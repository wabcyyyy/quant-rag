"""查询预处理（PLAN §5.2 / Phase 2 实测结论产品化）。

三项实测发现的落地（数据见 PLAN §5.3「已跑出的结果」）：
1. 聚合题（「关于 X 有哪些记录」）需要**实体聚焦查询**——剥离模板话术，
   只留实体词（实测周碧玉 0.06→0.22）
2. 聚合题需要**更宽检索预算**——覆盖率上限 = top_n / 答案集大小
   （top-8 0.13 → top-30 0.35）
3. 时间限定题接**元数据过滤**——覆盖率 0.61→0.96（+35pt）

用规则实现：零 LLM 成本、可复现、可解释；LLM 改写留作后续可选增强。
"""

from __future__ import annotations

import re

_QUOTE_RES = (
    re.compile(r"「(.+?)」"),
    re.compile(r"《(.+?)》"),
    re.compile(r"[\"“](.+?)[\"”]"),
)

# 聚合意图信号：问「有哪些/出现过哪些/都讨论了什么」而非单点事实
_AGG_PATTERNS = (
    "有哪些记录",
    "出现过哪些",
    "有哪些讨论",
    "有哪些安排",
    "哪些文档",
    "都讨论了什么",
    "都提到了什么",
    "相关的记录",
)

_YEAR_RE = re.compile(r"(20\d{2})\s*年")


class QueryRewriter:
    def __init__(self, retrieval_cfg: dict | None = None) -> None:
        self.cfg = retrieval_cfg or {}

    def rewrite(self, question: str) -> dict:
        """返回 {rewritten, filters, aggregate, top_n, reason}。

        aggregate=True 时调用方应：用 rewritten 检索、放宽预算到 top_n、按文档去重。
        """
        entity = self._extract_entity(question)
        is_agg = any(p in question for p in _AGG_PATTERNS)
        year = self._extract_year(question)

        filters: dict = {}
        # 日期过滤只在「明确时间限定的聚合题」上启用：
        # 实测本语料 doc_date 仅覆盖 ~19% 文档（会议档案 114/384 有完整日期，
        # 工作/议事档案几乎无日期），对普通事实题加年份过滤会误伤 80% 语料
        # （fact 覆盖率 1.00→0.33，见 PLAN §5.3 记录）。
        if year and is_agg:
            filters["doc_date"] = {
                "gte": f"{year}-01-01T00:00:00",
                "lt": f"{year + 1}-01-01T00:00:00",
            }

        reasons = []
        rewritten = question
        if is_agg and entity:
            rewritten = entity
            reasons.append(f"聚合意图 + 实体「{entity}」→ 实体聚焦查询")
        if filters:
            reasons.append(f"时间限定 {year} → doc_date 过滤")
        elif year and not is_agg:
            reasons.append(f"提及 {year} 但非时间限定聚合题 → 不加日期过滤（doc_date 稀疏）")

        top_n = None
        if is_agg:
            top_n = int(self.cfg.get("aggregate_top_n", 25))
            reasons.append(f"聚合题放宽预算 → top-{top_n} 且按文档去重")

        return {
            "rewritten": rewritten,
            "filters": filters or None,
            "aggregate": is_agg,
            "top_n": top_n,
            "reason": "；".join(reasons) or "无改写（单点事实题）",
        }

    @staticmethod
    def _extract_entity(question: str) -> str | None:
        for rx in _QUOTE_RES:
            m = rx.search(question)
            if m:
                return m.group(1).strip()
        return None

    @staticmethod
    def _extract_year(question: str) -> int | None:
        m = _YEAR_RE.search(question)
        return int(m.group(1)) if m else None

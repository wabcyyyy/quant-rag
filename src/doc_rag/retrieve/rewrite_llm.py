"""查询改写（LLM 版）。

取代改造前的 8 条中文子串规则。那套规则实测只在黄金集自己的措辞上触发：样例集
10 条里命中 2 条（恰好是 cross_doc 与 time_filter 那两行），另外 7 条同义改写
0 命中——于是「消融 #5 文档覆盖率 +35pt」是在**规则触发条件 == 评测集措辞**的前提
下测出来的，增益不可迁移。换成模型判意图之后，泛化能力单独设门禁
（`data/eval/rewrite_paraphrase.json` + `doc-rag check-rewrite`）。

两条边界纪律：
1. 检索预算 `top_n` 不交给模型——覆盖率上限是工程参数（实测随预算单调上升），
   由 `retrieval.aggregate_top_n` 决定。
2. 模型只报 `year`，日期区间由代码拼——`HybridRetriever._build_filter` 会照原样
   给任意字段名与任意算子建条件，让模型直接吐 filter 字典等于把未经校验的输入
   拿去查全库 payload。
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ValidationError

from ..generate import llm

_YEAR_MIN, _YEAR_MAX = 1990, 2100

SYSTEM_REWRITE = """\
你是文档检索的预处理模块。输入是一句针对企业会议纪要库的提问，只输出一个 JSON 对象，不要输出别的。

字段：
- rewritten: 真正送去检索的查询串。如果这个问题需要跨多篇文档汇总某个实体或主题，
  就只保留实体/主题词，剥掉「有哪些记录 / 都讨论了什么 / 列出所有」这类问话模板；
  否则原样返回整个问题。
- aggregate: 布尔值。问题的意图是「跨多篇文档枚举/汇总」（有哪些、都讨论了什么、
  列出所有、出现过几次、哪些部门参与、都安排了什么）时为 true；
  问单点事实、单个决议、某个名词的定义时为 false。
- year: 问题明确限定的年份（整数），如「2025 年」「25 年度」→ 2025；没有明确年份时 null。
  「上次 / 之前 / 最近」不算明确年份。
- reason: 一句话说明判断依据，供人核对。

例：
输入：关于「机房巡检」，各部门都安排过些什么？
输出：{"rewritten":"机房巡检","aggregate":true,"year":null,"reason":"跨文档枚举某主题的多个安排"}

输入：2025 年全员安全演练有哪些安排？
输出：{"rewritten":"全员安全演练","aggregate":true,"year":2025,"reason":"限定年份的枚举题"}

输入：新仓库计划什么时候投入使用？
输出：{"rewritten":"新仓库计划什么时候投入使用？","aggregate":false,"year":null,"reason":"单点事实题"}

输入：公司对采购审批权限做出了什么调整？
输出：{"rewritten":"公司对采购审批权限做出了什么调整？","aggregate":false,"year":null,"reason":"问单个决议"}
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class _ModelPlan(BaseModel):
    """模型返回的窄契约：只接受这四个字段，多余键忽略。"""

    rewritten: str = ""
    aggregate: bool = False
    year: int | None = None
    reason: str = ""


def _noop(question: str, reason: str) -> dict:
    """退化计划（不改写）。`degraded` 是给下游看的机器标志：靠嗅 `reason` 前缀判断
    「这条到底有没有真的改写」会在文案改动时静默失效，而 eval 臂和 /metrics 都依赖它。"""
    return {
        "rewritten": question,
        "filters": None,
        "aggregate": False,
        "top_n": None,
        "reason": reason,
        "degraded": True,
    }


def parse_reply(text: str) -> dict | None:
    """容忍围栏与前后杂文本；结构不合法返回 None（绝不猜）。"""
    if not text:
        return None
    fenced = text.strip()
    if "{" not in fenced:
        return None
    match = _JSON_RE.search(fenced)
    if not match:
        return None
    try:
        data = _ModelPlan.model_validate_json(match.group(0))
    except ValidationError:
        return None
    return data.model_dump()


def _date_range(year: int) -> dict:
    """与规则版逐字同形，保证 `DatetimeRange(**value)` 的解析口径不变。"""
    return {
        "gte": f"{year}-01-01T00:00:00",
        "lt": f"{year + 1}-01-01T00:00:00",
    }


class LLMQueryRewriter:
    """一次模型调用判定检索意图；失败就退化为「不改写」，并把它记下来。

    退化而非上抛是对的：改写失败不该让用户的提问整体失败；但必须计数——
    否则模型持续返回脏 JSON 时，会静默退回改造前那套「几乎不触发」的行为，
    而所有指标看起来仍是开着改写的。
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.retrieval_cfg = cfg.get("retrieval") or {}
        self.rewrite_cfg = cfg.get("rewrite") or {}
        self.failures: list[str] = []
        # 逐次调用耗时（ms + 是否缓存命中）：改写是「每次请求都付一次模型调用」的
        # 组件，它的延迟必须能被量出来，而不是只在评估的分阶段延迟里出现一次
        self.calls: list[dict] = []

    def endpoint(self) -> dict:
        """改写用的 LLM endpoint：默认继承 `llm`，`rewrite.model/base_url/api_key` 可单覆盖。

        为什么要能分开：改写是一次意图分类，挂在每次请求的关键路径上，花的是**延迟预算**；
        合成花的是质量预算。让两者锁死在同一个模型上，等于用合成的延迟上限给分类定价。
        为什么默认继承：不给部署加第二条必配路径。
        """
        llm_cfg = dict(self.cfg.get("llm") or {})
        for key in ("model", "base_url", "api_key"):
            value = str(self.rewrite_cfg.get(key) or "").strip()
            if value:
                llm_cfg[key] = value
        # 归因头是给 llm.base_url 那家用的；换了供应商就不该继续带过去
        if llm_cfg.get("base_url") != (self.cfg.get("llm") or {}).get("base_url"):
            llm_cfg.pop("headers", None)
        # 关键路径预算：合成侧「180s 超时 × 4 次重试」是给几十秒的聚合答案用的，
        # 照搬到一次分类上就是把 12 分钟的等待能力交给一个可退化的步骤。
        for key, cast in (("timeout_s", float), ("max_attempts", int)):
            budget = self.rewrite_cfg.get(key)
            if budget:
                llm_cfg[key] = cast(budget)
        return llm_cfg

    def rewrite(self, question: str) -> dict:
        llm_cfg = self.endpoint()
        if not (llm_cfg.get("api_key") and llm_cfg.get("model")):
            self.failures.append("no_llm")
            return _noop(question, "改写未生效：LLM 未配置（已计数）")
        llm_cfg["reasoning_effort"] = self.rewrite_cfg.get("reasoning_effort") or "none"
        try:
            reply, timing = llm.chat_timed(
                llm_cfg,
                f"输入：{question}",
                system_prompt=SYSTEM_REWRITE,
                temperature=self.rewrite_cfg.get("temperature"),
            )
        except Exception as exc:  # noqa: BLE001 调用失败退化为不改写，但要计数
            self.failures.append(f"call_failed:{type(exc).__name__}")
            return _noop(question, "改写未生效：调用失败（已计数）")
        self.calls.append(
            {
                "ms": timing.get("ms"),
                "cached": bool(timing.get("cached")),
                "model": timing.get("model"),
            }
        )

        parsed = parse_reply(reply)
        if parsed is None:
            self.failures.append("bad_json")
            return _noop(question, "改写未生效：模型返回不可解析（已计数）")

        aggregate = bool(parsed["aggregate"])
        year = parsed.get("year")
        if year is not None and not (_YEAR_MIN <= int(year) <= _YEAR_MAX):
            year = None
        year = int(year) if year is not None else None

        rewritten = (parsed["rewritten"] or "").strip() or question
        # 年份过滤只在「明确时间限定的聚合题」上启用：本语料 doc_date 只覆盖
        # 18.8% 文档（211/1121），对普通事实题加年份过滤会误伤八成语料
        # （实测 fact 文档覆盖率 1.00→0.33）。这条约束是语料属性，不随改写换实现而变。
        filters: dict[str, Any] | None = None
        reasons = []
        if aggregate and rewritten != question:
            reasons.append(f"聚合意图 → 实体聚焦查询「{rewritten}」")
        elif aggregate:
            reasons.append("聚合意图，但未剥离出更短的检索串")
        if filters is None and aggregate and year:
            filters = {"doc_date": _date_range(year)}
            reasons.append(f"时间限定 {year} → doc_date 过滤")
        elif year and not aggregate:
            reasons.append(f"提及 {year} 但非聚合题 → 不加日期过滤（doc_date 稀疏）")

        top_n = None
        if aggregate:
            top_n = int(self.retrieval_cfg.get("aggregate_top_n", 25))
            reasons.append(f"聚合题放宽预算 → top-{top_n} 且按文档去重")

        model_reason = (parsed.get("reason") or "").strip()
        if model_reason:
            reasons.append(f"模型依据：{model_reason}")
        return {
            "rewritten": rewritten,
            "filters": filters,
            "aggregate": aggregate,
            "top_n": top_n,
            "reason": "；".join(reasons) or "无改写（单点事实题）",
            "degraded": False,
        }

    @property
    def failure_count(self) -> int:
        return len(self.failures)


def endpoint_model(cfg: dict) -> str | None:
    """这份配置会把改写发给哪个模型（结果文件自证用，不发请求）。"""
    return LLMQueryRewriter(cfg).endpoint().get("model") or None

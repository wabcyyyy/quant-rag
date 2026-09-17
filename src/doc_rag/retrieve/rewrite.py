"""查询预处理（PLAN §7 Phase 2）：实体/时间识别 → 元数据过滤条件；可选查询改写。

时间/参会人识别直接决定消融 #5（无元数据过滤 vs 有）的上限。
"""

from __future__ import annotations


class QueryRewriter:
    def rewrite(self, query: str) -> dict:
        """返回 {"rewritten": str, "filters": {...}}。"""
        raise NotImplementedError("Phase 2 实装：LLM 实体/时间识别 + 改写")

"""黄金集条目 schema（PLAN §5.3）。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GoldItem(BaseModel):
    id: str
    type: str  # fact | decision | open_discussion | term | cross_doc | time_filter | no_answer
    question: str
    expected_answer: str
    must_contain: list[str] = Field(default_factory=list)
    source_doc_ids: list[str] = Field(default_factory=list)
    refusable: bool = False
    source_title: str | None = None
    # 题目来源标记：程序化构造题（如决议区提取）带 origin，重算 programmatic
    # 题型时据此整体重建——否则上一轮的程序化题会被当成 LLM 原题保留，越滚越多
    origin: str | None = None

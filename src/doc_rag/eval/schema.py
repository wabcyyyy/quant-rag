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
    # v3 窗口依赖题：答全这道题**必须同时看到的块**。文档级覆盖率对这道题是瞎的
    # （gold 只有 1 篇文档），性质全靠这两块分居，所以要把它们记下来——
    # `goldgen.v3_property_violations` 据此复检，重新生成时性质丢了会直接报。
    required_chunk_ids: list[str] = Field(default_factory=list)

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

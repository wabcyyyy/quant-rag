"""重排（PLAN §7 Phase 2）：bge-reranker（本地或 API）对 RRF 融合结果精排取 top_n。"""

from __future__ import annotations

from ..ingest.schema import Chunk


class Reranker:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def rerank(self, query: str, chunks: list[Chunk], top_n: int | None = None) -> list[Chunk]:
        raise NotImplementedError("Phase 2 实装：bge-reranker-v2-m3")

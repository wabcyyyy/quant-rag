"""混合检索（PLAN §5.2）：Qdrant Query API prefetch(Dense + BM25) → RRF 融合。

- BM25 路：Qdrant 内置稀疏模型 qdrant/bm25 + IDF 修饰符（服务端打分）。
  Qdrant 1.19.1 实测：单路可直接 query+using；双路必须 prefetch + 顶层 FusionQuery。
  喂入文本为 jieba 预分词空格拼接（内置模型按空白切词，见 ingest/bm25.py）。
- Dense 路：BGE-M3 dense(1024)，客户端先向量化再查询
- 两路 Prefetch（可带元数据过滤）→ FusionQuery(RRF) 融合
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from ..ingest.bm25 import build_bm25_text
from ..ingest.embedder import Embedder

_BM25_MODEL = "qdrant/bm25"


class HybridRetriever:
    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        collection: str,
        retrieval_cfg: dict,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self.cfg = retrieval_cfg

    def retrieve(
        self,
        question: str,
        top_n: int | None = None,
        filters: dict | None = None,
        mode: str | None = None,
    ) -> list[dict]:
        """检索。mode="dense" 走纯向量（消融对照组），默认 "hybrid" 走双路 RRF。

        filters 为简单匹配条件：{"topics": "预算"} / {"attendees": ["张三"]} /
        {"doc_date": {"gte": "2026-01-01"}}（消融 #5 的开关）。
        返回按排名的 [{chunk_id, doc_id, title, text, section_path, page,
        block_type, doc_date, score}]。
        """
        limit = top_n or int(self.cfg.get("fusion_limit", 12))
        mode = mode or self.cfg.get("mode", "hybrid")
        qvec = self.embedder.embed([question])[0]
        qfilter = self._build_filter(filters)

        if mode == "dense":
            response = self.client.query_points(
                self.collection,
                query=qvec,
                using="dense",
                limit=limit,
                query_filter=qfilter,
                with_payload=True,
            )
        else:
            qtext = build_bm25_text(question)
            prefetch = [
                models.Prefetch(
                    query=qvec,
                    using="dense",
                    limit=int(self.cfg.get("k_dense", 20)),
                    filter=qfilter,
                ),
                models.Prefetch(
                    query=models.Document(text=qtext, model=_BM25_MODEL),
                    using="bm25",
                    limit=int(self.cfg.get("k_bm25", 20)),
                    filter=qfilter,
                ),
            ]
            response = self.client.query_points(
                self.collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "doc_id": payload.get("doc_id"),
                    "title": payload.get("title"),
                    "text": payload.get("text"),
                    "section_path": payload.get("section_path") or [],
                    "page": payload.get("page"),
                    "block_type": payload.get("block_type"),
                    "doc_date": payload.get("doc_date"),
                    "score": point.score,
                }
            )
        return results

    @staticmethod
    def _build_filter(filters: dict | None) -> models.Filter | None:
        if not filters:
            return None
        must = []
        for key, value in filters.items():
            if isinstance(value, dict):  # 范围条件，如 {"gte": "2026-01-01"}
                must.append(
                    models.FieldCondition(key=key, range=models.DatetimeRange(**value))
                )
            elif isinstance(value, list):
                must.append(
                    models.FieldCondition(key=key, match=models.MatchAny(any=value))
                )
            else:
                must.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
        return models.Filter(must=must)

"""混合检索（PLAN §5.2）：Qdrant Query API prefetch(Dense + BM25) → RRF 融合。

- BM25 路：Qdrant 内置稀疏模型 qdrant/bm25 + IDF 修饰符（服务端打分）。
  Qdrant 1.19.1 实测：单路可直接 query+using；双路必须 prefetch + 顶层 FusionQuery。
  喂入文本为 jieba 预分词空格拼接（内置模型按空白切词，见 ingest/bm25.py）。
- Dense 路：BGE-M3 dense(1024)，客户端先向量化再查询
- 两路 Prefetch（可带元数据过滤）→ FusionQuery(RRF) 融合
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from ..ingest.bm25 import build_bm25_text
from ..ingest.embedder import Embedder

_BM25_MODEL = "qdrant/bm25"


@dataclass(frozen=True)
class RetrievalOutcome:
    """一次检索的结果 + 它走过的路。

    `filter_fallback` 不是装饰：过滤字段稀疏时回退会把过滤整个丢掉，于是「带元数据
    过滤」这条臂里有一部分条目其实没过滤。不落盘就没人知道覆盖率涨了几个点是靠
    放弃过滤拿到的，而且这条路径还要多付一次检索的延迟。
    """

    chunks: list[dict]
    filter_applied: bool = False
    filter_fallback: bool = False
    n_before_fallback: int = 0


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
        aggregate: bool = False,
    ) -> RetrievalOutcome:
        """检索。mode="dense" 走纯向量（消融对照组），默认 "hybrid" 走双路 RRF。

        aggregate=True：聚合检索——先取大池（pool_size），按 doc_id 去重后返回
        每篇文档的最佳块。跨文档聚合题（「关于 X 有哪些记录」）需要文档多样性，
        单纯 top-k 可能被同一篇文档的多个块占满。

        filters 为简单匹配条件：{"doc_date": {"gte": "2026-01-01"}} /
        {"category": "会议档案"} / {"doc_group": "办公会"}（消融 #5 的开关）。

        字段可用性是实测约束，不是想当然：默认入库只有 doc_date / category / doc_group
        有值（doc_date 覆盖率因语料与回填状态而异：公司语料周次回填后块级 62.5%、
        示例语料 100%；任何过滤前先看当次 profile，别引用历史常数），而
        topics / attendees 必须走 `ingest --llm-meta` 才会被填上——默认配置下它们
        在全库为 0。
        """
        limit = top_n or int(self.cfg.get("fusion_limit", 12))
        mode = mode or self.cfg.get("mode", "hybrid")
        pool = int(self.cfg.get("aggregate_pool", 50)) if aggregate else limit
        qvec = self.embedder.embed([question])[0]
        qfilter = self._build_filter(filters)

        results = self._query(question, qvec, qfilter, mode, pool, limit, aggregate)
        # 元数据过滤回退保护：字段稀疏时过滤可能清空结果（见 PLAN §5.3），
        # 结果过少则去掉过滤重试，保证不因过滤把正确答案滤没
        n_with_filter = len(results)
        fallback = False
        if qfilter is not None and n_with_filter < min(3, limit):
            results = self._query(question, qvec, None, mode, pool, limit, aggregate)
            fallback = True
        return RetrievalOutcome(
            chunks=results,
            filter_applied=qfilter is not None,
            filter_fallback=fallback,
            n_before_fallback=n_with_filter,
        )

    def _query(
        self,
        question: str,
        qvec: list[float],
        qfilter: models.Filter | None,
        mode: str,
        pool: int,
        limit: int,
        aggregate: bool,
    ) -> list[dict]:

        if mode == "dense":
            response = self.client.query_points(
                self.collection,
                query=qvec,
                using="dense",
                limit=pool,
                query_filter=qfilter,
                with_payload=True,
            )
        else:
            qtext = build_bm25_text(question)
            prefetch = [
                models.Prefetch(
                    query=qvec,
                    using="dense",
                    limit=max(pool, int(self.cfg.get("k_dense", 20))),
                    filter=qfilter,
                ),
                models.Prefetch(
                    query=models.Document(text=qtext, model=_BM25_MODEL),
                    using="bm25",
                    limit=max(pool, int(self.cfg.get("k_bm25", 20))),
                    filter=qfilter,
                ),
            ]
            response = self.client.query_points(
                self.collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=pool,
                with_payload=True,
            )
        results = []
        seen_docs: set[str] = set()
        for point in response.points:
            payload = point.payload or {}
            doc_id = payload.get("doc_id")
            if aggregate:
                doc_key = str(doc_id)
                if doc_key in seen_docs:  # 每篇文档只留最佳块 → 覆盖更多文档
                    continue
                seen_docs.add(doc_key)
            results.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "doc_id": doc_id,
                    "title": payload.get("title"),
                    "text": payload.get("text"),
                    "section_path": payload.get("section_path") or [],
                    "page": payload.get("page"),
                    "block_type": payload.get("block_type"),
                    "doc_date": payload.get("doc_date"),
                    "score": point.score,
                }
            )
            if len(results) >= limit:
                break
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

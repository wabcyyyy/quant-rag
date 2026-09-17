"""FastAPI（PLAN §2 交付行）。启动：uv run doc-rag serve"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="doc-rag", version="0.1.0")


class QueryIn(BaseModel):
    question: str
    kb: str | None = None
    top_n: int | None = None


@lru_cache(maxsize=1)
def _pipeline():
    from qdrant_client import QdrantClient

    from doc_rag.config import load_config
    from doc_rag.ingest.embedder import Embedder
    from doc_rag.retrieve.hybrid import HybridRetriever
    from doc_rag.retrieve.rewrite import QueryRewriter

    cfg = load_config()
    retriever = HybridRetriever(
        client=QdrantClient(url=cfg["qdrant"]["url"], timeout=60.0),
        embedder=Embedder(cfg["embedding"]),
        collection=cfg["qdrant"]["collection"],
        retrieval_cfg=cfg["retrieval"],
    )
    # 不含 Synthesizer：它的 last_meta 是可变属性，跨请求复用会串台（见 query()）
    return cfg, retriever, QueryRewriter(cfg["retrieval"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query")
def query(body: QueryIn) -> dict:
    import time

    from doc_rag.generate.synthesizer import Synthesizer

    cfg, retriever, rewriter = _pipeline()
    # 合成器每请求新建：它是无状态小对象，而 last_meta 是可变属性——
    # 复用 _pipeline 的缓存单例会让并发请求互相读到对方的计时（假延迟）。
    synthesizer = Synthesizer(cfg["llm"])
    if body.kb:
        retriever.collection = body.kb
    t0 = time.perf_counter()
    plan = rewriter.rewrite(body.question)
    t_rewrite = time.perf_counter()
    results = retriever.retrieve(
        plan["rewritten"],
        top_n=body.top_n or plan["top_n"],
        filters=plan["filters"],
        aggregate=plan["aggregate"],
    )
    t_retrieve = time.perf_counter()
    contexts = [
        {
            "no": i + 1,
            "text": r["text"],
            "doc": r["title"] or r["doc_id"],
            "page": r["page"],
        }
        for i, r in enumerate(results)
    ]
    answer = synthesizer.answer(body.question, contexts, aggregate=plan["aggregate"])
    t_synth = time.perf_counter()
    synth_meta = synthesizer.last_meta or {}
    return {
        "question": body.question,
        "rewrite": plan,
        "answer": answer,
        "citations": [
            {"no": c["no"], "doc": c["doc"], "page": c["page"], "doc_id": r["doc_id"]}
            for c, r in zip(contexts, results)
        ],
        # 延迟口径（PLAN「延迟口径」）：synth_cached 为真时 synthesize_ms 是缓存查询
        # 耗时而非模型延迟，客户端据此决定要不要信这个数
        "latency_ms": {
            "rewrite": round((t_rewrite - t0) * 1000, 1),
            "retrieve": round((t_retrieve - t_rewrite) * 1000, 1),
            "synthesize": synth_meta.get("ms"),
            "synth_cached": synth_meta.get("cached"),
            "total": round((t_synth - t0) * 1000, 1),
        },
    }


@app.post("/query/stream")
def query_stream(body: QueryIn):
    """流式问答（T7）：SSE 事件 = rewrite / delta / citations / done(含 latency_ms)。

    检索与合成路径与 /query 同构（同一 _pipeline、同一缓存），区别只在答案
    以增量事件推送——首 token 即达，不再等整段合成。旧端点行为不变。
    """
    import json
    import time
    from collections.abc import Iterator

    from fastapi.responses import StreamingResponse

    from doc_rag.generate.synthesizer import Synthesizer

    cfg, retriever, rewriter = _pipeline()
    synthesizer = Synthesizer(cfg["llm"])  # 每请求新建：last_meta 跨请求会串台（见 query）
    if body.kb:
        retriever.collection = body.kb

    def _event(name: str, data) -> str:
        return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _sse() -> Iterator[str]:
        t0 = time.perf_counter()
        plan = rewriter.rewrite(body.question)
        t_rewrite = time.perf_counter()
        yield _event("rewrite", plan)
        results = retriever.retrieve(
            plan["rewritten"],
            top_n=body.top_n or plan["top_n"],
            filters=plan["filters"],
            aggregate=plan["aggregate"],
        )
        t_retrieve = time.perf_counter()
        cap = int(cfg["retrieval"].get("max_contexts") or 0)
        if cap:
            results = results[:cap]
        contexts = [
            {
                "no": i + 1,
                "text": r["text"],
                "doc": r["title"] or r["doc_id"],
                "page": r["page"],
            }
            for i, r in enumerate(results)
        ]
        for piece in synthesizer.answer_stream(
            body.question, contexts, aggregate=plan["aggregate"]
        ):
            yield _event("delta", {"text": piece})
        synth_meta = synthesizer.last_meta or {}
        yield _event(
            "citations",
            [
                {"no": c["no"], "doc": c["doc"], "page": c["page"], "doc_id": r["doc_id"]}
                for c, r in zip(contexts, results)
            ],
        )
        yield _event(
            "done",
            {
                "latency_ms": {
                    "rewrite": round((t_rewrite - t0) * 1000, 1),
                    "retrieve": round((t_retrieve - t_rewrite) * 1000, 1),
                    "synthesize": synth_meta.get("ms"),
                    "synth_cached": synth_meta.get("cached"),
                    "total": round((time.perf_counter() - t0) * 1000, 1),
                }
            },
        )

    return StreamingResponse(_sse(), media_type="text/event-stream")

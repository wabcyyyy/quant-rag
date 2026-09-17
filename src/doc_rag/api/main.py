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
    from doc_rag.generate.synthesizer import Synthesizer
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
    return cfg, retriever, Synthesizer(cfg["llm"]), QueryRewriter(cfg["retrieval"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query")
def query(body: QueryIn) -> dict:
    cfg, retriever, synthesizer, rewriter = _pipeline()
    if body.kb:
        retriever.collection = body.kb
    plan = rewriter.rewrite(body.question)
    results = retriever.retrieve(
        plan["rewritten"],
        top_n=body.top_n or plan["top_n"],
        filters=plan["filters"],
        aggregate=plan["aggregate"],
    )
    contexts = [
        {
            "no": i + 1,
            "text": r["text"],
            "doc": r["title"] or r["doc_id"],
            "page": r["page"],
        }
        for i, r in enumerate(results)
    ]
    answer = synthesizer.answer(body.question, contexts)
    return {
        "question": body.question,
        "rewrite": plan,
        "answer": answer,
        "citations": [
            {"no": c["no"], "doc": c["doc"], "page": c["page"], "doc_id": r["doc_id"]}
            for c, r in zip(contexts, results)
        ],
    }

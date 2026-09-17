"""重排（PLAN §7 Phase 2）：bge-reranker-v2-m3（SiliconFlow /rerank API）。

动机：RRF 融合后的 top-k 里仍混有噪声块，而 Faithfulness 受上下文噪声拖累
（基线 0.62，目标 0.85）。重排后只把最相关的少数块送进合成，削减噪声。
"""

from __future__ import annotations

import httpx


class Reranker:
    def __init__(self, cfg: dict) -> None:
        self.base_url = cfg["base_url"].rstrip("/")
        self.api_key = cfg["api_key"]
        self.model = cfg.get("model", "BAAI/bge-reranker-v2-m3")
        self.top_n = int(cfg.get("top_n", 6))

    def rerank(
        self, query: str, chunks: list[dict], top_n: int | None = None
    ) -> list[dict]:
        """按与问题的相关性重排，返回 top_n 块（附加 rerank_score）。"""
        if not chunks:
            return []
        n = min(top_n or self.top_n, len(chunks))
        r = httpx.post(
            f"{self.base_url}/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": [c.get("text") or "" for c in chunks],
                "top_n": n,
            },
            timeout=60.0,
        )
        r.raise_for_status()
        out: list[dict] = []
        for item in r.json()["results"]:
            chunk = dict(chunks[item["index"]])
            chunk["rerank_score"] = item["relevance_score"]
            out.append(chunk)
        return out

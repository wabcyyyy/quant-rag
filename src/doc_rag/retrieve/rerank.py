"""重排（PLAN §7 Phase 2）：bge-reranker-v2-m3（SiliconFlow /rerank API）。

动机：RRF 融合后的 top-k 里仍混有噪声块，而 Faithfulness 受上下文噪声拖累
（基线 0.62，目标 0.85）。重排后只把最相关的少数块送进合成，削减噪声。

两个角色必须分开：`rerank()` 给下游一份**排好序的完整清单**（检索指标在它上面算），
`context_budget` 决定其中前几块进 LLM。让前者兼做截断，等于用一个上下文参数
偷偷改掉所有检索指标的分母。
"""

from __future__ import annotations

import httpx

from ..net import request_with_retry


class Reranker:
    def __init__(self, cfg: dict) -> None:
        self.base_url = cfg["base_url"].rstrip("/")
        self.api_key = cfg["api_key"]
        self.model = cfg.get("model", "BAAI/bge-reranker-v2-m3")
        self.top_n = int(cfg.get("top_n", 6))

    @property
    def context_budget(self) -> int:
        """送进合成的块数上限。只作用于上下文，**不作用于检索清单**（见 rerank）。"""
        return self.top_n

    def rerank(
        self, query: str, chunks: list[dict], top_n: int | None = None
    ) -> list[dict]:
        """按与问题的相关性重排，返回**全量重排**清单（附加 rerank_score）。

        `top_n=None`（默认）= 不截断：把候选按相关性重新排好整条返回。
        改造前这里默认取 `self.top_n`(6) 并把清单砍到 6 条，而调用方覆盖了自己的
        检索结果——于是 `retrieved` 这份「未截断清单」其实是 6 条，与对照臂的 8 条
        不等长。nDCG@8「−3.4pt」和 Recall@8「持平」都是这个长度差造成的，不是排序质量。
        截断仍然发生，但发生在上下文预算处（`context_budget`），送 LLM 的块数逐字不变。

        显式传 `top_n` 才截断（调用方知道自己在要什么）。
        """
        if not chunks:
            return []
        n = len(chunks) if top_n is None else min(top_n, len(chunks))

        def _once() -> dict:
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
            return r.json()

        out: list[dict] = []
        for item in request_with_retry(_once, label="rerank", attempts=3)["results"]:
            chunk = dict(chunks[item["index"]])
            chunk["rerank_score"] = item["relevance_score"]
            out.append(chunk)
        return out

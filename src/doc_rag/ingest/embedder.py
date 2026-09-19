"""BGE-M3 dense embedding 客户端：OpenAI 兼容 /embeddings，批量 + 统一重试判定。

改造前这里对**所有**异常盲重试 3 次（含 401/400 这类永久错误），而同一时间
`llm.py` 已经在按状态码分类——同一类 bug 修了一处留了一处。现在两处共用
`net.request_with_retry`。
"""

from __future__ import annotations

import httpx

from ..net import request_with_retry

_BATCH = 32


class Embedder:
    def __init__(self, emb_cfg: dict) -> None:
        self.base_url = emb_cfg["base_url"].rstrip("/")
        self.api_key = emb_cfg["api_key"]
        self.model = emb_cfg["model"]
        # 维度不匹配必须在这里挡下：否则它一路走到 upsert，变成逐文档失败
        # 沉进 stats["failed"]，1121 篇之后才发现
        self.expected_dim = int(emb_cfg.get("dense_dim") or 0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), _BATCH):
            out.extend(self._embed_batch(texts[i : i + _BATCH]))
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        def _once() -> list[list[float]]:
            r = httpx.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": batch},
                timeout=60.0,
            )
            r.raise_for_status()
            data = r.json()["data"]
            vectors = [
                item["embedding"] for item in sorted(data, key=lambda x: x["index"])
            ]
            if self.expected_dim:
                bad = [n for n, v in enumerate(vectors) if len(v) != self.expected_dim]
                if bad:
                    raise ValueError(
                        f"{len(bad)}/{len(vectors)} 条向量维度不是 {self.expected_dim}"
                        f"（首条 {len(vectors[bad[0]])}）——换嵌入模型必须 --recreate 重建 collection"
                    )
            return vectors

        return request_with_retry(_once, label="embedding", attempts=3)

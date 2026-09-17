"""BGE-M3 dense embedding 客户端：OpenAI 兼容 /embeddings，批量 + 简单重试。"""

from __future__ import annotations

import time

import httpx

_BATCH = 32
_RETRIES = 3


class Embedder:
    def __init__(self, emb_cfg: dict) -> None:
        self.base_url = emb_cfg["base_url"].rstrip("/")
        self.api_key = emb_cfg["api_key"]
        self.model = emb_cfg["model"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), _BATCH):
            out.extend(self._embed_batch(texts[i : i + _BATCH]))
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(_RETRIES):
            try:
                r = httpx.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": batch},
                    timeout=60.0,
                )
                r.raise_for_status()
                data = r.json()["data"]
                return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"embedding 请求失败（已重试 {_RETRIES} 次）：{last_exc}") from last_exc

"""库指纹（B4/B10）：让「同 prompt、不同索引状态」的 LLM 缓存不再互通。

为什么必须有：本地响应缓存的键 = endpoint+model+messages+params，messages 含检索
到的正文——块变了键自然变。但**块没变、语义变了**的情况键不变：重入库换了 BM25
文本策略、换了分块策略、OCR 把近空文档救回来，只要命中清单恰好同文同块，旧答案
照样被复用（「考卷没变、课本换了、还抄旧笔记」）。ingest 把每个 collection 的
`index_fp` 写进 `.cache/index_fingerprint.json`，检索路径（Orchestrator）把当前
`collection` 挂进 contextvar，`llm._cache_key` 便把「该 collection 的当前指纹」
织进键——索引一变，旧答案自动 miss。

口径：
- **裸检出不报错**：指纹文件不存在（新 clone、没跑过 ingest）→ identity 为空串，
  键退回原口径。绝不能因为缺文件把检索路径打断。
- **指纹只由 ingest 写**：读路径（eval / 服务）只消费，不许顺手写。
- 指纹内容 = collection + 点数/块数 + 嵌入模型 + 分块策略 + BM25 文本策略 +
  payload schema 版本（B10 `index_fp` 的定义，见 SPEC §6.2）。
"""

from __future__ import annotations

import contextvars
import hashlib
import json

from .config import project_root

_FP_FILE = project_root() / ".cache" / "index_fingerprint.json"

# 检索路径在「本次问答用的是哪个 collection」上置值；eval 单线程、API 走
# 每请求独立的 context（contextvar 不会跨请求串库——那是 W1 拆掉过的隐患）。
active_collection: contextvars.ContextVar[str] = contextvars.ContextVar(
    "active_collection", default=""
)

#: payload 字段集的版本号：字段集变了 = 旧答案的依据变了，指纹必须跟着变
_PAYLOAD_SCHEMA_VERSION = "2"  # v2 = +ctx_prefix（A3.4）与 +source 标注（A3.3）


def compute_index_fp(
    *,
    collection: str,
    n_points: int,
    n_chunks: int,
    embed_model: str,
    chunk_strategy: str,
    bm25_strategy: str,
) -> str:
    blob = json.dumps(
        {
            "collection": collection,
            "n_points": n_points,
            "n_chunks": n_chunks,
            "embed_model": embed_model,
            "chunk_strategy": chunk_strategy,
            "bm25_strategy": bm25_strategy,
            "payload_schema": _PAYLOAD_SCHEMA_VERSION,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_fingerprints() -> dict[str, str]:
    try:
        return {
            str(k): str(v)
            for k, v in json.loads(_FP_FILE.read_text(encoding="utf-8")).items()
        }
    except Exception:  # noqa: BLE001 裸检出（无文件/坏文件）= 无指纹可用，不报错
        return {}


def save_fingerprints(mapping: dict[str, str]) -> None:
    _FP_FILE.parent.mkdir(parents=True, exist_ok=True)
    _FP_FILE.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def identity_for(collection: str | None) -> str:
    """当前 collection 的指纹；缺文件 / 未登记 / 未置 collection → 空串。"""
    if not collection:
        return ""
    return load_fingerprints().get(collection, "")


def active_identity() -> str:
    return identity_for(active_collection.get())

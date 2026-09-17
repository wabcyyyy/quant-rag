"""入库（PLAN §5.1 后半）：建表 → 分块 → LLM 元数据 → Embed → Qdrant upsert。

幂等：按 doc_id 删旧再插，重复 ingest 不产生重复块；单文档失败不阻塞整批。
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from qdrant_client import QdrantClient, models

from .bm25 import build_bm25_text
from .chunker import chunk_by
from .embedder import Embedder
from .metadata import base_meta, extract_metadata
from .schema import Chunk, IntermediateDoc

_UPSERT_BATCH = 64
_META_WORKERS = 8
_META_EXCLUDE = {"profile.json"}


def ensure_collection(
    client: QdrantClient, name: str, dense_dim: int, recreate: bool = False
) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
            },
            # BM25 路：Qdrant 内置稀疏模型 + IDF 修饰符 = 服务端 BM25 打分
            #（Qdrant 1.19.1 实测可用；喂 jieba 预分词文本，见 ingest/bm25.py）
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
    client.create_payload_index(name, "doc_id", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "category", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "doc_group", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "block_type", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "doc_date", models.PayloadSchemaType.DATETIME)
    client.create_payload_index(name, "topics", models.PayloadSchemaType.KEYWORD)


def _embed_text(chunk: Chunk) -> str:
    """section_path 拼进向量输入，提升同质语料的区分度。"""
    prefix = " / ".join(chunk.section_path)
    return f"{prefix}\n{chunk.text}" if prefix else chunk.text


def index_parsed(
    parsed_dir: Path,
    cfg: dict,
    collection: str | None = None,
    recreate: bool = False,
    use_llm_meta: bool | None = None,
    chunk_strategy: str = "structural",
    limit: int | None = None,
) -> dict:
    """中间 JSON → Qdrant。

    `use_llm_meta` 缺省时读 `metadata_extraction.enabled`（默认 false）。
    **绝不能默认开**：LLM 抽取是 1 次/文档，1130 篇就是 1130 次调用，
    是本项目最大的单点成本（PLAN §8）。要用必须显式传 True 或走 `ingest --llm-meta`。
    `limit` 只取前 N 篇（试跑）——解析和入库都必须受它约束，否则试跑也会全量计费。
    """
    client = QdrantClient(url=cfg["qdrant"]["url"], timeout=60.0)
    name = collection or cfg["qdrant"]["collection"]
    ensure_collection(client, name, int(cfg["embedding"]["dense_dim"]), recreate)
    embedder = Embedder(cfg["embedding"])
    if use_llm_meta is None:
        use_llm_meta = bool((cfg.get("metadata_extraction") or {}).get("enabled"))
    llm_cfg = cfg["llm"] if use_llm_meta else {}
    stats: dict = {"docs": 0, "chunks": 0, "failed": [], "llm_meta": use_llm_meta}

    # 先装载全部中间 JSON
    docs: list[tuple[Path, IntermediateDoc]] = []
    for json_file in sorted(parsed_dir.glob("*.json")):
        if json_file.name in _META_EXCLUDE:
            continue
        try:
            doc = IntermediateDoc.model_validate_json(
                json_file.read_text(encoding="utf-8")
            )
            if doc.blocks:
                docs.append((json_file, doc))
        except Exception as exc:  # noqa: BLE001
            stats["failed"].append({"file": json_file.name, "error": str(exc)})
    if limit:
        docs = docs[:limit]

    # LLM 元数据是批量入库的瓶颈：并行抽取（文件名信号在 base_meta 里零成本）
    if llm_cfg:
        with ThreadPoolExecutor(max_workers=_META_WORKERS) as pool:
            metas = list(pool.map(lambda d: extract_metadata(d, llm_cfg), (d for _, d in docs)))
    else:
        metas = [base_meta(d) for _, d in docs]

    for (json_file, doc), meta in zip(docs, metas):
        try:
            chunks = chunk_by(chunk_strategy, doc)
            if not chunks:
                continue
            # 幂等：清掉本文档旧块再插
            client.delete(
                name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=doc.meta.doc_id),
                            )
                        ]
                    )
                ),
            )
            vectors = embedder.embed([_embed_text(c) for c in chunks])
            points = []
            for chunk, vector in zip(chunks, vectors):
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": doc.meta.doc_id,
                    "title": doc.meta.title,
                    "text": chunk.text,
                    "section_path": chunk.section_path,
                    "page": chunk.page,
                    "block_type": chunk.block_type,
                    "doc_date": meta["doc_date"],
                    "category": meta.get("category"),
                    "doc_group": meta.get("doc_group"),
                    "meeting_type": meta["meeting_type"],
                    "attendees": meta["attendees"],
                    "topics": meta["topics"],
                }
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                        vector={
                            "dense": vector,
                            "bm25": models.Document(
                                text=build_bm25_text(chunk.text), model="qdrant/bm25"
                            ),
                        },
                        payload={k: v for k, v in payload.items() if v is not None},
                    )
                )
            for i in range(0, len(points), _UPSERT_BATCH):
                client.upsert(name, points=points[i : i + _UPSERT_BATCH])
            stats["docs"] += 1
            stats["chunks"] += len(points)
        except Exception as exc:  # noqa: BLE001 单文档失败不阻塞整批
            stats["failed"].append({"file": json_file.name, "error": str(exc)})
    return stats

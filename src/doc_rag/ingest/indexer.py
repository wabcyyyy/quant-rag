"""入库（PLAN §5.1 后半）：建表 → 分块 → LLM 元数据 → Embed → Qdrant upsert。

幂等：按 doc_id 删旧再插，重复 ingest 不产生重复块；单文档失败不阻塞整批。
但**只删「本次要重插的那篇」的旧块**，所以源文件被删掉或改过内容（doc_id 是
sha256[:16]，改一个字节就是新 doc_id）时，旧点会永久留在库里并被检索命中——
`reconcile()` / `--prune` 就是为它存在的。
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from .bm25 import build_bm25_text
from .chunker import chunk_by
from .contextual import contextual_cfg, generate_prefixes
from .embedder import Embedder
from .metadata import base_meta, extract_metadata
from .schema import Chunk, IntermediateDoc

_UPSERT_BATCH = 64
_META_WORKERS = 8
_META_EXCLUDE = {"profile.json"}
_SCROLL_PAGE = 1024
_DELETE_BATCH = 256


def ensure_collection(
    client: QdrantClient,
    name: str,
    dense_dim: int,
    recreate: bool = False,
    llm_meta_fields: bool = False,
) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_dim, distance=models.Distance.COSINE
                )
            },
            # BM25 路：Qdrant 内置稀疏模型 + IDF 修饰符 = 服务端 BM25 打分
            # （Qdrant 1.19.1 实测可用；喂 jieba 预分词文本，见 ingest/bm25.py）
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
    client.create_payload_index(name, "doc_id", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "category", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "doc_group", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "block_type", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "doc_date", models.PayloadSchemaType.DATETIME)
    # topics / attendees 只在 `ingest --llm-meta` 时才有值（base_meta 留空列表）。
    # 默认入库不给永久为空的字段建索引：建了就会让人以为过滤可用（实测 0/3856 命中）。
    if llm_meta_fields:
        client.create_payload_index(name, "topics", models.PayloadSchemaType.KEYWORD)
        client.create_payload_index(name, "attendees", models.PayloadSchemaType.KEYWORD)


def _embed_text(chunk: Chunk) -> str:
    """向量输入 = 被索引文本（section_path 前缀 + 正文）。

    A3.2 之后它与 BM25 分词文本同源（`Chunk.indexed_text` 是唯一实现）——
    两路必须看到同一批词，否则「dense 配得上、BM25 没见过」的定位词（q019）
    会在融合时把其中一路变成噪声。
    """
    return chunk.indexed_text()


def collection_doc_ids(client: QdrantClient, name: str) -> dict[str, int]:
    """库里现存的 doc_id → 块数。只读 payload、不拉向量。"""
    counts: dict[str, int] = {}
    # 页令牌的类型由 SDK 决定（int / str / UUID 都可能），原样回传即可
    offset: Any = None
    while True:
        points, offset = client.scroll(
            name,
            limit=_SCROLL_PAGE,
            offset=offset,
            with_payload=["doc_id"],
            with_vectors=False,
        )
        for point in points:
            doc_id = (point.payload or {}).get("doc_id")
            if doc_id is not None:
                counts[str(doc_id)] = counts.get(str(doc_id), 0) + 1
        if offset is None:
            return counts


def reconcile(keep_doc_ids: set[str], existing: dict[str, int]) -> dict:
    """语料应有集合 vs 库里现存集合的双向对账。

    - `orphan_*`：库里有、语料里没有 → 幽灵块（源文件删除或改过内容）。它们会继续
      被检索命中，且带的是旧日期与旧正文——比漏入库更糟，因为没人怀疑结果里有它。
    - `missing_docs`：语料里有、库里没有 → 这批没灌进去（失败/截断/新文件）。
    """
    orphans = sorted(set(existing) - keep_doc_ids)
    return {
        "keep_docs": len(keep_doc_ids),
        "collection_docs": len(existing),
        "collection_chunks": sum(existing.values()),
        "orphan_doc_ids": orphans,
        "orphan_chunks": sum(existing[d] for d in orphans),
        "missing_docs": sorted(keep_doc_ids - set(existing)),
    }


def delete_docs(client: QdrantClient, name: str, doc_ids: list[str]) -> int:
    for i in range(0, len(doc_ids), _DELETE_BATCH):
        client.delete(
            name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchAny(any=doc_ids[i : i + _DELETE_BATCH]),
                        )
                    ]
                )
            ),
        )
    return len(doc_ids)


def _prune_decision(rec: dict, stats: dict, assume_yes: bool) -> dict:
    """删之前先把三道闸过一遍——误删一次索引就要重灌并重跑全部评估基线。"""
    orphans = rec["orphan_doc_ids"]
    if not orphans:
        return {"deleted": 0}
    if not assume_yes:
        return {
            "deleted": 0,
            "refused": f"发现 {len(orphans)} 篇幽灵文档，未加 --yes 只报告",
        }
    if stats["failed"]:
        return {
            "deleted": 0,
            "refused": f"{len(stats['failed'])} 个解析产物读失败，其旧点无法判定归属 → 拒绝清理",
        }
    if len(orphans) > rec["keep_docs"]:
        return {
            "deleted": 0,
            "refused": "要删的比留下的多，八成是 --kb 或 --parsed-dir 串了库 → 拒绝清理",
        }
    return {"deleted": len(orphans)}


def index_parsed(
    parsed_dir: Path,
    cfg: dict,
    collection: str | None = None,
    recreate: bool = False,
    use_llm_meta: bool | None = None,
    chunk_strategy: str = "structural",
    limit: int | None = None,
    prune: bool = False,
    assume_yes: bool = False,
    keep_doc_ids: set[str] | None = None,
    ctx_mode: str | None = None,
) -> dict:
    """中间 JSON → Qdrant。

    `use_llm_meta` 缺省时读 `metadata_extraction.enabled`（默认 false）。
    **绝不能默认开**：LLM 抽取是 1 次/文档，1130 篇就是 1130 次调用，
    是本项目最大的单点成本（PLAN §8）。要用必须显式传 True 或走 `ingest --llm-meta`。
    `limit` 只取前 N 篇（试跑）——解析和入库都必须受它约束，否则试跑也会全量计费。
    `keep_doc_ids` 是「语料应该有什么」的判据：默认取 parsed_dir 全体（含 0 块的，
    解析出空块只是可疑、不是确证该删）；调用方传本次 raw 语料的 sha 集合时，
    parsed_dir 里那些已经没有对应源文件的陈旧产物会被单列成 `stale_parsed_json`。
    `prune` 只报告幽灵文档；再加 `assume_yes` 才真删（删前过 `_prune_decision` 三道闸）。
    `ctx_mode`：contextual 前缀（A3.4/ADR-0004）的消融臂——
    None=读配置 `contextual.enabled`；"off"=无前缀；"both"=前缀进 dense+BM25；
    "bm25"=前缀只进 BM25（区分「前缀的信息价值」与「前缀对 dense 的扰动」）。
    前缀走 LLM 响应缓存，同 chunk 重入库零成本；失败降级「无前缀」并计数。
    """
    client = QdrantClient(url=cfg["qdrant"]["url"], timeout=60)
    name = collection or cfg["qdrant"]["collection"]
    if use_llm_meta is None:
        use_llm_meta = bool((cfg.get("metadata_extraction") or {}).get("enabled"))
    ctx = contextual_cfg(cfg)
    if ctx_mode is None:
        ctx_mode = "both" if ctx.get("enabled") else "off"
    if ctx_mode not in ("off", "both", "bm25"):
        raise ValueError(f"未知 ctx_mode：{ctx_mode}（可选 off / both / bm25）")
    ctx["enabled"] = ctx_mode != "off"
    ensure_collection(
        client,
        name,
        int(cfg["embedding"]["dense_dim"]),
        recreate,
        llm_meta_fields=use_llm_meta,
    )
    embedder = Embedder(cfg["embedding"])
    llm_cfg = cfg["llm"] if use_llm_meta else {}
    stats: dict = {
        "parsed": 0,
        "empty": 0,
        "docs": 0,
        "chunks": 0,
        "failed": [],
        "llm_meta": use_llm_meta,
        "ctx_mode": ctx_mode,
        "ctx_prefix_failed": 0,
    }

    # 先装载全部中间 JSON
    docs: list[tuple[Path, IntermediateDoc]] = []
    parsed_ids: set[str] = set()
    for json_file in sorted(parsed_dir.glob("*.json")):
        if json_file.name in _META_EXCLUDE:
            continue
        try:
            doc = IntermediateDoc.model_validate_json(
                json_file.read_text(encoding="utf-8")
            )
        except Exception as exc:  # noqa: BLE001
            stats["failed"].append({"file": json_file.name, "error": str(exc)})
            continue
        stats["parsed"] += 1
        parsed_ids.add(doc.meta.doc_id)
        if not doc.blocks:
            # 空文档必须被计数：改造前它被 `if doc.blocks` 静默丢弃，
            # 1127 篇解析产物只入库 1121 篇，差额在任何输出里都不存在
            stats["empty"] += 1
            continue
        docs.append((json_file, doc))
    if limit:
        docs = docs[:limit]

    # LLM 元数据是批量入库的瓶颈：并行抽取（文件名信号在 base_meta 里零成本）
    if llm_cfg:
        with ThreadPoolExecutor(max_workers=_META_WORKERS) as pool:
            metas = list(
                pool.map(lambda d: extract_metadata(d, llm_cfg), (d for _, d in docs))
            )
    else:
        metas = [base_meta(d) for _, d in docs]

    # A3.4：contextual 前缀整批先生成（可缓存；臂 ②/③ 共用同一批前缀文本）。
    # 走独立 LLM 配置（可换更便宜的模型）、关思考、失败降级「无前缀」并计数。
    ctx_prefixes: dict[str, str | None] = {}
    if ctx["enabled"]:
        chunk_lists = [chunk_by(chunk_strategy, d) for _, d in docs]
        ctx_prefixes, ctx_failed = generate_prefixes(
            ctx, [d for _, d in docs], chunk_lists
        )
        stats["ctx_prefix_failed"] = ctx_failed

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

            def indexed_text(c: Chunk) -> str:
                """被索引文本 = [contextual 前缀] + section_path 前缀 + 正文。

                `both`：dense 与 BM25 都带前缀；`bm25`：只有 BM25 带（dense 保持
                与无前缀臂同输入——区分「前缀的信息价值」与「前缀对 dense 的扰动」）。
                """
                prefix = ctx_prefixes.get(c.chunk_id)
                dense_src = _embed_text(c)
                if prefix and ctx_mode == "both":
                    dense_src = f"{prefix}\n{dense_src}"
                return dense_src

            def bm25_src(c: Chunk) -> str:
                prefix = ctx_prefixes.get(c.chunk_id)
                base = _embed_text(c)
                if prefix:  # both 与 bm25 两臂的 BM25 都带前缀
                    return f"{prefix}\n{base}"
                return base

            vectors = embedder.embed([indexed_text(c) for c in chunks])
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
                    # 前缀随 payload 落盘：读数与排查可回答「这块带没带前缀」
                    "ctx_prefix": ctx_prefixes.get(chunk.chunk_id),
                }
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                        vector={
                            "dense": vector,
                            "bm25": models.Document(
                                # A3.2 对称：与 dense 同一份被索引文本（含 A3.4
                                # 前缀臂的差异），jieba 预分词在 build_bm25_text
                                text=build_bm25_text(bm25_src(chunk)),
                                model="qdrant/bm25",
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

    # 反向对账：本次没碰过的那些旧点（源文件删了 / 改过内容）不会自己消失，
    # 而它们照样能被检索命中，带的是旧日期与旧正文。
    keep = keep_doc_ids if keep_doc_ids is not None else parsed_ids
    rec = reconcile(keep, collection_doc_ids(client, name))
    rec["source"] = "raw" if keep_doc_ids is not None else "parsed_dir"
    if keep_doc_ids is not None:
        rec["stale_parsed_json"] = sorted(parsed_ids - keep_doc_ids)
    stats["reconcile"] = rec
    if prune:
        decision = _prune_decision(rec, stats, assume_yes)
        if decision["deleted"]:
            decision["deleted"] = delete_docs(client, name, rec["orphan_doc_ids"])
        stats["prune"] = decision
    return stats

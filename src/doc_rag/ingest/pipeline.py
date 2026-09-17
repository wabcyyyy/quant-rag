"""Ingest 管线（PLAN §5.1）：三路接入 → 统一中间 JSON 落盘。

TODO(Phase 1 后半)：doc 级 LLM 元数据抽取 → BGE-M3 Embed → Qdrant upsert
（named vectors: dense + sparse，payload 建 doc_id/page/block_type 索引）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from .office import extract_office
from .pdf import extract_pdf
from .schema import IntermediateDoc

_ROUTES: dict[str, Callable[[Path], IntermediateDoc]] = {
    ".pdf": extract_pdf,
    ".docx": extract_office,
    ".doc": extract_office,
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(raw_dir: Path, parsed_dir: Path, limit: int | None = None) -> dict:
    """解析 raw_dir 全部可路由文件为中间 JSON；sha256 去重；逐文件容错。"""
    parsed_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {"total": 0, "parsed": 0, "skipped_duplicate": 0, "failed": []}
    seen: set[str] = set()
    files = [
        p
        for p in sorted(raw_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in _ROUTES
    ]
    if limit:
        files = files[:limit]
    for path in files:
        stats["total"] += 1
        digest = file_sha256(path)
        if digest in seen:
            stats["skipped_duplicate"] += 1
            continue
        seen.add(digest)
        try:
            doc = _ROUTES[path.suffix.lower()](path)
        except Exception as exc:  # noqa: BLE001 画像/入库阶段需要逐文件容错
            stats["failed"].append({"file": str(path), "error": str(exc)})
            continue
        doc.meta.doc_id = digest[:16]
        out = parsed_dir / f"{digest[:16]}.json"
        out.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        stats["parsed"] += 1
    return stats

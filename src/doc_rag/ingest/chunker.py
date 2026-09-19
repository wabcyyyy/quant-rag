"""结构感知分块（PLAN §5.1 分块默认）。

- 会议记录按议题/发言人段落优先，通用走标题层级 → 段落 → 滑动窗口
- 正文 300–600 字；标题进 section_path 不重复进正文
- 表格整块，禁止与正文混切；超长段落句级滑窗、相邻块约 12% 重叠
"""

from __future__ import annotations

import re

from .schema import Block, Chunk, IntermediateDoc

MIN_CHARS = 300
MAX_CHARS = 600
OVERLAP_RATIO = 0.12

_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")


def _sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def _slide(text: str) -> list[str]:
    """句级滑窗：块长 ≤ MAX_CHARS，相邻块句重叠约 OVERLAP_RATIO。"""
    sents = _sentences(text)
    if not sents:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(sents):
        part = ""
        end = start
        while end < len(sents) and len(part) + len(sents[end]) <= MAX_CHARS:
            part += sents[end]
            end += 1
        if end == start:  # 单句超长：硬切
            part = sents[start][:MAX_CHARS]
            end = start + 1
        parts.append(part)
        if end >= len(sents):
            break
        overlap_chars = int(MAX_CHARS * OVERLAP_RATIO)
        back = end
        acc = 0
        while back > start and acc < overlap_chars:
            back -= 1
            acc += len(sents[back])
        start = back if back > start else end
    return parts


def chunk_document(doc: IntermediateDoc) -> list[Chunk]:
    """结构感知分块（默认策略）。"""
    chunks: list[Chunk] = []
    section_path: list[str] = []
    current: list[Block] = []
    current_page: int | None = None

    def flush() -> None:
        nonlocal current, current_page
        if not current:
            return
        text = "\n".join(b.text for b in current if b.text.strip())
        for part in _slide(text) if len(text) > MAX_CHARS else [text]:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.meta.doc_id}:{len(chunks) + 1}",
                    doc_id=doc.meta.doc_id,
                    text=part,
                    section_path=list(section_path),
                    page=current_page,
                    block_type=current[0].type,
                )
            )
        current = []
        current_page = None

    for block in doc.blocks:
        if block.type == "table":
            flush()  # 表格整块，禁止与正文混切
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.meta.doc_id}:{len(chunks) + 1}",
                    doc_id=doc.meta.doc_id,
                    text=block.text,
                    section_path=list(section_path),
                    page=block.page,
                    block_type="table",
                )
            )
            continue
        if block.type == "heading" and block.heading_level:
            if sum(len(b.text) for b in current) >= MIN_CHARS:
                flush()
            level = block.heading_level
            section_path = section_path[: level - 1] + [block.text]
            continue
        if current_page is None and block.page is not None:
            current_page = block.page
        current.append(block)
        if sum(len(b.text) for b in current) >= MAX_CHARS:
            flush()
    flush()
    return chunks


def chunk_fixed(
    doc: IntermediateDoc, size: int = 512, overlap: int = 64
) -> list[Chunk]:
    """固定窗口切分（消融 #2 的对照组）。

    刻意忽略结构：把全文当纯文本按字符数硬切（表格也会被切断），
    用来量化「结构感知分块」相对「固定切分」的增益。
    """
    text = "\n".join(b.text for b in doc.blocks if b.text.strip())
    if not text:
        return []
    chunks: list[Chunk] = []
    step = max(size - overlap, 1)
    for start in range(0, len(text), step):
        part = text[start : start + size]
        if not part.strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc.meta.doc_id}:{len(chunks) + 1}",
                doc_id=doc.meta.doc_id,
                text=part,
                section_path=[],  # 固定切分无结构信息
                page=None,
                block_type="fixed",
            )
        )
        if start + size >= len(text):
            break
    return chunks


_STRATEGIES = {
    "structural": chunk_document,
    "fixed": chunk_fixed,
}


def chunk_by(strategy: str, doc: IntermediateDoc) -> list[Chunk]:
    if strategy not in _STRATEGIES:
        raise ValueError(f"未知分块策略：{strategy}（可选 {list(_STRATEGIES)}）")
    return _STRATEGIES[strategy](doc)

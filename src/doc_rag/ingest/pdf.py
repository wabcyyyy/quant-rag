"""PDF 快通道：PyMuPDF 抽取 born-digital 文本层 + find_tables 重建带框表格。

标题用字号启发识别（span 字号 ≥ 正文中位数 × 1.15 且短行，v1 层级粒度粗）。
无框线表格会退化为分段文本——质量由画像统计 + 抽样人审把关（PLAN Phase 0）；
不足率过高才触发飞书 OpenAPI 路线（PLAN §1 面试表）。
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pymupdf

from .schema import Block, IntermediateDoc, SourceMeta

_HEADING_SIZE_RATIO = 1.15
_HEADING_MAX_CHARS = 60
_TABLE_OVERLAP_RATIO = 0.5  # 文本块过半落在表格框内 → 属于表格内部
_SHORT_LINE_CHARS = 2  # ≤2 字的行视为碎化行（部分飞书导出每字符一行）
_LINE_GAP_FACTOR = 1.6  # 碎化行合并允许的纵向间距（相对字高）


def _merge_lines(lines: list[dict]) -> str:
    """块内行序列 → 文本。碎化导出（每字符一行）按纵向紧邻关系合并回正常文本。"""
    parts: list[str] = []
    buf = ""
    prev_y1: float | None = None
    for line in lines:
        text = "".join(span["text"] for span in line.get("spans", [])).strip().replace(
            "\u200b", ""
        )
        if not text:
            continue
        bbox = line["bbox"]
        height = max(bbox[3] - bbox[1], 1.0)
        short = len(text) <= _SHORT_LINE_CHARS
        gap_ok = (
            prev_y1 is not None and (bbox[1] - prev_y1) <= height * _LINE_GAP_FACTOR
        )
        if short and buf and gap_ok:
            buf += text
        elif short:
            if buf:
                parts.append(buf)
            buf = text
        else:
            if buf:
                parts.append(buf)
                buf = ""
            parts.append(text)
        prev_y1 = bbox[3]
    if buf:
        parts.append(buf)
    return "\n".join(parts)


def _overlaps_half(block_bbox: list[float], table_bbox: tuple[float, ...]) -> bool:
    bx0, by0, bx1, by1 = block_bbox
    tx0, ty0, tx1, ty1 = table_bbox
    w = max(0.0, min(bx1, tx1) - max(bx0, tx0))
    h = max(0.0, min(by1, ty1) - max(by0, ty0))
    block_area = max((bx1 - bx0) * (by1 - by0), 1e-6)
    return (w * h) / block_area > _TABLE_OVERLAP_RATIO


def _table_markdown(rows: list[list[str | None]]) -> str:
    return "\n".join(
        "| " + " | ".join((c or "").replace("\n", " ") for c in row) + " |" for row in rows
    )


def _detect_tables(page: pymupdf.Page) -> list[tuple[tuple[float, ...], str]]:
    try:
        found = page.find_tables()
    except Exception:  # noqa: BLE001 表格检测失败 → 该页退化为纯文本
        return []
    out: list[tuple[tuple[float, ...], str]] = []
    for tab in found.tables:
        rows = tab.extract()
        if not rows or not any(any(c for c in row) for row in rows):
            continue
        out.append((tuple(tab.bbox), _table_markdown(rows)))
    return out


def extract_pdf(path: Path) -> IntermediateDoc:
    doc = pymupdf.open(path)
    try:
        body_sizes: list[float] = []
        entries: list[tuple[int, float, float, Block, float]] = []

        for page_index, page in enumerate(doc):
            page_no = page_index + 1
            tables = _detect_tables(page)

            for bbox, markdown in tables:
                entries.append(
                    (
                        page_no,
                        bbox[1],
                        bbox[0],
                        Block(
                            type="table",
                            text=markdown,
                            page=page_no,
                            bbox=[round(v, 1) for v in bbox],
                        ),
                        0.0,
                    )
                )

            for block in page.get_text("dict")["blocks"]:
                if block.get("type") == 1:  # 图片：v1 丢弃，Phase 3 做 caption 入库
                    continue
                bbox = block.get("bbox")
                if bbox is None:
                    continue
                text = _merge_lines(block.get("lines", []))
                if not text:
                    continue
                if any(_overlaps_half(bbox, tbbox) for tbbox, _ in tables):
                    continue  # 单元格文字已由 find_tables 重建
                max_size = 0.0
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        if span["text"].strip():
                            body_sizes.append(span["size"])
                            max_size = max(max_size, span["size"])
                entries.append(
                    (
                        page_no,
                        bbox[1],
                        bbox[0],
                        Block(
                            type="paragraph",
                            text=text,
                            page=page_no,
                            bbox=[round(v, 1) for v in bbox],
                        ),
                        max_size,
                    )
                )

        body_median = statistics.median(body_sizes) if body_sizes else 10.0
        blocks: list[Block] = []
        for _, _, _, block, max_size in sorted(entries, key=lambda e: (e[0], e[1], e[2])):
            if (
                block.type == "paragraph"
                and max_size >= body_median * _HEADING_SIZE_RATIO
                and len(block.text) <= _HEADING_MAX_CHARS
                and "\n" not in block.text
            ):
                block.type = "heading"
                block.heading_level = 2
            blocks.append(block)
    finally:
        doc.close()
    return IntermediateDoc(
        meta=SourceMeta(source_type="pdf", doc_id=path.stem, title=path.stem),
        blocks=blocks,
    )

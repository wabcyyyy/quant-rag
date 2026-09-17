"""语料画像（PLAN Phase 0）：来源构成、文本可抽性、表格/图片占比、疑似扫描件。

「疑似扫描件」判定：PDF 平均每页可抽文本 < 50 字符 → 启用 MinerU 兜底的依据。
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

_SCAN_CHARS_PER_PAGE = 50


def profile_pdf(path: Path) -> dict:
    doc = pymupdf.open(path)
    try:
        pages = len(doc)
        empty_pages = 0
        total_chars = 0
        images = 0
        for page in doc:
            text = page.get_text("text").strip()
            total_chars += len(text)
            if len(text) < 20:
                empty_pages += 1
            images += len(page.get_images(full=True))
        chars_per_page = round(total_chars / pages, 1) if pages else 0.0
        return {
            "file": path.name,
            "pages": pages,
            "chars_per_page": chars_per_page,
            "near_empty_pages": empty_pages,
            "images": images,
            "scan_suspect": chars_per_page < _SCAN_CHARS_PER_PAGE,
        }
    finally:
        doc.close()


def run(raw_dir: Path, out_file: Path | None = None) -> dict:
    summary: dict[str, int] = {}
    pdf_details: list[dict] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower() or "(无后缀)"
        summary[suffix] = summary.get(suffix, 0) + 1
        if suffix == ".pdf":
            try:
                pdf_details.append(profile_pdf(path))
            except Exception as exc:  # noqa: BLE001 画像需要逐文件容错
                pdf_details.append({"file": path.name, "error": str(exc)})
    result = {
        "total_files": sum(summary.values()),
        "source_distribution": summary,
        "pdf_details": pdf_details,
        "scan_suspects": [d["file"] for d in pdf_details if d.get("scan_suspect")],
        "pdf_errors": [d["file"] for d in pdf_details if "error" in d],
    }
    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result

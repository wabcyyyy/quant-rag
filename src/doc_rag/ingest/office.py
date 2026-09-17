"""Office 通道：.docx 直读；.doc 先经 LibreOffice headless 归一为 docx（PLAN §2）。

mammoth 输出 Markdown 后解析回 Block（标题层级来自 # 数量）。
已知限制：mammoth 对表格的 Markdown 输出可能退化——Phase 1 画像若显示表格占比高，
改走 docx→HTML 保留 <table>（TODO）。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import mammoth

from .schema import Block, IntermediateDoc, SourceMeta

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^[-*]\s+|^\d+[.、)]\s*")
_CONVERT_TIMEOUT_SECONDS = 120


def _doc_to_docx(path: Path, out_dir: Path) -> Path:
    soffice = shutil.which("soffice") or shutil.which("soffice.exe")
    if soffice is None:
        raise RuntimeError(
            "未找到 LibreOffice（soffice）。.doc 需先经 LibreOffice 归一（PLAN §2 文件解析）；"
            "Windows 安装后需将其 bin 目录加入 PATH。"
        )
    subprocess.run(
        [
            soffice,
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            str(out_dir),
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=_CONVERT_TIMEOUT_SECONDS,
    )
    return out_dir / (path.stem + ".docx")


def _markdown_to_blocks(markdown: str) -> list[Block]:
    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(Block(type="paragraph", text="\n".join(paragraph).strip()))
            paragraph.clear()

    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if m := _HEADING_RE.match(stripped):
            flush()
            blocks.append(
                Block(type="heading", heading_level=len(m.group(1)), text=m.group(2).strip())
            )
        elif stripped.startswith("|") and stripped.endswith("|"):
            flush()
            blocks.append(Block(type="table", text=stripped))
        elif _LIST_RE.match(stripped):
            flush()
            blocks.append(Block(type="list_item", text=_LIST_RE.sub("", stripped, count=1)))
        else:
            paragraph.append(stripped)
    flush()
    return blocks


def extract_office(path: Path) -> IntermediateDoc:
    with tempfile.TemporaryDirectory() as tmp:
        src = path
        source_type = "docx"
        if path.suffix.lower() == ".doc":
            src = _doc_to_docx(path, Path(tmp))
            source_type = "doc"
        with src.open("rb") as f:
            result = mammoth.convert_to_markdown(f)
    return IntermediateDoc(
        meta=SourceMeta(source_type=source_type, doc_id=path.stem, title=path.stem),
        blocks=_markdown_to_blocks(result.value),
    )

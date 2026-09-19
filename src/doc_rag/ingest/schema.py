"""统一中间表示（PLAN §5.1）：双路来源归一化到同一套 Block/Chunk。

语料为飞书/Word 批量导出后手动迁入 data/raw，不对接平台 API。
- doc/docx → LibreOffice headless 归一 → mammoth
- PDF      → PyMuPDF 快通道（born-digital 文本层）
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Block(BaseModel):
    """中间 JSON 的最小单元。page/bbox 仅 PDF 来源携带。"""

    type: str  # heading | paragraph | list_item | table | quote
    text: str
    heading_level: int | None = None
    page: int | None = None
    bbox: list[float] | None = None


class SourceMeta(BaseModel):
    source_type: str  # docx | doc | pdf
    doc_id: str
    title: str | None = None
    # 文件来源默认为 None，由 LLM 元数据抽取补齐（PLAN §5.1）
    owner: str | None = None
    created_at: str | None = None
    edited_at: str | None = None


class IntermediateDoc(BaseModel):
    meta: SourceMeta
    blocks: list[Block] = Field(default_factory=list)

    def to_text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.text.strip())


class Chunk(BaseModel):
    """检索块（PLAN §5.1 分块默认：300–600 字、表格整块不切）。"""

    chunk_id: str
    doc_id: str
    text: str
    section_path: list[str] = Field(default_factory=list)
    page: int | None = None
    block_type: str = "paragraph"

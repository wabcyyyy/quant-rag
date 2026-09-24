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
    # 文本来源（A3.3）：None = 解析器文本层；"ocr" = OCR 兜底产物。
    # 不进 Qdrant payload、不参与检索，只在画像与排查时回答「这段字哪来的」。
    source: str | None = None


class SourceMeta(BaseModel):
    source_type: str  # docx | doc | pdf
    doc_id: str
    title: str | None = None
    # 文件来源默认为 None，由 LLM 元数据抽取补齐（PLAN §5.1）
    owner: str | None = None
    created_at: str | None = None
    edited_at: str | None = None
    # OCR 兜底状态（A3.3）：None=未触发（文本层正常）；"ocr_applied"=已兜底；
    # "ocr_unavailable"=疑似扫描件但引擎不可用（ingest 汇总必须点名，不许静默）；
    # "near_empty"=无图少字，没有可识别对象，OCR 帮不上。
    ocr_status: str | None = None


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

    def indexed_text(self) -> str:
        """被索引文本：section_path 前缀 + 正文，dense 与 BM25 共用的同一份。

        为什么必须在 schema 层给唯一实现（A3.2，q019 类缺陷的修复）：改前 dense
        的向量输入带 section_path 前缀，BM25 分词文本却是纯正文——「2025年第36周」
        「议题5」这类只出现在标题/路径里的定位词在 dense 可配、在 BM25 零出现，
        两路看到的是不同的索引。前缀组合逻辑只写在这一处，两个调用方引用它，
        「同一 chunk 的 dense 前缀词集合 ⊆ BM25 词集合」由构造保证。
        """
        prefix = " / ".join(self.section_path)
        return f"{prefix}\n{self.text}" if prefix else self.text

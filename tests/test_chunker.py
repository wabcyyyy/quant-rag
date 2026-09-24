from doc_rag.ingest.chunker import chunk_document
from doc_rag.ingest.schema import Block, IntermediateDoc, SourceMeta


def _doc(blocks: list[Block]) -> IntermediateDoc:
    return IntermediateDoc(
        meta=SourceMeta(source_type="docx", doc_id="t1"), blocks=blocks
    )


def test_table_is_never_merged_with_prose():
    doc = _doc(
        [
            Block(type="paragraph", text="前文" * 50),
            Block(type="table", text="| 列1 | 列2 |"),
        ]
    )
    chunks = chunk_document(doc)
    table_chunks = [c for c in chunks if c.block_type == "table"]
    assert len(table_chunks) == 1
    assert "前文" not in table_chunks[0].text
    assert table_chunks[0].text == "| 列1 | 列2 |"


def test_heading_builds_section_path():
    doc = _doc(
        [
            Block(type="heading", heading_level=1, text="会议纪要"),
            Block(type="heading", heading_level=2, text="议题A"),
            Block(type="paragraph", text="内容" * 100),
        ]
    )
    chunks = chunk_document(doc)
    assert chunks[0].section_path == ["会议纪要", "议题A"]
    assert all("会议纪要" not in c.text for c in chunks)  # 标题不重复进正文


def test_long_paragraph_slides_with_overlap():
    doc = _doc([Block(type="paragraph", text="这是第一句。" * 300)])
    chunks = chunk_document(doc)
    assert len(chunks) > 1
    assert all(len(c.text) <= 600 for c in chunks)
    # 相邻块有句级重叠
    assert chunks[0].text[-6:] in chunks[1].text


def test_chunk_ids_are_unique():
    doc = _doc(
        [
            Block(type="paragraph", text="甲" * 700),
            Block(type="paragraph", text="乙" * 700),
        ]
    )
    chunks = chunk_document(doc)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


# ── A3.1：标题文本必须进入被索引文本（q020 类缺陷的根形修复）────────────────


def _indexed_blob(chunks):
    """被索引文本的离线替身：dense 前缀（section_path）+ 块正文。"""
    return "\n".join(
        (" / ".join(c.section_path) + "\n" + c.text) if c.section_path else c.text
        for c in chunks
    )


def test_trailing_heading_anchor_is_indexed():
    """锚点词只出现在文档末尾的标题里：改前它不进任何块，检索永远够不着。"""
    doc = _doc(
        [
            Block(type="paragraph", text="正文内容" * 100),
            Block(type="heading", heading_level=2, text="动议区"),
        ]
    )
    blob = _indexed_blob(chunk_document(doc))
    assert "动议区" in blob


def test_consecutive_trailing_headings_are_indexed():
    doc = _doc(
        [
            Block(type="paragraph", text="正文内容" * 100),
            Block(type="heading", heading_level=2, text="动议区"),
            Block(type="heading", heading_level=2, text="投票区"),
        ]
    )
    blob = _indexed_blob(chunk_document(doc))
    assert "动议区" in blob and "投票区" in blob


def test_heading_before_table_enters_table_section_path():
    doc = _doc(
        [
            Block(type="heading", heading_level=2, text="预算明细"),
            Block(type="table", text="| 项目 | 金额 |"),
        ]
    )
    chunks = chunk_document(doc)
    table = next(c for c in chunks if c.block_type == "table")
    assert table.section_path == ["预算明细"]


def test_heading_before_content_enters_content_section_path():
    doc = _doc(
        [
            Block(type="heading", heading_level=2, text="议题五"),
            Block(type="paragraph", text="短正文"),
        ]
    )
    chunks = chunk_document(doc)
    assert any("议题五" in c.section_path for c in chunks if c.block_type != "heading")


def test_interior_heading_semantics_unchanged():
    """提交语义与改前逐字一致：level-2 标题把前者当父级、路径叠加；
    累积 ≥ MIN_CHARS 的正文先落块（用旧路径），标题提交只影响后随块。"""
    doc = _doc(
        [
            Block(type="heading", heading_level=2, text="议题A"),
            Block(type="paragraph", text="甲" * 300),
            Block(type="heading", heading_level=2, text="议题B"),
            Block(type="paragraph", text="乙" * 100),
        ]
    )
    chunks = chunk_document(doc)
    paths = [c.section_path for c in chunks if c.block_type != "heading"]
    assert ["议题A"] == paths[0]
    assert ["议题A", "议题B"] == paths[-1]


# ── A3.2：dense 与 BM25 的被索引文本对称（q019 类缺陷的修复）────────────────


def _doc_with_path(blocks):
    return IntermediateDoc(
        meta=SourceMeta(source_type="docx", doc_id="t2"), blocks=blocks
    )


def test_bm25_text_contains_the_dense_prefix():
    """同一 chunk 的 dense 前缀词集合 ⊆ BM25 词集合（构造上同源）。

    「2025年第36周」「议题5」这类只出现在标题里的定位词，改前只进 dense 的
    section_path 前缀，BM25 分词文本（纯正文）没有——两路看到的索引不同。
    """
    from doc_rag.ingest.bm25 import build_bm25_text

    doc = _doc_with_path(
        [
            Block(type="heading", heading_level=2, text="2025年第36周 议题5"),
            Block(type="paragraph", text="正文：动议区表决通过。"),
        ]
    )
    for chunk in chunk_document(doc):
        dense_text = chunk.indexed_text()
        bm25_text = build_bm25_text(dense_text)
        # 前缀的所有非空白字符都在 BM25 文本里（jieba 只切分、不丢字）
        prefix = " / ".join(chunk.section_path)
        assert _norm_str(prefix) in _norm_str(bm25_text)


def _norm_str(s: str) -> str:
    import re

    return re.sub(r"\s+", "", s)


def test_embed_text_and_bm25_share_one_composition():
    """indexer._embed_text 与 BM25 的输入必须逐字同源（唯一实现）。"""
    import inspect

    from doc_rag.ingest import indexer

    source = inspect.getsource(indexer)
    assert "build_bm25_text(_embed_text(chunk))" in source

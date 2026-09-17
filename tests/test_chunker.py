from doc_rag.ingest.chunker import chunk_document
from doc_rag.ingest.schema import Block, IntermediateDoc, SourceMeta


def _doc(blocks: list[Block]) -> IntermediateDoc:
    return IntermediateDoc(meta=SourceMeta(source_type="docx", doc_id="t1"), blocks=blocks)


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

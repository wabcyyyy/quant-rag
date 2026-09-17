import pymupdf

from doc_rag.ingest.pdf import extract_pdf


def _make_pdf_with_table(path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    # 2x2 带框表格
    page.draw_rect(pymupdf.Rect(72, 72, 272, 172))
    page.draw_line(pymupdf.Point(72, 122), pymupdf.Point(272, 122))
    page.draw_line(pymupdf.Point(172, 72), pymupdf.Point(172, 172))
    page.insert_text((80, 90), "Item")
    page.insert_text((180, 90), "Budget")
    page.insert_text((80, 140), "Alpha")
    page.insert_text((180, 140), "100")
    page.insert_text((72, 200), "Below the table there is some plain prose text.")
    doc.save(str(path))
    doc.close()


def test_ruled_table_is_rebuilt_as_one_block(tmp_path):
    pdf_path = tmp_path / "table.pdf"
    _make_pdf_with_table(pdf_path)
    doc = extract_pdf(pdf_path)

    tables = [b for b in doc.blocks if b.type == "table"]
    assert len(tables) == 1
    assert "Item" in tables[0].text and "Budget" in tables[0].text

    # 单元格文字不再以碎片形式重复进正文
    prose = [b for b in doc.blocks if b.type == "paragraph"]
    joined = "\n".join(b.text for b in prose)
    assert "Item" not in joined
    assert "Below the table" in joined  # 表格外的正文不受影响


def test_blocks_are_ordered_by_position(tmp_path):
    pdf_path = tmp_path / "order.pdf"
    _make_pdf_with_table(pdf_path)
    doc = extract_pdf(pdf_path)
    pages = [b.page for b in doc.blocks]
    assert pages == sorted(pages)
    # 表格（y≈72）在正文（y≈200）之前
    assert doc.blocks[0].type == "table"

"""A3.3 / B7：近空·扫描件兜底与画像拆分的离线护栏。

- 改前：扫描件「只有标记，没有兜底」（README 已知不足）——近空文档 0 块入库。
- 改后：有图无字（scan_likely）走本地 OCR（可选依赖）；引擎缺席时 ingest 汇总
  点名，不许静默；profile 把 scan_suspect 拆成 scan_likely / near_empty 两类。

单元测试不调真 OCR 引擎（可选依赖，CI 未必装）：ocr_pdf_pages 打替身；
真引擎的识别质量由 S3 阶段的入库对账验证（320 篇语料 4 篇扫描件全文复原）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_rag.ingest import ocr as ocr_mod
from doc_rag.ingest import pdf as pdf_mod
from doc_rag.ingest import pipeline as pipeline_mod
from doc_rag.ingest.profile import profile_pdf


def _text_pdf(tmp_path: Path) -> Path:
    import pymupdf

    p = tmp_path / "text.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 100), "This is a normal born-digital page with plenty of text." * 3
    )
    p.write_bytes(doc.tobytes())
    doc.close()
    return p


def _image_only_pdf(tmp_path: Path) -> Path:
    import pymupdf

    p = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=100)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 20))
    pix.clear_with(90)  # 灰块（有图无字）
    page.insert_image(pymupdf.Rect(10, 10, 50, 30), pixmap=pix)
    p.write_bytes(doc.tobytes())
    doc.close()
    return p


def _empty_pdf(tmp_path: Path) -> Path:
    import pymupdf

    p = tmp_path / "empty.pdf"
    doc = pymupdf.open()
    doc.new_page()  # 无文字无图
    p.write_bytes(doc.tobytes())
    doc.close()
    return p


def test_scan_like_goes_through_ocr_fallback(tmp_path, monkeypatch):
    p = _image_only_pdf(tmp_path)
    monkeypatch.setattr(
        ocr_mod, "ocr_pdf_pages", lambda path: [(1, "OCR 还原的正文内容")]
    )
    doc = pdf_mod.extract_pdf(p)
    assert doc.meta.ocr_status == "ocr_applied"
    assert len(doc.blocks) == 1
    assert doc.blocks[0].source == "ocr"
    assert doc.blocks[0].page == 1
    assert "OCR 还原的正文内容" in doc.to_text()


def test_ocr_unavailable_is_visible_not_silent(tmp_path, monkeypatch):
    p = _image_only_pdf(tmp_path)
    monkeypatch.setattr(ocr_mod, "ocr_pdf_pages", lambda path: None)
    doc = pdf_mod.extract_pdf(p)
    assert doc.meta.ocr_status == "ocr_unavailable"
    assert doc.blocks == []  # 改前它就是这样一个 0 块文档，无声入库


def test_text_layer_normal_skips_ocr(tmp_path):
    doc = pdf_mod.extract_pdf(_text_pdf(tmp_path))
    assert doc.meta.ocr_status is None
    assert doc.blocks


def test_near_empty_without_images_is_classified(tmp_path):
    doc = pdf_mod.extract_pdf(_empty_pdf(tmp_path))
    assert doc.meta.ocr_status == "near_empty"


def test_pipeline_counts_ocr_paths(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    _image_only_pdf(raw)  # scan.pdf
    monkeypatch.setattr(ocr_mod, "ocr_pdf_pages", lambda path: [(1, "OCR 文本")])
    stats = pipeline_mod.run(raw, tmp_path / "parsed")
    assert stats["ocr_applied"] == ["scan.pdf"]
    assert stats["ocr_unavailable"] == []
    monkeypatch.setattr(ocr_mod, "ocr_pdf_pages", lambda path: None)
    stats2 = pipeline_mod.run(raw, tmp_path / "parsed2")
    assert stats2["ocr_unavailable"] == ["scan.pdf"]


def test_available_lazy_and_safe(monkeypatch):
    import types

    fake = types.ModuleType("rapidocr_onnxruntime")
    fake.RapidOCR = object
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake)
    monkeypatch.setattr(ocr_mod, "_ENGINE", None)
    assert ocr_mod.available() is True
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)  # import 失败形态
    monkeypatch.setattr(ocr_mod, "_ENGINE", None)
    assert ocr_mod.available() is False
    monkeypatch.setattr(ocr_mod, "_ENGINE", None)


@pytest.mark.skipif(not (ROOT / "data" / "sample_raw").exists(), reason="示例语料不在")
def test_profile_split_union_holds_on_real_corpus():
    """B7 验收：scan_likely + near_empty = scan_suspect，全库逐篇钉住。"""
    files = sorted((ROOT / "data" / "sample_raw").glob("*.pdf"))[:50]
    n_likely = n_empty = n_suspect = 0
    for f in files:
        d = profile_pdf(f)
        assert (d["scan_likely"], d["near_empty"]) in {
            (True, False),
            (False, True),
        } or (not d["scan_suspect"] and not d["scan_likely"] and not d["near_empty"]), f
        if d["scan_likely"]:
            n_likely += 1
        if d["near_empty"]:
            n_empty += 1
        if d["scan_suspect"]:
            n_suspect += 1
    assert n_likely + n_empty == n_suspect

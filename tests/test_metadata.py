from doc_rag.ingest.bm25 import build_bm25_text
from doc_rag.ingest.metadata import date_from_filename, normalize_date, parse_llm_json


def test_date_from_filename():
    assert date_from_filename("20260901-XX项目周会.pdf") == "2026-09-01"
    assert date_from_filename("2026-9-1 周会.docx") == "2026-09-01"
    assert date_from_filename("2026年9月1日会议纪要.doc") == "2026-09-01"
    assert date_from_filename("会议纪要.pdf") is None
    assert date_from_filename("20261399.pdf") is None  # 非法月日


def test_normalize_date():
    assert normalize_date("2026-09-01") == "2026-09-01"
    assert normalize_date("2026-9-1") == "2026-09-01"
    assert normalize_date("2026年9月1日") is None  # LLM 偶发返回中文格式 → 拒绝
    assert normalize_date(None) is None


def test_parse_llm_json():
    assert parse_llm_json('```json\n{"date": "2026-09-01"}\n```') == {
        "date": "2026-09-01"
    }
    assert parse_llm_json('前缀 {"topics": ["预算"]} 后缀') == {"topics": ["预算"]}
    assert parse_llm_json("不是 json") is None
    assert parse_llm_json("") is None


def test_bm25_text_segments_chinese():
    out = build_bm25_text("供应商预付款政策。")
    assert " " in out  # 已分词
    assert "供应商" in out and "预付款" in out

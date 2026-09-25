"""LLM 措辞鲁棒性臂合并脚本——校验失败整批拒绝，产物逐条过 GoldItem 契约。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_rag.eval.schema import GoldItem

_spec = importlib.util.spec_from_file_location(
    "make_llm_rephrase", ROOT / "scripts" / "make_llm_rephrase.py"
)
mod = importlib.util.module_from_spec(_spec)
sys.modules["make_llm_rephrase"] = mod
_spec.loader.exec_module(mod)


def _gold() -> dict:
    return {
        "meta": {"gold_version": "gold_core", "sample": True, "count": 2},
        "items": [
            {
                "id": "c001",
                "type": "fact",
                "question": "《会议档案_专题会_2025-11-06-会议纪要》中「设备购置」的金额是多少？",
                "expected_answer": "差旅意外保险的预算为 197 万元。",
                "must_contain": ["设备购置", "197 万元"],
                "source_doc_ids": ["3ecdd843a3effae0"],
                "refusable": False,
                "source_title": "会议档案_专题会_2025-11-06-会议纪要",
                "origin": "programmatic",
                "key_points": [],
            },
            {
                "id": "c002",
                "type": "no_answer",
                "question": "公司有碳排放配额管理制度吗？",
                "expected_answer": "文档未记载。",
                "must_contain": [],
                "source_doc_ids": [],
                "refusable": True,
                "source_title": None,
                "origin": "programmatic",
                "key_points": [],
            },
        ],
    }


def _chunk_file(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_chunks_accepts_map_file_with_meta(tmp_path):
    """gold_core_llm_map.json 形态：dict 带 meta/variants——只消费 variants。"""
    p = tmp_path / "map.json"
    p.write_text(
        json.dumps(
            {
                "meta": {"purpose": "出处"},
                "variants": [{"id": "c001", "r1": "a？", "r2": "b？"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert mod.load_chunks([p]) == {"c001": ("a？", "b？")}


def test_merge_basic_two_variants_per_item(tmp_path):
    variants = {
        "c001": (
            "当时给差旅保险买设备批了多少钱来着？",
            "差旅保险那笔预算里设备购置具体批了多少？",
        ),
        "c002": ("咱们公司管碳排放配额不？", "公司对碳排放配额有没有成文的管理制度？"),
    }
    payload = mod.merge(_gold(), variants)

    assert payload["meta"]["count"] == 4
    assert payload["meta"]["gold_version"] == "gold_core_llm"
    assert payload["meta"]["derived_from"].startswith("gold_core")
    assert "不是外部锚点" in payload["meta"]["note"]
    ids = [i["id"] for i in payload["items"]]
    assert ids == ["c001r1", "c001r2", "c002r1", "c002r2"]
    q = {i["id"]: i for i in payload["items"]}
    # 判据逐字段继承，只有题面与 origin 变
    assert q["c001r1"]["must_contain"] == ["设备购置", "197 万元"]
    assert q["c001r1"]["origin"] == "llm_rephrase"
    assert q["c001r1"]["question"].startswith("当时给差旅保险")
    assert q["c002r2"]["refusable"] is True
    assert q["c002r2"]["question"] == "公司对碳排放配额有没有成文的管理制度？"
    assert payload["meta"]["type_distribution"] == {"fact": 2, "no_answer": 2}
    # eval 消费契约：逐条能构造 GoldItem
    for item in payload["items"]:
        GoldItem(**item)


def test_missing_coverage_fails_listing_ids():
    variants = {"c001": ("a？", "b？")}  # c002 缺
    with pytest.raises(ValueError, match="未覆盖 1 条.*c002"):
        mod.merge(_gold(), variants)


def test_load_chunks_rejects_duplicate_ids(tmp_path):
    p1 = _chunk_file(tmp_path / "c1.json", [{"id": "c001", "r1": "a？", "r2": "b？"}])
    p2 = _chunk_file(tmp_path / "c2.json", [{"id": "c001", "r1": "c？", "r2": "d？"}])
    with pytest.raises(ValueError, match="重复 id: c001"):
        mod.load_chunks([p1, p2])


def test_degenerate_variants_rejected():
    base = {"c001": ("正常问法？", "另一种问法？"), "c002": ("x？", "y？")}
    # r1 与 r2 相同
    bad_same = {**base, "c001": ("一样的？", "一样的？")}
    with pytest.raises(ValueError, match="r1 与 r2 相同"):
        mod.merge(_gold(), bad_same)
    # 变体为空
    bad_empty = {**base, "c002": ("x？", "  ")}
    with pytest.raises(ValueError, match="c002r2: 变体为空"):
        mod.merge(_gold(), bad_empty)
    # 与原题逐字相同（偷懒复读）
    orig_q = _gold()["items"][0]["question"]
    bad_copy = {**base, "c001": (orig_q, "另一种问法？")}
    with pytest.raises(ValueError, match="与原题逐字相同"):
        mod.merge(_gold(), bad_copy)

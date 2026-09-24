"""B1：人工复核一致率脚本的离线护栏（合成输入，含 κ 边界）。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "human_review_agreement", ROOT / "scripts" / "human_review_agreement.py"
)
mod = importlib.util.module_from_spec(_spec)
sys.modules["human_review_agreement"] = mod
_spec.loader.exec_module(mod)


def test_kappa_known_values():
    assert mod.cohen_kappa(20, 5, 5, 20) == 0.6
    assert mod.cohen_kappa(0, 0, 0, 10) is None  # 单边恒定 → 无定义
    assert mod.cohen_kappa(0, 0, 0, 0) is None


def test_analyze_counts_and_disagreements():
    review = {
        "study": "s",
        "date": "d",
        "judge": "非独立第三方",
        "grades": [
            {"id": "q1", "type": "fact", "G": 2, "F": 0},
            {"id": "q2", "type": "fact", "G": 0, "F": 0, "note": "检索没给到"},
            {"id": "q3", "type": "term", "G": 2, "F": 1, "note": "编造了一处"},
            {"id": "q4", "type": "term", "G": 1, "F": 0},  # G=1 → human_ok False
            {"id": "q9", "type": "fact", "G": 2, "F": 0},  # 不在结果文件 → skip
        ],
    }
    run = {
        "items": [
            {"id": "q1", "answered_ok": True},
            {"id": "q2", "answered_ok": False},
            {"id": "q3", "answered_ok": False},  # 人工 G=2（human_ok）但自动判否 → 分歧
            {"id": "q4", "answered_ok": None},  # 拒答条 → skip（None 不冒充 0）
        ]
    }
    out = mod.analyze(review, run)
    assert out["n_compared"] == 3
    assert out["n_skipped"] == 2
    t = out["table_2x2"]
    assert t == {
        "human_ok_auto_ok": 1,  # q1
        "human_ok_auto_ng": 1,  # q3：人工可（G=2）自动否
        "human_ng_auto_ok": 0,
        "human_ng_auto_ng": 1,  # q2：人工不可自动也不可
    }
    assert out["agreement_rate"] == round(2 / 3, 4)
    assert out["cohen_kappa"] == 0.4
    assert out["cohen_kappa"] is not None
    assert any(d["id"] == "q3" for d in out["disagreements"])
    assert "非独立第三方" in out["limitation"]

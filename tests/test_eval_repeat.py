"""B5：eval --repeat 的合并口径——极差即噪声地板，主口径取第一遍。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from doc_rag.eval import runner as eval_runner


def _run_summary(acc: float, n_items: int = 2) -> dict:
    return {
        "n_items": n_items,
        "strict_keyword_accuracy": acc,
        "hit_at_5": 0.5,
        "mrr": 0.5,
        "list_len": {"retrieved_min": 8},
        "latency": {"total": {"p50": 100.0}},
    }


def _run(acc: float, item_answer: str) -> dict:
    return {
        "meta": {"timestamp": "t"},
        "summary": _run_summary(acc),
        "ragas": None,
        "items": [{"id": "q1", "answer": item_answer}],
    }


def test_repeat_merges_span_and_keeps_first_pass_primary(monkeypatch, tmp_path):
    scripted = [_run(0.5, "甲"), _run(1.0, "乙")]
    calls = []

    def fake_evaluate(*args, **kwargs):
        calls.append(kwargs)
        return scripted[len(calls) - 1]

    monkeypatch.setattr(eval_runner, "evaluate", fake_evaluate)
    merged = eval_runner.evaluate_with_repeat(
        tmp_path / "gold.json", cfg={}, repeat=2, top_n=8
    )
    # 主口径 = 第一遍（既有消费方读顶层零改动）
    assert merged["summary"]["strict_keyword_accuracy"] == 0.5
    assert merged["items"][0]["answer"] == "甲"
    # 极差 = 跨遍 max−min，只对全数值键
    assert merged["summary"]["repeat_span"]["strict_keyword_accuracy"] == 0.5
    assert merged["summary"]["repeat_span"]["hit_at_5"] == 0.0
    assert "list_len" not in merged["summary"]["repeat_span"]  # 字典键不算
    assert "latency" not in merged["summary"]["repeat_span"]
    # 每一遍逐条保留
    assert [r["summary"]["strict_keyword_accuracy"] for r in merged["repeat_runs"]] == [
        0.5,
        1.0,
    ]
    assert merged["meta"]["repeat"] == 2


def test_repeat_one_returns_run_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_runner, "evaluate", lambda *a, **k: _run(0.75, "唯一一遍"))
    merged = eval_runner.evaluate_with_repeat(tmp_path / "gold.json", cfg={}, repeat=1)
    assert "repeat_span" not in merged["summary"]
    assert "repeat_runs" not in merged


def test_repeat_rejects_zero():
    with pytest.raises(ValueError):
        eval_runner.evaluate_with_repeat("x", repeat=0)

"""B5：eval --repeat 的合并口径——极差即噪声地板，主口径取第一遍。

N9/N12 追加：resume 必须真的进 repeat（且只给第一遍）；答案侧缓存闸门。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest
import typer

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


# ── N12：resume 必须真的进 repeat ───────────────────────────────────────


def test_cli_repeat_branch_passes_resume(monkeypatch, tmp_path):
    """`--repeat 3 --resume X` 不得静默丢弃 --resume（旧实现重复跑重付）。"""
    from typer.testing import CliRunner

    from doc_rag import cli

    seen: dict = {}

    def _fake_repeat(gold_file, cfg=None, repeat=1, **kw):
        seen.update(kw)
        seen["repeat"] = repeat
        raise typer.Exit(0)

    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"items": []}), encoding="utf-8")
    resume = tmp_path / "prior.json"
    resume.write_text("{}", encoding="utf-8")
    cfg = {
        "retrieval": {},
        "llm": {"model": "m"},
        "eval": {"gold_file": str(gold), "ragas_sample": 0},
        "paths": {"eval": str(tmp_path)},
    }
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(eval_runner, "evaluate_with_repeat", _fake_repeat)
    res = CliRunner().invoke(
        cli.app,
        ["eval", "--gold", str(gold), "--repeat", "3", "--resume", str(resume)],
    )
    assert res.exit_code == 0, res.output
    assert seen.get("resume_from") == resume


def test_repeat_gives_resume_to_first_run_only(monkeypatch, tmp_path):
    """第 2 遍起必须裸跑：若也 resume，全部条目被复用 → repeat_span 恒 0（假地板）。"""
    calls: list[dict] = []

    def fake_evaluate(*args, **kwargs):
        calls.append(kwargs)
        return {"meta": {}, "summary": {}, "items": []}

    monkeypatch.setattr(eval_runner, "evaluate", fake_evaluate)
    eval_runner.evaluate_with_repeat(
        tmp_path / "gold.json", cfg={}, repeat=3, resume_from="prior.json"
    )
    assert len(calls) == 3
    assert calls[0].get("resume_from") == "prior.json"
    assert calls[1].get("resume_from") is None
    assert calls[2].get("resume_from") is None


# ── N9：答案侧缓存闸门 ──────────────────────────────────────────────────


def _patch_fake_eval(monkeypatch):
    monkeypatch.setattr(
        eval_runner,
        "evaluate",
        lambda *a, **kw: {"meta": {}, "summary": {}, "items": []},
    )


def test_repeat_with_answers_refuses_when_cache_on(monkeypatch, tmp_path):
    """不加 --fresh-answers 的 --repeat 必须报错：缓存命中把地板测成 0（N9）。"""
    _patch_fake_eval(monkeypatch)
    monkeypatch.setenv("DOC_RAG_LLM_CACHE", "1")
    with pytest.raises(ValueError, match="fresh-answers"):
        eval_runner.evaluate_with_repeat(
            tmp_path / "gold.json",
            cfg={"llm": {"cache": True}},
            repeat=2,
            with_answers=True,
        )


def test_repeat_with_answers_allowed_when_cache_off(monkeypatch, tmp_path):
    """--fresh-answers 之后（env 关缓存）必须放行——这是 B5 的指定用法。"""
    _patch_fake_eval(monkeypatch)
    monkeypatch.setenv("DOC_RAG_LLM_CACHE", "0")
    out = eval_runner.evaluate_with_repeat(
        tmp_path / "gold.json",
        cfg={"llm": {"cache": True}},
        repeat=2,
        with_answers=True,
    )
    assert out["meta"]["repeat"] == 2


def test_repeat_retrieval_only_allows_cache(monkeypatch, tmp_path):
    """检索侧地板不依赖答案缓存：retrieval-only 的 repeat 不经此闸。"""
    _patch_fake_eval(monkeypatch)
    monkeypatch.setenv("DOC_RAG_LLM_CACHE", "1")
    out = eval_runner.evaluate_with_repeat(
        tmp_path / "gold.json",
        cfg={"llm": {"cache": True}},
        repeat=2,
        with_answers=False,
    )
    assert out["meta"]["repeat"] == 2

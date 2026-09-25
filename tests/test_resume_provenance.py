"""N11：`--resume` 必须做 provenance 全等校验。

旧实现只按「id 存在且无 error」复用条目，不比对任何状态；而最终 meta 用本轮
cfg 现算——可以合法产出「items 来自状态 A、meta 自称状态 B」的结果文件。下游
（compare 等长护栏、freeze_warnings、RAGAS 上下文重放）全部会基于错误的 meta
放行。judge 侧早有同类闸门（上下文无法复现即拒判），检索/答案侧这里补齐：
`index_fp` / `synth_fp` / `collection` / `prompt_fingerprint` 四键与本轮**全等**
才允许 resume；被复用条目带 `resumed_from` 标记，混源一眼可见。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_rag.eval import runner as eval_runner
from doc_rag.orchestrator import Result


class _FakeOrch:
    def __init__(self):
        self.calls: list[str] = []
        self.retriever = type("R", (), {"collection": "t", "cfg": {"mode": "hybrid"}})()

    def answer(self, question, **kw):
        self.calls.append(question)
        result = Result(
            plan={
                "rewritten": question,
                "filters": None,
                "aggregate": False,
                "top_n": None,
                "reason": "t",
                "degraded": False,
            },
            retrieved=[
                {
                    "chunk_id": "d1:1",
                    "doc_id": "d1",
                    "title": "文档",
                    "text": "预算 42 万元",
                    "section_path": [],
                    "page": 1,
                    "block_type": "paragraph",
                }
            ],
            contexts=[
                {
                    "no": 1,
                    "text": "预算 42 万元",
                    "doc": "文档",
                    "page": 1,
                    "doc_id": "d1",
                }
            ],
            citations=[{"no": 1, "doc": "文档", "page": 1, "doc_id": "d1"}],
            answer="预算 42 万元 [1]",
            retrieval_empty=False,
        )
        result.latency_ms = {
            "rewrite": 1.0,
            "retrieve": 1.0,
            "rerank": 1.0,
            "retrieval_total": 3.0,
            "synthesize": None,
            "synth_cached": None,
            "total": 3.0,
        }
        result.synth_meta = None
        return result


def _gold(tmp_path, n=3):
    p = tmp_path / "gold.json"
    p.write_text(
        json.dumps(
            {
                "meta": {"gold_version": "t"},
                "items": [
                    {
                        "id": f"q{i}",
                        "type": "fact",
                        "question": f"问题{i}",
                        "expected_answer": "答案",
                        "must_contain": ["42"],
                        "source_doc_ids": ["d1"],
                        "refusable": False,
                    }
                    for i in range(1, n + 1)
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return p


_CFG = {"retrieval": {"max_contexts": 3}, "llm": {"model": "m"}}


def _first_run(tmp_path, monkeypatch, fail_on: set[str] | None = None):
    out = tmp_path / "run.json"
    orch = _FlakyOrch(fail_on=fail_on or set())
    monkeypatch.setattr(eval_runner, "_build_orchestrator", lambda *a, **kw: orch)
    eval_runner.evaluate(_gold(tmp_path), cfg=_CFG, progress_file=out)
    return out


class _FlakyOrch(_FakeOrch):
    def __init__(self, fail_on: set[str]):
        super().__init__()
        self._fail_on = fail_on

    def answer(self, question, **kw):
        if question in self._fail_on:
            raise RuntimeError("embed endpoint timeout (simulated)")
        return super().answer(question, **kw)


def test_resume_refuses_provenance_mismatch(tmp_path, monkeypatch):
    """synth_fp 不等 → 拒绝：不许把状态 A 的条目拼进自称状态 B 的 meta。"""
    out = _first_run(tmp_path, monkeypatch)
    data = json.loads(out.read_text(encoding="utf-8"))
    data["meta"]["synth_fp"] = "deadbeefdeadbeef"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    orch = _FakeOrch()
    monkeypatch.setattr(eval_runner, "_build_orchestrator", lambda *a, **kw: orch)
    with pytest.raises(ValueError, match="synth_fp"):
        eval_runner.evaluate(_gold(tmp_path), cfg=_CFG, resume_from=out)
    assert orch.calls == []  # 一条都没跑：拒绝发生在任何付费/检索之前


def test_resume_marks_resumed_items(tmp_path, monkeypatch):
    """被复用条目带 resumed_from 标记；本轮新跑的条目不带。"""
    out = _first_run(tmp_path, monkeypatch, fail_on={"问题2"})
    orch = _FakeOrch()
    monkeypatch.setattr(eval_runner, "_build_orchestrator", lambda *a, **kw: orch)
    results = eval_runner.evaluate(_gold(tmp_path), cfg=_CFG, resume_from=out)
    by_id = {it["id"]: it for it in results["items"]}
    assert by_id["q1"]["resumed_from"] == out.name
    assert by_id["q3"]["resumed_from"] == out.name
    assert "resumed_from" not in by_id["q2"]  # 本轮补跑的条目不是复用
    assert len(orch.calls) == 1 and orch.calls[0] == "问题2"


def test_resume_provenance_equal_passes(tmp_path, monkeypatch):
    """同状态（真 meta 未动）必须照常续跑——护栏不许误伤 B2。"""
    out = _first_run(tmp_path, monkeypatch, fail_on={"问题2"})
    orch = _FakeOrch()
    monkeypatch.setattr(eval_runner, "_build_orchestrator", lambda *a, **kw: orch)
    results = eval_runner.evaluate(_gold(tmp_path), cfg=_CFG, resume_from=out)
    assert results["summary"]["n_resumed"] == 2
    assert results["summary"]["n_errors"] == 0

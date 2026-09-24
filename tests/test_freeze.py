"""B10：freeze 双指纹——幂等、eval meta 自证、只对相关的那一个报警。"""

from __future__ import annotations

import json
import sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_rag import freeze as freeze_mod
from doc_rag import index_identity


def _cfg(tmp_path: Path) -> dict:
    return {
        "llm": {
            "model": "deepseek-flash",
            "base_url": "https://api.deepseek.com/v1",
            "reasoning_effort": "",
            "reasoning_effort_by_type": {"cross_doc": "none"},
        },
        "retrieval": {"max_contexts": 10},
        "rerank": {"enabled": True, "top_n": 6},
        "rewrite": {"base_url": "", "model": ""},
        "eval": {"gold_file": str(tmp_path / "gold.json"), "judge": {}},
        "qdrant": {"collection": "c"},
    }


def _gold(tmp_path: Path) -> Path:
    p = tmp_path / "gold.json"
    p.write_text('{"items": []}', encoding="utf-8")
    return p


def test_freeze_idempotent(tmp_path, monkeypatch):
    freeze_dir = tmp_path / "cache"
    monkeypatch.setattr(freeze_mod, "_FREEZE_DIR", freeze_dir)
    monkeypatch.setattr(index_identity, "load_fingerprints", lambda: {"c": "aaaa" * 4})
    cfg = _cfg(tmp_path)
    r1 = freeze_mod.freeze(cfg, "c", _gold(tmp_path))
    r2 = freeze_mod.freeze(cfg, "c", _gold(tmp_path))
    assert r1["freeze_id"] == r2["freeze_id"]
    assert r1["file"] == r2["file"]
    assert (
        json.loads(Path(r1["file"]).read_text("utf-8"))["index_fp"]
        == "aaaaaaaaaaaaaaaa"
    )
    assert len(freeze_mod.scan_freezes()) == 1


def test_warnings_are_relevant_only(tmp_path, monkeypatch):
    freeze_dir = tmp_path / "cache"
    monkeypatch.setattr(freeze_mod, "_FREEZE_DIR", freeze_dir)
    monkeypatch.setattr(index_identity, "load_fingerprints", lambda: {"c": "ind-1"})
    cfg = _cfg(tmp_path)
    rec = freeze_mod.freeze(cfg, "c", _gold(tmp_path))  # 冻结 ind-1 / synth-1
    idx_fp, synth_fp = rec["index_fp"], rec["synth_fp"]

    # 只改 prompt（synth 变了）：只报 synth，检索侧不报
    w = freeze_mod.freeze_warnings(idx_fp, "other-synth", with_answers=True)
    assert any("synth_fp" in x for x in w)
    assert not any("index_fp" in x for x in w)

    # 双改（L2：分块/嵌入变更后两边都失配）：检索与答案都报
    w2 = freeze_mod.freeze_warnings("ind-2", "other-synth", with_answers=True)
    assert any("index_fp" in x for x in w2)
    assert any("synth_fp" in x for x in w2)

    # 只改索引、答案侧仍在冻结态：只报检索侧
    w2b = freeze_mod.freeze_warnings("ind-2", synth_fp, with_answers=True)
    assert any("index_fp" in x for x in w2b)
    assert not any("synth_fp 与" in x for x in w2b)

    # 检索侧专用评估（无答案）：synth 不匹配不报
    w3 = freeze_mod.freeze_warnings("ind-2", "whatever", with_answers=False)
    assert any("index_fp" in x for x in w3)
    assert not any("synth_fp 与" in x for x in w3)

    # 裸检出（无 freeze 记录）：静默
    monkeypatch.setattr(freeze_mod, "scan_freezes", list)
    assert freeze_mod.freeze_warnings("x", "y", with_answers=True) == []


def test_eval_meta_carries_fingerprints(tmp_path, monkeypatch):
    """evaluate() 的 meta 必须带 index_fp / synth_fp / freeze_warnings。"""
    from doc_rag.eval import runner as eval_runner

    gold = tmp_path / "g.json"
    gold.write_text(
        json.dumps(
            {
                "meta": {"gold_version": "t"},
                "items": [
                    {
                        "id": "q1",
                        "type": "fact",
                        "question": "问题1",
                        "expected_answer": "答案",
                        "must_contain": ["42"],
                        "source_doc_ids": ["d1"],
                        "refusable": False,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(index_identity, "load_fingerprints", lambda: {"t": "ind-9"})
    monkeypatch.setattr(freeze_mod, "scan_freezes", list)  # 无冻结记录 → 警告为空
    monkeypatch.setattr(freeze_mod, "_FREEZE_DIR", tmp_path / "cache")

    class _Retriever:
        collection = "t"
        cfg: typing.ClassVar = {"mode": "hybrid"}

    class _Orch:
        def __init__(self, *a, **kw):
            self.retriever = _Retriever()

        def answer(self, question, **kw):
            kw["with_answer"] = False
            from doc_rag.orchestrator import Result

            r = Result(
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
                contexts=[],
                citations=[],
            )
            r.latency_ms = {
                "rewrite": 0,
                "retrieve": 0,
                "rerank": 0,
                "retrieval_total": 0,
                "synthesize": None,
                "synth_cached": None,
                "total": 0,
            }
            return r

    monkeypatch.setattr(eval_runner, "_build_orchestrator", lambda *a, **kw: _Orch())
    results = eval_runner.evaluate(gold, with_answers=False)
    assert results["meta"]["index_fp"] == "ind-9"
    assert results["meta"]["synth_fp"]
    assert results["meta"]["freeze_warnings"] == []

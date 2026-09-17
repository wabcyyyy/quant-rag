import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from doc_rag import cli
from doc_rag.eval import runner
from doc_rag.generate import llm


@pytest.fixture
def offline_eval(tmp_path, monkeypatch):
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"items": [{
        "id": "q001", "type": "fact", "question": "费用？",
        "expected_answer": "67元", "source_doc_ids": ["d1"],
        "must_contain": ["67元"],
    }]}), encoding="utf-8")
    cfg = {"retrieval": {}, "eval": {"gold_file": str(gold)},
           "paths": {"eval": str(tmp_path)}}
    retriever = Mock(collection="test", cfg={})
    retriever.retrieve.return_value = [
        {"doc_id": "d1", "title": "报价", "page": 1, "text": "费用67元"}
    ]
    synthesizer = Mock()
    synthesizer.answer.return_value = "费用67元 [1]"
    monkeypatch.setattr(runner, "_build_retriever", Mock(return_value=(retriever, synthesizer)))
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(llm, "cache_stats", lambda: {
        "hit": 0, "miss": 0, "hit_rate": None, "cached_total": 0,
    })
    judge = Mock(return_value={"faithfulness": 1.0})
    monkeypatch.setattr(runner, "_run_ragas", judge)
    return gold, cfg, judge


@pytest.mark.parametrize("fresh", [False, True])
def test_cli_passes_judge_cache_policy_to_runner(offline_eval, fresh):
    _, _, judge = offline_eval
    args = ["eval", "--ragas"] + (["--fresh-judge"] if fresh else [])
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    judge.assert_called_once()
    assert judge.call_args.kwargs.get("use_cache", True) is (not fresh)


def test_direct_eval_keeps_judge_cache_by_default(offline_eval):
    gold, cfg, judge = offline_eval
    runner.evaluate(gold, cfg=cfg, with_ragas=True)
    assert judge.call_args.kwargs.get("use_cache", True) is True


def test_cli_rejects_fresh_judge_without_ragas_before_work(offline_eval, monkeypatch):
    build = Mock(side_effect=AssertionError("retrieval must not start"))
    monkeypatch.setattr(runner, "_build_retriever", build)
    result = CliRunner().invoke(cli.app, ["eval", "--fresh-judge"])
    assert result.exit_code != 0
    assert "--ragas" in result.output
    build.assert_not_called()


def test_ragas_from_keeps_forwarding_fresh_judge(offline_eval, monkeypatch):
    gold, _, _ = offline_eval
    replay = Mock(return_value={"faithfulness": 1.0})
    monkeypatch.setattr(runner, "ragas_from_results", replay)
    result = CliRunner().invoke(cli.app, [
        "eval", "--ragas-from", str(gold), "--fresh-judge",
    ])
    assert result.exit_code == 0, result.output
    assert replay.call_args.kwargs["use_cache"] is False

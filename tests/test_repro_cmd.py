"""C7：`doc-rag repro` 一键复现入口的离线验收。

守护三件事（都是「复现会不会复现错东西」的护栏，不测数字本身）：
1. **默认值全部指向公开语料**——`data/sample_parsed/s3`（公开 320 篇）与
   `gold_core.json`；绝不能默认吃 `data/parsed`（公司语料），那是串库事故的入口。
2. **collection 默认隔离**到 `doc_rag_repro`，不写生产库（`doc_rag_sample`）。
3. 检索侧默认不调 LLM（`with_answers=False`），`--answers` 才走合成。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from doc_rag import cli


@pytest.fixture()
def _stub(monkeypatch, tmp_path):
    """把入库与两次 evaluate 都换成替身，记录调用参数。"""
    from doc_rag.eval import runner as eval_runner
    from doc_rag.ingest import indexer

    seen: dict = {"eval": []}

    def fake_index(parsed_dir, cfg, **kw):
        seen["index"] = {"parsed_dir": Path(parsed_dir), **kw}
        return {"docs": 320, "chunks": 348, "parsed": 320, "empty": 0, "failed": []}

    def fake_evaluate(gold_file, cfg=None, **kw):
        seen["eval"].append({"gold": Path(gold_file), **kw})
        return {
            "meta": {"budget": "fixed:8", "retrieval": "dense+bm25+rrf[hybrid]+rerank"},
            "summary": {
                "n_items": 74,
                "hit_at_5": 0.9394,
                "hit_at_8": 0.9848,
                "mrr": 0.7884,
                "ndcg_at_8": 0.8394,
                "recall_at_list_macro": 0.9779,
                "recall_ceiling_macro": 0.9942,
                "strict_keyword_accuracy": 0.9394,
                "keypoint_recall_macro": 0.6694,
                "keypoint_n_items": 31,
                "refusal_acc": 1.0,
                "citation_valid_rate": 1.0,
            },
            "items": [],
        }

    monkeypatch.setattr(indexer, "index_parsed", fake_index)
    monkeypatch.setattr(eval_runner, "evaluate", fake_evaluate)
    return seen


def test_repro_defaults_point_at_public_corpus(_stub, tmp_path):
    res = CliRunner().invoke(cli.app, ["repro"])
    assert res.exit_code == 0, res.output
    # 路径按仓库根解析成绝对路径；判据是「指向公开示例语料」而不是 data/parsed
    assert _stub["index"]["parsed_dir"].as_posix().endswith("data/sample_parsed/s3")
    assert _stub["index"]["collection"] == "doc_rag_repro"
    assert _stub["index"]["use_llm_meta"] is None  # 读配置，不顺手开付费抽取
    assert _stub["eval"][0]["gold"].as_posix().endswith("data/eval/gold_core.json")


def test_repro_retrieval_only_by_default(_stub):
    CliRunner().invoke(cli.app, ["repro"])
    assert len(_stub["eval"]) == 1
    call = _stub["eval"][0]
    assert call["with_answers"] is False
    assert call["use_rerank"] is True
    assert call["use_rewrite"] is False  # 检索侧零 LLM（与阶段读数同口径）


def test_repro_answers_opt_in_runs_second_pass(_stub):
    res = CliRunner().invoke(cli.app, ["repro", "--answers"])
    assert res.exit_code == 0, res.output
    assert len(_stub["eval"]) == 2
    assert _stub["eval"][1]["with_answers"] is True
    assert _stub["eval"][1]["use_rewrite"] is True


def test_repro_parses_automatically_on_clean_clone(_stub, monkeypatch, tmp_path):
    """干净 clone 里没有解析产物（gitignored 构建件）→ 必须自动补一次解析。

    「两条命令跑通」这句话只对「解析产物已在盘上」的工作区成立；干净 clone 要成立，
    就必须让 repro 自己解析（纯本地、零 API 成本）。这是新目录实测发现的真缺口。
    """
    from doc_rag.ingest import pipeline as ingest_pipeline

    calls: list = []
    monkeypatch.setattr(
        ingest_pipeline,
        "run",
        lambda raw, parsed, **kw: (
            calls.append((Path(raw), Path(parsed))),
            {"total": 320, "parsed": 320, "skipped_duplicate": 0, "failed": []},
        )[1],
    )
    missing = tmp_path / "no-such-parsed"
    res = CliRunner().invoke(cli.app, ["repro", "--parsed-dir", str(missing)])
    assert res.exit_code == 0, res.output
    assert len(calls) == 1
    assert calls[0][0].as_posix().endswith("data/sample_raw")  # 从入库语料解析
    assert calls[0][1] == missing


def test_repro_skips_parse_when_parsed_dir_populated(_stub, monkeypatch, tmp_path):
    """解析产物已在盘上时必须跳过解析——否则每次重跑都白解析一遍 320 篇。"""
    from doc_rag.ingest import pipeline as ingest_pipeline

    seen: list = []
    monkeypatch.setattr(ingest_pipeline, "run", lambda *a, **kw: seen.append(a) or {})
    ready = tmp_path / "sample_parsed"
    ready.mkdir()
    (ready / "doc.json").write_text("{}", encoding="utf-8")
    res = CliRunner().invoke(cli.app, ["repro", "--parsed-dir", str(ready)])
    assert res.exit_code == 0, res.output
    assert seen == []


def test_repro_refuses_when_qdrant_unreachable(monkeypatch):
    """前置不满足要给可执行的下一步，不许半路崩在入库里。"""
    import qdrant_client

    class _Down:
        def __init__(self, *a, **kw):
            pass

        def get_collections(self):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(qdrant_client, "QdrantClient", _Down)
    res = CliRunner().invoke(cli.app, ["repro"])
    assert res.exit_code == 1
    assert "docker compose up -d" in res.output


def test_repro_gold_missing_fails_early(monkeypatch, tmp_path):
    res = CliRunner().invoke(cli.app, ["repro", "--gold", str(tmp_path / "nope.json")])
    assert res.exit_code == 1
    assert "黄金集不存在" in res.output

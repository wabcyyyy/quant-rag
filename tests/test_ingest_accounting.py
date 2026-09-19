"""入库对账：解析产物与入库篇数的差额必须逐项有归属。

动机（实测）：真实语料 1127 篇解析产物只入库 1121 篇，那 6 篇空文档被
`if doc.blocks` 静默丢弃——既不进 failed 也不进任何计数，差额无人知晓。
另一半是**反向**对账：源文件删了或改过内容（doc_id = sha256[:16]）时旧点永久留在
库里，还会被检索命中，所以 `--prune` 的判定与三道闸都在这里锁死。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from conftest import FakeQdrant

from doc_rag.ingest import indexer
from doc_rag.ingest.indexer import collection_doc_ids, delete_docs, reconcile


def _write(parsed_dir: Path, doc_id: str, blocks: list[dict]) -> None:
    (parsed_dir / f"{doc_id}.json").write_text(
        json.dumps(
            {
                "meta": {
                    "source_type": "pdf",
                    "doc_id": doc_id,
                    "title": f"档案_{doc_id}",
                },
                "blocks": blocks,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


@pytest.fixture
def wired(monkeypatch, request):
    """离线跑 index_parsed：不连 Qdrant、不打嵌入 API。"""
    calls: dict = {}
    fake = FakeQdrant(getattr(request, "param", None) or [])
    calls["client"] = fake

    monkeypatch.setattr(indexer, "QdrantClient", lambda **kw: fake)
    embedder = Mock()
    embedder.embed.return_value = [[0.0, 0.0]]
    monkeypatch.setattr(indexer, "Embedder", lambda cfg: embedder)
    monkeypatch.setattr(
        indexer, "ensure_collection", lambda *a, **k: calls.setdefault("ensure", k)
    )
    return calls


CFG = {
    "qdrant": {"url": "http://x", "collection": "kb"},
    "embedding": {"dense_dim": 2, "base_url": "http://e", "api_key": "k", "model": "m"},
    "llm": {"model": "m", "base_url": "http://l", "api_key": "k"},
}


def test_empty_docs_are_counted_not_dropped(tmp_path, wired):
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    _write(parsed, "d1", [{"type": "paragraph", "text": "正文一段", "page": 1}])
    _write(parsed, "d2", [])  # 解析成功但没有块
    _write(parsed, "d3", [])

    stats = indexer.index_parsed(parsed, CFG)

    assert stats["parsed"] == 3, "解析产物总数必须落地，否则差额无法对账"
    assert stats["empty"] == 2
    assert stats["docs"] == 1
    assert stats["chunks"] == 1
    assert stats["failed"] == []
    assert stats["parsed"] - stats["docs"] == stats["empty"] + len(stats["failed"])


def test_malformed_json_lands_in_failed(tmp_path, wired):
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    _write(parsed, "d1", [{"type": "paragraph", "text": "正文", "page": 1}])
    (parsed / "bad.json").write_text("{not json", encoding="utf-8")

    stats = indexer.index_parsed(parsed, CFG)

    assert stats["parsed"] == 1  # 解析失败的不进 parsed，进 failed
    assert [f["file"] for f in stats["failed"]] == ["bad.json"]


def test_empty_fields_are_not_indexed_by_default(tmp_path, wired):
    """默认入库不给 topics/attendees 建索引：它们在库里永久为空。"""
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    _write(parsed, "d1", [{"type": "paragraph", "text": "正文", "page": 1}])

    indexer.index_parsed(parsed, CFG)

    assert wired["ensure"]["llm_meta_fields"] is False


def test_llm_meta_run_indexes_those_fields(tmp_path, wired):
    """走 --llm-meta 时 topics/attendees 才会有值，索引也随之建立。

    这里把 llm api_key 置空：extract_metadata 会退回文件名信号，不产生 API 调用。
    """
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    _write(parsed, "d1", [{"type": "paragraph", "text": "正文", "page": 1}])
    cfg = {**CFG, "llm": {**CFG["llm"], "api_key": "", "model": ""}}

    stats = indexer.index_parsed(parsed, cfg, use_llm_meta=True)

    assert wired["ensure"]["llm_meta_fields"] is True
    assert stats["llm_meta"] is True


# --------------------------------------------------------------------- 反向对账


def test_collection_doc_ids_pages_until_the_offset_runs_out(monkeypatch):
    monkeypatch.setattr(indexer, "_SCROLL_PAGE", 2)
    fake = FakeQdrant(["a", "a", "a", "b", "b"])

    counts = collection_doc_ids(fake, "kb")

    assert counts == {"a": 3, "b": 2}
    assert fake.scroll_pages == 3  # 2 + 2 + 1


def test_reconcile_reports_both_directions():
    rec = reconcile({"d1", "d2"}, {"d1": 3, "d2": 1, "ghost": 4})
    assert rec["orphan_doc_ids"] == ["ghost"] and rec["orphan_chunks"] == 4
    assert rec["missing_docs"] == []
    assert (rec["keep_docs"], rec["collection_docs"], rec["collection_chunks"]) == (
        2,
        3,
        8,
    )
    # 语料里有、库里没有 → 漏灌（失败/新文件/上次截断）
    assert reconcile({"d1", "new"}, {"d1": 1})["missing_docs"] == ["new"]


def test_delete_docs_batches_the_matched_ids(monkeypatch):
    monkeypatch.setattr(indexer, "_DELETE_BATCH", 2)
    fake = FakeQdrant([])

    assert delete_docs(fake, "kb", ["a", "b", "c"]) == 3

    assert [len(b) for b in fake.deleted] == [2, 1]
    flat = [d for batch in fake.deleted for d in batch]
    assert sorted(flat) == ["a", "b", "c"]


def _decision(orphans: list[str], keep: int = 10, failed: list | None = None, yes=True):
    rec = {"keep_docs": keep, "orphan_doc_ids": orphans, "orphan_chunks": len(orphans)}
    return indexer._prune_decision(rec, {"failed": failed or []}, yes)


def test_prune_gates():
    assert _decision([]) == {"deleted": 0}
    assert "只报告" in _decision(["g"], yes=False)["refused"]
    # 有解析产物读不通时，它对应的旧点无法判定归属——宁可不删
    assert "拒绝清理" in _decision(["g"], failed=[{"file": "x.json"}])["refused"]
    # 要删的比留下的多 → 几乎一定是 --kb / --parsed-dir 串了库
    assert "拒绝清理" in _decision(["g1", "g2", "g3"], keep=2)["refused"]
    assert _decision(["g1", "g2"], keep=20) == {"deleted": 2}


@pytest.mark.parametrize("wired", [["d1", "ghost", "ghost"]], indirect=True)
def test_ingest_reports_ghost_points_and_only_deletes_with_yes(tmp_path, wired):
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    _write(parsed, "d1", [{"type": "paragraph", "text": "正文", "page": 1}])
    fake = wired["client"]

    report = indexer.index_parsed(parsed, CFG, prune=True)["reconcile"]
    assert report["orphan_doc_ids"] == ["ghost"] and report["orphan_chunks"] == 2
    assert report["collection_docs"] == 2 and report["collection_chunks"] == 3
    flat = [d for batch in fake.deleted for d in batch]
    assert "ghost" not in flat  # 没 --yes 不动手

    deleted = indexer.index_parsed(parsed, CFG, prune=True, assume_yes=True)["prune"]
    assert deleted == {"deleted": 1}
    assert ["ghost"] in fake.deleted


def test_keep_doc_ids_from_raw_marks_stale_parsed_products(tmp_path, wired):
    """parsed_dir 里那篇已经没有对应源文件了：它本次仍会被灌进去，但必须被点名。"""
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    _write(parsed, "d1", [{"type": "paragraph", "text": "正文", "page": 1}])
    _write(parsed, "d2", [{"type": "paragraph", "text": "正文", "page": 1}])

    stats = indexer.index_parsed(parsed, CFG, keep_doc_ids={"d1"})

    rec = stats["reconcile"]
    assert rec["source"] == "raw"
    assert rec["stale_parsed_json"] == ["d2"]
    assert rec["keep_docs"] == 1
    # 库里现存 d1、d2（刚灌进去），而语料只有 d1 → d2 是幽灵
    wired["client"].points = ["d1", "d2"]
    rec2 = indexer.index_parsed(parsed, {**CFG}, keep_doc_ids={"d1"}, prune=True)[
        "reconcile"
    ]
    assert rec2["orphan_doc_ids"] == ["d2"]

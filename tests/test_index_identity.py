"""B4：缓存键织入库指纹——同 prompt 不同指纹必 miss；裸检出不报错、键不变。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_rag import index_identity
from doc_rag.generate import llm


def _key() -> str:
    cfg = {"base_url": "http://x", "model": "m", "temperature": 0}
    messages = [{"role": "user", "content": "同一句话"}]
    return llm._cache_key(cfg, messages, temperature=0)


def _write_fp(tmp_path, monkeypatch, mapping):
    fp_file = tmp_path / "index_fingerprint.json"
    monkeypatch.setattr(index_identity, "_FP_FILE", fp_file)
    if mapping is not None:
        fp_file.write_text(json.dumps(mapping), encoding="utf-8")
    return fp_file


def test_missing_fingerprint_file_keeps_key_stable(tmp_path, monkeypatch):
    _write_fp(tmp_path, monkeypatch, None)
    index_identity.active_collection.set("doc_rag_sample")
    try:
        assert index_identity.active_identity() == ""  # 裸检出不报错
        k1 = _key()
        k2 = _key()
        assert k1 == k2
    finally:
        index_identity.active_collection.set("")


def test_same_prompt_different_fingerprint_gives_different_key(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": "同一句话"}]
    cfg = {"base_url": "http://x", "model": "m"}

    def key_for(fp: str | None) -> str:
        if fp is None:
            _write_fp(tmp_path, monkeypatch, None)
            index_identity.active_collection.set("")
        else:
            _write_fp(
                tmp_path,
                monkeypatch,
                {"doc_rag_sample": fp},
            )
            index_identity.active_collection.set("doc_rag_sample")
        try:
            return llm._cache_key(cfg, messages, temperature=0)
        finally:
            index_identity.active_collection.set("")

    k_none = key_for(None)  # 无指纹：历史键形状，零作废
    k_a = key_for("aaaaaaaaaaaaaaaa")
    k_b = key_for("bbbbbbbbbbbbbbbb")
    assert k_a != k_b, "索引变了必须 miss"
    assert k_a != k_none
    # 再读一遍无指纹形态：键回到历史值（指纹为空时完全不进键）
    assert key_for(None) == k_none


def test_compute_index_fp_is_stable_and_sensitive():
    base = {
        "collection": "c",
        "n_points": 348,
        "n_chunks": 320,
        "embed_model": "BAAI/bge-m3",
        "chunk_strategy": "structural",
        "bm25_strategy": "jieba+indexed_text+ctx_off",
    }
    a = index_identity.compute_index_fp(**base)
    assert a == index_identity.compute_index_fp(**base)
    assert a != index_identity.compute_index_fp(**{**base, "n_points": 349})
    assert a != index_identity.compute_index_fp(**{**base, "bm25_strategy": "x"})


def test_orchestrator_sets_active_collection(monkeypatch):
    """检索路径必须声明 collection：这是指纹进缓存键的挂载点。"""
    from doc_rag.orchestrator import Orchestrator

    captured = {}

    class FakeEmbedder:
        def __init__(self, cfg):
            pass

    class FakeRetriever:
        def __init__(self, **kw):
            captured["collection"] = kw.get("collection")

    monkeypatch.setattr("doc_rag.orchestrator.Embedder", FakeEmbedder)
    monkeypatch.setattr("doc_rag.orchestrator.HybridRetriever", FakeRetriever)
    cfg = {
        "qdrant": {"url": "http://fake", "collection": "default_kb"},
        "embedding": {},
        "retrieval": {},
        "llm": {},
    }
    index_identity.active_collection.set("")
    try:
        _ = Orchestrator(cfg, collection="my_kb").retriever
        assert captured["collection"] == "my_kb"
        assert index_identity.active_collection.get() == "my_kb"
    finally:
        index_identity.active_collection.set("")

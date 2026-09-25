"""A3.4 contextual retrieval 机制的离线护栏（真 LLM 调用由试算与 S 阶段读数覆盖）。

- 生成失败必须降级「无前缀」并计数（metadata.extract_metadata 先例）；
- 三臂的被索引文本组合：off=无前缀；both=dense+BM25 都带；bm25=只 BM25 带
  （dense 与无前缀臂同输入）；
- `contextual.enabled` 默认 true（2026-09-25 U3① 转正拍板；钉住防悄悄改动 = L2 作废）；
- model/base_url/api_key 留空 = 继承合成侧 llm。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_rag.ingest import indexer
from doc_rag.ingest.chunker import chunk_by
from doc_rag.ingest.contextual import contextual_cfg, generate_prefixes
from doc_rag.ingest.schema import Block, IntermediateDoc, SourceMeta


def _doc() -> IntermediateDoc:
    return IntermediateDoc(
        meta=SourceMeta(
            source_type="pdf",
            doc_id="d1",
            title="云帆科技 2026 年第 23 周办公会会议纪要",
        ),
        blocks=[
            Block(type="heading", heading_level=2, text="一、新仓库建设"),
            Block(
                type="paragraph",
                text="供应链部孙晓芸汇报，土建已完成 80%，预算 380 万元。",
            ),
        ],
    )


def _cfg() -> dict:
    return {
        "contextual": {
            "enabled": True,
            "model": "fake-model",
            "base_url": "http://fake",
            "api_key": "k",
        },
        "llm": {"model": "parent-model", "base_url": "http://parent", "api_key": "pk"},
    }


def test_cfg_inherits_llm_endpoint_when_blank():
    cfg = _cfg()
    cfg["contextual"].update(model="", base_url="", api_key="")
    ctx = contextual_cfg(cfg)
    assert ctx["model"] == "parent-model"
    assert ctx["base_url"] == "http://parent"
    assert ctx["api_key"] == "pk"


def test_generation_failure_degrades_to_no_prefix(monkeypatch):
    """真实失败模式：llm.chat 抛错 → generate_prefix 捕获 → 降级「无前缀」+ 计数。"""
    import doc_rag.ingest.contextual as ctx_mod

    def boom(llm_cfg, prompt, system_prompt=None, temperature=None):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(ctx_mod.llm, "chat", boom)
    doc = _doc()
    chunks = chunk_by("structural", doc)
    mapping, failed = generate_prefixes(contextual_cfg(_cfg()), [doc], [chunks])
    assert failed == len(chunks)
    assert all(v is None for v in mapping.values())


def test_disabled_ctx_returns_empty():
    cfg = _cfg()
    cfg["contextual"]["enabled"] = False
    mapping, failed = generate_prefixes(
        contextual_cfg(cfg), [_doc()], [chunk_by("structural", _doc())]
    )
    assert mapping == {} and failed == 0


def test_three_arms_compose_index_text(monkeypatch, tmp_path):
    """臂组合的构造断言：off/both/bm25 的 dense 与 BM25 输入符合 ADR-0004 定义。"""

    from doc_rag.ingest import contextual as ctx_mod

    parsed = tmp_path / "parsed"
    parsed.mkdir()
    doc = _doc()
    (parsed / "d1.json").write_text(doc.model_dump_json(), encoding="utf-8")

    def fake_llm_chat(llm_cfg, prompt, system_prompt=None, temperature=None):
        assert llm_cfg["reasoning_effort"] == "none"  # 前缀不思考（成本先例）
        return "本片段汇报新仓库建设进度与预算，位于议题一。"

    monkeypatch.setattr(ctx_mod.llm, "chat", fake_llm_chat)
    monkeypatch.setattr(indexer, "ensure_collection", lambda *a, **kw: None)

    embed_inputs: list[list[str]] = []

    class FakeEmbedder:
        def __init__(self, cfg):
            pass

        def embed(self, texts):
            embed_inputs.append(list(texts))  # 记录 dense 的真实输入文本
            return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(indexer, "Embedder", FakeEmbedder)

    captured: dict[str, list[tuple[str, str]]] = {}

    class FakeQdrant:
        def delete(self, name, points_selector=None):
            pass

        def upsert(self, name, points):
            rows = captured.setdefault(name, [])
            for p in points:
                rows.append(
                    (p.vector["bm25"].text, (p.payload or {}).get("ctx_prefix"))
                )

        def scroll(self, name, **kw):
            return [], None

    monkeypatch.setattr(indexer, "QdrantClient", lambda **kw: FakeQdrant())
    cfg = _cfg()
    cfg["embedding"] = {"dense_dim": 4}
    cfg["qdrant"] = {"url": "http://fake", "collection": "t"}

    # 臂 ①：off —— dense/BM25 都无前缀，payload 无 ctx_prefix
    indexer.index_parsed(
        parsed, cfg, collection="arm_off", recreate=True, ctx_mode="off"
    )
    off_dense = embed_inputs[-1]
    off_bm25, off_flag = captured["arm_off"][0]
    assert off_flag is None
    captured.clear()

    # 臂 ②：both —— dense 与 BM25 都以前缀开头
    indexer.index_parsed(
        parsed, cfg, collection="arm_both", recreate=True, ctx_mode="both"
    )
    both_dense = embed_inputs[-1]
    both_bm25, both_flag = captured["arm_both"][0]
    assert both_flag is not None
    assert both_dense[0].startswith("本片段汇报新仓库建设")
    # BM25 文本过 jieba 分词（字间插空格），前缀字符仍在开头
    assert "".join(both_bm25.split()).startswith("本片段汇报新仓库建设")
    captured.clear()

    # 臂 ③：bm25 —— 只有 BM25 带；dense 输入与 off 臂逐字相同
    indexer.index_parsed(
        parsed, cfg, collection="arm_bm25", recreate=True, ctx_mode="bm25"
    )
    bm25_dense = embed_inputs[-1]
    bm25_text, bm25_flag = captured["arm_bm25"][0]
    assert bm25_flag is not None
    assert bm25_dense == off_dense, "bm25 臂的 dense 输入必须与 off 臂逐字相同"
    assert "".join(bm25_text.split()).startswith("本片段汇报新仓库建设")
    assert bm25_text != off_bm25, "bm25 臂的 BM25 文本必须带上前缀"


def test_contextual_enabled_by_default():
    """U3① 已拍板转正（2026-09-25）：默认必须开（配置写谎 = 悄悄作废全部检索基线）。"""
    import yaml

    raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text("utf-8"))
    assert raw["contextual"]["enabled"] is True

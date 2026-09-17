"""Gradio 演示页（T8）冒烟测试——只构造 UI，不 launch、不打 API。

gradio 属 demo extra，未安装时整组跳过（核心依赖不含它，这是设计约束）。
"""

from __future__ import annotations

import pytest

gradio = pytest.importorskip("gradio")

import doc_rag.api.demo as demo_mod


def test_build_ui_returns_blocks_with_expected_components(monkeypatch):
    """构造 UI 必须全离线：pipeline 用 mock（不建 Qdrant 连接/嵌入器）。"""
    from types import SimpleNamespace

    cfg = {"llm": {}, "rerank": {"enabled": False}, "retrieval": {"max_contexts": 10}}
    retriever = SimpleNamespace(collection="c")
    rewriter = SimpleNamespace(rewrite=lambda q: {"rewritten": q, "filters": None, "aggregate": False, "top_n": 8})
    monkeypatch.setattr(demo_mod, "_pipeline", lambda: (cfg, retriever, rewriter))
    demo = demo_mod.build_ui(gradio)
    assert isinstance(demo, gradio.Blocks)


def test_example_questions_are_safe_placeholders():
    """示例问题必须是示意数据：不含真实文档标题模式（语料标题以档案类别前缀开头）。"""
    for q in demo_mod.EXAMPLE_QUESTIONS:
        assert "档案" not in q
        assert "@" not in q  # 会议纪要发言人格式是公司内容特征

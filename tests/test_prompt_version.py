"""prompt 版本开关（T1）的回归测试——全离线，不打真实 API。

守护对象是「三组对照的可复现性」：基线/收紧两组答案必须能靠结果文件 meta
自证出处（指纹不同、版本名落盘），且默认行为逐字不变（缓存键不漂移）。
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from doc_rag.eval import runner
from doc_rag.generate import llm as llm_mod
from doc_rag.generate import prompts
from doc_rag.generate.synthesizer import Synthesizer
from doc_rag.retrieve.hybrid import RetrievalOutcome


def test_default_version_is_tightened_and_unchanged():
    """SYSTEM_ANSWER 必须继续指向收紧版：所有既有调用点行为逐字不变。"""
    assert prompts.DEFAULT_PROMPT_VERSION == "tightened"
    assert prompts.ANSWER_PROMPTS["tightened"] is prompts.SYSTEM_ANSWER


def test_baseline_text_differs_from_tightened():
    """baseline 是收紧前的原文（取自 git 历史），与收紧版有可区分的措辞。"""
    base = prompts.ANSWER_PROMPTS["baseline"]
    tight = prompts.ANSWER_PROMPTS["tightened"]
    assert base != tight
    # 收紧版新增的关键约束（存在性自证两版确实不同）
    assert "只要上下文包含回答问题所需的信息" in tight
    assert "只要上下文包含回答问题所需的信息" not in base
    # baseline 的原文特征
    assert "不得使用上下文之外的知识" in base


def test_fingerprint_differs_between_versions():
    """两个版本的指纹必须不同——否则 meta 区分不了两组答案。"""
    fp_tight = prompts.fingerprint("tightened")
    fp_base = prompts.fingerprint("baseline")
    assert fp_tight != fp_base
    # 缺省 = tightened（fingerprint() 与 fingerprint(None) 等价）
    assert prompts.fingerprint() == fp_tight
    assert prompts.fingerprint(None) == fp_tight


def test_resolve_system_answer_unknown_version_raises():
    with pytest.raises(KeyError):
        prompts.resolve_system_answer(True, "no_such_version")


def _capture_system_prompt(monkeypatch):
    seen: list[str] = []

    def _fake_chat_timed(cfg, user_prompt, system_prompt=None, temperature=None):
        seen.append(system_prompt or "")
        return "答", {"ms": 1.0, "cached": False, "model": cfg["model"]}

    monkeypatch.setattr(llm_mod, "chat_timed", _fake_chat_timed)
    return seen


def test_synthesizer_default_uses_tightened(monkeypatch):
    seen = _capture_system_prompt(monkeypatch)
    Synthesizer({"model": "m"}).answer(
        "q", [{"no": 1, "text": "t", "doc": "d", "page": 1}]
    )
    assert seen[0] == prompts.SYSTEM_ANSWER


def test_synthesizer_baseline_switches_system_prompt(monkeypatch):
    seen = _capture_system_prompt(monkeypatch)
    syn = Synthesizer({"model": "m", "prompt_version": "baseline"})
    syn.answer("q", [{"no": 1, "text": "t", "doc": "d", "page": 1}])
    assert seen[0] == prompts.ANSWER_PROMPTS["baseline"]


def test_prompt_version_changes_cache_key(monkeypatch):
    """system prompt 进 messages → 缓存键必须随版本变化，不能拿 A 版答案答 B 版。"""
    keys = []
    monkeypatch.setattr(llm_mod, "cache_enabled", lambda cfg=None: True)
    monkeypatch.setattr(llm_mod, "_cache_get", lambda key: None)  # 强制 miss

    def _fake_put(key, model, response):
        keys.append(key)

    monkeypatch.setattr(llm_mod, "_cache_put", _fake_put)

    class _FakeClient:
        def __init__(self, **kwargs):
            def _create(**kw):
                resp = Mock()
                resp.choices = [Mock(message=Mock(content="x"))]
                resp.usage = None
                return resp

            self.chat = Mock(completions=Mock(create=_create))

    monkeypatch.setattr(llm_mod, "OpenAI", _FakeClient)
    base_cfg = {"model": "m", "base_url": "https://api.x.com", "api_key": "k"}
    llm_mod.chat_timed(base_cfg, "问题", system_prompt=prompts.SYSTEM_ANSWER)
    llm_mod.chat_timed(
        base_cfg, "问题", system_prompt=prompts.ANSWER_PROMPTS["baseline"]
    )
    assert len(keys) == 2 and keys[0] != keys[1]


# ------------------------------------------------------- runner / CLI 接线


def _gold_one_item(tmp_path):
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "q001",
                        "type": "fact",
                        "question": "费用？",
                        "expected_answer": "67元",
                        "source_doc_ids": ["d1"],
                        "must_contain": ["67元"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return gold


def _mock_pipeline(monkeypatch):
    retriever = Mock(collection="c", cfg={})
    retriever.retrieve.return_value = RetrievalOutcome(
        chunks=[
            {
                "doc_id": "d1",
                "title": "报价",
                "page": 1,
                "text": "费用67元",
                "block_type": "p",
            }
        ]
    )
    synthesizer = Mock()
    synthesizer.answer.return_value = "费用67元 [1]"
    synthesizer.last_meta = {"ms": 1.0, "cached": False, "model": "m"}
    monkeypatch.setattr(
        runner, "_build_retriever", Mock(return_value=(retriever, synthesizer))
    )


def test_evaluate_meta_records_prompt_version(tmp_path, monkeypatch):
    _gold_one_item(tmp_path)
    _mock_pipeline(monkeypatch)
    cfg = {"retrieval": {}, "llm": {"model": "m", "prompt_version": "baseline"}}
    results = runner.evaluate(tmp_path / "gold.json", cfg=cfg)
    assert results["meta"]["prompt_version"] == "baseline"
    assert results["meta"]["prompt_fingerprint"] == prompts.fingerprint("baseline")


def test_evaluate_meta_defaults_to_tightened(tmp_path, monkeypatch):
    _gold_one_item(tmp_path)
    _mock_pipeline(monkeypatch)
    results = runner.evaluate(
        tmp_path / "gold.json", cfg={"retrieval": {}, "llm": {"model": "m"}}
    )
    assert results["meta"]["prompt_version"] == "tightened"
    assert results["meta"]["prompt_fingerprint"] == prompts.fingerprint()

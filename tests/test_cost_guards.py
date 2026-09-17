"""成本护栏回归测试——全部离线（mock），不发任何真实 LLM/embedding 调用。

守护对象是「钱」：重试层数、缓存故障可见性、抽样先于重检索、
上下文不一致拒判、入库默认不开 LLM 抽取。
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest
from openai import APIStatusError

from doc_rag.eval import runner
from doc_rag.eval.runner import (
    _make_token_counter,
    _sample_rows,
    ragas_from_results,
)
from doc_rag.generate import llm as llm_mod
from doc_rag.ingest import indexer

_STAT_KEYS = ("hit", "miss", "prompt_tokens", "completion_tokens",
              "reasoning_tokens", "cache_read_errors", "cache_write_errors")


def _fresh_stats(monkeypatch):
    stats = dict.fromkeys(_STAT_KEYS, 0)
    monkeypatch.setattr(llm_mod, "_STATS", stats)
    return stats


class _FakeStatusError(APIStatusError):
    def __init__(self, status_code: int):
        super().__init__("boom", response=Mock(status_code=status_code), body=None)


class _Fake429(_FakeStatusError):
    def __init__(self):
        super().__init__(429)


class _Fake401(_FakeStatusError):
    def __init__(self):
        super().__init__(401)


def _ok_response(text="答案"):
    usage = Mock(prompt_tokens=10, completion_tokens=5)
    usage.completion_tokens_details = Mock(reasoning_tokens=3)
    resp = Mock()
    resp.choices = [Mock(message=Mock(content=text))]
    resp.usage = usage
    return resp


CFG = {"model": "m", "base_url": "https://api.x.com", "api_key": "k", "cache": False}


@pytest.fixture
def chat_env(monkeypatch):
    _fresh_stats(monkeypatch)
    monkeypatch.setattr(llm_mod, "cache_enabled", lambda cfg=None: False)
    calls: dict = {"n": 0, "sleeps": [], "client_kwargs": {}}

    class _Completions:
        def create(self, **kwargs):
            calls["n"] += 1
            out = calls["respond"]()
            if isinstance(out, Exception):
                raise out  # 假客户端用「返回异常实例」表达「这次调用抛异常」
            return out

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        def __init__(self, **kwargs):
            calls["client_kwargs"] = kwargs
            self.chat = _Chat()

    monkeypatch.setattr(llm_mod, "OpenAI", _FakeClient)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: calls["sleeps"].append(s))
    return calls


# ---------- 重试：只重试瞬时错误，且 SDK 层不再叠加 ----------


def test_retries_only_transient_errors(chat_env):
    chat_env["respond"] = lambda: (_Fake429() if chat_env["n"] < 2 else _ok_response())
    assert llm_mod.chat(CFG, "问题") == "答案"
    assert chat_env["n"] == 2  # 429 重试 1 次后成功
    assert len(chat_env["sleeps"]) == 1


def test_permanent_error_fails_fast_without_retry(chat_env):
    chat_env["respond"] = _Fake401
    with pytest.raises(RuntimeError, match="attempt=1"):
        llm_mod.chat(CFG, "问题")
    assert chat_env["n"] == 1  # 401 重试没有意义
    assert chat_env["sleeps"] == []


def test_sdk_retries_disabled(chat_env):
    """SDK 默认还会再重试 2 次，与应用层相乘最多 12 次传输——必须归零。"""
    chat_env["respond"] = _Fake429
    with pytest.raises(RuntimeError):
        llm_mod.chat(CFG, "问题")
    assert chat_env["client_kwargs"]["max_retries"] == 0
    assert chat_env["n"] == llm_mod._RETRIES  # 恰好应用层次数，没有相乘


def test_usage_accounted_and_reasoning_separated(chat_env):
    chat_env["respond"] = _ok_response
    llm_mod.chat(CFG, "问题")
    st = llm_mod.cache_stats()
    assert (st["prompt_tokens"], st["completion_tokens"], st["reasoning_tokens"]) == (10, 5, 3)


# ---------- 缓存键：必须含 endpoint；temperature 用配置值 ----------


def test_cache_key_distinguishes_providers():
    def mk(url):
        return llm_mod._cache_key(
            {"model": "m", "base_url": url}, [{"role": "user", "content": "q"}]
        )

    assert mk("https://api.a.com/") != mk("https://api.b.com")


def test_temperature_defaults_to_config(chat_env, monkeypatch):
    chat_env["respond"] = _ok_response
    captured = {}
    orig = llm_mod._cache_key

    def spy(llm_cfg, messages, **params):
        captured.update(params)
        return orig(llm_cfg, messages, **params)

    monkeypatch.setattr(llm_mod, "_cache_key", spy)
    llm_mod.chat({**CFG, "temperature": 0.25}, "问题")
    assert captured["temperature"] == 0.25  # 此前合成走的是供应商默认


# ---------- 缓存故障必须可见，不能静默变成重复付费 ----------


def test_cache_read_error_counted(monkeypatch):
    _fresh_stats(monkeypatch)
    monkeypatch.setattr(llm_mod, "_conn", Mock(side_effect=RuntimeError("db locked")))
    assert llm_mod._cache_get("k") is None  # 不抛，但必须留痕
    assert llm_mod.cache_stats()["cache_read_errors"] == 1


def test_cache_write_error_counted(monkeypatch):
    _fresh_stats(monkeypatch)
    monkeypatch.setattr(llm_mod, "_conn", Mock(side_effect=RuntimeError("read-only fs")))
    llm_mod._cache_put("k", "m", "v")
    assert llm_mod.cache_stats()["cache_write_errors"] == 1


def test_judge_cache_init_failure_blocks_uncached_full_run(monkeypatch):
    """缓存初始化失败还继续跑，就是无缓存的全量 judge（约 10 倍成本）——必须失败。"""
    _fresh_stats(monkeypatch)
    cfg = {"llm": {"cache": True}, "eval": {"ragas_metrics": ["faithfulness"]}, "embedding": {}}
    monkeypatch.setattr(runner, "cache_enabled", lambda cfg=None: True)
    import langchain.globals as lg
    import langchain_community.cache as lc

    monkeypatch.setattr(lg, "set_llm_cache", Mock())
    monkeypatch.setattr(lc, "SQLiteCache", Mock(side_effect=RuntimeError("no .cache dir")))
    with pytest.raises(RuntimeError, match="judge 缓存初始化失败"):
        runner._run_ragas([{"id": "q1", "type": "fact", "user_input": "q",
                            "response": "a", "retrieved_contexts": ["c"]}], cfg)


# ---------- reasoning token 计量：原始与归一化两种用量结构 ----------


class _Resp:
    def __init__(self, usage=None, generations=None):
        self.llm_output = {"token_usage": usage} if usage else {}
        self.generations = generations


def test_counter_reads_raw_openai_reasoning_key():
    c = _make_token_counter()
    c.on_llm_end(_Resp(usage={"prompt_tokens": 100, "completion_tokens": 500,
                              "completion_tokens_details": {"reasoning_tokens": 470}}))
    d = c.as_dict()
    assert d["reasoning_tokens"] == 470  # 此前读成 details["reasoning"] → 恒为 0
    assert d["completion_tokens"] == 500


def test_counter_accumulates_normalized_generations():
    gen = Mock()
    gen.message.usage_metadata = {"input_tokens": 7, "output_tokens": 9,
                                  "output_token_details": {"reasoning": 4}}
    c = _make_token_counter()
    c.on_llm_end(_Resp(generations=[[gen, gen]]))
    d = c.as_dict()
    assert (d["prompt_tokens"], d["completion_tokens"], d["reasoning_tokens"]) == (7, 9, 8)


# ---------- 补跑 RAGAS：抽样先于重检索 / 模式还原 / 不一致拒判 ----------


def _write_results(tmp_path, n=20, contexts="legacy"):
    items = []
    for i in range(n):
        body = f"正文{i}"
        ctx = [f"[1] （t 第1页）\n{body}"] if contexts == "current" else [body]
        items.append({"id": f"q{i:02d}", "type": "fact", "question": f"问题{i}",
                      "answer": f"答案{i}", "contexts": ctx})
    path = tmp_path / "results.json"
    path.write_text(json.dumps(
        {"meta": {"collection": "c", "retrieval":
                  "dense+bm25+rrf[dense]+rewrite+rerank", "top_n": 8},
         "items": items}, ensure_ascii=False), encoding="utf-8")
    return path, items


def _patch_retrieval(monkeypatch, texts_by_question=None):
    """假检索/重排：返回的正文按问题尾号回放，便于构造「能对上/对不上」两种情形。

    built_modes 记录 retriever 构建时拿到的检索模式（dense/hybrid 在构建时就定死）。"""
    seen: list[str] = []
    built_modes: list[str | None] = []

    class _Retriever:
        def __init__(self, cfg):
            built_modes.append((cfg.get("retrieval") or {}).get("mode"))

        def retrieve(self, question, **kw):
            seen.append(question)
            text = (texts_by_question or {}).get(question, "正文")
            return [{"doc_id": "d", "title": "t", "page": 1, "text": text}]

    monkeypatch.setattr(runner, "_build_retriever", lambda cfg, col: (_Retriever(cfg), None))
    monkeypatch.setattr(runner, "_maybe_rerank", lambda cfg, q, r, use: r)
    return seen, built_modes


REPLAY_CFG = {"eval": {"ragas_metrics": ["faithfulness"]},
              "retrieval": {"mode": "hybrid", "max_contexts": 10},
              "llm": {"model": "m", "base_url": "http://l", "api_key": "k",
                      "cache": False},
              "embedding": {"model": "e", "base_url": "http://e", "api_key": "k"}}


def test_sampling_happens_before_retrieval(tmp_path, monkeypatch):
    """sample_n=2 只能为抽中的 2 条重检索（检索/重排是真实计费 API），不能全量 20 条。"""
    path, _ = _write_results(tmp_path)
    seen, _ = _patch_retrieval(monkeypatch, {f"问题{i}": f"正文{i}" for i in range(20)})
    judge = Mock(return_value={"faithfulness": 1.0})
    monkeypatch.setattr(runner, "_run_ragas", judge)
    ragas_from_results(path, REPLAY_CFG, sample_n=2)
    assert len(seen) == 2
    assert judge.call_args.kwargs["total"] == 20  # 报告口径保持抽样前总数
    assert judge.call_args.kwargs["sample_n"] == 2


def test_no_retriever_built_when_contexts_are_current(tmp_path, monkeypatch):
    path, _ = _write_results(tmp_path, n=3, contexts="current")
    monkeypatch.setattr(
        runner, "_build_retriever", Mock(side_effect=AssertionError("不应构建 retriever")))
    judge = Mock(return_value={"faithfulness": 1.0})
    monkeypatch.setattr(runner, "_run_ragas", judge)
    ragas_from_results(path, REPLAY_CFG, sample_n=3)
    judge.assert_called_once()


def test_context_mismatch_refuses_to_judge(tmp_path, monkeypatch):
    """重放对不上 = 答案不是对着这份上下文生成的。拒判，而不是送进付费 judge。"""
    path, _ = _write_results(tmp_path, n=2)
    _, _modes = _patch_retrieval(monkeypatch, {"问题0": "别的正文", "问题1": "别的正文"})
    judge = Mock(side_effect=AssertionError("上下文对不上就不该进 judge"))
    monkeypatch.setattr(runner, "_run_ragas", judge)
    with pytest.raises(ValueError, match="无法按原样复现"):
        ragas_from_results(path, REPLAY_CFG, sample_n=2)
    judge.assert_not_called()


def test_replay_restores_saved_retrieval_mode(tmp_path, monkeypatch):
    """meta 记着 [dense] 就按 dense 构建 retriever；不还原就是拿 hybrid 上下文判 dense 的答案。"""
    path, _ = _write_results(tmp_path, n=2)
    seen, built_modes = _patch_retrieval(monkeypatch, {f"问题{i}": f"正文{i}" for i in range(2)})
    monkeypatch.setattr(runner, "_run_ragas", Mock(return_value={"faithfulness": 1.0}))
    ragas_from_results(path, REPLAY_CFG, sample_n=2)
    assert built_modes == ["dense"]  # 构建时注入（一次），配置里的 hybrid 被覆盖
    assert len(seen) == 2


def test_sample_rows_is_idempotent_for_pre_sampled_input():
    rows = list(range(55))
    once = _sample_rows(rows, 15)
    assert _sample_rows(once, 15) == once  # 调用方预抽样后二次抽样不得再变


# ---------- 入库：LLM 抽取默认必须关，limit 必须约束入库 ----------


def _write_parsed(parsed_dir, n=5):
    for i in range(n):
        (parsed_dir / f"d{i}.json").write_text(json.dumps({
            "meta": {"source_type": "pdf", "doc_id": f"d{i}", "title": f"t{i}"},
            "blocks": [{"type": "p", "text": "t", "section_path": [], "page": 1,
                        "block_type": "p"}]}), encoding="utf-8")


@pytest.fixture
def index_env(tmp_path, monkeypatch):
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    monkeypatch.setattr(indexer, "QdrantClient", Mock())
    monkeypatch.setattr(indexer, "ensure_collection", Mock())
    monkeypatch.setattr(indexer.Embedder, "__init__", lambda self, cfg: None)
    monkeypatch.setattr(indexer.Embedder, "embed",
                        lambda self, texts: [[0.0]] * len(texts))
    monkeypatch.setattr(indexer, "chunk_by", lambda strategy, doc: [Mock(
        chunk_id="c", section_path=[], text="t", page=1, block_type="p")])
    monkeypatch.setattr(indexer, "build_bm25_text", lambda t: t)
    monkeypatch.setattr(indexer, "base_meta", lambda d: {
        "doc_date": None, "category": None, "doc_group": None,
        "meeting_type": None, "attendees": [], "topics": []})
    return parsed


INDEX_CFG = {"qdrant": {"url": "http://x", "collection": "c"},
             "embedding": {"dense_dim": 4, "base_url": "http://e", "model": "m", "api_key": "k"},
             "llm": {"model": "m", "base_url": "http://l", "api_key": "k"},
             "metadata_extraction": {"enabled": False}}


def test_index_default_keeps_llm_meta_off(index_env, monkeypatch):
    """库函数默认绝不能开 LLM 抽取：1130 次调用是本项目最大的成本单点。"""
    _write_parsed(index_env)
    extract = Mock()
    monkeypatch.setattr(indexer, "extract_metadata", extract)
    stats = indexer.index_parsed(index_env, INDEX_CFG)
    assert stats["llm_meta"] is False
    assert stats["docs"] == 5
    extract.assert_not_called()


def test_index_honors_enabled_flag_and_explicit_optin(index_env, monkeypatch):
    """配置 enabled=true 时缺省生效；显式传 False 可以覆盖回关。"""
    _write_parsed(index_env, n=2)
    extract = Mock(side_effect=lambda d, cfg: {"doc_date": None, "category": None,
                                               "doc_group": None, "meeting_type": None,
                                               "attendees": [], "topics": []})
    monkeypatch.setattr(indexer, "extract_metadata", extract)
    cfg_on = {**INDEX_CFG, "metadata_extraction": {"enabled": True}}
    assert indexer.index_parsed(index_env, cfg_on)["llm_meta"] is True
    assert extract.call_count == 2
    extract.reset_mock()
    assert indexer.index_parsed(index_env, cfg_on, use_llm_meta=False)["llm_meta"] is False
    extract.assert_not_called()


def test_index_limit_bounds_indexing(index_env):
    """--limit 也必须约束入库：此前只限解析，试跑照样全量嵌入。"""
    _write_parsed(index_env)
    upserts: list[int] = []
    client = Mock()
    client.upsert.side_effect = lambda name, points: upserts.append(len(points))
    monkey = pytest.MonkeyPatch()
    monkey.setattr(indexer, "QdrantClient", lambda **kw: client)
    try:
        stats = indexer.index_parsed(index_env, INDEX_CFG, limit=2)
    finally:
        monkey.undo()
    assert stats["docs"] == 2  # 只入库 2 篇，不是 5 篇

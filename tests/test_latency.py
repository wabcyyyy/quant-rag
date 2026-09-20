"""延迟测量回归测试——全部离线（mock），不发任何真实 LLM/embedding 调用。

守护对象是「数字可信」：延迟曾因「无测量代码 + 缓存命中混淆 + 无模型归属」
而变成 PLAN 里两个互相矛盾的数（1.3s vs 5.3~7.4s）。这里锁住三件事：
1. 缓存命中的耗时必须带 cached 标志（它不是模型延迟）；
2. 分位数用最近秩法，小样本下不虚高；
3. 汇总把合成分位数限制在未命中缓存的条目上，并置 cache_contaminated。
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from doc_rag.eval import runner
from doc_rag.eval.runner import _latency_summary, _quantiles
from doc_rag.generate import llm as llm_mod
from doc_rag.retrieve.hybrid import RetrievalOutcome

# ---------------------------------------------------------------- _quantiles


def test_quantiles_nearest_rank_on_small_sample():
    """最近秩法：n=4 时 p50 取第 2 个、p95 取最大值（不做插值）。"""
    q = _quantiles([1.0, 2.0, 3.0, 4.0])
    assert q["n"] == 4
    assert q["p50"] == 2.0
    assert q["p95"] == 4.0
    assert q["max"] == 4.0
    assert q["min"] == 1.0


def test_quantiles_ignores_none_and_handles_empty():
    q = _quantiles([None, 10.0, None, 30.0])
    assert q["n"] == 2
    assert q["p50"] == 10.0
    assert _quantiles([]) == {}
    assert _quantiles([None]) == {}


def test_quantiles_never_exceeds_max():
    """插值实现会给出比任何真实请求都大的 p95；最近秩法必须落在样本内。"""
    xs = [5.0, 100.0, 700.0, 12000.0]
    q = _quantiles(xs)
    assert q["p95"] <= max(xs)


# ------------------------------------------------------- 缓存标志与计时口径


def test_chat_timed_flags_cache_hit_and_reports_ms(monkeypatch):
    """缓存命中必须带 cached=True —— 否则它的毫秒数会被当成模型延迟。"""
    monkeypatch.setattr(llm_mod, "cache_enabled", lambda cfg=None: True)
    monkeypatch.setattr(llm_mod, "_cache_get", lambda key: "缓存的答案")
    cfg = {"model": "m", "base_url": "https://api.x.com", "api_key": "k"}
    text, meta = llm_mod.chat_timed(cfg, "问题")
    assert text == "缓存的答案"
    assert meta["cached"] is True
    assert meta["model"] == "m"
    assert meta["ms"] >= 0


def test_chat_timed_flags_real_call_and_counts_attempts(monkeypatch):
    monkeypatch.setattr(llm_mod, "cache_enabled", lambda cfg=None: False)
    monkeypatch.setattr(llm_mod, "_cache_put", lambda *a, **k: None)
    usage = Mock(prompt_tokens=10, completion_tokens=5)
    usage.completion_tokens_details = Mock(reasoning_tokens=1)
    resp = Mock()
    resp.choices = [Mock(message=Mock(content="真答案"))]
    resp.usage = usage

    class _FakeClient:
        def __init__(self, **kwargs):
            self.chat = Mock(completions=Mock(create=lambda **kw: resp))

    monkeypatch.setattr(llm_mod, "OpenAI", _FakeClient)
    cfg = {"model": "m", "base_url": "https://api.x.com", "api_key": "k"}
    text, meta = llm_mod.chat_timed(cfg, "问题")
    assert text == "真答案"
    assert meta["cached"] is False
    assert meta["attempts"] == 1


def test_client_timeout_comes_from_cfg_not_the_180s_default(monkeypatch):
    """关键路径上的短调用必须能自带超时上限：180s 是给几十秒的聚合答案用的，
    一次意图分类吃同样的超时，等于让它有能力把整个请求拖停三分钟。"""
    seen: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            resp = Mock()
            resp.choices = [Mock(message=Mock(content="x"))]
            resp.usage = None
            self.chat = Mock(completions=Mock(create=lambda **kw: resp))

    monkeypatch.setattr(llm_mod, "OpenAI", _FakeClient)
    monkeypatch.setattr(llm_mod, "cache_enabled", lambda cfg=None: False)
    monkeypatch.setattr(llm_mod, "_cache_put", lambda *a, **k: None)
    llm_mod.chat_timed(
        {"model": "m", "base_url": "https://api.x", "api_key": "k", "timeout_s": 5},
        "问题",
    )
    assert seen["timeout"] == 5.0
    assert seen["max_retries"] == 0  # 重试只归应用层管，不与 SDK 相乘

    seen.clear()
    llm_mod.chat_timed(
        {"model": "m", "base_url": "https://api.x", "api_key": "k"}, "问题"
    )
    assert seen["timeout"] == llm_mod._DEFAULT_TIMEOUT_S


def test_max_attempts_bounds_the_retry_loop_and_is_reported_honestly(monkeypatch):
    """`已重试 N 次` 必须是真的 N；上限可由配置收紧，且退避不该把上限跑满。"""

    class _Flaky(Exception):
        status_code = 503

    attempts: list[int] = []

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        @property
        def chat(self):
            def _create(**kw):
                attempts.append(1)
                raise _Flaky("上游抖动")

            return Mock(completions=Mock(create=_create))

    monkeypatch.setattr(llm_mod, "OpenAI", _FakeClient)
    monkeypatch.setattr(llm_mod, "cache_enabled", lambda cfg=None: False)
    slept: list[float] = []
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: slept.append(s))
    base = {"model": "m", "base_url": "https://api.x", "api_key": "k"}
    with pytest.raises(RuntimeError, match="attempt=2"):
        llm_mod.chat_timed({**base, "max_attempts": 2}, "问题")
    assert len(attempts) == 2
    assert slept == [llm_mod._BACKOFF_BASE]  # 只在两次之间退避一次

    attempts.clear()
    slept.clear()
    with pytest.raises(RuntimeError, match=f"attempt={llm_mod._RETRIES}"):
        llm_mod.chat_timed(base, "问题")
    assert len(attempts) == llm_mod._RETRIES


def test_chat_still_returns_plain_text(monkeypatch):
    """chat() 是既有调用点依赖的接口，必须仍返回 str。"""
    monkeypatch.setattr(llm_mod, "cache_enabled", lambda cfg=None: True)
    monkeypatch.setattr(llm_mod, "_cache_get", lambda key: "答案")
    cfg = {"model": "m", "base_url": "https://api.x.com", "api_key": "k"}
    assert llm_mod.chat(cfg, "问题") == "答案"


def test_reasoning_effort_reaches_request_kwargs(monkeypatch):
    """合成侧的思考开关必须真的进请求参数（压聚合题延迟的唯一杠杆）。"""
    seen: dict = {}
    monkeypatch.setattr(llm_mod, "cache_enabled", lambda cfg=None: False)
    monkeypatch.setattr(llm_mod, "_cache_put", lambda *a, **k: None)

    class _FakeClient:
        def __init__(self, **kwargs):
            def _create(**kw):
                seen.update(kw)
                resp = Mock()
                resp.choices = [Mock(message=Mock(content="x"))]
                resp.usage = None
                return resp

            self.chat = Mock(completions=Mock(create=_create))

    monkeypatch.setattr(llm_mod, "OpenAI", _FakeClient)
    cfg = {
        "model": "m",
        "base_url": "https://api.x.com",
        "api_key": "k",
        "reasoning_effort": "none",
    }
    llm_mod.chat(cfg, "问题")
    assert seen.get("reasoning_effort") == "none"


def test_no_reasoning_effort_key_when_unset(monkeypatch):
    """默认（不配）不得凭空塞参数——那会改变现有基线行为。"""
    seen: dict = {}
    monkeypatch.setattr(llm_mod, "cache_enabled", lambda cfg=None: False)
    monkeypatch.setattr(llm_mod, "_cache_put", lambda *a, **k: None)

    class _FakeClient:
        def __init__(self, **kwargs):
            def _create(**kw):
                seen.update(kw)
                resp = Mock()
                resp.choices = [Mock(message=Mock(content="x"))]
                resp.usage = None
                return resp

            self.chat = Mock(completions=Mock(create=_create))

    monkeypatch.setattr(llm_mod, "OpenAI", _FakeClient)
    llm_mod.chat(
        {"model": "m", "base_url": "https://api.x.com", "api_key": "k"}, "问题"
    )
    assert "reasoning_effort" not in seen


# ------------------------------------------------------------ 延迟汇总口径


def _item(id_, type_, answer, synth_ms, cached, total_ms=None):
    return {
        "id": id_,
        "type": type_,
        "answer": answer,
        "latency": {
            "rewrite": 1.0,
            "retrieve": 200.0,
            "rerank": 50.0,
            "retrieval_total": 251.0,
            "synthesize": synth_ms,
            "synth_cached": cached,
            "total": total_ms if total_ms is not None else (synth_ms or 0) + 251.0,
        },
    }


def test_latency_summary_excludes_cached_from_synthesize_percentiles():
    """缓存命中的合成耗时不得进入合成分位数——否则 LLM 延迟被拉到毫秒级。"""
    rows = [
        _item("q1", "fact", "短答案", 3000.0, False),
        _item("q2", "fact", "短答案", 2.0, True),  # 缓存命中：2ms 不是延迟
        _item("q3", "fact", "短答案", 4000.0, False),
    ]
    lat = _latency_summary(rows)
    assert lat["synthesize_n_uncached"] == 2
    assert lat["synthesize"]["n"] == 2
    assert lat["synthesize"]["min"] == 3000.0
    assert lat["cached_answers"] == 1
    assert lat["cache_contaminated"] is True


def test_latency_summary_clean_run_not_contaminated():
    rows = [_item("q1", "fact", "答案", 3000.0, False)]
    lat = _latency_summary(rows)
    assert lat["cache_contaminated"] is False
    assert lat["cached_answers"] == 0


def test_latency_summary_by_type_exposes_the_tail():
    """延迟是双峰的：聚合题（长答案）必须单独可见，不能被混合 P95 藏起来。"""
    rows = [
        _item("q1", "fact", "短", 2000.0, False),
        _item("q2", "fact", "短", 2500.0, False),
        _item("q3", "cross_doc", "长" * 600, 30000.0, False),
    ]
    lat = _latency_summary(rows)
    assert lat["by_type"]["fact"]["synthesize"]["p95"] == 2500.0
    assert lat["by_type"]["cross_doc"]["synthesize"]["p95"] == 30000.0
    assert lat["by_type"]["cross_doc"]["answer_chars_mean"] == 600.0
    assert lat["by_type"]["fact"]["answer_chars_mean"] == 1.0


def test_latency_summary_target_verdict_tracks_p95():
    fast = _latency_summary([_item("q1", "fact", "a", 1000.0, False)])
    slow = _latency_summary([_item("q1", "cross_doc", "a", 30000.0, False)])
    assert fast["p95_meets_target"] is True
    assert slow["p95_meets_target"] is False
    assert fast["target_p95_ms"] == 8000


def test_latency_summary_empty_without_timing():
    assert _latency_summary([]) is None
    assert _latency_summary([{"id": "q1", "type": "fact"}]) is None


# ------------------------------------------------- runner 端到端接线（离线）


def test_evaluate_records_latency_and_self_documenting_meta(tmp_path, monkeypatch):
    """evaluate 必须落盘分阶段延迟，并让 meta 自证模型与缓存开关。"""
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

    retriever = Mock(collection="c", cfg={})
    retriever.retrieve.return_value = RetrievalOutcome(
        chunks=[
            {
                "doc_id": "d1",
                "title": "报价",
                "page": 1,
                "text": "费用67元",
                "block_type": "text",
            }
        ]
    )
    synthesizer = Mock()
    synthesizer.answer.return_value = "费用67元 [1]"
    synthesizer.last_meta = {"ms": 1234.5, "cached": False, "model": "deepseek-flash"}
    monkeypatch.setattr(
        runner, "_build_retriever", Mock(return_value=(retriever, synthesizer))
    )

    cfg = {"retrieval": {}, "llm": {"model": "deepseek-flash", "cache": False}}
    results = runner.evaluate(gold, cfg=cfg)

    item = results["items"][0]
    assert item["latency"]["synthesize"] == 1234.5
    assert item["latency"]["synth_cached"] is False
    assert item["latency"]["total"] is not None
    lat = results["summary"]["latency"]
    assert lat["synthesize"]["p95"] == 1234.5
    assert lat["cache_contaminated"] is False
    # 数字必须能追到「哪个模型、缓存开没开」
    assert results["meta"]["llm_model"] == "deepseek-flash"
    assert results["meta"]["answer_cache"] is False


def test_evaluate_marks_contaminated_when_cache_hit(tmp_path, monkeypatch):
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

    retriever = Mock(collection="c", cfg={})
    retriever.retrieve.return_value = RetrievalOutcome(
        chunks=[
            {
                "doc_id": "d1",
                "title": "报价",
                "page": 1,
                "text": "费用67元",
                "block_type": "text",
            }
        ]
    )
    synthesizer = Mock()
    synthesizer.answer.return_value = "费用67元 [1]"
    synthesizer.last_meta = {"ms": 1.2, "cached": True, "model": "m"}
    monkeypatch.setattr(
        runner, "_build_retriever", Mock(return_value=(retriever, synthesizer))
    )

    cfg = {"retrieval": {}, "llm": {"model": "m", "cache": True}}
    results = runner.evaluate(gold, cfg=cfg)
    lat = results["summary"]["latency"]
    assert lat["cache_contaminated"] is True
    assert lat["synthesize"] == {}  # 唯一一条命中缓存 → 合成分位数无样本
    assert results["meta"]["answer_cache"] is True


def test_evaluate_without_answers_has_no_synthesize_timing(tmp_path, monkeypatch):
    """检索模式不调 LLM：合成分位数应为空，但检索侧延迟照常记录。"""
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

    retriever = Mock(collection="c", cfg={})
    retriever.retrieve.return_value = RetrievalOutcome(
        chunks=[
            {
                "doc_id": "d1",
                "title": "报价",
                "page": 1,
                "text": "费用67元",
                "block_type": "text",
            }
        ]
    )
    monkeypatch.setattr(
        runner, "_build_retriever", Mock(return_value=(retriever, Mock()))
    )

    cfg = {"retrieval": {}, "llm": {"model": "m"}}
    results = runner.evaluate(gold, cfg=cfg, with_answers=False)
    lat = results["summary"]["latency"]
    assert lat["synthesize"] == {}
    assert lat["by_stage"]["retrieval_total"]["n"] == 1
    assert results["meta"]["answer_cache"] is None


def test_synthesizer_last_meta_is_none_before_first_answer():
    from doc_rag.generate.synthesizer import Synthesizer

    assert Synthesizer({"model": "m"}).last_meta is None


@pytest.mark.parametrize("cached", [True, False])
def test_synthesizer_records_meta(monkeypatch, cached):
    from doc_rag.generate.synthesizer import Synthesizer

    monkeypatch.setattr(
        llm_mod,
        "chat_timed",
        lambda *a, **k: ("答", {"ms": 42.0, "cached": cached, "model": "m"}),
    )
    syn = Synthesizer({"model": "m"})
    assert syn.answer("q", [{"no": 1, "text": "t", "doc": "d", "page": 1}]) == "答"
    assert syn.last_meta["cached"] is cached


# ------------------------------------------------------- 聚合题思考分流开关


def _capture_cfg(monkeypatch):
    seen: list = []
    monkeypatch.setattr(
        llm_mod,
        "chat_timed",
        lambda cfg, *a, **k: (
            seen.append(dict(cfg)),
            "答",
            {"ms": 1.0, "cached": False, "model": cfg["model"]},
        )[1:],
    )
    return seen


def test_by_type_table_overrides_only_listed_types(monkeypatch):
    """表里列出的题型用它，没列出的跟随全局——这张表的核心语义。"""
    from doc_rag.generate.synthesizer import Synthesizer

    seen = _capture_cfg(monkeypatch)
    syn = Synthesizer(
        {
            "model": "m",
            "reasoning_effort": "",
            "reasoning_effort_by_type": {"cross_doc": "none", "time_filter": "none"},
        }
    )
    ctx = [{"no": 1, "text": "t", "doc": "d", "page": 1}]
    syn.answer("单点题", ctx, question_type="single")
    syn.answer("聚合题", ctx, question_type="cross_doc")
    syn.answer("时间聚合题", ctx, question_type="time_filter")
    assert not seen[0].get("reasoning_effort")  # single 不在表里 → 全局空 = 不进参数
    assert seen[1]["reasoning_effort"] == "none"
    assert seen[2]["reasoning_effort"] == "none"


def test_unlisted_type_and_absent_table_both_follow_global(monkeypatch):
    """没配表 / 没传题型时，行为与引入这张表之前逐字一致。"""
    from doc_rag.generate.synthesizer import Synthesizer

    seen = _capture_cfg(monkeypatch)
    ctx = [{"no": 1, "text": "t", "doc": "d", "page": 1}]
    Synthesizer({"model": "m", "reasoning_effort": "none"}).answer(
        "聚合题", ctx, question_type="cross_doc"
    )
    assert seen[0]["reasoning_effort"] == "none"  # 回落全局
    Synthesizer({"model": "m", "reasoning_effort": ""}).answer("q", ctx)
    assert not seen[1].get("reasoning_effort")


def test_explicit_empty_cell_forces_thinking_on(monkeypatch):
    """`single: ""` 是合法配置：该题型强制开思考，即使全局是 none。

    「键不存在」与「键存在但值为空」必须是两件事，否则这张表表达不出
    「全局关思考、只给某一类留思考」这条政策。
    """
    from doc_rag.generate.synthesizer import Synthesizer

    seen = _capture_cfg(monkeypatch)
    syn = Synthesizer(
        {
            "model": "m",
            "reasoning_effort": "none",
            "reasoning_effort_by_type": {"single": ""},
        }
    )
    syn.answer(
        "单点题",
        [{"no": 1, "text": "t", "doc": "d", "page": 1}],
        question_type="single",
    )
    assert not seen[0].get("reasoning_effort")


def test_unpredictable_type_key_is_rejected_not_ignored():
    """填 `term: low` 必须炸：fact/term 是黄金集标注，服务侧预测不到。

    一个读不到的配置键比没有这个键更糟——设的人会以为分档已经生效。
    """
    import pytest

    from doc_rag.generate.synthesizer import Synthesizer

    with pytest.raises(ValueError, match="term"):
        Synthesizer(
            {
                "model": "m",
                "reasoning_effort_by_type": {"term": "low", "cross_doc": "none"},
            }
        )


def test_shipped_config_splits_aggregates_without_any_env(monkeypatch):
    """裸检出（不设任何环境变量）也必须拿到聚合题关思考这件事。

    旧形状里这个行为靠 `DOC_RAG_LLM_REASONING_EFFORT_AGGREGATE` 驱动，env 不设就静默
    退回全开思考——默认值不该由部署环境决定。
    """
    from doc_rag.config import load_config
    from doc_rag.generate.synthesizer import Synthesizer

    seen = _capture_cfg(monkeypatch)
    table = load_config()["llm"]["reasoning_effort_by_type"]
    assert table == {"cross_doc": "none", "time_filter": "none"}
    syn = Synthesizer(load_config()["llm"])
    syn.answer(
        "聚合题",
        [{"no": 1, "text": "t", "doc": "d", "page": 1}],
        question_type="cross_doc",
    )
    assert seen[0]["reasoning_effort"] == "none"


def test_original_llm_cfg_not_mutated(monkeypatch):
    """分档只影响本次调用：Synthesizer 的配置对象不得被原地改写。"""
    from doc_rag.generate.synthesizer import Synthesizer

    _capture_cfg(monkeypatch)
    cfg = {"model": "m", "reasoning_effort_by_type": {"cross_doc": "none"}}
    syn = Synthesizer(cfg)
    syn.answer(
        "聚合题",
        [{"no": 1, "text": "t", "doc": "d", "page": 1}],
        question_type="cross_doc",
    )
    assert "reasoning_effort" not in cfg
    assert cfg["reasoning_effort_by_type"] == {"cross_doc": "none"}


def test_runner_passes_predicted_type_to_synthesizer(tmp_path, monkeypatch):
    """runner 必须把**预测题型**传下去，否则思考档那张表在评估路径上不生效。"""
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "q001",
                        "type": "cross_doc",
                        "question": "关于X做过哪些决定？",
                        "expected_answer": "",
                        "source_doc_ids": ["d1"],
                        "must_contain": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    retriever = Mock(collection="c", cfg={})
    retriever.retrieve.return_value = RetrievalOutcome(
        chunks=[
            {"doc_id": "d1", "title": "t", "page": 1, "text": "正文", "block_type": "p"}
        ]
    )
    syn = Mock()
    syn.answer.return_value = "答案"
    monkeypatch.setattr(runner, "_build_retriever", Mock(return_value=(retriever, syn)))

    results = runner.evaluate(
        gold, cfg={"retrieval": {}, "llm": {"model": "m"}}, aggregate=True
    )
    assert results["items"][0]["answer"] == "答案"
    assert syn.answer.call_args.kwargs.get("question_type") == "cross_doc"


# ------------------------------------------------------- 逐条 token 用量（T5）


def _fake_client_with(resp_holder: dict, usage):
    class _FakeClient:
        def __init__(self, **kwargs):
            def _create(**kw):
                resp = Mock()
                resp.choices = [Mock(message=Mock(content="答案"))]
                resp.usage = usage
                return resp

            self.chat = Mock(completions=Mock(create=_create))

    return _FakeClient


def test_chat_timed_reports_usage_in_meta(monkeypatch):
    """每次真实调用的 usage 必须随 meta 透传——延迟归因（reasoning vs 耗时）靠它。"""
    monkeypatch.setattr(llm_mod, "cache_enabled", lambda cfg=None: False)
    monkeypatch.setattr(llm_mod, "_cache_put", lambda *a, **k: None)
    usage = Mock(prompt_tokens=100, completion_tokens=50)
    usage.completion_tokens_details = Mock(reasoning_tokens=40)
    monkeypatch.setattr(llm_mod, "OpenAI", _fake_client_with({}, usage))
    cfg = {"model": "m", "base_url": "https://api.x.com", "api_key": "k"}
    _, meta = llm_mod.chat_timed(cfg, "问题")
    assert meta["prompt_tokens"] == 100
    assert meta["completion_tokens"] == 50
    assert meta["reasoning_tokens"] == 40


def test_chat_timed_usage_defaults_to_none_without_usage(monkeypatch):
    """响应无 usage 时缺省为 None 而不是 0——0 会冒充「真实为零」的用量。"""
    monkeypatch.setattr(llm_mod, "cache_enabled", lambda cfg=None: False)
    monkeypatch.setattr(llm_mod, "_cache_put", lambda *a, **k: None)
    monkeypatch.setattr(llm_mod, "OpenAI", _fake_client_with({}, None))
    cfg = {"model": "m", "base_url": "https://api.x.com", "api_key": "k"}
    _, meta = llm_mod.chat_timed(cfg, "问题")
    assert meta["prompt_tokens"] is None
    assert meta["completion_tokens"] is None
    assert meta["reasoning_tokens"] is None


def test_chat_timed_cached_hit_has_no_usage(monkeypatch):
    """缓存命中没有本次调用可言，usage 必须为 None（不能拿上一次的数冒充）。"""
    monkeypatch.setattr(llm_mod, "cache_enabled", lambda cfg=None: True)
    monkeypatch.setattr(llm_mod, "_cache_get", lambda key: "缓存答案")
    cfg = {"model": "m", "base_url": "https://api.x.com", "api_key": "k"}
    _, meta = llm_mod.chat_timed(cfg, "问题")
    assert meta["cached"] is True
    assert meta["prompt_tokens"] is None


def test_evaluate_records_usage_per_item_and_totals(tmp_path, monkeypatch):
    """runner 把 synth_meta 的用量写进逐条 latency.usage，summary 给全量合计。"""
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
                    },
                    {
                        "id": "q002",
                        "type": "fact",
                        "question": "预算？",
                        "expected_answer": "9元",
                        "source_doc_ids": ["d1"],
                        "must_contain": ["9元"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    retriever = Mock(collection="c", cfg={})
    retriever.retrieve.return_value = RetrievalOutcome(
        chunks=[
            {
                "doc_id": "d1",
                "title": "报价",
                "page": 1,
                "text": "费用67元",
                "block_type": "text",
            }
        ]
    )
    synthesizer = Mock()
    synthesizer.answer.side_effect = ["费用67元 [1]", "预算9元 [1]"]
    synthesizer.last_meta = {
        "ms": 5000.0,
        "cached": False,
        "model": "m",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "reasoning_tokens": 40,
    }
    monkeypatch.setattr(
        runner, "_build_retriever", Mock(return_value=(retriever, synthesizer))
    )

    cfg = {"retrieval": {}, "llm": {"model": "m", "cache": False}}
    results = runner.evaluate(gold, cfg=cfg)
    item = results["items"][0]
    assert item["latency"]["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "reasoning_tokens": 40,
    }
    usage = results["summary"]["latency"]["usage"]
    assert usage["prompt_tokens"] == 200
    assert usage["completion_tokens"] == 100
    assert usage["reasoning_tokens"] == 80
    assert results["summary"]["latency"]["usage_n"] == 2


def test_evaluate_without_usage_still_writes_results(tmp_path, monkeypatch):
    """Mock/旧版 synthesizer 没有 usage 时：逐条为 None，summary 不造合计。"""
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
    retriever = Mock(collection="c", cfg={})
    retriever.retrieve.return_value = RetrievalOutcome(
        chunks=[
            {
                "doc_id": "d1",
                "title": "报价",
                "page": 1,
                "text": "费用67元",
                "block_type": "text",
            }
        ]
    )
    synthesizer = Mock()
    synthesizer.answer.return_value = "费用67元 [1]"
    synthesizer.last_meta = {"ms": 1234.5, "cached": False, "model": "m"}
    monkeypatch.setattr(
        runner, "_build_retriever", Mock(return_value=(retriever, synthesizer))
    )

    results = runner.evaluate(gold, cfg={"retrieval": {}, "llm": {"model": "m"}})
    assert results["items"][0]["latency"]["usage"] is None
    assert "usage" not in results["summary"]["latency"]


# ------------------------------------------------------- prompt 版本指纹


def test_prompt_fingerprint_changes_with_prompt_text():
    """prompt 任何一字改动都必须换指纹——答案归属的根子。"""
    from doc_rag.generate import prompts

    fp1 = prompts.fingerprint()
    original = prompts.USER_ANSWER
    try:
        prompts.USER_ANSWER = original + "\n"
        assert prompts.fingerprint() != fp1
    finally:
        prompts.USER_ANSWER = original
    assert prompts.fingerprint() == fp1


def test_evaluate_meta_records_prompt_fingerprint(tmp_path, monkeypatch):
    from doc_rag.generate import prompts

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
    retriever = Mock(collection="c", cfg={})
    retriever.retrieve.return_value = RetrievalOutcome(
        chunks=[
            {
                "doc_id": "d1",
                "title": "t",
                "page": 1,
                "text": "费用67元",
                "block_type": "p",
            }
        ]
    )
    synthesizer = Mock()
    synthesizer.answer.return_value = "费用67元 [1]"
    monkeypatch.setattr(
        runner, "_build_retriever", Mock(return_value=(retriever, synthesizer))
    )

    results = runner.evaluate(gold, cfg={"retrieval": {}, "llm": {"model": "m"}})
    assert results["meta"]["prompt_fingerprint"] == prompts.fingerprint()
    # 检索模式没有答案，prompt 指纹无意义
    results2 = runner.evaluate(
        gold, cfg={"retrieval": {}, "llm": {"model": "m"}}, with_answers=False
    )
    assert results2["meta"]["prompt_fingerprint"] is None

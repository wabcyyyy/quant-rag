"""两段式合成（ADR-0002）的离线单测。

锁四件事（UPGRADE §3.5 的清单）：
1. 路由只认预测题型：域外键报错，不静默失效；
2. map 退化三级路径：单篇失败降级原文首块 / 失败率超阈值整题退回单发；
3. 落盘纪律：summaries 缺失时拒绝重放；contexts 永远存原始块，摘要不进去；
4. reduce 上下文的重构与生成逐字一致（judge 所见 = LLM 所见的唯一前提）。
"""

from __future__ import annotations

import threading
from typing import ClassVar

import pytest

from doc_rag.generate import two_stage
from doc_rag.generate.synthesizer import Synthesizer
from doc_rag.orchestrator import Orchestrator
from doc_rag.retrieve.hybrid import RetrievalOutcome


def _cfg(**over):
    cfg = {
        "llm": {"model": "m", "base_url": "http://l", "api_key": "k"},
        "retrieval": {"mode": "hybrid", "max_contexts": 10},
        "rerank": {
            "enabled": False,
            "base_url": "http://r",
            "api_key": "k",
            "model": "rr",
            "top_n": 6,
        },
        "qdrant": {"url": "http://x", "collection": "kb"},
        "embedding": {"base_url": "http://e", "api_key": "k", "model": "e"},
        "synthesis": {
            "two_stage": {
                "enabled": True,
                "types": ["cross_doc", "time_filter"],
                "timeout_s": 20,
                "max_attempts": 1,
                "fail_ratio": 0.3,
                "workers": 2,
            }
        },
    }
    cfg["synthesis"]["two_stage"].update(over)
    return cfg


def _contexts(n: int = 4, per_doc: int = 2) -> list[dict]:
    """手工构造的原始块上下文（n 块，每 per_doc 块属同一篇文档；页码从 1 起）。"""
    out = []
    for i in range(n):
        out.append(
            {
                "no": i + 1,
                "text": f"正文{i}",
                "doc": f"文档{i // per_doc}",
                "page": i + 1,
                "doc_id": f"d{i // per_doc}",
            }
        )
    return out


# ── 1. 路由配置 ───────────────────────────────────────────────────────────


def test_route_rejects_out_of_domain_types():
    with pytest.raises(ValueError, match="预测不到的题型"):
        two_stage.two_stage_cfg(_cfg(types=["term"]))


def test_route_disabled_by_default():
    assert two_stage.two_stage_cfg({})["enabled"] is False
    assert not two_stage.is_routed({}, "cross_doc")


def test_route_follows_predicted_type_only():
    cfg = _cfg()
    assert two_stage.is_routed(cfg, "cross_doc")
    assert two_stage.is_routed(cfg, "time_filter")
    # 单发题型不在路由域内：即使开关开着，fact/term 也走单发口径
    assert not two_stage.is_routed(cfg, "single")


# ── 2. map 段与退化路径 ───────────────────────────────────────────────────


class _MapScript:
    """按文档名给微摘要；`fail` 集合里的文档抛错。线程安全（map 走线程池）。"""

    def __init__(self, fail: set[str]):
        self.fail = fail
        self.lock = threading.Lock()
        self.calls: list[str] = []

    def __call__(self, llm_cfg, user_prompt, system_prompt=None, temperature=None):
        assert system_prompt == two_stage.SYSTEM_MAP
        assert llm_cfg["reasoning_effort"] == "none", "map 段必须关思考"
        assert llm_cfg["timeout_s"] == 20, "map 段用自己的短预算，不是合成的 180s"
        with self.lock:
            self.calls.append(user_prompt)
        for doc in self.fail:
            if f"【文档】{doc}" in user_prompt:
                raise RuntimeError("503 map 抖动")
        return "该篇决定：事项甲", {
            "ms": 0.3,
            "cached": False,
            "model": "m",
            "prompt_tokens": 60,
            "completion_tokens": 12,
            "reasoning_tokens": 0,
        }


def test_map_degrades_single_doc_but_keeps_the_question(monkeypatch):
    script = _MapScript(fail={"文档1"})
    monkeypatch.setattr(two_stage.llm, "chat_timed", script)
    # 8 块 / 4 篇：1 篇失败 = 25% ≤ fail_ratio(30%) → 降级但不退回
    summaries, meta = two_stage.map_summaries(_cfg(), "问题", _contexts(8, per_doc=2))

    assert summaries is not None
    assert meta["n_docs"] == 4 and meta["n_degraded"] == 1
    degraded = next(s for s in summaries if s["degraded"])
    # 降级摘要 = 该篇原文首块，编号也指回原始块——reduce 对它无感知
    assert degraded["text"] == "正文2"
    assert degraded["source_context_idx"] == [3]
    ok = next(s for s in summaries if not s["degraded"])
    assert ok["source_context_idx"] == [1, 2]
    assert meta["usage"]["completion_tokens"] == 36  # 3 篇成功 × 12；失败那篇不冒充 0


def test_map_aborts_whole_item_when_fail_ratio_exceeded(monkeypatch):
    script = _MapScript(fail={"文档0", "文档1"})
    monkeypatch.setattr(two_stage.llm, "chat_timed", script)
    summaries, meta = two_stage.map_summaries(_cfg(), "问题", _contexts())

    assert summaries is None  # 2/2 = 100% > 30% → 整题退回单发
    assert meta["aborted"] == "fail_ratio_exceeded"


def test_fallback_returns_to_single_budget_and_single_prompt(monkeypatch):
    """整题退回 = 口径退回：上下文换回 min(max_contexts, top_n) 那份，合成走单发。"""

    class _Rec:
        collection = "kb"

        def retrieve(self, question, top_n=None, **kw):
            return RetrievalOutcome(
                chunks=[
                    {
                        "doc_id": f"d{i}",
                        "title": f"文档{i}",
                        "page": i + 1,
                        "text": f"正文{i}",
                        "block_type": "paragraph",
                        "score": 1.0,
                    }
                    for i in range(8)
                ]
            )

    monkeypatch.setattr(
        two_stage, "map_summaries", lambda cfg, q, ctx: (None, {"aborted": "x"})
    )
    seen: list[dict] = []

    def _timed(llm_cfg, user_prompt, system_prompt=None, temperature=None):
        seen.append({"user": user_prompt, "system": system_prompt})
        return "答案 [1]", {"ms": 1.0, "cached": False, "model": "m"}

    monkeypatch.setattr(two_stage.llm, "chat_timed", _timed)
    synth = Synthesizer(_cfg()["llm"])
    orch = Orchestrator(_cfg(), retriever=_Rec(), synthesizer=synth)
    plan = {
        "rewritten": "q",
        "filters": None,
        "aggregate": True,  # predicted = cross_doc → 命中路由域
        "top_n": None,
        "reason": "t",
        "degraded": False,
    }
    result = orch.answer("问题", use_rewrite=False, plan_override=plan)
    assert result.synthesis_route == "two_stage_fallback"
    assert result.summaries is None
    # 单发预算 = min(max_contexts=10, 无重排→不压) → 8 条全进；退回的是口径不是批块
    assert result.context_budget == 8
    assert result.map_meta == {"aborted": "x"}
    # 退回后走单发合成 prompt（不是 reduce 的摘要 prompt）
    assert len(seen) == 1
    assert "【该篇相关块" not in seen[0]["user"]


# ── 3. 落盘纪律 ───────────────────────────────────────────────────────────


def test_summaries_stay_out_of_contexts_and_citations_point_at_raw_blocks(
    monkeypatch,
):
    """`summaries` 不进 `contexts`（防两份清单合并的老错复发）：

    contexts 的正文必须仍是原始块文本——引用 [n] 与 citation_valid 的分母
    都建立在「contexts = 原始块」上（ADR-0002 的引用语义零改动）。
    """

    class _Rec:
        collection = "kb"

        def retrieve(self, question, top_n=None, **kw):
            return RetrievalOutcome(
                chunks=[
                    {
                        "doc_id": "d0",
                        "title": "文档0",
                        "page": 1,
                        "text": f"原始块{i}",
                        "block_type": "paragraph",
                        "score": 1.0,
                    }
                    for i in range(3)
                ]
            )

    summaries = [
        {
            "doc_id": "d0",
            "doc": "文档0",
            "text": "这是微摘要，不是原文",
            "source_context_idx": [1, 2, 3],
            "degraded": None,
        }
    ]
    monkeypatch.setattr(
        two_stage, "map_summaries", lambda cfg, q, ctx: (summaries, {"n_docs": 1})
    )

    def _fake_reduce(synth, q, ctxs, sums, **kw):
        # reduce 收到的是摘要串，但 Result.contexts 必须还是原始块
        assert "微摘要" in two_stage.format_summary_context(sums, ctxs)
        return "答案 [1]"

    monkeypatch.setattr(two_stage, "answer", _fake_reduce)
    synth = Synthesizer(_cfg()["llm"])
    orch = Orchestrator(_cfg(), retriever=_Rec(), synthesizer=synth)
    plan = {
        "rewritten": "q",
        "filters": None,
        "aggregate": True,
        "top_n": None,
        "reason": "t",
        "degraded": False,
    }
    result = orch.answer("问题", use_rewrite=False, plan_override=plan)
    assert result.synthesis_route == "two_stage"
    assert [c["text"] for c in result.contexts] == ["原始块0", "原始块1", "原始块2"]
    assert result.summaries[0]["text"] == "这是微摘要，不是原文"
    assert "微摘要" not in "".join(c["text"] for c in result.contexts)


def test_retrieval_only_never_pays_for_map():
    """只检索（judge 重放 / --retrieval-only）不路由两段式：map 是真金白银。"""

    class _Rec:
        collection = "kb"

        def retrieve(self, question, top_n=None, **kw):
            return RetrievalOutcome(
                chunks=[
                    {
                        "doc_id": "d0",
                        "title": "文档0",
                        "page": 1,
                        "text": "正文",
                        "block_type": "paragraph",
                    }
                ]
            )

    orch = Orchestrator(
        _cfg(), retriever=_Rec(), synthesizer=Synthesizer(_cfg()["llm"])
    )
    plan = {
        "rewritten": "q",
        "filters": None,
        "aggregate": True,
        "top_n": None,
        "reason": "t",
        "degraded": False,
    }
    result = orch.answer(
        "问题", with_answer=False, use_rewrite=False, plan_override=plan
    )
    assert result.synthesis_route == "single"
    assert result.summaries is None and result.map_meta is None


# ── 4. 重放：重建与生成逐字一致 ───────────────────────────────────────────


def test_rebuild_matches_format_byte_for_byte():
    ctx = _contexts()
    summaries = [
        {
            "doc_id": "d0",
            "doc": "文档0",
            "text": "摘要甲",
            "source_context_idx": [1, 2],
            "degraded": None,
        },
        {
            "doc_id": "d1",
            "doc": "文档1",
            "text": "正文2",
            "source_context_idx": [3],
            "degraded": "RuntimeError: 503",
        },
    ]
    from doc_rag.generate.prompts import format_context

    stored = [format_context([c]) for c in ctx]  # eval 落盘的 contexts 形态
    rebuilt = two_stage.rebuild_summary_context(summaries, stored)
    assert rebuilt == two_stage.format_summary_context(summaries, ctx)
    # 前缀形状与单发 format_context 同构：编号 + 定位一行、正文另起
    assert "[1][2] （文档0 第1页）\n摘要甲" in rebuilt
    assert "[3] （文档1 第3页）\n正文2" in rebuilt


def test_rebuild_rejects_out_of_range_idx():
    with pytest.raises(IndexError):
        two_stage.rebuild_summary_context(
            [{"source_context_idx": [99], "text": "x"}], ["[1] （a）\nb"]
        )


# ── 5. eval 层：全退化中止 + 重放守卫 ─────────────────────────────────────


class _SpyRetriever:
    collection = "kb"
    cfg: ClassVar[dict] = {"mode": "hybrid"}

    def retrieve(self, question, top_n=None, **kw):
        return RetrievalOutcome(
            chunks=[
                {
                    "doc_id": f"d{i}",
                    "title": f"文档{i}",
                    "page": i + 1,
                    "text": f"正文{i}",
                    "block_type": "paragraph",
                    "score": 1.0,
                }
                for i in range(8)
            ]
        )


def _gold(tmp_path, n: int):
    import json

    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": f"q{i}",
                        "type": "cross_doc",
                        "question": f"问题{i}",
                        "expected_answer": "正文0",
                        "source_doc_ids": ["d0"],
                        "must_contain": ["正文0"],
                    }
                    for i in range(n)
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return gold


def test_eval_aborts_when_two_stage_falls_back_for_every_item(monkeypatch, tmp_path):
    """全部退回单发的轮次不能自称 two_stage 臂——沿 rewrite 全退化中止的先例。"""
    from doc_rag.eval import runner
    from doc_rag.generate.synthesizer import Synthesizer as _Syn

    monkeypatch.setattr(
        runner,
        "_build_retriever",
        lambda c, col: (_SpyRetriever(), _Syn(c["llm"])),
    )
    monkeypatch.setattr(
        two_stage, "map_summaries", lambda cfg, q, ctx: (None, {"aborted": "x"})
    )
    # 退回后走的是单发合成：也拦掉，别让测试真打 API（先例照 rewrite 全退化那条）
    monkeypatch.setattr(
        two_stage.llm,
        "chat_timed",
        lambda cfg, p, system_prompt=None, temperature=None: (
            "答案 [1]",
            {"ms": 1.0, "cached": False, "model": "m"},
        ),
    )
    with pytest.raises(ValueError, match="两段式全部退回单发"):
        runner.evaluate(_gold(tmp_path, 2), cfg=_cfg(), aggregate=True)


def test_eval_persists_route_and_summaries(monkeypatch, tmp_path):
    """两段式条目落盘：synthesis_route / summaries / map 三件都在，供重放与对账。"""
    import json

    from doc_rag.eval import runner

    monkeypatch.setattr(
        runner,
        "_build_retriever",
        lambda c, col: (_SpyRetriever(), Synthesizer(c["llm"])),
    )
    summaries = [
        {
            "doc_id": "d0",
            "doc": "文档0",
            "text": "摘要",
            "source_context_idx": [1],
            "degraded": None,
        }
    ]
    monkeypatch.setattr(
        two_stage,
        "map_summaries",
        lambda cfg, q, ctx: (summaries, {"n_docs": 8, "n_degraded": 0}),
    )

    def _reduce(synth, q, ctxs, sums, **kw):
        return "答案 [1]"

    monkeypatch.setattr(two_stage, "answer", _reduce)
    out = runner.evaluate(_gold(tmp_path, 1), cfg=_cfg(), aggregate=True)
    item = out["items"][0]
    assert item["synthesis_route"] == "two_stage"
    assert item["summaries"] == summaries
    assert item["map"]["n_docs"] == 8
    assert out["meta"]["synthesis"]["n_two_stage"] == 1
    # contexts 仍是原始块编号串（不是摘要），引用 [1] 仍指原始块
    assert item["contexts"][0].startswith("[1] （文档0 第1页）\n正文0")
    json.dumps(out, ensure_ascii=False)  # 可序列化（落盘前提）


def test_ragas_replay_refuses_two_stage_item_without_summaries(tmp_path):
    """缺 summaries 的两段式条目拒绝判分——judge 看不到 LLM 实际所见的摘要。"""
    import json

    from doc_rag.eval import runner

    data = {
        "meta": {"retrieval": "dense+bm25+rrf[hybrid]+rewrite+rerank", "top_n": 8},
        "items": [
            {
                "id": "q1",
                "type": "cross_doc",
                "question": "问题",
                "answer": "答案 [1]",
                "synthesis_route": "two_stage",
                "contexts": ["[1] （文档0 第0页）\n正文0"],
            }
        ],
    }
    f = tmp_path / "results.json"
    f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="缺少落盘 summaries"):
        runner.ragas_from_results(f, cfg={"eval": {"ragas_sample": 0}})

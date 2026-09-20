"""Agentic policy 层的离线测试（PLAN §5.5 P1：零服务、零 API 成本）。

这里锁的是四件事，它们的共同点是「错了不会报错，只会让结论悄悄变味」：

1. 单发路径逐字不变（`Result.trace is None`、上下文预算仍是原来的口径）。
2. 停机条件由代码说了算，不由模型说了算（步数 / 预算 / 无新证据三条都不许被绕过）。
3. 判定挂掉时**停止扩展**而不是继续多查几步（坏掉的刹车不能去踩油门）。
4. trace 的形状能自证：动作、步数、预算、为什么停、开关来源逐项可查，
   形状不合规就判「不可重放」。
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from doc_rag import agent
from doc_rag.generate import llm as llm_mod
from doc_rag.orchestrator import Orchestrator
from doc_rag.retrieve.hybrid import RetrievalOutcome

COLLECTION = "kb_x"


def _chunk(n: int, doc: str = "d1", score: float | None = None) -> dict:
    return {
        "chunk_id": f"{doc}:{n}",
        "doc_id": doc,
        "title": f"文档{doc}",
        "text": f"正文{n}",
        "section_path": [],
        "page": n,
        "block_type": "paragraph",
        "doc_date": None,
        "score": score if score is not None else 1.0 - n / 1000,
    }


def _pool(n: int, doc: str = "d1") -> list[dict]:
    return [_chunk(i, doc=doc, score=2.0 - i * 0.01) for i in range(1, n + 1)]


def _cfg(**over: Any) -> dict:
    """一份「agent 已配好」的最小 cfg。默认开关落在 cross_doc / time_filter 上。"""
    agent_cfg: dict[str, Any] = {
        "enabled": True,
        "types": ["cross_doc", "time_filter"],
        "max_steps": 3,
        "judge_contexts": 6,
        "max_contexts": 12,
        "max_prompt_tokens": 100_000,
        "timeout_s": 20,
        "max_attempts": 1,
    }
    agent_cfg.update(over)
    return {
        "llm": {
            "model": "m",
            "base_url": "http://l",
            "api_key": "k",
            "headers": {"h": "1"},
        },
        "retrieval": {"max_contexts": 10, "aggregate_top_n": 25},
        "rerank": {"enabled": False, "top_n": 6},
        "qdrant": {"url": "http://x", "collection": COLLECTION},
        "embedding": {
            "base_url": "http://e",
            "api_key": "k",
            "model": "e",
            "dense_dim": 8,
        },
        "eval": {"judge": {"temperature": 0.0, "reasoning_effort": "none"}},
        "agent": agent_cfg,
    }


class FakeRetriever:
    """按查询串脚本化返回；`fresh_per_call` 用来造「每次都带来一个新块」的情形。"""

    cfg: ClassVar[dict] = {}

    def __init__(
        self, script: dict[str, list[dict]], default: list[dict] | None = None
    ):
        self.script = script
        self.default = default or []
        self.calls: list[str] = []
        self.fresh_per_call = False
        self._n = 0
        self.client = None  # 注入路径没有真 Qdrant 客户端：窗口步必须能安静跳过
        self.collection = COLLECTION

    def retrieve(self, question, top_n=None, filters=None, aggregate=False, **kw):
        self.calls.append(question)
        if self.fresh_per_call:
            self._n += 1
            return RetrievalOutcome(chunks=[_chunk(self._n, doc=f"new{self._n}")])
        return RetrievalOutcome(chunks=self.script.get(question, self.default))


class FakeClient:
    """只实现 `client.get(collection=..., points=[...], with_payload=True)`。"""

    def __init__(self, points: dict[str, dict]):
        self.points = points  # 键 = point-id
        self.requested: list[list[str]] = []

    def get(self, collection, points, with_payload=True):
        assert collection == COLLECTION
        self.requested.append(list(points))
        out = []
        for pid in points:
            payload = self.points.get(pid)
            if payload is not None:  # 真实 Qdrant 对不存在的点就是不返回，行为一致
                out.append(SimpleNamespace(id=pid, payload=payload, score=0.0))
        return out


class FakeReranker:
    """重排只重排序、不截断——它的 `context_budget` 才是上下文预算的来源。"""

    def __init__(self, cfg):
        self.context_budget = int(cfg.get("top_n") or 99)

    def rerank(self, query, chunks, top_n=None):
        assert top_n is None
        return [dict(c, rerank_score=c["score"]) for c in chunks]


def _verdict(sufficient=False, next_query="", widen=(), missing=""):
    return json.dumps(
        {
            "sufficient": sufficient,
            "missing": missing,
            "next_query": next_query,
            "widen_around": list(widen),
        },
        ensure_ascii=False,
    )


@pytest.fixture
def script_llm(monkeypatch):
    """按调用顺序喂判定输出。

    默认「多一次调用就炸」：agent 每一步都花真钱，任何一条测试要是多跑了一步判定，
    这里直接失败而不是悄悄多付一次（先例：test_rewrite_llm.py 的 `_forbidden`）。
    """
    state: dict[str, Any] = {
        "replies": [],
        "prompts": [],
        "usage": {},
        "raise_on": None,
    }

    def _timed(llm_cfg, user_prompt, system_prompt=None, temperature=None):
        if system_prompt != agent.SYSTEM_EVIDENCE:
            raise AssertionError(f"agent 只该发判定调用，收到 {system_prompt!r}")
        if state["raise_on"] is not None and len(state["prompts"]) >= state["raise_on"]:
            raise RuntimeError("503 判定服务抖动")
        assert len(state["prompts"]) < len(state["replies"]), (
            "判定调用次数超过脚本喂的份数"
        )
        reply = state["replies"][len(state["prompts"])]
        state["prompts"].append(user_prompt)
        meta = {"ms": 1.0, "cached": False, "model": "judge-m"}
        meta.update(state["usage"])
        return reply, meta

    monkeypatch.setattr(llm_mod, "chat_timed", _timed)
    return state


def _run(cfg, retriever, pool, plan=None):
    return agent.run_agent(
        cfg,
        question="关于供应商预付款做过哪些决定？",
        plan=plan
        or {
            "rewritten": "供应商预付款 决定",
            "filters": None,
            "aggregate": True,
            "top_n": 25,
            "reason": "test",
            "degraded": False,
        },
        retriever=retriever,
        pool=pool,
    )


def _orch_answer(cfg, retriever, question="聚合检索串", force_aggregate=True, **kw):
    orch = Orchestrator(cfg, retriever=retriever, synthesizer=None)
    return orch.answer(
        question,
        use_rewrite=False,
        force_aggregate=force_aggregate,
        with_answer=False,
        **kw,
    )


# ── 开关与「只扩展不替换」 ────────────────────────────────────────────────


def test_single_path_untouched_when_agent_off():
    """配置关着时 Result 与改造前同形状：trace=None，上下文预算仍是 max_contexts。"""
    cfg = _cfg()
    cfg["agent"]["enabled"] = False
    result = _orch_answer(cfg, FakeRetriever({"聚合检索串": _pool(14)}), "聚合检索串")
    assert result.trace is None
    assert len(result.contexts) == 10
    assert "agent" not in result.latency_ms


def test_config_gate_uses_predicted_type():
    """预测题型不在 types 里就不该跑——这是本期唯一的成本控制面。"""
    assert agent.predict_type({"aggregate": True, "filters": None}) == "cross_doc"
    assert (
        agent.predict_type(
            {"aggregate": True, "filters": {"doc_date": {"gte": "2026-01-01"}}}
        )
        == "time_filter"
    )
    assert agent.predict_type({"aggregate": False, "filters": None}) == "single"
    assert "single" not in agent.agent_cfg(_cfg(types=["cross_doc"]))["types"]


def test_gold_type_only_asks_for_opinion():
    """真题型只参与事后核对，不参与开关：否则五条入口的 trace 不可比。"""
    assert agent.type_matches_gold("cross_doc", "time_filter") is False
    assert agent.type_matches_gold("cross_doc", "cross_doc") is True
    assert agent.type_matches_gold("cross_doc", None) is None


def test_mode_single_overrides_enabled_config():
    """显式 single 压过配置：消融臂要能一键回到基线口径而不改配置文件。"""
    retriever = FakeRetriever({"原始问题": _pool(3)})
    result = _orch_answer(_cfg(), retriever, "原始问题", mode="single")
    assert result.trace is None
    assert retriever.calls == ["原始问题"]


def test_agent_records_why_the_switch_was_flipped(script_llm):
    script_llm["replies"] = [_verdict(sufficient=True)]
    result = _orch_answer(
        _cfg(),
        FakeRetriever({"聚合检索串": _pool(3)}),
        "聚合检索串",
        question_type="cross_doc",
    )
    trace = result.trace
    assert trace is not None
    assert trace["mode_explicit"] is False  # 这一次是配置开的
    assert trace["type_predicted"] == "cross_doc"
    assert trace["type_matches_gold"] is True
    assert trace["stop_reason"] in agent.STOP_REASONS


def test_explicit_mode_overrides_types_but_not_empty_retrieval(script_llm):
    """显式 mode="agent" 可以无视题型开关，但空检索永远不进扩展步。"""
    script_llm["replies"] = [_verdict(sufficient=True)]
    result = _orch_answer(
        _cfg(types=["cross_doc"]),
        FakeRetriever({"随便一问": _pool(2)}),
        "随便一问",
        mode="agent",
    )
    assert result.trace is not None
    assert result.trace["mode_explicit"] is True

    empty = _orch_answer(
        _cfg(), FakeRetriever({}), "空库问题", mode="agent", force_aggregate=False
    )
    assert empty.trace is None  # 没有块可判，一次判定都不该付
    assert empty.contexts == []


# ── 停机条件 ─────────────────────────────────────────────────────────────


def test_sufficient_verdict_stops_without_extra_search(script_llm):
    script_llm["replies"] = [_verdict(sufficient=True)]
    pool = _pool(3)
    retriever = FakeRetriever({})
    union, trace = _run(_cfg(), retriever, pool)
    assert trace["stop_reason"] == "sufficient"
    assert [s["action"] for s in trace["steps"]] == ["check_evidence"]
    assert retriever.calls == []
    assert union == pool  # 并集不改动第一步的结果
    assert trace["sub_queries"] == []


def test_second_search_unions_and_dedupes_by_chunk_id(script_llm):
    script_llm["replies"] = [_verdict(next_query="子查询二"), _verdict(sufficient=True)]
    pool = _pool(3, doc="d1")
    retriever = FakeRetriever(
        {"子查询二": [_chunk(1, doc="d2"), _chunk(1, doc="d1")]}  # 第二条与已有块同 id
    )
    union, trace = _run(_cfg(), retriever, pool)
    assert retriever.calls == ["子查询二"]
    assert [c["chunk_id"] for c in union].count("d1:1") == 1
    assert {c["doc_id"] for c in union} == {"d1", "d2"}
    search = next(s for s in trace["steps"] if s["action"] == "search")
    assert search["args"]["sub_query"] == "子查询二"
    assert search["n_retrieved"] == 2
    assert search["new_doc_ids_added"] == 1
    assert trace["stop_reason"] == "sufficient"


def test_no_new_evidence_stops_rather_than_spinning(script_llm):
    """返回的全是已有块时不许再兜第二轮——原地打转是最贵的那种浪费。"""
    script_llm["replies"] = [_verdict(next_query="同义反复")] * 3
    retriever = FakeRetriever({"同义反复": _pool(3, doc="d1")})
    _, trace = _run(_cfg(), retriever, _pool(3, doc="d1"))
    assert trace["stop_reason"] == "no_new_evidence"
    assert retriever.calls == ["同义反复"]
    assert len([s for s in trace["steps"] if s["action"] == "check_evidence"]) == 1


def test_max_steps_caps_judgement_calls_and_searches(script_llm):
    script_llm["replies"] = [_verdict(next_query="下一跳")] * 3
    retriever = FakeRetriever({})
    retriever.fresh_per_call = True
    _, trace = _run(_cfg(max_steps=3), retriever, _pool(2))
    checks = [s for s in trace["steps"] if s["action"] == "check_evidence"]
    searches = [s for s in trace["steps"] if s["action"] == "search"]
    assert len(checks) == 3
    assert len(searches) == 2  # 最后一轮判定之后已经没有「再检索」的预算
    assert trace["stop_reason"] == "steps_exhausted"


def test_max_steps_is_clamped_to_three(script_llm):
    """配置写 9 也只跑 3：步数上限是边界不是偏好（PLAN §2）。"""
    cfg = _cfg(max_steps=9)
    assert agent.agent_cfg(cfg)["max_steps"] == 3
    script_llm["replies"] = [_verdict(next_query="下一跳")] * 3
    retriever = FakeRetriever({})
    retriever.fresh_per_call = True
    _, trace = _run(cfg, retriever, _pool(2))
    assert len([s for s in trace["steps"] if s["action"] == "check_evidence"]) == 3


def test_token_budget_stops_before_the_next_search(script_llm):
    script_llm["replies"] = [_verdict(next_query="下一跳")]
    script_llm["usage"] = {"prompt_tokens": 5000, "completion_tokens": 3}
    retriever = FakeRetriever({})
    retriever.fresh_per_call = True
    _, trace = _run(_cfg(max_prompt_tokens=4000), retriever, _pool(2))
    assert trace["stop_reason"] == "token_budget"
    assert retriever.calls == []  # 预算已被这轮判定吃掉，就不该再付一次检索
    assert trace["budget_used"]["prompt_tokens"] == 5000


def test_judge_failure_stops_instead_of_extending(script_llm):
    """判定抖动 → 按「证据已足够」停下，且一次检索都不许多做。

    退化成继续扩展等于让坏掉的刹车去踩油门：花真钱，依据还是空的。
    """
    script_llm["raise_on"] = 0
    pool = _pool(3)
    retriever = FakeRetriever({})
    union, trace = _run(_cfg(), retriever, pool)
    assert trace["stop_reason"] == "judge_degraded"
    assert trace["steps"][0]["degraded"].startswith("call_failed")
    assert trace["steps"][0]["decision"] == "degraded"
    assert retriever.calls == []
    assert union == pool


def test_unparsable_verdict_also_stops(script_llm):
    script_llm["replies"] = ["判定模型今天写散文，不给 JSON"]
    retriever = FakeRetriever({})
    _, trace = _run(_cfg(), retriever, _pool(2))
    assert trace["stop_reason"] == "judge_degraded"
    assert trace["steps"][0]["degraded"] == "bad_json"
    assert retriever.calls == []


def test_missing_llm_config_degrades_without_calling_anything(script_llm):
    """没配 key 时退化成「停」，而且不能去碰 chat_timed。"""
    cfg = _cfg()
    cfg["llm"] = {"model": "", "base_url": "http://l", "api_key": ""}
    _, trace = _run(cfg, FakeRetriever({}), _pool(2))
    assert trace["steps"][0]["degraded"] == "no_llm"
    assert script_llm["prompts"] == []


# ── read_window ──────────────────────────────────────────────────────────


def test_neighbour_ids_only_walks_the_ordinal_and_stops_at_one():
    assert agent.neighbour_ids("ab:7") == ["ab:6", "ab:8"]
    assert agent.neighbour_ids("ab:1") == ["ab:2"]  # 序号从 1 起，没有 0 号块
    assert agent.neighbour_ids("没有序号") == []


def test_point_id_matches_the_ingest_side():
    """point-id 的算法只有一个来源：indexer 的 uuid5(NAMESPACE_URL, chunk_id)。"""
    assert agent.chunk_point_id("d1:2") == str(uuid.uuid5(uuid.NAMESPACE_URL, "d1:2"))


def test_read_window_fetches_only_missing_neighbours():
    have = [_chunk(2, doc="d1")]
    wanted = {
        agent.chunk_point_id(cid): {
            "chunk_id": cid,
            "doc_id": "d1",
            "title": "文档d1",
            "text": f"正文{cid.rsplit(':', 1)[-1]}",
            "page": 1,
            "block_type": "paragraph",
        }
        for cid in ("d1:1", "d1:3")
    }
    client = FakeClient(wanted)
    out = agent.read_window(client, COLLECTION, have)
    assert {c["chunk_id"] for c in out} == {"d1:1", "d1:3"}
    assert all(c["from_window"] and c["score"] == 0.0 for c in out)
    assert "d1:2" not in {c["chunk_id"] for c in out}  # 已有的块不再取回来


def test_read_window_without_chunk_id_or_client_is_quiet():
    """注入路径没有真客户端、旧结果文件没有 chunk_id —— 跳过而不是抛错。"""
    assert agent.read_window(None, COLLECTION, [{"doc_id": "d1"}]) == []
    assert agent.read_window(FakeClient({}), "", [_chunk(1)]) == []


def test_window_step_only_runs_when_judge_asks_for_it(script_llm):
    script_llm["replies"] = [_verdict(sufficient=True, widen=[1])]
    client = FakeClient({agent.chunk_point_id("d1:2"): {"chunk_id": "d1:2"}})
    retriever = FakeRetriever({})
    retriever.client = client
    _, trace = _run(_cfg(), retriever, _pool(2))
    assert [s["action"] for s in trace["steps"]] == ["check_evidence"]
    assert trace["stop_reason"] == "sufficient"
    assert client.requested == []  # 证据够了就不必补窗口


def test_window_step_runs_and_is_recorded(script_llm):
    script_llm["replies"] = [_verdict(widen=[1]), _verdict(sufficient=True)]
    pool = [_chunk(2, doc="d1", score=2.0)]
    client = FakeClient(
        {
            agent.chunk_point_id("d1:3"): {
                "chunk_id": "d1:3",
                "doc_id": "d1",
                "title": "文档d1",
                "text": "正文3",
                "page": 3,
                "block_type": "table",
            }
        }
    )
    retriever = FakeRetriever({})
    retriever.client = client
    union, trace = _run(_cfg(), retriever, pool)
    assert [s["action"] for s in trace["steps"]] == ["check_evidence", "read_window"]
    assert union[1]["chunk_id"] == "d1:3"
    window = trace["steps"][1]
    assert window["decision"] == "extended"
    assert window["new_doc_ids_added"] == 0  # 邻居是同一篇文档，不是新文档
    assert trace["stop_reason"] == "no_next_query"


def test_widen_indices_outside_the_view_are_dropped(script_llm):
    """判定只能指它真看到的编号；越界一律丢弃，不能借道取别的块。"""
    script_llm["replies"] = [_verdict(widen=[99]), _verdict(sufficient=True)]
    client = FakeClient({agent.chunk_point_id("d1:2"): {"chunk_id": "d1:2"}})
    retriever = FakeRetriever({})
    retriever.client = client
    _, trace = _run(_cfg(), retriever, _pool(3))
    window = next(s for s in trace["steps"] if s["action"] == "read_window")
    assert window["decision"] == "nothing_new"
    assert client.requested == []  # 一次点取都没发生


def test_window_blocks_land_after_scored_ones(script_llm):
    """窗口块 score=0：并集重排时必须沉底，不许挤掉榜首。"""
    script_llm["replies"] = [_verdict(widen=[1]), _verdict(sufficient=True)]
    pool = [_chunk(2, doc="d1", score=2.0)]
    client = FakeClient(
        {
            agent.chunk_point_id(f"d1:{n}"): {
                "chunk_id": f"d1:{n}",
                "doc_id": "d1",
                "text": f"正文{n}",
                "page": n,
                "block_type": "paragraph",
            }
            for n in (1, 3)
        }
    )
    retriever = FakeRetriever({"再查一次": [_chunk(7, doc="d7", score=1.5)]})
    retriever.client = client
    union, _ = _run(_cfg(), retriever, pool)
    assert union[0]["chunk_id"] == "d1:2"
    assert union[-1]["from_window"] is True


# ── 上下文预算与延迟档位 ──────────────────────────────────────────────────


def test_agent_context_budget_replaces_the_single_shot_one(monkeypatch, script_llm):
    """agent 臂的进 LLM 块数由 `agent.max_contexts` 说了算，不被 rerank.top_n 砍回 6。

    砍回去就等于「多查几次」的收益在最后一米被自己的配置扔掉——而这正是本层存在的
    理由（PLAN §5.3 的覆盖率封顶）。代价由答案轨的等长护栏兜，不在这里兜。
    """
    import doc_rag.retrieve.rerank as rerank_mod

    monkeypatch.setattr(rerank_mod, "Reranker", FakeReranker)
    script_llm["replies"] = [_verdict(sufficient=True)]
    cfg = _cfg(max_contexts=12)
    cfg["rerank"] = {"enabled": True, "top_n": 6}
    result = _orch_answer(cfg, FakeRetriever({"聚合检索串": _pool(14)}), "聚合检索串")
    assert result.trace is not None
    assert len(result.contexts) == 12
    assert len(result.retrieved) == 14  # 未截断清单仍是全量：检索指标的分母
    assert [c["no"] for c in result.citations] == list(range(1, 13))


def test_single_shot_context_budget_still_follows_rerank(monkeypatch):
    """同一份配置下把 agent 关掉，预算口径必须回到 min(max_contexts, rerank.top_n)。"""
    import doc_rag.retrieve.rerank as rerank_mod

    monkeypatch.setattr(rerank_mod, "Reranker", FakeReranker)
    cfg = _cfg()
    cfg["agent"]["enabled"] = False
    cfg["rerank"] = {"enabled": True, "top_n": 6}
    result = _orch_answer(cfg, FakeRetriever({"聚合检索串": _pool(14)}), "聚合检索串")
    assert result.trace is None
    assert len(result.contexts) == 6


def test_agent_latency_is_its_own_stage(script_llm):
    """`retrieval_total` 的口径是「不含 LLM」，判定调用是 LLM —— 所以单列一档。"""
    script_llm["replies"] = [_verdict(sufficient=True)]
    result = _orch_answer(_cfg(), FakeRetriever({"q": _pool(2)}), "q")
    assert set(result.latency_ms) == {
        "rewrite",
        "retrieve",
        "rerank",
        "retrieval_total",
        "synthesize",
        "synth_cached",
        "total",
        "agent",
    }
    assert result.latency_ms["agent"] >= 0.0
    assert result.latency_ms["retrieval_total"] <= result.latency_ms["total"]


# ── trace 形状与重放准入 ──────────────────────────────────────────────────


def test_produced_trace_is_replay_admissible(script_llm):
    script_llm["replies"] = [_verdict(next_query="下一跳"), _verdict(sufficient=True)]
    retriever = FakeRetriever({"下一跳": [_chunk(9, doc="d9")]})
    _, trace = _run(_cfg(), retriever, _pool(2))
    assert agent.trace_from_dict(trace) is trace
    assert json.dumps(trace, ensure_ascii=False)  # 必须能整份落进结果文件


@pytest.mark.parametrize(
    "broken",
    [
        None,
        "sufficient",
        {},
        {
            "stop_reason": "模型心情好",
            "steps": [{"action": "search"}],
            "budget_used": {},
        },
        {"stop_reason": "sufficient", "steps": [], "budget_used": {}},
        {
            "stop_reason": "sufficient",
            "steps": [{"action": "browse_web"}],
            "budget_used": {},
        },
        {"stop_reason": "sufficient", "steps": [{"action": "search"}]},
    ],
)
def test_malformed_trace_is_refused(broken):
    """形状不合就判不可重放——宁可拒绝，不要拿另一批证据去判旧答案。"""
    assert agent.trace_from_dict(broken) is None


def test_every_step_carries_its_own_cost(script_llm):
    """步级归因是成本-质量前沿的数据源（PLAN §5：没有前沿图，agentic 就是玄学）。"""
    script_llm["replies"] = [_verdict(next_query="跳"), _verdict(sufficient=True)]
    script_llm["usage"] = {"prompt_tokens": 120, "completion_tokens": 7}
    retriever = FakeRetriever({"跳": [_chunk(3, doc="dz")]})
    _, trace = _run(_cfg(), retriever, _pool(2))
    for step in trace["steps"]:
        assert isinstance(step["n"], int)
        assert step["action"] in agent.ACTIONS
        assert "ms" in step and "prompt_tokens" in step
    assert [s["n"] for s in trace["steps"]] == list(range(1, len(trace["steps"]) + 1))
    assert trace["budget_used"]["calls"] == 2
    assert trace["budget_used"]["prompt_tokens"] == 240
    assert trace["n_docs_union"] == 2


def test_judge_prompt_sees_only_the_capped_contexts(script_llm):
    """判定的输入块数受 `judge_contexts` 限：它每轮都跑，不设上限就是每轮付一次长 prompt。"""
    script_llm["replies"] = [_verdict(sufficient=True)]
    _run(_cfg(judge_contexts=4), FakeRetriever({}), _pool(9))
    prompt = script_llm["prompts"][0]
    assert "正文4" in prompt
    assert "正文5" not in prompt


# ── 判定调用的 endpoint 装配 ─────────────────────────────────────────────


def test_endpoint_cfg_borrows_judge_but_not_its_timeout():
    """走 judge 的装配（能不同源、换家不带生成侧 key），但预算按 `agent.*` 压下来。"""
    cfg = _cfg(timeout_s=7.5, max_attempts=1)
    cfg["eval"] = {"judge": {"timeout_s": 180, "max_attempts": 2, "model": "j"}}
    built = agent.endpoint_cfg(cfg)
    assert built["timeout_s"] == 7.5
    assert built["max_attempts"] == 1
    assert built["model"] == "j"
    assert built["reasoning_effort"] == "none"


def test_endpoint_cfg_does_not_leak_generation_key_cross_vendor():
    """把判定指到另一家时绝不顺手带生成侧的 key（与 eval/judge.py 同一条规则）。"""
    cfg = _cfg()
    cfg["eval"] = {"judge": {"base_url": "http://other/v1"}}
    built = agent.endpoint_cfg(cfg)
    assert built["api_key"] == ""
    assert "headers" not in built

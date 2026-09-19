"""四条问答路径必须走同一条管线（W1 的回归护栏）。

背景：这条链此前在 6 处内联重复，并且已经漂移成——FastAPI 两个端点漏掉重排、
`/query` 还漏掉 `max_contexts` 截断；README 的头条数字测的是「改写 + 重排 + 截断」，
而唯一对外交付的 `/query` 跑的是另一条配置。

这里用「送进 LLM 的 prompt 必须逐字相同」来锁死：任何一条路径少拼一个环节，
prompt 就会与其他路径不一致。另两条护栏是重排调用次数与上下文块数——它们能在
prompt 相同的情况下仍然悄悄漂移（比如顺序不同但内容相同）。
"""

from __future__ import annotations

import json
import re
from typing import ClassVar

import pytest

from doc_rag.api import demo as demo_mod
from doc_rag.api import main as api_main
from doc_rag.eval import runner
from doc_rag.generate import llm as llm_mod
from doc_rag.generate.synthesizer import Synthesizer
from doc_rag.orchestrator import Orchestrator
from doc_rag.retrieve.hybrid import RetrievalOutcome

QUESTION = "客服系统升级的预算是多少？"
N_RETRIEVED = 12
MAX_CONTEXTS = 10
_CTX_HEAD = re.compile(r"^\[\d+\] ", re.MULTILINE)


def _cfg():
    return {
        "llm": {"model": "m", "base_url": "http://l", "api_key": "k"},
        "retrieval": {"mode": "hybrid", "max_contexts": MAX_CONTEXTS},
        "rerank": {
            "enabled": True,
            "base_url": "http://r",
            "api_key": "k",
            "model": "rr",
            # 故意与 max_contexts 相等：parity 断言的是「10 块上下文」这件事，
            # 谁把上下文砍到 6 是下面 test_rerank_budget_... 那条单独钉的。
            "top_n": MAX_CONTEXTS,
        },
        "qdrant": {"url": "http://x", "collection": "kb_default"},
        "api": {
            "auth_token": "t",
            "allowed_collections": ["kb_default", "kb_a", "kb_b"],
        },
        "embedding": {
            "base_url": "http://e",
            "api_key": "k",
            "model": "e",
            "dense_dim": 1024,
        },
    }


class SpyRetriever:
    """固定返回同一批块，忽略 top_n 与过滤：让断言聚焦在管线拼装顺序上。

    `collections` 记录每次构建时绑定的 collection，用于验证 kb 不跨请求残留。
    """

    collections: ClassVar[list[str | None]] = []

    def __init__(self, client=None, embedder=None, collection=None, retrieval_cfg=None):
        self.collection = collection
        self.cfg = retrieval_cfg or {}
        SpyRetriever.collections.append(collection)

    def retrieve(self, question, **kw):
        return RetrievalOutcome(
            chunks=[
                {
                    "doc_id": f"d{i}",
                    "title": f"文档{i}",
                    "page": i,
                    "text": f"正文{i}",
                    "block_type": "paragraph",
                    "score": 1.0 - i / 100,
                }
                for i in range(N_RETRIEVED)
            ]
        )


class SpyReranker:
    calls = 0

    def __init__(self, cfg):
        # 真实 Reranker 的 context_budget 来自 cfg，替身必须一样：
        # 重排不再截断清单，它对下游的唯一影响就是这个上下文预算。
        self.context_budget = int(cfg.get("top_n") or 99)

    def rerank(self, query, chunks, top_n=None):
        SpyReranker.calls += 1
        assert top_n is None, "管线不该再让重排截断清单"
        return [dict(c, rerank_score=c["score"]) for c in chunks]


@pytest.fixture
def prompt_spy(monkeypatch):
    """拦在 llm 层：记录真实 Synthesizer 拼出来的 prompt，而不是替身的答案。

    改写阶段也走同一个 chat_timed，因此必须按 system_prompt 分流——只记答案侧的
    调用（否则「四条路径各一次合成」会数成八次），且四条路径共用同一份改写结果，
    parity 比较才不被改写抖动污染。
    """
    from doc_rag.retrieve.rewrite_llm import SYSTEM_REWRITE

    seen: list[dict] = []

    def _rewrite_reply(user_prompt: str) -> str:
        return '{"rewritten": "改写后的检索串", "aggregate": false, "year": null, "reason": "test"}'

    def _timed(llm_cfg, user_prompt, system_prompt=None, temperature=None):
        if system_prompt == SYSTEM_REWRITE:
            return _rewrite_reply(user_prompt), {
                "ms": 0.5,
                "cached": False,
                "model": "m",
            }
        seen.append({"user": user_prompt, "system": system_prompt})
        return "答案 [1]", {"ms": 1.0, "cached": False, "model": "m"}

    def _stream(llm_cfg, user_prompt, system_prompt=None, temperature=None):
        if system_prompt == SYSTEM_REWRITE:
            return iter([_rewrite_reply(user_prompt)]), {
                "ms": 0.5,
                "cached": False,
                "model": "m",
            }
        seen.append({"user": user_prompt, "system": system_prompt})
        return iter(["答案 ", "[1]"]), {"ms": 1.0, "cached": False, "model": "m"}

    monkeypatch.setattr(llm_mod, "chat_timed", _timed)
    monkeypatch.setattr(llm_mod, "chat_stream", _stream)
    return seen


@pytest.fixture
def wired(monkeypatch, prompt_spy):
    import doc_rag.orchestrator as orch_mod
    import doc_rag.retrieve.rerank as rerank_mod

    monkeypatch.setattr(orch_mod, "HybridRetriever", SpyRetriever)
    monkeypatch.setattr(orch_mod, "QdrantClient", lambda **kw: object())
    monkeypatch.setattr(orch_mod, "Embedder", lambda cfg: object())
    monkeypatch.setattr(rerank_mod, "Reranker", SpyReranker)
    SpyReranker.calls = 0
    SpyRetriever.collections = []
    return prompt_spy


def _drive_all_four(monkeypatch, tmp_path, cfg):
    """跑 /query、/query/stream、演示页、eval 单条，各自触发一次合成。"""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api_main, "_orchestrator", lambda: Orchestrator(cfg))
    client = TestClient(api_main.app, headers={"Authorization": "Bearer t"})

    client.post("/query", json={"question": QUESTION})
    client.post("/query/stream", json={"question": QUESTION})
    list(demo_mod.render_answer(Orchestrator(cfg), QUESTION))

    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "q1",
                        "type": "fact",
                        "question": QUESTION,
                        "expected_answer": "正文0",
                        "source_doc_ids": ["d0"],
                        "must_contain": ["正文0"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "_build_retriever",
        lambda c, col: (SpyRetriever(), Synthesizer(c["llm"])),
    )
    runner.evaluate(gold, cfg=cfg, use_rewrite=True, use_rerank=True)


def test_all_four_paths_send_identical_prompt(monkeypatch, tmp_path, wired):
    cfg = _cfg()
    _drive_all_four(monkeypatch, tmp_path, cfg)

    assert len(wired) == 4, "四条路径各自应且只应触发一次合成调用"
    users = [p["user"] for p in wired]
    assert users[0] == users[1] == users[2] == users[3]
    assert len({p["system"] for p in wired}) == 1
    # 重排：cfg.rerank.enabled 为真时没有任何一条路径可以跳过它
    assert SpyReranker.calls == 4
    # 截断：max_contexts 对四条路径同样生效
    for user in users:
        assert len(_CTX_HEAD.findall(user)) == MAX_CONTEXTS


def test_kb_param_does_not_stick_across_requests(monkeypatch, wired):
    """回归：改造前端点用 `retriever.collection = body.kb` 改共享单例，并发会串库。"""
    from fastapi.testclient import TestClient

    cfg = _cfg()
    monkeypatch.setattr(api_main, "_orchestrator", lambda: Orchestrator(cfg))
    client = TestClient(api_main.app, headers={"Authorization": "Bearer t"})

    for kb in ("kb_a", "kb_b", None):
        client.post("/query", json={"question": QUESTION, "kb": kb})

    assert SpyRetriever.collections == ["kb_a", "kb_b", "kb_default"]


def test_eval_ablation_switch_overrides_rerank_config(monkeypatch, tmp_path, wired):
    """eval 的消融臂必须能绕过 `rerank.enabled` 强制关重排（旧 `_maybe_rerank` 口径）。"""
    cfg = _cfg()
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "q1",
                        "type": "fact",
                        "question": QUESTION,
                        "expected_answer": "x",
                        "source_doc_ids": ["d0"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "_build_retriever",
        lambda c, col: (SpyRetriever(), Synthesizer(c["llm"])),
    )
    runner.evaluate(gold, cfg=cfg, use_rerank=False)
    assert SpyReranker.calls == 0


def test_retrieved_stays_uncapped_while_contexts_is_capped(wired):
    """口径分离：检索指标在未截断清单上算，LLM 只看截断后那份。

    合并两者会静默改掉 Hit@k / MRR / nDCG / 文档覆盖率的分母。
    重排也在这条线上：它只重排序，不砍清单——砍由上下文预算负责。
    """
    result = Orchestrator(_cfg()).answer(QUESTION)
    assert len(result.retrieved) == N_RETRIEVED  # 重排后仍是全量 12 条
    assert len(result.contexts) == MAX_CONTEXTS
    assert result.context_budget == MAX_CONTEXTS
    assert [c["no"] for c in result.citations] == list(range(1, MAX_CONTEXTS + 1))
    assert all(c["block_type"] for c in result.citations)


def test_rerank_budget_caps_contexts_not_the_retrieved_list(wired):
    """`rerank.top_n` 是上下文预算，不是清单长度。

    它一旦兼做截断，「有重排」臂的检索指标就在 6 条清单上算、对照臂在 8 条上算，
    nDCG@8 与覆盖率的差就变成清单长度的函数（消融 #3 当时正是这样）。
    """
    cfg = _cfg()
    cfg["rerank"] = dict(cfg["rerank"], top_n=6)
    result = Orchestrator(cfg).answer(QUESTION)
    assert len(result.retrieved) == N_RETRIEVED
    assert len(result.contexts) == 6
    assert [c["no"] for c in result.contexts] == list(range(1, 7))


def test_rerank_failure_falls_back_and_is_visible(monkeypatch, wired):
    """重排挂了要退回融合顺序，但失败必须记在结果上，不能静默冒充「重排过」。"""

    class BrokenReranker:
        def __init__(self, cfg):
            pass

        def rerank(self, query, chunks, top_n=None):
            raise RuntimeError("429 too many requests")

    import doc_rag.retrieve.rerank as rerank_mod

    monkeypatch.setattr(rerank_mod, "Reranker", BrokenReranker)
    result = Orchestrator(_cfg()).answer(QUESTION)
    assert "429" in result.rerank_error
    assert len(result.retrieved) == N_RETRIEVED  # 退回未重排的融合顺序


def test_stream_events_precede_the_final_result(wired):
    """事件序列 = rewrite / delta* / citations / done，done 带完整计时（SSE 契约）。"""
    events = list(Orchestrator(_cfg()).answer_stream(QUESTION))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "rewrite"
    assert kinds[-1] == "done"
    assert kinds[-2] == "citations"
    assert set(kinds[1:-2]) == {"delta"}
    result = events[-1]["result"]
    assert result.answer == "答案 [1]"
    assert set(result.latency_ms) == {
        "rewrite",
        "retrieve",
        "rerank",
        "retrieval_total",
        "synthesize",
        "synth_cached",
        "total",
    }


def test_stop_on_empty_skips_synthesis(wired):
    """CLI 的省费护栏：空库不该花一次必然无据的合成。"""
    from types import SimpleNamespace

    cfg = _cfg()
    orch = Orchestrator(
        cfg,
        retriever=SimpleNamespace(
            retrieve=lambda q, **kw: RetrievalOutcome(chunks=[]), cfg={}
        ),
        synthesizer=Synthesizer(cfg["llm"]),
    )
    result = orch.answer(QUESTION, stop_on_empty=True)
    assert result.answer == ""
    assert wired == []


def _write_gold(tmp_path, n: int):
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": f"q{i}",
                        "type": "fact",
                        "question": f"{QUESTION}{i}",
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


def _patch_eval_pipeline(monkeypatch, cfg):
    monkeypatch.setattr(
        runner,
        "_build_retriever",
        lambda c, col: (SpyRetriever(), Synthesizer(c["llm"])),
    )


def _flaky_reranker(monkeypatch, fail_after: int):
    """前 fail_after 次成功，之后一律抛错——用来构造「部分失败」与「全失败」。"""
    import doc_rag.retrieve.rerank as rerank_mod

    state = {"calls": 0}

    class FlakyReranker:
        def __init__(self, cfg):
            self.context_budget = int(cfg.get("top_n") or 99)

        def rerank(self, query, chunks, top_n=None):
            state["calls"] += 1
            if state["calls"] > fail_after:
                raise RuntimeError("429 too many requests")
            return [dict(c, rerank_score=0.5) for c in chunks]

    monkeypatch.setattr(rerank_mod, "Reranker", FlakyReranker)
    return state


def test_eval_aborts_when_rerank_fails_for_every_item(monkeypatch, tmp_path, wired):
    """全量重排失败必须中止：那是「无重排」的结果，标成 +rerank 就是度量伪影。"""
    cfg = _cfg()
    _patch_eval_pipeline(monkeypatch, cfg)
    _flaky_reranker(monkeypatch, fail_after=0)
    with pytest.raises(ValueError, match="全部重排失败"):
        runner.evaluate(_write_gold(tmp_path, 2), cfg=cfg, use_rerank=True)


def test_eval_meta_stops_claiming_rerank_on_partial_failure(
    monkeypatch, tmp_path, wired
):
    """部分失败：meta 不再自称 +rerank，且失败条数落在 meta 上可追。"""
    cfg = _cfg()
    _patch_eval_pipeline(monkeypatch, cfg)
    _flaky_reranker(monkeypatch, fail_after=1)

    out = runner.evaluate(_write_gold(tmp_path, 2), cfg=cfg, use_rerank=True)

    assert "+rerank" not in out["meta"]["retrieval"]
    assert out["meta"]["rerank_failed"] == 1
    assert sum(1 for r in out["items"] if r["rerank_error"]) == 1


def test_over_refusal_must_have_actually_withheld_the_answer(monkeypatch, tmp_path):
    """被 prompt 鼓励写出的「文档没记载…」hedge 不算过度拒答。

    实测动机：2026-09-19 全量 72 条那轮，被标记的 26 条**全部**答对了关键内容，
    `over_refusal_rate` 0.406 是 100% 假阳性。一个恒真的指标比一个偏高的更糟——
    它会让人去修一个不存在的问题。
    """
    cfg = _cfg()
    _patch_eval_pipeline(monkeypatch, cfg)

    # 按调用顺序给两份答案（条目问题串是 `f"{QUESTION}{i}"`，别拿它做分流键）
    replies = iter(
        [
            # 带拒答措辞「未记载」、但同时把上下文里的关键信息给了出来——正确答案的常见写法
            "文档未记载单独的决议编号，但记录为 正文0 [1]。",
            # 上下文里就有 正文0，答案却直接拒了——这才是过度拒答
            "根据现有文档无法回答。",
        ]
    )

    def _timed(llm_cfg, user_prompt, system_prompt=None, temperature=None):
        return next(replies), {"ms": 1.0, "cached": False, "model": "m"}

    monkeypatch.setattr(llm_mod, "chat_timed", _timed)
    out = runner.evaluate(_write_gold(tmp_path, 2), cfg=cfg, use_rewrite=False)

    assert {r["id"]: r["over_refusal"] for r in out["items"]} == {
        "q0": False,
        "q1": True,
    }
    assert out["summary"]["over_refusal_rate"] == 0.5
    assert out["summary"]["contains_acc"] == 0.5


def _flaky_rewrite(monkeypatch, fail_after: int):
    """前 fail_after 次改写成功、之后一律抛错：构造「部分退化」与「全退化」。"""
    from doc_rag.retrieve.rewrite_llm import SYSTEM_REWRITE

    ok = '{"rewritten":"改写后的检索串","aggregate":false,"year":null,"reason":"r"}'
    state = {"calls": 0}

    def _timed(llm_cfg, user_prompt, system_prompt=None, temperature=None):
        if system_prompt != SYSTEM_REWRITE:
            return "答案 [1]", {"ms": 1.0, "cached": False, "model": "m"}
        state["calls"] += 1
        if state["calls"] > fail_after:
            raise RuntimeError("503 改写服务抖动")
        return ok, {"ms": 0.5, "cached": False, "model": "m"}

    monkeypatch.setattr(llm_mod, "chat_timed", _timed)
    return state


def test_eval_aborts_when_rewrite_degrades_for_every_item(monkeypatch, tmp_path):
    """全量改写退化必须中止：那是「无改写」的结果，标成 +rewrite 就是度量伪影。"""
    cfg = _cfg()
    _patch_eval_pipeline(monkeypatch, cfg)
    _flaky_rewrite(monkeypatch, fail_after=0)
    with pytest.raises(ValueError, match="改写全部退化"):
        runner.evaluate(_write_gold(tmp_path, 2), cfg=cfg, use_rewrite=True)


def test_eval_reports_which_model_rewrote_and_how_many_degraded(monkeypatch, tmp_path):
    """部分退化不中止（逐条可追），但必须计数——且 `+rewrite` 标记要留着：
    重放靠它决定「读记录的改写串」，抹掉就等于让重放拿原始问题去配旧答案。"""
    cfg = _cfg()
    cfg["rewrite"] = {"model": "small-rewrite"}
    _patch_eval_pipeline(monkeypatch, cfg)
    _flaky_rewrite(monkeypatch, fail_after=1)

    out = runner.evaluate(_write_gold(tmp_path, 2), cfg=cfg, use_rewrite=True)

    assert "+rewrite" in out["meta"]["retrieval"]
    assert out["meta"]["rewrite_degraded"] == 1
    # 改写与合成可以不同源：只记 llm_model 会把改写的归属记错
    assert out["meta"]["rewrite_model"] == "small-rewrite"
    assert out["meta"]["llm_model"] == "m"
    degraded = [r for r in out["items"] if r["rewrite_degraded"]]
    assert len(degraded) == 1
    assert degraded[0]["rewritten"] == degraded[0]["question"]


def test_retrieval_budget_precedence(wired):
    """三条预算规则：生产听改写建议、显式 top_n 压住它（消融口径）、honor 标志反过来。

    「聚合题放宽到 aggregate_top_n」是生产行为；eval 一直显式传 top_n=8，于是
    这条行为从来没被任何评估臂量过。顺序写反一次，生产就会悄悄退回 fusion_limit。
    """
    seen: list[int | None] = []

    class _Rec:
        collection = "kb_default"
        cfg: ClassVar[dict] = {}

        def retrieve(self, question, top_n=None, **kw):
            seen.append(top_n)
            return RetrievalOutcome(
                chunks=[
                    {
                        "doc_id": "d1",
                        "title": "文档",
                        "page": 1,
                        "text": "正文",
                        "block_type": "paragraph",
                    }
                ]
            )

    plan = {
        "rewritten": "机房巡检",
        "filters": None,
        "aggregate": True,
        "top_n": 25,  # 改写给聚合题的预算
        "reason": "test",
        "degraded": False,
    }
    orch = Orchestrator(_cfg(), retriever=_Rec(), synthesizer=None)
    orch.answer("问题", with_answer=False, use_rewrite=False, plan_override=plan)
    assert seen == [25]  # 生产路径：不传 top_n → 听改写
    orch.answer(
        "问题", with_answer=False, use_rewrite=False, plan_override=plan, top_n=8
    )
    assert seen == [25, 8]  # 消融路径：显式预算优先
    orch.answer(
        "问题",
        with_answer=False,
        use_rewrite=False,
        plan_override=plan,
        top_n=8,
        honor_rewrite_budget=True,
    )
    assert seen == [25, 8, 25]  # 生产口径的评估臂：改写的建议压回显式预算

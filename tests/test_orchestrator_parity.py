"""五条问答入口必须走同一条管线（W1 的回归护栏）。

背景：这条链此前在 6 处内联重复，并且已经漂移成——FastAPI 两个端点漏掉重排、
`/query` 还漏掉 `max_contexts` 截断；README 的头条数字测的是「改写 + 重排 + 截断」，
而唯一对外交付的 `/query` 跑的是另一条配置。

这里用「送进 LLM 的 prompt 必须逐字相同」来锁死：任何一条路径少拼一个环节，
prompt 就会与其他路径不一致。另两条护栏是重排调用次数与上下文块数——它们能在
prompt 相同的情况下仍然悄悄漂移（比如顺序不同但内容相同）。

CLI 原本**不在**这条护栏里（helper 名为 `_drive_all_four`，PLAN §5.4 W1 却写着「五条入口
含 CLI」——2026-09-20 核对时改正）。它确实复用同一个 `Orchestrator`，但正因为只是「确实」，
少传一个参数、多带一个默认值都不会有任何测试报警，所以这里通过 `CliRunner` 驱动真正的
命令行入口（连 typer 的参数解析一起过），CLI 的默认与 `--stream` 两种形态各算一次调用。
"""

from __future__ import annotations

import json
import re
import threading
from types import SimpleNamespace
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from doc_rag import agent
from doc_rag.api import demo as demo_mod
from doc_rag.api import main as api_main
from doc_rag.eval import runner
from doc_rag.generate import llm as llm_mod
from doc_rag.generate import two_stage
from doc_rag.generate.synthesizer import Synthesizer
from doc_rag.orchestrator import Orchestrator
from doc_rag.retrieve.hybrid import RetrievalOutcome

QUESTION = "客服系统升级的预算是多少？"
SUB_QUERY = (
    "子查询二"  # agent 第二步的检索串；SpyRetriever 靠它区分「第一次」与「扩展步」
)
N_RETRIEVED = 12
MAX_CONTEXTS = 10
AGENT_CONTEXTS = 12  # agent 臂替换单发口径的那份预算（configs 里的 agent.max_contexts）
_CTX_HEAD = re.compile(r"^\[\d+\] ", re.MULTILINE)


def _cfg(agent_on: bool = False, two_stage_on: bool = False):
    cfg = {
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
    if agent_on:
        cfg["agent"] = {
            "enabled": True,
            "types": ["cross_doc", "time_filter"],
            "max_steps": 3,
            "judge_contexts": 6,
            "max_contexts": AGENT_CONTEXTS,
            "max_prompt_tokens": 100_000,
            "timeout_s": 20,
            "max_attempts": 1,
        }
    if two_stage_on:
        cfg["synthesis"] = {
            "two_stage": {
                "enabled": True,
                "types": ["cross_doc", "time_filter"],
                "timeout_s": 20,
                "max_attempts": 1,
                "fail_ratio": 0.3,
                "workers": 4,
            }
        }
    return cfg


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
        chunks = [
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
        if question == SUB_QUERY:
            # agent 的第二步要真带回来一篇新文档，否则「并集」在替身下永远等于
            # 第一步，测试会去断言一个根本没发生的行为（并直接 no_new_evidence 停住）。
            chunks.append(
                {
                    "doc_id": "d_new",
                    "title": "文档新",
                    "page": 99,
                    "text": "正文新",
                    "block_type": "paragraph",
                    "score": 0.5,
                }
            )
        return RetrievalOutcome(chunks=chunks)


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
def judge():
    """agent 判定的脚本。`replies` 是**每条入口一份**，驱动每条之前会清空 `calls`。

    为什么必须可重置也不许多跑：五条入口各跑一遍，判定输出必须逐字相同，否则
    「trace 步数逐项相等」根本没有可比对象——这是 PLAN §5.5 把 parity 扩到 agent mode
    的唯一前提（判定本身不可复现，mock 里不脚本化就没法断言）。
    """
    return SimpleNamespace(replies=[], calls=[])


@pytest.fixture
def mapper():
    """map 段（两段式微摘要）的脚本，语义同上面的 `judge`：逐篇逐入口可重置。

    回复按调用序循环取用：同一入口内每篇文档一份，跨入口必须看到同一序列。
    map 段走线程池（workers>1），取号必须持锁——两篇文档取到同一个序号会让
    「每入口 12 次调用」这条断言随机翻车。
    """
    return SimpleNamespace(
        replies=["相关事实：该篇决定了事项甲", "无关"],
        calls=[],
        lock=threading.Lock(),
    )


@pytest.fixture
def prompt_spy(monkeypatch, judge, mapper):
    """拦在 llm 层：记录真实 Synthesizer 拼出来的 prompt，而不是替身的答案。

    四个调用方共用同一个 chat 入口（改写 / agent 判定 / map 微摘要 / 合成），必须按
    system_prompt 分流：只把答案侧的调用记进 `seen`（否则「六次合成」会数错），
    判定记进 `judge`，微摘要记进 `mapper`——后两者的次数与内容就是 agent /
    two_stage mode 下要比的那几项。
    """
    from doc_rag.retrieve.rewrite_llm import SYSTEM_REWRITE

    seen: list[dict] = []

    def _rewrite_reply(user_prompt: str) -> str:
        # aggregate=true：agent 的分题型开关键在改写预测出来的题型上，
        # 不喂聚合意图就只有合成侧一条臂可比，agent 那条永远开不起来。
        return (
            '{"rewritten": "改写后的检索串", "aggregate": true, "year": null,'
            ' "reason": "test"}'
        )

    def _evidence(user_prompt: str) -> str:
        i = len(judge.calls)
        assert i < len(judge.replies), (
            f"判定调用超出脚本喂的份数（第 {i + 1} 次）——agent 多走了一步，"
            "预算或停机条件写坏了"
        )
        judge.calls.append({"user": user_prompt})
        return judge.replies[i]

    def _evidence_meta() -> dict:
        # token 固定：agent 的预算按 prompt token 封顶，跨入口必须逐项相等
        return {
            "ms": 0.4,
            "cached": False,
            "model": "m",
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "reasoning_tokens": 0,
        }

    def _map(user_prompt: str) -> str:
        with mapper.lock:
            i = len(mapper.calls)
            mapper.calls.append({"user": user_prompt})
        # 循环取用：篇数由 parity 断言锁（每入口 = 文档数），脚本只负责确定性
        return mapper.replies[i % len(mapper.replies)]

    def _map_meta() -> dict:
        return {
            "ms": 0.3,
            "cached": False,
            "model": "m",
            "prompt_tokens": 60,
            "completion_tokens": 12,
            "reasoning_tokens": 0,
        }

    def _timed(llm_cfg, user_prompt, system_prompt=None, temperature=None):
        if system_prompt == SYSTEM_REWRITE:
            return _rewrite_reply(user_prompt), {
                "ms": 0.5,
                "cached": False,
                "model": "m",
            }
        if system_prompt == agent.SYSTEM_EVIDENCE:
            return _evidence(user_prompt), _evidence_meta()
        if system_prompt == two_stage.SYSTEM_MAP:
            return _map(user_prompt), _map_meta()
        seen.append({"user": user_prompt, "system": system_prompt})
        return "答案 [1]", {"ms": 1.0, "cached": False, "model": "m"}

    def _stream(llm_cfg, user_prompt, system_prompt=None, temperature=None):
        if system_prompt == SYSTEM_REWRITE:
            return iter([_rewrite_reply(user_prompt)]), {
                "ms": 0.5,
                "cached": False,
                "model": "m",
            }
        if system_prompt == agent.SYSTEM_EVIDENCE:
            raise AssertionError("证据判定不该走流式：它是一次二元判定")
        if system_prompt == two_stage.SYSTEM_MAP:
            raise AssertionError("微摘要不该走流式：它是 map 段的短输出")
        seen.append({"user": user_prompt, "system": system_prompt})
        return iter(["答案 ", "[1]"]), {"ms": 1.0, "cached": False, "model": "m"}

    monkeypatch.setattr(llm_mod, "chat_timed", _timed)
    monkeypatch.setattr(llm_mod, "chat_stream", _stream)
    return seen


@pytest.fixture
def trace_spy(monkeypatch):
    """接住每条入口产出的 trace。

    在 `run_agent` 这一层收而不是从各入口的返回值里读：这样「某条入口悄悄没走 agent 层」
    会直接表现为 traces 少一份，而不是六个 trace 都长得一样地错。
    """
    real = agent.run_agent
    traces: list[dict] = []

    def _spy(cfg, **kw):
        union, trace = real(cfg, **kw)
        traces.append(trace)
        return union, trace

    monkeypatch.setattr(agent, "run_agent", _spy)
    return traces


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


def _drive_eval_one(monkeypatch, tmp_path, cfg):
    """eval 单条：一次合成。"""
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


# 五条入口，CLI 的两种形态各一次 → 六次合成调用
ENTRIES: tuple[str, ...] = (
    "/query",
    "/query/stream",
    "演示页",
    "eval 单条",
    "doc-rag query",
    "doc-rag query --stream",
)


def _drive_all_entries(monkeypatch, tmp_path, cfg, judge=None, mapper=None):
    """把五条入口各跑一遍，各自触发一次合成；返回与调用顺序同名的清单。

    `judge` / `mapper` 非空时每条入口之前清空对应脚本的游标：agent mode 下五条入口
    必须看到同一份判决序列，two_stage mode 下必须看到同一份微摘要序列，否则
    「逐项相等」这条断言没有意义。
    """
    from typer.testing import CliRunner

    from doc_rag import cli as cli_mod

    def _reset():
        if judge is not None:
            judge.calls.clear()
        if mapper is not None:
            mapper.calls.clear()

    monkeypatch.setattr(api_main, "_orchestrator", lambda: Orchestrator(cfg))
    client = TestClient(api_main.app, headers={"Authorization": "Bearer t"})

    _reset()
    client.post("/query", json={"question": QUESTION})
    _reset()
    client.post("/query/stream", json={"question": QUESTION})
    _reset()
    list(demo_mod.render_answer(Orchestrator(cfg), QUESTION))
    _reset()
    _drive_eval_one(monkeypatch, tmp_path, cfg)

    monkeypatch.setattr(cli_mod, "load_config", lambda *a, **k: cfg)
    for args in (["query", QUESTION], ["query", QUESTION, "--stream"]):
        _reset()
        res = CliRunner().invoke(cli_mod.app, args)
        assert res.exit_code == 0, f"{args} 退出码 {res.exit_code}：{res.output}"

    return list(ENTRIES)


def test_every_entry_sends_identical_prompt(monkeypatch, tmp_path, wired):
    cfg = _cfg()
    entries = _drive_all_entries(monkeypatch, tmp_path, cfg)

    assert len(wired) == len(entries) == 6, "五条入口各应且只应触发一次合成调用"
    users = [p["user"] for p in wired]
    assert users == [users[0]] * len(users), f"prompt 漂移：{entries}"
    assert len({p["system"] for p in wired}) == 1
    # 重排：cfg.rerank.enabled 为真时没有任何一条路径可以跳过它
    assert SpyReranker.calls == len(entries)
    # 截断：max_contexts 对每条入口同样生效
    for entry, user in zip(entries, users, strict=True):
        assert len(_CTX_HEAD.findall(user)) == MAX_CONTEXTS, entry


# ── mode 维度：agent 臂下五条入口同样要逐项相等 ─────────────────────────

# 两步停：先判「不够 + 再查子查询二」，再判「够了」。判决序列脚本化是 mock 下能
# 比 trace 的唯一前提（见 judge fixture 的 docstring）。
AGENT_SCRIPT = (
    json.dumps(
        {
            "sufficient": False,
            "missing": "缺另一半决定",
            "next_query": SUB_QUERY,
            "widen_around": [],
        },
        ensure_ascii=False,
    ),
    '{"sufficient": true, "missing": "", "next_query": "", "widen_around": []}',
)


def test_agent_mode_traces_match_across_all_entries(
    monkeypatch, tmp_path, wired, trace_spy, judge
):
    """agent mode 追加的 parity：步数、每步 action 集合、调用次数、token 合计逐项相等。

    这是 §4 那条「五入口一致性」在 agent 层的延伸。任何一条入口悄悄绕过 policy 层
    （或自己多带一份默认预算），这里都会表现为 trace 少一份或某一项不等。
    """
    cfg = _cfg(agent_on=True)
    judge.replies = list(AGENT_SCRIPT)
    entries = _drive_all_entries(monkeypatch, tmp_path, cfg, judge)

    assert len(trace_spy) == len(entries), f"有入口没走 agent 层：{entries}"
    actions = [tuple(s["action"] for s in t["steps"]) for t in trace_spy]
    assert set(actions) == {("check_evidence", "search", "check_evidence")}, actions
    assert {t["stop_reason"] for t in trace_spy} == {"sufficient"}
    assert len({t["budget_used"]["calls"] for t in trace_spy}) == 1
    assert len({t["budget_used"]["prompt_tokens"] for t in trace_spy}) == 1
    assert len({tuple(t["sub_queries"]) for t in trace_spy}) == 1
    # 并集进来的新文档也要一致：某条入口少并了一篇，答案就不是同一批证据
    assert len({t["n_docs_union"] for t in trace_spy}) == 1


# ── synthesis_route 维度：两段式臂下五条入口同样要逐项相等 ────────────────


def test_two_stage_mode_matches_across_all_entries(
    monkeypatch, tmp_path, wired, mapper
):
    """两段式的 parity（UPGRADE §3.5）：每入口 map 调用数、map 与 reduce 的 prompt、
    latency 键集合逐项相等。

    路由是 Orchestrator 内的分支而不是新入口，所以这条护栏的形态与 agent mode
    那条相同：任何一条入口悄悄少跑 map、多喂一篇文档、或换一份 reduce prompt，
    都会在这里表现为计数或内容不等。
    """
    cfg = _cfg(two_stage_on=True)
    entries = _drive_all_entries(monkeypatch, tmp_path, cfg, mapper=mapper)

    # mapper.calls 在每条入口前清空：收尾时剩下的就是**最后一条入口**的 map 调用。
    # SpyRetriever 每次返回 12 篇不同文档 → 每入口应恰 12 次；其余入口若少跑/多跑，
    # 它们的 reduce prompt（由摘要拼成）就不可能与其他入口逐字相等——下面那条接住。
    assert len(mapper.calls) == 12, (
        f"最后一条入口的 map 调用数应为文档篇数 12，实得 {len(mapper.calls)}——"
        "清单长度或归组方式漂移了"
    )
    docs_mapped = {re.search(r"【文档】(.+)", c["user"]).group(1) for c in mapper.calls}
    assert len(docs_mapped) == 12, "map 调用没有按文档归组：同一篇被摘要了不止一次"

    reduces = [p for p in wired if p["system"] == two_stage.SYSTEM_ANSWER_TWO_STAGE]
    assert len(reduces) == len(entries), "每入口应且只应有一次 reduce 调用"
    assert {p["user"] for p in reduces} == {reduces[0]["user"]}, "reduce prompt 漂移"
    # reduce 是两段式下唯一的「合成」调用；其余 seen 项不该存在
    assert len(wired) == len(reduces), f"意外多出的合成调用：{entries}"

    results = Orchestrator(cfg).answer(QUESTION)
    assert results.synthesis_route == "two_stage"
    assert results.summaries is not None and len(results.summaries) == 12
    # 引用语义：contexts 仍是原始块（12 篇各 1 块），citation 编号指原始块序号
    assert len(results.contexts) == N_RETRIEVED
    assert [c["no"] for c in results.citations] == list(range(1, N_RETRIEVED + 1))
    assert set(results.latency_ms) == {
        "rewrite",
        "retrieve",
        "rerank",
        "retrieval_total",
        "map_ms",
        "synthesize",
        "synth_cached",
        "total",
    }
    assert results.map_meta is not None and results.map_meta["n_docs"] == 12
    assert results.map_meta["n_degraded"] == 0


def test_agent_mode_widens_the_answer_prompt_by_design(wired, judge):
    """agent 臂进 LLM 的块数由 `agent.max_contexts` 替换单发口径——这条锁的是「口径不混用」。

    两臂的 prompt 本来就该不同（更多证据是这层存在的理由），但差值必须是**配置里写着
    的那个值**，不能是某条入口顺手多带几块。反过来它也警告读数字的人：agent 臂与单发
    臂的 faithfulness 不可直接对读，要等长对照臂（PLAN §5.5 门槛 2）。
    """
    judge.replies = [
        '{"sufficient": true, "missing": "", "next_query": "", "widen_around": []}'
    ]
    Orchestrator(_cfg()).answer(QUESTION)
    single = len(_CTX_HEAD.findall(wired[-1]["user"]))
    Orchestrator(_cfg(agent_on=True)).answer(QUESTION)
    widened = len(_CTX_HEAD.findall(wired[-1]["user"]))

    assert single == MAX_CONTEXTS
    assert widened == AGENT_CONTEXTS
    assert len(judge.calls) == 1  # 证据够 → 只付一次判定，没有第二次检索


def test_kb_param_does_not_stick_across_requests(monkeypatch, wired):
    """回归：改造前端点用 `retriever.collection = body.kb` 改共享单例，并发会串库。"""
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

"""RGB 跑批的容错与续跑测试：全离线（假 Orchestrator，不调 LLM）。

这些断言护的是**几小时长跑**的承重面：一次网络抖动不该让整轮白花，而失败条目更不该
被算成「答错」——那是把「没测」印成「测了且全错」，本项目在 `refusal_acc` 上踩过。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_rag.benchmarks import rgb, rgb_runner


def _records(n: int) -> list[rgb.Record]:
    out = []
    for i in range(n):
        raw = {
            "id": f"t{i}",
            "query": f"问题{i}",
            "answer": ["答案"],
            "positive": ["正面文档"],
            "negative": ["噪声文档"],
        }
        out.append(
            rgb.Record(id=raw["id"], query=raw["query"], answer=raw["answer"], raw=raw)
        )
    return out


def _fact_records(n: int) -> list[rgb.Record]:
    """反事实形状：官方那一族用 `positive_wrong`（被篡改的文档），字段不能少。"""
    out = []
    for i in range(n):
        raw = {
            "id": f"f{i}",
            "query": f"问题{i}",
            "answer": "70",
            "fakeanswer": "170",
            "positive": ["正确文档"],
            "positive_wrong": ["错误文档"],
            "negative": ["噪声文档"],
        }
        out.append(
            rgb.Record(id=raw["id"], query=raw["query"], answer=raw["answer"], raw=raw)
        )
    return out


class _Orch:
    """假 Orchestrator：对指定 id 抛异常，其余返回固定答案。"""

    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self.fail_ids = fail_ids or set()
        self.calls: list[str] = []

    def answer_given_contexts(self, question, docs, **kw):
        self.calls.append(question)
        if question in self.fail_ids:
            raise RuntimeError("LLM 调用失败（attempt=4）：Connection error.")
        return type(
            "R",
            (),
            {
                "answer": "根据文档，答案是 答案 [1]",
                "synth_meta": {
                    "model": "m",
                    "cached": False,
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "reasoning_tokens": 0,
                },
                "contexts": [{"no": 1}],
            },
        )()


def test_a_single_failure_does_not_abort_the_run_and_is_not_scored_as_wrong():
    """一条失败 → 记录 error 后继续；且它**不进分母**。

    若失败条目进分母，空答案会被判据算成答错，拒绝率/准确率会被网络抖动污染——
    报出来的数与实际测过的东西就不是一回事了。
    """
    rows = rgb_runner.run_combo(
        {"llm": {}},
        _records(4),
        "zh",
        0.0,
        orch=_Orch(fail_ids={"问题1"}),
    )
    assert len(rows) == 4, "失败也必须留一行记录，否则没人知道它没跑"
    errs = [r for r in rows if r.get("error")]
    assert len(errs) == 1 and "Connection error" in errs[0]["error"]
    assert errs[0]["label"] is None and errs[0]["prediction"] == ""

    summary = rgb_runner.summarize(rows, {})
    assert summary["n"] == 3, "分母只能是真正测过的条数"
    assert summary["n_errors"] == 1
    assert summary["all_rate"] == 1.0, "3 条全对 → 1.0，失败那条不该把它拉到 0.75"


def test_done_index_skips_finished_rows_but_retries_failed_ones(tmp_path):
    """续跑只补未完成的：已完成的复用（不再花钱），失败的必须重试。"""
    orch = _Orch(fail_ids={"问题0"})
    first = rgb_runner.run_combo({"llm": {}}, _records(3), "zh", 0.0, orch=orch)
    assert [r["id"] for r in first if r.get("error")] == ["t0"]

    done = rgb_runner.done_index(first)
    assert "t1" in {k[3] for k in done} and "t0" not in {k[3] for k in done}

    orch2 = _Orch()  # 这次全通
    second = rgb_runner.run_combo(
        {"llm": {}}, _records(3), "zh", 0.0, orch=orch2, done=done
    )
    assert orch2.calls == ["问题0"], "已完成的 t1/t2 不该被重复调用"
    assert not any(r.get("error") for r in second), "失败条目在续跑里应被补上"
    assert len(second) == 3


def test_sink_writes_each_row_immediately():
    """逐条落盘：长跑被打断时，已花的调用必须留在盘上。"""
    written: list[dict] = []
    rgb_runner.run_combo(
        {"llm": {}}, _records(3), "zh", 0.0, orch=_Orch(), sink=written.append
    )
    assert len(written) == 3
    assert all("id" in row for row in written)


def test_load_rows_round_trips_and_tolerates_blank_lines(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"id":"a"}\n\n{"id":"b"}\n', encoding="utf-8")
    assert [r["id"] for r in rgb_runner.load_rows(p)] == ["a", "b"]
    assert rgb_runner.load_rows(tmp_path / "missing.jsonl") == []


def test_parallel_run_matches_the_serial_run_row_for_row():
    """并发只改墙钟：同样输入必须给出逐字相同的行、且顺序一致。

    若并发引入了跨条共享状态（比如共用一个可变 prompt 缓冲），这里会立刻不一致。
    端点上偶发分钟级停顿，所以并发不是优化而是可行性——但它的正确性必须被钉住。
    """
    serial = rgb_runner.run_combo(
        {"llm": {}}, _records(6), "zh", 0.0, orch=_Orch(), workers=1
    )
    parallel = rgb_runner.run_combo(
        {"llm": {}}, _records(6), "zh", 0.0, orch=_Orch(), workers=4
    )
    assert [r["id"] for r in parallel] == [r["id"] for r in serial], "map 必须保序"
    for a, b in zip(serial, parallel, strict=True):
        # ms 是墙钟，必然不同；其余字段必须逐字相同
        assert {k: v for k, v in a.items() if k != "ms"} == {
            k: v for k, v in b.items() if k != "ms"
        }


def test_parallel_run_keeps_error_tolerance_and_sink():
    """并发下失败照样只记录不中断，且每条都落盘一次。"""
    written: list[dict] = []
    rows = rgb_runner.run_combo(
        {"llm": {}},
        _records(5),
        "zh",
        0.0,
        orch=_Orch(fail_ids={"问题3"}),
        workers=3,
        sink=written.append,
    )
    assert len(rows) == len(written) == 5
    assert sum(1 for r in rows if r.get("error")) == 1
    assert rgb_runner.summarize(rows, {})["n"] == 4


def test_report_marks_combos_whose_denominator_was_shrunk():
    """表里的 n 必须带出失败条数——不然读者会以为分母是完整的。"""
    rows = rgb_runner.run_combo(
        {"llm": {}}, _records(4), "zh", 0.0, orch=_Orch(fail_ids={"问题2"})
    )
    s = rgb_runner.summarize(rows, {})
    s["instruction"] = "production"
    text = rgb_runner.render_reports([s])
    assert "3+1错" in text


def test_call_plan_counts_every_combo_including_rejection():
    """组合数就是钱：漏掉拒答档（noise=1）会少算几百次调用。"""
    plan = rgb_runner.call_plan(
        ["zh", "zh_refine", "zh_int", "zh_fact"],
        {"zh": 300, "zh_refine": 300, "zh_int": 100, "zh_fact": 100},
    )
    # zh 5 档 + 拒答 1 档 = 6；zh_refine 同；zh_int 3；zh_fact 1 → 16 个组合
    assert plan["combos"] == 16
    assert plan["calls"] == 300 * 6 + 300 * 6 + 100 * 3 + 100


def test_call_plan_follows_the_sample_size():
    """抽样降档时账要跟着变，否则闸门会按全量拦人、白跑一次。"""
    plan = rgb_runner.call_plan(["zh", "zh_int"], {"zh": 300, "zh_int": 100}, sample=50)
    assert plan["calls"] == 50 * 6 + 50 * 3


def test_sample_is_uniform_reproducible_and_a_subset_of_the_full_run():
    """抽样必须均匀、可复现，且**是全量的子集**——这样之后续跑到全量不重复付费。

    为什么不能用 `--limit`：前 N 条是按 id 排的（RGB 的 id 与难度无关但也不是随机），
    本项目明令「抽样必须均匀覆盖，不能取前 N 条」。
    """
    records = _records(50)
    a = rgb_runner.select_records(records, sample=10)
    b = rgb_runner.select_records(records, sample=10)
    assert [r.id for r in a] == [r.id for r in b], "同种子必须可复现"
    assert len(a) == 10
    assert {r.id for r in a} <= {r.id for r in records}
    # 均匀：抽样不能等于「前 10 条」
    assert [r.id for r in a] != [r.id for r in records[:10]]


def test_sample_larger_than_the_pool_means_full_run():
    records = _records(5)
    assert len(rgb_runner.select_records(records, sample=99)) == 5
    assert len(rgb_runner.select_records(records, sample=5)) == 5


def test_limit_and_sample_are_mutually_exclusive():
    with pytest.raises(ValueError, match="只能给一个"):
        rgb_runner.select_records(_records(3), limit=1, sample=1)
    with pytest.raises(ValueError, match="只能给一个"):
        rgb_runner.run_combo(
            {"llm": {}}, _records(3), "zh", 0.0, orch=_Orch(), limit=1, sample=1
        )


def test_extrapolate_scales_tokens_from_the_small_run():
    """外推的输入是「本次全部组合的合计」，不是某一个组合。

    第一版把每组合条数当成了全量调用数（4000 被说成 250，差 16 倍）——成本纪律
    要防的正是这类错，所以这里把「全量 = 组合数 × 每组合条数」这个关系钉住。
    """
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "reasoning_tokens": 150,
        "total_ms": 10_000,
        "n": 10,
    }
    out = rgb_runner.extrapolate(usage, n_limited=10, n_full=400)
    assert out["calls"] == 400
    assert out["prompt_tokens"] == 40_000
    assert out["per_call_prompt_tokens"] == 100
    assert out["per_call_completion_tokens"] == 20
    # 1000ms/条 × 400 条 = 400s ≈ 0.1 小时
    assert out["wall_hours"] == 0.1


def test_total_usage_sums_every_combo_not_just_one():
    """合计必须跨组合：只看一个组合会把总账算小一个数量级。"""
    summaries = [
        {
            "n": 2,
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "reasoning_tokens": 40,
            },
            "ms": {"p50": 1000.0},
        },
        {
            "n": 3,
            "usage": {
                "prompt_tokens": 300,
                "completion_tokens": 90,
                "reasoning_tokens": 70,
            },
            "ms": {"p50": 2000.0},
        },
    ]
    total = rgb_runner.total_usage(summaries)
    assert total["n"] == 5
    assert total["prompt_tokens"] == 400
    assert total["completion_tokens"] == 140
    # 耗时按各组合 p50×n 近似：1000×2 + 2000×3 = 8000ms
    assert total["total_ms"] == 8000


def test_judge_sidecar_round_trips_and_reuses_cache(tmp_path, monkeypatch):
    """判分旁挂：第二次读同一批不该再调 judge（重跑判分不重花钱）。"""
    from doc_rag.generate import llm as llm_mod

    calls = {"n": 0}

    def _fake(cfg, user_prompt, system_prompt=None, temperature=None):
        calls["n"] += 1
        if "not addressed" in user_prompt:
            return "No, the question is not addressed by the documents.", {}
        return "Yes, the model has identified the factual errors.", {}

    monkeypatch.setattr(llm_mod, "chat_timed", _fake)
    rows = rgb_runner.run_combo({"llm": {}}, _records(2), "zh", 1.0, orch=_Orch())
    side = rgb_runner.judge_sidecar_path(tmp_path / "r.jsonl")
    cache = rgb_runner.load_judge_sidecar(side)
    with open(side, "a", encoding="utf-8") as f:
        rgb_runner.judge_rows(
            rows, {}, cache, lambda item: f.write(json.dumps(item) + "\n")
        )
    first_calls = calls["n"]
    # zh@noise=1 只该判 Rej*（ED* 是反事实族的口径）→ 2 条 × 1 个判据
    assert first_calls == 2

    reloaded = rgb_runner.load_judge_sidecar(side)
    rows2 = rgb_runner.run_combo({"llm": {}}, _records(2), "zh", 1.0, orch=_Orch())
    rgb_runner.judge_rows(rows2, {}, reloaded, None)
    assert calls["n"] == first_calls, "命中缓存不该再调 judge"
    assert rgb_runner.star_rates(rows2)["rej_star"] == 1.0


def test_judge_only_covers_the_combos_the_official_rubric_defines(monkeypatch):
    """judge 只跑官方口径覆盖得到的组合：Rej* 在 noise=1、ED* 在反事实族。

    无差别地对全部条目跑两个判据 = 把 ~150 次该花的调用变成 ~1600 次（实测 10 倍）。
    """
    from doc_rag.generate import llm as llm_mod

    calls: list[str] = []

    def _fake(cfg, user_prompt, system_prompt=None, temperature=None):
        calls.append("rej" if "not addressed" in user_prompt else "fact")
        return "No, the question is not addressed by the documents.", {}

    monkeypatch.setattr(llm_mod, "chat_timed", _fake)
    # zh@0.0：官方没定义 → 一次都不该调
    mid = rgb_runner.run_combo({"llm": {}}, _records(3), "zh", 0.0, orch=_Orch())
    rgb_runner.judge_rows(mid, {}, {}, None)
    assert calls == []
    assert all(r["rej_star"] is None and r["ed_star"] is None for r in mid)
    assert rgb_runner.star_rates(mid) == {}, "全未判时不该造出一个 0 分的读数"

    # zh_fact：只判 ED*，不判 Rej*
    fact = rgb_runner.run_combo(
        {"llm": {}}, _fact_records(3), "zh_fact", 0.0, orch=_Orch()
    )
    rgb_runner.judge_rows(fact, {}, {}, None)
    assert calls == ["fact"] * 3
    assert all(r["rej_star"] is None for r in fact)
    assert rgb_runner.star_rates(fact)["ed_star_n"] == 3


def test_star_rates_excludes_unjudged_rows_from_the_denominator(monkeypatch):
    """未判（None）不能混进分母当 False——那正是 refusal_acc 上修过的错。"""
    rows = [
        {"rej_star": True, "ed_star": None},
        {"rej_star": True, "ed_star": None},
        {"rej_star": None, "ed_star": None},
    ]
    out = rgb_runner.star_rates(rows)
    assert out["rej_star"] == 1.0 and out["rej_star_n"] == 2
    assert "ed_star" not in out, "一个都没判过就不该有这个读数"


def test_unknown_instruction_and_missing_orch_are_rejected():
    with pytest.raises(ValueError, match="未知 instruction"):
        rgb_runner.run_combo(
            {"llm": {}}, _records(1), "zh", 0.0, instruction="x", orch=_Orch()
        )
    with pytest.raises(ValueError, match="需要传入 Orchestrator"):
        rgb_runner.run_combo({"llm": {}}, _records(1), "zh", 0.0)


def test_fetch_manifest_records_the_pinned_commit():
    """结果文件要靠 commit 自证测的是哪一版数据——常量必须与抓取用的一致。"""
    assert rgb.UPSTREAM_COMMIT == "65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615"
    assert "NonCommercial" in rgb.UPSTREAM_LICENSE or "非商用" in rgb.UPSTREAM_LICENSE


def test_dataset_path_joins_root():
    assert rgb.dataset_path(Path("d"), "zh") == Path("d") / "zh.json"


def test_only_rates_filters_combos_for_a_same_question_comparison():
    """同题对照只该跑那一档：跑全档扫一遍等于多花 5 倍钱买与对照无关的档位。

    检索变体是「自己找文档」，它要对比的是 `noise_rate=0.0`（喂 5 篇正确文档）——
    所以过滤必须同时作用于调用计划与实际循环，否则账与实际调用会对不上。
    """
    all_combos = rgb_runner.protocol_combos(["zh"])
    assert len(all_combos) == 6, "zh 是 5 档 + 拒答档"
    only = rgb_runner.protocol_combos(["zh"], only_rates=(0.0,))
    assert only == [("zh", 0.0)]
    plan = rgb_runner.call_plan(["zh"], {"zh": 300}, only_rates=(0.0,))
    assert plan["combos"] == 1 and plan["calls"] == 300

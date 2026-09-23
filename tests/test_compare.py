"""配对判读工具测试：重点是「子集抽样能造出假差异」这一失败模式可被复现。"""

from __future__ import annotations

import json

import pytest

from doc_rag.eval.compare import (
    compare,
    compare_retrieval,
    format_report,
    holm_bonferroni,
    load_scores,
)


def _write(
    tmp_path,
    name: str,
    scores: dict[str, float],
    types: dict[str, str] | None = None,
    ctx: dict[str, int] | None = None,
):
    types = types or {}
    per_item = []
    for i, s in scores.items():
        entry: dict = {"id": i, "type": types.get(i, "fact"), "faithfulness": s}
        if ctx:
            entry["n_contexts"] = ctx[i]
        per_item.append(entry)
    payload = {
        "meta": {"collection": "c", "retrieval": "r"},
        "summary": {
            "n": len(per_item),
            "metrics": ["faithfulness"],
            "faithfulness": round(sum(scores.values()) / len(scores), 4),
            "per_item": per_item,
        },
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_scores_rejects_legacy_mean_only_file(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {"summary": {"n": 15, "metrics": ["faithfulness"], "faithfulness": 0.64}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="没有逐条分数"):
        load_scores(legacy)


def test_paired_diff_and_sign_test(tmp_path):
    # A 全 1.0；B 在 10 条里 7 条更低、3 条持平
    a = _write(tmp_path, "a.json", {f"q{i:02d}": 1.0 for i in range(10)})
    b_scores = {f"q{i:02d}": (0.0 if i < 7 else 1.0) for i in range(10)}
    b = _write(tmp_path, "b.json", b_scores)
    rep = compare([{"label": "A", "files": [a]}, {"label": "B", "files": [b]}])
    paired = rep["paired"][0]
    assert paired["n_paired"] == 10
    assert paired["mean_diff"] == pytest.approx(-0.7)
    assert (paired["wins"], paired["losses"], paired["ties"]) == (0, 7, 3)
    assert paired["sign_test_p"] < 0.05  # 7 负 0 正，双侧 p=2/2^7
    assert rep["groups"][1]["mean"] == pytest.approx(0.3)


def test_repeat_runs_give_noise_floor(tmp_path):
    # 同条件两轮：逐条分数不同 → 极差就是噪声地板，应被报告
    r1 = _write(tmp_path, "r1.json", {"q1": 1.0, "q2": 1.0})
    r2 = _write(tmp_path, "r2.json", {"q1": 0.5, "q2": 1.0})
    rep = compare([{"label": "A", "files": [r1, r2]}, {"label": "B", "files": [r1]}])
    assert rep["groups"][0]["run_means"] == [1.0, 0.75]
    assert rep["groups"][0]["noise_range"] == pytest.approx(0.25)
    assert rep["groups"][1]["noise_range"] is None  # 只跑一轮不报噪声
    assert rep["groups"][0]["items"]["q1"] == pytest.approx(0.75)  # 重跑取逐条均值


def test_subset_sensitivity_exposes_order_bias(tmp_path):
    """黄金集按题型排序时，前 N 条能造出与全量**方向相反**的结论（真实踩过的坑）。"""
    ids = [f"q{i:02d}" for i in range(40)]
    types = {i: ("cross_doc" if n >= 10 else "fact") for n, i in enumerate(ids)}
    # A 在 fact 段全对、cross_doc 段全错；B 正好相反
    a_scores = {i: (1.0 if n < 10 else 0.0) for n, i in enumerate(ids)}
    b_scores = {i: (0.0 if n < 10 else 1.0) for n, i in enumerate(ids)}
    a = _write(tmp_path, "a.json", a_scores, types)
    b = _write(tmp_path, "b.json", b_scores, types)
    rep = compare(
        [{"label": "A", "files": [a]}, {"label": "B", "files": [b]}], subset_sizes=(10,)
    )
    rows = {r["note"]: r for r in rep["subset_sensitivity"]}
    assert rows["前 10 条"]["B"] == 0.0  # 子集：B 比 A 差 1.0
    assert rows["全量"]["A"] == 0.25
    assert rows["全量"]["B"] == 0.75  # 全量：B 反而比 A 好 0.5 —— 方向翻转
    assert rep["paired"][0]["mean_diff"] == pytest.approx(0.5)
    assert rep["by_type"]["cross_doc"]["B"] == 1.0


def test_format_report_renders_all_sections(tmp_path):
    a = _write(
        tmp_path, "a.json", {"q1": 1.0, "q2": 0.0}, {"q1": "fact", "q2": "cross_doc"}
    )
    b = _write(
        tmp_path, "b.json", {"q1": 0.5, "q2": 0.5}, {"q1": "fact", "q2": "cross_doc"}
    )
    text = format_report(
        compare([{"label": "A", "files": [a]}, {"label": "B", "files": [b]}])
    )
    assert "全量同题配对差异" in text
    assert "子集敏感性" in text
    assert "cross_doc" in text
    # 两条轨的极差不是同一种噪声，标题不能串台
    assert "RAGAS 配对判读" in text
    assert "judge 噪声地板" in text
    assert "检索" not in text.splitlines()[0]


def test_load_scores_accepts_eval_results_format(tmp_path):
    """T4 三组对照的 compare 直接吃 eval 结果文件：ragas 摘要嵌在 results["ragas"]。"""
    payload = {
        "meta": {"collection": "c", "retrieval": "r", "llm_model": "m"},
        "summary": {
            "n_items": 2,
            "hit_at_5": 0.9,
            "mrr": 0.8,
        },  # 客观指标，无 per_item
        "ragas": {
            "n": 2,
            "metrics": ["faithfulness"],
            "faithfulness": 0.95,
            "per_item": [
                {"id": "q1", "type": "fact", "faithfulness": 1.0},
                {"id": "q2", "type": "term", "faithfulness": 0.9},
            ],
        },
        "items": [],
    }
    path = tmp_path / "eval_results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_scores(path)
    assert loaded["metric"] == "faithfulness"
    assert loaded["items"] == {"q1": 1.0, "q2": 0.9}
    assert loaded["types"] == {"q1": "fact", "q2": "term"}


# --------------------------------------------------------------------------- 检索轨


def _ret_row(
    rid: str,
    rank: int | None,
    cov: float | None,
    ceiling: float | None,
    ndcg: float | None,
    n_ret: int,
    n_ctx: int,
    rtype: str = "fact",
) -> dict:
    return {
        "id": rid,
        "type": rtype,
        "question": rid,
        "first_hit_rank": rank,
        "doc_coverage": cov,
        "doc_coverage_ceiling": ceiling,
        "ndcg_at_8": ndcg,
        "n_retrieved": n_ret,
        "n_contexts": n_ctx,
        "contexts": [f"[{i}]（doc_{i} 第1页）正文" for i in range(1, n_ctx + 1)],
    }


def _write_ret(tmp_path, name: str, rows: list[dict]):
    payload = {
        "meta": {"collection": "c", "retrieval": "r", "top_n": 8, "budget": "fixed:8"},
        "summary": {"n_items": len(rows)},
        "ragas": None,
        "items": rows,
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


_ROWS = [
    _ret_row("q1", 3, 0.5, 1.0, 0.7, 8, 8),
    _ret_row("q2", 9, 0.0, 1.0, 0.0, 8, 8),
    _ret_row("q3", None, 0.0, 1.0, 0.0, 8, 8),
    _ret_row("q4", None, None, None, None, 8, 6, rtype="no_answer"),  # 无 gold
]


def test_retrieval_track_derives_per_item_scores_from_items(tmp_path):
    f = _write_ret(tmp_path, "a.json", _ROWS)
    assert load_scores(f, "hit_at_5")["items"] == {"q1": 1.0, "q2": 0.0, "q3": 0.0}
    assert load_scores(f, "hit_at_8")["items"]["q2"] == 0.0
    assert load_scores(f, "hit_within_budget")["items"] == {
        "q1": 1.0,
        "q2": 1.0,
        "q3": 0.0,
    }
    assert load_scores(f, "mrr")["items"]["q1"] == pytest.approx(1 / 3)
    assert load_scores(f, "ndcg_at_8")["items"]["q1"] == 0.7
    assert load_scores(f, "mean_doc_coverage")["items"]["q1"] == 0.5
    assert load_scores(f, "coverage_vs_ceiling")["items"]["q1"] == pytest.approx(0.5)
    assert load_scores(f, "hit_at_5")["track"] == "retrieval"
    # 拒答题没有 gold：出现在 types 里，但一个检索指标的分母都不进
    assert "q4" not in load_scores(f, "hit_at_5")["items"]
    assert load_scores(f, "hit_at_5")["types"]["q4"] == "no_answer"


def test_retrieval_file_without_metric_points_at_the_right_command(tmp_path):
    f = _write_ret(tmp_path, "a.json", _ROWS)
    with pytest.raises(ValueError, match="compare-retrieval"):
        load_scores(f)


def test_unequal_list_lengths_between_arms_are_flagged(tmp_path):
    """重排的「−3.4pt / −1.6pt」就是靠这个伪影活下来的：两臂清单 6 格 vs 8 格。"""
    a = _write_ret(
        tmp_path,
        "a.json",
        [_ret_row(f"q{i}", i + 1, 1.0, 1.0, 1.0, 8, 8) for i in range(4)],
    )
    b = _write_ret(
        tmp_path,
        "b.json",
        [_ret_row(f"q{i}", i + 1, 1.0, 0.75, 1.0, 6, 6) for i in range(4)],
    )
    rep = compare(
        [{"label": "A", "files": [a]}, {"label": "B", "files": [b]}], metric="hit_at_5"
    )
    warns = rep["paired"][0]["warnings"]
    assert any("检索清单不等长" in w for w in warns)
    assert any("上下文块不等长" in w for w in warns)
    assert any("覆盖率上限不同" in w for w in warns)
    assert rep["groups"][1]["ceiling_mean"] == pytest.approx(0.75)


def test_identical_config_reruns_have_zero_metric_noise(tmp_path):
    """同配置两遍：doc 级检索指标逐条全等（实测 64/64），所以噪声地板是 0 而不是估值。"""
    a = _write_ret(tmp_path, "a.json", [dict(r) for r in _ROWS])
    b = _write_ret(tmp_path, "b.json", [dict(r) for r in _ROWS])
    rep = compare(
        [{"label": "跑法一", "files": [a]}, {"label": "跑法二", "files": [b]}],
        metric="hit_at_8",
    )
    paired = rep["paired"][0]
    assert (paired["wins"], paired["losses"], paired["ties"]) == (0, 0, 3)
    assert paired["mean_diff"] == 0.0
    assert paired["sign_test_p"] == 1.0
    assert paired["warnings"] == []  # 等长同配置，不该报伪影
    text = format_report(rep)
    assert text.startswith("== 检索 配对判读")
    assert "检索重跑噪声地板" in text


# ── 答案轨的等长护栏（PLAN §5.5 门槛 2） ────────────────────────────────


def test_ragas_track_flags_unequal_context_counts(tmp_path):
    """两臂块数不等时必须报出来：agent 臂必然送更多块，而 faithfulness 随块数走高。

    没有这条告警，「agent 的 faithfulness 更高」就可能是构造出来的偏差，
    而且偏差量级正好在 judge 自身方差（≥1.9pt）那一档。
    """
    a = _write(
        tmp_path,
        "single.json",
        {f"q{i}": 0.90 for i in range(4)},
        ctx={f"q{i}": 6 for i in range(4)},
    )
    b = _write(
        tmp_path,
        "agent.json",
        {f"q{i}": 0.95 for i in range(4)},
        ctx={f"q{i}": 12 for i in range(4)},
    )
    rep = compare([{"label": "单发", "files": [a]}, {"label": "agent", "files": [b]}])
    warns = rep["paired"][0]["warnings"]
    hit = [w for w in warns if "进 LLM 的上下文块不等长" in w]
    assert hit, warns
    # 措辞要分轨：答案轨的后果是「偏向块数多的一臂」，不是检索轨的分母问题
    assert "faithfulness" in hit[0] and "偏向块数多的一臂" in hit[0]
    assert "nDCG" not in hit[0]


def test_ragas_track_with_equal_context_counts_has_no_length_warning(tmp_path):
    """同块数对照臂存在时不该报警——这条护栏的意义就是逼着那条臂存在。"""
    a = _write(
        tmp_path,
        "a.json",
        {f"q{i}": 0.90 for i in range(4)},
        ctx={f"q{i}": 8 for i in range(4)},
    )
    b = _write(
        tmp_path,
        "b.json",
        {f"q{i}": 0.95 for i in range(4)},
        ctx={f"q{i}": 8 for i in range(4)},
    )
    rep = compare([{"label": "A", "files": [a]}, {"label": "B", "files": [b]}])
    assert rep["paired"][0]["warnings"] == []


def test_legacy_ragas_without_n_contexts_stays_silent(tmp_path):
    """旧产物没这个键 → 护栏不置可否：既不假装等长，也不凭空报错。"""
    a = _write(tmp_path, "a.json", {f"q{i}": 0.9 for i in range(4)})
    b = _write(tmp_path, "b.json", {f"q{i}": 0.8 for i in range(4)})
    rep = compare([{"label": "A", "files": [a]}, {"label": "B", "files": [b]}])
    assert rep["paired"][0]["warnings"] == []
    assert load_scores(a)["aux"] == {}


def test_ragas_aux_records_per_item_context_counts(tmp_path):
    """aux 是护栏的输入：逐条块数要能读出来，均值才有意义。"""
    path = _write(
        tmp_path,
        "a.json",
        {f"q{i}": 0.9 for i in range(3)},
        ctx={"q0": 6, "q1": 7, "q2": 6},
    )
    aux = load_scores(path)["aux"]
    assert aux == {"q0": {"ctx_len": 6}, "q1": {"ctx_len": 7}, "q2": {"ctx_len": 6}}


def test_coverage_gain_that_is_purely_list_length(tmp_path):
    """清单从 4 格扩到 8 格：覆盖率翻倍，但 `coverage_vs_ceiling` 一动不动。"""
    a = _write_ret(
        tmp_path,
        "a.json",
        [
            _ret_row("q1", 1, 0.5, 0.5, 1.0, 4, 4),
            _ret_row("q2", 2, 0.5, 0.5, 1.0, 4, 4),
        ],
    )
    b = _write_ret(
        tmp_path,
        "b.json",
        [
            _ret_row("q1", 1, 1.0, 1.0, 1.0, 8, 8),
            _ret_row("q2", 2, 1.0, 1.0, 1.0, 8, 8),
        ],
    )
    groups = [{"label": "短清单", "files": [a]}, {"label": "长清单", "files": [b]}]
    cov, norm = compare_retrieval(
        groups, metrics=("mean_doc_coverage", "coverage_vs_ceiling")
    )
    assert cov["paired"][0]["mean_diff"] == pytest.approx(0.5)
    assert norm["paired"][0]["mean_diff"] == pytest.approx(0.0)  # 归一化后没有差
    assert cov["correction"]["family_size"] == 2 == norm["correction"]["family_size"]


def test_holm_family_spans_every_arm_and_metric(tmp_path):
    a = _write_ret(tmp_path, "a.json", [dict(r) for r in _ROWS])
    b = _write_ret(tmp_path, "b.json", [dict(r) for r in _ROWS])
    c = _write_ret(tmp_path, "c.json", [dict(r) for r in _ROWS])
    groups = [
        {"label": "基线", "files": [a]},
        {"label": "臂2", "files": [b]},
        {"label": "臂3", "files": [c]},
    ]
    reports = compare_retrieval(groups, metrics=("hit_at_5", "mrr", "ndcg_at_8"))
    # 3 臂 × 3 指标 = 2 个对照 × 3 = 6 个配对检验，算一个家族
    assert [len(r["paired"]) for r in reports] == [2, 2, 2]
    assert {r["correction"]["family_size"] for r in reports} == {6}


def test_holm_bonferroni_is_monotone_and_clipped():
    assert holm_bonferroni([]) == []
    assert holm_bonferroni([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert holm_bonferroni([0.9, 0.8]) == [1.0, 1.0]
    # 只有一名挑战者时，Holm 与原始 p 相同（校正不该无中生有）
    assert holm_bonferroni([0.04]) == [0.04]


def _arms_with_wins(n_items: int, wins: int) -> list[dict]:
    """基线全 miss（首命中在第 9 位），挑战臂把前 `wins` 题提到第 1 位。"""
    ids = [f"q{i:02d}" for i in range(n_items)]
    return [
        _ret_row(i, 1 if n < wins else 9, 1.0 if n < wins else 0.0, 1.0, 1.0, 8, 8)
        for n, i in enumerate(ids)
    ]


def test_holm_with_a_single_challenger_leaves_p_untouched(tmp_path):
    from doc_rag.eval.compare import apply_holm

    base = _write_ret(tmp_path, "a.json", _arms_with_wins(20, 0))  # 基线：全 miss
    arm = _write_ret(tmp_path, "b.json", _arms_with_wins(20, 6))
    rep = compare(
        [{"label": "A", "files": [base]}, {"label": "B", "files": [arm]}],
        metric="hit_at_5",
    )
    apply_holm([rep])
    paired = rep["paired"][0]
    assert (paired["wins"], paired["losses"]) == (6, 0)
    assert paired["sign_test_p"] == pytest.approx(0.03125)
    assert paired["p_holm"] == pytest.approx(paired["sign_test_p"])  # 家族=1，不放大
    assert paired["significant_holm"] is True
    text = format_report(rep)
    assert "p_holm=" in text
    assert "多重比较校正" in text


def test_multiple_arms_can_take_a_significant_result_away(tmp_path):
    """两个臂各自单次 p=0.031 过线，同族校正后都是 0.0625 → 报告必须翻成「不显著」。"""
    base = _write_ret(tmp_path, "a.json", _arms_with_wins(20, 0))
    arm2 = _write_ret(tmp_path, "b.json", _arms_with_wins(20, 6))
    arm3 = _write_ret(tmp_path, "c.json", _arms_with_wins(20, 6))
    reports = compare_retrieval(
        [
            {"label": "基线", "files": [base]},
            {"label": "臂2", "files": [arm2]},
            {"label": "臂3", "files": [arm3]},
        ],
        metrics=("hit_at_5",),
    )
    assert len(reports) == 1
    assert reports[0]["correction"]["family_size"] == 2
    for paired in reports[0]["paired"]:
        assert paired["sign_test_p"] == pytest.approx(0.03125)
        assert paired["p_holm"] == pytest.approx(0.0625)
        assert paired["significant_holm"] is False
    assert "校正后不显著" in format_report(reports[0])


def test_cli_accepts_legacy_metric_names_and_dedupes(tmp_path):
    """改名之后旧命令必须照样能跑，且同一指标被两个名字点两次只算一次比较。

    这条是被自己的回归逼出来的：别名回落在 `load_scores` 里做，而 CLI 在更早一步
    就按 `RETRIEVAL_METRICS` 校验 `--metric` → 旧名被直接拒。库层测试全绿拦不住它，
    因为它们绕过了那道校验。PLAN/README 里写下的复现命令是文档的一部分，不能失效。
    """
    from typer.testing import CliRunner

    from doc_rag import cli as cli_mod

    f = _write_ret(tmp_path, "a.json", _ROWS)
    g = _write_ret(tmp_path, "b.json", _ROWS)
    res = CliRunner().invoke(
        cli_mod.app,
        [
            "compare-retrieval",
            "--group",
            f"A={f}",
            "--group",
            f"B={g}",
            "--metric",
            "mean_doc_coverage",  # 旧名
            "--metric",
            "recall_at_list",  # 同一个指标的标准名
            "--metric",
            "coverage_vs_ceiling",  # 另一个旧名
        ],
    )
    assert res.exit_code == 0, res.output
    assert "指标：recall_at_list" in res.output
    assert "指标：recall_vs_ceiling" in res.output
    # 旧名与新名指向同一个指标 → 只能占一格，否则 Holm 家族被同一指标占两次，
    # 白白收紧其余指标的 α
    assert res.output.count("指标：recall_at_list") == 1
    assert "mean_doc_coverage" not in res.output

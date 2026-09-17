"""配对判读工具测试：重点是「子集抽样能造出假差异」这一失败模式可被复现。"""

from __future__ import annotations

import json

import pytest

from doc_rag.eval.compare import compare, format_report, load_scores


def _write(tmp_path, name: str, scores: dict[str, float], types: dict[str, str] | None = None):
    types = types or {}
    per_item = [
        {"id": i, "type": types.get(i, "fact"), "faithfulness": s} for i, s in scores.items()
    ]
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
        json.dumps({"summary": {"n": 15, "metrics": ["faithfulness"], "faithfulness": 0.64}}),
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
    a = _write(tmp_path, "a.json", {"q1": 1.0, "q2": 0.0}, {"q1": "fact", "q2": "cross_doc"})
    b = _write(tmp_path, "b.json", {"q1": 0.5, "q2": 0.5}, {"q1": "fact", "q2": "cross_doc"})
    text = format_report(compare([{"label": "A", "files": [a]}, {"label": "B", "files": [b]}]))
    assert "全量同题配对差异" in text
    assert "子集敏感性" in text
    assert "cross_doc" in text


def test_load_scores_accepts_eval_results_format(tmp_path):
    """T4 三组对照的 compare 直接吃 eval 结果文件：ragas 摘要嵌在 results["ragas"]。"""
    payload = {
        "meta": {"collection": "c", "retrieval": "r", "llm_model": "m"},
        "summary": {"n_items": 2, "recall_at_5": 0.9, "mrr": 0.8},  # 客观指标，无 per_item
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

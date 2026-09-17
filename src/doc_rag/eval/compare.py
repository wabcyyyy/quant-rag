"""RAGAS 结果的配对判读（PLAN §5.3 可信度口径）。

动机：只比均值会骗人。本语料的黄金集按题型分块排序，早先 `rows[:15]` 抽样
恰好整段漏掉排在末尾的 cross_doc / time_filter，18pt 的"差异"可能只是抽样偏置。
判读一个消融结论至少要回答三件事：

1. **全量配对差**：同题配对（同问题、不同上下文）的均值差 + 自助法置信区间 + 符号检验；
2. **judge 噪声地板**：同条件重跑的分差——差异小于它就谈不上结论；
3. **子集敏感性**：同样的逐条分数，换成前 N 条会得出什么结论。

只用 numpy / 标准库（无 scipy 依赖）。
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

_SUPPORTED = ("faithfulness", "answer_relevancy")


def load_scores(path: str | Path) -> dict:
    """读一个 RAGAS 结果文件 → {metric, items{id: score}, meta, summary}。"""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    summary = data.get("summary") or {}
    if "per_item" not in summary:
        raise ValueError(f"{p.name} 没有逐条分数（旧版只存均值，需用 --ragas-sample 重跑）")
    metric = next((m for m in summary.get("metrics", []) if m in _SUPPORTED), None)
    if metric is None:
        raise ValueError(f"{p.name} 未包含可判读指标：{summary.get('metrics')}")
    items = {
        row["id"]: row[metric]
        for row in summary["per_item"]
        if row.get(metric) is not None
    }
    return {
        "path": p,
        "metric": metric,
        "items": items,
        "types": {row["id"]: row.get("type") for row in summary["per_item"]},
        "meta": data.get("meta") or {},
        "summary": summary,
    }


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _bootstrap_ci(diffs: list[float], n: int = 5000, seed: int = 42) -> tuple[float, float]:
    """自助法 95% CI：不假设正态，适合 0/1 截断的比例型指标。"""
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def _sign_test_p(wins: int, losses: int) -> float:
    """双侧符号检验（精确二项）。n≤55 时直接算组合数。"""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2**n)


def _group_items(group: dict) -> tuple[dict[str, float], list[float]]:
    """把一组内多次重跑聚成逐条均值，并返回每轮的总均值（用于噪声地板）。"""
    per_run = [load_scores(f)["items"] for f in group["files"]]
    run_means = [_mean(list(it.values())) for it in per_run]
    ids = sorted(set().union(*[set(it) for it in per_run]))
    pooled = {
        i: _mean([it[i] for it in per_run if i in it])
        for i in ids
        if any(i in it for it in per_run)
    }
    return pooled, [m for m in run_means if m is not None]


def compare(
    groups: list[dict],
    baseline: int = 0,
    subset_sizes: tuple[int, ...] = (15, 30),
    bootstrap: int = 5000,
) -> dict:
    """groups=[{"label": str, "files": [Path, ...]}, ...]；组内多文件 = 同条件重跑。"""
    loaded = []
    for g in groups:
        items, run_means = _group_items(g)
        loaded.append({"label": g["label"], "items": items, "run_means": run_means})

    metric = load_scores(groups[0]["files"][0])["metric"]
    base = loaded[baseline]

    report: dict = {
        "metric": metric,
        "groups": [
            {
                "label": g["label"],
                "n_items": len(g["items"]),
                "mean": _mean(list(g["items"].values())),
                "run_means": g["run_means"],
                "items": g["items"],
                "noise_range": (
                    max(g["run_means"]) - min(g["run_means"])
                    if len(g["run_means"]) > 1
                    else None
                ),
            }
            for g in loaded
        ],
        "paired": [],
        "by_type": {},
        "subset_sensitivity": [],
    }

    # 全量配对差
    for idx, g in enumerate(loaded):
        if idx == baseline:
            continue
        common = sorted(set(base["items"]) & set(g["items"]))
        diffs = [g["items"][i] - base["items"][i] for i in common]
        wins = sum(1 for d in diffs if d > 1e-9)
        losses = sum(1 for d in diffs if d < -1e-9)
        lo, hi = _bootstrap_ci(diffs, n=bootstrap)
        report["paired"].append(
            {
                "vs": f"{g['label']} − {base['label']}",
                "n_paired": len(common),
                "mean_diff": _mean(diffs),
                "ci95": [lo, hi],
                "wins": wins,
                "losses": losses,
                "ties": len(diffs) - wins - losses,
                "sign_test_p": _sign_test_p(wins, losses),
                "largest_drops": sorted(
                    ((i, round(d, 4)) for i, d in zip(common, diffs)),
                    key=lambda t: t[1],
                )[:5],
            }
        )

    # 分题型
    types = load_scores(groups[0]["files"][0])["types"]
    for t in sorted({v for v in types.values() if v}):
        row = {}
        for g in loaded:
            vals = [s for i, s in g["items"].items() if types.get(i) == t]
            row[g["label"]] = _mean(vals)
            row[f"{g['label']}#n"] = len(vals)
        report["by_type"][t] = row

    # 子集敏感性：同一批逐条分数，只换取样口径
    ordered = sorted(set(base["items"]))
    for size in (*subset_sizes, len(ordered)):
        row = {"size": size, "note": "全量" if size == len(ordered) else f"前 {size} 条"}
        for g in loaded:
            ids = [i for i in ordered[:size] if i in g["items"]]
            row[g["label"]] = _mean([g["items"][i] for i in ids])
            row[f"{g['label']}#n"] = len(ids)
        report["subset_sensitivity"].append(row)

    return report


def format_report(report: dict) -> str:
    labels = [g["label"] for g in report["groups"]]
    out = [f"== RAGAS 配对判读（指标：{report['metric']}）==", "", "每轮均值 / 组内重跑极差（judge 噪声地板）："]
    for g in report["groups"]:
        rounds = "、".join(f"{m:.4f}" for m in g["run_means"])
        noise = f"{g['noise_range']:.4f}" if g["noise_range"] is not None else "—（仅一轮）"
        out.append(f"  {g['label']:<12} n={g['n_items']:<3} 均值 {g['mean']:.4f}  各轮 [{rounds}]  极差 {noise}")

    out += ["", "全量同题配对差异："]
    for p in report["paired"]:
        out.append(
            f"  {p['vs']}: {p['mean_diff']:+.4f}"
            f"  95%CI [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"
            f"  赢/输/平 {p['wins']}/{p['losses']}/{p['ties']}"
            f"  符号检验 p={p['sign_test_p']:.4f}"
        )
        drops = "、".join(f"{i}{d:+.2f}" for i, d in p["largest_drops"])
        out.append(f"      单条最大下降：{drops}")

    out += ["", "分题型均值："]
    header = "  " + f"{'题型':<16}" + "".join(f"{lab:>12}" for lab in labels)
    out.append(header)
    for t, row in report["by_type"].items():
        out.append(
            f"  {t:<16}"
            + "".join(
                f"{(row[lab] if row[lab] is not None else float('nan')):>12.4f}" for lab in labels
            )
            + "".join(f"{row[f'{lab}#n']:>4}" for lab in labels)
        )

    out += ["", "子集敏感性（同一批逐条分数，只换取样口径）："]
    for row in report["subset_sensitivity"]:
        out.append(
            f"  {row['note']:<10}"
            + "".join(
                f"{lab} {(row[lab] if row[lab] is not None else float('nan')):.4f}  "
                for lab in labels
            )
        )
    return "\n".join(out)

"""配对判读：RAGAS 轨与检索轨的同题差值（PLAN §5.3 可信度口径）。

动机：只比均值会骗人。本语料的黄金集按题型分块排序，早先 `rows[:15]` 抽样
恰好整段漏掉排在末尾的 cross_doc / time_filter，18pt 的"差异"可能只是抽样偏置。
判读一个消融结论至少要回答三件事：

1. **全量配对差**：同题配对（同问题、不同上下文）的均值差 + 自助法置信区间 + 符号检验；
2. **噪声地板**：同条件重跑的分差——差异小于它就谈不上结论；
3. **子集敏感性**：同样的逐条分数，换成前 N 条会得出什么结论。

两条轨共用同一套判读，区别只在逐条分数从哪读：

- **RAGAS 轨**（`faithfulness` / `answer_relevancy`）：在 `summary.per_item`；
- **检索轨**（`hit_at_*` / `mrr` / `ndcg_at_8` / 覆盖率）：从 `items[]` 的
  `first_hit_rank` / `doc_coverage` / `ndcg_at_8` 派生，分母 = 有 gold 的条目
  （拒答题无 gold，不进检索轨）。派生均值已实测与落盘的 `summary` 逐字一致。

检索轨额外两条护栏，因为这里栽过两次：重排的「nDCG@8 −3.4pt、覆盖率 −1.6pt」
曾经是**两臂清单不等长**（6 格 vs 8 格）造出来的伪影，不是排序质量差。所以只要
两臂的 `n_retrieved`、`n_contexts` 或 `doc_coverage_ceiling` 不同，报告就明说这个
差值不能当质量差读；覆盖率另列一个 `coverage_vs_ceiling`（拿到的 / 本可以拿到的）
作为等长口径。

多重比较：四臂 × 六个指标 = 20 个配对检验，只看 `p<0.05` 迟早出假阳性 → 一个
家族内做 Holm–Bonferroni 校正（`p_holm`）。家族 = **同一次运行**里的全部配对比较，
分几次跑不会互相校正，所以对照臂要一次跑齐。

只用标准库（无 scipy / numpy 依赖）。
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

SUPPORTED_RAGAS = ("faithfulness", "answer_relevancy")

#: 目标轨（逐条分数从 `items[]` 派生、判分不依赖 LLM）可选的指标。
#: 里面混着两个层级，所以不叫「检索轨」：前七个是**检索级**，后两个是**答案级**
#: （它们读的是答案文本，只是和检索级一样躺在 `items[]` 里、走同一套配对判读）。
RETRIEVAL_METRICS = (
    "hit_at_5",
    "hit_at_8",
    "hit_within_budget",
    "mrr",
    "ndcg_at_8",
    "mean_doc_coverage",
    "coverage_vs_ceiling",
    "answered_ok",
    "keypoint_hit_ratio",
)

#: 答案级的那两个：判据是字符串匹配，但读数随答案变，所以块数/清单长度是**实验变量**
#: 而不是伪影（见 `_length_warnings` 的 answer 分支）。
ANSWER_METRICS = ("answered_ok", "keypoint_hit_ratio")
_KEYPOINT_METRIC = "keypoint_hit_ratio"

#: `compare-retrieval` 默认一次跑齐的指标族（`hit_within_budget` 要看清单末再单独点）。
#: 两个答案级指标也在族内：它们只在有判据的条目上有值（聚合题 17 条 / 可答题 64 条），
#: n 更小，但与其余指标同属一次运行、一起进 Holm 家族，所以只会更保守。
RETRIEVAL_SWEEP = (
    "hit_at_5",
    "hit_at_8",
    "mrr",
    "ndcg_at_8",
    "mean_doc_coverage",
    "coverage_vs_ceiling",
    "answered_ok",
    _KEYPOINT_METRIC,
)

_NOISE_LABEL = {
    "retrieval": "检索重跑噪声地板",
    # 答案级：判据本身是字符串匹配（同一答案必得同一分），极差全部来自**答案生成**
    # 的 run-to-run 差异。说成「judge 噪声地板」会把一个不经过 judge 的指标挂上
    # judge 的地板，读的人会以为 1.9pt 那条下限适用于它——不适用。
    "answer": "答案重跑极差（判据确定性，差异只来自生成）",
    "ragas": "judge 噪声地板",
}


def _retrieval_value(metric: str, row: dict) -> float | None:
    """从一条检索结果算出该指标的逐条值；无 gold 的条目返回 None（不进分母）。"""
    if metric == _KEYPOINT_METRIC:
        # 答案级判据：分母是「有 key_points 的条目」，与有没有 gold 文档无关。
        # 旧结果文件没这个键 → None，配对时自然落进「两臂都无值」的第三类，
        # 与 P1-e 那条护栏同一个处置（不假装等长，也不凭空报错）。
        value = row.get("keypoint_hit")
        return None if value is None else float(value)
    if metric == "answered_ok":
        # 答案级：runner 算好的「must_contain 是否逐条齐」（全有或全无）。
        # **它不叫 contains_acc，因为分母不同**：summary 的 `contains_acc` 只在
        # `scorable`（非拒答且有判据）上算 = 64 条；这里能拿到的最大集合是所有
        # `answered_ok` 非 None 的条目（含拒答题，= 72 条）。差值不受影响（两臂
        # 同一分母），但均值不许借那个名字。
        value = row.get("answered_ok")
        return None if value is None else (1.0 if value else 0.0)
    if row.get("doc_coverage") is None:
        return None
    rank = row.get("first_hit_rank")
    if metric.startswith("hit_at_"):
        k = int(metric.removeprefix("hit_at_"))
        return 1.0 if rank and rank <= k else 0.0
    if metric == "hit_within_budget":
        return 1.0 if rank else 0.0
    if metric == "mrr":
        return 1.0 / rank if rank else 0.0
    if metric == "ndcg_at_8":
        value = row.get("ndcg_at_8")
        return None if value is None else float(value)
    if metric == "mean_doc_coverage":
        return float(row["doc_coverage"])
    if metric == "coverage_vs_ceiling":
        ceiling = row.get("doc_coverage_ceiling")
        return None if not ceiling else float(row["doc_coverage"]) / float(ceiling)
    raise ValueError(
        f"未知检索轨指标：{metric}（可选：{', '.join(RETRIEVAL_METRICS)}）"
    )


def _load_retrieval(p: Path, data: dict, metric: str) -> dict:
    rows = data.get("items")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{p.name} 没有 items[]，读不出检索轨的逐条分数")
    items: dict[str, float] = {}
    aux: dict[str, dict] = {}
    types: dict[str, str] = {}
    for row in rows:
        rid = row["id"]
        types[rid] = row.get("type") or ""
        value = _retrieval_value(metric, row)
        if value is None:
            continue
        items[rid] = value
        # 清单长度与覆盖率上限是"这个差值能不能当质量差读"的判据，必须逐条带着走。
        # `k` 是答案级指标的分母（这道题有几个要点）：两臂 K 不同就意味着分母变了，
        # 和「清单不等长」在检索轨上的地位完全一样。
        aux[rid] = {
            "list_len": row.get("n_retrieved"),
            "ctx_len": row.get("n_contexts"),
            "ceiling": row.get("doc_coverage_ceiling"),
            "k": row.get("n_key_points"),
        }
    return {
        "path": p,
        "metric": metric,
        "track": "answer" if metric in ANSWER_METRICS else "retrieval",
        "items": items,
        "types": types,
        "aux": aux,
        "meta": data.get("meta") or {},
        "summary": data.get("summary") or {},
    }


def load_scores(path: str | Path, metric: str | None = None) -> dict:
    """读一个结果文件 → {metric, track, items{id: score}, aux, meta, summary}。

    `metric` 是检索轨键名时走 `items[]`；留空则走 RAGAS 轨。RAGAS 轨兼容两种落盘
    格式：`--ragas-from` 产物（ragas 摘要就是顶层 summary）与 `eval` 直跑产物
    （ragas 摘要嵌在 results["ragas"]，顶层 summary 是客观指标）——三组对照（T4）的
    compare 直接吃 eval 结果文件，不该要求用户再手工拆一份。
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if metric in RETRIEVAL_METRICS:
        return _load_retrieval(p, data, metric)

    summary = data.get("summary") or {}
    if "per_item" not in summary:
        ragas = data.get("ragas") or {}
        if isinstance(ragas, dict) and "per_item" in ragas:
            summary = ragas
        elif isinstance(data.get("items"), list):
            raise ValueError(
                f"{p.name} 是检索轨结果文件（逐条分数在 items[]）——"
                f"请用 compare-retrieval，或传 metric ∈ {RETRIEVAL_METRICS}"
            )
        else:
            raise ValueError(
                f"{p.name} 没有逐条分数（旧版只存均值，需用 --ragas-sample 重跑）"
            )
    if metric and metric not in SUPPORTED_RAGAS:
        raise ValueError(f"{p.name} 未包含指标 {metric}：{summary.get('metrics')}")
    chosen = metric or next(
        (m for m in summary.get("metrics", []) if m in SUPPORTED_RAGAS), None
    )
    if chosen is None:
        raise ValueError(f"{p.name} 未包含可判读指标：{summary.get('metrics')}")
    items = {
        row["id"]: row[chosen]
        for row in summary["per_item"]
        if row.get(chosen) is not None
    }
    # 答案轨的等长护栏：块数来自 per_item 的 n_contexts。旧产物没这个键 → aux 为空，
    # 护栏退化成「不置可否」而不是硬判，与改动前一致（但新产物不会再漏掉这个偏差）。
    aux = {
        row["id"]: {"ctx_len": row.get("n_contexts")}
        for row in summary["per_item"]
        if row.get("n_contexts") is not None
    }
    return {
        "path": p,
        "metric": chosen,
        "track": "ragas",
        "items": items,
        "types": {row["id"]: row.get("type") for row in summary["per_item"]},
        "aux": aux,
        "meta": data.get("meta") or {},
        "summary": summary,
    }


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _bootstrap_ci(
    diffs: list[float], n: int = 5000, seed: int = 42
) -> tuple[float, float]:
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


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni 逐步校正，返回与输入同序的调整后 p 值。

    比 Bonferroni 一样强控制 FWER，但不像它那样把弱效应一律压死——n 个检验里最小
    的 p 乘 n、第二小的乘 n−1……再对前缀取单调化上包络，最后截到 1.0。
    """
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (n - rank) * pvals[idx]))
        adjusted[idx] = running
    return adjusted


def apply_holm(reports: list[dict]) -> dict:
    """就地把校正后的 p 写进每个报告，返回本次判读用的家族说明。"""
    entries = [p for r in reports for p in r["paired"]]
    family = len(entries)
    for entry, adj in zip(
        entries, holm_bonferroni([e["sign_test_p"] for e in entries])
    ):
        entry["p_holm"] = round(adj, 6)
        entry["significant_holm"] = adj < 0.05
    note = {
        "method": "holm-bonferroni",
        "family_size": family,
        "alpha": 0.05,
        "note": "家族 = 本次运行的全部配对比较；分多次运行不会互相校正",
    }
    for r in reports:
        r["correction"] = note
    return note


def _group_items(
    group: dict, metric: str | None = None
) -> tuple[dict, list[float], dict]:
    """把一组内多次重跑聚成逐条均值，返回 (pooled, 每轮总均值, pooled aux)。"""
    per_run = [load_scores(f, metric) for f in group["files"]]
    run_means = [_mean(list(it["items"].values())) for it in per_run]
    ids = sorted(set().union(*[set(it["items"]) for it in per_run]))
    pooled = {
        i: _mean([it["items"][i] for it in per_run if i in it["items"]])
        for i in ids
        if any(i in it["items"] for it in per_run)
    }
    aux: dict[str, dict] = {}
    for it in per_run:
        for rid, meta in it["aux"].items():
            aux.setdefault(rid, meta)
    return pooled, [m for m in run_means if m is not None], aux


def _aux_mean(aux: dict, ids: list[str], key: str) -> float | None:
    vals = [aux[i][key] for i in ids if i in aux and aux[i].get(key) is not None]
    return _mean([float(v) for v in vals])


def _length_warnings(
    base: dict, other: dict, common: list[str], track: str = "retrieval"
) -> list[str]:
    """两臂清单长度 / 上下文块数 / 覆盖率上限不等 → 差值不能整体当质量差读。"""
    warns: list[str] = []
    if track == "answer":
        # 答案级指标只看答案文本：块数不等是 E2 的**实验变量**而不是伪影，清单长度与
        # 覆盖率上限也只影响检索级指标的分母。这个轨上唯一会破坏可比性的是分母 K 变了
        # （同一份黄金集才谈得上配对），所以只报那一条。
        ka = _aux_mean(base["aux"], common, "k")
        kb = _aux_mean(other["aux"], common, "k")
        if ka is not None and kb is not None and abs(ka - kb) > 1e-9:
            warns.append(
                f"要点数 K 不同（{base['label']} {ka:.2f} vs {other['label']} "
                f"{kb:.2f}）→ 分母变了，命中率差不能当质量差读；两臂必须用同一份黄金集"
            )
        return warns
    # 两条轨上「不等长」的后果不是同一件事，措辞必须分开：检索轨是指标分母被长度
    # 改变；答案轨是 faithfulness 天然偏向块数多的一臂（可验证的陈述更多、每条更
    # 容易找到依据）。后者正是 agent 臂必然踩到的那个坑。
    ctx_tail = (
        "→ faithfulness 随上下文变长单调走高，这个差偏向块数多的一臂；"
        "结论要等同块数对照臂（PLAN §5.5 门槛 2）"
        if track == "ragas"
        else "→ hit/nDCG/覆盖率的差部分是长度的函数，不是排序质量"
    )
    for key, label, tail in (
        (
            "list_len",
            "检索清单",
            "→ hit/nDCG/覆盖率的差部分是长度的函数，不是排序质量",
        ),
        ("ctx_len", "进 LLM 的上下文块", ctx_tail),
    ):
        a = _aux_mean(base["aux"], common, key)
        b = _aux_mean(other["aux"], common, key)
        if a is not None and b is not None and abs(a - b) > 1e-9:
            warns.append(
                f"{label}不等长（{base['label']} {a:.2f} vs {other['label']} "
                f"{b:.2f} 格）{tail}"
            )
    ca = _aux_mean(base["aux"], common, "ceiling")
    cb = _aux_mean(other["aux"], common, "ceiling")
    if ca is not None and cb is not None and abs(ca - cb) > 1e-9:
        warns.append(
            f"覆盖率上限不同（{ca:.4f} vs {cb:.4f}）→ 覆盖率的差主要是「清单能装几篇」，"
            "要看同口径的 `coverage_vs_ceiling`"
        )
    return warns


def compare(
    groups: list[dict],
    baseline: int = 0,
    subset_sizes: tuple[int, ...] = (15, 30),
    bootstrap: int = 5000,
    metric: str | None = None,
) -> dict:
    """groups=[{"label": str, "files": [Path, ...]}, ...]；组内多文件 = 同条件重跑。

    `metric` 为检索轨键名时走检索轨（逐条分数来自 `items[]`），留空走 RAGAS 轨。
    """
    loaded = []
    for g in groups:
        items, run_means, aux = _group_items(g, metric)
        loaded.append(
            {"label": g["label"], "items": items, "run_means": run_means, "aux": aux}
        )

    probe = load_scores(groups[0]["files"][0], metric)
    base = loaded[baseline]

    report: dict = {
        "metric": probe["metric"],
        "track": probe["track"],
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
                "list_len_mean": _aux_mean(g["aux"], list(g["items"]), "list_len"),
                "ctx_len_mean": _aux_mean(g["aux"], list(g["items"]), "ctx_len"),
                "ceiling_mean": _aux_mean(g["aux"], list(g["items"]), "ceiling"),
                "k_mean": _aux_mean(g["aux"], list(g["items"]), "k"),
            }
            for g in loaded
        ],
        "paired": [],
        "by_type": {},
        "subset_sensitivity": [],
    }

    # 全量同题配对差
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
                "other_label": g["label"],
                "n_paired": len(common),
                "mean_diff": _mean(diffs),
                "ci95": [lo, hi],
                "wins": wins,
                "losses": losses,
                "ties": len(diffs) - wins - losses,
                "sign_test_p": _sign_test_p(wins, losses),
                "warnings": _length_warnings(base, g, common, track=probe["track"]),
                "largest_drops": sorted(
                    ((i, round(d, 4)) for i, d in zip(common, diffs)),
                    key=lambda t: t[1],
                )[:5],
            }
        )

    # 分题型
    types = probe["types"]
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
        sub_row: dict = {
            "size": size,
            "note": "全量" if size == len(ordered) else f"前 {size} 条",
        }
        for g in loaded:
            ids = [i for i in ordered[:size] if i in g["items"]]
            sub_row[g["label"]] = _mean([g["items"][i] for i in ids])
            sub_row[f"{g['label']}#n"] = len(ids)
        report["subset_sensitivity"].append(sub_row)

    return report


def compare_retrieval(
    groups: list[dict],
    metrics: tuple[str, ...] = RETRIEVAL_SWEEP,
    baseline: int = 0,
    subset_sizes: tuple[int, ...] = (15, 30),
    bootstrap: int = 5000,
) -> list[dict]:
    """一次跑齐指标族并跨指标做 Holm 校正（多臂 × 多指标 = 一个家族）。

    族里不是每个指标都有逐条分数：改动前的结果文件没有 `keypoint_hit`，而非聚合题
    本来也不该有。这类指标**不进 Holm 家族**（没有配对比较就没有多重比较，把它们
    算进分母会白白收紧其余指标的 α），但报告仍然带出来并标成 `unscored` ——
    静默少一个指标，读的人会以为它被测了并且没差异。
    """
    reports = [
        compare(groups, baseline, subset_sizes, bootstrap, metric=m) for m in metrics
    ]
    live = {
        i
        for i, r in enumerate(reports)
        if any(g["n_items"] for g in r["groups"]) and r["paired"]
    }
    apply_holm([reports[i] for i in sorted(live)])
    for i, r in enumerate(reports):
        r["unscored"] = i not in live
    return reports


def format_report(report: dict) -> str:
    if report.get("unscored"):
        return (
            f"== {report['metric']}：本轮无逐条分数，未参与判读与 Holm 校正 ==\n"
            "   （结果文件里没有这个指标的键，或所有条目都缺该判据——"
            "「未测」不等于「测了且无差异」）"
        )
    labels = [g["label"] for g in report["groups"]]
    track = report["track"]
    title = {"retrieval": "检索", "answer": "答案级", "ragas": "RAGAS"}.get(
        track, "检索"
    )
    out = [
        f"== {title} 配对判读（指标：{report['metric']}）==",
        "",
        f"每轮均值 / 组内重跑极差（{_NOISE_LABEL.get(track, _NOISE_LABEL['ragas'])}）：",
    ]
    for g in report["groups"]:
        rounds = "、".join(f"{m:.4f}" for m in g["run_means"])
        noise = (
            f"{g['noise_range']:.4f}" if g["noise_range"] is not None else "—（仅一轮）"
        )
        # 一臂有值另一臂整段缺值时 `live` 判真（any），那臂的均值就是 None——
        # 印成 0.0000 是把「没测」读成「测了且是零」，和 coverage_by_type 那条禁止同源
        mean = f"{g['mean']:.4f}" if g["mean"] is not None else "—（无逐条分数）"
        out.append(
            f"  {g['label']:<12} n={g['n_items']:<3} 均值 {mean}  各轮 [{rounds}]  极差 {noise}"
        )
        lens = []
        if g["list_len_mean"] is not None:
            lens.append(f"清单 {g['list_len_mean']:.2f} 格")
        if g["ctx_len_mean"] is not None:
            lens.append(f"上下文 {g['ctx_len_mean']:.2f} 块")
        if g["ceiling_mean"] is not None:
            lens.append(f"覆盖率上限 {g['ceiling_mean']:.4f}")
        if g.get("k_mean") is not None:
            lens.append(f"要点数 K {g['k_mean']:.2f}")
        if lens:
            out.append(f"      {' · '.join(lens)}")

    out += ["", "全量同题配对差异："]
    for p in report["paired"]:
        holm = (
            f"  p_holm={p['p_holm']:.4f}{'（校正后仍显著）' if p['significant_holm'] else '（校正后不显著）'}"
            if "p_holm" in p
            else ""
        )
        out.append(
            f"  {p['vs']}: {p['mean_diff']:+.4f}"
            f"  95%CI [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"
            f"  赢/输/平 {p['wins']}/{p['losses']}/{p['ties']}"
            f"  符号检验 p={p['sign_test_p']:.4f}{holm}"
        )
        for w in p["warnings"]:
            out.append(f"      ⚠ {w}")
        if p["largest_drops"] and p["largest_drops"][0][1] < -1e-9:
            drops = "、".join(f"{i}{d:+.2f}" for i, d in p["largest_drops"])
            out.append(f"      单条最大下降：{drops}")

    if report.get("correction"):
        c = report["correction"]
        out += [
            "",
            (
                f"多重比较校正：{c['method']}，家族 = {c['family_size']} 个配对比较"
                f"（α={c['alpha']}）；{c['note']}"
            ),
        ]

    out += ["", "分题型均值："]
    header = "  " + f"{'题型':<16}" + "".join(f"{lab:>12}" for lab in labels)
    out.append(header)
    for t, row in report["by_type"].items():
        out.append(
            f"  {t:<16}"
            + "".join(
                f"{(row[lab] if row[lab] is not None else float('nan')):>12.4f}"
                for lab in labels
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

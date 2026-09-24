"""B1：人工复核数据 vs 自动指标的一致率（含 2×2 表、Cohen's κ、分歧清单）。

输入两份文件：
- 人工复核（如 data/eval/human_review_20260922.json）：grades[] 每条 {id, G, F, note}，
  `human_ok = (G == 2)`（rubric：2=可直接用 / 1=可用但需复核 / 0=不可用）；
- 对应的自动评估结果文件（results_*.json）：items[].answered_ok。

输出：一致率、2×2 交叉表、κ、逐条分歧；**限制必须随结论一起报**——那份复核的
judge 字段自述「不是独立第三方」（考卷与判据同一会话写的），一致率只回答
「自动判据与该次人工判定贴不贴」，不回答「判据本身对不对」（那是 U2 的事：
若 κ 低到判据不可信，再请独立第三方盲判 15~20 条）。

运行：
  uv run python scripts/human_review_agreement.py \
    --review data/eval/human_review_20260922.json \
    --run data/eval/results_20260920_224315.json \
    [--out data/eval/human_review_agreement.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float | None:
    """2×2 的 Cohen's κ。分母为 0（一方恒定）时无定义，返回 None。"""
    n = tp + fp + fn + tn
    if n == 0:
        return None
    po = (tp + tn) / n
    yes = tp + fp
    no = fn + tn
    pe = (yes * (tp + fn) + no * (fp + tn)) / (n * n)
    if pe == 1.0:
        return None
    return round((po - pe) / (1 - pe), 4)


def analyze(review: dict, run: dict) -> dict:
    auto_by_id = {it["id"]: it for it in run.get("items", [])}
    rows = []
    skipped = []
    for g in review.get("grades", []):
        item = auto_by_id.get(g["id"])
        if item is None:
            skipped.append({"id": g["id"], "reason": "复核条目不在结果文件里"})
            continue
        auto = item.get("answered_ok")
        if auto is None:
            skipped.append({"id": g["id"], "reason": "自动判据为 None（拒答/无判据）"})
            continue
        rows.append(
            {
                "id": g["id"],
                "type": g.get("type"),
                "human_ok": g["G"] == 2,
                "auto_ok": bool(auto),
                "G": g["G"],
                "F": g.get("F"),
                "note": g.get("note"),
            }
        )
    tp = sum(1 for r in rows if r["human_ok"] and r["auto_ok"])
    fp = sum(1 for r in rows if r["human_ok"] and not r["auto_ok"])
    fn = sum(1 for r in rows if not r["human_ok"] and r["auto_ok"])
    tn = sum(1 for r in rows if not r["human_ok"] and not r["auto_ok"])
    n = tp + fp + fn + tn
    disagreements = [r for r in rows if r["human_ok"] != r["auto_ok"]]
    return {
        "review_meta": {
            "study": review.get("study"),
            "date": review.get("date"),
            "judge": review.get("judge"),
            "source_run": review.get("source_run"),
            "rubric": review.get("rubric"),
        },
        "n_compared": n,
        "n_skipped": len(skipped),
        "skipped": skipped,
        "agreement_rate": round((tp + tn) / n, 4) if n else None,
        "table_2x2": {
            # 行 = 人工（human_ok），列 = 自动（auto_ok）
            "human_ok_auto_ok": tp,
            "human_ok_auto_ng": fp,
            "human_ng_auto_ok": fn,
            "human_ng_auto_ng": tn,
        },
        "cohen_kappa": cohen_kappa(tp, fp, fn, tn),
        "disagreements": disagreements,
        "limitation": (
            "复核人非独立第三方（judge 字段自述：考卷与判据同一会话所写）。"
            "一致率回答「自动判据与该次人工判定贴不贴」，不回答「判据本身对不对」；"
            "κ 低到判据不可信时的动作 = U2 独立第三方盲判 15~20 条，不是调判据。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B1：人工复核 vs 自动指标一致率")
    parser.add_argument(
        "--review",
        default="data/eval/human_review_20260922.json",
        help="人工复核文件（grades[].G/F）",
    )
    parser.add_argument(
        "--run",
        default="data/eval/results_20260920_224315.json",
        help="对应批次的自动评估结果文件",
    )
    parser.add_argument("--out", default=None, help="结果落盘路径（可选）")
    args = parser.parse_args()

    review_path = Path(args.review)
    run_path = Path(args.run)
    if not review_path.exists():
        raise SystemExit(f"复核文件不存在：{review_path}")
    if not run_path.exists():
        raise SystemExit(
            f"结果文件不存在：{run_path}——该批次为 gitignored 历史产物，"
            "若有备份请放回原路径或用 --run 指向现存文件"
        )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    result = analyze(review, run)

    print(f"比较条数 n = {result['n_compared']}（跳过 {result['n_skipped']}）")
    t = result["table_2x2"]
    print(f"一致率 = {result['agreement_rate']} · κ = {result['cohen_kappa']}")
    print(
        f"2×2（行=人工 human_ok，列=自动 answered_ok）："
        f"双可 {t['human_ok_auto_ok']} · 人工可/自动否 {t['human_ok_auto_ng']} · "
        f"人工否/自动可 {t['human_ng_auto_ok']} · 双否 {t['human_ng_auto_ng']}"
    )
    for r in result["disagreements"]:
        print(
            f"  [分歧] {r['id']}（{r['type']}）G={r['G']} auto={r['auto_ok']}：{r['note']}"
        )
    print(f"限制：{result['limitation']}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()

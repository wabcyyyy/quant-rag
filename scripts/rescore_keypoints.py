"""用当前黄金集的要点，离线重判已有结果文件里的 `keypoint_hit`。

为什么这是合法的而不是抄近路：`keypoint_hit` 只由两样东西决定——黄金集里的要点短语，
和结果文件里**已经落盘的答案文本**。答案不依赖要点，所以「改判据后重判」与
「改判据后重新合成一遍」在这一个指标上逐字等价，而前者是 ¥0。

E2 就是这么对齐三臂的：判据在 A 阶段之后又修过一次（源文档把 markdown 转义 `2\\.3`、`\\-`
与 `__…__` 强调带进了正文，它们被当成判据的一部分，答案写「2.3 张果…」就永远不命中；
修好匹配口径后实测救回 A 臂 2 条、25 块臂 5 条），三臂若各自重跑合成要多花一份钱，
而这里一次重判就把它对齐了。

口径边界（别把这个脚本当成通用重算器）：
- 只重算 `keypoint_hit` / `n_key_points` / `answer_grade` 及它们在 summary 里的聚合；
  `contains_acc`、检索指标、延迟、token 用量**原样保留**——它们依赖真正重跑。
- 输出 `<原名>_kc.json`，不动原文件。

用法：`python scripts/rescore_keypoints.py data/eval/results_*.json`
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doc_rag.eval.schema import LEGACY_SUMMARY_KEYS, kp_normalize

GOLD_FILE = "data/eval/gold.json"
# 与 runner._GRADE_FULL_RATIO 同值。这里不复用而是抄一份常量：runner 会连 Qdrant
# 依赖一起 import，而本脚本必须在无服务的环境下可跑。改一处必须改两处，测试锁它。
GRADE_FULL_RATIO = 0.8


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rescore(path: str, gold_file: str = GOLD_FILE) -> str:
    gold = {i["id"]: i for i in _read_json(gold_file)["items"]}
    data = _read_json(path)
    scored: list[dict] = []
    for row in data["items"]:
        points = (gold.get(row["id"]) or {}).get("key_points") or []
        answer = kp_normalize(row.get("answer") or "")
        if not points:
            row["keypoint_hit"] = None
            row["n_key_points"] = None
            row["answer_grade"] = None
            continue
        hits = sum(1 for p in points if kp_normalize(p["phrase"]) in answer)
        k = len(points)
        row["keypoint_hit"] = round(hits / k, 4)
        row["n_key_points"] = k
        row["answer_grade"] = (
            "full"
            if hits >= math.ceil(GRADE_FULL_RATIO * k)
            else ("half" if hits else "zero")
        )
        scored.append(row)

    summary = data["summary"]
    summary["keypoint_recall_macro"] = (
        round(mean(r["keypoint_hit"] for r in scored), 4) if scored else None
    )
    by_type: dict[str, list[float]] = {}
    for r in scored:
        by_type.setdefault(r["type"], []).append(r["keypoint_hit"])
    summary["keypoint_recall_by_type"] = {
        t: round(mean(v), 4) for t, v in sorted(by_type.items())
    }
    summary["keypoint_n_items"] = len(scored)
    ks = [r["n_key_points"] for r in scored]
    summary["keypoint_k"] = {
        "min": min(ks) if ks else None,
        "max": max(ks) if ks else None,
        "mean": round(sum(ks) / len(ks), 2) if ks else None,
    }
    grades = {"full": 0, "half": 0, "zero": 0}
    for r in scored:
        grades[r["answer_grade"]] += 1
    summary["answer_grade_counts"] = grades
    # 旧键从 runner 那张表派生，和 eval 直跑写的是同一份映射——两处各写一遍就会漂移
    for legacy, canonical in LEGACY_SUMMARY_KEYS.items():
        if canonical in summary:
            summary[legacy] = summary[canonical]
    # 自证这份文件是重判出来的，而不是合成时算出来的：两件事的判据版本可能不同
    summary["keypoint_rescored_from"] = str(path)

    out = Path(path).with_name(Path(path).stem + "_kc.json")
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(
        f"{Path(path).name}: n={len(scored)} mean={summary['keypoint_recall_macro']} "
        f"grades={grades} -> {out.name}"
    )
    return str(out)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        rescore(arg)

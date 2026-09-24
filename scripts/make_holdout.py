"""B9：外部 holdout 脚手架（U1 的另一半——代理只做格式，不代编问题、不代打分）。

输入：外部人提的问题清单（纯文本，一行一问）。**问题必须来自外部**（SPEC U1：
把虚构语料给非作者本人，「你想知道什么就问」）——代理绝不代写问题，那会把
「无用户、无需求」这个唯一的断裂缝用自问自答糊上。

产出：
- gold 文件（origin: external）：无 must_contain、无 source_doc_ids——外部问题
  没有程序化判据，检索/答案自动指标对它**结构性为空**，这本身就是设计：
  holdout 的判分是人工 rubric（G 可用性 2/1/0、F 编造 0/1），不是自动化指标；
- 空白评分卷：判分人先写 G/F，再看任何自动输出（协议照 human_review 先例）。

之后：`doc-rag eval --gold <holdout gold> --holdout` 把 meta.holdout=true 落盘，
结果文件与调参集可溯源地区分（holdout 冻结后永不进调参集）。

运行：uv run python scripts/make_holdout.py questions.txt [--out gold_holdout.json]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WORKSHEET_RUBRIC = """\
# Holdout 人工评分卷（{n} 条外部问题）

> 协议（照人工复核先例）：先逐条把 G/F 填完，再打开任何自动评估输出；
> G 判「答案对所问是否可用」，F 判「有没有引用块里找不到却以确定语气写出的主张」。

rubric：
- G 可用性：2 = 可直接用（答到所问、无可核对错误）；1 = 可用但需复核
  （有可核对的小错、答得绕、掺了没问的内容）；0 = 不可用（答错/答非所问/该答未答）
- F 编造：1 = 有（引用块里找不到、却以确定语气写出）；0 = 无

| # | id | 问题 | G | F | 备注 |
|---|----|------|---|---|------|
"""


def make_holdout(lines: list[str], source_note: str) -> tuple[dict, str]:
    items = []
    worksheet_rows = []
    for i, q in enumerate(lines, 1):
        qid = f"h{i:03d}"
        items.append(
            {
                "id": qid,
                "type": "holdout",
                "question": q,
                "expected_answer": "",
                "must_contain": [],
                "source_doc_ids": [],
                "refusable": False,
                "source_title": None,
                "origin": "external",
            }
        )
        worksheet_rows.append(f"| {i} | {qid} | {q} |  |  |  |")
    gold = {
        "meta": {
            "gold_version": f"holdout-{datetime.now(tz=UTC).date().isoformat()}",
            "sample": True,
            "holdout": True,
            "note": (
                "外部 holdout（U1）：问题由非作者的外部人提出（origin=external），"
                "判分走人工 rubric（G 2/1/0、F 0/1），无程序化判据——"
                "自动指标对本集结构性为空是设计而非缺陷。冻结后永不进调参集。"
                + (f" 来源：{source_note}。" if source_note else "")
            ),
            "count": len(items),
        },
        "items": items,
    }
    worksheet = WORKSHEET_RUBRIC.format(n=len(items)) + "\n".join(worksheet_rows) + "\n"
    return gold, worksheet


def main() -> None:
    parser = argparse.ArgumentParser(description="B9：外部 holdout 脚手架")
    parser.add_argument("questions", help="外部问题清单（纯文本，一行一问）")
    parser.add_argument("--out", default="data/eval/gold_holdout.json")
    parser.add_argument("--worksheet", default="data/eval/holdout_worksheet.md")
    parser.add_argument(
        "--source-note", default="", help="问题来源备注（如「外部人 X，2026-09-25」）"
    )
    args = parser.parse_args()

    qpath = Path(args.questions)
    if not qpath.exists():
        raise SystemExit(f"问题清单不存在：{qpath}")
    lines = [
        ln.strip()
        for ln in qpath.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if not lines:
        raise SystemExit(
            "问题清单为空——holdout 至少要 1 条外部问题（U1 红线：代理不代写）"
        )
    gold, worksheet = make_holdout(lines, args.source_note)
    out = Path(args.out)
    out.write_text(
        json.dumps(gold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    Path(args.worksheet).write_text(worksheet, encoding="utf-8")
    print(f"holdout gold：{len(lines)} 条（origin=external）→ {out}")
    print(f"空白评分卷 → {args.worksheet}")
    print("下一步：拿语料回答这些题（doc-rag eval --holdout），再把评分卷交人工填写")


if __name__ == "__main__":
    main()

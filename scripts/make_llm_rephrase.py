"""LLM 措辞鲁棒性臂：gold_core 的程序化问题 → LLM 口语化重写（判据原样继承）。

诚实边界（引用这批数字前必须带上）：
- **这不是外部锚点**。问题由 ZCode 子代理（GLM）生成，与语料、判据同生态——买不到
  「真人提问分布」，买的是**措辞分布**：gold_core 的题面带着《文档标题》这类
  「知道答案在哪」的程序化伪影（检索等于吃了 title 提示），重写臂在防锚定输入
  （代理看不到原题）下去掉这个提示，量的是「换个问法掉多少」。
- **判据零 LLM**：must_contain / key_points / source_doc_ids 逐字段继承 gold_core，
  origin 标 `llm_rephrase`。同一事实两种问法 → 配对可比（compare-retrieval 直接吃）。
- 数量弥补不了来源单一；生成协议（两 persona、硬规则、逐条产物）与出处见
  data/eval/gold_core_llm_map.json 的 meta——该文件与产物同批入库，作为出处凭证。

输入：gold json + 若干 chunk 文件（子代理产物，每项 {"id","r1","r2"}）。
输出：gold json（每项 ×2 变体，id 加 r1/r2 后缀，原序交错）。
校验（违反即整批拒绝，不落盘半成品）：
  ① chunk 对 gold 全覆盖——缺 id 或重复 id 都报错并列出；
  ② 变体非空、r1≠r2、变体≠原题（防偷懒复读）；
  ③ 每条产物过 GoldItem（eval 消费的同一契约）。

运行：
  uv run python scripts/make_llm_rephrase.py --gold data/eval/gold_core.json \
      --out data/eval/gold_core_llm.json data/eval/_rephrase_work/chunk1_output.json \
      data/eval/_rephrase_work/chunk2_output.json data/eval/_rephrase_work/chunk3_output.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_rag.eval.schema import GoldItem


def load_chunks(paths: list[Path]) -> dict[str, tuple[str, str]]:
    """合并 chunk 文件 → {id: (r1, r2)}；重复 id 直接报错。

    接受两种形态：裸数组（子代理产物 chunk），或 {"meta":…, "variants":[…]}
    （gold_core_llm_map.json 出处文件——只消费 variants，meta 原样留在出处里）。
    """
    variants: dict[str, tuple[str, str]] = {}
    for p in paths:
        rows = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows["variants"]
        for row in rows:
            qid = row["id"]
            if qid in variants:
                raise ValueError(f"chunk 重复 id: {qid}（{p.name} 与先前文件冲突）")
            variants[qid] = (row["r1"], row["r2"])
    return variants


def merge(gold: dict, variants: dict[str, tuple[str, str]]) -> dict:
    """gold × 变体 → 交错产物；三道校验全过才返回。"""
    errors: list[str] = []
    missing = [i["id"] for i in gold["items"] if i["id"] not in variants]
    if missing:
        errors.append(f"chunk 未覆盖 {len(missing)} 条: {missing}")
    extra = sorted(set(variants) - {i["id"] for i in gold["items"]})
    if extra:
        errors.append(f"chunk 多出未知 id: {extra}")

    out_items: list[dict] = []
    for item in gold["items"]:
        if item["id"] not in variants:
            continue
        r1, r2 = variants[item["id"]]
        for suffix, q in (("r1", r1), ("r2", r2)):
            if not q or not q.strip():
                errors.append(f"{item['id']}{suffix}: 变体为空")
            if q.strip() == r1.strip() and suffix == "r2" and q.strip():
                errors.append(f"{item['id']}: r1 与 r2 相同")
            if q.strip() == item["question"].strip():
                errors.append(f"{item['id']}{suffix}: 与原题逐字相同（防偷懒复读）")
            new_item = {
                **item,
                "id": f"{item['id']}{suffix}",
                "question": q.strip(),
                "origin": "llm_rephrase",
            }
            GoldItem(**new_item)  # eval 消费契约，此处只做校验
            out_items.append(new_item)
    if errors:
        raise ValueError("重写产物校验失败:\n  " + "\n  ".join(errors))

    type_dist = Counter(i["type"] for i in out_items)
    src_version = gold["meta"]["gold_version"]
    return {
        "meta": {
            "gold_version": f"{src_version}_llm",
            "sample": gold["meta"].get("sample", True),
            "note": (
                f"LLM 措辞鲁棒性臂：{src_version} 每条 ×2 口语化变体（origin=llm_rephrase）。"
                "问题措辞由 ZCode 子代理生成（协议与出处见 gold_core_llm_map.json），"
                "判据零 LLM、逐字段继承源集。它不是外部锚点——引用读数必须带"
                "「LLM 生成措辞」，禁止当『真实用户分布』讲。"
            ),
            "count": len(out_items),
            "type_distribution": dict(type_dist),
            "derived_from": f"{src_version}（{gold['meta'].get('count', '?')} 条 × 2 变体）",
        },
        "items": out_items,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("chunks", type=Path, nargs="+", help="子代理产物 chunk 文件")
    args = ap.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    variants = load_chunks(args.chunks)
    payload = merge(gold, variants)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    by_type = Counter(i["type"] for i in payload["items"])
    print(f"写出 {args.out}: {payload['meta']['count']} 条（{dict(by_type)}）")


if __name__ == "__main__":
    main()

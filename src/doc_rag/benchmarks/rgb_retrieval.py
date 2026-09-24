"""RGB 的**检索变体**：把数据集文档并成一份语料，让本系统自己检索再回答。

## 为什么要这一层

`rgb.py` 那条口径是 RGB 的官方协议：**文档由数据集提供**，检索层不参与。所以它测到的
只有「prompt + 模型选型」——本项目的分块、Dense+BM25 混合、RRF 融合、重排、上下文预算
在那份读数里**一分都没体现**。而「这个 RAG 做得好不好」恰恰要看那一层。

这一层的做法：把该数据集所有文档（正面 + 噪声）去重成一份语料灌进 Qdrant，然后
**跑本仓库真实的 `eval` 命令**（`answer()` → 改写 → 混合检索 → 重排 → 上下文预算 →
合成）。于是指标是同一套已验过口径的东西：检索侧的 Hit@k / Recall@k / nDCG / MRR /
覆盖率，答案侧的 `strict_keyword_accuracy`。

## 判据等价性（为什么这两个数字可以直接对比）

`eval/runner.py:349` 的规则是 `all(must_contain 都在答案里)`，与 RGB 官方
`checkanswer` 的「全部 ground-truth 元素命中」是**同一条规则**。所以把 RGB 的
`answer` 逐项填进 `must_contain` 之后：

    本仓库的 strict_keyword_accuracy  ≡  RGB 的 all_rate

**这不是「另一套指标」，是同一个判据在两条路径上跑**——所以「自己检索」与「给定文档」
两行的差，就是**检索层贡献的那部分**。这正是要看的东西。

## 与官方协议的差异（报告里必须声明）

- 官方是给定文档；这里是**自己检索**。所以这一行**不能**与论文基线并排比大小
  （论文没有这个设置），只能与「给定文档」那一行自比。
- 官方没有 `noise_rate` 这个旋钮；这里对应的是**检索条数 / 上下文预算**（k）。
- 拒答那一族（`noise_rate=1`）在这个设置下没有对应物：语料里明明有答案文档，
  「该拒答」就无从构造。所以检索变体只覆盖噪声鲁棒与信息整合两族。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import rgb

#: 数据集 → 项目 gold 的题型标签。`zh_int` 是「两组文档各答一半」= 跨文档聚合。
GOLD_TYPE: dict[str, str] = {"zh": "fact", "zh_refine": "fact", "zh_int": "cross_doc"}

#: 每个数据集一份独立语料（照官方「每个数据集自带文档池」的形状）
COLLECTION: dict[str, str] = {
    "zh": "rgb_zh",
    "zh_refine": "rgb_zh_refine",
    "zh_int": "rgb_zh_int",
}


def passage_doc_id(text: str) -> str:
    """内容寻址：与 ingest 的 `sha256(源文件)[:16]` 同一约定。

    同一段文字在不同题里既是正面又是噪声时只会有一篇——这也让「gold 里写的
    source_doc_ids」与库里真实的点一一对应。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _docs_of(rec: rgb.Record) -> list[str]:
    """该题涉及的全部文档（正面 + 噪声）。`zh_int` 的 positive 是「每组一个列表」。"""
    raw = rec.raw
    out: list[str] = []
    pos = raw.get("positive") or []
    if pos and isinstance(pos[0], list):
        out.extend(d for group in pos for d in group)
    else:
        out.extend(pos)
    out.extend(raw.get("negative") or [])
    return out


def build_corpus(records: list[rgb.Record]) -> dict[str, str]:
    """去重后的语料：`{doc_id: 正文}`。顺序按 doc_id 排序 → 可复现。"""
    corpus: dict[str, str] = {}
    for rec in records:
        for text in _docs_of(rec):
            corpus.setdefault(passage_doc_id(text), text)
    return dict(sorted(corpus.items()))


def write_parsed(corpus: dict[str, str], out_dir: Path) -> int:
    """把语料写成 ingest 认的**统一中间 JSON**（一篇一个文件）。

    走中间表示而不是直接 upsert：这样分块、嵌入、入库全部是**真实链路**，
    检索变体测到的分块策略与生产一致。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, text in corpus.items():
        payload = {
            "meta": {
                "source_type": "rgb",
                "doc_id": doc_id,
                # RGB 的段落没有标题；给一个可追溯的编号而不是编一个标题
                "title": f"RGB文档 {doc_id[:8]}",
            },
            "blocks": [{"type": "paragraph", "text": text}],
        }
        (out_dir / f"{doc_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    return len(corpus)


def build_gold(
    records: list[rgb.Record], dataset: str, *, limit: int | None = None
) -> dict[str, Any]:
    """把 RGB 的题转成本仓库的黄金集格式。

    `must_contain` 直接填 RGB 的 `answer` 列表：因为两侧的判据是同一条规则
    （全部命中），所以填进去之后 `strict_keyword_accuracy` 就是 RGB 的 `all_rate`。
    """
    items: list[dict[str, Any]] = []
    for rec in records[:limit] if limit else records:
        raw = rec.raw
        answer = raw["answer"]
        must = [str(a) for a in (answer if isinstance(answer, list) else [answer])]
        pos = raw.get("positive") or []
        pos_docs = (
            [d for group in pos for d in group]
            if pos and isinstance(pos[0], list)
            else list(pos)
        )
        items.append(
            {
                "id": rec.id,
                "type": GOLD_TYPE[dataset],
                "question": rec.query,
                "expected_answer": " / ".join(must),
                "must_contain": must,
                "source_doc_ids": sorted({passage_doc_id(d) for d in pos_docs}),
                "refusable": False,
                "origin": "rgb",
            }
        )
    return {
        "meta": {
            "gold_version": f"rgb-{dataset}-retrieval",
            "note": (
                "由 RGB（CC BY-NC-SA 4.0）派生，仅本地评测用，不再分发。"
                "must_contain 直接取 RGB 的 answer——两侧判据是同一条「全部命中」规则，"
                "所以 strict_keyword_accuracy ≡ RGB 的 all_rate。"
            ),
            "source": {
                "repo": rgb.UPSTREAM_REPO,
                "commit": rgb.UPSTREAM_COMMIT,
                "dataset": dataset,
                "collection": COLLECTION[dataset],
            },
            "count": len(items),
        },
        "items": items,
    }


def write_gold(gold: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")

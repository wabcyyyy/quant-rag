"""拒答答案的「上下文外断言」审计（PLAN §5.3 拒答口径）。

现有拒答判分只查措辞：`_refusal_ok` 命中 12 个词之一就算正确拒答。而 RAGAS 轨按定义
把拒答题排除在 faithfulness 之外（正确拒答的套话与 hedge 天然不可验证，实测会被结构性
扣分）。两条轨合起来的结果是：**8 条拒答题没有任何一条被检查过「里面有没有编造」**。
最极端的一类不是短句，而是几百字、带 `[1][2]` 引用、还转述了实质内容的「拒答」。

这里只问一个二元问题：这条拒答有没有把上下文里没记载的东西当成事实讲出来。
判定所用的上下文直接取结果文件里落盘的那份（`[n]（文档名 第p页）` 前缀版，与 LLM
当时看到的逐字一致），所以**审计不需要 Qdrant，也不需要重新检索**——只要结果文件在。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from doc_rag.config import load_config
from doc_rag.eval.judge import judge_cfg, judge_is_cross_vendor
from doc_rag.generate.llm import chat_timed

PROMPT_VERSION = "refusal-audit-v1"

_SYSTEM = (
    "你在审计一套企业文档问答系统的拒答答案。系统被要求：文档里没有依据时必须明说答不了。\n"
    "只做一个判断：这条回答有没有把**上下文里没有记载的内容**当成事实讲出来"
    "（具体数字、日期、决议、人名、承诺、因果关系）。\n"
    "只输出一个 JSON 对象，不要多余文字：\n"
    '{"fabricated": true, "quote": "回答里最像编造的那句原文，没有则空串",'
    ' "why": "一句话说明上下文撑不住这句的理由"}\n'
    "注意三件事：① 回答里出现「无法回答/没有记载」这类措辞本身不算编造；"
    "② 回答只是复述问题里的词、或明确说「不知道」，不算编造；"
    "③ 如果它一边说答不了、一边又转述了具体结论，逐字对照上下文判那条结论有没有依据。"
)

_USER = """【问题】
{question}

【系统当时看到的上下文（逐字，含 [n]（文档名 第p页） 前缀）】
{contexts}

【待审回答】
{answer}
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def refusal_rows(data: dict) -> list[dict]:
    """结果文件里需要审计的条目：无答案题（refusable）且真给出了回答。"""
    return [
        row
        for row in data.get("items", [])
        if row.get("type") == "no_answer" and row.get("answer")
    ]


def _parse_verdict(text: str) -> dict[str, Any]:
    """judge 偶尔在 JSON 前后带一句寒暄，取第一个花括号块。"""
    match = _JSON_RE.search(text)
    if not match:
        return {"fabricated": None, "error": f"判分输出里没有 JSON：{text[:120]}"}
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {"fabricated": None, "error": f"JSON 解析失败：{exc}；{text[:120]}"}
    return {
        "fabricated": bool(raw.get("fabricated")),
        "quote": str(raw.get("quote") or ""),
        "why": str(raw.get("why") or ""),
    }


def _usable_contexts(row: dict) -> bool:
    """落盘上下文必须是 `[n]（文档名 第p页）` 那份。

    旧格式只存正文，用它判会把答案里的归属陈述当成无据断言——那是当年 faithfulness
    被低估 32pt 的同一个 bug，别在审计里再犯一次。
    """
    contexts = row.get("contexts") or []
    return bool(contexts) and str(contexts[0]).startswith("[1]")


def compare_audits(prev: dict, new: dict) -> dict:
    """两份审计的逐条判读差异（同一批落盘答案、不同 judge 或不同 judge 配置）。

    只比双方都判出了结论的条目——一方跳过/判分失败的不能算「一致」。
    """
    a = {i["id"]: i for i in prev["items"] if i.get("fabricated") is not None}
    b = {i["id"]: i for i in new["items"] if i.get("fabricated") is not None}
    common = sorted(set(a) & set(b))
    flips = [
        {
            "id": rid,
            "prev": a[rid]["fabricated"],
            "new": b[rid]["fabricated"],
            "new_quote": b[rid].get("quote") or a[rid].get("quote") or "",
            "new_why": b[rid].get("why") or a[rid].get("why") or "",
        }
        for rid in common
        if a[rid]["fabricated"] != b[rid]["fabricated"]
    ]
    return {
        "compared": {
            "prev_judge": prev["meta"]["judge_model"],
            "new_judge": new["meta"]["judge_model"],
            "prev_prompt": prev["meta"]["prompt_version"],
            "new_prompt": new["meta"]["prompt_version"],
        },
        "n_both_judged": len(common),
        "n_only_prev": sorted(set(a) - set(b)),
        "n_only_new": sorted(set(b) - set(a)),
        "agreement": round(1 - len(flips) / len(common), 4) if common else None,
        "prev_rate": prev["fabrication_rate"],
        "new_rate": new["fabrication_rate"],
        "flips": flips,
    }


def audit_results(
    results_file: Path,
    cfg: dict | None = None,
    limit: int | None = None,
    **judge_over: str,
) -> dict:
    """对一份 eval 结果文件里的拒答答案做二元审计。串行调用（n 只有个位数）。

    `judge_over` 传 `model` / `base_url` / `api_key` 就是一次跨供应商复判；不传则用
    配置里的 judge（默认继承生成侧模型，即同源判分）。
    """
    cfg = cfg or load_config()
    data = json.loads(Path(results_file).read_text(encoding="utf-8"))
    rows = refusal_rows(data)
    if limit:
        rows = rows[:limit]
    llm_cfg = judge_cfg(cfg, **{k: v for k, v in judge_over.items() if v})

    items: list[dict] = []
    for row in rows:
        if not _usable_contexts(row):
            items.append(
                {
                    "id": row["id"],
                    "fabricated": None,
                    "answer_len": len(row.get("answer") or ""),
                    "skipped": "落盘上下文不是 LLM 实际看到的那份（缺 [n]（文档名 第p页）前缀）",
                }
            )
            continue
        prompt = _USER.format(
            question=row["question"],
            contexts="\n\n".join(row["contexts"]),
            answer=row["answer"],
        )
        text, meta = chat_timed(
            llm_cfg, prompt, system_prompt=_SYSTEM, temperature=llm_cfg["temperature"]
        )
        verdict = _parse_verdict(text)
        items.append(
            {
                "id": row["id"],
                "answer_len": len(row["answer"]),
                "n_citations": row.get("n_citations"),
                "refusal_word_hit": row.get("answered_ok"),
                "ms": meta.get("ms"),
                "cached": meta.get("cached"),
                **verdict,
            }
        )

    judged = [i for i in items if i.get("fabricated") is not None]
    fabricated = [i for i in judged if i["fabricated"]]
    return {
        "meta": {
            "results_file": str(results_file),
            "eval_timestamp": (data.get("meta") or {}).get("timestamp"),
            "judge_model": llm_cfg["model"],
            # 只落 host，不落 key：跨供应商复判的证据要能说明是哪一家判的
            "judge_base_url": (llm_cfg.get("base_url") or "")
            .split("//")[-1]
            .split("/")[0],
            "judge_cross_vendor": judge_is_cross_vendor(cfg, **judge_over),
            "reasoning_effort": llm_cfg.get("reasoning_effort"),
            "prompt_version": PROMPT_VERSION,
            "temperature": llm_cfg["temperature"],
            "n_refusable_with_answer": len(rows),
            "n_judged": len(judged),
            "n_skipped": len(items) - len(judged),
        },
        "fabrication_rate": (
            round(len(fabricated) / len(judged), 4) if judged else None
        ),
        "items": items,
    }

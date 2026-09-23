"""两段式合成（ADR-0002 / UPGRADE_2026 §3）：聚合题 map（逐篇微摘要）→ reduce（聚合合成）。

动机（ADR-0002 的三条已量事实）：C25 直喂 25 块拿到 +26.7pt 但 3.4 倍输入破 SLO；
agent policy 层与 A6 不可区分还多花 2.2 倍；只动上下文预算是半个杠杆。
两段式让 reduce 只吃逐篇微摘要（token 可控），map 并行可缓存，是唯一有机会
同时拿到质量与 SLO 的路线。

四条边界（沿 agent.py / synthesizer.py 的写法）：

1. 路由键 = **预测**题型（`synthesizer.PREDICTED_TYPES`），与 agent 门控、思考档
   同一个信号源；真题型只做事后核对。单发口径逐字不动：`enabled: false` 时
   本模块的任何函数都不会被走到。
2. `item.contexts` 仍存**原始块**，微摘要存独立字段（`summaries`），
   引用编号 [n] 指原始块的序号——`citation_valid = 1 ≤ n ≤ len(contexts)`
   的现行实现零改动即用，`n_contexts` 与历史臂可比。
3. 重放纪律：reduce 的输入可从落盘 `summaries` + 原始块串逐字重建
   （`rebuild_summary_context`）；缺记录值拒绝重放（eval/runner 的重放守卫）。
4. 退化三级：单篇微摘要失败 → 该篇降级用原文首块；单题失败率超 `fail_ratio`
   → 整题退回单发口径并计数（`synthesis_route = two_stage_fallback`）。
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import llm
from .synthesizer import PREDICTED_TYPES, resolve_effort_cfg

# 与 agent.DEFAULT_TYPES 同一组：聚合两类的命中率问题（清单格数封顶）只在它们身上
DEFAULT_TYPES = ("cross_doc", "time_filter")


def two_stage_cfg(cfg: dict) -> dict:
    """`synthesis.two_stage` 段的读取入口：一处集中默认值。

    填错键（`types: [term]`）在这里就炸，而不是线上悄悄不生效——
    与 `Synthesizer` 对思考档表、`agent` 对题型键的处置一致。
    """
    raw = (cfg.get("synthesis") or {}).get("two_stage") or {}
    types = tuple(raw.get("types") or DEFAULT_TYPES)
    unknown = sorted(t for t in types if t not in PREDICTED_TYPES)
    if unknown:
        raise ValueError(
            f"synthesis.two_stage.types 含服务侧预测不到的题型：{unknown}"
            f"（可用键：{list(PREDICTED_TYPES)}）。fact/term 是黄金集标注，"
            "线上不可得；路由只能建在改写预测出来的信号上。"
        )
    return {
        "enabled": bool(raw.get("enabled")),
        "types": types,
        # map 段自己的预算：可退化的步骤不该有合成侧 180s×4 的等待能力（W3 教训）
        "timeout_s": float(raw.get("timeout_s") or 20.0),
        "max_attempts": int(raw.get("max_attempts") or 1),
        # 单题微摘要失败率超过它 → 整题退回单发口径（宁退不猜）
        "fail_ratio": min(1.0, max(0.0, float(raw.get("fail_ratio") or 0.3))),
        # map 段并行度：照 ingest/indexer 的 ThreadPoolExecutor(8) 先例
        "workers": max(1, int(raw.get("workers") or 8)),
    }


def is_routed(cfg: dict, predicted: str) -> bool:
    ts = two_stage_cfg(cfg)
    return ts["enabled"] and predicted in ts["types"]


# ── map 段 prompt ─────────────────────────────────────────────────────────────
# 与 agent.SYSTEM_EVIDENCE 同一性质：prompt 随代码走，不进 prompt 版本注册表
# （那张表管的是单发合成侧的基线口径；两段式自己的 prompt 由 parity 测试锁）。

SYSTEM_MAP = """\
你在为跨文档聚合问答做逐篇微摘要。给你一条问题和某篇文档的全部相关块，只输出该篇中
与问题直接相关的决定/事实：逐字保留人名、数值、日期、文号，一条一行，总共不超过 150 字。
该篇与问题无关时，只输出两个字：无关。除此之外不要输出任何其他内容。
"""

_USER_MAP = """【问题】
{question}

【文档】{doc}

【该篇相关块（[n] 是原始块编号，共 {n} 块）】
{blocks}

只输出微摘要；该篇与问题无关则只输出：无关"""

# ── reduce 段 prompt：在单发合成 prompt 基础上改「上下文为逐篇摘要」──────────
# 合成规则 3 **不放宽**（ADR-0002 / F1）：逐篇枚举陈述，禁止跨篇合并——
# 这里显式重申，另由 audit-refusals 式抽查兜底。

SYSTEM_ANSWER_TWO_STAGE = """\
你是企业文档问答助手。下方上下文不是原文，而是逐篇微摘要：每条摘要以它覆盖的
原始上下文块编号开头（如 [3][4]），摘要正文是该篇与问题相关的决定/事实。规则：
1. 每一句事实陈述都必须在句末标注**原始块编号**（摘要开头方括号里的编号），
   格式为 [3] 或 [3][4]。没有编号的事实陈述视为无效；不要给摘要编新号。
2. 回答前先在摘要中查找依据。只要摘要包含回答问题所需的信息（即使措辞与问题不同、
   需要你自己归纳），就必须作答并标注来源；信息不完整但已包含答案时，也要回答已知部分。
3. 禁止推断、补全、概括：不得把摘要没有写出的内容写成结论。**逐篇枚举陈述**：
   不得把多篇文档的信息合并成任何单篇文档的摘要都不支持的陈述。
4. 严格区分「会上决定」与「会上讨论过但未决定」——按摘要措辞表述，不得把讨论升级为决议。
5. 仅当全部摘要确实不含回答所需信息时才拒答，回答"根据现有文档无法回答"并说明缺少什么
   信息（此句无需编号）。不要因为信息分散在多篇、或需要归纳而拒答。
6. 摘要相互矛盾时分别陈述，并各自标注来源编号。
7. 涉及日期、参会人、数值、文号时逐字引用摘要中的原词。
"""

USER_ANSWER_TWO_STAGE = """\
上下文（逐篇微摘要，方括号编号为原始上下文块号）：
{context}

问题：{question}
"""

# 单发上下文串的前缀形状：`[3] （文档名 第p页）`（见 prompts.format_context）。
# 重放侧从存档串里剥掉编号拿回定位前缀，两段必须逐字对得上。
_LOC_RE = re.compile(r"^\[\d+\]\s?")


def _loc_of(context_line: str) -> str:
    return _LOC_RE.sub("", context_line, count=1)


def format_summary_context(summaries: list[dict], contexts: list[dict]) -> str:
    """reduce 的上下文段：每篇一条摘要，前缀 = 它覆盖的**原始块**编号 + 定位。

    `contexts` 是 Orchestrator.Result.contexts（原始块，编号 1 起）。摘要携带
    `source_context_idx`（1 起的原始块编号），定位前缀取自**首个**编号那块的
    `doc`/`page`——与单发 `format_context` 的前缀形状逐字同构，单篇覆盖时两者
    完全相同。降级摘要（原文首块）走同一条路径，reduce 无需知道它降级过。
    """
    parts = []
    for s in summaries:
        idxs = list(s["source_context_idx"])
        first = contexts[idxs[0] - 1]
        loc = (
            f"（{first['doc']}"
            + (f" 第{first['page']}页" if first.get("page") else "")
            + "）"
        )
        prefix = "".join(f"[{i}]" for i in idxs)
        parts.append(f"{prefix} {loc}\n{s['text']}")
    return "\n\n".join(parts)


def rebuild_summary_context(summaries: list[dict], context_strings: list[str]) -> str:
    """重放侧的唯一入口：从落盘摘要 + 存档的原始块串逐字重建 reduce 的上下文段。

    与 `format_summary_context` 逐字一致（编号拼接 + 定位前缀两个来源分别是
    摘要里的 `source_context_idx` 与存档串首行），否则 judge 看到的就不是
    LLM 当时看到的那份。`contexts` 是 `_judge_contexts` 存下的字符串列表。
    """
    parts = []
    for s in summaries:
        idxs = list(s["source_context_idx"])
        loc = _loc_of(context_strings[idxs[0] - 1].splitlines()[0])
        prefix = "".join(f"[{i}]" for i in idxs)
        parts.append(f"{prefix} {loc}\n{s['text']}")
    return "\n\n".join(parts)


def _map_llm_cfg(cfg: dict, ts: dict) -> dict:
    """map 段的调用配置：继承合成模型，但关思考 + 自己的短预算。

    思考关死不看思考档表：表里聚合两类本来就是 none，但 map 在 `types` 可配的
    前提下不能假设键域——微摘要是分类性质的短输出，reasoning 只烧钱不改结果
    （judge / 改写 / agent 判定三处同一结论）。
    """
    out = dict(cfg["llm"])
    out["reasoning_effort"] = "none"
    out["timeout_s"] = ts["timeout_s"]
    out["max_attempts"] = ts["max_attempts"]
    return out


def _group_by_doc(contexts: list[dict]) -> list[dict]:
    """把上下文块按文档归组，保持首次出现顺序（编号即原始块号，不许重排）。"""
    groups: dict[str, dict] = {}
    order: list[str] = []
    for c in contexts:
        key = str(c.get("doc_id") or c.get("doc"))
        if key not in groups:
            groups[key] = {
                "doc_id": c.get("doc_id"),
                "doc": c.get("doc"),
                "page": c.get("page"),
                "blocks": [],
            }
            order.append(key)
        groups[key]["blocks"].append(c)
    return [groups[k] for k in order]


def map_summaries(
    cfg: dict, question: str, contexts: list[dict]
) -> tuple[list[dict] | None, dict]:
    """逐篇微摘要。返回 (summaries, meta)；失败率超阈值时 summaries 为 None。

    summaries 元素：`{doc_id, doc, text, source_context_idx, degraded}`——
    逐条落盘供重放（`rewritten` / trace 同一条纪律）。单篇失败 → 该篇降级用
    原文首块（`degraded` 记原因）；失败率超 `fail_ratio` → 返回 None，
    调用方整题退回单发口径。
    """
    ts = two_stage_cfg(cfg)
    llm_cfg = _map_llm_cfg(cfg, ts)
    groups = _group_by_doc(contexts)
    t0 = time.perf_counter()

    def _one(group: dict) -> tuple[dict, bool, dict]:
        """返回 (summary_entry, degraded, usage)。usage 三键缓存命中时为 None。"""
        blocks = group["blocks"]
        first = blocks[0]
        prompt = _USER_MAP.format(
            question=question,
            doc=group["doc"],
            n=len(blocks),
            blocks="\n".join(f"[{c['no']}] {c['text']}" for c in blocks),
        )
        usage = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
        }
        try:
            text, _meta = llm.chat_timed(
                llm_cfg, prompt, system_prompt=SYSTEM_MAP, temperature=0.0
            )
            usage = {
                k: _meta.get(k)
                for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")
            }
        except Exception as exc:  # noqa: BLE001 单篇失败降级，不拖死整题
            return (
                {
                    "doc_id": group["doc_id"],
                    "doc": group["doc"],
                    "text": first["text"],
                    "source_context_idx": [first["no"]],
                    "degraded": f"{type(exc).__name__}: {exc}"[:200],
                },
                True,
                usage,
            )
        body = (text or "").strip()
        if not body:
            return (
                {
                    "doc_id": group["doc_id"],
                    "doc": group["doc"],
                    "text": first["text"],
                    "source_context_idx": [first["no"]],
                    "degraded": "empty_reply",
                },
                True,
                usage,
            )
        return (
            {
                "doc_id": group["doc_id"],
                "doc": group["doc"],
                "text": body,
                "source_context_idx": [c["no"] for c in blocks],
                "degraded": None,
            },
            False,
            usage,
        )

    with ThreadPoolExecutor(max_workers=ts["workers"]) as ex:
        entries = list(ex.map(_one, groups))
    summaries = [e for e, _, _ in entries]
    n_degraded = sum(1 for _, d, _ in entries if d)
    # 用量合计只累真实调用：缓存命中（None）不算 0——0 会冒充「真实为零」
    usage_tot = {
        k: sum(int(u[k] or 0) for _, _, u in entries if u.get(k) is not None)
        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")
    }
    usage_n = sum(
        1 for _, _, u in entries if any(u.get(k) is not None for k in usage_tot)
    )
    meta = {
        "ms": round((time.perf_counter() - t0) * 1000, 1),
        "n_docs": len(groups),
        "n_degraded": n_degraded,
        "fail_ratio": ts["fail_ratio"],
        "workers": ts["workers"],
        "usage": usage_tot,
        "usage_n_calls": usage_n,
    }
    if groups and n_degraded / len(groups) > ts["fail_ratio"]:
        meta["aborted"] = "fail_ratio_exceeded"
        return None, meta
    meta["aborted"] = None
    return summaries, meta


def answer(
    synthesizer: Any,
    question: str,
    contexts: list[dict],
    summaries: list[dict],
    *,
    question_type: str | None = None,
) -> str:
    """reduce 段（非流式）。计时照常写进 `synthesizer.last_meta`。

    只支持带引用口径：无引用约束的消融（#4）不路由两段式（调用方守卫）。
    `question_type` 是预测题型，只用来查思考档表——与单发 `Synthesizer.answer`
    同一条规则；map 段自己关思考，不经这张表。
    """
    llm_cfg = resolve_effort_cfg(synthesizer.llm_cfg, question_type)
    text, meta = llm.chat_timed(
        llm_cfg,
        USER_ANSWER_TWO_STAGE.format(
            context=format_summary_context(summaries, contexts), question=question
        ),
        system_prompt=SYSTEM_ANSWER_TWO_STAGE,
    )
    synthesizer.last_meta = meta
    return text


def answer_stream(
    synthesizer: Any,
    question: str,
    contexts: list[dict],
    summaries: list[dict],
    *,
    question_type: str | None = None,
):
    """reduce 段（流式）。与单发 `Synthesizer.answer_stream` 同构：
    逐段 yield，生成器耗尽后 `last_meta` 才填充完整。"""
    llm_cfg = resolve_effort_cfg(synthesizer.llm_cfg, question_type)
    stream, meta = llm.chat_stream(
        llm_cfg,
        USER_ANSWER_TWO_STAGE.format(
            context=format_summary_context(summaries, contexts), question=question
        ),
        system_prompt=SYSTEM_ANSWER_TWO_STAGE,
    )

    def _gen():
        yield from stream
        synthesizer.last_meta = dict(meta)

    return _gen()

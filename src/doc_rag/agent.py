"""Agentic RAG 的 policy 层（PLAN §5.5）。

一句话：单条问题内的有界多步检索 + 一次证据自判。工具全是现有部件的薄封装，
本模块只新增两样东西——「够不够」的判定，和邻居块的精确取用。

三条边界（写在这里是为了下次有人想顺手加时看得见）：

1. **不做自由 tool 选择**。每一步的动作集合是固定的：判定 → 按判定给出的
   `next_query` 再检索一次 → 按判定给出的 `widen_around` 补窗口。模型只能在
   「问什么 /  widening 哪几块」这两处说话，不能自己发明动作，也不能决定循环结构
   （步数、预算、停机条件全在代码里）。
2. **不做多轮会话状态**（F3）。这里没有任何跨请求可变的东西：collection 从外面读，
   绝不写回 `retriever.collection`——那正是 W1 拆掉的串库隐患（`api/main.py` 的注释）。
3. **每一轮都必须落 trace，重放只读 trace**。判定是模型产出、不可复现的，
   和改写侧 W3 同一个性质；不落盘就拒绝重放（见 `trace_from_dict` 的调用方）。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from .generate import llm
from .log import get_logger

log = get_logger("agent")

# 停机条件的取值集合。trace 里的 stop_reason 必须落在这个集合内——重放侧要靠它
# 判断这一轮到底停在哪，写成别的字符串就等于把「为什么停」这个信息丢掉。
STOP_REASONS = (
    "sufficient",  # 判定说证据够了
    "judge_degraded",  # 判定本身挂了/解析不出，按「够」停（宁停不猜）
    "no_next_query",  # 判定没给出下一步
    "no_new_evidence",  # 又检索了一次，但一个新块都没带来（原地打转）
    "steps_exhausted",  # 用完 max_steps
    "token_budget",  # 用完 max_prompt_tokens
)

ACTIONS = ("check_evidence", "search", "read_window")

# 默认只对这两类开：它们既是「覆盖率被清单格数封顶」的两类（PLAN §5.3），
# 也已有 ≤5s 的聚合 SLO 档。真正的开关在配置里，这里只是缺省值。
DEFAULT_TYPES = ("cross_doc", "time_filter")

# 邻居块的跨度：±1。写死成常量而不是配置——它是「窗口」这个词的定义的一部分，
# 做成可调只会让每次评估臂多一个要对齐的维度，而收益没人量过。
_WINDOW_SPAN = 1
_WINDOW_CHUNK_LIMIT = 3  # 单步最多 widening 几块：并集后 prompt 递增，这里先掐住

SYSTEM_EVIDENCE = (
    "你在决定一条问题要不要继续检索。给你的是问题和已经检索到的上下文。\n"
    "只输出一个 JSON 对象，不要多余文字：\n"
    '{"sufficient": false, "missing": "还缺的具体信息", '
    '"next_query": "下一步该检索的查询串", "widen_around": [1]}\n'
    "判据：\n"
    "① sufficient = 上下文的**并集**够不够逐句回答这条问题——不是「相关」，是「够」。\n"
    "② 不够时 missing 写清缺哪一条具体信息，next_query 给出下一步检索串"
    "（换词、换角度、指向还没出现的那一类文档）。\n"
    "③ 想不出有意义的下一步就把 next_query 留空串——宁可停，不要用同义反复的查询原地打转。\n"
    "④ widen_around 只在「某条上下文看着是从中间截断的、需要前后文才读得懂」时给出，"
    "里面是上下文的编号（1 起）。不需要就留空列表。"
)

_USER_EVIDENCE = """【问题】
{question}

【已检索到的上下文（并集，共 {n} 条）】
{contexts}
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def agent_cfg(cfg: dict) -> dict:
    """`agent` 段的读取入口：一处集中默认值，别在循环里散落 `get(..., 3)`。

    开关、题型、预算全是配置而不是代码常量——PLAN §5.5 把「分题型开关」叫作
    本期唯一的成本控制面，常量形式的开关等于没有开关。
    """
    raw = cfg.get("agent") or {}
    return {
        "enabled": bool(raw.get("enabled")),
        "types": tuple(raw.get("types") or DEFAULT_TYPES),
        # 步数上限夹在 [1, 3]：>3 是 PLAN §2 写死的边界；=1 就是「只判一次、不再
        # 检索」，那等于没开，但它是合法的对照臂口径（消融要能只留判定这一层）。
        "max_steps": min(3, max(1, int(raw.get("max_steps") or 3))),
        "judge_contexts": int(raw.get("judge_contexts") or 6),
        "max_contexts": int(raw.get("max_contexts") or 12),
        "max_prompt_tokens": int(raw.get("max_prompt_tokens") or 12000),
        "timeout_s": float(raw.get("timeout_s") or 20.0),
        "max_attempts": int(raw.get("max_attempts") or 1),
    }


def predict_type(plan: dict) -> str:
    """生产侧唯一可得的题型信号：改写计划的 aggregate / year。

    为什么不用真题型：`item.type` 只存在于黄金集里，CLI/API 拿不到。如果评估臂按
    真题型开关、服务臂按预测开关，五条入口的 trace 步数根本不可比（parity 当场失效）。
    所以**五处统一按预测值开**，真题型只用来在 eval 里量「这个预测准不准」
    （见 `type_matches_gold`）。
    """
    if not plan.get("aggregate"):
        return "single"
    return "time_filter" if (plan.get("filters") or {}).get("doc_date") else "cross_doc"


def type_matches_gold(predicted: str, gold_type: str | None) -> bool | None:
    """预测题型与黄金集题型是否一致。没有真题型时返回 None（不是 False）。"""
    if not gold_type:
        return None
    return predicted == gold_type


def endpoint_cfg(cfg: dict) -> dict:
    """判定调用的 endpoint：走 judge 那一套装配（可以和生成不同源）。

    为什么走 judge 而不是 llm：这一步是判分性质的二元判定，`eval.judge.*` 已经为它
    解决了「自偏」与「别带生成侧 key」两件事；而为什么不直接复用 judge 的超时——
    它在关键路径上，180s×2 次等于给一个可退化的步骤 6 分钟的等待能力（改写侧 W3
    踩过的同一个坑），所以超时与重试按 `agent.*` 压下来。
    """
    from .eval.judge import judge_cfg

    out = judge_cfg(cfg)
    ac = agent_cfg(cfg)
    out["timeout_s"] = ac["timeout_s"]
    out["max_attempts"] = ac["max_attempts"]
    out["reasoning_effort"] = "none"  # 判定不需要思考，reasoning token 实测占输出 97%
    return out


def _parse_verdict(text: str) -> dict | None:
    """解析判定输出；形状不对就返回 None（调用方按降级处理，不猜）。"""
    match = _JSON_RE.search(text or "")
    if not match:
        return None
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("sufficient"), bool):
        return None
    widen = raw.get("widen_around")
    idx: list[int] = []
    if isinstance(widen, list):
        for value in widen:
            try:
                n = int(value)
            except (TypeError, ValueError):
                continue
            if n >= 1:
                idx.append(n)
    return {
        "sufficient": raw["sufficient"],
        "missing": str(raw.get("missing") or ""),
        "next_query": str(raw.get("next_query") or "").strip(),
        "widen_around": idx[:_WINDOW_CHUNK_LIMIT],
    }


def _judge_contexts(pool: list[dict], cap: int) -> str:
    """送给判定的上下文：带 [n] 前缀、按 cap 截断。

    编号必须与 widen_around 的编号同源——判定看到的是第 n 条，回传的 n 才指得对。
    """
    lines = []
    for i, chunk in enumerate(pool[:cap], start=1):
        page = f" 第{chunk.get('page')}页" if chunk.get("page") else ""
        lines.append(
            f"[{i}]（{chunk.get('title') or chunk.get('doc_id')}{page}）{chunk.get('text')}"
        )
    return "\n\n".join(lines)


_EMPTY_USAGE = {
    "ms": 0.0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "reasoning_tokens": 0,
    "cached": False,
    "model": None,
}


def check_evidence(cfg: dict, question: str, pool: list[dict]) -> dict:
    """一次「证据够不够」的二元判定。失败一律退化成「够」——宁停不猜。

    退化成 sufficient=True 而不是 False：判定挂了还继续多查几步，等于让一个坏掉的
    刹车去踩油门，花的是真钱而依据是空的。降级会记在 trace 的 decision 上，
    停机由 `run_agent` 里那句 `if verdict["degraded"]` 兜住——即使哪天这个函数改成
    返回 sufficient=False，循环也不会多走一步（突变验过）。
    """
    ac = agent_cfg(cfg)
    llm_cfg = endpoint_cfg(cfg)
    if not (llm_cfg.get("api_key") and llm_cfg.get("model")):
        return {
            "sufficient": True,
            "missing": "",
            "next_query": "",
            "widen_around": [],
            "degraded": "no_llm",
            **_EMPTY_USAGE,
        }
    prompt = _USER_EVIDENCE.format(
        question=question,
        n=min(len(pool), ac["judge_contexts"]),
        contexts=_judge_contexts(pool, ac["judge_contexts"]),
    )
    try:
        reply, meta = llm.chat_timed(
            llm_cfg, prompt, system_prompt=SYSTEM_EVIDENCE, temperature=0.0
        )
    except Exception as exc:  # noqa: BLE001 关键路径上的可退化步骤：停，但留下痕迹
        log.warning(
            "证据判定调用失败，按「证据已足够」停止扩展",
            extra={"stage": "agent_check", "error": f"{type(exc).__name__}: {exc}"},
        )
        verdict_degraded = {
            "sufficient": True,
            "missing": "",
            "next_query": "",
            "widen_around": [],
            "degraded": f"call_failed:{type(exc).__name__}",
        }
        return {**verdict_degraded, **_EMPTY_USAGE}
    parsed = _parse_verdict(reply)
    if parsed is None:
        return {
            "sufficient": True,
            "missing": "",
            "next_query": "",
            "widen_around": [],
            "degraded": "bad_json",
            **_EMPTY_USAGE,
        }
    return {
        **parsed,
        "degraded": None,
        "ms": meta.get("ms"),
        "prompt_tokens": meta.get("prompt_tokens") or 0,
        "completion_tokens": meta.get("completion_tokens") or 0,
        "reasoning_tokens": meta.get("reasoning_tokens") or 0,
        "cached": bool(meta.get("cached")),
        "model": meta.get("model"),
    }


def chunk_point_id(chunk_id: str) -> str:
    """point-id 的唯一来源就是 `ingest/indexer.py` 那一条 uuid5，别在这儿再造一份。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def neighbour_ids(chunk_id: str, span: int = _WINDOW_SPAN) -> list[str]:
    """由 `doc_id:序号` 推邻居块的 chunk_id（不含自己）。

    为什么走 point-id 而不是 filter：`chunk_id` 没有 payload 索引，按它过滤是全扫。
    doc_id 是 sha256[:16]，不含冒号，所以 rpartition 取最后一段就是序号。
    """
    head, sep, tail = str(chunk_id).rpartition(":")
    if not sep or not tail.isdigit():
        return []
    ordinal = int(tail)
    out = []
    for offset in range(-span, span + 1):
        if offset == 0:
            continue
        neighbour = ordinal + offset
        if neighbour >= 1:
            out.append(f"{head}:{neighbour}")
    return out


def read_window(
    client: Any, collection: str, chunks: list[dict], span: int = _WINDOW_SPAN
) -> list[dict]:
    """按 point-id 精确取邻居块。返回的是**新**块（调用方已有的不重复给）。

    这是 §3 设计原则 2「父块/窗口进 LLM」的第一次真正实现——W2 删掉的那个
    `parent_expand` 键从来没人读过。零前置成本：不重新 ingest、不 backfill、不建索引。
    """
    have = {c.get("chunk_id") for c in chunks}
    wanted: list[str] = []
    queued: set[str] = set()
    for chunk in chunks:
        cid = chunk.get("chunk_id")
        if not cid:
            continue  # 注入的替身或旧结果文件可能没有 chunk_id，跳过而不是炸
        for neighbour in neighbour_ids(cid, span):
            if neighbour not in have and neighbour not in queued:
                queued.add(neighbour)
                wanted.append(neighbour)
    if not wanted or client is None or not collection:
        return []
    points = client.get(
        collection=collection,
        points=[chunk_point_id(cid) for cid in wanted],
        with_payload=True,
    )
    out = []
    for point in points:
        payload = point.payload or {}
        cid = payload.get("chunk_id")
        if not cid or cid in have:
            continue
        out.append(
            {
                "chunk_id": cid,
                "doc_id": payload.get("doc_id"),
                "title": payload.get("title"),
                "text": payload.get("text"),
                "section_path": payload.get("section_path") or [],
                "page": payload.get("page"),
                "block_type": payload.get("block_type"),
                "doc_date": payload.get("doc_date"),
                "score": 0.0,  # 窗口块是补齐用的，不参与排序，分数留 0 让它排在后面
                "from_window": True,
            }
        )
    return out


def _step(n: int, action: str, **kw: Any) -> dict:
    return {"n": n, "action": action, **kw}


def run_agent(
    cfg: dict,
    *,
    question: str,
    plan: dict,
    retriever: Any,
    pool: list[dict],
) -> tuple[list[dict], dict]:
    """有界多步。返回（并集后的完整清单, trace），清单第一项起就是原来的第一步结果。

    `pool` 是第一步 rerank 之后的未截断清单——agent 不重新做第一步，这样
    「agent 臂的第 1 步」与「单发臂」逐字同源，差异全部落在扩展步上（可比性要求）。
    """
    ac = agent_cfg(cfg)
    t_start = time.perf_counter()
    steps: list[dict] = []
    sub_queries: list[str] = []
    union = list(pool)
    seen = {c.get("chunk_id") or id(c) for c in union}
    used_tokens = 0
    calls = 0
    stop_reason = "steps_exhausted"
    n_step = 0

    def _budget_used() -> dict:
        return {
            "calls": calls,
            "prompt_tokens": used_tokens,
            "ms": round((time.perf_counter() - t_start) * 1000, 1),
        }

    # 一次「扩展」= 判定 + （ widening + 检索）。步数算的是判定轮次，含第 1 轮。
    for round_no in range(1, ac["max_steps"] + 1):
        verdict = check_evidence(cfg, question, union)
        calls += 1
        used_tokens += int(verdict["prompt_tokens"] or 0)
        n_step += 1
        steps.append(
            _step(
                n_step,
                "check_evidence",
                args={
                    "n_contexts": min(len(union), ac["judge_contexts"]),
                    "widen_around": verdict["widen_around"],
                },
                decision=(
                    "degraded"
                    if verdict["degraded"]
                    else ("sufficient" if verdict["sufficient"] else "extend")
                ),
                missing=verdict["missing"],
                degraded=verdict["degraded"],
                ms=verdict["ms"],
                prompt_tokens=verdict["prompt_tokens"],
                completion_tokens=verdict["completion_tokens"],
                reasoning_tokens=verdict["reasoning_tokens"],
                cached=verdict["cached"],
            )
        )
        if verdict["degraded"]:
            stop_reason = "judge_degraded"
            break
        if verdict["sufficient"]:
            stop_reason = "sufficient"
            break
        if used_tokens >= ac["max_prompt_tokens"]:
            stop_reason = "token_budget"
            break
        if round_no == ac["max_steps"]:
            # 最后一次判定之后已经没有「再检索」的预算了，别再白检索一步。
            stop_reason = "steps_exhausted"
            break

        if verdict["widen_around"]:
            picked = [
                union[i - 1]
                for i in verdict["widen_around"]
                if 0 < i <= min(len(union), ac["judge_contexts"])
            ]
            window = read_window(
                getattr(retriever, "client", None),
                getattr(retriever, "collection", None) or "",
                picked,
            )
            n_step += 1
            prev_docs = {c.get("doc_id") for c in union}
            added = [c for c in window if (c.get("chunk_id") or id(c)) not in seen]
            for chunk in added:
                seen.add(chunk.get("chunk_id") or id(chunk))
            union += added
            steps.append(
                _step(
                    n_step,
                    "read_window",
                    args={"around": [c.get("chunk_id") for c in picked]},
                    n_retrieved=len(window),
                    # 邻居块几乎总是同一篇文档——按「新增了几篇文档」数就该是 0。
                    # 数成 1 会让成本-质量前沿把窗口步误记成检索步的收益。
                    new_doc_ids_added=len({c.get("doc_id") for c in added} - prev_docs),
                    decision="extended" if added else "nothing_new",
                    ms=0.0,
                    prompt_tokens=0,
                )
            )

        next_query = verdict["next_query"]
        if not next_query:
            stop_reason = "no_next_query"
            break
        sub_queries.append(next_query)
        outcome = retriever.retrieve(
            next_query,
            top_n=plan.get("top_n"),
            filters=plan.get("filters"),
            aggregate=bool(plan.get("aggregate")),
        )
        fresh = [c for c in outcome.chunks if (c.get("chunk_id") or id(c)) not in seen]
        for chunk in fresh:
            seen.add(chunk.get("chunk_id") or id(chunk))
        # 新块并进来后按分数重排：窗口块 score=0 自然沉底，不会挤掉榜首。
        union = sorted(union + fresh, key=lambda c: -(c.get("score") or 0.0))
        n_step += 1
        steps.append(
            _step(
                n_step,
                "search",
                args={"sub_query": next_query},
                n_retrieved=len(outcome.chunks),
                new_doc_ids_added=len({c.get("doc_id") for c in fresh}),
                decision="extended" if fresh else "nothing_new",
                filter_fallback=outcome.filter_fallback,
                ms=0.0,
                prompt_tokens=0,
            )
        )
        if not fresh:
            stop_reason = "no_new_evidence"
            break

    trace = {
        "steps": steps,
        "sub_queries": sub_queries,
        "stop_reason": stop_reason,
        "budget_used": _budget_used(),
        # 这几项让 trace 能自证它跑的是哪条口径——和结果文件里 top_n_used 同一个理由。
        "max_steps": ac["max_steps"],
        "max_contexts": ac["max_contexts"],
        "max_prompt_tokens": ac["max_prompt_tokens"],
        "judge_contexts": ac["judge_contexts"],
        "n_docs_union": len({c.get("doc_id") for c in union}),
    }
    return union, trace


def trace_from_dict(raw: Any) -> dict | None:
    """重放侧的唯一入口：形状不对就判「不可重放」，不做宽容修复。"""
    if not isinstance(raw, dict):
        return None
    if raw.get("stop_reason") not in STOP_REASONS:
        return None
    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    if any(not isinstance(s, dict) or s.get("action") not in ACTIONS for s in steps):
        return None
    if not isinstance(raw.get("budget_used"), dict):
        return None
    return raw

"""客观指标评估器（PLAN §5.3 双轨的确定性一轨）。

指标（不依赖 LLM judge，可复现）：
- Recall@5 / Recall@top_n：黄金来源文档是否被检回
- MRR：首个命中来源的排名倒数
- contains_acc：非拒答题答案包含全部 must_contain 关键词
- refusal_acc：拒答题正确拒答（出现拒答语且未编造 must_contain）
- citation_valid_rate：答案中 [n] 引用编号全部落在上下文范围内
- 延迟：分阶段（改写/检索/重排/合成/端到端）p50/p95/max，按题型拆分

延迟口径（PLAN「延迟口径」）：合成耗时只在**缓存未命中**时才算延迟；命中缓存
返回的是本地查询耗时。只要本轮有命中，`latency.cache_contaminated` 置真，
提示 LLM 延迟被低估——测真实延迟必须关合成缓存。
"""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient

from ..config import load_config
from ..generate import prompts
from ..generate.llm import cache_enabled
from ..generate.synthesizer import Synthesizer
from ..ingest.embedder import Embedder
from ..retrieve.hybrid import HybridRetriever
from ..retrieve.rewrite import QueryRewriter
from .schema import GoldItem

_REFUSAL_MARKERS = [
    "无法回答", "无法确定", "未形成决议", "未记载", "未提到", "没有提到",
    "缺少", "没有足够", "未能找到", "根据现有文档", "没有讨论", "未讨论",
]
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _build_retriever(cfg: dict, collection: str | None) -> tuple[HybridRetriever, Synthesizer]:
    retriever = HybridRetriever(
        client=QdrantClient(url=cfg["qdrant"]["url"], timeout=60.0),
        embedder=Embedder(cfg["embedding"]),
        collection=collection or cfg["qdrant"]["collection"],
        retrieval_cfg=cfg["retrieval"],
    )
    return retriever, Synthesizer(cfg["llm"])


def _maybe_rerank(cfg: dict, question: str, results: list[dict], use_rerank: bool) -> list[dict]:
    """重排（PLAN §7 Phase 2）：削减上下文噪声，Faithfulness 的主要手段。"""
    if not use_rerank or not results:
        return results
    from ..retrieve.rerank import Reranker

    try:
        return Reranker(cfg["rerank"]).rerank(question, results)
    except Exception:  # noqa: BLE001 重排失败不阻塞（退回融合顺序）
        return results


def _refusal_ok(answer: str) -> bool:
    return any(marker in answer for marker in _REFUSAL_MARKERS)


def _contains_loose(keyword: str, answer: str) -> bool:
    """宽松包含：关键词的字符按序出现即算命中（容忍「整理纪要」vs「整理会议纪要」）。"""
    pos = 0
    target = _norm(keyword)
    hay = _norm(answer)
    for ch in hay:
        if pos < len(target) and ch == target[pos]:
            pos += 1
    return pos == len(target)


def _contexts(results: list[dict], max_n: int | None = None) -> list[dict]:
    """构造送 LLM 的编号上下文；max_n 控制上限以约束输入 token（PLAN §8 成本控制）。"""
    if max_n:
        results = results[:max_n]
    return [
        {"no": i + 1, "text": r["text"], "doc": r["title"] or r["doc_id"], "page": r["page"]}
        for i, r in enumerate(results)
    ]


def _judge_contexts(ctx: list[dict]) -> list[str]:
    """judge 必须看到与 LLM 完全相同的上下文串（含 `[n]（文档名 第p页）` 前缀）。

    实测 bug：此前只把正文传给 RAGAS，而合成 prompt 要求答案标注来源文档名 ——
    答案里「《某文档》中…」这类归属陈述在 judge 眼里无据可依，一律判为不忠实。
    量化：提及文档名的 18 条均值 0.475，不提的 37 条 0.716，24pt 的差距全部来自
    度量口径而非答案质量；零分条目 7 条里 5 条提及文档名。
    同时，上下文带 `[n]` 编号后，答案里的引用标记也变成可验证的陈述。
    """
    return [prompts.format_context([c]) for c in ctx]


def _retrieve_contexts(
    question: str,
    meta: dict,
    retriever: HybridRetriever,
    cfg: dict,
    aggregate: bool = False,
) -> list[dict]:
    """按 meta 记录的重检索参数重放检索，还原 LLM 实际看到的编号上下文。

    旧结果文件只存了正文；要补文档名必须重放**同一套**参数（改写 / 聚合 / 重排 /
    上下文上限），否则 judge 拿到的是另一批块——等于用 A 的上下文去判 B 的答案。
    meta.retrieval 形如 `dense+bm25+rrf[hybrid]+rewrite+rerank`。
    """
    flags = meta.get("retrieval") or ""
    use_rewrite = "+rewrite" in flags
    use_rerank = "+rerank" in flags
    aggregate = aggregate or "+aggregate" in flags
    # 检索模式已在 ragas_from_results 构建 retriever 前还原（dense/hybrid 构建时定死）
    rewriter = QueryRewriter(cfg["retrieval"]) if use_rewrite else None
    plan = (
        rewriter.rewrite(question)
        if rewriter
        else {"rewritten": question, "filters": None, "aggregate": aggregate}
    )
    results = retriever.retrieve(
        plan["rewritten"],
        top_n=meta.get("top_n") or 8,
        filters=plan["filters"],
        aggregate=plan["aggregate"] or aggregate,
    )
    results = _maybe_rerank(cfg, plan["rewritten"], results, use_rerank)
    return _contexts(
        results, max_n=int(cfg["retrieval"].get("max_contexts") or 0) or None
    )


def evaluate(
    gold_file: Path,
    cfg: dict | None = None,
    collection: str | None = None,
    top_n: int = 8,
    limit: int | None = None,
    with_ragas: bool = False,
    with_answers: bool = True,
    mode: str | None = None,
    aggregate: bool = False,
    use_rewrite: bool = False,
    use_rerank: bool = False,
    require_citation: bool = True,
    ragas_sample: int | None = None,
    use_judge_cache: bool = True,
) -> dict:
    cfg = cfg or load_config()
    if mode:
        cfg["retrieval"]["mode"] = mode  # 消融开关：dense / hybrid
    retriever, synthesizer = _build_retriever(cfg, collection)
    rewriter = QueryRewriter(cfg["retrieval"]) if use_rewrite else None
    payload = json.loads(gold_file.read_text(encoding="utf-8"))
    items_raw = payload["items"][:limit] if limit else payload["items"]
    items = [GoldItem.model_validate(i) for i in items_raw]

    per_item: list[dict] = []
    for item in items:
        t_item = time.perf_counter()
        t0 = time.perf_counter()
        plan = (
            rewriter.rewrite(item.question)
            if rewriter
            else {"rewritten": item.question, "filters": None, "aggregate": aggregate, "top_n": None}
        )
        t_rewrite = time.perf_counter()
        results = retriever.retrieve(
            plan["rewritten"],
            top_n=top_n,
            filters=plan["filters"],
            aggregate=plan["aggregate"] or aggregate,
        )
        t_retrieve = time.perf_counter()
        results = _maybe_rerank(cfg, plan["rewritten"], results, use_rerank)
        t_rerank = time.perf_counter()
        got_ids = [r["doc_id"] for r in results]
        hit_ranks = [got_ids.index(s) + 1 for s in item.source_doc_ids if s in got_ids]
        first_rank = min(hit_ranks) if hit_ranks else None
        # 文档覆盖率：聚合题（答案集 5~30 篇）真正该量的指标
        coverage = (
            len(set(got_ids) & set(item.source_doc_ids)) / len(item.source_doc_ids)
            if item.source_doc_ids
            else None
        )

        ctx = _contexts(
            results, max_n=int(cfg["retrieval"].get("max_contexts") or 0) or None
        )
        synth_meta: dict | None = None
        if with_answers:
            answer = synthesizer.answer(
                item.question,
                ctx,
                require_citation=require_citation,
                aggregate=bool(plan.get("aggregate") or aggregate),
            )
            # 计时从 Synthesizer 实例上取：answer() 的返回类型保持不变，
            # 现有调用点与测试（Mock synthesizer）都不用改。
            # 必须是 dict——Mock 的自动属性会造出一个不可序列化的假 meta。
            candidate = getattr(synthesizer, "last_meta", None)
            synth_meta = candidate if isinstance(candidate, dict) else None
        else:
            answer = ""
        t_synth = time.perf_counter()

        if not with_answers:
            answered_ok = None  # 检索模式不评回答
            answered_ok_loose = None
            over_refusal = None
        elif item.refusable:
            answered_ok = _refusal_ok(answer)
            answered_ok_loose = answered_ok
            over_refusal = None
        elif item.must_contain:
            answered_ok = all(_norm(m) in _norm(answer) for m in item.must_contain)
            answered_ok_loose = all(_contains_loose(m, answer) for m in item.must_contain)
            # 过度拒答：声称无法回答，但上下文里其实含有关键信息
            ctx_text = _norm(" ".join(c["text"] for c in ctx))
            ctx_has = all(_norm(m) in ctx_text for m in item.must_contain)
            over_refusal = bool(_refusal_ok(answer) and ctx_has)
        else:
            answered_ok = None  # 无判据（如部分聚合题），不计入 answer 准确率
            answered_ok_loose = None
            over_refusal = None

        refs = [int(n) for n in _CITATION_RE.findall(answer)]
        citation_valid = all(1 <= n <= len(ctx) for n in refs) if refs else None
        citation_present = bool(refs) if answer else None

        per_item.append(
            {
                "id": item.id,
                "type": item.type,
                "question": item.question,
                "first_hit_rank": first_rank,
                "doc_coverage": round(coverage, 4) if coverage is not None else None,
                "n_source_docs": len(item.source_doc_ids),
                "answered_ok": answered_ok,
                "answered_ok_loose": answered_ok_loose,
                "over_refusal": over_refusal,
                "citation_valid": citation_valid,
                "citation_present": citation_present,
                "n_citations": len(refs),
                "answer": answer,
                "contexts": _judge_contexts(ctx),
                "latency": {
                    "rewrite": round((t_rewrite - t0) * 1000, 1),
                    "retrieve": round((t_retrieve - t_rewrite) * 1000, 1),
                    "rerank": round((t_rerank - t_retrieve) * 1000, 1),
                    # 检索侧不含 LLM：这部分与模型无关，换模型不必重测
                    "retrieval_total": round((t_rerank - t0) * 1000, 1),
                    "synthesize": (synth_meta or {}).get("ms") if with_answers else None,
                    "synth_cached": bool((synth_meta or {}).get("cached")) if with_answers else None,
                    "total": round((t_synth - t_item) * 1000, 1),
                },
            }
        )

    # 聚合（no_answer 题无来源文档，不计入 Recall/MRR 分母，由 refusal_acc 单独评）
    with_source = [
        r for r in per_item if not _is_refusable(items, r["id"])
    ]
    hits5 = [r for r in with_source if r["first_hit_rank"] and r["first_hit_rank"] <= 5]
    hitsN = [r for r in with_source if r["first_hit_rank"]]
    mrr_scores = [1.0 / r["first_hit_rank"] for r in with_source if r["first_hit_rank"]]
    scorable = [
        r for r in per_item
        if not _is_refusable(items, r["id"]) and r["answered_ok"] is not None
    ]
    refusables = [r for r in per_item if _is_refusable(items, r["id"])]
    cites = [r for r in per_item if r["citation_valid"] is not None]

    covs = [r["doc_coverage"] for r in per_item if r["doc_coverage"] is not None]

    summary = {
        "n_items": len(per_item),
        "recall_at_5": len(hits5) / len(with_source),
        f"recall_at_{top_n}": len(hitsN) / len(with_source),
        "mrr": sum(mrr_scores) / len(with_source),
        "mean_doc_coverage": round(sum(covs) / len(covs), 4) if covs else None,
        "contains_acc": _safe_div(
            sum(1 for r in scorable if r["answered_ok"]), len(scorable)
        ),
        # 宽松包含（容忍改写）：与严格值一起看，差值即「度量伪影」大小
        "contains_acc_loose": _safe_div(
            sum(1 for r in scorable if r["answered_ok_loose"]), len(scorable)
        ),
        # 过度拒答：上下文含关键信息却答「无法回答」
        "over_refusal_rate": _safe_div(
            sum(1 for r in per_item if r["over_refusal"]),
            sum(1 for r in per_item if r["over_refusal"] is not None),
        ),
        "refusal_acc": _safe_div(
            sum(1 for r in refusables if r["answered_ok"]), len(refusables)
        ),
        "citation_valid_rate": _safe_div(
            sum(1 for r in cites if r["citation_valid"]), len(cites)
        ),
        # 引用存在率：无引用也算未遵守（否则「从不引用」的模型会显示 None 而非 0）
        "citation_presence_rate": _safe_div(
            sum(1 for r in per_item if r["citation_present"]),
            sum(1 for r in per_item if r["citation_present"] is not None),
        ),
        "coverage_by_type": {
            t: round(
                sum(r["doc_coverage"] for r in per_item if r["type"] == t and r["doc_coverage"] is not None)
                / max(sum(1 for r in per_item if r["type"] == t and r["doc_coverage"] is not None), 1),
                4,
            )
            for t in sorted({r["type"] for r in per_item})
        },
    }
    latency = _latency_summary(per_item)
    if latency:
        summary["latency"] = latency

    ragas_summary = None
    if with_ragas:
        ragas_summary = _run_ragas(
            _ragas_rows(items, per_item), cfg, sample_n=ragas_sample,
            use_cache=use_judge_cache,
        )

    results = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "top_n": top_n,
            "collection": retriever.collection,
            "retrieval": f"dense+bm25+rrf[{retriever.cfg.get('mode', 'hybrid')}]"
            + ("+aggregate" if aggregate else "")
            + ("+rewrite" if use_rewrite else "")
            + ("+rerank" if use_rerank else ""),
            "with_answers": with_answers,
            # 让结果文件自证身份：延迟数字曾因「不知道是哪个模型、缓存开没开」
            # 而无法归属（PLAN 里 1.3s 与 5.3~7.4s 的矛盾）。事后靠人回忆不可靠。
            "llm_model": (cfg.get("llm") or {}).get("model"),
            "answer_cache": cache_enabled(cfg.get("llm") or {}) if with_answers else None,
            # prompt 指纹：三组对照的基线/收紧两组答案曾因 meta 不记 prompt 版本
            # 而无法归属（哪组用了哪个 prompt 靠猜），结论只能整体作废
            "prompt_fingerprint": prompts.fingerprint() if with_answers else None,
        },
        "summary": summary,
        "ragas": ragas_summary,
        "items": per_item,
    }
    return results


def _quantiles(values: list[float], ns: tuple[float, ...] = (50, 95)) -> dict:
    """分位数（标准库实现，不引 numpy）。

    用最近秩法（nearest-rank）：小样本下比线性插值更保守，也更贴近
    「P95 到底有没有超 8s」这种判定——插值会给出一个任何真实请求都没出现过的值。
    样本量不足时照常返回，但调用方应结合 n 判读（n<20 的 p95 基本等于 max）。
    """
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return {}
    out = {"n": len(xs), "min": round(xs[0], 1), "max": round(xs[-1], 1),
           "mean": round(sum(xs) / len(xs), 1)}
    for n in ns:
        idx = max(0, min(len(xs) - 1, math.ceil(n / 100 * len(xs)) - 1))
        out[f"p{int(n)}"] = round(xs[idx], 1)
    return out


def _latency_summary(per_item: list[dict]) -> dict | None:
    """按阶段汇总延迟 + 按题型拆分合成与端到端。

    按题型拆分不是装饰：本语料的延迟是**双峰**的——短答案（fact/term）约 2~3s，
    聚合题（cross_doc/time_filter，答案 600~900 字）25~34s。只报一个混合 P95
    会把尾部藏起来，也解释不了「均值达标但聚合题超 8s」。

    合成分位数只在**未命中缓存**的条目上算：缓存命中的 ms 是本地查询耗时，
    混进去会把 LLM 延迟拉到毫秒级（这正是 PLAN 里 1.3s 的来路）。
    """
    rows = [r for r in per_item if r.get("latency")]
    if not rows:
        return None
    uncached = [r for r in rows if r["latency"].get("synthesize") is not None
                and not r["latency"].get("synth_cached")]
    stages = ("rewrite", "retrieve", "rerank", "retrieval_total")
    summary: dict = {
        "unit": "ms",
        # 检索侧与 LLM 无关（实测换模型完全一致），全部条目都算
        "by_stage": {s: _quantiles([r["latency"].get(s) for r in rows]) for s in stages},
        "synthesize": _quantiles([r["latency"].get("synthesize") for r in uncached]),
        "synthesize_n_uncached": len(uncached),
        "total": _quantiles([r["latency"].get("total") for r in rows]),
        "by_type": {},
        "n": len(rows),
        "cached_answers": sum(1 for r in rows if r["latency"].get("synth_cached")),
    }
    summary["cache_contaminated"] = summary["cached_answers"] > 0
    # 端到端是否达标：PLAN 目标 P95 ≤ 8s（用全量 total，含缓存命中的快条目）
    e2e = summary["total"]
    summary["target_p95_ms"] = 8000
    summary["p95_meets_target"] = bool(e2e) and e2e.get("p95", 0) <= 8000
    for t in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == t]
        sub_uncached = [r for r in uncached if r["type"] == t]
        summary["by_type"][t] = {
            "n": len(sub),
            "synthesize": _quantiles([r["latency"].get("synthesize") for r in sub_uncached]),
            "total": _quantiles([r["latency"].get("total") for r in sub]),
            # 答案长度是延迟的主因（实测），不报它就无法解释聚合题为何慢
            "answer_chars_mean": round(
                sum(len(r.get("answer") or "") for r in sub) / len(sub), 1
            ),
        }
    return summary


def _is_refusable(items: list[GoldItem], item_id: str) -> bool:
    return next((i.refusable for i in items if i.id == item_id), False)


def _safe_div(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def _ragas_rows(items: list[GoldItem], per_item: list[dict]) -> list[dict]:
    """构造 RAGAS 输入行：只评可答题（拒答题的拒答措辞天然不在原文中，会被忠实度惩罚）。"""
    return [
        {
            "id": item.id,
            "type": item.type,
            "user_input": item.question,
            "response": r["answer"],
            "retrieved_contexts": r.get("contexts") or [],
        }
        for item, r in zip(items, per_item)
        if not item.refusable and r["answer"]
    ]


def _sample_rows(rows: list[dict], sample_n: int) -> list[dict]:
    """均匀抽样：黄金集按题型分块排序，取前 N 条会整段漏掉排在末尾的题型。

    实测教训：63 条黄金集里 cross_doc(8) / time_filter(5) 排在最末，
    旧的 rows[:15] 抽样 15 条完全不含这两类聚合题 —— 而分块策略的差异
    恰恰最可能体现在聚合题上（上下文来自大池子，跨块拼接更多）。
    """
    if not sample_n or sample_n >= len(rows):
        return rows
    step = len(rows) / sample_n
    return [rows[int(i * step)] for i in range(sample_n)]


def _make_token_counter():
    """构造 judge 调用量计数器（PLAN §8 成本可见）。

    必须继承 langchain 的 BaseCallbackHandler：回调管理器会读 `ignore_chain` /
    `raise_error` 等属性，鸭子类型会直接抛 AttributeError。
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class _TokenCounter(BaseCallbackHandler):
        def __init__(self) -> None:
            self.calls = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.reasoning_tokens = 0

        def on_llm_end(self, response, **kwargs) -> None:
            self.calls += 1
            llm_output = getattr(response, "llm_output", None) or {}
            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            if usage:
                # 原始 OpenAI 用量键是 reasoning_tokens；langchain 归一化后是 reasoning
                details = usage.get("completion_tokens_details") or {}
                reasoning = int(details.get("reasoning_tokens") or 0)
            else:
                reasoning = 0
                for gen_list in getattr(response, "generations", None) or []:
                    for gen in gen_list:
                        meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
                        if not meta:
                            continue
                        usage = meta  # 多代时取最后一份，但 reasoning 要累加
                        reasoning += int(
                            (meta.get("output_token_details") or {}).get("reasoning") or 0
                        )
            self.prompt_tokens += int(
                usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            )
            self.completion_tokens += int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            self.reasoning_tokens += reasoning

        def as_dict(self) -> dict:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                # 推理型模型把输出预算大部分花在看不见的 reasoning 上（实测 judge 占 97%），
                # 单列出来才能看出「关掉思考」省的是哪一块
                "reasoning_tokens": self.reasoning_tokens,
            }

    return _TokenCounter()


def _judge_chat_kwargs(cfg: dict) -> dict:
    """judge 的统一构造参数（`_run_ragas` 与 `probe-judge` 必须同源，否则探针看到的行为
    和正式判分不一致——这正是当初定位口径 bug 时踩过的坑）。"""
    llm_cfg = cfg["llm"]
    kwargs: dict = {
        "model": llm_cfg["model"],
        "base_url": llm_cfg["base_url"],
        "api_key": llm_cfg["api_key"],
        "temperature": 0,
    }
    # judge 的两个子任务（拆陈述 / 逐条判定）几乎不需要思考，但推理型模型会把
    # 输出预算的 97% 花在看不见的 reasoning token 上（实测单次 1554 → 79，全量 10.4×）。
    # 同一个模型、只关思考，不改 judge 身份，不破坏 §5.3 的 judge 固定口径。
    effort = (cfg.get("eval", {}).get("judge") or {}).get("reasoning_effort")
    if effort:
        kwargs["reasoning_effort"] = effort
    return kwargs


def _run_ragas(
    rows: list[dict],
    cfg: dict,
    sample_n: int | None = None,
    use_cache: bool = True,
    total: int | None = None,
) -> dict | None:
    """RAGAS 第二轨：rows=[{id,type,user_input,response,retrieved_contexts}] → judge 指标。

    可信度口径（PLAN §5.3）：judge 固定模型、temperature=0；只看与客观指标的相对一致性。
    `use_cache=False` 用于测 judge 自身的运行间随机性（temperature=0 也不保证跨请求逐字复现）。
    `total`：调用方已自行抽样时传入抽样前的总数，保证报告口径（n_answerable_total / sampled）准确。
    """
    try:
        from langchain.globals import set_llm_cache
        from langchain_community.cache import SQLiteCache
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, evaluate as ragas_evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, Faithfulness
    except Exception as exc:  # noqa: BLE001
        return {"error": f"ragas 不可用：{exc}"}

    if sample_n is None:
        sample_n = int((cfg.get("eval") or {}).get("ragas_sample") or 0)
    all_n = total if total is not None else len(rows)
    rows = _sample_rows(rows, sample_n)

    # judge 是最大调用方（每指标每条多次内部调用），必须走缓存：
    # langchain 有自己的缓存层，不复用 llm.chat 的缓存（PLAN §8 成本控制）
    judge_cache_path = None
    if use_cache and cache_enabled(cfg.get("llm")):
        try:
            from ..generate.llm import _CACHE_PATH as _llm_cache_path

            judge_cache_path = str(_llm_cache_path.parent / "judge_cache.sqlite")
            Path(judge_cache_path).parent.mkdir(parents=True, exist_ok=True)
            set_llm_cache(SQLiteCache(database_path=judge_cache_path))
        except Exception as exc:  # noqa: BLE001
            set_llm_cache(None)
            # 静默降级 = 无缓存跑完全量 judge（实付约 10 倍）。宁可失败也不白花。
            raise RuntimeError(
                f"judge 缓存初始化失败（{exc}）——已阻止无缓存的全量判分。"
                f"可用 --fresh-judge 显式跳过缓存，或修复 .cache 目录权限后重试。"
            ) from exc
    else:
        set_llm_cache(None)  # 显式关缓存：set_llm_cache 是进程级全局，必须清掉

    # 指标可选（PLAN §8 成本控制）：AnswerRelevancy 在中文场景噪声大且需嵌入调用
    wanted = [m.lower() for m in (cfg.get("eval", {}).get("ragas_metrics") or ["faithfulness"])]
    metric_map = {"faithfulness": Faithfulness(), "answer_relevancy": AnswerRelevancy()}
    metrics = [metric_map[m] for m in wanted if m in metric_map]
    if not metrics:
        return {"skipped": f"未配置有效指标：{wanted}"}

    llm_cfg = cfg["llm"]
    judge = LangchainLLMWrapper(ChatOpenAI(**_judge_chat_kwargs(cfg), max_retries=0))
    # AnswerRelevancy 需要嵌入模型：用 SiliconFlow 的 BGE-M3（DeepSeek 无 embedding API）
    emb_cfg = cfg["embedding"]
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=emb_cfg["model"],
            base_url=emb_cfg["base_url"],
            api_key=emb_cfg["api_key"],
        )
    )
    counter = _make_token_counter()
    try:
        from ragas.run_config import RunConfig

        # RAGAS 默认 max_retries=10，叠上 judge 自身重试会把限流放大成几十次请求；
        # SDK 层已在 ChatOpenAI(max_retries=0) 关掉，这里给个收紧的应用层上限
        run_config = RunConfig(max_retries=2, max_wait=30, timeout=180, max_workers=8)
    except Exception:  # noqa: BLE001 旧版 ragas 无 RunConfig 时不设限，但不阻塞评估
        run_config = None
    try:
        ds = EvaluationDataset.from_list(
            [{k: v for k, v in r.items() if k not in ("id", "type")} for r in rows]
        )
        out = ragas_evaluate(
            dataset=ds,
            metrics=metrics,
            llm=judge,
            embeddings=embeddings,
            callbacks=[counter],
            show_progress=False,
            **({"run_config": run_config} if run_config is not None else {}),
        )
        df = out.to_pandas()
        summary = {
            "n": len(rows),
            "n_answerable_total": all_n,
            "sampled": sample_n if 0 < sample_n < all_n else None,
            "judge_cache": judge_cache_path,
            "metrics": [m.name for m in metrics],
            "token_usage": counter.as_dict(),
        }
        per_item = []
        for idx, row in enumerate(rows):
            entry = {"id": row["id"], "type": row["type"]}
            for m in metrics:
                if m.name in df.columns and idx < len(df):
                    val = df[m.name].iloc[idx]
                    entry[m.name] = (
                        None if val is None or math.isnan(float(val)) else round(float(val), 4)
                    )
            per_item.append(entry)
        for m in metrics:
            if m.name in df.columns:
                summary[m.name] = round(float(df[m.name].mean()), 4)
        summary["per_item"] = per_item
        # 分题型均值：抽样偏置最容易在题型维度暴露（聚合题上下文最长、最易失分）
        by_type: dict[str, list[float]] = {}
        for entry in per_item:
            for m in metrics:
                v = entry.get(m.name)
                if v is not None:
                    by_type.setdefault(f"{m.name}:{entry['type']}", []).append(v)
        summary["by_type"] = {
            k: round(sum(v) / len(v), 4) for k, v in sorted(by_type.items())
        }
        return summary
    except Exception as exc:  # noqa: BLE001
        return {"error": f"ragas 运行失败：{exc}"}


def probe_judge(results_file: Path, item_id: str, cfg: dict | None = None) -> dict:
    """打印单条答案的 judge 中间产物（抽出的陈述 + 逐条判定 + 理由）。

    绝对分值可疑时的定位手段：能区分「答案真不忠实」与「judge 抽错/判错」。
    消融 #2 的口径 bug 就是这样定位到的——q007 逐字引用上下文原文却被判 0.00，
    理由暴露了「上下文没有提到《某文档》这一文档标题」。
    """
    import asyncio

    from langchain.globals import set_llm_cache
    from langchain_community.cache import SQLiteCache
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    cfg = cfg or load_config()
    data = json.loads(results_file.read_text(encoding="utf-8"))
    item = next((i for i in data["items"] if i["id"] == item_id), None)
    if item is None:
        return {"error": f"{results_file.name} 里没有 {item_id}"}

    # 复用 judge 缓存：探针常被连着调好几条，重付一遍 judge 调用纯属浪费
    if cache_enabled(cfg.get("llm")):
        try:
            from ..generate.llm import _CACHE_PATH as _llm_cache_path

            cache_path = _llm_cache_path.parent / "judge_cache.sqlite"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            set_llm_cache(SQLiteCache(database_path=str(cache_path)))
        except Exception:  # noqa: BLE001 探针失败可见即可，不中断
            set_llm_cache(None)

    metric = Faithfulness()
    # 必须与正式判分同源（含 reasoning_effort / max_retries），否则探针看到的行为
    # 与 `_run_ragas` 不一致——探针的价值就在于反映真实判分时 judge 在做什么
    metric.llm = LangchainLLMWrapper(
        ChatOpenAI(**_judge_chat_kwargs(cfg), max_retries=0)
    )
    row = {
        "user_input": item["question"],
        "response": item["answer"],
        "retrieved_contexts": item.get("contexts") or [],
    }

    async def _run() -> dict:
        stmts = await metric._create_statements(row, [])
        verdicts = await metric._create_verdicts(row, stmts.statements, [])
        return {
            "statements": [
                {"statement": v.statement, "verdict": bool(v.verdict), "reason": v.reason}
                for v in verdicts.statements
            ],
            "score": metric._compute_score(verdicts),
        }

    out = asyncio.run(_run())
    return {"id": item_id, "type": item["type"], "question": item["question"], **out}


def _legacy_contexts(contexts: list[str] | None) -> bool:
    """旧结果文件只存了正文（无 `[n]（文档名 第p页）` 前缀），不是 LLM 实际看到的上下文。

    用它跑 judge 会系统性低估忠实度（见 `_judge_contexts`），必须重检索还原。
    """
    if not contexts:
        return True
    return not contexts[0].startswith("[1]")


def ragas_from_results(
    results_file: Path,
    cfg: dict | None = None,
    sample_n: int | None = None,
    use_cache: bool = True,
    out_file: Path | None = None,
) -> dict | None:
    """对已保存的评估结果补跑 RAGAS：答案复用，上下文过期时按需重检索还原。

    三个成本/正确性要点：
    - **先抽样、后重检索**。检索要调 embedding、重排要调远端 API，都不是免费的；
      旧实现会给全部可答题重检索，哪怕只要判 15 条。
    - 旧结果文件的 `meta.retrieval` 里记着 dense/hybrid，重放必须还原，否则是在
      用另一套检索的上下文判分。
    - 重放后正文对不上 → 这条的答案本来就不是对着这份上下文生成的，判了也是假数据。
      **拒判并报错**，而不是拿替换后的上下文悄悄送进付费 judge。
    """
    cfg = cfg or load_config()
    data = json.loads(results_file.read_text(encoding="utf-8"))
    meta = data.get("meta") or {}
    candidates = [i for i in data["items"] if i["type"] != "no_answer" and i.get("answer")]

    resolved_n = sample_n
    if resolved_n is None:
        resolved_n = int((cfg.get("eval") or {}).get("ragas_sample") or 0)
    picked = _sample_rows(candidates, resolved_n)

    needs_rebuild = [i for i in picked if _legacy_contexts(i.get("contexts"))]
    replay_cfg = cfg
    if needs_rebuild:
        # dense/hybrid 在 retriever 构建时就被定死，必须在构建**前**注入保存的模式，
        # 否则是拿当前配置（默认 hybrid）的上下文去判 dense 跑出来的答案
        mode_match = re.search(r"\[([a-z0-9_]+)\]", meta.get("retrieval") or "")
        replay_cfg = dict(cfg)
        replay_cfg["retrieval"] = dict(cfg.get("retrieval") or {})
        if mode_match:
            replay_cfg["retrieval"]["mode"] = mode_match.group(1)
    retriever = _build_retriever(replay_cfg, meta.get("collection"))[0] if needs_rebuild else None

    rows = []
    rebuilt = 0
    mismatched: list[str] = []
    for raw in picked:
        contexts = raw.get("contexts")
        if _legacy_contexts(contexts):
            ctx = _retrieve_contexts(raw["question"], meta, retriever, cfg)
            # 自检：重放检索必须逐字复现旧文件里的正文，否则等于换了上下文再判分
            stored = [_norm(c) for c in (contexts or [])]
            got = [_norm(c["text"]) for c in ctx]
            if stored and stored != got:
                mismatched.append(raw["id"])
                continue
            contexts = _judge_contexts(ctx)
            rebuilt += 1
        rows.append(
            {
                "id": raw["id"],
                "type": raw["type"],
                "user_input": raw["question"],
                "response": raw["answer"],
                "retrieved_contexts": contexts,
            }
        )
    if mismatched:
        raise ValueError(
            f"{results_file.name} 有 {len(mismatched)} 条上下文无法按原样复现"
            f"（{', '.join(mismatched[:8])}…）——rerank 非位级可复现所致。"
            f"拿重放后的上下文去判这些答案会得到假数据，已拒绝判分。"
            f"请重新执行 `doc-rag eval` 生成与答案同源的上下文后再补跑 RAGAS。"
        )
    summary = _run_ragas(
        rows, cfg, sample_n=resolved_n, use_cache=use_cache, total=len(candidates)
    )
    if summary is None:
        return None
    summary["contexts_rebuilt"] = rebuilt
    payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source_results": results_file.name,
            "collection": meta.get("collection"),
            "retrieval": meta.get("retrieval"),
            "judge_model": cfg["llm"]["model"],
            "judge_temperature": 0,
            "judge_cache": use_cache,
        },
        "summary": summary,
    }
    dest = out_file or results_file.with_name(f"{results_file.stem}_ragas.json")
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

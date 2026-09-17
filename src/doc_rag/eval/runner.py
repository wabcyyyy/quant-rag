"""客观指标评估器（PLAN §5.3 双轨的确定性一轨）。

指标（不依赖 LLM judge，可复现）：
- Recall@5 / Recall@top_n：黄金来源文档是否被检回
- MRR：首个命中来源的排名倒数
- contains_acc：非拒答题答案包含全部 must_contain 关键词
- refusal_acc：拒答题正确拒答（出现拒答语且未编造 must_contain）
- citation_valid_rate：答案中 [n] 引用编号全部落在上下文范围内
"""

from __future__ import annotations

import json
import math
import re
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
    return _contexts(results, max_n=int(cfg["retrieval"].get("max_contexts") or 0) or None)


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
        plan = (
            rewriter.rewrite(item.question)
            if rewriter
            else {"rewritten": item.question, "filters": None, "aggregate": aggregate, "top_n": None}
        )
        results = retriever.retrieve(
            plan["rewritten"],
            top_n=top_n,
            filters=plan["filters"],
            aggregate=plan["aggregate"] or aggregate,
        )
        results = _maybe_rerank(cfg, plan["rewritten"], results, use_rerank)
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
        answer = (
            synthesizer.answer(item.question, ctx, require_citation=require_citation)
            if with_answers
            else ""
        )

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

    ragas_summary = None
    if with_ragas:
        ragas_summary = _run_ragas(
            _ragas_rows(items, per_item), cfg, sample_n=ragas_sample
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
        },
        "summary": summary,
        "ragas": ragas_summary,
        "items": per_item,
    }
    return results


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

        def on_llm_end(self, response, **kwargs) -> None:
            self.calls += 1
            llm_output = getattr(response, "llm_output", None) or {}
            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            if not usage:
                for gen_list in getattr(response, "generations", None) or []:
                    for gen in gen_list:
                        meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
                        if meta:
                            usage = meta
            self.prompt_tokens += int(
                usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            )
            self.completion_tokens += int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )

        def as_dict(self) -> dict:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }

    return _TokenCounter()


def _run_ragas(
    rows: list[dict],
    cfg: dict,
    sample_n: int | None = None,
    use_cache: bool = True,
) -> dict | None:
    """RAGAS 第二轨：rows=[{id,type,user_input,response,retrieved_contexts}] → judge 指标。

    可信度口径（PLAN §5.3）：judge 固定模型、temperature=0；只看与客观指标的相对一致性。
    `use_cache=False` 用于测 judge 自身的运行间随机性（temperature=0 也不保证跨请求逐字复现）。
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
    all_n = len(rows)
    rows = _sample_rows(rows, sample_n)

    # judge 是最大调用方（每指标每条多次内部调用），必须走缓存：
    # langchain 有自己的缓存层，不复用 llm.chat 的缓存（PLAN §8 成本控制）
    judge_cache_path = None
    if use_cache and cache_enabled(cfg.get("llm")):
        try:
            from ..generate.llm import _CACHE_PATH as _llm_cache_path

            judge_cache_path = str(_llm_cache_path.parent / "judge_cache.sqlite")
            set_llm_cache(SQLiteCache(database_path=judge_cache_path))
        except Exception:  # noqa: BLE001 缓存设置失败不影响评估
            judge_cache_path = None
    else:
        set_llm_cache(None)  # 显式关缓存：set_llm_cache 是进程级全局，必须清掉

    # 指标可选（PLAN §8 成本控制）：AnswerRelevancy 在中文场景噪声大且需嵌入调用
    wanted = [m.lower() for m in (cfg.get("eval", {}).get("ragas_metrics") or ["faithfulness"])]
    metric_map = {"faithfulness": Faithfulness(), "answer_relevancy": AnswerRelevancy()}
    metrics = [metric_map[m] for m in wanted if m in metric_map]
    if not metrics:
        return {"skipped": f"未配置有效指标：{wanted}"}

    llm_cfg = cfg["llm"]
    judge = LangchainLLMWrapper(
        ChatOpenAI(
            model=llm_cfg["model"],
            base_url=llm_cfg["base_url"],
            api_key=llm_cfg["api_key"],
            temperature=0,
        )
    )
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

    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    cfg = cfg or load_config()
    data = json.loads(results_file.read_text(encoding="utf-8"))
    item = next((i for i in data["items"] if i["id"] == item_id), None)
    if item is None:
        return {"error": f"{results_file.name} 里没有 {item_id}"}

    metric = Faithfulness()
    metric.llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=cfg["llm"]["model"],
            base_url=cfg["llm"]["base_url"],
            api_key=cfg["llm"]["api_key"],
            temperature=0,
        )
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
    """对已保存的评估结果补跑 RAGAS：答案复用，上下文缺失/口径过期时免费重检索。

    结果落盘（`<results>_ragas.json`）：此前只打印到控制台，导致 PLAN 里的
    Faithfulness 数字无法追溯到逐条分数，也就无法回答「差异是题型抽样还是 judge 噪声」。
    """
    cfg = cfg or load_config()
    data = json.loads(results_file.read_text(encoding="utf-8"))
    meta = data.get("meta") or {}
    retriever, _ = _build_retriever(cfg, meta.get("collection"))
    rows = []
    rebuilt = mismatched = 0
    for raw in data["items"]:
        if raw["type"] == "no_answer" or not raw.get("answer"):
            continue
        contexts = raw.get("contexts")
        if _legacy_contexts(contexts):
            ctx = _retrieve_contexts(raw["question"], meta, retriever, cfg)
            # 自检：重放检索必须逐字复现旧文件里的正文，否则等于换了上下文再判分
            stored = [_norm(c) for c in (contexts or [])]
            got = [_norm(c["text"]) for c in ctx]
            if stored and stored != got:
                mismatched += 1
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
    summary = _run_ragas(rows, cfg, sample_n=sample_n, use_cache=use_cache)
    if summary is None:
        return None
    summary["contexts_rebuilt"] = rebuilt
    summary["contexts_mismatched"] = mismatched
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

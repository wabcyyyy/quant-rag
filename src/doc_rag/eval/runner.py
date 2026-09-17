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
import re
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient

from ..config import load_config
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
        answer = synthesizer.answer(item.question, ctx) if with_answers else ""

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
                "contexts": [c["text"] for c in ctx],
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
        rows = [
            {
                "user_input": item.question,
                "response": r["answer"],
                "retrieved_contexts": r.get("contexts") or [],
            }
            for item, r in zip(items, per_item)
            if not item.refusable and r["answer"]
        ]
        # 成本杠杆：RAGAS 是调用大户（每指标每条 ≈1 次 judge 调用），
        # 抽样即可校准趋势（PLAN §5.3：只报相对变化）。0 或未设 = 全量。
        sample_n = int((cfg.get("eval") or {}).get("ragas_sample") or 0)
        if sample_n and sample_n < len(rows):
            rows = rows[:sample_n]
        ragas_summary = _run_ragas(rows, cfg)

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


def _run_ragas(rows: list[dict], cfg: dict) -> dict | None:
    """RAGAS 第二轨：rows=[{user_input, response, retrieved_contexts}] → judge 指标。

    可信度口径（PLAN §5.3）：judge 固定模型、temperature=0；只看与客观指标的相对一致性。
    """
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, evaluate as ragas_evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, Faithfulness
    except Exception as exc:  # noqa: BLE001
        return {"error": f"ragas 不可用：{exc}"}

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
    try:
        ds = EvaluationDataset.from_list(rows)
        out = ragas_evaluate(
            dataset=ds,
            metrics=metrics,
            llm=judge,
            embeddings=embeddings,
        )
        df = out.to_pandas()
        summary = {"n": len(rows), "metrics": [m.name for m in metrics]}
        for m in metrics:
            if m.name in df.columns:
                summary[m.name] = round(float(df[m.name].mean()), 4)
        return summary
    except Exception as exc:  # noqa: BLE001
        return {"error": f"ragas 运行失败：{exc}"}


def ragas_from_results(results_file: Path, cfg: dict | None = None) -> dict | None:
    """对已保存的评估结果补跑 RAGAS：答案复用，上下文缺失时免费重检索。"""
    cfg = cfg or load_config()
    data = json.loads(results_file.read_text(encoding="utf-8"))
    retriever, _ = _build_retriever(cfg, data["meta"].get("collection"))
    rows = []
    for item in data["items"]:
        if item["type"] == "no_answer" or not item.get("answer"):
            continue
        contexts = item.get("contexts")
        if not contexts:
            results = retriever.retrieve(item["question"], top_n=6)
            contexts = [r["text"] for r in results]
        rows.append(
            {
                "user_input": item["question"],
                "response": item["answer"],
                "retrieved_contexts": contexts,
            }
        )
    return _run_ragas(rows, cfg)

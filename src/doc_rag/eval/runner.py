"""客观指标评估器（PLAN §5.3 双轨的确定性一轨）。

指标（不依赖 LLM judge，可复现）：
- Hit@5 / Hit@8 / Hit@清单末：gold 来源文档的**首个**命中落在哪一位之前
  （旧名 `recall_at_k` 是误称：64 条可答题里 44 条只有 1 篇 gold，剩下 20 条
  gold 有 10~56 篇，在 8~25 槽位上真 Recall 不可能高，报出来的其实是命中率）
- 文档覆盖率：逐条 `|命中∩gold| / |gold|` 再取均值——这才是真 Recall（macro）
  它的结构上限 `coverage_ceiling_mean` 一起报，低于上限的差才是系统的真实空间
- MRR：首个命中来源的排名倒数；nDCG@8：二值相关、按文档去重的整段排序质量
- contains_acc：非拒答题答案包含全部 must_contain 关键词
  （另有一路 `contains_acc_subseq`：字符子序列匹配，假阳性无上界，见
  `_contains_as_subsequence`——它只在「与严格口径不同值」时才提供信息）
- keypoint_hit_ratio：聚合题的**分档**答案命中——答案命中了几条逐篇要点 / K。
  与 contains_acc 并列而不是替换它：聚合题原先只有 1 个 must_contain（那个人名），
  而答案集 10~56 篇，「答出 2 篇」与「答出 18 篇」同分，上下文预算消融读不出差。
  它是纯字符串判据，**不依赖清单长度**，所以是 E2 两臂块数不等时唯一干净的指标
- refusal_acc：拒答题答案出现拒答措辞（**只查措辞**，见 over_refusal 那条口径；
  「拒答里有没有编造」另有一条独立检查，在 `eval/refusal.py` / `doc-rag audit-refusals`）
- citation_valid_rate / citation_presence_rate
- 延迟：分阶段（改写/检索/重排/合成/端到端）p50/p95/max，按题型拆分

延迟口径（PLAN「延迟口径」）：合成耗时只在**缓存未命中**时才算延迟；命中缓存
返回的是本地查询耗时。只要本轮有命中，`latency.cache_contaminated` 置真，
提示 LLM 延迟被低估——测真实延迟必须关合成缓存。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient

from ..agent import agent_cfg
from ..config import load_config
from ..generate import prompts
from ..generate.llm import cache_enabled
from ..generate.synthesizer import Synthesizer
from ..ingest.embedder import Embedder
from ..orchestrator import Orchestrator
from ..retrieve.hybrid import HybridRetriever
from ..retrieve.rewrite_llm import endpoint_model
from .judge import judge_cfg
from .schema import LEGACY_SUMMARY_KEYS, GoldItem

# 「答全」的门槛：命中 ≥ 80% 的要点。这个数是**拍的**，没有校准过——所以它只做展示
# 分档，不进任何门禁；能被当作结论的是 `keypoint_hit_ratio` 本身。
_GRADE_FULL_RATIO = 0.8

_REFUSAL_MARKERS = [
    "无法回答",
    "无法确定",
    "未形成决议",
    "未记载",
    "未提到",
    "没有提到",
    "缺少",
    "没有足够",
    "未能找到",
    "根据现有文档",
    "没有讨论",
    "未讨论",
]
_CITATION_RE = re.compile(r"\[(\d+)\]")

#: 标准 IR 读数的**唯一截断**。选 5 的理由见 `_rank_metrics` 的 docstring
#: （LLM 只读 6 块；@8 里有两三格模型没看过；@10 在 8 格清单上是假数据点）。
_STD_K = 5


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _build_retriever(
    cfg: dict, collection: str | None
) -> tuple[HybridRetriever, Synthesizer]:
    retriever = HybridRetriever(
        client=QdrantClient(url=cfg["qdrant"]["url"], timeout=60),
        embedder=Embedder(cfg["embedding"]),
        collection=collection or cfg["qdrant"]["collection"],
        retrieval_cfg=cfg["retrieval"],
    )
    return retriever, Synthesizer(cfg["llm"])


def _build_orchestrator(cfg: dict, collection: str | None):
    """把 `_build_retriever` 的产物注入 Orchestrator。

    eval 是顺序执行，复用同一个 Synthesizer 没有跨请求串台风险；测试也只需 patch
    `_build_retriever` 这一处缝就能让整条链路离线。
    """
    retriever, synthesizer = _build_retriever(cfg, collection)
    return Orchestrator(cfg, retriever=retriever, synthesizer=synthesizer)


def _refusal_ok(answer: str) -> bool:
    return any(marker in answer for marker in _REFUSAL_MARKERS)


def _rank_metrics(got_ids: list[str], gold: set[str], k: int = _STD_K) -> dict:
    """业界标准读数的**唯一计算处**：Recall@k、Precision@k、AP@k（k 固定 = 5）。

    为什么以前只能报 Hit@k：逐条只存了「首命中位次」，而 Recall@k 需要前 k 个槽位里
    **命中了几篇**——首命中一个数答不了「gold 有 56 篇时前 5 位捞回几篇」。当年
    `recall_at_k` 被改名成 `hit_at_k` 就是因为名字说了假话（护栏在
    `tests/test_eval_metric_definitions.py` 第 1 条）；这一份是真的 Recall，
    代价是 `retrieved_doc_ids` 必须逐条落盘，否则事后重算不出来。

    **为什么 K 只取 5 而不做扫描**：一个截断一套数，读的人不必再猜哪列对哪列。
    5 也是这条链路上唯一有意义的标准点——LLM 实际只读 6 块（`rerank.top_n`），
    @8 里有 2~3 格模型根本没看过。曾经扫过 1/3/5/10，其中 **@10 是假数据点**：
    清单只有 8 格，`recall_at_10` 在 64/64 条上逐字等于整条清单的召回，
    看起来像一个独立测量点其实不是。k 超过清单长度的读数一律不出。

    **precision 的分母是 k，不是清单条数**：早期版本写的是「命中的不同文档数 ÷ 块数」，
    分子按文档、分母按块，单位不一致，于是单文档题的满分只有 1/8 = 0.125——
    那个数读起来像质量差，其实是量纲错。现在与 recall 同用**去重后的文档排名**。
    """
    if not gold:
        return {}
    ranked = list(dict.fromkeys(got_ids))  # 去重保序：同一篇的第二个块不再占位
    top = ranked[:k]
    hits = len(set(top) & gold)
    num = 0.0
    seen = 0
    for i, doc_id in enumerate(top, start=1):
        if doc_id in gold:
            seen += 1
            num += seen / i
    return {
        f"recall_at_{k}": round(hits / len(gold), 4),
        f"precision_at_{k}": round(hits / k, 4),
        f"ap_at_{k}": round(num / min(len(gold), k), 4),
    }


def _contains_as_subsequence(keyword: str, answer: str) -> bool:
    """字符子序列匹配：关键词的每个字按序出现即算命中。

    比「整理纪要」vs「整理会议纪要」这种容忍更宽得多——假阳性**没有上界**：
    「通过决议」能在一段毫无关系的话里命中（只要「通」「过」「决」「议」依次出现）。
    名字里带 subsequence 而不是 loose，是为了让这层语义在报告里可见。
    """
    pos = 0
    target = _norm(keyword)
    hay = _norm(answer)
    for ch in hay:
        if pos < len(target) and ch == target[pos]:
            pos += 1
    return pos == len(target)


def _ndcg_at_k(got_ids: list[str], relevant: set[str], k: int = 8) -> float | None:
    """二值相关 nDCG@k，与 recall/mrr 同口径：来源文档=相关，排除无来源题。

    检索结果按 doc 去重取首个命中位次（同一文档的多个块不重复计贡献——
    与 first_hit_rank 的口径一致）；IDCG 按相关文档数截断到 k。
    消融 #3 此前只有 MRR（只看首个命中），nDCG 补上「命中越多越靠前越好」
    的梯度——重排的价值本就该体现在整段排序质量上。
    """
    dcg = 0.0
    seen: set[str] = set()
    rank = 0
    for doc_id in got_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rank += 1
        if rank > k:
            break
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return None
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg


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
    retriever: HybridRetriever | None,
    cfg: dict,
    aggregate: bool = False,
    recorded: dict | None = None,
) -> list[dict]:
    """按 meta 记录的重检索参数重放检索，还原 LLM 实际看到的编号上下文。

    旧结果文件只存了正文；要补文档名必须重放**同一套**参数（改写 / 聚合 / 重排 /
    上下文上限），否则 judge 拿到的是另一批块——等于用 A 的上下文去判 B 的答案。
    meta.retrieval 形如 `dense+bm25+rrf[hybrid]+rewrite+rerank`。

    改写自 W3 起由模型产出、不再可复现，因此重放只读条目里记下的 `rewritten`，
    绝不当场重跑改写：拿新一次改写的结果去配旧答案，就是把度量对象换掉了。
    """
    assert retriever is not None  # 只在需要重建时才构建（构建它要花钱）
    if recorded and recorded.get("trace") is not None:
        # agent 条目的上下文是「逐步判定 + 逐步检索」的产物，重放检索只能还原出
        # 第一步那份清单——用它判旧答案等于换了度量对象。走到这里说明这份文件的
        # 上下文不是编号版（旧格式），那就没有可重建的正确路径。
        raise ValueError(
            f"条目「{question[:24]}…」带 agent trace，而它的落盘上下文不是 LLM 实际"
            "看到的那份：agent 的并集无法靠重放单次检索还原。请重跑 "
            "`doc-rag eval --agent`（结果文件会带编号上下文）。"
        )
    flags = meta.get("retrieval") or ""
    use_rerank = "+rerank" in flags
    force_agg = aggregate or "+aggregate" in flags
    recorded = recorded or {}
    override: dict | None = None
    if "+rewrite" in flags:
        if not recorded.get("rewritten"):
            raise ValueError(
                f"条目「{question[:24]}…」的旧结果文件没记改写后的检索串，"
                "而实时改写不可复现——不能拿另一条查询检索出的上下文去判旧答案。"
                "请重跑 `doc-rag eval`。"
            )
        override = {
            "rewritten": recorded["rewritten"],
            "filters": recorded.get("rewrite_filters"),
            "aggregate": bool(recorded.get("rewrite_aggregate")) or force_agg,
            "top_n": None,
            "reason": "重放已记录的改写结果",
            "degraded": False,  # 重放不是「改写失败」，别把重放计成退化
        }
    result = Orchestrator(cfg, retriever=retriever, synthesizer=None).answer(
        question,
        top_n=meta.get("top_n") or 8,
        use_rewrite=False,
        use_rerank=use_rerank,
        force_aggregate=force_agg,
        with_answer=False,
        plan_override=override,
        # 预算口径也要照原样重放：旧文件是 `budget: rewrite` 时，用固定 top_n
        # 还原出来的上下文与 LLM 当时看到的那份不是同一批块。
        honor_rewrite_budget=meta.get("budget") == "rewrite",
    )
    return result.contexts


def evaluate(
    gold_file: Path,
    cfg: dict | None = None,
    collection: str | None = None,
    top_n: int = 8,
    limit: int | None = None,
    sample: int | None = None,
    with_ragas: bool = False,
    with_answers: bool = True,
    mode: str | None = None,
    agent_mode: str | None = None,
    aggregate: bool = False,
    use_rewrite: bool = False,
    use_rerank: bool = False,
    require_citation: bool = True,
    ragas_sample: int | None = None,
    use_judge_cache: bool = True,
    honor_rewrite_budget: bool = False,
    judge_over: dict | None = None,
) -> dict:
    cfg = cfg or load_config()
    if mode:
        cfg["retrieval"]["mode"] = mode  # 消融开关：dense / hybrid
    orchestrator = _build_orchestrator(cfg, collection)
    retriever = orchestrator.retriever  # 结果文件 meta 自证 collection / 检索模式用
    payload = json.loads(gold_file.read_text(encoding="utf-8"))
    items_raw = payload["items"]
    if limit:
        items_raw = items_raw[:limit]
    if sample:
        # 抽样必须均匀：黄金集按题型分块排序，取前 N 条会整段漏掉末尾题型
        # （cross_doc / time_filter 全在表尾）——RAGAS 轨当年就是这么漏的。
        items_raw = _sample_rows(items_raw, sample)
    items = [GoldItem.model_validate(i) for i in items_raw]

    per_item: list[dict] = []
    for item in items:
        result = orchestrator.answer(
            item.question,
            top_n=top_n,
            use_rewrite=use_rewrite,
            use_rerank=use_rerank,
            force_aggregate=aggregate,
            require_citation=require_citation,
            with_answer=with_answers,
            honor_rewrite_budget=honor_rewrite_budget,
            # `agent_mode` 与上面那个 `mode` 不是一回事：后者是检索模式（dense/hybrid），
            # 前者是 single/agent。名字分开是因为这两个词在项目里都出现过，混用一次
            # 就会让评估臂悄悄换掉 policy 而 meta 上还自称同一条。
            mode=agent_mode,
            # 真题型只进 trace 做「预测准不准」的核对，不参与开关（见 agent.predict_type）
            question_type=item.type,
        )
        results = result.retrieved
        ctx = result.contexts
        got_ids = [r["doc_id"] for r in results]
        hit_ranks = [got_ids.index(s) + 1 for s in item.source_doc_ids if s in got_ids]
        first_rank = min(hit_ranks) if hit_ranks else None
        gold = set(item.source_doc_ids)
        # 文档覆盖率：聚合题（答案集 5~56 篇）真正该量的指标——它就是逐条真 Recall
        coverage = len(set(got_ids) & gold) / len(gold) if gold else None
        # 同一清单长度下覆盖率的结构性上限：清单只有 8 格而 gold 有 56 篇时，
        # 0.143 就是满分。不把这个数一起报出来，0.18 与 0.96 都会被读成同一回事。
        coverage_ceiling = min(len(gold), len(results)) / len(gold) if gold else None
        ndcg = _ndcg_at_k(got_ids, gold, k=8) if gold else None
        # 与标准族同截断的 nDCG（@8 是已发布基线，保留；@5 才和上面那三个数同 k）
        ndcg_std = _ndcg_at_k(got_ids, gold, k=_STD_K) if gold else None
        rank_std = _rank_metrics(got_ids, gold)

        synth_meta = result.synth_meta
        answer = result.answer

        if not with_answers:
            answered_ok = None  # 检索模式不评回答
            answered_ok_subseq = None
            over_refusal = None
        elif item.refusable:
            answered_ok = _refusal_ok(answer)
            answered_ok_subseq = answered_ok
            over_refusal = None
        elif item.must_contain:
            answered_ok = all(_norm(m) in _norm(answer) for m in item.must_contain)
            answered_ok_subseq = all(
                _contains_as_subsequence(m, answer) for m in item.must_contain
            )
            # 过度拒答：声称无法回答，但上下文里其实含有关键信息。
            # `not answered_ok` 是必须的：收紧后的 prompt 要求模型说明「文档里没记载
            # 什么」，于是正确答案也常带拒答措辞。2026-09-19 全量重跑实测：被标记的
            # 26 条**全部**答对（over_refusal_rate 0.406 是 100% 假阳性）。
            # 一个恒真的指标比一个偏高的指标更糟——它会让人去修一个不存在的问题。
            ctx_text = _norm(" ".join(c["text"] for c in ctx))
            ctx_has = all(_norm(m) in ctx_text for m in item.must_contain)
            over_refusal = bool(_refusal_ok(answer) and ctx_has and not answered_ok)
        else:
            answered_ok = None  # 无判据 → 不计入 answer 准确率（当前 72 条里只有 8 条
            # no_answer 走到这里：聚合题是有 must_contain 的，只是只有 1 个词，所以才
            # 另设下面的分档口径。旧注释写成「如部分聚合题」是过时的，会误导人去找
            # 一个并不存在的判据缺口。
            answered_ok_subseq = None
            over_refusal = None

        # 逐篇要点命中（分档）。与 answered_ok 完全独立：不共用判据、不改它的分母，
        # 旧口径逐字不动。`item.key_points` 为空（旧黄金集、非聚合题）时判 None，
        # 不进任何均值——把「未定义」印成「测出来是 0」是这个项目明令禁止的那类错误。
        if not with_answers or not item.key_points:
            keypoint_hit = None
            answer_grade = None
            n_key_points = len(item.key_points) or None
        else:
            hits = sum(1 for kp in item.key_points if kp.hit_in(answer))
            k = len(item.key_points)
            keypoint_hit = round(hits / k, 4)
            answer_grade = (
                "full"
                if hits >= math.ceil(_GRADE_FULL_RATIO * k)
                else ("half" if hits else "zero")
            )
            n_key_points = k

        refs = [int(n) for n in _CITATION_RE.findall(answer)]
        citation_valid = all(1 <= n <= len(ctx) for n in refs) if refs else None
        citation_present = bool(refs) if answer else None

        # 过度拒答的第二条口径：gold 文档已经**进了上下文**，答案却说「无法回答」。
        # 上面的 `over_refusal` 要求 must_contain 逐字出现在块里，因而漏掉一整类
        # 真过度拒答——检回了正确文档、但那块没含那句原话。2026-09-19 全量实测：
        # q018/q026/q034 首命中在第 1/3/2 位、doc_coverage=1.0，答案仍是
        # 「根据现有文档无法回答」，而 `over_refusal_rate` 报 0.0。
        # 两条都报：`over_refusal_rate` 是「上下文含原话」口径，
        # `over_refusal_gold_rate` 是「正确文档已进上下文」口径。
        ctx_doc_ids = {c["doc_id"] for c in result.citations}
        over_refusal_gold = (
            None
            if (not with_answers or not answer or item.refusable)
            else bool(
                answered_ok is False and _refusal_ok(answer) and (ctx_doc_ids & gold)
            )
        )

        # 逐条用量（T5）：延迟数字此前只记耗时没记用量，无法解释「term 单条 28s
        # 但答案仅 308 字」——reasoning token 与耗时必须能对上号。缓存命中时
        # usage 为 None（缓存的答案当时没记用量），合计时按缺失跳过。
        usage = None
        if synth_meta:
            u = {
                k: synth_meta.get(k)
                for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")
            }
            if any(v is not None for v in u.values()):
                usage = u

        lat = result.latency_ms
        per_item.append(
            {
                "id": item.id,
                "type": item.type,
                "question": item.question,
                "first_hit_rank": first_rank,
                "doc_coverage": round(coverage, 4) if coverage is not None else None,
                "doc_coverage_ceiling": (
                    round(coverage_ceiling, 4) if coverage_ceiling is not None else None
                ),
                "ndcg_at_8": round(ndcg, 4) if ndcg is not None else None,
                "ndcg_at_5": round(ndcg_std, 4) if ndcg_std is not None else None,
                # 标准读数（Recall@5 / Precision@5 / AP@5）与 ** ranked 清单**：
                # 没有后者，这些数事后重算不出来，而它们正是「Hit@k 之外还能不能说
                # Recall」的前提。无 gold 的条目（no_answer）这里是 null。
                "retrieved_doc_ids": list(dict.fromkeys(got_ids)),
                **rank_std,
                "n_source_docs": len(item.source_doc_ids),
                # 清单与上下文的长度必须逐条可见：消融臂之间若清单不等长，
                # hit/nDCG/覆盖率的差就部分是长度的函数（重排以前正是如此）。
                "n_retrieved": len(results),
                "n_contexts": len(ctx),
                # agent 臂的每一步都必须可追。判定是模型产出、不可复现的，和
                # `rewritten` 同一个性质：不落 trace 的 agent 条目就没有重放资格
                # （见 `_retrieve_contexts` 里那条拒绝）。单发条目恒为 null。
                "trace": result.trace,
                "top_n_used": result.top_n_used,
                "rewrite_top_n": result.rewrite_top_n,
                "filter_applied": result.filter_applied,
                "filter_fallback": result.filter_fallback,
                "answered_ok": answered_ok,
                "answered_ok_subseq": answered_ok_subseq,
                # 分档判据：`n_key_points` 必须逐条落盘，否则两臂的 K 是不是同一个
                # 分母只能靠回忆（和 `n_retrieved` 同一条理由）。
                "keypoint_hit": keypoint_hit,
                "n_key_points": n_key_points,
                "answer_grade": answer_grade,
                "over_refusal": over_refusal,
                "over_refusal_gold": over_refusal_gold,
                "citation_valid": citation_valid,
                "citation_present": citation_present,
                "n_citations": len(refs),
                "answer": answer,
                "rerank_error": result.rerank_error,
                # 改写结果必须随条目落盘：改写自 W3 起由模型产出、不可复现，
                # 事后重放检索还原 judge 上下文时只能读这里记下的那一条。
                "rewritten": result.plan["rewritten"],
                "rewrite_filters": result.plan["filters"],
                "rewrite_aggregate": bool(result.plan["aggregate"]),
                "rewrite_degraded": bool(result.plan.get("degraded")),
                "contexts": _judge_contexts(ctx),
                "latency": {
                    "rewrite": lat["rewrite"],
                    "retrieve": lat["retrieve"],
                    "rerank": lat["rerank"],
                    # 检索侧不含 LLM：这部分与模型无关，换模型不必重测
                    "retrieval_total": lat["retrieval_total"],
                    "synthesize": lat["synthesize"],
                    "synth_cached": lat["synth_cached"],
                    "usage": usage,
                    "total": lat["total"],
                },
            }
        )

    # 聚合（no_answer 题无来源文档，不计入分母，由 refusal_acc 单独评）
    with_source = [r for r in per_item if not _is_refusable(items, r["id"])]
    mrr_scores = [1.0 / r["first_hit_rank"] for r in with_source if r["first_hit_rank"]]
    ndcg_scores = [r["ndcg_at_8"] for r in with_source if r["ndcg_at_8"] is not None]
    covs = [r["doc_coverage"] for r in per_item if r["doc_coverage"] is not None]
    ceilings = [
        r["doc_coverage_ceiling"]
        for r in per_item
        if r["doc_coverage_ceiling"] is not None
    ]
    # 整条清单口径的归一化召回（逐条比值取均值，**不是两个均值相除**）
    rvc = [
        r["doc_coverage"] / r["doc_coverage_ceiling"]
        for r in per_item
        if r.get("doc_coverage") is not None and r.get("doc_coverage_ceiling")
    ]

    def _hit_rate(k: int | None) -> float | None:
        """首命中落在前 k 位的条目占比；`k=None` = 整条清单内任一位。

        这量的是「找没找到」，不是「找全没有」——找全的程度看 `mean_doc_coverage`。
        """
        if not with_source:
            return None
        n = sum(
            1
            for r in with_source
            if r["first_hit_rank"] and (k is None or r["first_hit_rank"] <= k)
        )
        return round(n / len(with_source), 4)

    def _mean_by_type(key: str) -> dict[str, float]:
        """按题型取均值；**没有值的题型一个 key 都不建**。

        给 `no_answer`（按定义无 gold、覆盖率恒 None）编一个 0.0，就是把「未定义」
        印成「测出来是 0」——和那个恒真的 over_refusal 是同一类错误。
        取值用 `.get()` 而不是 `[key]`：标准族那几个键在无 gold 的条目上是**整个
        不存在**（`_rank_metrics` 返回空 dict），按 `[key]` 取会 KeyError 而不是跳过。
        """
        out: dict[str, float] = {}
        for t in sorted({r["type"] for r in per_item}):
            vals = [
                r[key] for r in per_item if r["type"] == t and r.get(key) is not None
            ]
            if vals:
                out[t] = round(sum(vals) / len(vals), 4)
        return out

    def _macro_mean(key: str) -> float | None:
        """有 gold 条目上的 macro 均值；该键缺失（旧文件）或全 None 时返回 None。"""
        vals = [r[key] for r in with_source if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    scorable = [
        r
        for r in per_item
        if not _is_refusable(items, r["id"]) and r["answered_ok"] is not None
    ]
    refusables = [r for r in per_item if _is_refusable(items, r["id"])]
    cites = [r for r in per_item if r["citation_valid"] is not None]
    # 分档命中的分母 = 真有 key_points 的那些条目。它必须和均值一起报，否则
    # 「20 条聚合题的 0.31」与「4 条题的 0.31」读起来是同一个数。
    kp_rows = [r for r in per_item if r["keypoint_hit"] is not None]
    ks = [r["n_key_points"] for r in kp_rows if r["n_key_points"]]
    grades = Counter(r["answer_grade"] for r in kp_rows if r["answer_grade"])

    lens = [r["n_retrieved"] for r in per_item]
    summary = {
        "n_items": len(per_item),
        # ↓ 标准名是主键。`hit_at_*` 保留这个名字是因为它**确实**是 Hit Rate@k
        # （首命中在前 k 位），不是 Recall——这个区别当年被搞错过一次，护栏见
        # `tests/test_eval_metric_definitions.py` 第 1 条。
        "hit_at_5": _hit_rate(5),
        "hit_at_8": _hit_rate(8),
        "hit_at_list": _hit_rate(None),
        "mrr": sum(mrr_scores) / len(with_source),
        # nDCG@8（二值相关，doc 去重）：整段排序质量，与 hit/mrr 同分母
        "ndcg_at_8": round(sum(ndcg_scores) / len(ndcg_scores), 4)
        if ndcg_scores
        else None,
        # 逐条真 Recall（macro，整条清单口径）。它必须和自己的上限一起读：
        # 清单 8 格、gold 56 篇的那种条目，上限 0.143 就是满分。
        "recall_at_list_macro": round(sum(covs) / len(covs), 4) if covs else None,
        "recall_ceiling_macro": round(sum(ceilings) / len(ceilings), 4)
        if ceilings
        else None,
        "recall_by_type": _mean_by_type("doc_coverage"),
        "recall_ceiling_by_type": _mean_by_type("doc_coverage_ceiling"),
        # ↓ 标准族：**同一个截断 K=5** 的一套数（@8 那几列是已发布基线，原样保留）。
        # 与 `hit_at_*` 的区别是「捞回几篇 / 几格相关」而不是「捞回没捞回」。
        "recall_at_5": _macro_mean("recall_at_5"),
        "precision_at_5": _macro_mean("precision_at_5"),
        "ndcg_at_5": _macro_mean("ndcg_at_5"),
        "map_at_5": _macro_mean("ap_at_5"),
        # 分题型是这套数唯一有讲相法的地方：单文档题的 Precision@5 = 0.2 是
        # 「5 格里 1 格相关」的教科书值，而 cross_doc 的 0.95 说明槽位几乎没浪费
        # ——两个 0.2 与 0.95 混在全量均值里，谁也看不见谁。
        "recall_at_5_by_type": _mean_by_type("recall_at_5"),
        "precision_at_5_by_type": _mean_by_type("precision_at_5"),
        # 归一化召回。K=5 窗口版不另列：#gold ≥ 5 时它与 Precision@5 是同一个数，
        # 列两遍只会让人以为是两件事。它必须与 Recall 并排读——cross_doc 的 gold
        # 中位 37 篇，5 格的天花板只有 0.135，裸读 Recall@5 会把检索判成失败。
        "recall_vs_ceiling_macro": round(sum(rvc) / len(rvc), 4) if rvc else None,
        # 两臂可比性的自证：清单长度必须逐条落盘，不然「有重排」臂悄悄短一截
        "list_len": {
            "retrieved_min": min(lens) if lens else None,
            "retrieved_max": max(lens) if lens else None,
            "contexts_min": min((r["n_contexts"] for r in per_item), default=None),
            "contexts_max": max((r["n_contexts"] for r in per_item), default=None),
        },
        "strict_keyword_accuracy": _safe_div(
            sum(1 for r in scorable if r["answered_ok"]), len(scorable)
        ),
        # 字符子序列口径（见 `_contains_as_subsequence`）：与严格值一起看，差值即度量
        # 口径的松紧。2026-09-19 全量实测两者**同值**（都 0.8438）→ 这一路当前不提供信息，
        # 留着是因为它能证伪「严格口径在惩罚措辞改写」，不是因为它是独立信号。
        "subseq_keyword_accuracy": _safe_div(
            sum(1 for r in scorable if r["answered_ok_subseq"]), len(scorable)
        ),
        # 聚合题的分档命中（macro，只看有 key_points 的条目）。它是纯字符串判据，
        # 不受清单/上下文块数影响 → 上下文预算消融（E2）两臂块数不等时，这是唯一
        # 不用先扣除长度伪影的答案级读数。
        "keypoint_recall_macro": _safe_div(
            sum(r["keypoint_hit"] for r in kp_rows), len(kp_rows)
        ),
        "keypoint_recall_by_type": _mean_by_type("keypoint_hit"),
        "keypoint_n_items": len(kp_rows),
        "keypoint_k": {
            "min": min(ks) if ks else None,
            "max": max(ks) if ks else None,
            "mean": round(sum(ks) / len(ks), 2) if ks else None,
        },
        # 三值分档只作展示（阈值未校准，见 `_GRADE_FULL_RATIO`），不进门禁
        "answer_grade_counts": {g: grades.get(g, 0) for g in ("full", "half", "zero")},
        # 过度拒答：两条口径，缺任何一条都会把问题看漏一半
        "over_refusal_rate": _safe_div(
            sum(1 for r in per_item if r["over_refusal"]),
            sum(1 for r in per_item if r["over_refusal"] is not None),
        ),
        "over_refusal_gold_rate": _safe_div(
            sum(1 for r in per_item if r["over_refusal_gold"]),
            sum(1 for r in per_item if r["over_refusal_gold"] is not None),
        ),
        # 分母只算**真判过**的拒答题：`--retrieval-only` 下八条拒答题的 answered_ok
        # 全是 None，用 len(refusables) 当分母会把「没测」印成 0.0——正是本文件
        # 已经禁止过两次的那类错误（coverage_by_type 不给 no_answer 编 0.0、
        # over_refusal 的 None 不参与）。
        "refusal_acc": _safe_div(
            sum(1 for r in refusables if r["answered_ok"]),
            sum(1 for r in refusables if r["answered_ok"] is not None),
        ),
        "citation_valid_rate": _safe_div(
            sum(1 for r in cites if r["citation_valid"]), len(cites)
        ),
        # 引用存在率：无引用也算未遵守（否则「从不引用」的模型会显示 None 而非 0）
        "citation_presence_rate": _safe_div(
            sum(1 for r in per_item if r["citation_present"]),
            sum(1 for r in per_item if r["citation_present"] is not None),
        ),
    }
    # 旧名从标准名派生（见 `schema.LEGACY_SUMMARY_KEYS`）：外部脚本与历史工具还能读旧键，
    # 而两个名字永远同值——写两遍才会漂移，派生不会。
    for _legacy, _canonical in LEGACY_SUMMARY_KEYS.items():
        if _canonical in summary:
            summary[_legacy] = summary[_canonical]
    latency = _latency_summary(per_item)
    if latency:
        summary["latency"] = latency

    # 生效的 prompt 版本：meta 必须能区分「基线 / 收紧」两组答案——三组对照
    # 当年就死在这里。指纹按生效版本算，两个版本指纹必然不同。
    llm_section = cfg.get("llm") or {}
    prompt_version = llm_section.get("prompt_version") or prompts.DEFAULT_PROMPT_VERSION
    # 重排失败必须改口径，不能让结果文件继续自称「+rerank」：改造前 `_maybe_rerank`
    # 把所有异常吞成「退回融合顺序」，于是一次重排服务抖动就产出一份
    # 自称 A 组、实为 B 组的评估——正是 README 里宣布已消灭的那类度量伪影。
    rerank_failed = sum(1 for r in per_item if r.get("rerank_error"))
    if use_rerank and per_item and rerank_failed == len(per_item):
        raise ValueError(
            f"{len(per_item)} 条全部重排失败（首条：{per_item[0]['rerank_error']}）——"
            "本轮实际是「无重排」组，不能标成 +rerank 使用；修好 rerank 服务后重跑。"
        )
    # 改写失败会静默退化成「不改写」（对用户是对的：不该让一次分类拖死问答），
    # 但对评估是另一回事：一条自称 +rewrite 的臂里混进 N 条没改写的条目，
    # 就是在拿混合条件跟纯改写条件比。超时预算（rewrite.timeout_s）让这条路径
    # 在真实抖动下会被走到，所以必须计数而不是靠人回忆。
    rewrite_degraded = sum(1 for r in per_item if r.get("rewrite_degraded"))
    if use_rewrite and per_item and rewrite_degraded == len(per_item):
        raise ValueError(
            f"{len(per_item)} 条改写全部退化（未生效）——本轮实际是「无改写」组，"
            "不能标成 +rewrite 使用；先修改写（见 doc-rag check-rewrite）。"
        )

    # 过滤回退必须计数：字段稀疏时它会整条丢掉过滤（并多付一次检索延迟）。
    # 一条自称「带元数据过滤」的臂里混进 N 条没过滤的条目，覆盖率就不是那个机制的
    # 效果了——和 rerank_failed / rewrite_degraded 是同一类自证。
    filter_fallback_n = sum(1 for r in per_item if r.get("filter_fallback"))

    ragas_summary = None
    if with_ragas:
        ragas_summary = _run_ragas(
            _ragas_rows(items, per_item),
            cfg,
            sample_n=ragas_sample,
            use_cache=use_judge_cache,
            judge_over=judge_over,
        )

    acfg = agent_cfg(cfg)
    traces = [r["trace"] for r in per_item if r.get("trace")]
    agent_meta: dict | None = None
    if agent_mode or acfg["enabled"]:
        agent_meta = {
            "requested": agent_mode or ("config" if acfg["enabled"] else None),
            "enabled": acfg["enabled"],
            "types": list(acfg["types"]),
            "max_steps": acfg["max_steps"],
            "max_contexts": acfg["max_contexts"],
            "max_prompt_tokens": acfg["max_prompt_tokens"],
            "n_items_with_trace": len(traces),
            # 停机原因的分布就是 P2/P3 要报的「失败分类」：用完步数仍答不出 vs
            # 证据自判说够了但答案错，是两类完全不同的病。
            "stop_reasons": {
                reason: sum(1 for t in traces if t["stop_reason"] == reason)
                for reason in sorted({str(t["stop_reason"]) for t in traces})
            },
            # 非零就说明有些「停」其实是判定挂了，成本与质量结论都要打折看
            "judge_degraded": sum(
                1 for t in traces if any(s.get("degraded") for s in t["steps"])
            ),
            # 开关建在预测题型上（服务侧拿不到真题型），所以预测本身要能被度量
            "type_matches_gold": {
                "n": sum(1 for t in traces if t.get("type_matches_gold") is not None),
                "hits": sum(1 for t in traces if t.get("type_matches_gold")),
            },
        }

    results = {
        "meta": {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "top_n": top_n,
            # 预算口径必须自证：`fixed:8` 下所有题都只取 8 块（连聚合题也是），
            # 而生产路径听改写的建议（聚合题 25）。以前 eval 只能测前者，
            # 于是「聚合题放宽预算」这条生产行为从来没被任何数字量过。
            "budget": "rewrite" if honor_rewrite_budget else f"fixed:{top_n}",
            "context_budget": {
                "max_contexts": int(
                    (cfg.get("retrieval") or {}).get("max_contexts") or 0
                )
                or None,
                "rerank_top_n": int((cfg.get("rerank") or {}).get("top_n") or 0) or None
                if use_rerank
                else None,
            },
            "collection": retriever.collection,
            # agent 层的口径必须自证：并集进来的块比单发多，而 faithfulness 随
            # 上下文变长单调走高——不知道某条结果开没开 agent、开了几步，就不能拿它
            # 跟单发基线并排读数（PLAN §5.5 门槛 2 的同一条纪律）。
            "agent": agent_meta,
            "retrieval": f"dense+bm25+rrf[{retriever.cfg.get('mode', 'hybrid')}]"
            + ("+aggregate" if aggregate else "")
            # 与 +rerank 的规则故意不同：这条串还是**重放**的输入（`_retrieve_contexts`
            # 靠 `+rewrite` 决定「必须读记录的改写串」），部分退化时把它抹掉会让重放
            # 拿原始问题去配旧答案。退化条数靠下面的 rewrite_degraded 逐条追。
            + ("+rewrite" if use_rewrite else "")
            + ("+rerank" if use_rerank and not rerank_failed else ""),
            # 部分失败时逐条 rerank_error 可追；全失败直接中止（见下）
            "rerank_failed": rerank_failed if use_rerank else None,
            # 改写侧同款自证：退化条数（0 才是干净的 +rewrite 臂）+ 哪个模型做的改写
            # （改写可以与合成不同源，只记 llm_model 会把改写的归属记错）
            "rewrite_degraded": rewrite_degraded if use_rewrite else None,
            "rewrite_model": endpoint_model(cfg) if use_rewrite else None,
            # 有多少条其实没带着过滤跑完（0 才是干净的「过滤生效」臂）
            "filter_fallback_n": filter_fallback_n,
            "filters_applied_n": sum(1 for r in per_item if r.get("filter_applied")),
            "with_answers": with_answers,
            # 让结果文件自证身份：延迟数字曾因「不知道是哪个模型、缓存开没开」
            # 而无法归属（PLAN 里 1.3s 与 5.3~7.4s 的矛盾）。事后靠人回忆不可靠。
            "llm_model": llm_section.get("model"),
            "answer_cache": cache_enabled(llm_section) if with_answers else None,
            # prompt 指纹：三组对照的基线/收紧两组答案曾因 meta 不记 prompt 版本
            # 而无法归属（哪组用了哪个 prompt 靠猜），结论只能整体作废
            "prompt_version": prompt_version if with_answers else None,
            "prompt_fingerprint": (
                prompts.fingerprint(prompt_version) if with_answers else None
            ),
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
    out = {
        "n": len(xs),
        "min": round(xs[0], 1),
        "max": round(xs[-1], 1),
        "mean": round(sum(xs) / len(xs), 1),
    }
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
    uncached = [
        r
        for r in rows
        if r["latency"].get("synthesize") is not None
        and not r["latency"].get("synth_cached")
    ]
    stages = ("rewrite", "retrieve", "rerank", "retrieval_total")
    summary: dict = {
        "unit": "ms",
        # 检索侧与 LLM 无关（实测换模型完全一致），全部条目都算
        "by_stage": {
            s: _quantiles([r["latency"].get(s) for r in rows]) for s in stages
        },
        "synthesize": _quantiles([r["latency"].get("synthesize") for r in uncached]),
        "synthesize_n_uncached": len(uncached),
        "total": _quantiles([r["latency"].get("total") for r in rows]),
        "by_type": {},
        "n": len(rows),
        "cached_answers": sum(1 for r in rows if r["latency"].get("synth_cached")),
    }
    summary["cache_contaminated"] = summary["cached_answers"] > 0
    # 全量用量合计（T5）：把「耗时 → 用量」的归因做到汇总层。只累计真实记录的
    # 用量；缓存命中（usage=None）不计入也不冒充 0，分母口径见 usage_n
    usage_rows = [r["latency"].get("usage") for r in rows if r["latency"].get("usage")]
    if usage_rows:
        summary["usage_n"] = len(usage_rows)
        summary["usage"] = {
            k: sum(int(u.get(k) or 0) for u in usage_rows)
            for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")
        }
    # 端到端是否达标：PLAN 目标 P95 ≤ 8s（用全量 total，含缓存命中的快条目）
    e2e = summary["total"]
    summary["target_p95_ms"] = 8000
    summary["p95_meets_target"] = bool(e2e) and e2e.get("p95", 0) <= 8000
    for t in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == t]
        sub_uncached = [r for r in uncached if r["type"] == t]
        summary["by_type"][t] = {
            "n": len(sub),
            "synthesize": _quantiles(
                [r["latency"].get("synthesize") for r in sub_uncached]
            ),
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
                        meta = getattr(
                            getattr(gen, "message", None), "usage_metadata", None
                        )
                        if not meta:
                            continue
                        usage = meta  # 多代时取最后一份，但 reasoning 要累加
                        reasoning += int(
                            (meta.get("output_token_details") or {}).get("reasoning")
                            or 0
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


def _judge_chat_kwargs(cfg: dict, judge: dict | None = None) -> dict:
    """judge 的统一构造参数（`_run_ragas` 与 `probe-judge` 必须同源，否则探针看到的行为
    和正式判分不一致——这正是当初定位口径 bug 时踩过的坑）。

    endpoint 由 `eval/judge.py` 装配：默认继承生成侧 `llm`，`eval.judge.*` 或调用方传的
    `judge` 覆盖可以换到另一家供应商（跨供应商复判，PLAN「judge 自偏」）。
    """
    built = judge_cfg(cfg, **(judge or {}))
    kwargs: dict = {
        "model": built["model"],
        "base_url": built["base_url"],
        "api_key": built["api_key"],
        "temperature": built["temperature"],
    }
    # judge 的两个子任务（拆陈述 / 逐条判定）几乎不需要思考，但推理型模型会把
    # 输出预算的 97% 花在看不见的 reasoning token 上（实测单次 1554 → 79，全量 10.4×）。
    if built.get("reasoning_effort"):
        kwargs["reasoning_effort"] = built["reasoning_effort"]
    if built.get("timeout_s"):
        kwargs["timeout"] = built["timeout_s"]
    return kwargs


def _judge_identity(cfg: dict, judge: dict | None) -> dict:
    """判分产物必须自证是**谁**判的：换 judge 就是换度量身份，混用比没有更糟。"""
    built = judge_cfg(cfg, **(judge or {}))
    return {
        "model": built["model"],
        "base_url": (built.get("base_url") or "").split("//")[-1].split("/")[0],
        "cross_vendor": built.get("base_url") != (cfg.get("llm") or {}).get("base_url"),
        "reasoning_effort": built.get("reasoning_effort"),
    }


def _run_ragas(
    rows: list[dict],
    cfg: dict,
    sample_n: int | None = None,
    use_cache: bool = True,
    total: int | None = None,
    judge_over: dict | None = None,
) -> dict | None:
    """RAGAS 第二轨：rows=[{id,type,user_input,response,retrieved_contexts}] → judge 指标。

    可信度口径（PLAN §5.3）：judge 固定模型、temperature=0；只看与客观指标的相对一致性。
    `use_cache=False` 用于测 judge 自身的运行间随机性（temperature=0 也不保证跨请求逐字复现）。
    `total`：调用方已自行抽样时传入抽样前的总数，保证报告口径（n_answerable_total / sampled）准确。
    `judge_over`：`model` / `base_url` / `api_key` 任一非空即是一次跨供应商复判；**换 judge
    就是换度量身份**，所以是谁判的必须写进产物 meta，不能只留在命令行历史里。
    """
    try:
        from langchain.globals import set_llm_cache
        from langchain_community.cache import SQLiteCache
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset
        from ragas import evaluate as ragas_evaluate
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
        except Exception as exc:
            set_llm_cache(None)
            # 静默降级 = 无缓存跑完全量 judge（实付约 10 倍）。宁可失败也不白花。
            raise RuntimeError(
                f"judge 缓存初始化失败（{exc}）——已阻止无缓存的全量判分。"
                f"可用 --fresh-judge 显式跳过缓存，或修复 .cache 目录权限后重试。"
            ) from exc
    else:
        set_llm_cache(None)  # 显式关缓存：set_llm_cache 是进程级全局，必须清掉

    # 指标可选（PLAN §8 成本控制）：AnswerRelevancy 在中文场景噪声大且需嵌入调用
    wanted = [
        m.lower()
        for m in (cfg.get("eval", {}).get("ragas_metrics") or ["faithfulness"])
    ]
    metric_map = {"faithfulness": Faithfulness(), "answer_relevancy": AnswerRelevancy()}
    metrics = [metric_map[m] for m in wanted if m in metric_map]
    if not metrics:
        return {"skipped": f"未配置有效指标：{wanted}"}

    judge = LangchainLLMWrapper(
        ChatOpenAI(**_judge_chat_kwargs(cfg, judge_over), max_retries=0)
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
            "judge": _judge_identity(cfg, judge_over),
            "metrics": [m.name for m in metrics],
            "token_usage": counter.as_dict(),
        }
        per_item = []
        for idx, row in enumerate(rows):
            entry = {"id": row["id"], "type": row["type"]}
            # 每条判分带上的上下文块数：faithfulness 随上下文变长单调走高（可验证的
            # 陈述更多、每条更容易找到依据），所以两臂块数不等时这个差**偏向块数多的
            # 一臂**。不带这个数，compare 的等长护栏在答案轨上就是瞎的（PLAN §5.5 门槛 2）。
            contexts = row.get("retrieved_contexts") or []
            entry["n_contexts"] = len(contexts)
            for m in metrics:
                if m.name in df.columns and idx < len(df):
                    val = df[m.name].iloc[idx]
                    entry[m.name] = (
                        None
                        if val is None or math.isnan(float(val))
                        else round(float(val), 4)
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
                {
                    "statement": v.statement,
                    "verdict": bool(v.verdict),
                    "reason": v.reason,
                }
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
    judge_over: dict | None = None,
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
    candidates = [
        i for i in data["items"] if i["type"] != "no_answer" and i.get("answer")
    ]

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
    retriever = (
        _build_retriever(replay_cfg, meta.get("collection"))[0]
        if needs_rebuild
        else None
    )

    rows = []
    rebuilt = 0
    mismatched: list[str] = []
    for raw in picked:
        contexts = raw.get("contexts")
        if _legacy_contexts(contexts):
            ctx = _retrieve_contexts(
                raw["question"], meta, retriever, cfg, recorded=raw
            )
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
        rows,
        cfg,
        sample_n=resolved_n,
        use_cache=use_cache,
        total=len(candidates),
        judge_over=judge_over,
    )
    if summary is None:
        return None
    summary["contexts_rebuilt"] = rebuilt
    # judge 身份在这里独立算，不去读 `_run_ragas` 的返回：那条函数可能被调用方桩掉，
    # 而「谁判的」不该依赖判分是否成功才有值。
    ident = _judge_identity(cfg, judge_over)
    payload = {
        "meta": {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_results": results_file.name,
            "collection": meta.get("collection"),
            "retrieval": meta.get("retrieval"),
            "judge_model": ident["model"],
            "judge_base_url": ident["base_url"],
            "judge_cross_vendor": ident["cross_vendor"],
            "judge_reasoning_effort": ident["reasoning_effort"],
            "judge_temperature": 0,
            "judge_cache": use_cache,
        },
        "summary": summary,
    }
    dest = out_file or results_file.with_name(f"{results_file.stem}_ragas.json")
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

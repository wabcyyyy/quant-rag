"""在线管线单一入口：README 架构图里那个 Orchestrator。

rewrite → retrieve → rerank → max_contexts 截断 → synthesize 只在这里拼装一次。
此前这条链在 6 处内联重复（api/main.py 两处、api/demo.py、cli.py、eval/runner.py 两处），
并且已经漂移：FastAPI 两个端点漏掉重排、`/query` 还漏掉上下文截断——评估测的是 A 配置，
服务跑的是 B 配置，README 的头条数字因此描述的是一条没有交付入口在跑的管线。

三条口径必须记住：
1. `retrieved`（未截断）与 `contexts`（已截断）是两份不同清单：检索指标
   （Hit@k / MRR / nDCG / 文档覆盖率）在未截断那份上算，LLM 只看截断那份。
   合并它们会静默改变所有检索指标的分母。重排**也只重排序、不截断**——
   此前它把清单砍到 `rerank.top_n`(6) 又冒充「未截断」，于是「有重排」臂的
   nDCG@8 是在 6 条清单上算的、对照臂在 8 条上算，两臂不可比。现在上下文
   块数仍由 `min(max_contexts, rerank.top_n)` 决定，进 LLM 的那份逐字没变。
2. 合成器拿的是**原始问题**，不是改写后的——改写只服务于检索。
3. 重排失败退回融合顺序并把错误记在 `rerank_error` 上，不静默吞掉；过滤回退
   丢掉过滤同理，记在 `filter_fallback` 上。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient

from .generate import synthesizer as synthesizer_mod
from .ingest.embedder import Embedder
from .log import get_logger
from .retrieve.hybrid import HybridRetriever
from .retrieve.rewrite_llm import LLMQueryRewriter

log = get_logger("orchestrator")


@dataclass
class Result:
    plan: dict
    retrieved: list[dict]
    contexts: list[dict]
    citations: list[dict]
    answer: str = ""
    synth_meta: dict | None = None
    rerank_error: str | None = None
    latency_ms: dict[str, Any] = field(default_factory=dict)
    # 这次检索的实际预算与过滤轨迹：`top_n_used` 记录「显式预算」压掉了改写的
    # 建议预算（eval 固定 top_n 时必然发生），`filter_fallback` 记录过滤被放弃。
    # 没有这两个字段，结果文件就无法自证它测的是哪条配置。
    top_n_used: int | None = None
    rewrite_top_n: int | None = None
    filter_applied: bool = False
    filter_fallback: bool = False
    n_before_fallback: int = 0
    context_budget: int | None = None


def _ms(a: float, b: float) -> float:
    """b 到 a 的毫秒数（b 是更早的 time.perf_counter() 读数）。"""
    return round((a - b) * 1000, 1)


def _rerank_requested(cfg: dict, use_rerank: bool | None) -> bool:
    """`use_rerank=None` 跟随配置开关；显式传 bool 则覆盖它。

    eval 的消融臂必须能绕过 `rerank.enabled` 强制开/关（与改造前的 `_maybe_rerank`
    逐字一致），而服务和 CLI 只认配置。
    """
    if use_rerank is None:
        return bool((cfg.get("rerank") or {}).get("enabled"))
    return bool(use_rerank)


class Orchestrator:
    """一条在线问答管线。

    构造有两种：
    - `Orchestrator(cfg)`：生产路径。Qdrant 客户端与嵌入器**首次使用时**才建，
      collection 由每次调用的 `kb` 参数决定——不存在跨请求可变的共享状态。
    - `Orchestrator(cfg, retriever=..., synthesizer=...)`：注入现成部件。顺序执行的
      调用方（eval）和离线测试用这条，因此 `cfg` 可以不含 qdrant / embedding 段。
    """

    def __init__(
        self,
        cfg: dict,
        *,
        retriever: HybridRetriever | None = None,
        synthesizer: Any | None = None,
        collection: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.collection = (
            collection or (cfg.get("qdrant") or {}).get("collection") or ""
        )
        self._injected_retriever = retriever
        self._injected_synthesizer = synthesizer
        self._client: QdrantClient | None = None
        self._embedder: Embedder | None = None

    def _parts(self, kb: str | None) -> tuple[Any, Any]:
        if self._injected_retriever is not None:
            return self._injected_retriever, self._injected_synthesizer
        if self._client is None:
            self._client = QdrantClient(url=self.cfg["qdrant"]["url"], timeout=60)
            self._embedder = Embedder(self.cfg["embedding"])
        client, embedder = self._client, self._embedder
        assert client is not None and embedder is not None  # 同一分支内成对赋值
        retriever = HybridRetriever(
            client=client,
            embedder=embedder,
            collection=kb or self.collection,
            retrieval_cfg=self.cfg["retrieval"],
        )
        # 合成器逐调用新建：它的 last_meta 是可变属性，复用会让并发请求互相读到
        # 对方的计时（假延迟）。eval 走注入路径，顺序执行下复用同一个实例。
        return retriever, synthesizer_mod.Synthesizer(self.cfg["llm"])

    @property
    def retriever(self) -> Any:
        """当前绑定的检索器（默认 collection）。

        eval 的结果文件 meta 要用它自证 `collection` / 检索模式；注入路径下它返回
        的就是注入的那个对象，不会因此建立 Qdrant 连接。
        """
        return self._parts(None)[0]

    def _prepare(
        self,
        question: str,
        *,
        kb: str | None,
        top_n: int | None,
        use_rewrite: bool,
        use_rerank: bool | None,
        force_aggregate: bool,
        plan_override: dict | None = None,
        honor_rewrite_budget: bool = False,
    ) -> tuple[Result, Any, bool, dict[str, float]]:
        retriever, synthesizer = self._parts(kb)
        t0 = time.perf_counter()

        if plan_override is not None:
            # 重放已记录的改写结果。实时改写从 W3 起由模型产出、不可复现，
            # 用它重放等于拿另一批上下文去判旧答案——judge 的忠实度就失去意义。
            plan = plan_override
        elif use_rewrite:
            plan = LLMQueryRewriter(self.cfg).rewrite(question)
        else:
            plan = {
                "rewritten": question,
                "filters": None,
                "aggregate": force_aggregate,
                "top_n": None,
                "reason": "已关闭改写",
                "degraded": False,  # 主动关的，不是模型失败的
            }
        t_rewrite = time.perf_counter()

        aggregate = bool(plan["aggregate"] or force_aggregate)
        # 预算优先级：
        # - `honor_rewrite_budget=True` → 一律听改写的建议（生产口径的评估臂）
        # - 否则：调用方显式给的 top_n 优先（消融要在同一预算下比），
        #   没给（CLI / API 的默认路径）时仍然听改写的建议——生产路径一直是这样。
        if honor_rewrite_budget:
            effective_top_n: int | None = plan.get("top_n") or top_n
        else:
            effective_top_n = top_n or plan.get("top_n")
        outcome = retriever.retrieve(
            plan["rewritten"],
            top_n=effective_top_n,
            filters=plan["filters"],
            aggregate=aggregate,
        )
        results = outcome.chunks
        t_retrieve = time.perf_counter()

        rerank_error: str | None = None
        context_budget = 0
        if _rerank_requested(self.cfg, use_rerank) and results:
            from .retrieve.rerank import Reranker

            try:
                reranker = Reranker(self.cfg["rerank"])
                # 全量重排、不截断：清单长度与「无重排」臂相同，两臂的 nDCG@8 /
                # 覆盖率才有可比性。截断在下面按上下文预算做。
                results = reranker.rerank(plan["rewritten"], results)
                context_budget = reranker.context_budget
            except Exception as exc:  # noqa: BLE001 退回融合顺序，但失败必须可见
                rerank_error = str(exc)
                log.warning(
                    "重排失败，退回融合顺序",
                    extra={"stage": "rerank", "rerank_error": rerank_error, "kb": kb},
                )
        t_rerank = time.perf_counter()

        # 进 LLM 的块数 = min(max_contexts, rerank.top_n)。重排开启时生效值仍是
        # rerank.top_n（改造前是重排把清单砍到 6，然后 [:10] 不再动它）——同一批
        # 块、同一个顺序，所以这份上下文逐字没变，只有指标的分母被修正了。
        cap = int(self.cfg["retrieval"].get("max_contexts") or 0)
        budgets = [b for b in (cap, context_budget) if b > 0]
        capped = results[: min(budgets)] if budgets else list(results)
        contexts = [
            {
                "no": i + 1,
                "text": r["text"],
                "doc": r["title"] or r["doc_id"],
                "page": r["page"],
            }
            for i, r in enumerate(capped)
        ]
        citations = [
            {
                "no": c["no"],
                "doc": c["doc"],
                "page": c["page"],
                "doc_id": r["doc_id"],
                "block_type": r.get("block_type"),
            }
            for c, r in zip(contexts, capped)
        ]
        result = Result(
            plan=plan,
            retrieved=results,
            contexts=contexts,
            citations=citations,
            rerank_error=rerank_error,
            top_n_used=effective_top_n,
            rewrite_top_n=plan.get("top_n"),
            filter_applied=outcome.filter_applied,
            filter_fallback=outcome.filter_fallback,
            n_before_fallback=outcome.n_before_fallback,
            context_budget=len(capped),
        )
        marks = {
            "t0": t0,
            "t_rewrite": t_rewrite,
            "t_retrieve": t_retrieve,
            "t_rerank": t_rerank,
        }
        return result, synthesizer, aggregate, marks

    @staticmethod
    def _latency(
        marks: dict[str, float], synth_meta: dict | None, answered: bool
    ) -> dict:
        return {
            "rewrite": _ms(marks["t_rewrite"], marks["t0"]),
            "retrieve": _ms(marks["t_retrieve"], marks["t_rewrite"]),
            "rerank": _ms(marks["t_rerank"], marks["t_retrieve"]),
            # 检索侧不含 LLM：这部分与模型无关，换模型不必重测
            "retrieval_total": _ms(marks["t_rerank"], marks["t0"]),
            "synthesize": (synth_meta or {}).get("ms") if answered else None,
            "synth_cached": bool((synth_meta or {}).get("cached"))
            if answered
            else None,
            "total": _ms(time.perf_counter(), marks["t0"]),
        }

    def answer(
        self,
        question: str,
        *,
        kb: str | None = None,
        top_n: int | None = None,
        use_rewrite: bool = True,
        use_rerank: bool | None = None,
        force_aggregate: bool = False,
        require_citation: bool = True,
        with_answer: bool = True,
        stop_on_empty: bool = False,
        plan_override: dict | None = None,
        honor_rewrite_budget: bool = False,
    ) -> Result:
        """跑完整管线，返回 `Result`。

        `with_answer=False`：只检索（供 judge 重放上下文）。
        `stop_on_empty=True`：检索为空时**不发合成**——CLI 用它兜住"忘了 ingest"，
        省掉一次必然无据的调用。eval 不能用它：空上下文下的拒答行为本身就是被测对象。
        `plan_override`：重放已记录的改写计划；与 `use_rewrite` 同时给时以它为准。
        `honor_rewrite_budget=True`：检索预算听改写的建议（生产口径），此时 `top_n`
        被忽略。默认 False = 用调用方给的固定预算（消融口径）。
        """
        result, synthesizer, aggregate, marks = self._prepare(
            question,
            kb=kb,
            top_n=top_n,
            use_rewrite=use_rewrite,
            use_rerank=use_rerank,
            force_aggregate=force_aggregate,
            plan_override=plan_override,
            honor_rewrite_budget=honor_rewrite_budget,
        )
        if not with_answer:
            result.latency_ms = self._latency(marks, None, answered=False)
            return result
        if stop_on_empty and not result.contexts:
            result.latency_ms = self._latency(marks, None, answered=False)
            return result
        result.answer = synthesizer.answer(
            question,
            result.contexts,
            require_citation=require_citation,
            aggregate=aggregate,
        )
        # 必须是 dict——Mock 的自动属性会造出一个不可序列化的假 meta
        candidate = getattr(synthesizer, "last_meta", None)
        result.synth_meta = candidate if isinstance(candidate, dict) else None
        result.latency_ms = self._latency(marks, result.synth_meta, answered=True)
        return result

    def answer_stream(
        self,
        question: str,
        *,
        kb: str | None = None,
        top_n: int | None = None,
        use_rewrite: bool = True,
        use_rerank: bool | None = None,
        force_aggregate: bool = False,
        require_citation: bool = True,
        stop_on_empty: bool = False,
    ) -> Iterator[dict]:
        """流式问答，事件序列 = rewrite / delta* / citations / done。

        与 `answer` 共用 `_prepare` 和同一个 prompt 构造逻辑，区别只在答案以增量
        事件推送。请求参数与缓存键在 `llm._build_request` 处同源，所以流式与非流式
        互相命中缓存。`stop_on_empty` 时不产出 delta，直接走到 citations/done。
        """
        result, synthesizer, aggregate, marks = self._prepare(
            question,
            kb=kb,
            top_n=top_n,
            use_rewrite=use_rewrite,
            use_rerank=use_rerank,
            force_aggregate=force_aggregate,
        )
        yield {"type": "rewrite", "plan": result.plan}
        if stop_on_empty and not result.contexts:
            result.latency_ms = self._latency(marks, None, answered=False)
            yield {"type": "citations", "citations": result.citations}
            yield {"type": "done", "result": result}
            return
        pieces: list[str] = []
        for piece in synthesizer.answer_stream(
            question,
            result.contexts,
            require_citation=require_citation,
            aggregate=aggregate,
        ):
            pieces.append(piece)
            yield {"type": "delta", "text": piece}
        result.answer = "".join(pieces)
        candidate = getattr(synthesizer, "last_meta", None)
        result.synth_meta = candidate if isinstance(candidate, dict) else None
        result.latency_ms = self._latency(marks, result.synth_meta, answered=True)
        yield {"type": "citations", "citations": result.citations}
        yield {"type": "done", "result": result}

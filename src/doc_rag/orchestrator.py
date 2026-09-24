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

from . import agent as agent_mod
from .generate import synthesizer as synthesizer_mod
from .generate import two_stage as two_stage_mod
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
    # agent 臂的每一步（含停机原因与预算用量）。单发路径恒为 None——
    # 「只扩展不替换」：现有结果文件的条目形状、检索指标分母都不该因为
    # 加了一层 policy 就悄悄变样。落盘与重放约束见 agent.py 的模块 docstring。
    trace: dict | None = None
    # 两段式合成（ADR-0002）的路由结果与逐篇微摘要。`two_stage` / `two_stage_fallback`
    # 时 summaries 非 None（重放纪律：reduce 输入可从落盘摘要逐字重建，缺记录值拒绝重放）；
    # `single` 恒为 None——上下文仍存原始块，引用 [n] 指原始块序号，判分口径零改动。
    synthesis_route: str = "single"
    summaries: list[dict] | None = None
    # map 段自己的开销（并行微摘要的墙钟与篇数/退化数）。单发路径恒为 None。
    map_meta: dict | None = None
    # 检索是否为空（B3 拒答归因）：True = 「索引挂了/语料没进来」，False = 检索有产出。
    # 没有这个字段，「检索为空导致的拒答」与「文档真没记载的正确拒答」在结果里
    # 长得一模一样——前者是事故，后者是被测行为，必须可区分。
    retrieval_empty: bool = False


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


def _entries(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """检索结果行 → (contexts, citations)。

    为什么是模块级而不是 `_prepare` 里的闭包：`answer_given_contexts`（外部基准
    那种「文档由数据集提供」的协议）必须与检索路径产出**形状逐字相同**的
    contexts/citations——否则引用编号 `[n]` 的含义会在两条路径上分叉，而被引用的
    是同一份 LLM 输出。
    """
    ctxs = [
        {
            "no": i + 1,
            "text": r["text"],
            "doc": r["title"] or r["doc_id"],
            "page": r["page"],
            "doc_id": r["doc_id"],
        }
        for i, r in enumerate(rows)
    ]
    cites = [
        {
            "no": c["no"],
            "doc": c["doc"],
            "page": c["page"],
            "doc_id": r["doc_id"],
            "block_type": r.get("block_type"),
        }
        for c, r in zip(ctxs, rows)
    ]
    return ctxs, cites


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

    def _synthesizer(self) -> Any:
        """只取合成器，不建 Qdrant 客户端与嵌入器。

        单独成一个方法而不是走 `_parts`：后者在生产路径下会顺手建连接与嵌入器，
        而给定上下文的入口（`answer_given_contexts`）根本不检索，不该为它付一次
        连接与一次模型装配。注入路径仍返回注入的那个实例，语义与 `_parts` 一致。
        """
        if self._injected_synthesizer is not None:
            return self._injected_synthesizer
        return synthesizer_mod.Synthesizer(self.cfg["llm"])

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
        mode: str | None = None,
        question_type: str | None = None,
        require_citation: bool = True,
        with_answer: bool = True,
    ) -> tuple[Result, Any, str, dict[str, float]]:
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

        # ── agent 层。它只在「第一步已经跑完」之后扩展，所以单发臂与 agent 臂的第 1 步
        # 逐字同源，两臂的差异全部落在扩展步上——这是可比性的要求，不是风格。
        ac = agent_mod.agent_cfg(self.cfg)
        predicted = agent_mod.predict_type(plan)
        wants_agent = mode == "agent" or (
            mode is None and ac["enabled"] and predicted in ac["types"]
        )
        trace: dict | None = None
        if wants_agent and results:
            results, trace = agent_mod.run_agent(
                self.cfg,
                question=question,
                plan=plan,
                retriever=retriever,
                pool=results,
            )
            # 把「这条臂为什么开了 agent」记进 trace：显式 mode（评估臂）与配置开关
            # （部署）是两条不同来源，混起来就没人能从结果反推它跑的是什么。
            trace.update(
                {
                    "type_predicted": predicted,
                    "type_requested": question_type,
                    "type_matches_gold": agent_mod.type_matches_gold(
                        predicted, question_type
                    ),
                    "mode_explicit": mode == "agent",
                }
            )
        t_agent = time.perf_counter()

        # ── 两段式合成路由（ADR-0002）。与 agent 互斥：agent 已经替换了上下文预算
        # 口径，再叠 map/reduce 等于一条臂混两个实验变量。require_citation=False 是
        # 消融 #4 的无引用对照组，reduce prompt 没有无引用变体，同样不路由；
        # 只检索（with_answer=False）与空清单更不该花 map 的钱。
        route = "single"
        if (
            with_answer
            and trace is None
            and require_citation
            and results
            and two_stage_mod.is_routed(self.cfg, predicted)
        ):
            route = "two_stage"

        # 进 LLM 的块数 = min(max_contexts, rerank.top_n)。重排开启时生效值仍是
        # rerank.top_n（改造前是重排把清单砍到 6，然后 [:10] 不再动它）——同一批
        # 块、同一个顺序，所以这份上下文逐字没变，只有指标的分母被修正了。
        cap = int(self.cfg["retrieval"].get("max_contexts") or 0)
        normal_budgets = [b for b in (cap, context_budget) if b > 0]
        if trace is not None:
            # agent 臂的上下文预算换成 `agent.max_contexts`，**不再**受 rerank.top_n 约束：
            # 沿用它就把多步并集又砍回 6 块，被砍掉的正是这层存在的理由。
            # 代价写在 PLAN §5.5 门槛 2——faithfulness 随上下文变长单调走高，所以
            # agent 臂必须配一条同块数的对照臂，才准它进答案轨结论。
            budgets = [int(trace["max_contexts"])]
        elif route == "two_stage":
            # 两段式臂的 map 段吃 retrieved 去重后的**全部**文档清单——照 agent 先例
            # 开自己的预算分支：被 min(max_contexts, rerank.top_n) 截在 6 块的话，
            # 逐篇微摘要就只剩 6 篇可摘，检索放宽的收益全被截掉（UPGRADE §3.1）。
            budgets = [len(results)]
        else:
            budgets = normal_budgets
        capped = results[: min(budgets)] if budgets else list(results)

        contexts, citations = _entries(capped)
        summaries: list[dict] | None = None
        map_meta: dict | None = None
        map_marks: dict[str, float] = {}
        if route == "two_stage":
            map_marks["t_map_start"] = time.perf_counter()
            summaries, map_meta = two_stage_mod.map_summaries(
                self.cfg, question, contexts
            )
            map_marks["t_map"] = time.perf_counter()
            if summaries is None:
                # 失败率超阈值 → 整题退回单发口径并计数（`synthesis_route` 落到
                # two_stage_fallback，eval 层对「全部退回」中止）。上下文换回单发
                # 预算那份——退回的是**口径**，不是同一批块换个 prompt。
                route = "two_stage_fallback"
                summaries = None
                capped = (
                    results[: min(normal_budgets)] if normal_budgets else list(results)
                )
                contexts, citations = _entries(capped)
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
            trace=trace,
            synthesis_route=route,
            summaries=summaries,
            map_meta=map_meta,
            retrieval_empty=not results,
        )
        marks = {
            "t0": t0,
            "t_rewrite": t_rewrite,
            "t_retrieve": t_retrieve,
            "t_rerank": t_rerank,
        }
        if trace is not None:
            # 只有真跑了扩展步才产出 `agent` 这一档延迟：单发路径的 latency_ms
            # 键集合是所有延迟分位数统计的既有口径，不能因为加了层就悄悄多一键。
            marks["t_agent"] = t_agent
        if "t_map" in map_marks:
            marks.update(map_marks)
        # 往外传的是**预测题型**而不是 aggregate 布尔：思考档那张表的键就是它
        # （`synthesizer.PREDICTED_TYPES`），传布尔等于让合成层再造一次同样的判断。
        return result, synthesizer, predicted, marks

    @staticmethod
    def _latency(
        marks: dict[str, float], synth_meta: dict | None, answered: bool
    ) -> dict:
        out = {
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
        if "t_agent" in marks:
            # 单独一档，不并进 `retrieval_total`：扩展步里含判定调用（那是 LLM），
            # 而 `retrieval_total` 的口径是「不含 LLM、换模型不必重测」。端到端的
            # `total` 自然包含它；SLO 判读要看 total 与这一档的分布。
            out["agent"] = _ms(marks["t_agent"], marks["t_rerank"])
        if "t_map" in marks:
            # map 段（并行微摘要）单独一档，同理：它是 LLM 时间，不该混进
            # retrieval_total；ADR-0002 的延迟门槛判读看 total 与这一档的和。
            out["map_ms"] = _ms(marks["t_map"], marks["t_map_start"])
        return out

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
        mode: str | None = None,
        question_type: str | None = None,
    ) -> Result:
        """跑完整管线，返回 `Result`。

        `with_answer=False`：只检索（供 judge 重放上下文）。
        `stop_on_empty=True`：检索为空时**不发合成**——CLI 用它兜住"忘了 ingest"，
        省掉一次必然无据的调用。eval 不能用它：空上下文下的拒答行为本身就是被测对象。
        `plan_override`：重放已记录的改写计划；与 `use_rewrite` 同时给时以它为准。
        `honor_rewrite_budget=True`：检索预算听改写的建议（生产口径），此时 `top_n`
        被忽略。默认 False = 用调用方给的固定预算（消融口径）。
        `mode`：`"agent"` 强制走扩展步、`"single"` 强制不走、`None`（默认）跟随配置
        `agent.enabled` + 分题型开关。评估臂用显式值，部署用配置——来源不同，
        所以要落进 trace 的 `mode_explicit`。
        `question_type`：黄金集的真题型，**只用于事后核对预测**（`type_matches_gold`），
        不参与开关决策。理由见 `agent.predict_type`：真题型在服务侧不存在，
        拿它当开关会让五条入口的 trace 不可比。
        """
        result, synthesizer, predicted, marks = self._prepare(
            question,
            kb=kb,
            top_n=top_n,
            use_rewrite=use_rewrite,
            use_rerank=use_rerank,
            force_aggregate=force_aggregate,
            plan_override=plan_override,
            honor_rewrite_budget=honor_rewrite_budget,
            mode=mode,
            question_type=question_type,
            require_citation=require_citation,
            with_answer=with_answer,
        )
        if not with_answer:
            result.latency_ms = self._latency(marks, None, answered=False)
            return result
        if stop_on_empty and not result.contexts:
            result.latency_ms = self._latency(marks, None, answered=False)
            return result
        if result.synthesis_route == "two_stage" and result.summaries is not None:
            # ADR-0002 的 reduce 段：只吃逐篇微摘要，引用编号仍指原始块
            # （`result.contexts` 存的就是原始块，citation 语义零改动）。
            result.answer = two_stage_mod.answer(
                synthesizer,
                question,
                result.contexts,
                result.summaries,
                question_type=predicted,
            )
        else:
            result.answer = synthesizer.answer(
                question,
                result.contexts,
                require_citation=require_citation,
                question_type=predicted,
            )
        # 必须是 dict——Mock 的自动属性会造出一个不可序列化的假 meta
        candidate = getattr(synthesizer, "last_meta", None)
        result.synth_meta = candidate if isinstance(candidate, dict) else None
        result.latency_ms = self._latency(marks, result.synth_meta, answered=True)
        return result

    def answer_given_contexts(
        self,
        question: str,
        docs: list[str],
        *,
        require_citation: bool = True,
        question_type: str | None = None,
        doc_names: list[str] | None = None,
    ) -> Result:
        """给定上下文的问答：跳过改写/检索/重排，直接合成。

        存在理由：外部基准（RGB）是**给文档**的协议——文档由数据集提供，检索层不
        参与。此前没有这条路径，评测脚本只能自己在 Orchestrator 之外拼 contexts 再
        调 Synthesizer，那正是「同一条链在多处内联重复并漂移」的复发条件。这里与
        检索路径共用同一份 `_entries` 与同一个 Synthesizer，所以引用编号 `[n]`、
        上下文渲染格式、思考档查表在两条路径上逐字同源——`tests/test_orchestrator_parity.py`
        用同 contexts 对照把这件事钉住。

        `question_type` 是**预测题型**（生产里来自 `agent.predict_type(plan)`），只用来
        查思考档。本入口不经过改写、拿不到计划，所以由调用方给出；基准按数据集自带的
        任务标签填（聚合类的题因此走生产的「聚合关思考」档）——这是一处**声明过的
        偏差**，不是静默改动，见 docs/guides/benchmark-rgb.md。

        `doc_names` 缺省按「文档1..n」编号：基准语料没有标题，而生产路径的
        `format_context` 一定给每块带一个 `（doc 第p页）` 前缀，不能省。
        """
        names = (
            list(doc_names)
            if doc_names is not None
            else [f"文档{i + 1}" for i in range(len(docs))]
        )
        if len(names) != len(docs):
            raise ValueError(
                f"doc_names 与 docs 数量不一致：{len(names)} vs {len(docs)}"
            )
        rows = [
            {
                "text": text,
                "title": name,
                # doc_id 参与 citations 与落盘、不参与 prompt：固定前缀是为了让
                # 「这条引用来自给定上下文而不是检索」在结果文件里可自证。
                "doc_id": f"given-{i + 1:04d}",
                "page": None,
                "block_type": "paragraph",
            }
            for i, (name, text) in enumerate(zip(names, docs, strict=True))
        ]
        contexts, citations = _entries(rows)
        synthesizer = self._synthesizer()
        t0 = time.perf_counter()
        result = Result(
            plan={
                "rewritten": question,
                "filters": None,
                "aggregate": False,
                "top_n": None,
                "reason": "给定上下文，未改写",
                "degraded": False,
            },
            # `retrieved` 是**检索**的结果；本入口没有检索，留空而不是填 contexts。
            # 填了会让「检索指标算在未截断清单上」这条口径在这条路径上变成假的。
            retrieved=[],
            contexts=contexts,
            citations=citations,
            context_budget=len(contexts),
        )
        result.answer = synthesizer.answer(
            question,
            contexts,
            require_citation=require_citation,
            question_type=question_type,
        )
        # 必须是 dict——Mock 的自动属性会造出一个不可序列化的假 meta（同 answer）
        candidate = getattr(synthesizer, "last_meta", None)
        result.synth_meta = candidate if isinstance(candidate, dict) else None
        marks = {"t0": t0, "t_rewrite": t0, "t_retrieve": t0, "t_rerank": t0}
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
        mode: str | None = None,
        question_type: str | None = None,
    ) -> Iterator[dict]:
        """流式问答，事件序列 = rewrite / delta* / citations / done。

        与 `answer` 共用 `_prepare` 和同一个 prompt 构造逻辑，区别只在答案以增量
        事件推送。请求参数与缓存键在 `llm._build_request` 处同源，所以流式与非流式
        互相命中缓存。`stop_on_empty` 时不产出 delta，直接走到 citations/done。
        """
        result, synthesizer, predicted, marks = self._prepare(
            question,
            kb=kb,
            top_n=top_n,
            use_rewrite=use_rewrite,
            use_rerank=use_rerank,
            force_aggregate=force_aggregate,
            mode=mode,
            question_type=question_type,
            require_citation=require_citation,
            with_answer=True,
        )
        yield {"type": "rewrite", "plan": result.plan}
        if stop_on_empty and not result.contexts:
            result.latency_ms = self._latency(marks, None, answered=False)
            yield {"type": "citations", "citations": result.citations}
            yield {"type": "done", "result": result}
            return
        pieces: list[str] = []
        if result.synthesis_route == "two_stage" and result.summaries is not None:
            gen = two_stage_mod.answer_stream(
                synthesizer,
                question,
                result.contexts,
                result.summaries,
                question_type=predicted,
            )
        else:
            gen = synthesizer.answer_stream(
                question,
                result.contexts,
                require_citation=require_citation,
                question_type=predicted,
            )
        for piece in gen:
            pieces.append(piece)
            yield {"type": "delta", "text": piece}
        result.answer = "".join(pieces)
        candidate = getattr(synthesizer, "last_meta", None)
        result.synth_meta = candidate if isinstance(candidate, dict) else None
        result.latency_ms = self._latency(marks, result.synth_meta, answered=True)
        yield {"type": "citations", "citations": result.citations}
        yield {"type": "done", "result": result}

"""Gradio 演示页（T8，PLAN §2 交付清单最后一项）。

启动：`uv run --extra demo doc-rag demo`（gradio 在 demo extra，不进核心依赖）。
组件：问题输入 + 示例下拉（**示意问题，不含公司内容**）、答案 Markdown（引用编号）、
引用列表（文档名/页码/块类型）、分阶段延迟面板。检索与合成复用 API 层的 `_pipeline`，
T7 完成后接入流式（逐段 yield 渐进更新）。
"""

from __future__ import annotations

# 示例问题用标注过的示意措辞（合规：不出现真实文档标题/人名/公司议题）
EXAMPLE_QUESTIONS = [
    "关于供应商预付款，我们做过哪些决定？",
    "上季度的管理层例会定了哪些事情？",
    "公司关于员工持股计划的制度是什么？",
]

# 复用 API 层的单例 pipeline（cfg/retriever/rewriter）；放全局段末避免 import 排序告警
from doc_rag.api.main import _pipeline  # 复用 API 层单例 pipeline


def _citations_markdown(contexts: list[dict], results: list[dict]) -> str:
    lines = []
    for c, r in zip(contexts, results):
        page = f" 第{c['page']}页" if c["page"] else ""
        lines.append(f"[{c['no']}] {c['doc']}{page}（{r['block_type']}）")
    return "\n".join(lines) if lines else "（无检索结果）"


def build_ui(gr):
    """构建 gr.Blocks 演示页。gradio 由调用方传入（核心依赖不含它）。"""
    import time

    from doc_rag.generate.synthesizer import Synthesizer
    from doc_rag.retrieve.rerank import Reranker

    cfg, retriever, rewriter = _pipeline()

    def run(question: str):
        """流式回答：yield (部分答案, 引用面板, 延迟面板)。"""
        if not question or not question.strip():
            yield "请输入问题。", "", ""
            return
        synthesizer = Synthesizer(cfg["llm"])
        t0 = time.perf_counter()
        plan = rewriter.rewrite(question)
        results = retriever.retrieve(
            plan["rewritten"],
            top_n=plan["top_n"],
            filters=plan["filters"],
            aggregate=plan["aggregate"],
        )
        t_retrieve = time.perf_counter()
        rerank_ms = None
        if results and cfg["rerank"].get("enabled"):
            t_r0 = time.perf_counter()
            try:
                results = Reranker(cfg["rerank"]).rerank(plan["rewritten"], results)
            except Exception:  # noqa: BLE001 重排失败退回融合顺序（与 CLI 同口径）
                rerank_ms = None
            rerank_ms = (time.perf_counter() - t_r0) * 1000
        cap = int(cfg["retrieval"].get("max_contexts") or 0)
        if cap:
            results = results[:cap]
        contexts = [
            {"no": i + 1, "text": r["text"], "doc": r["title"] or r["doc_id"], "page": r["page"]}
            for i, r in enumerate(results)
        ]
        cites_md = "（生成中…）"
        lat_md = (
            f"改写 {(t_retrieve - t0) * 1000:.0f}ms · 检索+重排 "
            f"{(time.perf_counter() - t_retrieve) * 1000:.0f}ms · 合成中…"
        )
        answer = ""
        for piece in synthesizer.answer_stream(
            question, contexts, aggregate=plan["aggregate"]
        ):
            answer += piece
            yield answer, cites_md, lat_md
        meta = synthesizer.last_meta or {}
        synth_ms = meta.get("ms")
        cached_note = "（缓存命中）" if meta.get("cached") else ""
        lat_md = (
            f"改写+检索 {(t_retrieve - t0) * 1000:.0f}ms · "
            + (f"重排 {rerank_ms:.0f}ms · " if rerank_ms is not None else "")
            + f"合成 {synth_ms:.0f}ms{cached_note} · 端到端 {(time.perf_counter() - t0) * 1000:.0f}ms"
        )
        yield answer, _citations_markdown(contexts, results), lat_md

    with gr.Blocks(title="doc-rag 演示") as demo:
        gr.Markdown("## 企业文档 RAG 演示\n检索问答：改写 → Dense+BM25+RRF → 重排 → 强制引用合成")
        with gr.Row():
            with gr.Column(scale=2):
                q_in = gr.Textbox(label="问题", placeholder="输入关于文档库的问题…")
                example = gr.Dropdown(
                    choices=EXAMPLE_QUESTIONS, label="示例问题（示意数据）", interactive=True
                )
                example.input(lambda q: q, inputs=example, outputs=q_in)
                btn = gr.Button("提交", variant="primary")
            with gr.Column(scale=3):
                answer_md = gr.Markdown(label="答案")
                with gr.Accordion("引用", open=True):
                    cites_md = gr.Markdown()
                with gr.Accordion("延迟", open=True):
                    lat_md = gr.Markdown()
        btn.click(run, inputs=q_in, outputs=[answer_md, cites_md, lat_md])
        q_in.submit(run, inputs=q_in, outputs=[answer_md, cites_md, lat_md])
    return demo

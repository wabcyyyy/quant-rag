"""Gradio 演示页（T8，PLAN §2 交付清单最后一项）。

启动：`uv run --extra demo doc-rag demo`（gradio 在 demo extra，不进核心依赖）。
组件：问题输入 + 示例下拉（**示意问题，不含公司内容**）、答案 Markdown（引用编号）、
引用列表（文档名/页码/块类型）、分阶段延迟面板。管线复用 API 层的 `_orchestrator`，
逐段 yield 渐进更新（T7 流式）。
"""

from __future__ import annotations

# 示例问题用标注过的示意措辞（合规：不出现真实文档标题/人名/公司议题）
EXAMPLE_QUESTIONS = [
    "关于供应商预付款，我们做过哪些决定？",
    "上季度的管理层例会定了哪些事情？",
    "公司关于员工持股计划的制度是什么？",
]

# 复用 API 层的单例 Orchestrator；放全局段末避免 import 排序告警
from doc_rag.api.main import _orchestrator


def _citations_markdown(citations: list[dict]) -> str:
    lines = []
    for c in citations:
        page = f" 第{c['page']}页" if c.get("page") else ""
        block = f"（{c['block_type']}）" if c.get("block_type") else ""
        lines.append(f"[{c['no']}] {c['doc']}{page}{block}")
    return "\n".join(lines) if lines else "（无检索结果）"


def _latency_markdown(result) -> str:
    lat = result.latency_ms
    if result.rerank_error:
        head = f"重排失败，退回融合顺序（{result.rerank_error}）\n\n"
    else:
        head = ""
    cached = "（缓存命中，非模型延迟）" if lat.get("synth_cached") else ""
    return (
        f"{head}"
        f"改写 {lat['rewrite']:.0f}ms · 检索 {lat['retrieve']:.0f}ms · "
        f"重排 {lat['rerank']:.0f}ms · 合成 {lat['synthesize'] or 0:.0f}ms{cached} · "
        f"端到端 {lat['total']:.0f}ms"
    )


def render_answer(orch, question: str):
    """一次问答的三面板流式产出（模块级：parity 测试可直接驱动，不必起 gradio）。"""
    answer = ""
    cites_md = "（生成中…）"
    for ev in orch.answer_stream(question):
        kind = ev["type"]
        if kind == "delta":
            answer += ev["text"]
            yield answer, cites_md, "生成中…"
        elif kind == "citations":
            cites_md = _citations_markdown(ev["citations"])
        elif kind == "done":
            yield answer, cites_md, _latency_markdown(ev["result"])


def build_ui(gr, orchestrator=None):
    """构建 gr.Blocks 演示页。gradio 由调用方传入（核心依赖不含它）。

    `orchestrator` 显式传入时用它，否则复用 API 层单例——`doc-rag demo --kb X` 靠
    这条换库，不需要改共享状态。
    """
    orch = orchestrator if orchestrator is not None else _orchestrator()

    def run(question: str):
        """gradio 绑定的是这个函数，它必须是生成器函数才能逐段刷新。"""
        if not question or not question.strip():
            yield "请输入问题。", "", ""
            return
        yield from render_answer(orch, question)

    with gr.Blocks(title="doc-rag 演示") as demo:
        gr.Markdown(
            "## 企业文档 RAG 演示\n检索问答：改写 → Dense+BM25+RRF → 重排 → 强制引用合成"
        )
        with gr.Row():
            with gr.Column(scale=2):
                q_in = gr.Textbox(label="问题", placeholder="输入关于文档库的问题…")
                example = gr.Dropdown(
                    choices=EXAMPLE_QUESTIONS,
                    label="示例问题（示意数据）",
                    interactive=True,
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

"""合成层（PLAN §5.2）：编号引用 + 无据拒答。"""

from __future__ import annotations

from . import llm, prompts


class Synthesizer:
    def __init__(self, llm_cfg: dict) -> None:
        self.llm_cfg = llm_cfg
        # 最近一次 answer 的计时元数据（llm.chat_timed 的返回）。
        # 用实例属性而不是改 answer 的返回类型：调用点（CLI/API/eval）都不用变，
        # 需要延迟的一方读它即可（PLAN「延迟口径」）。
        self.last_meta: dict | None = None

    def answer(
        self,
        question: str,
        chunks: list[dict],
        require_citation: bool = True,
        aggregate: bool | None = None,
    ) -> str:
        """chunks: [{"no", "text", "doc", "page"}] → 带编号引用的回答文本。

        require_citation=False 用于消融 #4（无引用约束对照组）。
        aggregate=True（跨文档聚合题）时改用 `llm.reasoning_effort_aggregate`：
        实测聚合题关思考包含匹配不降（1.00→1.00）而端到端 24~32s → 2s（PLAN「延迟实测」），
        是唯一「免费」的延迟收益；其余题型关思考 −8~−25pt，仍走全局 `llm.reasoning_effort`。
        未配置聚合档时回落到全局值（默认行为与不开关完全一致）。
        prompt 版本：`llm.prompt_version` 缺省 = tightened（行为逐字不变）；
        设 baseline 切回收紧前的 system prompt（三组对照的实验变量，PLAN §5.3）。
        计时见 `last_meta`：`cached=True` 时它的 ms 是本地缓存查询耗时，**不是模型延迟**，
        测真实延迟必须关缓存（`DOC_RAG_LLM_CACHE=0`）。
        """
        version = self.llm_cfg.get("prompt_version")
        system = prompts.resolve_system_answer(require_citation, version)
        llm_cfg = self.llm_cfg
        if aggregate:
            agg_effort = llm_cfg.get("reasoning_effort_aggregate")
            if agg_effort:  # 只在显式配置时覆盖，否则与全局一致（不悄悄改变行为）
                llm_cfg = {**llm_cfg, "reasoning_effort": agg_effort}
        text, meta = llm.chat_timed(
            llm_cfg,
            prompts.USER_ANSWER.format(
                context=prompts.format_context(chunks), question=question
            ),
            system_prompt=system,
        )
        self.last_meta = meta
        return text

    def answer_stream(
        self,
        question: str,
        chunks: list[dict],
        require_citation: bool = True,
        aggregate: bool | None = None,
    ):
        """流式回答：逐段 yield 文本；生成器耗尽后 `last_meta` 照常记录（含总耗时与用量）。

        与 `answer` 完全同构：同一 system prompt 选择逻辑（含 prompt_version）、
        同一聚合思考分流、同一缓存键——流式拼装写下的缓存，非流式调用直接命中。
        体感延迟方案（T7）：不等整段生成，首 token 即出。
        """
        version = self.llm_cfg.get("prompt_version")
        system = prompts.resolve_system_answer(require_citation, version)
        llm_cfg = self.llm_cfg
        if aggregate:
            agg_effort = llm_cfg.get("reasoning_effort_aggregate")
            if agg_effort:  # 只在显式配置时覆盖，否则与全局一致（不悄悄改变行为）
                llm_cfg = {**llm_cfg, "reasoning_effort": agg_effort}
        stream, meta = llm.chat_stream(
            llm_cfg,
            prompts.USER_ANSWER.format(
                context=prompts.format_context(chunks), question=question
            ),
            system_prompt=system,
        )

        def _gen():
            yield from stream
            self.last_meta = dict(meta)  # 生成器耗尽后 meta 才填充完整

        return _gen()

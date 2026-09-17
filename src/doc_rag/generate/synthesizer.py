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
        self, question: str, chunks: list[dict], require_citation: bool = True
    ) -> str:
        """chunks: [{"no", "text", "doc", "page"}] → 带编号引用的回答文本。

        require_citation=False 用于消融 #4（无引用约束对照组）。
        计时见 `last_meta`：`cached=True` 时它的 ms 是本地缓存查询耗时，**不是模型延迟**，
        测真实延迟必须关缓存（`DOC_RAG_LLM_CACHE=0`）。
        """
        system = prompts.SYSTEM_ANSWER if require_citation else prompts.SYSTEM_ANSWER_NO_CITE
        text, meta = llm.chat_timed(
            self.llm_cfg,
            prompts.USER_ANSWER.format(
                context=prompts.format_context(chunks), question=question
            ),
            system_prompt=system,
        )
        self.last_meta = meta
        return text

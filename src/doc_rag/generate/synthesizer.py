"""合成层（PLAN §5.2）：编号引用 + 无据拒答。"""

from __future__ import annotations

from . import llm, prompts


class Synthesizer:
    def __init__(self, llm_cfg: dict) -> None:
        self.llm_cfg = llm_cfg

    def answer(self, question: str, chunks: list[dict]) -> str:
        """chunks: [{"no", "text", "doc", "page"}] → 带编号引用的回答文本。"""
        return llm.chat(
            self.llm_cfg,
            prompts.USER_ANSWER.format(
                context=prompts.format_context(chunks), question=question
            ),
            system_prompt=prompts.SYSTEM_ANSWER,
        )

"""合成层（PLAN §5.2）：编号引用 + 无据拒答。

思考档是**一张按题型的表**（`llm.reasoning_effort_by_type`），不是「聚合题特例」：
未列出的题型跟随全局 `llm.reasoning_effort`。表的键只能是服务侧预测得出来的题型
（`PREDICTED_TYPES`）——fact / term 这些是黄金集的标注，线上拿不到。
"""

from __future__ import annotations

from . import llm, prompts

# `agent.predict_type` 的值域。写在这里而不是 import agent：合成层不该依赖 policy 层，
# 而 parity 测试保证两侧对同一个 plan 给出同一个预测。
PREDICTED_TYPES = ("single", "cross_doc", "time_filter")


def resolve_effort_cfg(llm_cfg: dict, question_type: str | None) -> dict:
    """按**预测**题型挑思考档。

    没配表、题型未知、或该题型不在表里 → 原样返回，行为与引入这张表之前逐字相同。
    表里显式写 `single: ""` 是合法的：意思是「这一类强制开思考」，即使全局是 none。
    """
    table = llm_cfg.get("reasoning_effort_by_type") or {}
    if not table or not question_type or question_type not in table:
        return llm_cfg
    return {**llm_cfg, "reasoning_effort": table[question_type]}


class Synthesizer:
    def __init__(self, llm_cfg: dict) -> None:
        self.llm_cfg = llm_cfg
        # 最近一次 answer 的计时元数据（llm.chat_timed 的返回）。
        # 用实例属性而不是改 answer 的返回类型：调用点（CLI/API/eval）都不用变，
        # 需要延迟的一方读它即可（PLAN「延迟口径」）。
        self.last_meta: dict | None = None
        # 填错键（`term: low`）在这一步就炸，而不是等到线上悄悄不生效：一个读不到的
        # 配置键比没有这个键更糟——它会让人以为策略已经落地。
        table = llm_cfg.get("reasoning_effort_by_type") or {}
        unknown = sorted(k for k in table if k not in PREDICTED_TYPES)
        if unknown:
            raise ValueError(
                f"reasoning_effort_by_type 含服务侧预测不到的题型：{unknown}"
                f"（可用键：{list(PREDICTED_TYPES)}）。fact/term 是黄金集标注，"
                "线上不可得；要按题型分档得先有一个可预测的信号。"
            )

    def answer(
        self,
        question: str,
        chunks: list[dict],
        require_citation: bool = True,
        question_type: str | None = None,
    ) -> str:
        """chunks: [{"no", "text", "doc", "page"}] → 带编号引用的回答文本。

        require_citation=False 用于消融 #4（无引用约束对照组）。
        `question_type` 是**预测**题型（`agent.predict_type` 的输出），只用来查思考档；
        真题型在服务侧不存在，拿它当开关会让五条入口的口径分叉（同 agent 层的理由）。
        实测依据：聚合题关思考包含匹配不降（1.00→1.00）而端到端 24~32s → 2s；
        其余题型关思考 −8~−25pt。2026-09-20 又量了一档 `low`：它只在思考量大的请求上
        起约束作用（term 慢尾 2,029→1,195 token、10.2s→5.9s），轻项上等于不传
        （337 vs 363），而在聚合题上只砍掉 20% 思考（5,463→4,360、22.9s→19.5s）——
        所以「全局 low 替换这张表」是拿聚合题 11 倍延迟换非聚合侧的一部分尾部。
        prompt 版本：`llm.prompt_version` 缺省 = tightened（行为逐字不变）；
        设 baseline 切回收紧前的 system prompt（三组对照的实验变量，PLAN §5.3）。
        计时见 `last_meta`：`cached=True` 时它的 ms 是本地缓存查询耗时，**不是模型延迟**，
        测真实延迟必须关缓存（`DOC_RAG_LLM_CACHE=0`）。
        """
        version = self.llm_cfg.get("prompt_version")
        system = prompts.resolve_system_answer(require_citation, version)
        llm_cfg = resolve_effort_cfg(self.llm_cfg, question_type)
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
        question_type: str | None = None,
    ):
        """流式回答：逐段 yield 文本；生成器耗尽后 `last_meta` 照常记录（含总耗时与用量）。

        与 `answer` 完全同构：同一 system prompt 选择逻辑（含 prompt_version）、
        同一思考档位查表、同一缓存键——流式拼装写下的缓存，非流式调用直接命中。
        体感延迟方案（T7）：不等整段生成，首 token 即出。
        """
        version = self.llm_cfg.get("prompt_version")
        system = prompts.resolve_system_answer(require_citation, version)
        llm_cfg = resolve_effort_cfg(self.llm_cfg, question_type)
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

"""Prompt 模板（PLAN §5.1 元数据抽取 / §5.2 引用合成 / §5.3 决议-讨论区分）。"""

SYSTEM_ANSWER = """\
你是企业文档问答助手。仅依据下方编号上下文回答，规则：
1. 每一句事实陈述都必须能在上下文中找到依据，并在句末标注来源编号，如 [1][2]。
2. 禁止推断、补全、概括：不得把上下文没有写出的内容写成结论；不得把多篇文档的信息
   合并成一个在任何单篇文档中都不存在的陈述。
3. 严格区分「会上决定」与「会上讨论过但未决定」——按原文措辞引用，不得把讨论升级为决议。
4. 上下文相互矛盾时分别陈述，并各自标注来源编号。
5. 上下文不足以回答时，明确回答"根据现有文档无法回答"，并说明缺少什么信息。
6. 涉及日期、参会人、数值、文号时逐字引用原文。
"""

USER_ANSWER = """\
上下文：
{context}

问题：{question}
"""


def format_context(chunks: list[dict]) -> str:
    """chunks: [{"no": 1, "text": ..., "doc": ..., "page": ...}] → 编号上下文块。"""
    parts = []
    for c in chunks:
        loc = f"（{c['doc']}" + (f" 第{c['page']}页" if c.get("page") else "") + "）"
        parts.append(f"[{c['no']}] {loc}\n{c['text']}")
    return "\n\n".join(parts)


METADATA_EXTRACTION = """\
从下面这份会议记录中抽取 JSON 元数据，字段：
- date: 会议日期（YYYY-MM-DD，找不到填 null）
- meeting_type: 会议类型（如 周会/项目会/评审会）
- attendees: 参会人列表（仅文中明确列出的）
- topics: 议题列表（每项不超过 15 字）
- decisions: 决议列表（仅"明确决定/一致通过"的事项；讨论过但未决定的事项不要列入）

会议记录：
{document}

只输出 JSON，不要输出其他内容。
"""

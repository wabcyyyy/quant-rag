"""Prompt 模板（PLAN §5.1 元数据抽取 / §5.2 引用合成 / §5.3 决议-讨论区分）。"""

SYSTEM_ANSWER = """\
你是企业文档问答助手。仅依据下方编号上下文回答，规则：
1. 每个事实陈述必须标注来源编号，如 [1][2]；不得使用上下文之外的知识。
2. 严格区分「会上决定」与「会上讨论过但未决定」——按原文措辞引用，不得把讨论升级为决议。
3. 上下文不足以回答时，明确回答"根据现有文档无法回答"，并说明缺少什么信息。
4. 涉及日期、参会人、数值、文号时逐字引用原文。
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

"""黄金集条目 schema（PLAN §5.3）。"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

# 语料里的发言人格式是 `@姓名：`，飞书导出还常带 `__…__` 装饰；markdown 标题/列表的
# 转义（`2\.3`）与强调号（`*`）也会被原样带进导出正文。这些字符在**文档侧**是原文的
# 一部分，但模型转述时不会照抄——留着就会把「答对了」判成没命中。
# 实测：臂 A 上限内的 7 条未命中里有 3 条只是装饰字符不同（`__…__` 强调、`2\.3` 与
# `\-` 的 markdown 转义），把它们从匹配口径里剔掉后 A 救回 2 条、25 块臂救回 5 条。
# 所以这不是假想敌：判据短语本身仍存原文（保 grounding），只在**比较**时去装饰。
_KP_STRIP_CHARS = "@_\\*"
_KP_WHITESPACE = re.compile(r"\s+")


def kp_normalize(s: str) -> str:
    """聚合题要点的唯一归一化口径：空白去掉，`@`/`_` 装饰去掉。

    出题侧（候选短语、全库唯一性）与判分侧（答案里有没有这句）必须走同一个函数，
    否则「逐字可核对」这件事会在某一环悄悄换成另一种字符串。放在 schema 而不是
    goldgen：判分器不该反向依赖出题器。
    """
    out = _KP_WHITESPACE.sub("", s or "")
    for ch in _KP_STRIP_CHARS:
        out = out.replace(ch, "")
    return out


class KeyPoint(BaseModel):
    """聚合题的一个「可归属要点」：某篇 gold 文档独有、可逐字核对的一句话。

    存在的理由：聚合题原先只有 1 个 `must_contain`（那个人名），而答案集有 10~56 篇。
    于是「答出 2 篇」和「答出 18 篇」在答案轨上得分相同——判据密度配不上答案集规模。
    一条要点绑死一篇文档，命中它才等于把那一篇答出来了。
    """

    doc_id: str
    phrase: str

    def hit_in(self, answer: str) -> bool:
        """这条要点是否出现在答案里（与出题侧同一个归一化口径）。"""
        return kp_normalize(self.phrase) in kp_normalize(answer)


class GoldItem(BaseModel):
    id: str
    type: str  # fact | decision | open_discussion | term | cross_doc | time_filter | no_answer
    question: str
    expected_answer: str
    must_contain: list[str] = Field(default_factory=list)
    source_doc_ids: list[str] = Field(default_factory=list)
    refusable: bool = False
    source_title: str | None = None
    # 聚合题的逐篇要点（见 `KeyPoint`）。缺省空列表 → runner 的分档指标为 None，
    # 不进分母：旧黄金集跑出来的每个既有数字必须逐字不变。
    key_points: list[KeyPoint] = Field(default_factory=list)
    # 题目来源标记：程序化构造题（如决议区提取）带 origin，重算 programmatic
    # 题型时据此整体重建——否则上一轮的程序化题会被当成 LLM 原题保留，越滚越多
    origin: str | None = None
    # v3 窗口依赖题：答全这道题**必须同时看到的块**。文档级覆盖率对这道题是瞎的
    # （gold 只有 1 篇文档），性质全靠这两块分居，所以要把它们记下来——
    # `goldgen.v3_property_violations` 据此复检，重新生成时性质丢了会直接报。
    required_chunk_ids: list[str] = Field(default_factory=list)

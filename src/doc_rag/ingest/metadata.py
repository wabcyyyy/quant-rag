"""doc 级元数据抽取（PLAN §5.1 基础版）：LLM 抽 date / meeting_type / attendees / topics。

文件名日期做第二信号交叉校验；LLM 失败不阻塞入库（PLAN §8 风险表）。
"""

from __future__ import annotations

import json
import re
from datetime import date

from ..generate import llm, prompts
from .schema import IntermediateDoc

# 「2026年42周」这类无「第」/无月日的周次串，曾被下面的日期正则回溯拆成
# 2026-04-02（「42」→ 月 4 日 2）。尾部 (?!周) 把周次串让给 _FILENAME_WEEK_RE，
# 日期正则只吃真有月日的东西。
_FILENAME_DATE_RE = re.compile(r"(20\d{2})[年\-/.]?(\d{1,2})[月\-/.]?(\d{1,2})(?!周)")
# 「2025年第44周」式周次（「第」可省，与语料两种命名都咬合）→ ISO 周一日期。
# 精度 ±1 周：年份过滤场景无损（过滤条件是整年区间）。
_FILENAME_WEEK_RE = re.compile(r"(20\d{2})年第?(\d{1,2})周")
_ISO_DATE_RE = re.compile(r"^(20\d{2})-(\d{1,2})-(\d{1,2})$")
# 正文日期（会议纪要开头通常写明）：2020-03-14 / 2020年3月14日 / 2020.03.14
_TEXT_DATE_RES = (
    re.compile(r"(20\d{2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})"),
    re.compile(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"),
)
_PROMPT_TEXT_CHARS = 3000
_TEXT_SCAN_CHARS = 600  # 正文头部扫描范围（日期一般在开头）


def week_from_filename(name: str) -> str | None:
    """「2025年第44周」式周次 → 该 ISO 周的周一日期；无周次串返回 None。

    为什么允许 ±1 周：周次解析的唯一消费方是 `doc_date` 的年份过滤（聚合题的
    `doc_date ∈ 年份`），整年区间下周一是周几无所谓。两个边界都往年内钳：
    ISO 周 1 的周一可能落在上一年（如 2019-W01 → 2018-12-31）→ 钳到 1 月 1 日；
    公司周制的「第 53 周」在 ISO 只有 52 周的年份不存在（2024/2025 即如此）→
    钳到 12 月 31 日——它只可能是 12 月下旬那几天，不处理就会重演 q063/q064 的
    「年份过滤把整周文档排除」。
    """
    m = _FILENAME_WEEK_RE.search(name or "")
    if not m:
        return None
    y, w = int(m.group(1)), int(m.group(2))
    if not 1 <= w <= 53:
        return None
    try:
        monday = date.fromisocalendar(y, w, 1)
    except ValueError:
        # 该年 ISO 无第 53 周：公司周制的年末周，钳到年内最后一天
        return date(y, 12, 31).isoformat()
    if monday.year != y:  # 仅周 1 的周一会落到上一年
        monday = date(y, 1, 1)
    return monday.isoformat()


def date_from_filename(name: str) -> str | None:
    m = _FILENAME_DATE_RE.search(name or "")
    if m:
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y}-{mo:02d}-{d:02d}"
    # 没有真日期再看周次：完整日期更准，优先；周次串在这里永远不会被
    # 上面的正则误吞（(?!周) 已把那条回溯路径封掉）
    return week_from_filename(name or "")


def normalize_date(value) -> str | None:
    """LLM 返回的日期规范化为 YYYY-MM-DD；不合法返回 None。"""
    if not isinstance(value, str):
        return None
    m = _ISO_DATE_RE.match(value.strip())
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not 1 <= mo <= 12 or not 1 <= d <= 31:
        return None
    return f"{y}-{mo:02d}-{d:02d}"


def date_from_text(text: str, scan_chars: int = _TEXT_SCAN_CHARS) -> str | None:
    """从正文抽取日期（会议纪要开头一般写明）。

    实测动机：仅从文件名解析时全库只有 14.2% 的块带 doc_date，
    年份过滤会误伤 86% 语料（见 PLAN §5.3 记录）。
    """
    for chunk in (text[:scan_chars], text):  # 先扫头部，再全篇兜底
        for rx in _TEXT_DATE_RES:
            m = rx.search(chunk)
            if m:
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 1 <= mo <= 12 and 1 <= d <= 31:
                    return f"{y}-{mo:02d}-{d:02d}"
    return None


def parse_llm_json(text: str) -> dict | None:
    """容忍 ```json 围栏与前后杂文本；解析失败返回 None。"""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def base_meta(doc: IntermediateDoc) -> dict:
    title = doc.meta.title or doc.meta.doc_id
    parts = title.split("_")
    return {
        # 文件名优先，正文兜底（实测：仅文件名时覆盖率仅 14.2%）
        "doc_date": date_from_filename(title) or date_from_text(doc.to_text()),
        # 文件名规范（实测语料）：大类_分组-描述_日期 → 零成本结构化信号
        "category": parts[0] if len(parts) > 1 else None,
        "doc_group": parts[1] if len(parts) > 2 else None,
        "meeting_type": None,
        "attendees": [],
        "topics": [],
    }


def extract_metadata(doc: IntermediateDoc, llm_cfg: dict) -> dict:
    """LLM 抽取；无 Key / 失败时退回文件名日期（不阻塞入库）。"""
    meta = base_meta(doc)
    if not (llm_cfg.get("api_key") and llm_cfg.get("model")):
        return meta
    try:
        reply = llm.chat(
            llm_cfg,
            prompts.METADATA_EXTRACTION.format(
                document=doc.to_text()[:_PROMPT_TEXT_CHARS]
            ),
            temperature=0,
        )
        data = parse_llm_json(reply) or {}
    except Exception:  # noqa: BLE001 元数据失败不阻塞入库
        return meta
    normalized = normalize_date(data.get("date"))
    if normalized:
        meta["doc_date"] = normalized
    meta["meeting_type"] = data.get("meeting_type") or None
    meta["attendees"] = [str(a) for a in (data.get("attendees") or [])][:30]
    meta["topics"] = [str(t)[:20] for t in (data.get("topics") or [])][:10]
    return meta

"""doc 级元数据抽取（PLAN §5.1 基础版）：LLM 抽 date / meeting_type / attendees / topics。

文件名日期做第二信号交叉校验；LLM 失败不阻塞入库（PLAN §8 风险表）。
"""

from __future__ import annotations

import json
import re

from ..generate import llm, prompts
from .schema import IntermediateDoc

_FILENAME_DATE_RE = re.compile(r"(20\d{2})[年\-/.]?(\d{1,2})[月\-/.]?(\d{1,2})")
_ISO_DATE_RE = re.compile(r"^(20\d{2})-(\d{1,2})-(\d{1,2})$")
_PROMPT_TEXT_CHARS = 3000


def date_from_filename(name: str) -> str | None:
    m = _FILENAME_DATE_RE.search(name or "")
    if not m:
        return None
    y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
    if not 1 <= mo <= 12 or not 1 <= d <= 31:
        return None
    return f"{y}-{mo:02d}-{d:02d}"


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


def parse_llm_json(text: str) -> dict | None:
    """容忍 ```json 围栏与前后杂文本；解析失败返回 None。"""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
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
        "doc_date": date_from_filename(title),
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
            prompts.METADATA_EXTRACTION.format(document=doc.to_text()[:_PROMPT_TEXT_CHARS]),
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

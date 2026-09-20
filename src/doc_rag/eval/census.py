"""v3 出题前的语料普查（PLAN §5.5 P0.5 的第②项：表格密度 + 时间线可造性）。

为什么这件事必须先做：PLAN §5.3 的「全库 179 个表格、固定 512 切下 85 个被切断」
只证明了**表格存在**与**分块会切断它们**，没证明**能凑出跨篇数值对比题**。
如果 179 个表格集中在极少数几篇、或者表格里根本没有可对齐的数值行，那么
「v3 的新题型之一 = 跨篇表格数值」这条动机就是空的——出题前先量一次，
比出完 24 条再发现判不了要便宜得多。

零成本：只读 `data/parsed` 的中间 JSON，不碰 Qdrant、不碰 LLM。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

# 语料是从飞书/Word 导出后迁进来的，单元格里夹着零宽空格：不去掉会把同一列名
# 数成两个标签，密度就被高估。
_ZERO_WIDTH = re.compile("[\u200b\ufeff]")
# 数值单元格：数字为主，允许千分位、小数、金额单位与百分号。
# 只要「有数字」就算数值行——普查要的是密度，不是记账精度。
_NUMERIC = re.compile(r"\d+(?:[.,]\d+)?\s*(?:元|万元|万|亿|%)?")
# 标题里的时间线信号：`2025年第30周` / `2026年3月` / 裸年份。doc_date 只有 18.8%
# 覆盖（文件名推断来的），所以出题时日期靠标题而不是靠 payload。
_PERIOD = re.compile(r"(20\d{2})(?:年第(\d{1,2})周|年(\d{1,2})月|年)")
# 标题结构：`类别_子类_主题`（`会议档案_周会_2025年第33周-议题4-吴抒允绩效`）
_TOPIC_SEP = "_"
# 「跨篇同一行标签」里要先剔掉的假标签：日期行与序号行。它们形似可对比项，
# 实际不是——实测 179 个表格里凑出的 13 个跨篇标签中，8 个是日期行、4 个是序号列，
# 剩 1 个才是真项目标签。不剔就会把「能出 8 条题」误判成成立。
_DATE_LABEL = re.compile(r"^(?:20\d{2}[-/年.]|\d{1,2}[-/月.]|20\d{2}$)")
_ORDINAL_LABEL = re.compile(r"^\d{1,3}$|^[一二三四五六七八九十]{1,3}$")


def meaningless_label(label: str) -> bool:
    """日期行 / 序号列 / 合计行：出现在多篇也不构成「同一项目的跨篇数值对比」。"""
    stripped = label.strip()
    return bool(
        _DATE_LABEL.match(stripped)
        or _ORDINAL_LABEL.match(stripped)
        or stripped in {"合计", "总计", "小计", "Total", "total", "合计:", "合计："}
    )


def _cells(row: str) -> list[str]:
    parts = [p for p in row.strip().strip("|").split("|")]
    return [_ZERO_WIDTH.sub("", p).strip() for p in parts]


def table_rows(text: str) -> list[list[str]]:
    """markdown 表格 → 行列表（丢掉 `---` 分隔行与空行）。"""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = _cells(line)
        if not cells or all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows


def numeric_pairs(rows: list[list[str]]) -> list[tuple[str, str]]:
    """(行标签, 首个数值单元格)。表头行天然被跳过：它的第二格通常是列名而非数字。"""
    out = []
    for cells in rows:
        if len(cells) < 2:
            continue
        label = cells[0]
        if not label:
            continue
        for cell in cells[1:]:
            m = _NUMERIC.search(cell)
            if m and not cell.strip().startswith(("√", "×", "✓")):
                out.append((label, cell.strip()))
                break
    return out


def period_of(title: str) -> str | None:
    m = _PERIOD.search(title or "")
    if not m:
        return None
    year, week, month = m.group(1), m.group(2), m.group(3)
    if week:
        return f"{year}W{int(week):02d}"
    if month:
        return f"{year}-{int(month):02d}"
    return year


def topic_of(title: str) -> str:
    """标题去掉类别前缀与时间戳之后的主题干。

    `会议档案_周会_2025年第33周-议题4-吴抒允绩效` → `周会/吴抒允绩效`：
    保留子类（周会/月会…）是因为同名主题在不同子类下不是同一条线索。
    """
    parts = (title or "").split(_TOPIC_SEP)
    if len(parts) >= 3:
        sub, rest = parts[1], _TOPIC_SEP.join(parts[2:])
    elif len(parts) == 2:
        sub, rest = parts[0], parts[1]
    else:
        sub, rest = "", title or ""
    rest = re.sub(r"20\d{2}(?:年第\d{1,2}周|年\d{1,2}月|年)", "", rest)
    rest = re.sub(r"议题\d+", "", rest)
    tail = [p for p in rest.replace("/", "-").split("-") if p.strip()]
    leaf = tail[-1].strip() if tail else rest.strip()
    return f"{sub}/{leaf}" if sub else leaf


def _iter_docs(parsed_dir: Path):
    for path in sorted(parsed_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # 解析产物读不通是 indexer 的清理判据，不是普查的事
        yield path, doc


def census(parsed_dir: Path | str, sample_pairs: int = 8) -> dict:
    """普查 `parsed_dir`，返回表格密度与时间线可造性两份结论。"""
    root = Path(parsed_dir)
    docs = 0
    docs_with_tables = 0
    tables_total = 0
    tables_with_numbers = 0
    per_doc_tables: list[int] = []
    # 标签 → {doc_id: 该文档里这个标签的数值}：跨篇对比题的候选池
    by_label: dict[str, dict[str, str]] = defaultdict(dict)
    topics: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    dated_docs = 0

    for path, doc in _iter_docs(root):
        meta = doc.get("meta") or {}
        title = str(meta.get("title") or path.stem)
        doc_id = str(meta.get("doc_id") or path.stem)
        blocks = doc.get("blocks") or []
        docs += 1
        tables = [b for b in blocks if (b.get("type") == "table")]
        if tables:
            docs_with_tables += 1
        tables_total += len(tables)
        per_doc_tables.append(len(tables))
        any_number = False
        for table in tables:
            pairs = numeric_pairs(table_rows(table.get("text") or ""))
            if not pairs:
                continue
            any_number = True
            for label, value in pairs:
                by_label.setdefault(label, {}).setdefault(doc_id, value)
        if any_number:
            tables_with_numbers += 1
        period = period_of(title)
        if period:
            dated_docs += 1
            topics[topic_of(title)][period].append(doc_id)

    cross_all = {
        label: sorted(values) for label, values in by_label.items() if len(values) >= 2
    }
    cross = {
        label: docs_
        for label, docs_ in cross_all.items()
        if not meaningless_label(label)
    }
    # 同一主题、≥2 个不同时间点 → 时间线题的候选（gold 只取后一次决议，
    # 见 PLAN §5.5 的 F1：规则 3 不放宽时「推翻」只能作为检索挑战存在）
    timelines = {
        topic: {p: ids for p, ids in sorted(periods.items())}
        for topic, periods in topics.items()
        if len(periods) >= 2
    }
    ordered = sorted(per_doc_tables)

    return {
        "parsed_dir": str(root),
        "docs": docs,
        "docs_with_tables": docs_with_tables,
        "tables_total": tables_total,
        "tables_with_numbers": tables_with_numbers,
        "tables_per_doc": {
            "max": ordered[-1] if ordered else 0,
            "p50": ordered[len(ordered) // 2] if ordered else 0,
            "docs_with_ge_2_tables": sum(1 for n in ordered if n >= 2),
        },
        "numeric_labels_total": len(by_label),
        # 跨篇数值对比题的原料：同一行标签在 ≥2 篇各带一个数值。`raw` 含日期行与序号列，
        # `cross_doc_labels` 是剔掉它们之后的数——两者差得很远，只报一个就会把
        # 「这份语料其实凑不出题」读成「凑得出」。
        "cross_doc_labels_raw": len(cross_all),
        "cross_doc_labels": len(cross),
        "cross_doc_label_docs": len({d for values in cross.values() for d in values}),
        "cross_doc_examples": [
            {"label": label, "docs": docs_[:3]}
            for label, docs_ in sorted(cross.items(), key=lambda kv: -len(kv[1]))[
                :sample_pairs
            ]
        ],
        "docs_with_date_in_title": dated_docs,
        "timeline_topics": len(timelines),
        "timeline_examples": [
            {"topic": t, "periods": dict(list(p.items())[:4])}
            for t, p in sorted(timelines.items(), key=lambda kv: -len(kv[1]))[
                :sample_pairs
            ]
        ],
    }


def verdict(report: dict, want_items: int = 8) -> dict:
    """把普查数字翻译成「这两类题能不能出」的结论，不让人自己心算。

    跨篇数值题的下限是**标签数**：同两篇文档可以供给多条题（三个项目标签就是三条
    「A 与 B 的 X 各是多少」），所以不要求标签数 × 2 篇文档；只要求真的覆盖 ≥2 篇。
    时间线题要的是**同主题多时间点**的组数。任一项不足就说明那条动机是空的，得降级题型。
    """
    labels = int(report["cross_doc_labels"])
    docs = int(report["cross_doc_label_docs"])
    topics = int(report["timeline_topics"])
    return {
        "cross_doc_numeric_viable": labels >= want_items and docs >= 2,
        "cross_doc_labels": labels,
        "cross_doc_docs_covered": docs,
        "timeline_viable": topics >= want_items,
        "timeline_topics": topics,
    }

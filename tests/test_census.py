"""v3 出题前普查的测试。

普查的唯一职责是回答「这两类新题型到底凑不凑得出来」，所以测试锁的是**判据**而不是
字符串：日期行与序号列必须被剔掉（它们在真实语料里占了跨篇标签的绝大多数，留着就会
把「凑不出题」判成「凑得出」），而时间线要按「同主题 + 多时间点」数。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_rag.eval.census import (
    census,
    meaningless_label,
    numeric_pairs,
    period_of,
    table_rows,
    topic_of,
    verdict,
)


def _doc(doc_id: str, title: str, blocks: list[dict]) -> dict:
    return {
        "meta": {
            "source_type": "pdf",
            "doc_id": doc_id,
            "title": title,
            "owner": None,
            "created_at": None,
            "edited_at": None,
        },
        "blocks": blocks,
    }


def _table(text: str) -> dict:
    return {"type": "table", "text": text, "page": 1}


def _write(root: Path, docs: list[dict]) -> Path:
    for doc in docs:
        doc_id = doc["meta"]["doc_id"]
        (root / f"{doc_id}.json").write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8"
        )
    return root


QUOTATION = """| 费用项目 | 报价（元） | 说明 |
| 方案设计 | 500 | 总体方案 |
| 开发 | 1500 | 三个爬虫 |
| 运维 | 100元 | 监控运行 |
"""

# 同一张表的两个版本：行标签是日期，数值也确实是数，但它们不是「同一项目的跨篇对比」
SCHEDULE = """| 日期 | 到场人数 |
| 2025-12-01 | 8 |
| 2025-12-05 | 9 |
"""

VOTE = """|  | 同意 | 反对 | 弃权 |
| 刘雨杉 | √ |  |  |
| 严淑 | √ |  |  |
"""


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "parsed"
    root.mkdir()
    return _write(
        root,
        [
            _doc("a1", "工作档案_SSC职能_贴吧项目报价单", [_table(QUOTATION)]),
            _doc("a2", "工作档案_SSC职能_官网项目报价单", [_table(QUOTATION)]),
            _doc("b1", "会议档案_周会_2025年第33周-议题1-排班", [_table(SCHEDULE)]),
            _doc("b2", "会议档案_周会_2025年第34周-议题1-排班", [_table(SCHEDULE)]),
            _doc("c1", "会议档案_周会_2025年第35周-议题2-表决", [_table(VOTE)]),
            # 无表格 / 读不通的文件：都不该把整次普查带崩
            _doc(
                "d1",
                "工作档案_SSC职能_周报",
                [{"type": "paragraph", "text": "没有表格"}],
            ),
        ],
    )


def test_counts_tables_and_docs(corpus):
    report = census(corpus)
    assert report["docs"] == 6
    assert report["docs_with_tables"] == 5
    assert report["tables_total"] == 5
    # 表决表只有 √，没有任何数值行——它不该被算进「有数值的表格」
    assert report["tables_with_numbers"] == 4


def test_cross_doc_labels_drop_dates_and_ordinals(corpus):
    report = census(corpus)
    # raw 把日期行也算进来了（2025-12-01 / 2025-12-05 各跨两篇）
    assert report["cross_doc_labels_raw"] == 5
    # 剔掉日期之后只剩报价单的三个项目标签
    assert report["cross_doc_labels"] == 3
    labels = {e["label"] for e in report["cross_doc_examples"]}
    assert labels == {"方案设计", "开发", "运维"}
    assert report["cross_doc_label_docs"] == 2


@pytest.mark.parametrize(
    "label",
    ["2025-12-01", "2026年3月", "1", "12", "三", "合计", "总计", "Total", "小计"],
)
def test_meaningless_labels(label):
    assert meaningless_label(label)


@pytest.mark.parametrize("label", ["方案设计", "开发", "入职日期", "履约保证金"])
def test_meaningful_labels(label):
    assert not meaningless_label(label)


def test_verdict_uses_the_configured_bar(corpus):
    report = census(corpus)
    assert verdict(report, want_items=8)["cross_doc_numeric_viable"] is False
    # 3 个标签来自同两篇文档：题数由标签数决定，不是标签数 × 2 篇
    assert verdict(report, want_items=3)["cross_doc_numeric_viable"] is True
    assert verdict(report, want_items=4)["cross_doc_numeric_viable"] is False


def test_verdict_needs_at_least_two_documents(corpus):
    """标签再多，只有一篇文档带数值表就出不了「跨篇」题。"""
    one = census(corpus)
    one["cross_doc_label_docs"] = 1
    assert verdict(one, want_items=1)["cross_doc_numeric_viable"] is False


def test_timeline_topics_need_same_topic_multiple_periods(corpus):
    report = census(corpus)
    # b1/b2 同主题两个周次算一条候选；c1 只有一个周次，不构成时间线
    assert report["docs_with_date_in_title"] == 3
    topics = {e["topic"] for e in report["timeline_examples"]}
    assert "周会/排班" in topics
    assert "周会/表决" not in topics
    assert report["timeline_topics"] == 1
    assert verdict(report, want_items=1)["timeline_viable"] is True
    # 判据不能恒真：门槛抬到 2 就该不成立
    assert verdict(report, want_items=2)["timeline_viable"] is False


def test_zero_width_space_does_not_split_a_label():
    """「刘雨杉\\u200b」与「刘雨杉」必须是同一个标签，否则密度被虚高。"""
    assert table_rows(VOTE)[1][0] == "刘雨杉"


def test_period_and_topic_parsing():
    assert period_of("会议档案_周会_2025年第33周-议题4-吴抒允绩效") == "2025W33"
    assert period_of("工作档案_2026年3月-总结") == "2026-03"
    assert period_of("没有时间的标题") is None
    assert topic_of("会议档案_周会_2025年第33周-议题4-吴抒允绩效") == "周会/吴抒允绩效"
    assert topic_of("单段标题") == "单段标题"


def test_vote_table_yields_no_numeric_pairs():
    assert numeric_pairs(table_rows(VOTE)) == []
    pairs = numeric_pairs(table_rows(QUOTATION))
    assert ("方案设计", "500") in pairs
    assert ("运维", "100元") in pairs  # 带单位的数值照收，普查要的是密度不是精度


def test_unreadable_parsed_file_is_skipped(corpus):
    (corpus / "broken.json").write_text("{not json", encoding="utf-8")
    assert census(corpus)["docs"] == 6

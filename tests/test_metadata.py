from doc_rag.ingest.bm25 import build_bm25_text
from doc_rag.ingest.metadata import (
    date_from_filename,
    normalize_date,
    parse_llm_json,
    week_from_filename,
)


def test_date_from_filename():
    assert date_from_filename("20260901-XX项目周会.pdf") == "2026-09-01"
    assert date_from_filename("2026-9-1 周会.docx") == "2026-09-01"
    assert date_from_filename("2026年9月1日会议纪要.doc") == "2026-09-01"
    assert date_from_filename("会议纪要.pdf") is None
    assert date_from_filename("20261399.pdf") is None  # 非法月日


def test_week_from_filename_iso_monday():
    """周次 → ISO 周一（±1 周精度，年份过滤场景无损；UPGRADE 2026 §2.2）。"""
    assert week_from_filename("会议档案_周会_2025年第36周-议题5- Proposal") == (
        "2025-09-01"
    )
    # 无「第」的变体（正是曾经被误解析成日期的那种串）
    assert week_from_filename("档案_2025年44周-会议纪要") == "2025-10-27"
    assert week_from_filename("会议纪要.pdf") is None
    assert week_from_filename("2025年第99周-会议纪要") is None  # 非法周号
    # 带「第」的周次串不是被 month/day 误吞的日期——回归锁
    assert date_from_filename("会议档案_周会_2025年第36周-议题5-X") == "2025-09-01"


def test_week_from_filename_does_not_misparse_as_date():
    """「2026年42周」曾被日期正则回溯拆成 2026-04-02（月 4 日 2）——修复后必须是
    ISO 周 42 的周一，且任何情况下不再吐出那个假日期。"""
    got = date_from_filename("工作档案_战略职能_2026年42周-客户研讨")
    assert got == "2026-10-12"
    assert got != "2026-04-02"


def test_week53_in_52_week_year_clamped_to_year_end():
    """公司周制的「第53周」在 ISO 只有 52 周的年份不存在（2024/2025 即如此）：
    fromisocalendar 会直接抛 ValueError。不处理 = 这 19 篇文档没有 doc_date，
    年份过滤照样把整周排除（q063/q064 同款病）；钳到 12-31 落回年份区间。
    2020 年 ISO 真有 53 周，取真实周一，不受钳制影响。"""
    assert week_from_filename("档案_2025年第53周-会议纪要") == "2025-12-31"
    assert week_from_filename("档案_2024年第53周-会议纪要") == "2024-12-31"
    assert week_from_filename("档案_2020年第53周-会议纪要") == "2020-12-28"


def test_week1_monday_clamped_into_the_year():
    """ISO 周 1 的周一可能落在上一年（2019-W01 → 2018-12-31、2020-W01 → 2019-12-30）：
    钳到 1 月 1 日，否则「2019 年的文档」会被 `doc_date ∈ 2019` 过滤整族排除
    （q063/q064 同款病）。2021-W01 周一在年内，是「不触发钳制」的对照。"""
    assert week_from_filename("档案_2019年第1周-会议纪要") == "2019-01-01"
    assert week_from_filename("档案_2020年第1周-会议纪要") == "2020-01-01"
    assert week_from_filename("档案_2021年第1周-会议纪要") == "2021-01-04"


def test_full_date_wins_over_week_in_same_title():
    """同一标题里真日期与周次并存：完整日期更准，优先。"""
    assert (
        date_from_filename("会议档案_2025年周会_2025-04-28_2025年第17周-纪要")
        == "2025-04-28"
    )


def test_normalize_date():
    assert normalize_date("2026-09-01") == "2026-09-01"
    assert normalize_date("2026-9-1") == "2026-09-01"
    assert normalize_date("2026年9月1日") is None  # LLM 偶发返回中文格式 → 拒绝
    assert normalize_date(None) is None


def test_parse_llm_json():
    assert parse_llm_json('```json\n{"date": "2026-09-01"}\n```') == {
        "date": "2026-09-01"
    }
    assert parse_llm_json('前缀 {"topics": ["预算"]} 后缀') == {"topics": ["预算"]}
    assert parse_llm_json("不是 json") is None
    assert parse_llm_json("") is None


def test_bm25_text_segments_chinese():
    out = build_bm25_text("供应商预付款政策。")
    assert " " in out  # 已分词
    assert "供应商" in out and "预付款" in out

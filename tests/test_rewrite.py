from doc_rag.retrieve.rewrite import QueryRewriter


def _rw():
    return QueryRewriter({"aggregate_top_n": 25})


def test_aggregation_question_uses_entity_focused_query():
    plan = _rw().rewrite("关于「黄日航」，公司文档里出现过哪些讨论或安排？")
    assert plan["aggregate"] is True
    assert plan["rewritten"] == "黄日航"  # 剥离模板话术
    assert plan["top_n"] == 25  # 放宽预算
    assert plan["filters"] is None


def test_year_question_adds_date_filter():
    plan = _rw().rewrite("2021年的文档中，关于「何李健」有哪些记录？")
    assert plan["aggregate"] is True
    assert plan["rewritten"] == "何李健"
    rng = plan["filters"]["doc_date"]
    assert rng["gte"] == "2021-01-01T00:00:00" and rng["lt"] == "2022-01-01T00:00:00"


def test_factual_question_is_untouched():
    plan = _rw().rewrite("2020年3月14日周会讨论了李小均和陈凡的去留吗？")
    assert plan["aggregate"] is False
    assert plan["rewritten"] == "2020年3月14日周会讨论了李小均和陈凡的去留吗？"
    assert plan["top_n"] is None
    # 非时间限定聚合题不加日期过滤：本语料 doc_date 仅覆盖 ~19% 文档，
    # 过滤会误伤 80% 语料（实测 fact 覆盖率 1.00→0.33）
    assert plan["filters"] is None
    assert "doc_date 稀疏" in plan["reason"]


def test_quoted_entity_without_aggregation_intent_stays_question():
    plan = _rw().rewrite("「量潮课堂」的定位是什么？")
    assert plan["aggregate"] is False
    assert plan["rewritten"] == "「量潮课堂」的定位是什么？"

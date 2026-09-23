"""判分口径的回归护栏：名字必须说实话，未定义不能印成 0。

六条各挡一类曾经过的错误读数：
1. `recall_at_k` 这个名字当年被叫错过一次：手上有 ranked 清单之前，逐条只存了
   「首命中位次」，所以报出来的其实是 Hit@k——64 条可答题里 44 条只有 1 篇 gold，
   剩下 20 条 gold 有 10~56 篇，在 8 个槽位上报 Recall 是自欺。于是它被改名成
   `hit_at_k`，`recall_at_*` 空出来等真正的定义。2026-09-21 起 ranked 清单逐条落盘，
   **两个名字都在，且必须给出不同的数**（相等就说明有一个是假的）。
2. 覆盖率必须和自己的结构上限一起报：清单 8 格、gold 56 篇时 0.143 就是满分。
3. `coverage_by_type` 不能给无 gold 的题型（no_answer）编一个 0.0——那是把
   「未定义」印成「测出来是 0」，和那个恒真的 over_refusal 同源。
4. 过度拒答只看「must_contain 逐字在上下文里」会漏掉「正确文档已进上下文却拒答」。
5. 过滤回退丢掉过滤必须计数。
6. 那条宽松包含轨是**字符子序列**匹配，假阳性没有上界——名字必须说实话（原叫
   `_contains_loose`，读数与严格口径同值时它还看不出来）。
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from doc_rag.eval import runner
from doc_rag.retrieve.hybrid import RetrievalOutcome

_ANSWERS = {
    "健身房预算多少？": "预算 67 元 [1]。",
    "关于张三有哪些记录？": "关于张三有多次记录 [1]。",
    "调度方案的决议是什么？": "根据现有文档无法回答：上下文未记载该决议。",
    "公司碳排放制度是什么？": "根据现有文档无法回答。",
}


def _chunk(doc_id: str, text: str) -> dict:
    return {
        "doc_id": doc_id,
        "title": f"文档{doc_id}",
        "page": 1,
        "text": text,
        "block_type": "paragraph",
    }


def _gold_file(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "q001",
                        "type": "fact",
                        "question": "健身房预算多少？",
                        "expected_answer": "67元",
                        "source_doc_ids": ["d1"],
                        "must_contain": ["67元"],
                    },
                    {
                        "id": "q002",
                        "type": "cross_doc",
                        "question": "关于张三有哪些记录？",
                        "expected_answer": "聚合题",
                        "source_doc_ids": [f"x{i}" for i in range(12)],
                        "must_contain": ["张三"],
                    },
                    {
                        "id": "q003",
                        "type": "decision",
                        "question": "调度方案的决议是什么？",
                        "expected_answer": "周一晚7点",
                        "source_doc_ids": ["d3"],
                        "must_contain": ["周一晚7点"],
                    },
                    {
                        "id": "q004",
                        "type": "no_answer",
                        "question": "公司碳排放制度是什么？",
                        "expected_answer": "应拒答",
                        "source_doc_ids": [],
                        "must_contain": [],
                        "refusable": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _run(monkeypatch, gold, **kw):
    # 8 条清单，长度故意不等于 max_contexts，用来验 list_len 是否落盘
    chunks = [
        _chunk("d1", "费用67元"),
        _chunk("x1", "张三出席了会议"),
        _chunk("d3", "议题：调度方案进度汇报"),
    ] + [_chunk(f"o{i}", "无关正文") for i in range(5)]

    retriever = Mock(collection="c", cfg={})
    retriever.retrieve.return_value = RetrievalOutcome(chunks=chunks)
    syn = Mock()
    syn.answer.side_effect = lambda q, *a, **k: _ANSWERS[q]
    syn.last_meta = None
    monkeypatch.setattr(runner, "_build_retriever", Mock(return_value=(retriever, syn)))
    return runner.evaluate(
        gold,
        cfg={"retrieval": {"max_contexts": 6}, "llm": {"model": "m"}},
        use_rewrite=False,
        use_rerank=False,
        **kw,
    )


@pytest.fixture
def run(tmp_path, monkeypatch):
    return _run(monkeypatch, _gold_file(tmp_path))


def test_hit_and_recall_are_both_reported_and_differ(run):
    """Hit@5 与 Recall@5/Precision@5 同时存在、且**数值不同**——这是各自说实话的证据。

    当年只有 `recall_at_k` 这个名字、没有逐条 ranked 清单，报的其实是 Hit@k，
    于是把它改名成 hit 并把 recall 空出来（本文件模块 docstring 第 1 条）。
    2026-09-21 起 ranked 清单落盘，recall 才第一次真可算。
    **同一截断下几个数若相等，说明其中有一个是假的。**
    """
    s = run["summary"]
    assert s["hit_at_5"] == 1.0 and s["hit_at_8"] == 1.0
    assert s["recall_at_list_macro"] == pytest.approx((1 + 1 / 12 + 1) / 3, abs=1e-4)
    assert s["recall_at_5"] != s["hit_at_5"]
    # 标准族只有一个截断：K=5。曾经扫 1/3/5/10，其中 @10 在 8 格清单上逐条等于
    # 整条清单的召回——一个看着独立、其实是复制的假数据点，所以删。
    for m in ("recall_at_1", "recall_at_3", "recall_at_10", "precision_at_list"):
        assert m not in s, m
    for m in ("recall_at_5", "precision_at_5", "ndcg_at_5", "map_at_5"):
        assert s[m] is not None, m
    # 归一化召回是**逐条比值取均值**，不是两个均值相除——两者在这里就不相等
    assert s["recall_vs_ceiling_macro"] != pytest.approx(
        s["recall_at_list_macro"] / s["recall_ceiling_macro"], abs=1e-3
    )
    # 旧名从标准名派生、同值：外部脚本与历史命令不失效，但两个名字不可能各自漂移
    assert s["mean_doc_coverage"] == s["recall_at_list_macro"]
    assert s["hit_within_budget"] == s["hit_at_list"]
    assert s["coverage_ceiling_mean"] == s["recall_ceiling_macro"]


def test_rank_metrics_are_the_textbook_definitions():
    """`_rank_metrics` 的手算锁：去重、截断、AP 的分子分母都要能一格格数出来。

    清单（含重复，同篇第二个块不占新位）：d1 d2 d1 d3 d4 d5
    去重保序后：                        d1 d2 d3 d4 d5
    gold = {d2, d5}

      K=5：top5 = d1..d5，命中 2 篇
        Recall@5    = 2/2 = 1.0
        Precision@5 = 2/5 = 0.4      ← 分母是 k，不是块数（旧写法给的是 2/6）
        AP@5        = (1/2 + 2/5) / min(#gold=2, 5) = 0.9/2 = 0.45   命中在第 2、5 位
      K=3：top3 = d1 d2 d3，命中 1 篇
        Recall@3 = 1/2 = 0.5 · Precision@3 = 1/3 · AP@3 = (1/2)/min(2,3) = 0.25
    """
    got = ["d1", "d2", "d1", "d3", "d4", "d5"]
    gold = {"d2", "d5"}
    out = runner._rank_metrics(got, gold)
    assert out["recall_at_5"] == pytest.approx(1.0)
    assert out["precision_at_5"] == pytest.approx(2 / 5, abs=1e-4)
    assert out["ap_at_5"] == pytest.approx((1 / 2 + 2 / 5) / 2, abs=1e-4)
    k3 = runner._rank_metrics(got, gold, k=3)
    assert k3["recall_at_3"] == pytest.approx(0.5)
    assert k3["precision_at_3"] == pytest.approx(1 / 3, abs=1e-4)
    assert k3["ap_at_3"] == pytest.approx(0.25)
    # 无 gold（no_answer 题）→ 空 dict，不是全 0：未定义不能印成 0
    assert runner._rank_metrics(got, set()) == {}


def test_coverage_is_reported_with_its_own_ceiling(run):
    s = run["summary"]
    # q001 1/1, q002 1/12, q003 1/1 → macro 覆盖率
    assert s["mean_doc_coverage"] == pytest.approx((1 + 1 / 12 + 1) / 3, abs=1e-4)
    # 清单只有 8 条：q002 的上限是 8/12，不是 1.0
    assert s["coverage_ceiling_mean"] == pytest.approx((1 + 8 / 12 + 1) / 3, abs=1e-4)
    assert s["coverage_by_type"]["cross_doc"] == pytest.approx(1 / 12, abs=1e-4)
    assert s["coverage_ceiling_by_type"]["cross_doc"] == pytest.approx(8 / 12, abs=1e-4)


def test_types_without_gold_get_no_phantom_zero(run):
    """no_answer 题按定义无 gold —— 它的覆盖率是未定义，不是 0.0。"""
    s = run["summary"]
    assert "no_answer" not in s["coverage_by_type"]
    assert "no_answer" not in s["coverage_ceiling_by_type"]
    assert s["refusal_acc"] == 1.0


def test_over_refusal_gold_catches_what_ctx_definition_misses(run):
    """q003：正确文档 d3 已进上下文（第 3 位）却答「无法回答」。

    must_contain（周一晚7点）不在那块正文里，所以按「上下文含原话」口径它不算
    过度拒答——那正是漏报。按「gold 已进上下文」口径必须抓到它。
    """
    s = run["summary"]
    row = {r["id"]: r for r in run["items"]}["q003"]
    assert row["over_refusal"] is False  # 旧口径：原话不在块里
    assert row["over_refusal_gold"] is True
    assert s["over_refusal_rate"] == 0.0
    assert s["over_refusal_gold_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_arm_self_documents_list_and_context_lengths(run):
    """臂间可比性的证据：清单长度与上下文长度都落盘。"""
    assert run["summary"]["list_len"] == {
        "retrieved_min": 8,
        "retrieved_max": 8,
        "contexts_min": 6,
        "contexts_max": 6,
    }
    assert run["items"][0]["n_retrieved"] == 8
    assert run["items"][0]["n_contexts"] == 6
    assert run["meta"]["filter_fallback_n"] == 0  # 这条臂没走过回退
    assert run["meta"]["budget"] == "fixed:8"


def test_filter_fallback_is_counted_not_silently_swallowed():
    """过滤后结果过少 → 回退成「不过滤」，这件事必须留在结果上。

    回退不是 bug（字段稀疏时它保证不把答案滤没），但一条自称带过滤的臂里混进
    多少条其实没过滤，只有计数才知道——覆盖率的涨有多少来自放弃过滤才量得出来。
    """
    from doc_rag.retrieve.hybrid import HybridRetriever

    class _Point:
        def __init__(self, i):
            self.payload = {"doc_id": f"d{i}", "text": "正文", "title": "t", "page": 1}
            self.score = 1.0

    class _Client:
        def __init__(self):
            self.queries: list = []

        def query_points(self, collection, **kw):
            self.queries.append(kw.get("query_filter"))
            n = 1 if len(self.queries) == 1 else 5  # 带过滤时只有 1 条 → 触发回退
            return Mock(points=[_Point(i) for i in range(n)])

    class _Embedder:
        def embed(self, texts):
            return [[0.0] * 4 for _ in texts]

    client = _Client()
    r = HybridRetriever(
        client=client,
        embedder=_Embedder(),
        collection="c",
        retrieval_cfg={"mode": "dense", "fusion_limit": 8},
    )
    out = r.retrieve("问题", filters={"doc_date": {"gte": "2026-01-01T00:00:00"}})
    assert out.filter_applied is True
    assert out.filter_fallback is True
    assert out.n_before_fallback == 1
    assert len(out.chunks) == 5
    assert client.queries[0] is not None and client.queries[1] is None  # 第二次没带过滤

    kept = r.retrieve("问题", filters=None)
    assert kept.filter_applied is False and kept.filter_fallback is False


def test_retrieval_only_leaves_answer_metrics_undefined(tmp_path, monkeypatch):
    """只评检索时没有任何答案：两条拒答口径必须是 None，不能报 0.0。

    「未定义」印成「测出来是 0」是本项目反复踩的那类 bug——加新指标时最容易复发。
    """
    out = _run(monkeypatch, _gold_file(tmp_path), with_answers=False)
    s = out["summary"]
    assert s["over_refusal_rate"] is None
    assert s["over_refusal_gold_rate"] is None
    assert s["contains_acc"] is None
    assert s["contains_acc_subseq"] is None
    assert s["hit_at_5"] == 1.0  # 检索指标照常算


def test_subsequence_track_matches_the_strict_one_on_this_arm(run):
    """本轮全量两路同值（0.8438 == 0.8438）：这条轨现在不提供额外信息，得能看出来。"""
    s = run["summary"]
    assert (
        s["contains_acc"] == s["contains_acc_subseq"] == pytest.approx(2 / 3, abs=1e-4)
    )
    rows = {r["id"]: r for r in run["items"]}
    assert rows["q003"]["answered_ok"] is False  # 拒答：两路都判否
    assert rows["q003"]["answered_ok_subseq"] is False


def test_subsequence_matching_has_no_false_positive_bound():
    """「通过决议」能命中一句毫不相干的话——它就是有序的字符子序列，不是同义容忍。

    原名 `_contains_loose` 听起来像「容忍改写」，实测更像「只要字按序出现就算答对」。
    改名之后这层语义在报告里可见，收紧与否才是一个能被讨论的决定。
    """
    unrelated = "这条街名通达不过是个巧合，决胜球还没踢完，会议明天开。"

    assert "通过决议" not in unrelated  # 严格口径挡住
    assert runner._contains_as_subsequence("通过决议", unrelated)  # 子序列放行
    # 有序但无上界：换个字序就不命中，说明它唯一的作用是放宽距离
    assert not runner._contains_as_subsequence("决议通过", unrelated)


def test_refusal_acc_is_undefined_not_zero_without_answers(monkeypatch, tmp_path):
    """`--retrieval-only` 下拒答正确率必须是 None，不是 0.0。

    拒答题压根没生成答案时 `answered_ok` 全是 None，拿 `len(refusables)` 当分母
    就把「没测」印成「测了且全错」——同一类错误本文件已经禁过两次（no_answer 不给
    覆盖率编 0.0、over_refusal 的 None 不参与）。这条是 2026-09-21 跑全量
    retrieval-only 时从输出里看出来的（它当时印 0.0）。
    """
    s = _run(monkeypatch, _gold_file(tmp_path), with_answers=False)["summary"]
    assert s["refusal_acc"] is None
    # 反证：带答案跑时同一条必须给出真数，否则上面那个 None 只是「压根没算」
    assert _run(monkeypatch, _gold_file(tmp_path))["summary"]["refusal_acc"] == 1.0

"""聚合题分档判据（keypoint_hit_ratio）的回归护栏。

动机是量出来的一件事：gold v2 的 20 条聚合题每条只有 1 个 `must_contain`（那个人名），
而答案集 10~56 篇。于是「上下文 6 块答出 2 篇」与「25 块答出 18 篇」在答案轨同分，
上下文预算消融（E2）跑完读不出任何差别。四条护栏各挡一种会悄悄失效的地方：

1. **分档必须真的分档**：同一道聚合题，多答出几篇要点，旧口径不变、新口径要动。
   这条如果哪天变红，说明新指标退化成了旧指标的同义反复。
2. **没有要点时判 None 而不是 0**：旧黄金集、非聚合题都不该进这个指标的分母——
   把「未定义」印成「测出来是 0」是本项目明令禁止的那类错误。
3. **判据要挑得出有信息量的句子**：全库唯一的发言人点名册行通得过唯一性判据，
   却什么都不主张。
4. **@ 装饰不能吃掉命中**：文档侧写 `@康少云制作了方案`，答案侧转述成
   「康少云制作了方案」，两侧必须按同一口径归一化。
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from doc_rag.eval import compare, runner
from doc_rag.eval.goldgen import _KeyPointIndex, _kp_candidates, _kp_says_something
from doc_rag.eval.schema import KeyPoint, kp_normalize
from doc_rag.retrieve.hybrid import RetrievalOutcome

_A = "康少云负责薪酬制度的初步方案"
_B = "与康少云研究开源协议问题"
_C = "康少云跟进机房巡检排班"
_D = "康少云提交采购审批调整建议"

# 检索清单里必须有这些 doc_id，否则覆盖率/gold 那一套会判 None、条目会被逐条过滤掉
_KP_GOLD = {
    "id": "q100",
    "type": "cross_doc",
    "question": "关于「康少云」，公司文档里出现过哪些讨论或安排？",
    "expected_answer": "散见于多篇文档",
    "must_contain": ["康少云"],
    "source_doc_ids": ["d1", "d2", "d3", "d4"],
    "key_points": [
        {"doc_id": "d1", "phrase": "康少云负责薪酬制度的初步方案"},
        {"doc_id": "d2", "phrase": "与康少云研究开源协议问题"},
        {"doc_id": "d3", "phrase": "康少云跟进机房巡检排班"},
        {"doc_id": "d4", "phrase": "康少云提交采购审批调整建议"},
    ],
}

_ONE_POINT = "关于康少云的记录 [1]：康少云负责薪酬制度的初步方案。"
_ALL_POINTS = f"关于康少云的记录 [1][2][3][4]：{_A}；{_B}；{_C}；{_D}。"


def _chunk(doc_id: str, text: str) -> dict:
    return {
        "doc_id": doc_id,
        "title": f"文档{doc_id}",
        "page": 1,
        "text": text,
        "block_type": "paragraph",
    }


def _gold_file(tmp_path, items):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")
    return path


def _run(monkeypatch, gold, answer, n_contexts=4):
    chunks = [
        _chunk("d1", _A),
        _chunk("d2", _B),
        _chunk("d3", _C),
        _chunk("d4", _D),
    ] + [_chunk(f"o{i}", "无关正文") for i in range(4)]

    retriever = Mock(collection="c", cfg={})
    retriever.retrieve.return_value = RetrievalOutcome(chunks=chunks)
    syn = Mock()
    syn.answer.side_effect = lambda q, *a, **k: answer
    syn.last_meta = None
    monkeypatch.setattr(runner, "_build_retriever", Mock(return_value=(retriever, syn)))
    out = runner.evaluate(
        gold,
        cfg={"retrieval": {"max_contexts": n_contexts}, "llm": {"model": "m"}},
        use_rewrite=False,
        use_rerank=False,
    )
    out["_n_contexts"] = n_contexts
    return out


@pytest.fixture
def arm_one_point(tmp_path, monkeypatch):
    gold = _gold_file(tmp_path, [_KP_GOLD])
    return _run(monkeypatch, gold, _ONE_POINT)


@pytest.fixture
def arm_all_points(tmp_path, monkeypatch):
    gold = _gold_file(tmp_path, [_KP_GOLD])
    return _run(monkeypatch, gold, _ALL_POINTS)


def test_graded_metric_reads_what_contains_acc_cannot(arm_one_point, arm_all_points):
    """同一道聚合题答出 1 篇 vs 4 篇：旧口径同分，新口径必须分得开。"""
    strict = [r["summary"]["contains_acc"] for r in (arm_one_point, arm_all_points)]
    graded = [
        r["summary"]["keypoint_hit_mean"] for r in (arm_one_point, arm_all_points)
    ]
    assert strict[0] == strict[1] == 1.0, "contains_acc 只查那 1 个人名，两臂必然同分"
    assert graded == [0.25, 1.0], f"分档读数丢了：{graded}"
    # 分母必须逐条落盘，否则两臂的 K 是不是同一个只能靠回忆
    assert arm_one_point["items"][0]["n_key_points"] == 4
    assert arm_one_point["summary"]["keypoint_n_items"] == 1
    assert arm_one_point["summary"]["keypoint_k"] == {"min": 4, "max": 4, "mean": 4.0}


def test_answer_grade_is_display_only_and_thresholds_hold(
    arm_one_point, arm_all_points
):
    """≥80% 记 full、≥1 记 half、0 记 zero（阈值未校准，只作展示不进门禁）。"""
    assert arm_all_points["items"][0]["answer_grade"] == "full"
    assert arm_one_point["items"][0]["answer_grade"] == "half"
    assert arm_all_points["summary"]["answer_grade_counts"]["full"] == 1
    assert arm_one_point["summary"]["answer_grade_counts"]["half"] == 1


def test_items_without_key_points_stay_undefined_not_zero(tmp_path, monkeypatch):
    """旧黄金集（没有 key_points）：新指标全 None，且既有指标逐字不动。"""
    legacy = {k: v for k, v in _KP_GOLD.items() if k != "key_points"}
    gold = _gold_file(tmp_path, [legacy])
    out = _run(monkeypatch, gold, _ONE_POINT)
    row = out["items"][0]
    assert row["keypoint_hit"] is None and row["answer_grade"] is None
    assert out["summary"]["keypoint_hit_mean"] is None
    assert out["summary"]["keypoint_n_items"] == 0
    # 「未定义」不能顺着均值函数变成 0.0
    assert out["summary"].get("keypoint_hit_by_type") == {}
    assert row["answered_ok"] is True and row["doc_coverage"] == 1.0


def test_roster_line_is_not_a_key_point():
    """全库唯一的点名册行没有信息量：唯一性它通得过，这条判据挡它。"""
    assert not _kp_says_something("__@陈凡@康少云@黄日航@张果@田锃__")
    assert _kp_says_something("@康少云制作了薪酬制度的初步方案")
    # 判据是「摘掉 @姓名 后还剩不剩话」，不是数 @ 的个数：三个名字也可以说真事
    assert _kp_says_something("@康少云@何李健@田锃三人负责机房巡检排班")
    # 候选句层面同样不能放行（实测这类句子在语料里既含实体又够长）
    text = (
        "议题讨论\n__@陈凡@康少云@黄日航@张果@田锃__\n@康少云制作了薪酬制度的初步方案"
    )
    cands = _kp_candidates(text, "康少云")
    # 候选保留 `@`（它是文档原文的一部分），`@`/`_` 只在匹配时按 `kp_normalize` 去掉
    assert "@康少云制作了薪酬制度的初步方案" in cands
    assert "__@陈凡@康少云@黄日航@张果@田锃__" not in cands


def test_phrase_must_be_unique_across_the_corpus():
    """同一句出现在两篇里时不能当要点：命中一次会同时算答出两篇。"""
    shared = "康少云负责薪酬制度的初步方案"
    docs = [
        {"doc_id": "d1", "text": f"其他内容 {shared} 结尾"},
        {"doc_id": "d2", "text": f"开头 {shared} 别的"},
        {"doc_id": "d3", "text": "独有句：康少云提交了巡检排班表"},
    ]
    index = _KeyPointIndex(docs)
    assert not index.is_unique(shared)
    assert index.is_unique("康少云提交了巡检排班表")


def test_at_decoration_does_not_eat_a_hit():
    """文档侧带 @ / markdown 转义、答案侧转述不带：两侧同一口径，必须算命中。

    `\\` 与 `*` 不是假想敌：E2 首轮臂 A 上限内的 7 条未命中里有 3 条只是装饰字符不同，
    把它们从匹配口径剔掉后实测救回 A 臂 2 条、25 块臂 5 条（`__…__` 强调、`2\\.3`
    与 `\\-` 的 markdown 转义被当成了判据正文）。
    """
    kp = KeyPoint(doc_id="d1", phrase="@康少云制作了__薪酬制度的初步方案")
    assert kp.hit_in("会议记录里康少云制作了薪酬制度的初步方案 [1]。")
    assert not kp.hit_in("康少云没有参与薪酬制度的制定。")
    esc = KeyPoint(doc_id="d2", phrase="2\\.3@张果点评大家近期的情况并给出指导")
    assert esc.hit_in("2.3 张果点评大家近期的情况并给出指导 [2]。")


def _write_result(tmp_path, name, metric_rows):
    p = tmp_path / name
    p.write_text(
        json.dumps({"meta": {}, "items": metric_rows}, ensure_ascii=False),
        encoding="utf-8",
    )
    return p


def _rows(hits, n_ctx):
    return [
        {
            "id": f"q{i:03d}",
            "type": "cross_doc",
            "doc_coverage": 0.5,
            "doc_coverage_ceiling": 0.6,
            "first_hit_rank": 2,
            "ndcg_at_8": 0.4,
            "n_retrieved": 8,
            "n_contexts": n_ctx,
            "n_key_points": 4,
            "keypoint_hit": h,
        }
        for i, h in enumerate(hits)
    ]


def test_compare_puts_keypoints_on_its_own_track_without_length_artifact(tmp_path):
    """块数不等是 E2 的实验变量，不是伪影；这个轨上只有 K 变了才该告警。"""
    a = _write_result(tmp_path, "a.json", _rows([0.25, 0.5], 6))
    b = _write_result(tmp_path, "b.json", _rows([1.0, 0.75], 25))
    report = compare.compare(
        [{"label": "臂A", "files": [a]}, {"label": "臂B", "files": [b]}],
        metric="keypoint_hit_ratio",
    )
    assert report["track"] == "answer"
    # 旧名照样能跑（PLAN/README 里的复现命令不失效），但报告里印的是标准名
    assert report["metric"] == "keypoint_recall"
    paired = report["paired"][0]
    assert paired["mean_diff"] == pytest.approx(0.5)
    assert paired["warnings"] == [], f"块数不等不该在这个轨上报警：{paired['warnings']}"
    assert report["groups"][1]["ctx_len_mean"] == 25  # 长度仍然可见，只是不算伪影
    assert "答案级" in compare.format_report(report)


def test_compare_warns_when_the_denominator_changed(tmp_path):
    """两臂 K 不同 = 分母变了，这才是这个轨上真正该拦下的伪影。"""
    rows_b = _rows([1.0, 0.75], 25)
    for r in rows_b:
        r["n_key_points"] = 8
    a = _write_result(tmp_path, "a.json", _rows([0.25, 0.5], 6))
    b = _write_result(tmp_path, "b.json", rows_b)
    report = compare.compare(
        [{"label": "臂A", "files": [a]}, {"label": "臂B", "files": [b]}],
        metric="keypoint_hit_ratio",
    )
    warns = report["paired"][0]["warnings"]
    assert len(warns) == 1 and "K" in warns[0], warns


def test_old_result_files_without_the_key_degrade_to_silence(tmp_path):
    """改动前的结果文件没有 keypoint_hit：判 None 落进「两臂都无值」，不报错也不假装。"""
    legacy = [
        {k: v for k, v in r.items() if k != "keypoint_hit"} for r in _rows([0.5], 6)
    ]
    a = _write_result(tmp_path, "a.json", legacy)
    loaded = compare.load_scores(a, "keypoint_hit_ratio")
    assert loaded["items"] == {}
    assert loaded["track"] == "answer"


def test_rescore_script_agrees_with_the_runner(tmp_path, monkeypatch):
    """`scripts/rescore_keypoints.py` 是独立入口，分档阈值必须与 runner 同值。

    脚本注释里写了「测试锁它」，这条就是那句注释的兑现：两处常量一旦漂移，
    离线重判出来的数就不再等于合成时算出来的数，而 E2 的三臂全靠离线重判对齐。
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "rescore_keypoints",
        Path(__file__).resolve().parents[1] / "scripts" / "rescore_keypoints.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.GRADE_FULL_RATIO == runner._GRADE_FULL_RATIO

    gold = _gold_file(tmp_path, [_KP_GOLD])
    evaluated = _run(monkeypatch, gold, _ALL_POINTS)
    src = tmp_path / "results_x.json"
    src.write_text(json.dumps(evaluated, ensure_ascii=False), encoding="utf-8")
    # 先用「已被 runner 判过一遍」的同一份答案重判，结果必须逐字相同（幂等）
    mod.rescore(str(src), str(gold))
    out = json.loads((tmp_path / "results_x_kc.json").read_text(encoding="utf-8"))
    assert out["items"][0]["keypoint_hit"] == evaluated["items"][0]["keypoint_hit"]
    assert (
        out["summary"]["keypoint_hit_mean"] == evaluated["summary"]["keypoint_hit_mean"]
    )


def test_markdown_escape_is_stripped_on_both_sides():
    """判据里留着源文档的 `\\.`，答案侧就不许因此漏判——E2 冤案的回归护栏。"""
    assert kp_normalize("2\\.3@张果点评大家近期的情况") == kp_normalize(
        "2.3 张果点评大家近期的情况"
    )


def test_default_sweep_survives_files_lacking_the_new_metric(tmp_path):
    """新指标进了默认指标族，老结果文件上它必须被标成「未测」而不是把整条命令炸掉。

    「静默少一个指标」和「测了但没差异」是两件事：前者会让人以为族里那格被检查过。
    """
    rows = _rows([0.5, 0.5], 6)
    for r in rows:
        r.pop("keypoint_hit")
    a = _write_result(tmp_path, "a.json", rows)
    b = _write_result(tmp_path, "b.json", rows)
    reports = compare.compare_retrieval(
        [{"label": "臂A", "files": [a]}, {"label": "臂B", "files": [b]}]
    )
    by_metric = {r["metric"]: r for r in reports}
    assert by_metric["keypoint_recall"]["unscored"] is True
    assert "未参与判读" in compare.format_report(by_metric["keypoint_recall"])
    # 2026-09-21 新加的标准读数在同一批旧文件上同样是「未测」而不是 0：
    # 它们要的是逐条 ranked 清单，而旧文件没落盘过这个字段。
    for m in ("recall_at_5", "precision_at_5", "ndcg_at_5", "map_at_5"):
        assert by_metric[m]["unscored"] is True, m
    # 其余指标照旧判读，且不因为多了空指标而被拉进 Holm 家族
    assert by_metric["hit_at_5"]["unscored"] is False
    family = by_metric["hit_at_5"]["correction"]["family_size"]
    assert family == sum(1 for r in reports if not r["unscored"])


def test_answered_ok_metric_pairs_and_skips_undefined(tmp_path):
    """`--metric answered_ok` 是答案级配对判据：None（拒答题/无判据）不进分母。

    名字故意不叫 contains_acc —— 后者的分母是 `scorable`（64 条），这里的最大集合
    是所有 answered_ok 非 None 的条目（含拒答题，72 条）。借名字就是把两个分母混成一个。
    """
    rows = [
        {"id": "q001", "type": "fact", "answered_ok": True, "n_contexts": 6},
        {"id": "q002", "type": "fact", "answered_ok": False, "n_contexts": 6},
        {"id": "q003", "type": "no_answer", "answered_ok": None, "n_contexts": 6},
    ]
    a = _write_result(tmp_path, "a.json", [dict(r) for r in rows])
    loaded = compare.load_scores(a, "answered_ok")
    assert loaded["track"] == "answer"
    assert loaded["items"] == {"q001": 1.0, "q002": 0.0}
    b = _write_result(tmp_path, "b.json", [dict(r, answered_ok=True) for r in rows])
    report = compare.compare(
        [{"label": "A", "files": [a]}, {"label": "B", "files": [b]}],
        metric="answered_ok",
    )
    assert report["paired"][0]["n_paired"] == 2
    assert report["paired"][0]["mean_diff"] == pytest.approx(0.5)


def test_shipped_gold_only_gained_key_points():
    """随仓库那份 gold.json：只有聚合题被加了要点，题目本身一条都没变。"""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "eval" / "gold.json"
    if not path.exists():
        pytest.skip("本机没有黄金集文件")
    data = json.loads(path.read_text(encoding="utf-8"))
    agg = [i for i in data["items"] if i["type"] in ("cross_doc", "time_filter")]
    assert len(agg) == 20
    with_points = [i for i in agg if i.get("key_points")]
    # K=0 的那几条是语料属性（该人名的出现几乎都是纯 @ 提及），不是构造失败
    assert len(with_points) >= 15, f"要点覆盖率过低：{len(with_points)}/20"
    for i in with_points:
        assert len(i["key_points"]) <= 8
        assert {p["doc_id"] for p in i["key_points"]} <= set(i["source_doc_ids"])
        for p in i["key_points"]:
            # 每条要点都必须「摘掉名字之后还剩 ≥6 个字」——点名册行过不了这关
            assert _kp_says_something(p["phrase"]), p["phrase"]
            assert KeyPoint(doc_id=p["doc_id"], phrase=p["phrase"]).hit_in(
                p["phrase"]
            ), "要点必须能命中它自己（逐字可核对的下界）"

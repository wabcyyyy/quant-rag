"""检索变体的备料测试：全离线，用合成的假数据（不碰 RGB 真内容）。

这一层护的是**「让系统自己检索」这件事本身**：语料要真的去重、gold 的
`source_doc_ids` 必须与库里的 doc_id 一一对应、`must_contain` 必须与 RGB 的
`answer` 同形（否则 `strict_keyword_accuracy` 就不再等价于 RGB 的 `all_rate`，
两行数字就不能对比了）。
"""

from __future__ import annotations

import json

from doc_rag.benchmarks import rgb, rgb_retrieval


def _zh_raw(rid: str = "0", n_pos: int = 2, n_neg: int = 3) -> dict:
    return {
        "id": rid,
        "query": f"问题{rid}",
        "answer": ["42万元"],
        "positive": [f"正面{rid}-{i}" for i in range(n_pos)],
        "negative": [f"噪声{rid}-{i}" for i in range(n_neg)],
    }


def _int_raw(rid: str = "0") -> dict:
    return {
        "id": rid,
        "query": f"两问{rid}",
        "answer": ["3月1日", "5月1日"],
        "positive": [
            [f"甲{rid}-{i}" for i in range(2)],
            [f"乙{rid}-{i}" for i in range(2)],
        ],
        "negative": [f"噪声{rid}-{i}" for i in range(2)],
    }


def _rec(raw: dict) -> rgb.Record:
    return rgb.Record(
        id=str(raw["id"]), query=raw["query"], answer=raw["answer"], raw=raw
    )


# ── 语料 ────────────────────────────────────────────────────────────────────


def test_passage_doc_id_is_content_addressed_and_stable():
    assert rgb_retrieval.passage_doc_id("同一段文字") == rgb_retrieval.passage_doc_id(
        "同一段文字"
    )
    assert rgb_retrieval.passage_doc_id("甲") != rgb_retrieval.passage_doc_id("乙")
    assert len(rgb_retrieval.passage_doc_id("x")) == 16


def test_corpus_dedups_identical_passages_across_questions():
    """同一段文字在不同题里既是正面又是噪声时只能有一篇——否则 gold 的
    `source_doc_ids` 会与库里的点对不上（库里只有一份）。"""
    a = _zh_raw("0")
    b = _zh_raw("1")
    b["negative"] = [a["positive"][0], "另一个噪声"]  # 复用 a 的正面文档
    corpus = rgb_retrieval.build_corpus([_rec(a), _rec(b)])
    expected = (
        {rgb_retrieval.passage_doc_id(t) for t in a["positive"]}
        | {rgb_retrieval.passage_doc_id(t) for t in a["negative"]}
        | {rgb_retrieval.passage_doc_id(t) for t in b["positive"]}
        | {rgb_retrieval.passage_doc_id("另一个噪声")}
    )
    assert set(corpus) == expected


def test_corpus_includes_both_groups_of_an_integration_question():
    """zh_int 的 positive 是「每组一个列表」，两组文档都必须在语料里。"""
    raw = _int_raw()
    corpus = rgb_retrieval.build_corpus([_rec(raw)])
    for group in raw["positive"]:
        for d in group:
            assert rgb_retrieval.passage_doc_id(d) in corpus


def test_write_parsed_emits_ingest_shaped_json(tmp_path):
    """中间 JSON 必须是 ingest 认的形状：meta.doc_id + blocks。"""
    corpus = {"a" * 16: "某段正文"}
    assert rgb_retrieval.write_parsed(corpus, tmp_path) == 1
    payload = json.loads((tmp_path / f"{'a' * 16}.json").read_text(encoding="utf-8"))
    assert payload["meta"]["doc_id"] == "a" * 16
    assert payload["meta"]["source_type"] == "rgb"
    assert payload["blocks"] == [{"type": "paragraph", "text": "某段正文"}]


def test_write_parsed_never_names_a_file_profile(tmp_path):
    """`profile.json` 会被 indexer 显式排除（`_META_EXCLUDE`），命名不能撞上。"""
    rgb_retrieval.write_parsed({"b" * 16: "正文"}, tmp_path)
    assert not (tmp_path / "profile.json").exists()


# ── 黄金集 ──────────────────────────────────────────────────────────────────


def test_gold_must_contain_mirrors_rgb_answer():
    """`must_contain` 必须逐项等于 RGB 的 `answer`——两侧判据同规则才可直接对比。"""
    raw = _zh_raw()
    gold = rgb_retrieval.build_gold([_rec(raw)], "zh")
    item = gold["items"][0]
    assert item["must_contain"] == ["42万元"]
    assert item["type"] == "fact"
    assert item["refusable"] is False
    assert gold["meta"]["source"]["commit"] == rgb.UPSTREAM_COMMIT


def test_gold_integration_needs_both_subanswers():
    """信息整合题两个子答案都要进 must_contain（缺一不可的规则由 runner 保证）。"""
    gold = rgb_retrieval.build_gold([_rec(_int_raw())], "zh_int")
    item = gold["items"][0]
    assert item["must_contain"] == ["3月1日", "5月1日"]
    assert item["type"] == "cross_doc"


def test_gold_source_doc_ids_point_at_the_corpus():
    """gold 的来源集合必须与语料用同一个 doc_id（否则检索指标的分母是错的）。"""
    raw = _int_raw()
    rec = _rec(raw)
    corpus = rgb_retrieval.build_corpus([rec])
    gold = rgb_retrieval.build_gold([rec], "zh_int")
    expected = {
        rgb_retrieval.passage_doc_id(d) for group in raw["positive"] for d in group
    }
    assert set(gold["items"][0]["source_doc_ids"]) == expected
    assert expected <= set(corpus), "来源文档必须在库里"


def test_gold_limit_truncates_for_rehearsals():
    gold = rgb_retrieval.build_gold(
        [_rec(_zh_raw("0")), _rec(_zh_raw("1"))], "zh", limit=1
    )
    assert gold["meta"]["count"] == 1
    assert [i["id"] for i in gold["items"]] == ["0"]


def test_collection_names_are_per_dataset():
    assert len(set(rgb_retrieval.COLLECTION.values())) == len(rgb_retrieval.COLLECTION)
    assert all(ds in rgb.PROTOCOL for ds in rgb_retrieval.COLLECTION), (
        "语料名表必须覆盖真实数据集"
    )

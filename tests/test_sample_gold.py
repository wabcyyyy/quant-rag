"""公开黄金集（gold_core / gold_full / gold_core_agg）的消费侧护栏。

生成脚本（scripts/make_gold_from_corpus.py）自带解析往返自检（要解析 PDF，
约 40 秒，不搬进 pytest）；这里钉住的是 eval 消费端的契约：
1. 条目能被 eval.schema.GoldItem 校验（eval --gold 直接可消费）；
2. source_doc_ids / key_points.doc_id 与语料 manifest 一致（在库里真实存在）；
3. 聚合题 key_points 的 K 分布（1 ≤ K ≤ 8，绑死的 doc_id 属于来源集合）；
4. 题型分布下限（cross_doc / time_filter 各 ≥15）与 id 唯一；
5. gold_core_agg 恰为 gold_core 的聚合子集。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_rag.eval.schema import GoldItem

MANIFEST = ROOT / "data" / "eval" / "sample_corpus_manifest.json"


def _load(name: str) -> dict:
    return json.loads((ROOT / "data" / "eval" / name).read_text("utf-8"))


def _check_set(
    payload: dict, manifest_doc_ids: set[str], floor: dict[str, int]
) -> None:
    items = payload["items"]
    assert payload["meta"]["count"] == len(items)
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), "id 重复"

    dist: dict[str, int] = {}
    for raw in items:
        item = GoldItem.model_validate(raw)  # eval 消费端契约
        dist[item.type] = dist.get(item.type, 0) + 1
        unknown = set(item.source_doc_ids) - manifest_doc_ids
        assert not unknown, f"{item.id}: source_doc_ids 不在语料里 {unknown}"
        if item.type == "no_answer":
            assert item.refusable and not item.source_doc_ids
        else:
            assert item.must_contain, f"{item.id}: 非拒答题缺判据"
        if item.key_points:
            assert 1 <= len(item.key_points) <= 8, (
                f"{item.id}: K={len(item.key_points)} 越界"
            )
            for kp in item.key_points:
                assert kp.doc_id in set(item.source_doc_ids), (
                    f"{item.id}: keypoint doc_id 不在来源集合里"
                )
                assert kp.phrase.strip(), f"{item.id}: keypoint 空短语"
    for t, n in floor.items():
        assert dist.get(t, 0) >= n, f"{t} 只有 {dist.get(t, 0)} 条（要求 ≥{n}）"
    assert dist == payload["meta"]["type_distribution"], "meta 分布与逐条统计不一致"


def test_gold_core_and_full_contract() -> None:
    if not MANIFEST.exists():
        raise AssertionError("manifest 缺失：先跑 make_sample_corpus.py --scale 300")
    manifest_doc_ids = {
        d["doc_id"] for d in json.loads(MANIFEST.read_text("utf-8"))["docs"]
    }
    floor = {"cross_doc": 15, "time_filter": 15}
    core = _load("gold_core.json")
    full = _load("gold_full.json")
    assert 60 <= len(core["items"]) <= 80, f"核心集 {len(core['items'])} 条越界"
    assert len(full["items"]) >= 150, f"扩展集 {len(full['items'])} 条过少"
    _check_set(core, manifest_doc_ids, floor)
    _check_set(full, manifest_doc_ids, floor)


def test_gold_core_agg_is_core_subset() -> None:
    core = {i["id"]: i for i in _load("gold_core.json")["items"]}
    agg = _load("gold_core_agg.json")["items"]
    assert agg, "聚合子集为空"
    for item in agg:
        assert core[item["id"]]["question"] == item["question"]
        assert item["type"] in ("cross_doc", "time_filter")
    assert len(agg) == sum(
        1 for i in core.values() if i["type"] in ("cross_doc", "time_filter")
    )

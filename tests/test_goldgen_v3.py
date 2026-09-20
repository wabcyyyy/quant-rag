"""v3 窗口依赖题的生成与性质复检（合成语料，零 LLM、零 Qdrant）。

这批题的全部价值在一个性质上：**值只在 B 块、说的是哪件事只在 A 块**——只装单块的清单
按构造答不全它，而 `read_window` 补的正是邻居块。所以测试两头都要锁：
① 满足性质的题要出得来；② 四种「看起来像但其实不满足」的候选必须被剔掉，
③ 性质复检函数要真能报错（恒真的复检比没有更糟）。
"""

from __future__ import annotations

import json
from pathlib import Path

from doc_rag.eval.goldgen import generate_v3, v3_property_violations
from doc_rag.eval.schema import GoldItem

LONG = (
    "供应商预付款方案的背景说明：本方案由秘书处起草，讨论了付款节奏、发票类型与验收节点，"
    "并比较了三种安排方式的资金占用差异。与会人认为一次性付清会挤占当月现金流，"
    "分两期则要与合同签订时点对齐；财务侧提出需要留出验收款的尾期。"
) * 3

TABLE = """| 付款比例 | 交付节点 |
| 30%：11130 | 签订合同 |
| 40%：14840 | 前三门课程教研资料交付 |
"""


def _doc(doc_id: str, title: str, blocks: list[dict]) -> dict:
    return {
        "meta": {"source_type": "pdf", "doc_id": doc_id, "title": title},
        "blocks": blocks,
    }


def _para(text: str) -> dict:
    return {"type": "paragraph", "text": text, "page": 1}


def _table(text: str) -> dict:
    return {"type": "table", "text": text, "page": 1}


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "parsed"
    root.mkdir()
    docs = [
        # ① 正例：主语在正文块、带单位的数值只在表格块（表格块永远单独成块）
        _doc(
            "good1", "工作档案_SSC职能_供应商预付款方案", [_para(LONG), _table(TABLE)]
        ),
        # ①b 形状对但句子里没有决议线索词：值只是排期里的「2-3 天」→ 为质量丢掉
        _doc(
            "nocue",
            "工作档案_SSC职能_实习带教安排",
            [
                _para(
                    "实习带教安排的讨论记录：导师人选、带教内容与验收方式都过了两轮。"
                    * 20
                ),
                _table("| 项目 | 频次 |\n| 每 2-3 天 | 简短同步 |\n"),
            ],
        ),
        # ② 主语同时出现在 B 块 → 单块就能答，必须剔除
        _doc(
            "subinb",
            "工作档案_SSC职能_设备租赁方案",
            [
                _para("设备租赁方案的讨论记录" * 40),
                _table("| 项目 | 金额 |\n| 设备租赁方案 30%：900 | 首付 |"),
            ],
        ),
        # ③ 主语只有模板词（会议纪要N）→ 出的题不指认任何东西，必须剔除
        _doc(
            "template",
            "会议档案_周会_会议纪要3",
            [
                _para(
                    "会议纪要3 的正文：讨论了供应商预付款方案，付款比例见下表。" * 12
                ),
                _table(TABLE),
            ],
        ),
        # ④ 值在 A 块里也有 → 不是「各记一半」，必须剔除
        _doc(
            "valueina",
            "工作档案_SSC职能_差旅报销方案",
            [_para(LONG + " 其中 30%：11130 已在上面给出"), _table(TABLE)],
        ),
    ]
    for doc in docs:
        (root / f"{doc['meta']['doc_id']}.json").write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8"
        )
    return root


def test_only_quality_candidates_become_items(tmp_path):
    root = _corpus(tmp_path)
    meta = generate_v3(root, tmp_path / "gold_v3.json", limit=12)
    assert meta["count"] == 1, meta
    assert meta["candidates_total"] == 2  # good1 + nocue（②③④ 在成题前就被判据排除）
    assert meta["candidates_with_cue"] == 1
    assert meta["dropped_for_quality"] == 1
    assert meta["bar_met"] is False  # 合成语料只有 1 条，判据 ≥8 不成立要如实写出来
    items = json.loads((tmp_path / "gold_v3.json").read_text(encoding="utf-8"))["items"]
    item = items[0]
    assert item["type"] == "window"
    assert item["must_contain"] == ["30%"]  # 带单位的那一段才算「值」
    # 题面用的是 A 块里的主语，且没把值漏进问题里
    assert "供应商预付款方案" in item["question"]
    assert "30%" not in item["question"]
    assert item["required_chunk_ids"][1].endswith(":2")
    assert meta["property_violations"] == []


def test_rejects_subject_only_in_b(tmp_path):
    root = _corpus(tmp_path)
    ids = _item_doc_ids(root)
    assert "subinb" not in ids  # B 自带主语 → 单块可答，不算窗口依赖


def test_rejects_template_subject(tmp_path):
    root = _corpus(tmp_path)
    assert "template" not in _item_doc_ids(root)


def test_rejects_value_present_in_a(tmp_path):
    root = _corpus(tmp_path)
    assert "valueina" not in _item_doc_ids(root)


def _item_doc_ids(root: Path) -> set[str]:
    """跑一遍生成，返回出题的 doc_id 集合（负例用例的判据都在这上面）。"""
    meta = generate_v3(root, root.parent / "out.json", limit=20)
    items = json.loads((root.parent / "out.json").read_text(encoding="utf-8"))["items"]
    ids = {i["source_doc_ids"][0] for i in items}
    # 防空跑：正例必须出得来，否则「某负例不在集合里」这个断言毫无意义
    assert "good1" in ids, meta
    return ids


def test_property_check_catches_a_stale_item(tmp_path):
    """分块策略一变，这批题就失去性质——复检必须报出来，不能继续给平均分。"""
    root = _corpus(tmp_path)
    good = GoldItem(
        id="w001",
        type="window",
        question="关于「供应商预付款方案」，最后定下来的具体数字是多少？",
        expected_answer="30%：11130",
        must_contain=["30%：11130"],
        source_doc_ids=["good1"],
        required_chunk_ids=["good1:1", "good1:2"],
    )
    assert v3_property_violations(root, [good]) == []

    stale = good.model_copy(update={"required_chunk_ids": ["good1:1", "good1:9"]})
    assert any("已经不在当前分块" in v for v in v3_property_violations(root, [stale]))

    wrong_value = good.model_copy(update={"must_contain": ["77%：1"]})
    assert any("值不在 B 块" in v for v in v3_property_violations(root, [wrong_value]))

    leaked = good.model_copy(update={"must_contain": ["30%：11130", "现金流"]})
    # 「现金流」在 A 块正文里 → 它不是「只在 B 的那一半」，同样要报
    assert any("不再是各记一半" in v for v in v3_property_violations(root, [leaked]))


def test_non_window_items_are_not_checked(tmp_path):
    root = _corpus(tmp_path)
    fact = GoldItem(id="q1", type="fact", question="x", expected_answer="y")
    assert v3_property_violations(root, [fact]) == []

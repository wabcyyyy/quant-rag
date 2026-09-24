"""RGB 判据的离线测试：全部用**合成的假数据**，形状照官方字段、内容不是数据集内容。

为什么不用真数据做夹具：RGB 是 CC BY-NC-SA 4.0（非商用），把它的数据副本放进
仓库等于再分发。夹具只需要字段形状对，就够了——真实内容由
`doc-rag bench-rgb-fetch` 在本地抓（不入库）。

这里钉的主要是**复刻是否忠实**：官方那几条行为很容易被"顺手改好"而破坏可比性，
所以要显式断言（包括一条官方已知的假阳性）。
"""

from __future__ import annotations

import json

import pytest

from doc_rag.benchmarks import rgb
from doc_rag.generate import llm as llm_mod


def _zh_record(rid: str = "t1") -> dict:
    return {
        "id": rid,
        "query": "某次会议定下的预算是多少",
        "answer": ["42万元"],
        "positive": [f"正面文档{i}" for i in range(6)],
        "negative": [f"噪声文档{i}" for i in range(8)],
    }


def _int_record(rid: str = "t2") -> dict:
    return {
        "id": rid,
        "query": "两次会议分别定在什么时候",
        "answer": ["3月1日", "5月1日"],
        "asnwer1": ["3月1日"],  # 官方拼写如此，夹具照抄以免掩盖真实字段
        "answer2": ["5月1日"],
        "positive": [
            [f"甲组文档{i}" for i in range(3)],
            [f"乙组文档{i}" for i in range(2)],
        ],
        "negative": [f"噪声文档{i}" for i in range(8)],
    }


def _fact_record(rid: str = "t3") -> dict:
    return {
        "id": rid,
        "query": "议席有多少个",
        "answer": "70",
        "fakeanswer": "170",
        "positive": ["正确文档0", "正确文档1", "正确文档2"],
        "positive_wrong": ["错误文档0", "错误文档1", "错误文档2"],
        "negative": [f"噪声文档{i}" for i in range(5)],
    }


def _rec(raw: dict) -> rgb.Record:
    return rgb.Record(
        id=str(raw["id"]), query=raw["query"], answer=raw["answer"], raw=raw
    )


def _row(
    rid: str,
    prediction: str,
    answer,
    dataset: str,
    noise_rate: float,
) -> dict:
    labels, factlabel, _ = rgb.label_and_flags(prediction, answer, dataset)
    return {
        "id": rid,
        "dataset": dataset,  # 真实行必带：记录索引的键是 (数据集, id)
        "prediction": prediction,
        "label": labels,
        "factlabel": factlabel,
        "noise_rate": noise_rate,
    }


# ── 数据加载 ────────────────────────────────────────────────────────────────


def test_load_records_reads_jsonl_not_a_json_object(tmp_path):
    """官方数据文件是一行一条的 JSONL——当成 JSON 对象读会直接解析失败。"""
    p = tmp_path / "zh.json"
    p.write_text(
        "\n".join(json.dumps(_zh_record(f"t{i}"), ensure_ascii=False) for i in range(3))
        + "\n",
        encoding="utf-8",
    )
    recs = rgb.load_records(p)
    assert [r.id for r in recs] == ["t0", "t1", "t2"]


# ── 文档组装（复刻官方 processdata）────────────────────────────────────────


@pytest.mark.parametrize("noise_rate", [0.0, 0.2, 0.4, 0.6, 0.8])
def test_every_rate_gives_exactly_passage_num_docs(noise_rate):
    docs = rgb.assemble_docs(_rec(_zh_record()), "zh", noise_rate)
    assert len(docs) == rgb.PASSAGE_NUM


def test_noise_one_gives_only_negatives():
    """拒答的测试条件就是「一条正面文档都不给」——这是整个 Rej 口径的前提。"""
    raw = _zh_record()
    docs = rgb.assemble_docs(_rec(raw), "zh", rgb.REJECTION_NOISE)
    assert len(docs) == rgb.PASSAGE_NUM
    assert all(d in raw["negative"] for d in docs)


def test_doc_selection_is_reproducible_across_calls():
    """官方每条记录前 `random.seed(2333)`，所以选文档与顺序都可精确复现。"""
    rec = _rec(_zh_record())
    assert rgb.assemble_docs(rec, "zh", 0.6) == rgb.assemble_docs(rec, "zh", 0.6)


def test_integration_docs_come_from_every_group():
    """信息整合题必须同时拿到两组文档，否则「整合」无从谈起。"""
    raw = _int_record()
    docs = rgb.assemble_docs(_rec(raw), "zh_int", 0.0)
    heads = {raw["positive"][0][0], raw["positive"][1][0]}
    assert heads <= set(docs)


def test_counterfactual_uses_the_falsified_documents():
    """反事实题在 noise 0 时**全部**喂 positive_wrong——喂对了就没这个任务了。"""
    raw = _fact_record()
    docs = rgb.assemble_docs(_rec(raw), "zh_fact", 0.0)
    assert docs
    assert all(d in raw["positive_wrong"] for d in docs)
    assert not any(d in raw["positive"] for d in docs)


# ── 判据（复刻官方 checkanswer / predict）──────────────────────────────────


def test_string_ground_truth_must_appear_and_list_is_or():
    assert rgb.check_labels("预算为42万元", ["42万元"]) == [1]
    assert rgb.check_labels("没有提到", ["42万元"]) == [0]
    # 列表元素 = 任一子串命中即算（OR），这是官方语义，别"顺手收紧"
    assert rgb.check_labels("提到了42万元", [["42万元", "9万元"]]) == [1]
    assert rgb.check_labels("两个都没提", [["42万元", "9万元"]]) == [0]


def test_rejection_marker_wins_over_keyword_hit():
    """官方先判拒答标记、再判关键词——两者同时出现时按拒答记（label=[-1]）。"""
    pred = "文档信息不足，因此我无法回答。不过文中出现过42万元"
    labels, _fact, rejected = rgb.label_and_flags(pred, ["42万元"], "zh")
    assert rejected is True
    assert labels == [-1]


def test_zh_space_stripping_applies_before_marker_matching():
    """官方对 zh 先 replace(" ","")：掺了空格的中文答案同样要判得出拒答。

    漏掉这一步会让中文分数系统性偏低（模型的答案里常有全角/半角混排与空格）。
    """
    labels, _fact, rejected = rgb.label_and_flags("信 息 不 足", ["42万元"], "zh")
    assert rejected is True and labels == [-1]
    # 同一句话在 en 口径下不触发（英文标记才管用）
    assert rgb.label_and_flags("信 息 不 足", ["42万元"], "en")[2] is False


def test_fact_marker_sets_factlabel():
    _labels, factlabel, _rej = rgb.label_and_flags(
        "文档存在事实性错误，正确答案是 70", "70", "zh_fact"
    )
    assert factlabel == 1


# ── 记分（复刻官方 __main__ 的内联循环）──────────────────────────────────


def test_official_rejection_bucket_also_counts_a_lucky_correct_answer():
    """官方 `all_rate` 在 noise=1 时走 `if label[0]==-1 … elif 全部命中`：答对也算。

    README 把这一档叫 rejection rate，实现却不是纯拒答率（没有正面文档却答对，
    只可能来自参数记忆）。这里同时钉住官方口径与严格口径，两者之差就是侥幸那部分
    ——报数时两个都要给，否则「拒答率」这个词会被读成它不是的东西。
    """
    rows = [
        _row("a", "文档信息不足，无法回答", ["x"], "zh", 1.0),
        _row("b", "答案是 x", ["x"], "zh", 1.0),
    ]
    assert rgb.accuracy(rows, 1.0) == 1.0  # 官方口径：两条都算
    assert rgb.rejection_rate_strict(rows) == 0.5  # 严格口径：只有 a 真的拒答了


def test_rejection_is_penalised_below_noise_one():
    """非 1 档位下拒答**不算对**——那正是噪声鲁棒性要罚的行为（该答却没答）。"""
    rows = [_row("a", "文档信息不足，无法回答", ["x"], "zh", 0.0)]
    assert rgb.accuracy(rows, 0.0) == 0.0


def test_production_wording_is_counted_by_its_own_marker_set():
    """生产 prompt 的拒答措辞不含官方关键词，所以必须单列一个读数。

    `prompts.SYSTEM_ANSWER` 第 5 条要求回答「根据现有文档无法回答」；官方判据认
    「信息不足」。若只报官方口径，生产行的拒绝率会被读成「系统不拒答」——
    而它其实拒答了，只是换了措辞。这条同时钉住两件事：官方口径确实为 0、
    自己的措辞确实计数。
    """
    rows = [_row("a", "根据现有文档无法回答：缺少预算金额", ["x"], "zh", 1.0)]
    assert rgb.rejection_rate_strict(rows) == 0.0  # 官方关键词认不出来
    assert rgb.rejection_rate_marker(rows, rgb.PRODUCTION_REJECT_MARKERS) == 1.0


def test_integration_needs_both_subanswers():
    """zh_int 的两个子答案缺一不可（官方 `0 not in label`）。"""
    both = _row("a", "分别是3月1日与5月1日", ["3月1日", "5月1日"], "zh_int", 0.0)
    one = _row("b", "只有3月1日", ["3月1日", "5月1日"], "zh_int", 0.0)
    assert rgb.accuracy([both], 0.0) == 1.0
    assert rgb.accuracy([one], 0.0) == 0.0


def test_fact_rates_follow_the_official_definition():
    """ED = 打出标记的占比；CR（官方非星号口径）= 打出标记且标签全中的占比。"""
    rows = [
        _row("a", "文档存在事实性错误，正确答案是 70", "70", "zh_fact", 0.0),
        _row("b", "文档存在事实性错误，但我也没答对", "70", "zh_fact", 0.0),
        _row("c", "70", "70", "zh_fact", 0.0),
    ]
    ed, cr = rgb.fact_rates(rows)
    assert ed == pytest.approx(2 / 3)
    assert cr == pytest.approx(0.5)


def test_official_substring_rule_marks_a_falsified_answer_as_correct():
    """照抄官方判据的**已知假阳性**：'70' ⊂ '170'，所以答了被篡改的 170 也算命中。

    这条断言存在的意义不是认可它，而是把假阳性钉在明面上：换判据就失去与论文的
    可比性，所以这里选择照抄 + 单列计数（`fakeanswer_false_positives`），
    而不是"顺手修好"。
    """
    records = {rgb.record_key("zh_fact", "t3"): _rec(_fact_record())}
    row = _row("t3", "共有170个议席", "70", "zh_fact", 0.0)
    assert rgb.is_correct(row, 0.0) is True
    assert rgb.fakeanswer_false_positives([row], records) == 1


def test_fakeanswer_counter_ignores_rows_that_are_not_correct():
    records = {rgb.record_key("zh_fact", "t3"): _rec(_fact_record())}
    row = _row("t3", "共有70个议席", "70", "zh_fact", 0.0)
    assert rgb.fakeanswer_false_positives([row], records) == 0


def test_record_index_must_be_keyed_by_dataset_too():
    """四个数据集的 id 各自从 0 开始，只按 id 建索引会互相覆盖。

    实测踩过：跨数据集建索引时 `zh` 的记录覆盖了 `zh_fact` 的同 id 记录，
    `fakeanswer` 取不到 → 假阳性被静默算成 0（真值 5）。这条把键的形状钉住。
    """
    from doc_rag.benchmarks import rgb_runner

    zh = _rec(
        {"id": "0", "query": "q", "answer": ["a"], "positive": ["p"], "negative": ["n"]}
    )
    fact = _rec(_fact_record("0"))
    idx = rgb_runner.record_index({"zh": [zh], "zh_fact": [fact]})
    assert set(idx) == {("zh", "0"), ("zh_fact", "0")}
    assert idx[("zh_fact", "0")].fakeanswer == "170"
    assert idx[("zh", "0")].fakeanswer is None


# ── 星号判据（judge）的解析规则 ─────────────────────────────────────────────


def test_judge_reject_reads_the_official_marker(monkeypatch):
    seen: list[str] = []

    def _fake(cfg, user_prompt, system_prompt=None, temperature=None):
        seen.append(user_prompt)
        return "No, the question is not addressed by the documents.", {"ms": 1.0}

    monkeypatch.setattr(llm_mod, "chat_timed", _fake)
    assert rgb.judge_reject("q", "a", {}) is True
    # 判据必须把问题与答案都送进去，否则「文档解不解得了这题」无从判断
    assert "Question: q" in seen[0] and "Answer: a" in seen[0]


def test_judge_fact_reads_the_official_marker(monkeypatch):
    monkeypatch.setattr(
        llm_mod,
        "chat_timed",
        lambda *a, **k: ("NO, the model fail to identify the factual errors.", {}),
    )
    assert rgb.judge_fact("a", {}) is False
    monkeypatch.setattr(
        llm_mod,
        "chat_timed",
        lambda *a, **k: ("Yes, the model has identified the factual errors.", {}),
    )
    assert rgb.judge_fact("a", {}) is True


# ── 协议常量 ────────────────────────────────────────────────────────────────


def test_protocol_covers_the_four_chinese_files():
    """四个文件四个任务，档位照论文中文侧报的那些。"""
    assert set(rgb.PROTOCOL) == {"zh", "zh_refine", "zh_int", "zh_fact"}
    assert rgb.PROTOCOL["zh"] == (0.0, 0.2, 0.4, 0.6, 0.8)
    assert rgb.PROTOCOL["zh_int"] == (0.0, 0.2, 0.4)
    assert rgb.REJECTION_NOISE == 1.0
    # 精修版与 zh 同任务，档位必须一致，否则两者不可比
    assert rgb.PROTOCOL["zh_refine"] == rgb.PROTOCOL["zh"]

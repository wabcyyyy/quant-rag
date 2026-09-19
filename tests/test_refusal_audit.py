"""拒答审计（C1）：现有口径下 8 条拒答题没人看过内容。

锁三件事：① 判定用的是**落盘的那份上下文**（与 LLM 当时看到的逐字一致），不是重新
检索出来的；② 旧格式（只有正文、没有 `[n]（文档名 第p页）` 前缀）必须**跳过而不是照判**
——那正是当年 faithfulness 被低估 32pt 的同一个 bug；③ 结论只报「有没有把上下文外
的内容当事实讲」，不去评措辞。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_rag.eval import refusal

CFG = {"llm": {"model": "judge-m", "api_key": "k", "base_url": "http://x"}}

GOOD_CTX = [
    "[1]（会议纪要_办公会 第2页）\n本次未涉及碳排放制度。",
    "[2]（会议纪要_办公会 第3页）\n下一季度议题待定。",
]


def _row(rid: str, rtype: str, answer: str, contexts: list[str]) -> dict:
    return {
        "id": rid,
        "type": rtype,
        "question": f"{rid} 的问题",
        "answer": answer,
        "contexts": contexts,
        "n_citations": 2,
        "answered_ok": True,
    }


def _results(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {"meta": {"timestamp": "2026-09-19T00:00:00"}, "summary": {}, "items": rows}
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def stub(monkeypatch):
    """替身 judge：记录送进去的 prompt，按队列返回预置判定。"""
    calls: dict = {"prompts": [], "replies": []}

    def fake_chat(cfg, prompt, system_prompt=None, temperature=None):
        calls["prompts"].append(prompt)
        calls["system"] = system_prompt
        calls["temperature"] = temperature
        return calls["replies"].pop(0), {"ms": 421, "cached": False}

    monkeypatch.setattr(refusal, "chat_timed", fake_chat)
    return calls


ROWS = [
    _row("q01", "no_answer", "根据现有文档无法回答。", GOOD_CTX),
    _row(
        "q02",
        "no_answer",
        "无法回答该问题，但公司已于 2026 年 3 月通过每吨 120 元的碳税决议 [1][2]。",
        GOOD_CTX,
    ),
    _row("q03", "fact", "预算 67 元 [1]。", GOOD_CTX),  # 非拒答题：不进审计
]


def test_only_refusable_rows_with_answers_are_audited(tmp_path, stub):
    stub["replies"] = [json.dumps({"fabricated": False})] * 2

    out = refusal.audit_results(_results(tmp_path, ROWS), cfg=CFG)

    assert out["meta"]["n_refusable_with_answer"] == 2
    assert [i["id"] for i in out["items"]] == ["q01", "q02"]


def test_judge_sees_the_recorded_contexts_verbatim(tmp_path, stub):
    stub["replies"] = [json.dumps({"fabricated": False})] * 2

    refusal.audit_results(_results(tmp_path, ROWS), cfg=CFG)

    assert len(stub["prompts"]) == 2
    for prompt in stub["prompts"]:
        for ctx in GOOD_CTX:
            assert ctx in prompt  # 逐字，含 [n]（文档名 第p页）前缀
    assert stub["temperature"] == 0.0
    assert "无法回答" in stub["system"]  # 措辞本身不算编造这条纪律在 system 里


def test_fabricated_answer_is_reported_with_rate_and_quote(tmp_path, stub):
    stub["replies"] = [
        json.dumps({"fabricated": False, "quote": "", "why": ""}),
        json.dumps(
            {
                "fabricated": True,
                "quote": "通过每吨 120 元的碳税决议",
                "why": "上下文只说未涉及碳排放制度",
            }
        ),
    ]

    out = refusal.audit_results(_results(tmp_path, ROWS), cfg=CFG)

    assert out["fabrication_rate"] == 0.5
    bad = out["items"][1]
    assert bad["fabricated"] is True and bad["quote"].startswith("通过每吨")
    assert bad["answer_len"] == len(ROWS[1]["answer"])
    assert out["meta"]["judge_model"] == "judge-m"
    assert out["meta"]["prompt_version"] == refusal.PROMPT_VERSION


def test_legacy_contexts_are_skipped_not_judged(tmp_path, stub):
    """只存正文的旧结果文件：判了就是假数据，必须跳过并计数。"""
    rows = [_row("q09", "no_answer", "无法回答。", ["碳排放制度未被提及"])]
    stub["replies"] = [json.dumps({"fabricated": True})]

    out = refusal.audit_results(_results(tmp_path, rows), cfg=CFG)

    assert out["items"][0]["fabricated"] is None
    assert "前缀" in out["items"][0]["skipped"]
    assert out["fabrication_rate"] is None  # 一条都没判 → 不编一个率出来
    assert stub["prompts"] == []


def test_limit_bounds_the_paid_calls(tmp_path, stub):
    stub["replies"] = [json.dumps({"fabricated": False})]

    out = refusal.audit_results(_results(tmp_path, ROWS), cfg=CFG, limit=1)

    assert len(stub["prompts"]) == 1 and out["meta"]["n_judged"] == 1


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"fabricated": true, "quote": "x"}', True),
        ('判定如下：\n{"fabricated": false}\n以上。', False),
        ("模型没按格式回答", None),
        ('{"fabricated": ', None),
    ],
)
def test_verdict_parsing_tolerates_prose_but_never_guesses(text, expected):
    got = refusal._parse_verdict(text)
    assert got["fabricated"] is expected
    if expected is None:
        assert got["error"]


# ------------------------------------------------------------------ 判读对照


def _audit(model: str, verdicts: dict[str, bool | None], skipped=()) -> dict:
    items = [
        {"id": rid, "fabricated": v}
        for rid, v in verdicts.items()
        if rid not in skipped
    ]
    return {
        "meta": {"judge_model": model, "prompt_version": refusal.PROMPT_VERSION},
        "fabrication_rate": None,
        "items": items,
    }


def test_compare_audits_reports_agreement_and_flips():
    prev = _audit("deepseek-flash", {"q1": False, "q2": False, "q3": True})
    new = _audit("qwen", {"q1": False, "q2": True, "q3": True})
    cmp = refusal.compare_audits(prev, new)
    assert cmp["n_both_judged"] == 3
    assert cmp["agreement"] == pytest.approx(2 / 3, abs=1e-4)
    assert [f["id"] for f in cmp["flips"]] == ["q2"]
    assert cmp["flips"][0]["prev"] is False and cmp["flips"][0]["new"] is True


def test_compare_audits_never_counts_a_skipped_row_as_agreement():
    """一方跳过或判分失败的条目不能悄悄进分母——那会把一致率刷高。"""
    prev = _audit("a", {"q1": False, "q2": False})
    new = _audit("b", {"q1": False, "q2": None, "q3": False})
    cmp = refusal.compare_audits(prev, new)
    assert cmp["n_both_judged"] == 1 and cmp["agreement"] == 1.0
    assert cmp["n_only_prev"] == ["q2"] and cmp["n_only_new"] == ["q3"]

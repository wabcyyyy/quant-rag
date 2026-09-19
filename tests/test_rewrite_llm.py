"""LLM 查询改写的离线测试：判定逻辑、退化计数、过滤字段边界。

真调模型的泛化能力不在这里测——那是 `doc-rag check-rewrite` 与
`data/eval/rewrite_paraphrase.json` 的职责（门禁会花 API 成本，不能进 CI）。
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from doc_rag.generate import llm as llm_mod
from doc_rag.retrieve import rewrite_llm as rw
from doc_rag.retrieve.hybrid import RetrievalOutcome

_CFG = {
    "llm": {"model": "m", "base_url": "http://l", "api_key": "k"},
    "retrieval": {"aggregate_top_n": 25, "max_contexts": 10},
    "rewrite": {"temperature": 0.0, "reasoning_effort": "none"},
}


@pytest.fixture
def reply(monkeypatch):
    """让 chat_timed 返回指定文本，并记录请求参数。"""
    captured: dict = {}

    def _install(text):
        def _timed(llm_cfg, user_prompt, system_prompt=None, temperature=None):
            captured["llm_cfg"] = llm_cfg
            captured["user"] = user_prompt
            captured["system"] = system_prompt
            captured["temperature"] = temperature
            return text, {"ms": 1.0, "cached": False, "model": "m"}

        monkeypatch.setattr(llm_mod, "chat_timed", _timed)
        return captured

    return _install


def test_parse_plain_and_fenced_json():
    assert rw.parse_reply('{"aggregate": true}')["aggregate"] is True
    assert rw.parse_reply('```json\n{"aggregate": true}\n```')["aggregate"] is True
    assert rw.parse_reply('前置废话 {"aggregate": false} 尾巴')["aggregate"] is False


def test_parse_rejects_garbage_rather_than_guessing():
    assert rw.parse_reply("") is None
    assert rw.parse_reply("我觉得这不是聚合题") is None
    assert rw.parse_reply('{"aggregate": "也许"}') is None  # 类型不符也不猜


def test_parse_ignores_unknown_keys():
    """模型多吐字段时按已知契约取用，不能整体失败。"""
    out = rw.parse_reply('{"aggregate": true, "topics": ["预算"], "evil": 1}')
    assert out["aggregate"] is True


def test_aggregate_year_builds_same_filter_shape_as_the_old_rule(reply):
    reply(
        '{"rewritten":"安全演练","aggregate":true,"year":2025,"reason":"限定年份的枚举题"}'
    )
    plan = rw.LLMQueryRewriter(_CFG).rewrite("2025 年全员安全演练有哪些安排？")
    assert plan["rewritten"] == "安全演练"
    assert plan["aggregate"] is True
    # 与改造前规则版逐字同形：`DatetimeRange(**value)` 的解析口径不能悄悄变
    assert plan["filters"] == {
        "doc_date": {"gte": "2025-01-01T00:00:00", "lt": "2026-01-01T00:00:00"}
    }
    assert plan["top_n"] == 25


def test_model_cannot_inject_filter_fields(reply):
    """过滤维度只有 doc_date：模型没有渠道把任意 payload 字段塞进检索条件。"""
    reply(
        '{"rewritten":"x","aggregate":true,"year":2025,'
        '"filters":{"topics":{"gte":"2020-01-01"}},"reason":"r"}'
    )
    plan = rw.LLMQueryRewriter(_CFG).rewrite("2025 年的记录")
    assert set(plan["filters"]) == {"doc_date"}


def test_out_of_range_year_is_dropped(reply):
    reply('{"rewritten":"x","aggregate":true,"year":9999,"reason":"r"}')
    plan = rw.LLMQueryRewriter(_CFG).rewrite("某年的记录")
    assert plan["filters"] is None
    assert plan["aggregate"] is True


def test_single_fact_question_gets_no_filter_and_no_budget_change(reply):
    reply(
        '{"rewritten":"预算是多少","aggregate":false,"year":null,"reason":"单点事实"}'
    )
    plan = rw.LLMQueryRewriter(_CFG).rewrite("客服系统升级的预算是多少？")
    assert plan["aggregate"] is False
    assert plan["filters"] is None
    assert plan["top_n"] is None


def test_year_mentioned_without_aggregate_intent_stays_unfiltered(reply):
    """doc_date 只覆盖 18.8% 语料：对单点题加年份过滤会误伤八成语料。"""
    reply('{"rewritten":"x","aggregate":false,"year":2026,"reason":"单点"}')
    plan = rw.LLMQueryRewriter(_CFG).rewrite("2026 年定的消防验收标准是什么？")
    assert plan["filters"] is None
    assert "doc_date 稀疏" in plan["reason"]


def test_blank_rewritten_falls_back_to_original_question(reply):
    reply('{"rewritten":"   ","aggregate":true,"year":null,"reason":"r"}')
    q = "关于机房巡检的安排"
    assert rw.LLMQueryRewriter(_CFG).rewrite(q)["rewritten"] == q


def test_unparsable_reply_degrades_and_is_counted(reply):
    reply("抱歉，我无法判断")
    rewriter = rw.LLMQueryRewriter(_CFG)
    plan = rewriter.rewrite("随便一个问题")
    assert plan["rewritten"] == "随便一个问题"
    assert plan["aggregate"] is False
    assert rewriter.failures == ["bad_json"]
    assert "已计数" in plan["reason"]


def test_missing_llm_config_degrades_without_calling(reply):
    cfg = {"llm": {"model": "", "api_key": ""}, "retrieval": {}}
    rewriter = rw.LLMQueryRewriter(cfg)
    assert rewriter.rewrite("问题")["aggregate"] is False
    assert rewriter.failures == ["no_llm"]
    assert reply is not None  # 未安装桩 → 真打出去就会炸，说明没被调用


def test_call_failure_is_counted(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("LLM 调用失败：connection reset")

    monkeypatch.setattr(llm_mod, "chat_timed", _boom)
    rewriter = rw.LLMQueryRewriter(_CFG)
    assert rewriter.rewrite("问题")["rewritten"] == "问题"
    assert rewriter.failures == ["call_failed:RuntimeError"]


def test_reasoning_effort_is_forced_off_for_rewrite(reply):
    """改写是意图分类：思考 token 只会把一次分类变成关键路径上的长尾。"""
    captured = reply(
        '{"rewritten":"机房巡检","aggregate":true,"year":null,"reason":"r"}'
    )
    rw.LLMQueryRewriter(_CFG).rewrite("关于机房巡检都安排过什么？")
    assert captured["llm_cfg"]["reasoning_effort"] == "none"
    assert captured["temperature"] == 0.0
    assert captured["system"] == rw.SYSTEM_REWRITE


def test_rewrite_config_without_section_still_forces_no_thinking(monkeypatch):
    """缺 rewrite 段也要走默认 none——配置缺失不该让关键路径吃上推理 token。"""
    captured: dict = {}

    def _timed(llm_cfg, user_prompt, system_prompt=None, temperature=None):
        captured.update(llm_cfg=llm_cfg, temperature=temperature)
        return '{"rewritten":"x","aggregate":false,"year":null,"reason":""}', {
            "ms": 1.0,
            "cached": False,
            "model": "m",
        }

    monkeypatch.setattr(llm_mod, "chat_timed", _timed)
    cfg = {k: v for k, v in _CFG.items() if k != "rewrite"}
    rw.LLMQueryRewriter(cfg).rewrite("问题")
    assert captured["llm_cfg"]["reasoning_effort"] == "none"


# ---------- 改写 endpoint 与合成 endpoint 的分/合 ----------


def _endpoint_cfg(**rewrite):
    return {
        "llm": {
            "model": "big",
            "base_url": "http://big",
            "api_key": "K",
            "headers": {"X-Title": "doc-rag"},
        },
        "retrieval": {},
        "rewrite": {"reasoning_effort": "none", **rewrite},
    }


@pytest.mark.parametrize(
    "rewrite",
    [
        {},  # 整段没配 → 继承
        {"model": "", "base_url": "", "api_key": ""},  # 配了但留空（env 未设）→ 继承
    ],
)
def test_empty_rewrite_endpoint_overrides_inherit_the_synthesis_llm(reply, rewrite):
    """留空必须是「继承」而不是「空 endpoint」：`env:VAR` 未设时解析出的就是空串。"""
    captured = reply('{"rewritten":"x","aggregate":false,"year":null,"reason":""}')
    rw.LLMQueryRewriter(_endpoint_cfg(**rewrite)).rewrite("问题")
    cfg = captured["llm_cfg"]
    assert (cfg["model"], cfg["base_url"], cfg["api_key"]) == ("big", "http://big", "K")
    assert cfg["headers"] == {"X-Title": "doc-rag"}


def test_rewrite_can_run_on_a_different_endpoint(reply):
    """改写花的是延迟预算、合成花的是质量预算，两者不必锁死在同一个模型上。"""
    captured = reply('{"rewritten":"x","aggregate":false,"year":null,"reason":""}')
    rw.LLMQueryRewriter(
        _endpoint_cfg(model="small", base_url="http://small", api_key="S")
    ).rewrite("问题")
    cfg = captured["llm_cfg"]
    assert (cfg["model"], cfg["base_url"], cfg["api_key"]) == (
        "small",
        "http://small",
        "S",
    )


def test_switching_provider_does_not_carry_the_other_providers_headers(reply):
    """归因头是给 `llm.base_url` 那家用的：换供应商还带着它就是把 A 家的标识发给 B 家。"""
    captured = reply('{"rewritten":"x","aggregate":false,"year":null,"reason":""}')
    rw.LLMQueryRewriter(_endpoint_cfg(model="small", base_url="http://small")).rewrite(
        "问题"
    )
    assert "headers" not in captured["llm_cfg"]


def test_same_endpoint_keeps_headers_when_only_model_changes(reply):
    captured = reply('{"rewritten":"x","aggregate":false,"year":null,"reason":""}')
    rw.LLMQueryRewriter(_endpoint_cfg(model="small")).rewrite("问题")
    assert captured["llm_cfg"]["headers"] == {"X-Title": "doc-rag"}


def test_endpoint_override_without_key_or_model_degrades_not_crashes(reply):
    """只覆盖 model 而 llm 本身没配好时，仍然是「不改写 + 计数」，不能上抛。"""
    cfg = _endpoint_cfg(model="small")
    cfg["llm"]["api_key"] = ""
    rewriter = rw.LLMQueryRewriter(cfg)
    assert rewriter.rewrite("问题")["rewritten"] == "问题"
    assert rewriter.failures == ["no_llm"]


def test_critical_path_budget_overrides_the_synthesis_defaults(reply):
    """改写的超时/重试上限必须能压下来：合成侧的 180s × 4 是给几十秒的答案用的，
    照搬过来就是一个可退化的步骤拥有了拖死整个请求的能力。"""
    captured = reply('{"rewritten":"x","aggregate":false,"year":null,"reason":""}')
    rw.LLMQueryRewriter(_endpoint_cfg(timeout_s=5, max_attempts=2)).rewrite("问题")
    assert captured["llm_cfg"]["timeout_s"] == 5.0
    assert captured["llm_cfg"]["max_attempts"] == 2

    captured.clear()
    rw.LLMQueryRewriter(_endpoint_cfg()).rewrite("问题")
    # 不配就不注入：默认行为必须与改造前逐字一致（沿用 llm 的默认超时）
    assert "timeout_s" not in captured["llm_cfg"]
    assert "max_attempts" not in captured["llm_cfg"]


def test_calls_are_timed_and_cache_hits_stay_distinguishable(reply, monkeypatch):
    """延迟必须逐次可取，且缓存命中不能混进延迟——否则 `check-rewrite` 会把
    本地 sqlite 查询报成模型延迟（这是本仓库延迟口径上犯过的老错）。"""
    captured = reply('{"rewritten":"x","aggregate":false,"year":null,"reason":""}')
    rewriter = rw.LLMQueryRewriter(_CFG)
    rewriter.rewrite("问题一")
    monkeypatch.setattr(
        llm_mod,
        "chat_timed",
        lambda *a, **k: (
            '{"rewritten":"x","aggregate":false,"year":null,"reason":""}',
            {"ms": 0.4, "cached": True, "model": "m"},
        ),
    )
    rewriter.rewrite("问题二")
    assert captured  # 桩确实被走过
    assert [c["ms"] for c in rewriter.calls] == [1.0, 0.4]
    assert [c["cached"] for c in rewriter.calls] == [False, True]


def test_failed_rewrite_is_marked_by_a_flag_not_by_sniffing_the_text(
    reply, monkeypatch
):
    """`degraded` 是机器标志：下游（/metrics、eval 的臂口径）靠嗅 `reason` 前缀
    判断「这条没真的改写」，会在文案改动时静默失效。"""

    def _boom(*a, **k):
        raise RuntimeError("503 抖动")

    ok = '{"rewritten":"x","aggregate":true,"year":null,"reason":""}'
    rewriter = rw.LLMQueryRewriter(_CFG)
    reply(ok)
    assert rewriter.rewrite("问题")["degraded"] is False
    monkeypatch.setattr(llm_mod, "chat_timed", _boom)
    assert rewriter.rewrite("问题")["degraded"] is True
    assert rewriter.rewrite("问题")["degraded"] is True
    assert rewriter.calls and len(rewriter.calls) == 1  # 失败的调用不计耗时


def test_endpoint_model_helper_reports_the_effective_model():
    assert rw.endpoint_model(_endpoint_cfg()) == "big"
    assert rw.endpoint_model(_endpoint_cfg(model="small")) == "small"
    assert rw.endpoint_model({"retrieval": {}}) is None


# ---------- 重放路径的改写可复现性 ----------
#
# 回归动机：改写换成 LLM 之后，judge 的「重放检索还原上下文」如果当场再改写一次，
# 就会拿另一条查询检索出的块去配旧答案——忠实度判的是别人的上下文。
# 实测过更糟的版本：那条路径会真的打出去，测试套件从 9s 涨到 160s 并开始烧钱。


def _legacy_results_file(tmp_path, *, with_rewritten: bool):
    import json

    items = []
    for i in range(2):
        item = {
            "id": f"q{i}",
            "type": "fact",
            "question": f"问题{i}",
            "answer": f"答案{i}",
            "contexts": [f"正文{i}"],  # 旧格式 → 触发重放
        }
        if with_rewritten:
            item |= {
                "rewritten": f"问题{i}",
                "rewrite_filters": None,
                "rewrite_aggregate": False,
            }
        items.append(item)
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "collection": "c",
                    "retrieval": "dense+bm25+rrf[dense]+rewrite",
                    "top_n": 8,
                },
                "items": items,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


class _ReplayRetriever:
    def __init__(self):
        self.seen: list[str] = []

    def retrieve(self, question, **kw):
        self.seen.append(question)
        return RetrievalOutcome(
            chunks=[
                {
                    "doc_id": "d",
                    "title": "t",
                    "page": 1,
                    "text": question.replace("问题", "正文"),
                    "block_type": "p",
                }
            ]
        )


REPLAY_CFG = {
    "eval": {"ragas_metrics": ["faithfulness"]},
    "retrieval": {"mode": "dense", "max_contexts": 10},
    "llm": {"model": "m", "base_url": "http://l", "api_key": "k", "cache": False},
    "embedding": {"base_url": "http://e", "api_key": "k", "model": "e"},
}


def test_replay_uses_recorded_query_and_never_calls_the_llm(tmp_path, monkeypatch):
    from doc_rag.eval import runner

    path = _legacy_results_file(tmp_path, with_rewritten=True)
    fake = _ReplayRetriever()
    monkeypatch.setattr(runner, "_build_retriever", lambda cfg, col: (fake, None))

    def _forbidden(*a, **k):
        raise AssertionError("重放路径不得调用模型：改写必须来自结果文件里记录的那一条")

    monkeypatch.setattr(llm_mod, "chat_timed", _forbidden)
    monkeypatch.setattr(runner, "_run_ragas", Mock(return_value={"faithfulness": 1.0}))

    runner.ragas_from_results(path, REPLAY_CFG, sample_n=2)
    assert fake.seen == ["问题0", "问题1"]


def test_replay_refuses_when_recorded_query_is_missing(tmp_path, monkeypatch):
    from doc_rag.eval import runner

    path = _legacy_results_file(tmp_path, with_rewritten=False)
    monkeypatch.setattr(
        runner, "_build_retriever", lambda cfg, col: (_ReplayRetriever(), None)
    )

    with pytest.raises(ValueError, match="不可复现"):
        runner.ragas_from_results(path, REPLAY_CFG, sample_n=2)


def test_gate_questions_are_not_taught_by_the_prompt(tmp_path):
    """泛化门禁的题目不能被改写 prompt 的 few-shot 例子教过。

    本项目作废「消融 #5」的理由就是「规则触发条件 == 评测集措辞」。同一把尺子
    必须量到自己：改造前门禁集里 `fact-02` / `fact-03` 与 SYSTEM_REWRITE 的两个
    示例输入**逐字相同**，`aggyear-01` 与第三个示例只差几个字——那样的 0.95
    里有一部分是从 prompt 里抄来的。
    """
    import difflib
    import json
    import re

    from doc_rag.config import project_root
    from doc_rag.retrieve.rewrite_llm import SYSTEM_REWRITE

    gate = project_root() / "data" / "eval" / "rewrite_paraphrase.json"
    items = json.loads(gate.read_text(encoding="utf-8"))["items"]
    shots = re.findall(r"^输入：(.+)$", SYSTEM_REWRITE, flags=re.MULTILINE)
    assert len(shots) >= 3, "few-shot 示例必须能被解析出来，否则这条护栏会空转"

    def _norm(s: str) -> str:
        return re.sub(r"[\s，。？?！「」『』《》、：；;]", "", s)

    for item in items:
        q = _norm(item["question"])
        for shot in shots:
            s = _norm(shot)
            assert q != s, f"{item['id']} 与 few-shot 示例逐字相同"
            ratio = difflib.SequenceMatcher(None, q, s).ratio()
            assert ratio < 0.6, (
                f"{item['id']} 与 few-shot 示例过于相似（{ratio:.2f}）："
                f"{item['question']} ≠ {shot}"
            )

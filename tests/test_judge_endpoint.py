"""判分 endpoint 的装配规则（`eval/judge.py`）。

RAGAS judge 与拒答审计共用这一处构造：换供应商判分时，两边的身份必须来自同一个地方，
否则「探针看到的行为」和「正式判分」会分叉——那是当年定位 faithfulness 口径 bug 时
踩过的同一类坑。
"""

from __future__ import annotations

from doc_rag.eval.judge import judge_cfg, judge_is_cross_vendor

# ------------------------------------------------------------------ judge 端点


BASE = {
    "llm": {
        "model": "gen-m",
        "base_url": "https://api.deepseek.com",
        "api_key": "k-gen",
        "temperature": 0.7,
        "headers": {"X-Title": "doc-rag"},  # 归因头是给生成那家用的
        "reasoning_effort": "",
    },
    "embedding": {"base_url": "https://api.siliconflow.cn/v1", "api_key": "k-sf"},
    "eval": {"judge": {"temperature": 0.0, "reasoning_effort": "none"}},
}


def test_judge_defaults_to_the_same_endpoint_as_the_generator():
    """默认同源是取舍不是遗漏：不给部署强加第二条必配路径。"""
    cfg = judge_cfg(BASE)
    assert cfg["model"] == "gen-m" and cfg["api_key"] == "k-gen"
    # 但 judge 自己的两条口径必须生效，不受生成侧配置牵连
    assert cfg["temperature"] == 0.0 and cfg["reasoning_effort"] == "none"


def test_cli_flag_beats_the_config_section():
    cfg = {**BASE, "eval": {"judge": {"model": "cfg-judge"}}}
    assert judge_cfg(cfg, model="cli-judge")["model"] == "cli-judge"
    assert judge_cfg(cfg)["model"] == "cfg-judge"


def test_cross_vendor_judge_reuses_the_account_key_and_drops_headers():
    """换供应商：归因头不能跟过去；指向 embedding 那家时不必再填一遍 key。"""
    cfg = judge_cfg(
        BASE, base_url="https://api.siliconflow.cn/v1", model="Qwen/Qwen3-30B-A3B"
    )
    assert cfg["api_key"] == "k-sf"
    assert "headers" not in cfg
    # 显式给 key 时不被复用逻辑覆盖
    explicit = judge_cfg(
        BASE, base_url="https://api.siliconflow.cn/v1", api_key="k-explicit"
    )
    assert explicit["api_key"] == "k-explicit"
    # 第三家、又没给 key：只能是空，绝不能拿生成侧的 key 去打别人的端点
    assert judge_cfg(BASE, base_url="https://api.other.example/v1")["api_key"] == ""


def test_budget_from_the_judge_section_reaches_the_client():
    cfg = {
        **BASE,
        "eval": {
            "judge": {"reasoning_effort": "none", "timeout_s": 7, "max_attempts": 1}
        },
    }
    out = judge_cfg(cfg)
    assert out["timeout_s"] == 7.0 and out["max_attempts"] == 1


def test_ragas_judge_uses_the_same_endpoint_builder():
    """`_judge_chat_kwargs` 不能自己再拼一份 endpoint，否则两条轨会分叉。"""
    from doc_rag.eval.runner import _judge_chat_kwargs

    kw = _judge_chat_kwargs(BASE)
    assert kw["model"] == "gen-m" and kw["base_url"] == "https://api.deepseek.com"
    assert kw["temperature"] == 0.0 and kw["reasoning_effort"] == "none"

    cross = _judge_chat_kwargs(BASE, {"base_url": "https://api.siliconflow.cn/v1"})
    assert cross["api_key"] == "k-sf" and cross["model"] == "gen-m"
    assert "headers" not in BASE["llm"] or "headers" not in cross
    assert judge_is_cross_vendor(BASE, base_url="https://api.siliconflow.cn/v1")
    assert not judge_is_cross_vendor(BASE)


def test_judge_budget_reaches_the_client_kwargs():
    from doc_rag.eval.runner import _judge_chat_kwargs

    cfg = {**BASE, "eval": {"judge": {"reasoning_effort": "none", "timeout_s": 45}}}
    assert _judge_chat_kwargs(cfg)["timeout"] == 45.0

"""判分 endpoint 的装配：judge 可以和生成不同源（PLAN §5.3「judge 自偏」）。

为什么要有这一层：judge 与生成同源时，「LLM 打分」这条轨上没有独立视角——正对照
只能证明「这条检查不是恒真」，证不出「不是自偏」。默认仍然继承 `llm`，因为给部署
强加第二条必配路径会让多数人干脆不配。

**换 judge 就是换度量身份**：只有同一批落盘答案上的判读差异可比，跨 judge 的绝对
分数不能混用（现有 Faithfulness 基线全部判于同源 judge）。
"""

from __future__ import annotations


def judge_cfg(cfg: dict, **over: str) -> dict:
    """判分用的 LLM endpoint 配置：`eval.judge.*` 覆盖 `llm`，`over` 再压一层。

    三条与安全/口径有关的规则，都有测试锁着：
    1. 换供应商时丢掉 `llm.headers`（归因头是给默认端点那家用的）；
    2. 换供应商时**不带生成侧的 api_key 过去**——那是另一家的凭证，宁可留空；
    3. 只有当 judge 端点与 embedding 端点同属一家时才复用那把 key（SiliconFlow 的
       embed 与 chat 共用一把账号 key），让密钥少一处落地。
    """
    out = dict(cfg.get("llm") or {})
    judge = dict((cfg.get("eval") or {}).get("judge") or {})
    for key in ("model", "base_url", "api_key"):
        value = str(over.get(key) or judge.get(key) or "").strip()
        if value:
            out[key] = value
    gen_base = (cfg.get("llm") or {}).get("base_url")
    explicit_key = bool(over.get("api_key") or judge.get("api_key"))
    if out.get("base_url") != gen_base:
        out.pop("headers", None)
        if not explicit_key:
            out["api_key"] = ""
    emb = cfg.get("embedding") or {}
    if not out.get("api_key") and out.get("base_url") == emb.get("base_url"):
        out["api_key"] = str(emb.get("api_key") or "")
    out["reasoning_effort"] = judge.get("reasoning_effort") or "none"
    out["temperature"] = (
        0.0 if judge.get("temperature") is None else judge["temperature"]
    )
    for key, cast in (("timeout_s", float), ("max_attempts", int)):
        if judge.get(key):
            out[key] = cast(judge[key])
    return out


def judge_is_cross_vendor(cfg: dict, **over: str) -> bool:
    """这一轮判分是不是独立视角（端点与生成侧不同家）。"""
    built = judge_cfg(cfg, **over)
    return built.get("base_url") != (cfg.get("llm") or {}).get("base_url")

"""LLM 调用共用封装：OpenAI 兼容 chat（OpenRouter 免费档 / DeepSeek 官方均可）。

- 免费档（OpenRouter :free 模型）有速率限制，故带指数退避重试
- 可选 extra_headers：OpenRouter 建议带 HTTP-Referer / X-Title 做归因
- 思考型模型若需关闭思考，用 llm.extra_body 按官方文档传参
"""

from __future__ import annotations

import time

from openai import OpenAI

_RETRIES = 4
_BACKOFF_BASE = 2.0  # 秒；免费档 429 常见，退避要够长


def chat(
    llm_cfg: dict,
    user_prompt: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
) -> str:
    client = OpenAI(
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"],
        timeout=180.0,
        default_headers=llm_cfg.get("headers") or None,
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if llm_cfg.get("max_tokens"):
        kwargs["max_tokens"] = llm_cfg["max_tokens"]
    extra = llm_cfg.get("extra_body") or None
    if extra:
        kwargs["extra_body"] = extra

    last_exc: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=llm_cfg["model"], messages=messages, **kwargs
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 免费档 429/5xx 需退避重试
            last_exc = exc
            if attempt < _RETRIES - 1:
                time.sleep(_BACKOFF_BASE * (2**attempt))
    raise RuntimeError(f"LLM 调用失败（已重试 {_RETRIES} 次）：{last_exc}") from last_exc

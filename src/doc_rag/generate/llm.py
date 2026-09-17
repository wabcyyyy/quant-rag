"""LLM 调用共用封装：OpenAI 兼容 chat（DeepSeek 官方 / SiliconFlow 可切换）。

思考型模型（如 deepseek-flash）若需关闭思考或放宽输出，用 llm.extra_body 按官方文档传参。
"""

from __future__ import annotations

from openai import OpenAI


def chat(
    llm_cfg: dict,
    user_prompt: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
) -> str:
    client = OpenAI(base_url=llm_cfg["base_url"], api_key=llm_cfg["api_key"], timeout=120.0)
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
    resp = client.chat.completions.create(model=llm_cfg["model"], messages=messages, **kwargs)
    return resp.choices[0].message.content or ""

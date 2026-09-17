"""LLM 调用共用封装：OpenAI 兼容 chat + **本地响应缓存** + 指数退避重试。

缓存设计（成本控制核心，见 PLAN §8）：
- 键 = (model, messages, temperature, max_tokens, extra_body) 的 sha256
- 存储 = .cache/llm_cache.sqlite（gitignore）
- 命中即返回，不发请求 → 同一批黄金集反复评估零成本
- 换模型/prompt 会自然产生新键，不会误用旧答案
- 关闭方式：环境变量 DOC_RAG_LLM_CACHE=0 或配置 llm.cache=false
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

from openai import OpenAI

_RETRIES = 4
_BACKOFF_BASE = 2.0  # 秒；免费档 429 常见，退避要够长
_CACHE_PATH = Path(__file__).resolve().parents[3] / ".cache" / "llm_cache.sqlite"
_STATS = {"hit": 0, "miss": 0}
_LOCK = threading.Lock()


def cache_enabled(llm_cfg: dict | None = None) -> bool:
    env = os.environ.get("DOC_RAG_LLM_CACHE")
    if env is not None:
        return env.strip() not in ("0", "false", "False", "")
    if llm_cfg is not None and "cache" in llm_cfg:
        return bool(llm_cfg["cache"])
    return True


def _conn() -> sqlite3.Connection:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_CACHE_PATH, timeout=30)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS responses ("
        "key TEXT PRIMARY KEY, model TEXT, response TEXT, created_at TEXT)"
    )
    return conn


def _cache_key(model: str, messages: list[dict], **params) -> str:
    blob = json.dumps(
        {"model": model, "messages": messages, "params": params},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> str | None:
    with _LOCK:
        try:
            conn = _conn()
            row = conn.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
            conn.close()
        except Exception:  # noqa: BLE001 缓存故障不应影响主流程
            return None
    return row[0] if row else None


def _cache_put(key: str, model: str, response: str) -> None:
    with _LOCK:
        try:
            conn = _conn()
            conn.execute(
                "INSERT OR REPLACE INTO responses (key, model, response, created_at) "
                "VALUES (?, ?, ?, ?)",
                (key, model, response, time.strftime("%Y-%m-%dT%H:%M:%S")),
            )
            conn.commit()
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def cache_stats() -> dict:
    """返回本次进程的命中统计 + 缓存总量（评估结束时打印，让节省可见）。"""
    total = 0
    try:
        conn = _conn()
        total = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        conn.close()
    except Exception:  # noqa: BLE001
        pass
    hit, miss = _STATS["hit"], _STATS["miss"]
    denom = hit + miss
    return {
        "hit": hit,
        "miss": miss,
        "hit_rate": round(hit / denom, 4) if denom else None,
        "cached_total": total,
    }


def chat(
    llm_cfg: dict,
    user_prompt: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
) -> str:
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

    use_cache = cache_enabled(llm_cfg)
    key = _cache_key(llm_cfg["model"], messages, **kwargs)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            _STATS["hit"] += 1
            return cached
    _STATS["miss"] += 1

    client = OpenAI(
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"],
        timeout=180.0,
        default_headers=llm_cfg.get("headers") or None,
    )
    last_exc: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=llm_cfg["model"], messages=messages, **kwargs
            )
            content = resp.choices[0].message.content or ""
            if use_cache and content:
                _cache_put(key, llm_cfg["model"], content)
            return content
        except Exception as exc:  # noqa: BLE001 免费档 429/5xx 需退避重试
            last_exc = exc
            if attempt < _RETRIES - 1:
                time.sleep(_BACKOFF_BASE * (2**attempt))
    raise RuntimeError(f"LLM 调用失败（已重试 {_RETRIES} 次）：{last_exc}") from last_exc

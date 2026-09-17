"""配置加载：configs/default.yaml + .env + 环境变量注入。

字符串值以 "env:VAR" 开头时从环境变量读取，避免密钥落盘（PLAN §6）。
.env 由 python-dotenv 加载（已 gitignore），优先级低于真实环境变量。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


_BASE_URL_STRIP_SUFFIXES = ("/chat/completions", "/responses", "/embeddings")


def _normalize_base_url(url: str) -> str:
    """归一化 base_url：容忍误填完整端点路径。

    实测坑：把 OpenRouter 的 Responses API 端点（/api/v1/responses）当 base_url 填入，
    OpenAI SDK 会拼成 /api/v1/responses/chat/completions → 404。
    """
    out = (url or "").rstrip("/")
    for suffix in _BASE_URL_STRIP_SUFFIXES:
        if out.endswith(suffix):
            out = out[: -len(suffix)]
            break
    return out


def _expand_env(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("env:"):
        return os.environ.get(value[4:], "")
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | Path | None = None) -> dict:
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = _expand_env(cfg)
    for section in ("llm", "embedding", "rerank"):
        if isinstance(cfg.get(section), dict) and cfg[section].get("base_url"):
            cfg[section]["base_url"] = _normalize_base_url(cfg[section]["base_url"])
    return cfg

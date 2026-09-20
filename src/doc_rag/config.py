"""配置加载：configs/default.yaml + .env + 环境变量注入。

字符串值以 "env:VAR" 开头时从环境变量读取，避免密钥落盘（PLAN §6）。
.env 由 python-dotenv 加载（已 gitignore），优先级低于真实环境变量。
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def project_root() -> Path:
    """项目根：DOC_RAG_HOME 优先，其次向上找带 configs/default.yaml 的目录，最后回落到 cwd。

    改造前这里是 `Path(__file__).parents[2]`，只在 editable 安装下成立；
    打成 wheel 装进 site-packages 后 `load_config()` 会去 site-packages 找配置文件。
    缓存路径（llm._CACHE_PATH）同理，所以两处共用这个解析器。
    """
    env = os.environ.get("DOC_RAG_HOME")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "default.yaml").is_file():
            return parent
    return Path.cwd()


ROOT = project_root()


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


_RETIRED_ENV = {
    # 旧的分流键挂在 aggregate 布尔上，已被 `llm.reasoning_effort_by_type` 取代。
    # 留着不报错的话，设了它的人会以为分流还开着——实际值已被字面量表覆盖，读都不读。
    "DOC_RAG_LLM_REASONING_EFFORT_AGGREGATE": "configs/default.yaml 的 llm.reasoning_effort_by_type"
}


def _warn_retired_env() -> None:
    for name, replacement in _RETIRED_ENV.items():
        if os.environ.get(name):
            warnings.warn(
                f"{name} 已废弃且不再被读取（改用 {replacement}）；"
                "本行请从 .env 删掉，否则你会以为思考分流仍由它控制。",
                stacklevel=3,
            )


def load_config(path: str | Path | None = None) -> dict:
    load_dotenv(ROOT / ".env")
    _warn_retired_env()
    cfg_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = _expand_env(cfg)
    for section in ("llm", "embedding", "rerank", "rewrite"):
        if isinstance(cfg.get(section), dict) and cfg[section].get("base_url"):
            cfg[section]["base_url"] = _normalize_base_url(cfg[section]["base_url"])
    # judge 可以指向另一家供应商（跨供应商复判，PLAN「judge 自偏」），同样要归一化
    judge = (cfg.get("eval") or {}).get("judge")
    if isinstance(judge, dict) and judge.get("base_url"):
        judge["base_url"] = _normalize_base_url(judge["base_url"])
    return cfg

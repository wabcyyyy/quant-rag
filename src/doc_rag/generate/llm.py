"""LLM 调用共用封装：OpenAI 兼容 chat + **本地响应缓存** + 指数退避重试。

缓存设计（成本控制核心，见 PLAN §8）：
- 键 = (model, messages, temperature, max_tokens, extra_body) 的 sha256
- 存储 = .cache/llm_cache.sqlite（gitignore）
- 命中即返回，不发请求 → 同一批黄金集反复评估零成本
- 换模型/prompt 会自然产生新键，不会误用旧答案
- 关闭方式：环境变量 DOC_RAG_LLM_CACHE=0 或配置 llm.cache=false

计时：`chat_timed` 返回每次调用的墙钟耗时 + `cached` 标志。缓存命中的耗时是
一次本地查询，**不能当作模型延迟**——测延迟必须关缓存（PLAN「延迟口径」）。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI

_RETRIES = 4
_BACKOFF_BASE = 2.0  # 秒；免费档 429 常见，退避要够长
_CACHE_PATH = Path(__file__).resolve().parents[3] / ".cache" / "llm_cache.sqlite"
# 计费口径：缓存命中不产生调用，所以只累加真实 API 调用的 token。
# reasoning 单列——推理型模型把输出预算大部分花在看不见的思考上（实测 judge 占 97%），
# 不单列就会像 PLAN 早先那样按「可见文本长度」估成本，低估一个数量级。
_STATS = {
    "hit": 0,
    "miss": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "reasoning_tokens": 0,
    # 缓存故障必须可见：读失败=重复付费，写失败=缓存永远不生效
    "cache_read_errors": 0,
    "cache_write_errors": 0,
}
_LOCK = threading.Lock()

# 只重试瞬时错误。此前无差别重试 4 次 × SDK 默认重试 2 次 = 最多 12 次传输，
# 401/400 这类永久错误也会被白白打满
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, APIStatusError):
        return getattr(exc, "status_code", 0) in _RETRYABLE_STATUS
    # 含 APITimeoutError（其子类）
    return isinstance(exc, APIConnectionError)


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


def _cache_key(llm_cfg: dict, messages: list[dict], **params) -> str:
    """键必须包含 endpoint：同名模型走不同供应商（OpenRouter / 官方）答案不同，
    混用会拿 A 家缓存回答 B 家的问题。"""
    blob = json.dumps(
        {
            "base_url": (llm_cfg.get("base_url") or "").rstrip("/"),
            "model": llm_cfg["model"],
            "messages": messages,
            "params": params,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> str | None:
    try:
        with _LOCK:
            conn = _conn()
            try:
                row = conn.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
            finally:
                conn.close()
    except Exception:  # noqa: BLE001 记数并上抛给调用方可见，不能静默变成一次重复付费
        with _LOCK:
            _STATS["cache_read_errors"] += 1
        return None
    return row[0] if row else None


def _cache_put(key: str, model: str, response: str) -> None:
    try:
        with _LOCK:
            conn = _conn()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO responses (key, model, response, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (key, model, response, time.strftime("%Y-%m-%dT%H:%M:%S")),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception:  # noqa: BLE001 写失败意味着缓存永远不生效，必须计数可见
        with _LOCK:
            _STATS["cache_write_errors"] += 1


def _timing(t0: float, cached: bool, model: str) -> dict:
    """构造计时元数据。`cached=True` 时 ms 是本地 sqlite 查询耗时，不是模型延迟。

    usage 三键固定在 meta 上（缺数据时为 None）：延迟数字此前只记耗时没记用量，
    「term 单条 28s 但答案仅 308 字」这类归因无从做起——用量必须与耗时逐条同落盘。
    """
    return {
        "ms": round((time.perf_counter() - t0) * 1000, 1),
        "cached": cached,
        "model": model,
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
    }


def cache_stats() -> dict:
    """返回本次进程的命中统计 + 真实调用 token + 缓存故障 + 缓存总量（让成本与浪费可见）。"""
    total = 0
    try:
        conn = _conn()
        total = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        conn.close()
    except Exception:  # noqa: BLE001, S110  # 缓存库不可用（锁/损坏）只丢统计，不拖垮查询主流程
        pass
    with _LOCK:
        hit, miss = _STATS["hit"], _STATS["miss"]
        denom = hit + miss
        return {
            "hit": hit,
            "miss": miss,
            "hit_rate": round(hit / denom, 4) if denom else None,
            "cached_total": total,
            # 只含真实 API 调用（缓存命中不计费不计量）
            "prompt_tokens": _STATS["prompt_tokens"],
            "completion_tokens": _STATS["completion_tokens"],
            "reasoning_tokens": _STATS["reasoning_tokens"],
            "cache_read_errors": _STATS["cache_read_errors"],
            "cache_write_errors": _STATS["cache_write_errors"],
        }


def chat(
    llm_cfg: dict,
    user_prompt: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
) -> str:
    return chat_timed(llm_cfg, user_prompt, system_prompt, temperature)[0]


def _build_request(
    llm_cfg: dict,
    user_prompt: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
) -> tuple[list[dict], dict, str, bool]:
    """构造 (messages, 请求参数, 缓存键, 是否用缓存)。

    `chat_timed` 与 `chat_stream` 必须共用这里：参数任何一处不一致都会产生
    不同的缓存键，流式写下的缓存非流式就命中不了（T7 的「缓存互通」契约）。
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    # temperature 未显式给时用配置值：否则合成走供应商默认，且配置改动不影响缓存键
    if temperature is None:
        temperature = llm_cfg.get("temperature")

    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if llm_cfg.get("max_tokens"):
        kwargs["max_tokens"] = llm_cfg["max_tokens"]
    extra = llm_cfg.get("extra_body") or None
    if extra:
        kwargs["extra_body"] = extra
    # 合成侧的思考开关（judge 走 eval.judge.reasoning_effort，见 eval/runner.py）。
    # 推理型模型把输出预算大部分花在看不见的 reasoning token 上——这是聚合题
    # 端到端延迟的主因（实测墙钟 25~34s），也是唯一有效的压延迟杠杆。
    if llm_cfg.get("reasoning_effort"):
        kwargs["reasoning_effort"] = llm_cfg["reasoning_effort"]

    use_cache = cache_enabled(llm_cfg)
    key = _cache_key(llm_cfg, messages, **kwargs)
    return messages, kwargs, key, use_cache


def chat_timed(
    llm_cfg: dict,
    user_prompt: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
) -> tuple[str, dict]:
    """同 `chat`，但返回 (文本, 计时元数据)。

    计时口径（PLAN「延迟口径」）：`ms` 是这次调用的墙钟耗时，**缓存命中时它衡量的是
    一次本地 sqlite 查询，不是模型延迟**——所以 `cached` 标志必须与耗时一起看，
    否则缓存命中会把延迟低估到毫秒级（PLAN 里 1.3s 与 6.3s 的矛盾就是这类混淆）。
    测真实延迟必须关缓存（`DOC_RAG_LLM_CACHE=0` / `--fresh-answers`）。
    """
    messages, kwargs, key, use_cache = _build_request(
        llm_cfg, user_prompt, system_prompt, temperature
    )
    t0 = time.perf_counter()
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            _STATS["hit"] += 1
            return cached, _timing(t0, cached=True, model=llm_cfg["model"])
    _STATS["miss"] += 1

    client = OpenAI(
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"],
        timeout=180.0,
        # 重试只归应用层管：SDK 再叠一层会相乘（4 × (1+SDK) 最多 12 次传输）
        max_retries=0,
        default_headers=llm_cfg.get("headers") or None,
    )
    last_exc: Exception | None = None
    for attempt in range(_RETRIES):
        t0 = time.perf_counter()  # 每轮重置：ms 只算最后一次成功尝试，不含退避等待
        try:
            resp = client.chat.completions.create(
                model=llm_cfg["model"], messages=messages, **kwargs
            )
        except Exception as exc:
            last_exc = exc
            # 永久错误（401/400/404…）重试没有意义，直接失败并说明原因
            if not _is_retryable(exc) or attempt == _RETRIES - 1:
                raise RuntimeError(f"LLM 调用失败（attempt={attempt + 1}）：{exc}") from exc
            time.sleep(_BACKOFF_BASE * (2**attempt))
            continue
        msg = resp.choices[0].message
        content = msg.content or ""
        usage = getattr(resp, "usage", None)
        if usage:
            details = getattr(usage, "completion_tokens_details", None)
            prompt_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            reasoning_tok = int(getattr(details, "reasoning_tokens", 0) or 0)
            with _LOCK:
                _STATS["prompt_tokens"] += prompt_tok
                _STATS["completion_tokens"] += completion_tok
                _STATS["reasoning_tokens"] += reasoning_tok
        else:
            prompt_tok = completion_tok = reasoning_tok = None
        if use_cache and content:
            _cache_put(key, llm_cfg["model"], content)
        meta = _timing(t0, cached=False, model=llm_cfg["model"])
        meta["attempts"] = attempt + 1  # >1 说明发生过重试，耗时含退避等待
        meta["reasoning_effort"] = kwargs.get("reasoning_effort")
        # 单次用量随 meta 透传（缓存命中/响应无 usage 时为 None，不算 0——0 会冒充「真实为零」）
        meta["prompt_tokens"] = prompt_tok
        meta["completion_tokens"] = completion_tok
        meta["reasoning_tokens"] = reasoning_tok
        return content, meta
    raise RuntimeError(f"LLM 调用失败（已重试 {_RETRIES} 次）：{last_exc}")


def chat_stream(
    llm_cfg: dict,
    user_prompt: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
) -> tuple:
    """流式调用：返回 (逐段 yield 文本的生成器, 逐步填充的 meta 字典)。

    契约（T7，PLAN §5.2 体感延迟方案，不改任何基线）：
    - 请求参数与缓存键经 `_build_request` 与 `chat_timed` 完全同源——流式完整拼装
      后按**同一缓存键**写缓存，后续非流式调用直接命中，反之亦然；
    - 流式同样受 `reasoning_effort` 等配置影响，行为与 `chat_timed` 一致；
    - 缓存命中时整段文本作为单个 chunk 返回，`meta.cached=True`（无本次用量）；
    - `meta` 在生成器耗尽后包含 ms / cached / model / usage 三键 / attempts。
    中途断流：已 yield 的内容不重发也不缓存（部分答案入缓存会污染非流式调用）。
    """
    messages, kwargs, key, use_cache = _build_request(
        llm_cfg, user_prompt, system_prompt, temperature
    )
    meta: dict = {"model": llm_cfg["model"]}

    def _gen():
        t0 = time.perf_counter()
        if use_cache:
            cached = _cache_get(key)
            if cached is not None:
                _STATS["hit"] += 1
                meta.update(
                    ms=round((time.perf_counter() - t0) * 1000, 1),
                    cached=True,
                    prompt_tokens=None,
                    completion_tokens=None,
                    reasoning_tokens=None,
                )
                yield cached
                return
        _STATS["miss"] += 1

        client = OpenAI(
            base_url=llm_cfg["base_url"],
            api_key=llm_cfg["api_key"],
            timeout=180.0,
            max_retries=0,  # 同 chat_timed：重试只归应用层管，避免与 SDK 相乘
            default_headers=llm_cfg.get("headers") or None,
        )
        parts: list[str] = []
        usage_data = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
        }
        try:
            # stream_options 让 DeepSeek 在流末尾补一个带 usage 的块（choices 为空）
            stream = client.chat.completions.create(
                model=llm_cfg["model"],
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )
            for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage:
                    details = getattr(usage, "completion_tokens_details", None)
                    prompt_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
                    completion_tok = int(getattr(usage, "completion_tokens", 0) or 0)
                    reasoning_tok = int(getattr(details, "reasoning_tokens", 0) or 0)
                    with _LOCK:
                        _STATS["prompt_tokens"] += prompt_tok
                        _STATS["completion_tokens"] += completion_tok
                        _STATS["reasoning_tokens"] += reasoning_tok
                    usage_data = {
                        "prompt_tokens": prompt_tok,
                        "completion_tokens": completion_tok,
                        "reasoning_tokens": reasoning_tok,
                    }
                    continue  # usage 块没有 choices
                for choice in getattr(chunk, "choices", None) or []:
                    piece = getattr(getattr(choice, "delta", None), "content", None)
                    if piece:
                        parts.append(piece)
                        yield piece
        except Exception as exc:
            raise RuntimeError(f"LLM 流式调用失败：{exc}") from exc
        content = "".join(parts)
        if use_cache and content:
            _cache_put(key, llm_cfg["model"], content)
        meta.update(
            ms=round((time.perf_counter() - t0) * 1000, 1),
            cached=False,
            attempts=1,
            reasoning_effort=kwargs.get("reasoning_effort"),
            **usage_data,
        )

    return _gen(), meta

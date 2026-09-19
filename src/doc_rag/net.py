"""HTTP 出口的统一重试与错误分类：嵌入 / 重排 / LLM 三条外呼共用一套判定。

改造前三处各写一遍且互不一致：`llm.py` 精细区分瞬时与永久错误，`embedder.py` 对
401/400 也照盲重试 3 次（永久错误白打三遍），`rerank.py` 一次重试都没有——
一次 429 就能打死整条查询。判定表只留这一份。
"""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx

# 408 请求超时 · 409 冲突 · 429 限流 · 5xx 服务端
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def _status_of(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:  # httpx.HTTPStatusError 把状态码放在 .response 上
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return int(status) if status is not None else None


def is_retryable(exc: BaseException, extra_types: tuple[type, ...] = ()) -> bool:
    if extra_types and isinstance(exc, extra_types):
        return True
    status = _status_of(exc)
    if status is not None:
        return status in RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError)


def request_with_retry[T](
    fn: Callable[[], T],
    *,
    label: str,
    attempts: int = 4,
    backoff_base: float = 2.0,
    extra_retryable: tuple[type, ...] = (),
) -> T:
    """指数退避重试，只重试瞬时错误。永久错误立即上抛并说明原因。

    退避不计入调用方计时：`llm.chat_timed` 每轮重置 t0，重试过的痕迹靠
    `meta.attempts` 而不是靠耗时膨胀来体现。
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # 分类后决定是否重试，最终仍会上抛
            last = exc
            if not is_retryable(exc, extra_retryable) or attempt == attempts - 1:
                raise RuntimeError(
                    f"{label} 请求失败（attempt={attempt + 1}）：{exc}"
                ) from exc
            time.sleep(backoff_base * (2**attempt))
    raise RuntimeError(f"{label} 请求失败（已重试 {attempts} 次）：{last}") from last

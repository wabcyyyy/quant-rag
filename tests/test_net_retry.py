"""统一重试判定的测试：net.py 是三条外呼共用的判定表，错了会同时影响三路。"""

from __future__ import annotations

import httpx
import pytest

from doc_rag import net


class _Status(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _WithResponse(Exception):
    """httpx.HTTPStatusError 的形状：状态码挂在 .response 上。"""

    def __init__(self, status_code):
        super().__init__(f"response {status_code}")
        self.response = httpx.Response(status_code=status_code, request=_REQUEST)


class _NoStatus(Exception):
    pass


_REQUEST = httpx.Request("POST", "http://x/y")


@pytest.mark.parametrize("code", sorted(net.RETRYABLE_STATUS))
def test_transient_status_codes_are_retryable(code):
    assert net.is_retryable(_Status(code)) is True


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_permanent_status_codes_are_not_retried(code):
    """永久错误重试没有意义：改造前 embedder 对 401 也照打 3 次。"""
    assert net.is_retryable(_Status(code)) is False
    assert net.is_retryable(_WithResponse(code)) is False


def test_transport_errors_are_retryable():
    assert net.is_retryable(httpx.ConnectError("boom")) is True


def test_unknown_exception_is_not_retryable():
    assert net.is_retryable(_NoStatus()) is False


def test_extra_types_cover_wrappers_without_status_code():
    """openai.APIConnectionError 没有 status_code，靠调用方补一类。"""

    class ConnErr(Exception):
        pass

    assert net.is_retryable(ConnErr(), (ConnErr,)) is True
    assert net.is_retryable(ConnErr()) is False


def test_retries_then_succeeds(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(net.time, "sleep", slept.append)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Status(429)
        return "ok"

    assert net.request_with_retry(fn, label="t", attempts=4) == "ok"
    assert calls["n"] == 3
    assert slept == [2.0, 4.0]  # 指数退避，不是固定间隔


def test_permanent_error_fails_on_first_attempt(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(net.time, "sleep", slept.append)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _Status(401)

    with pytest.raises(RuntimeError, match="attempt=1"):
        net.request_with_retry(fn, label="embedding", attempts=4)
    assert calls["n"] == 1  # 永久错误一次就停
    assert slept == []


def test_exhausted_retries_wrap_with_attempt_count(monkeypatch):
    monkeypatch.setattr(net.time, "sleep", lambda *_: None)

    def fn():
        raise _Status(503)

    with pytest.raises(RuntimeError, match="attempt=3"):
        net.request_with_retry(fn, label="rerank", attempts=3)

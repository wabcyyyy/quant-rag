"""FastAPI（PLAN §2 交付行）。启动：uv run doc-rag serve

两个问答端点都是 `Orchestrator` 的薄壳：拼装逻辑（含重排与上下文截断）不在这里，
所以端点跑的配置与 `doc-rag eval` 测的配置是同一份，不会再各写一份而漂移。

鉴权与 kb 边界是后补的：改造前三条路由零 Depends、零鉴权，且 `body.kb` 未校验即可
指向该 Qdrant 实例上的任意 collection。
"""

from __future__ import annotations

import secrets
from collections import defaultdict
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

app = FastAPI(title="doc-rag", version="0.1.0")

# 限流窗口：进程内、按令牌聚合。跨实例部署时换成共享存储
import threading

_RATE_LOCK = threading.Lock()
_RATE_WINDOW: dict[str, list[float]] = defaultdict(list)


class QueryIn(BaseModel):
    question: str
    kb: str | None = None
    top_n: int | None = None


@lru_cache(maxsize=1)
def _orchestrator():
    """进程级单例可以缓存：Orchestrator 只持有不可变的 cfg 与懒建的连接。

    改造前这里缓存的是 (cfg, retriever, rewriter) 三元组，而端点用
    `retriever.collection = body.kb` 改这个共享对象——并发请求会互相串库。
    现在 collection 由每次调用的 `kb` 参数传入，没有跨请求可变状态。
    """
    from doc_rag.config import load_config
    from doc_rag.orchestrator import Orchestrator

    return Orchestrator(load_config())


def _api_cfg() -> dict:
    return _orchestrator().cfg.get("api") or {}


def require_token(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """令牌 + 限流：两条都在依赖里，端点 body 之前就生效。"""
    expected = str(_api_cfg().get("auth_token") or "")
    if not expected:
        # 未配令牌时拒绝服务，而不是悄悄裸奔
        raise HTTPException(503, "服务端未配置 DOC_RAG_API_TOKEN，拒绝提供问答")
    given = (authorization or "").encode("utf-8", "surrogateescape")
    # 常数时间比较：这不是登录口令而是全库只读令牌，逐字节短路仍会把匹配前缀
    # 的长度泄漏给能计时的人。
    ok = any(
        secrets.compare_digest(given, form.encode("utf-8", "surrogateescape"))
        for form in (expected, f"Bearer {expected}")
    )
    if not ok:
        raise HTTPException(401, "Authorization 缺失或不正确")
    _rate_limit_or_raise()


def _rate_limit_or_raise() -> None:
    """进程内滑动窗口限流（按服务端令牌聚合，等价于全局每分钟配额）。

    口径要老实：窗口按**服务端配置里的那一个令牌**聚合，而所有合法调用方带的都是
    同一个令牌——所以现在它是「全局每分钟问答数」的闸门，不是按调用方分的。
    要按调用方限流，得先有真正的身份（多令牌或 API key → 主体映射）。
    `uvicorn --workers N` 或多实例下每份进程各算各的，配额会翻倍——那要换共享存储。

    为什么必须有限流：一次问答在慢路径上要占住 starlette 线程池槽位到 180s
    （llm 超时），默认池 40 线程——十几个并发聚合请求就能把服务拖成队列。
    """
    import time as _time

    limit = int(_api_cfg().get("rate_limit_rpm") or 0)
    if limit <= 0:
        return
    from ..log import request_id  # noqa: F401  仅确保日志层已就绪

    token = str(_api_cfg().get("auth_token") or "")
    now = _time.monotonic()
    with _RATE_LOCK:
        hits = _RATE_WINDOW.setdefault(token, [])
        while hits and now - hits[0] > 60.0:
            hits.pop(0)
        if len(hits) >= limit:
            raise HTTPException(429, f"超过 {limit} 次/分钟，请稍后重试")
        hits.append(now)


def _check_kb(kb: str | None) -> None:
    if kb is None:
        return
    allowed = list(_api_cfg().get("allowed_collections") or [])
    if kb not in allowed:
        raise HTTPException(403, f"kb 不在允许列表内：{allowed}")


def _record(result, *, endpoint: str) -> None:
    """把一次问答的度量推进进程内注册表（`GET /metrics` 读的就是这些）。"""
    from .. import metrics

    metrics.inc("doc_rag_requests_total", endpoint=endpoint)
    lat = result.latency_ms
    metrics.observe_ms("doc_rag_e2e_ms", lat.get("total"))
    metrics.observe_ms(
        "doc_rag_retrieval_ms", lat.get("retrieval_total"), endpoint=endpoint
    )
    metrics.observe_ms("doc_rag_rewrite_ms", lat.get("rewrite"))
    if result.rerank_error:
        metrics.inc("doc_rag_rerank_failures_total")
    if result.plan.get("degraded"):
        metrics.inc("doc_rag_rewrite_failures_total")
    if result.filter_fallback:
        # 过滤后结果太少 → 这轮其实是「无过滤」。它悄悄改变答案依据，必须能看见。
        metrics.inc("doc_rag_filter_fallback_total")
    trace = result.trace
    if trace:
        # agent 层的四条：开了几步、停在哪、花了多少、有多少「停」其实是判定挂了。
        # 停机原因的分布就是失败分类的数据源，只报平均步数会把两类病混成一个数。
        steps = trace.get("steps") or []
        metrics.inc("doc_rag_agent_requests_total")
        metrics.observe_ms("doc_rag_agent_ms", result.latency_ms.get("agent"))
        metrics.inc("doc_rag_agent_steps_total", by=len(steps))
        metrics.inc(
            "doc_rag_agent_stops_total", stop_reason=str(trace.get("stop_reason"))
        )
        if any(s.get("degraded") for s in steps):
            metrics.inc("doc_rag_agent_judge_degraded_total")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics", dependencies=[Depends(require_token)])
def metrics_endpoint() -> PlainTextResponse:
    """进程内累计度量（Prometheus 文本格式）。要令牌：不对外裸奔。"""
    from .. import metrics

    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


@app.post("/query", dependencies=[Depends(require_token)])
def query(body: QueryIn) -> dict:
    from ..log import get_logger, new_request_id, request_id

    _check_kb(body.kb)
    log = get_logger("api")
    reset = new_request_id()
    try:
        result = _orchestrator().answer(body.question, kb=body.kb, top_n=body.top_n)
        # 只记结构化度量，不记问题原文与答案：语料是公司内容，日志外传等于第二条泄露路径
        log.info(
            "query",
            extra={
                "stage": "query",
                "ms": result.latency_ms.get("total"),
                "kb": body.kb,
                "n_retrieved": len(result.retrieved),
                "n_contexts": len(result.contexts),
                "rerank_error": result.rerank_error,
                "aggregate": result.plan.get("aggregate"),
                "model": (result.synth_meta or {}).get("model"),
                "cached": (result.synth_meta or {}).get("cached"),
                "agent_steps": len((result.trace or {}).get("steps") or []),
                "agent_stop_reason": (result.trace or {}).get("stop_reason"),
            },
        )
        _record(result, endpoint="query")
    finally:
        request_id.reset(reset)
    return {
        "question": body.question,
        "rewrite": result.plan,
        "answer": result.answer,
        "citations": result.citations,
        "rerank_error": result.rerank_error,
        # 延迟口径（PLAN「延迟口径」）：synth_cached 为真时 synthesize 是缓存查询
        # 耗时而非模型延迟，客户端据此决定要不要信这个数
        "latency_ms": result.latency_ms,
        # agent 臂的逐步轨迹；单发路径是 null。调用方要用它核对成本，
        # 也要能在事后回答「这条答案是哪几步检索拼出来的」。
        "trace": result.trace,
    }


@app.post("/query/stream", dependencies=[Depends(require_token)])
def query_stream(body: QueryIn):
    """流式问答（T7）：SSE 事件 = rewrite / delta / citations / done(含 latency_ms)。

    事件名与 `Orchestrator.answer_stream` 的产出逐字对应；该生成器耗尽时 `done`
    带上的计时与用量与非流式 `answer()` 同口径（缓存键亦同源，见 llm._build_request）。
    """
    import json
    from collections.abc import Iterator

    from fastapi.responses import StreamingResponse

    _check_kb(body.kb)

    def _event(name: str, data) -> str:
        return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _sse() -> Iterator[str]:
        from .. import metrics

        # 客户端中途断开时 `done` 永远不到，度量会整条消失——请求数、分位数、
        # 失败计数全都少算。断连必须单独计数，不然「成功率」是拿自己没记的
        # 请求当分母算出来的。
        completed = False
        try:
            for ev in _orchestrator().answer_stream(
                body.question, kb=body.kb, top_n=body.top_n
            ):
                kind = ev["type"]
                if kind == "delta":
                    yield _event("delta", {"text": ev["text"]})
                elif kind == "citations":
                    yield _event("citations", ev["citations"])
                elif kind == "done":
                    _record(ev["result"], endpoint="query_stream")
                    completed = True
                    yield _event(
                        "done",
                        {
                            "latency_ms": ev["result"].latency_ms,
                            "rerank_error": ev["result"].rerank_error,
                            # 与 /query 的 done 载荷同字段：流式调用方也要能事后
                            # 核对「这条答案是那几步检索拼出来的」
                            "trace": ev["result"].trace,
                        },
                    )
                else:
                    yield _event(kind, ev["plan"])
        finally:
            if not completed:
                metrics.inc("doc_rag_stream_incomplete_total")

    return StreamingResponse(_sse(), media_type="text/event-stream")

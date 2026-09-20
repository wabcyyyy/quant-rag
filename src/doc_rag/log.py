"""最小结构化日志：一行一个 JSON。

改造前 `src/` 里没有任何日志（grep "import logging" 为空）。后果不是「不好排查」这么轻：
`eval.runner._maybe_rerank` 把重排异常静默吞成「退回融合顺序」，结果文件却继续自称
`+rerank`——没有日志也没有别的痕迹，一次重排服务抖动就能产出一份标错组的评估。
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar, Token

# 每次问答一个 id：分阶段日志要能串起来，否则并发下只能看到一串无主的时间
request_id: ContextVar[str] = ContextVar("request_id", default="-")

_FIELDS = (
    "stage",
    "ms",
    "kb",
    "n_contexts",
    "n_retrieved",
    "rerank_error",
    "rewrite_failures",
    "error",
    "model",
    "cached",
    "aggregate",
    # agent 层的两条必须能在日志里看见：步数与停机原因。少了它们，「这条请求为什么
    # 慢/为什么拒答」在日志里就没有答案——而 trace 本身太长，不该整份塞进日志行。
    "agent_steps",
    "agent_stop_reason",
)


def new_request_id() -> Token[str]:
    return request_id.set(uuid.uuid4().hex[:12])


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id.get(),
        }
        for key in _FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure(level: str = "INFO") -> None:
    """幂等安装到 stderr。库本身不配 handler，由入口（serve / CLI）调用。"""
    root = logging.getLogger("doc_rag")
    if any(getattr(h, "_doc_rag", False) for h in root.handlers):
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler._doc_rag = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"doc_rag.{name}")

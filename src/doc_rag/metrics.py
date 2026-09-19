"""进程内最小指标注册表：给探针和压测用，不引 prometheus 客户端。

为什么要有：README 立了「聚合 ≤5s / 短答 ≤8s」的分题型 SLO，但此前没有任何
可查询的运行时出口——SLO 只能靠人工跑 `doc-rag eval` 复算。Phase 3 压测也需要
一个能持续读的口子，所以先落一个零依赖的计数/分位数出口。
"""

from __future__ import annotations

import threading
from collections import defaultdict

_LOCK = threading.Lock()
_COUNTERS: dict[str, int] = defaultdict(int)
# 有界样本：延迟分位数够用即可，长跑进程不能无上限吃内存
_SAMPLES: dict[str, list[float]] = defaultdict(list)
_MAX_SAMPLES = 500


def _key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    rendered = ",".join(f'{k}="{labels[k]}"' for k in sorted(labels))
    return f"{name}{{{rendered}}}"


def inc(name: str, by: int = 1, **labels: str) -> None:
    with _LOCK:
        _COUNTERS[_key(name, {k: str(v) for k, v in labels.items()})] += by


def observe_ms(name: str, ms: float | None, **labels: str) -> None:
    """记 count/sum 与分位数样本。

    分位数按**指标名**聚合（样本窗口有界），labels 只参与 count/sum 的键；
    Prometheus 的标签必须跟在指标名后面，所以拼键时先加后缀再套标签。
    """
    if ms is None:
        return
    lab = {k: str(v) for k, v in labels.items()}
    with _LOCK:
        _COUNTERS[_key(f"{name}_count", lab)] += 1
        _COUNTERS[_key(f"{name}_sum", lab)] += round(ms)
        samples = _SAMPLES[_key(name, {})]
        samples.append(float(ms))
        if len(samples) > _MAX_SAMPLES:
            del samples[: len(samples) - _MAX_SAMPLES]


def render() -> str:
    """Prometheus 文本格式（counter + 已算好的 p50/p95 快照）。"""
    lines: list[str] = []
    with _LOCK:
        items = dict(_COUNTERS)
        for name, xs in _SAMPLES.items():
            xs_sorted = sorted(xs)
            if not xs_sorted:
                continue
            for pct in (50, 95):
                idx = max(
                    0, min(len(xs_sorted) - 1, round(pct / 100 * len(xs_sorted)) - 1)
                )
                lines.append(f"{name}_p{pct} {xs_sorted[idx]}")
        # TYPE 只声明一次：同名指标重复 # TYPE 行不合规，会被抓取端拒掉
        names: set[str] = set()
        for key in sorted(items):
            base = key.partition("{")[0]
            metric_name = base.removesuffix("_sum")
            if metric_name not in names:
                names.add(metric_name)
                lines.insert(0, f"# TYPE {metric_name} untyped")
            lines.append(f"{key} {items[key]}")
    return "\n".join(lines) + "\n"


def reset() -> None:
    """测试用：清掉累加状态，避免用例之间互相污染。"""
    with _LOCK:
        _COUNTERS.clear()
        _SAMPLES.clear()

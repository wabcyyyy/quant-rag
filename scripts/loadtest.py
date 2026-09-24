"""B8 并发压测：本地假 LLM/embed/rerank 端点 + 真 Qdrant，打真 /query 服务。

回答两个问题（SPEC B8）：
1. 「什么并发下 p95 超 8s」——假端点用 --delay-ms 模拟慢路径（合成的真实量级
   见 PLAN 分题型延迟表：短答 ~2s、聚合 ~25s；默认 1500ms 介于其间）；
2. 「限流 30 RPM 与 40 线程池谁是先到的瓶颈」——服务端 uvicorn 线程池 40、
   进程内限流 `rate_limit_rpm`（默认 30）。压测轮的 4xx 计数会直接显示谁先到：
   429 = 限流先到，排队延迟暴涨 + 无 4xx = 线程池先到。

架构：假端点（本进程 ThreadingHTTPServer，:8790）+ doc-rag serve（子进程，
环境变量把三个端点全指到假服务）+ 真 Qdrant（localhost:6333 的既有 collection）。
零外部 API 成本：LLM/embed/rerank 全是本地桩。

运行（本轮只冒烟不进 PLAN 表，正式曲线下一轮采）：
  uv run python scripts/loadtest.py --kb doc_rag_sample --rounds 20
  uv run python scripts/loadtest.py --smoke   # 1 并发 × 3 题，验证链路
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FAKE_PORT = 8790
SERVE_PORT = 8791
TOKEN = "loadtest-token"


class _FakeHandler(BaseHTTPRequestHandler):
    """三个桩：embeddings / chat/completions / rerank（OpenAI 或 SiliconFlow 形状）。"""

    delay_s: float = 1.5

    def log_message(self, *args):
        return

    def _respond(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) or b"{}"
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            req = {}
        time.sleep(self.delay_s)
        if self.path.endswith("embeddings"):
            self._respond(
                {
                    "data": [{"embedding": [0.001] * 1024, "index": 0}],
                    "model": "fake-embed",
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                }
            )
        elif self.path.endswith("rerank"):
            n_docs = len(req.get("documents") or [])
            self._respond(
                {
                    "results": [
                        {"index": i, "relevance_score": 1.0 / (i + 1)}
                        for i in range(n_docs)
                    ]
                }
            )
        else:  # chat/completions：改写要 JSON plan，合成/判定要正文
            blob = raw.decode("utf-8", errors="ignore")
            if '"rewritten"' in blob or "aggregate" in blob:
                content = json.dumps(
                    {
                        "aggregate": False,
                        "rewritten": "压测改写",
                        "filters": None,
                        "top_n": None,
                        "reason": "stub",
                        "degraded": False,
                    },
                    ensure_ascii=False,
                )
            else:
                content = "压测桩答案 [1]"
            self._respond(
                {
                    "id": "fake",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                    },
                }
            )


def start_fake_server(delay_ms: int) -> ThreadingHTTPServer:
    _FakeHandler.delay_s = delay_ms / 1000.0
    server = ThreadingHTTPServer(("127.0.0.1", FAKE_PORT), _FakeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _kill_stale_serve() -> None:
    """上一轮没死透的 serve 会占住端口：新进程绑定失败、health 打到旧进程上，
    压测的就成了旧代码（本脚本开发时就踩了这一下）。启动前先清端口。"""
    try:
        # Windows 的 netstat 输出是本地编码（GBK），按 utf-8 解会炸
        out = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="ignore",
        ).stdout
        if out is None or ":8791" not in out:
            out = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                check=False,
                encoding="gbk",
                errors="ignore",
            ).stdout
    except Exception:  # noqa: BLE001 非 Windows 下跳过（netstat 形态不同）
        return
    if not out:
        return
    pids = set()
    for line in out.splitlines():
        if f":{SERVE_PORT}" in line and "LISTENING" in line:
            parts = line.split()
            if parts:
                pids.add(parts[-1])
    for pid in pids:
        subprocess.run(
            ["taskkill", "/F", "/PID", pid], capture_output=True, check=False
        )
    if pids:
        time.sleep(1.0)


def start_serve(kb: str) -> subprocess.Popen:
    env = {
        **os.environ,
        "DOC_RAG_LLM_BASE_URL": f"http://127.0.0.1:{FAKE_PORT}/v1",
        "DOC_RAG_LLM_API_KEY": "fake",
        "DOC_RAG_LLM_MODEL": "fake-llm",
        "DOC_RAG_LLM_CACHE": "0",
        "DOC_RAG_EMBEDDING_BASE_URL": f"http://127.0.0.1:{FAKE_PORT}/v1",
        "DOC_RAG_EMBEDDING_API_KEY": "fake",
        "DOC_RAG_API_TOKEN": TOKEN,
    }
    if kb:
        env["DOC_RAG_KB"] = kb  # /query 默认 kb（allowed_collections 含示例库）
    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "doc-rag",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(SERVE_PORT),
        ],
        env=env,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def wait_health(proc: subprocess.Popen, timeout_s: float = 90) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise SystemExit("doc-rag serve 进程退出了（检查端口占用 / Qdrant）")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{SERVE_PORT}/health", timeout=2
            ) as resp:
                if resp.status == 200:
                    return
        except Exception:  # noqa: BLE001 还没起来
            time.sleep(1.0)
    raise SystemExit(f"serve {timeout_s}s 内未就绪")


def load_questions(n: int) -> list[str]:
    """取 gold_core 的前 n 个问题（全部虚构语料，合规）。"""
    gold = ROOT / "data" / "eval" / "gold_core.json"
    if gold.exists():
        qs = [it["question"] for it in json.loads(gold.read_text("utf-8"))["items"]]
        return (qs * ((n // len(qs)) + 1))[:n]
    return [f"压测问题{i}：新仓库建设进展如何？" for i in range(n)]


def one_request(question: str, kb: str) -> tuple[int, float]:
    body = json.dumps({"question": question, "kb": kb}).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{SERVE_PORT}/query",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
        },
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            _ = resp.read()
            return resp.status, (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as exc:
        _ = exc.read()
        return exc.code, (time.perf_counter() - t0) * 1000


def run_round(questions: list[str], concurrency: int, kb: str) -> dict:
    codes: list[int] = []
    latencies: list[float] = []
    lock = threading.Lock()

    def work(q: str) -> None:
        code, ms = one_request(q, kb)
        with lock:
            codes.append(code)
            latencies.append(ms)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(work, questions))
    wall = time.perf_counter() - t0
    ok = [m for c, m in zip(codes, latencies) if c == 200]
    lat_sorted = sorted(ok)
    p = lambda q: (
        lat_sorted[min(len(lat_sorted) - 1, round(q / 100 * len(lat_sorted)) - 1)]
        if lat_sorted
        else None
    )
    return {
        "concurrency": concurrency,
        "n": len(questions),
        "ok": len(ok),
        "p50_ms": round(p(50), 1) if ok else None,
        "p95_ms": round(p(95), 1) if ok else None,
        "max_ms": round(max(ok), 1) if ok else None,
        "mean_ms": round(statistics.mean(ok), 1) if ok else None,
        "wall_s": round(wall, 1),
        "codes": {str(c): codes.count(c) for c in sorted(set(codes)) if codes.count(c)},
    }


def read_serve_metrics() -> dict[str, float]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{SERVE_PORT}/metrics",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            text = resp.read().decode("utf-8")
    except Exception:  # noqa: BLE001 metrics 读不到就空着
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            parts = line.split()
            if len(parts) == 2:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="B8 并发压测（本地桩，零 API 成本）")
    parser.add_argument("--kb", default="doc_rag_sample")
    parser.add_argument("--rounds", type=int, default=20, help="每档并发打多少题")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--delay-ms", type=int, default=1500, help="假端点延迟")
    parser.add_argument("--smoke", action="store_true", help="1 并发 × 3 题链路验证")
    parser.add_argument("--keep-serve", action="store_true", help="调试：不关 serve")
    args = parser.parse_args()
    if args.smoke:
        args.rounds = 3
        args.concurrency = [1]
        args.delay_ms = min(args.delay_ms, 200)

    questions = load_questions(args.rounds)
    _kill_stale_serve()
    fake = start_fake_server(args.delay_ms)
    proc = start_serve(args.kb)
    try:
        wait_health(proc)
        rows = [run_round(questions, c, args.kb) for c in args.concurrency]
        metrics = read_serve_metrics()
    finally:
        if args.keep_serve:
            print(f"serve 保留中（pid {proc.pid}，端口 {SERVE_PORT}）")
        else:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            fake.shutdown()

    print(
        f"\n=== B8 压测（假端点 delay={args.delay_ms}ms · {args.rounds} 题/档 · "
        f"kb={args.kb} · 服务端限流 30RPM/线程池 40）==="
    )
    print("并发 | n | ok/4xx/5xx | p50 | p95 | max | 墙钟")
    for r in rows:
        codes = r["codes"]
        print(
            f"{r['concurrency']:>4} | {r['n']:>3} | {r['ok']}/{codes.get('429', 0) + codes.get('403', 0)}4xx/"
            f"{codes.get('500', 0) + codes.get('503', 0)}5xx "
            f"| {r['p50_ms']} | {r['p95_ms']} | {r['max_ms']} | {r['wall_s']}s"
        )
    if metrics:
        keys = [k for k in metrics if k.startswith("doc_rag_")]
        print("服务端指标：")
        for k in sorted(keys):
            print(f"  {k} = {metrics[k]}")
    print(
        "判读：p95 首超 8000ms 的并发档 = SLO 边界；429 出现 = 限流先到，"
        "无 4xx 且延迟随并发暴涨 = 线程池排队先到。"
    )


if __name__ == "__main__":
    main()

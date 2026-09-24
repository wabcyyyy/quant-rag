"""B6：文档漂移护栏——已知写错过一次的数字/措辞，测试里钉死。

这四条都是真实漂移（B6 清单）：README 的测试数、hybrid.py docstring 的 doc_date
覆盖率、PLAN 选型表的「本地小模型」、README 的「~15 份扫描件」。护栏分两层：
已知的**陈旧串必须缺席**；README 的测试数必须与实测一致（加测试不改 README 会红）。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(*parts: str) -> str:
    return (ROOT.joinpath(*parts)).read_text("utf-8")


def test_known_stale_strings_are_gone():
    assert "18.8%" not in _read("src", "doc_rag", "retrieve", "hybrid.py"), (
        "hybrid.py docstring 的 doc_date 覆盖率是回填前的历史常数——过滤前的字段"
        "覆盖率因语料而异，写死任何百分比都会再漂移"
    )
    assert "本地小模型" not in _read("docs", "design", "PLAN.md"), (
        "Reranker 从未跑过本地小模型方案：实际是 bge-reranker-v2-m3 via API rerank"
    )
    readme = _read("README.md")
    assert "~15 份" not in readme, (
        "扫描件数量以 profile 实测为准：92 = 21 scan_likely + 71 near_empty（A3.3）"
    )
    assert "371 项" not in readme, "README 的测试数漂移过一次（实测 371→428→455→…）"


def test_readme_test_count_matches_reality():
    """README「测试（N 项」必须等于 tests/ 下 def test_ 的实际数量。

    数一遍只要几毫秒；加参数化用例时这里的口径是「函数数」，需要在 README 同步
    （护栏的价值就是逼着这次同步发生，而不是静默漂移）。
    """
    n = 0
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        n += len(re.findall(r"^def test_", p.read_text("utf-8"), flags=re.MULTILINE))
    readme = _read("README.md")
    m = re.search(r"测试（(\d+) 项", readme)
    assert m, "README 缺少「测试（N 项」标记"
    assert int(m.group(1)) == n, (
        f"README 写 {m.group(1)} 项，实测 def test_ = {n}——改 README（数字来自实测）"
    )

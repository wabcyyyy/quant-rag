"""配置里写的键必须真的被代码读取。

动机（实测）：`retrieval.parent_expand: true` 在 src 里零引用——配置文件在宣称一个
根本不存在的「父文档扩展」能力；`rrf_k`、`rerank_top_n` 同样无人读取。而
`embedding.topics` 之类被建了索引却永久为空的字段，是同一类问题的另一面。
这个测试把「配置即规格」变成硬约束：加一个键，就必须同时加一处读取。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "doc_rag"

# 整棵子树原样透传给外部 SDK，逐键查引用只会产生假阳性
PASS_THROUGH = ("llm.headers",)


def _leaf_paths(node: dict, prefix: str = ""):
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from _leaf_paths(value, path)
        else:
            yield path, str(key)


def _leaves() -> list[tuple[str, str]]:
    cfg = yaml.safe_load(
        (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    )
    out = []
    for path, key in _leaf_paths(cfg):
        if any(path.startswith(p + ".") for p in PASS_THROUGH):
            continue
        out.append((path, key))
    return out


@pytest.fixture(scope="module")
def sources() -> list[str]:
    return [p.read_text(encoding="utf-8") for p in SRC.rglob("*.py")]


def test_scanner_actually_inspects_the_config():
    """防假阳性：扫描器坏了会让下面那条测试空跑通过。"""
    paths = {p for p, _ in _leaves()}
    assert len(paths) >= 25, f"只解析出 {len(paths)} 个配置键，扫描逻辑可能失效"
    assert any(p.endswith("max_contexts") for p in paths)


def test_default_yaml_has_no_unread_keys(sources):
    """每个叶子键都要以 ["k"] 或 .get("k" 的形式出现在 src 里。"""
    dead = []
    for path, key in _leaves():
        needles = (f'["{key}"]', f'.get("{key}"', f".get('{key}'")
        if not any(n in text for text in sources for n in needles):
            dead.append(path)
    assert not dead, f"配置里有代码从不读取的键（配置在说谎）：{dead}"

"""docker-compose 钉的镜像 tag 必须是「真的存在」的那种。

动机（实测踩过）：把 Qdrant 从 `latest` 改成 `1.19.1` 看着像已经锁死了，实际 Docker
Hub 上没有这个标签——它叫 `v1.19.1`。本地容器一直在跑所以毫无察觉，而 README 承诺的
「clone 即可端到端复现」会在第一条 `docker compose up -d` 上以
`failed to resolve reference` 断掉。镜像仓库的标签是否存在没法离线查，所以这里锁住
可离线验证的那一半：**必须有 `v` 前缀、必须是三段号、必须不是 latest**。
运行时的那一半由 `doc-rag check` 补（它把跑着的 server 版本与这里的钉版对一遍）。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from doc_rag.cli import _pinned_qdrant_tag

ROOT = Path(__file__).resolve().parents[1]


def _image() -> str:
    spec = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    return spec["services"]["qdrant"]["image"]


def test_qdrant_image_is_a_three_part_version_with_the_registry_v_prefix():
    """`1.19.1` 这种写法在 Docker Hub 上不存在，必须写成 `v1.19.1`。"""
    image = _image()
    assert image.startswith("qdrant/qdrant:"), f"镜像仓库被改动：{image}"
    tag = image.split(":", 1)[1]
    assert re.fullmatch(r"v\d+\.\d+\.\d+", tag), (
        f"Qdrant tag {tag!r} 不是 `v主.次.补丁` 形式："
        "`latest` 会让检索行为随时间漂移（prefetch+RRF 是服务端融合），"
        "少一个 `v` 前缀则拉不到镜像（Docker Hub 实测只有 v 前缀的标签）"
    )


def test_parser_agrees_with_the_image_line(tmp_path):
    assert _pinned_qdrant_tag() == _image().split(":", 1)[1].removeprefix("v")

    case = tmp_path / "docker-compose.yml"
    case.write_text("services:\n  qdrant:\n    image: qdrant/qdrant:1.19.1\n")
    assert (
        _pinned_qdrant_tag(case) == "1.19.1"
    )  # 不带 v 也解析得出来：交给上一条测试去拦

    case.write_text("services:\n  qdrant:\n    image: qdrant/qdrant\n")
    assert _pinned_qdrant_tag(case) is None  # 没 tag = 不比对，而不是拿 "latest" 去比


def test_parser_survives_a_broken_or_missing_file(tmp_path):
    assert _pinned_qdrant_tag(tmp_path / "nope.yml") is None
    bad = tmp_path / "docker-compose.yml"
    bad.write_text("services: [unclosed\n")
    assert _pinned_qdrant_tag(bad) is None
    bad.write_text("services:\n  redis:\n    image: redis:7\n")
    assert _pinned_qdrant_tag(bad) is None


@pytest.mark.parametrize("tag", ["latest", "1.19", "v1.19", "1.19.1"])
def test_the_rule_rejects_the_shapes_that_bite(tag):
    """把踩过的四种写法逐个钉死：漂移（latest）、不完整、缺前缀、缺 v。"""
    assert not re.fullmatch(r"v\d+\.\d+\.\d+", tag)

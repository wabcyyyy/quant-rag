"""测试期进程级护栏。

离线 mock 测试不许对开发机状态有副作用：`llm` 的响应缓存路径是模块级常量，
任何没 patch 它的用例都会把假答案写进真实的 `.cache/llm_cache.sqlite`
（实测漏进去过一条 `model=m / response=你好`）。那个文件是「重复评估近乎零成本」
的依据，被测试污染之后命中率与成本结论就都不干净了。

单个用例要测缓存本身时自己 `monkeypatch.setattr(llm, "_CACHE_PATH", tmp_path/...)`，
它在 fixture 之后执行，会覆盖这里的路径。
"""

from __future__ import annotations

import os
from unittest.mock import Mock

import pytest

# ragas 在 import 时就构造遥测 batcher（`_analytics.py` 模块级），此后每次判分都可能
# 向 t.explodinggradients.com POST。离线门禁不该有出网行为——本机装了 `--extra eval`
# 时那三个判分测试是真跑的。必须在 ragas 被导入之前设好，conftest 就是最早的位置。
# 用 setdefault：要观察 ragas 自身行为的人可以 externally 设成 false。
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


class FakeQdrant:
    """离线替身：只实现 `ingest/indexer.py` 用到的 scroll / delete / upsert。

    `points` 是预置的 doc_id 列表（一个 id 出现几次就是几块）；`deleted` 记每次删除
    命中的 doc_id，`upserted` 记每次 upsert 的块数。入库侧现在总要做一次反向对账
    （翻遍全库 payload），所以裸 `Mock()` 已经不够用了。
    """

    def __init__(self, doc_ids=()) -> None:
        self.points = list(doc_ids)
        self.deleted: list[list[str]] = []
        self.upserted: list[int] = []
        self.scroll_pages = 0

    def scroll(
        self, name, limit=None, offset=None, with_payload=None, with_vectors=None
    ):
        self.scroll_pages += 1
        start = int(offset) if offset else 0
        chunk = self.points[start : start + limit]
        nxt = start + limit
        return (
            [Mock(payload={"doc_id": d}) for d in chunk],
            str(nxt) if nxt < len(self.points) else None,
        )

    def delete(self, name, points_selector=None):
        match = points_selector.filter.must[0].match
        self.deleted.append(
            list(match.any) if hasattr(match, "any") else [match.value]  # type: ignore[attr-defined]
        )

    def upsert(self, name, points=None):
        self.upserted.append(len(points or []))


@pytest.fixture(autouse=True)
def _isolated_llm_cache(tmp_path, monkeypatch):
    from doc_rag.generate import llm as llm_mod

    monkeypatch.setattr(llm_mod, "_CACHE_PATH", tmp_path / "test-only-llm_cache.sqlite")

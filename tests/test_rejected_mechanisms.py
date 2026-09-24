"""A5：已否机制的默认值护栏（防后人误开）。

两个机制「已建、被自己数据否掉」（PLAN §5.7 已否机制台账）：
- `agent.enabled`：P2 实测与现基线不可区分（−2.75pt 在 3.1pt 地板内），代价 2.2 倍；
- `synthesis.two_stage.enabled`：ADR-0002 双门槛皆未过（+9.57/+13.93pt、CI 跨 0；
  干净栏 p95 仍 >8s），负结果收档、禁止迭代 prompt 挽救。

默认值翻 true = 悄悄换掉全部已发布基线的载体——这条测试就是那道闸。
「配置无未读键」由 tests/test_config_keys_are_wired.py 整体覆盖，不在此重复。
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _default_yaml() -> dict:
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text("utf-8"))


def test_agent_disabled_by_default():
    assert _default_yaml()["agent"]["enabled"] is False


def test_two_stage_disabled_by_default():
    assert _default_yaml()["synthesis"]["two_stage"]["enabled"] is False

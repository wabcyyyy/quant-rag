"""freeze 双指纹（B10，SPEC §6.2）：防止「跨基线状态并排报数」的技术保障。

两个指纹把「改动会作废什么」变成可机读的判定：
- `index_fp`（ingest 落盘，见 index_identity）：检索侧基线依赖它——
  Hit/Recall/MRR/nDCG/覆盖率族 + 上下文内容本身；
- `synth_fp`（本模块在冻结/评估时计算）：答案侧基线依赖它——prompt 指纹 +
  思考档 + 上下文预算 + 重排开关 + 合成/改写/判分模型 + gold sha256 + commit。

`doc-rag freeze` 把两者写进 `.cache/freeze_<id>.json`（id = 双指纹哈希前 12 位，
幂等：同状态重复 freeze 得到同一 id 与同一文件）。eval 的 meta 带上当前双指纹，
并对**只相关的那一个**报警：只改 prompt → 只报 `synth_fp` 不匹配（检索基线仍可
引用）；改了分块或嵌入 → 两个都报。裸检出（从未 freeze）不报警——没有冻结状态
可对照时，报警只会是噪音。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .config import project_root
from .generate import prompts
from .index_identity import identity_for

_FREEZE_DIR = project_root() / ".cache"


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=project_root(),
        ).stdout.strip()
    except Exception:  # noqa: BLE001 非 git 环境（如打包分发）不阻塞
        return "unknown"


def _host_of(url: str) -> str:
    return (url or "").split("//")[-1].split("/")[0]


def compute_synth_fp(cfg: dict, gold_file: Path) -> str:
    """答案侧状态指纹。模型记 host+model（同 host 不同 key 视为同身份）。"""
    llm_sec = cfg.get("llm") or {}
    judge = cfg.get("eval", {}).get("judge") or {}
    parts = {
        "prompt_fingerprint": prompts.fingerprint(None),
        "reasoning_effort": llm_sec.get("reasoning_effort") or "",
        "reasoning_effort_by_type": llm_sec.get("reasoning_effort_by_type") or {},
        "max_contexts": (cfg.get("retrieval") or {}).get("max_contexts"),
        "rerank_enabled": bool((cfg.get("rerank") or {}).get("enabled")),
        "rerank_top_n": (cfg.get("rerank") or {}).get("top_n"),
        "synth_model": f"{_host_of(llm_sec.get('base_url') or '')}/{llm_sec.get('model')}",
        "rewrite_model": f"{_host_of((cfg.get('rewrite') or {}).get('base_url') or '')}/"
        f"{(cfg.get('rewrite') or {}).get('model') or llm_sec.get('model')}",
        "judge_model": f"{_host_of(judge.get('base_url') or '')}/"
        f"{judge.get('model') or llm_sec.get('model')}",
        "gold_sha256": hashlib.sha256(Path(gold_file).read_bytes()).hexdigest()[:16]
        if Path(gold_file).exists()
        else "missing",
        "commit": _commit(),
    }
    blob = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def freeze(cfg: dict, collection: str, gold_file: Path) -> dict:
    """产出双指纹冻结记录并落盘（幂等）。要求先 ingest（index_fp 已登记）。"""
    index_fp = identity_for(collection)
    if not index_fp:
        raise SystemExit(
            f"collection「{collection}」没有库指纹——先对它 ingest 一次（B4 落盘后才能冻结）"
        )
    synth_fp = compute_synth_fp(cfg, gold_file)
    freeze_id = hashlib.sha256(f"{index_fp}:{synth_fp}".encode()).hexdigest()[:12]
    record = {
        "freeze_id": freeze_id,
        "index_fp": index_fp,
        "synth_fp": synth_fp,
        "collection": collection,
        "gold_file": str(gold_file),
        "gold_sha256": hashlib.sha256(Path(gold_file).read_bytes()).hexdigest()[:16]
        if Path(gold_file).exists()
        else None,
        "commit": _commit(),
    }
    _FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    out = _FREEZE_DIR / f"freeze_{freeze_id}.json"
    out.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    record["file"] = str(out)
    return record


def scan_freezes() -> list[dict]:
    out: list[dict] = []
    if not _FREEZE_DIR.exists():
        return out
    for p in sorted(_FREEZE_DIR.glob("freeze_*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 坏文件不阻塞评估
            continue
    return out


def freeze_warnings(
    current_index_fp: str,
    current_synth_fp: str,
    *,
    with_answers: bool,
) -> list[str]:
    """对**只相关的那一个**指纹报警。

    检索指标看 index_fp、答案指标看 synth_fp（SPEC §6.2 的依赖表）；
    两侧都没有任何 freeze 记录 → 静默（裸检出不吵）。
    """
    freezes = scan_freezes()
    if not freezes:
        return []
    warnings: list[str] = []
    index_ok = any(f.get("index_fp") == current_index_fp for f in freezes)
    synth_ok = any(f.get("synth_fp") == current_synth_fp for f in freezes)
    if not current_index_fp:
        warnings.append(
            "库指纹未登记（该 collection 从未在新代码下 ingest）——"
            "检索侧读数无法对照任何冻结状态"
        )
    elif not index_ok:
        warnings.append(
            "index_fp 与全部冻结记录不匹配——本轮检索读数不能与既有检索基线并排报数"
            "（L2 全作废口径）"
        )
    if with_answers:
        if not current_synth_fp:
            warnings.append("synth_fp 无法计算（缺 gold 或配置缺失）")
        elif not synth_ok:
            warnings.append(
                "synth_fp 与全部冻结记录不匹配——本轮答案侧读数不能与既有答案基线"
                "并排报数（L1 口径：检索基线仍可引用）"
            )
    return warnings

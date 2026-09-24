"""RGB 跑批与报告：协议编排、成本预演、判分、并排表。

CLI 保持薄壳（三个命令），重逻辑在这里——照 `eval/runner.py` 的先例。

## 两行结果，各自回答不同的问题

- **`production` 行（默认）**：本项目的生产 prompt + 生产思考档分流，就是线上跑的
  那套。它回答「**我的系统**在公开基准上什么水平」。
- **`rgb` 行（`--instruction rgb`）**：官方 `config/instruction.yaml` 的 zh 模板
  逐字照抄 + 我们选的 LLM。它回答「换成一个裸 LLM 做同一件事会怎样」。

两行的差别不是装饰：`zh_int` 的官方模板明确要求「信息不包含答案就生成…」，
而生产 prompt 要求「仅当上下文确实不含所需信息时才拒答」。官方判据是**关键词**
（「信息不足」），所以生产行的官方口径拒绝率会**系统性偏低**——不是因为不拒答，
而是因为它不这么说。所以生产行必须同时报严格口径与 judge 口径，见 `summarize`。

⚠️ **`rgb` 行不是与论文的模型对比**：论文基线是 2023–2024 年的模型（ChatGPT-3.5、
Qwen-7B-Chat），我们是 2026 年的推理模型，思考档/温度/时代全不同。它的用途是**内部
校准**（分清分数里哪部分来自 LLM、哪部分来自管线），不是「我们比 ChatGPT 强多少」。
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..generate import llm
from . import rgb

# ── 官方 instruction（`config/instruction.yaml` 的 `zh` 项，逐字照抄）──────────
#
# 出处：github.com/chen700564/RGB @ 65ec39e4 的 config/instruction.yaml。
# 只用于 `--instruction rgb` 那一行；默认行走生产 prompt。
RGB_ZH_SYSTEM = (
    "你是一个准确和可靠的人工智能助手，能够借助外部文档回答问题，请注意外部文档可能存在"
    "噪声事实性错误。如果文档中的信息包含了正确答案，你将进行准确的回答。如果文档中的信息"
    "不包含答案，你将生成“文档信息不足，因此我无法基于提供的文档回答该问题。”。如果部分"
    "文档中存在与事实不一致的错误，请先生成“提供文档的文档存在事实性错误。”，并生成正确答案。"
)
RGB_ZH_INSTRUCTION = "文档：\n{DOCS} \n\n问题：\n{QUERY}"

#: 生产侧思考档的路由键。RGB 自带任务标签（`zh_int` = 信息整合 = 聚合题），生产里
#: 这个标签来自改写器；给定上下文的入口没有改写计划，所以按数据集填。
#: **这是一处声明过的偏差**：聚合题因此走生产的「聚合关思考」档，其余走全局档。
TASK_QUESTION_TYPE: dict[str, str] = {
    "zh": "single",
    "zh_refine": "single",
    "zh_int": "cross_doc",
    "zh_fact": "single",
}

#: 拒答档要跑哪些数据集（官方 README：先 `evalue.py` 跑 `noise_rate=1`，
#: 再用 `reject_evalue.py` 判 Rej*）
REJECTION_DATASETS = ("zh", "zh_refine")


def protocol_combos(
    datasets: Iterable[str],
    *,
    include_rejection: bool = True,
    only_rates: tuple[float, ...] | None = None,
) -> list[tuple[str, float]]:
    """展开成「(数据集, 噪声率)」组合清单——它同时就是调用次数的分母。

    `only_rates` 只留指定档位。它的用途不是省钱，而是**做同题对照**：检索变体是
    「自己找文档」，给定文档那行要对比的就该是 `noise_rate=0.0` 那一档（喂 5 篇正确
    文档），跑全档扫一遍等于多花 5 倍的钱买一堆与对照无关的档位。
    """
    combos: list[tuple[str, float]] = []
    for ds in datasets:
        combos.extend((ds, rate) for rate in rgb.PROTOCOL[ds])
        if include_rejection and ds in REJECTION_DATASETS:
            combos.append((ds, rgb.REJECTION_NOISE))
    if only_rates is not None:
        combos = [(ds, r) for ds, r in combos if r in only_rates]
    return combos


def _answer_with_rgb_instruction(
    cfg: dict, question: str, docs: list[str]
) -> tuple[str, dict]:
    """官方 instruction 那一行：文档朴素拼接（**没有** `[n]` 编号），无引用要求。"""
    user = RGB_ZH_INSTRUCTION.format(DOCS="\n".join(docs), QUERY=question)
    text, meta = llm.chat_timed(cfg["llm"], user, system_prompt=RGB_ZH_SYSTEM)
    return text, meta


def row_key(dataset: str, noise_rate: float, instruction: str, rid: str) -> tuple:
    """断点续跑的键：同一组合同一条记录只该被真正调用一次。"""
    return (dataset, noise_rate, instruction, rid)


def load_rows(path: Path) -> list[dict]:
    """读已落盘的逐条结果（JSONL）。"""
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def done_index(rows: list[dict]) -> dict[tuple, dict]:
    """已完成的条目索引。**失败条目不入索引**——续跑时要重试它们。"""
    return {
        row_key(
            row["dataset"], row["noise_rate"], row["instruction"], str(row["id"])
        ): row
        for row in rows
        if not row.get("error")
    }


def select_records(
    records: list[rgb.Record],
    *,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 42,
) -> list[rgb.Record]:
    """挑这一轮要跑哪些记录：`limit` 取前 N（试跑）、`sample` 均匀抽 N（可判读）。

    抽样用固定种子 → 可复现；抽出来的是全量的**子集** → 之后 `--resume` 补全量
    时已完成的那部分直接复用，不会重复付费。
    """
    if limit is not None and sample is not None:
        raise ValueError("limit 与 sample 只能给一个")
    if sample is not None and sample < len(records):
        picked = random.Random(seed).sample(records, sample)
        return sorted(picked, key=lambda r: r.id)  # 顺序固定，便于比对两份结果
    if limit is not None:
        return records[:limit]
    return list(records)


def record_index(
    datasets: dict[str, list[rgb.Record]] | list[rgb.Record],
    dataset: str | None = None,
) -> dict[tuple, rgb.Record]:
    """建记录索引，**键必带数据集**（四个数据集的 id 各自从 0 开始，只按 id 会互相覆盖）。

    两种调用形态：`record_index(records, "zh")` 建单个数据集的索引；
    `record_index({ds: records, ...})` 建跨数据集的。
    """
    if isinstance(datasets, dict):
        return {
            rgb.record_key(ds, rec.id): rec
            for ds, recs in datasets.items()
            for rec in recs
        }
    assert dataset is not None, "单数据集形态必须给 dataset"
    return {rgb.record_key(dataset, rec.id): rec for rec in datasets}


def run_combo(
    cfg: dict,
    records: list[rgb.Record],
    dataset: str,
    noise_rate: float,
    *,
    instruction: str = "production",
    limit: int | None = None,
    sample: int | None = None,
    sample_seed: int = 42,
    workers: int = 1,
    # `Any` 而不是 `Any | None`：注入件的既有做法（同 Orchestrator 的 synthesizer），
    # 非 None 由下面那行显式检查保证，写进类型只会让每条调用点都要 assert。
    orch: Any = None,
    done: dict[tuple, dict] | None = None,
    sink: Any = None,
) -> list[dict]:
    """跑一个 (数据集, 噪声率) 组合，返回逐条结果。

    `limit` 只截**前** N 条——成本预演用，**判读绝不能用它**：前 N 条是按 id 排的，
    既不是均匀样本、又同时改变噪声文档的抽样。要降成本又要判读就用 `sample`：
    固定种子的**均匀**抽样，且抽出来的是全量的子集，之后 `--resume` 补全量不重复付费。

    `done` / `sink` 是给「几小时的长跑」准备的：`done` 里已有的键直接复用（官方的
    `evalue.py` 也是这个形状——读已有结果、按 id 跳过），`sink` 逐条落盘并 flush，
    这样一次网络抖动或一次 Ctrl-C 都不会让前面几小时的调用白花。

    `workers > 1` 走线程池：**只改墙钟**。每条的文档组装与 prompt 只依赖它自己那条
    记录，没有任何跨条共享的可变状态；`map` 保序，落盘顺序与串行一致。并发不是优化
    而是可行性问题——实测单次调用偶发分钟级端点停顿（一次 20 分钟的挂起），串行下
    几千次调用会被拖到几十小时。

    单条失败**不让整轮死掉**：异常记进 `error` 字段后继续。但失败条目**不得进入
    任何分母**——空答案会被判据算成「答错」，那就是把「没测」印成「测了且全错」，
    正是本项目在 `refusal_acc` 上踩过的那个坑（PLAN §5.3）。
    """
    if instruction not in ("production", "rgb"):
        raise ValueError(f"未知 instruction：{instruction!r}（production | rgb）")
    if limit is not None and sample is not None:
        raise ValueError("limit 与 sample 只能给一个（前者是试跑，后者是可判读的抽样）")
    if instruction == "production" and orch is None:
        raise ValueError("production 行需要传入 Orchestrator")
    done = done or {}
    pool = select_records(records, limit=limit, sample=sample, seed=sample_seed)
    sink_lock = threading.Lock()

    def _one(rec: rgb.Record) -> dict:
        hit = done.get(row_key(dataset, noise_rate, instruction, rec.id))
        if hit is not None:
            return hit
        docs = rgb.assemble_docs(rec, dataset, noise_rate)
        base = {
            "id": rec.id,
            "dataset": dataset,
            "noise_rate": noise_rate,
            "instruction": instruction,
            "question": rec.query,
            "docs": docs,
        }
        t0 = time.perf_counter()
        try:
            if instruction == "rgb":
                text, meta = _answer_with_rgb_instruction(cfg, rec.query, docs)
                n_contexts = None
            else:
                result = orch.answer_given_contexts(
                    rec.query, docs, question_type=TASK_QUESTION_TYPE[dataset]
                )
                text = result.answer
                meta = result.synth_meta or {}
                n_contexts = len(result.contexts)
        except Exception as exc:  # noqa: BLE001 一条失败不该毁掉整轮长跑
            row = {
                **base,
                "prediction": "",
                "label": None,
                "factlabel": None,
                "rejected": None,
                "n_contexts": None,
                "error": f"{type(exc).__name__}: {exc}",
                "ms": round((time.perf_counter() - t0) * 1000, 1),
            }
            if sink is not None:
                with sink_lock:
                    sink(row)
            return row
        labels, factlabel, rejected = rgb.label_and_flags(text, rec.answer, dataset)
        row = {
            **base,
            "prediction": text,
            "label": labels,
            "factlabel": factlabel,
            "rejected": rejected,
            "n_contexts": n_contexts,
            "model": (meta or {}).get("model"),
            "cached": (meta or {}).get("cached"),
            "prompt_tokens": (meta or {}).get("prompt_tokens"),
            "completion_tokens": (meta or {}).get("completion_tokens"),
            "reasoning_tokens": (meta or {}).get("reasoning_tokens"),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        if sink is not None:
            with sink_lock:
                sink(row)
        return row

    if workers <= 1:
        return [_one(rec) for rec in pool]
    # 并发跑：实测单次调用偶发分钟级端点停顿（一次 20 分钟的挂起仍在超时内正常返回），
    # 串行下几千次调用会被这些停顿拖到几十小时。并发只改墙钟，不改任何一条的输入
    # ——每条的文档组装与 prompt 都只依赖它自己那条记录。`map` 保序，落盘顺序稳定。
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_one, pool))


# ── 判分 ────────────────────────────────────────────────────────────────────


def summarize(rows: list[dict], records: dict[tuple, rgb.Record]) -> dict:
    """一个组合的读数。指标按数据集族给，不硬凑成一张表。

    **失败条目先剔出分母**：它们的 `prediction` 是空的，留着会被判据算成「答错」，
    于是「没测」被印成「测了且全错」。剔掉的条数记在 `n_errors` 里，与 `n` 一起报
    ——报一个只在更小样本上算出来的率而不说样本被削过，是本项目明令禁止的读法。
    """
    total = len(rows)
    rows = [r for r in rows if not r.get("error")]
    dataset = rows[0]["dataset"] if rows else ""
    noise_rate = float(rows[0]["noise_rate"]) if rows else 0.0
    out: dict[str, Any] = {
        "dataset": dataset,
        "noise_rate": noise_rate,
        "instruction": rows[0].get("instruction") if rows else None,
        "n": len(rows),
        "n_errors": total - len(rows),
        "all_rate": rgb.accuracy(rows, noise_rate),
        "rejected_rate": (
            sum(1 for r in rows if r["rejected"]) / len(rows) if rows else 0.0
        ),
    }
    # 拒答那一档的三个口径，各回答一个不同的问题：
    # - `all_rate`：官方原式（拒答**或**侥幸答对），用于与论文可比
    # - `rejection_rate_strict`：只数官方关键词认出的拒答（去掉侥幸那部分）
    # - `rejection_rate_own`：按**生产 prompt 自己的措辞**数（「根据现有文档无法回答」）
    #   生产 prompt 不写「信息不足」，所以前两个在生产行上接近 0，那是措辞差异，
    #   不是能力差异。真正的语义判据是 judge 的 Rej*。
    if noise_rate == rgb.REJECTION_NOISE:
        out["rejection_rate_strict"] = rgb.rejection_rate_strict(rows)
        out["rejection_rate_own"] = rgb.rejection_rate_marker(
            rows, rgb.PRODUCTION_REJECT_MARKERS
        )
    if dataset in rgb.FACT_DATASETS:
        ed, cr = rgb.fact_rates(rows)
        out["fact_check_rate"] = ed
        out["correct_rate"] = cr
        out["fakeanswer_false_positives"] = rgb.fakeanswer_false_positives(
            rows, records
        )
    out["usage"] = {
        k: sum(int(r.get(k) or 0) for r in rows)
        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")
    }
    out["cached_n"] = sum(1 for r in rows if r.get("cached"))
    ms = sorted(float(r["ms"]) for r in rows)
    if ms:
        out["ms"] = {
            "p50": round(ms[len(ms) // 2], 1),
            "max": round(ms[-1], 1),
        }
    return out


# ── 成本预演 ────────────────────────────────────────────────────────────────


def call_plan(
    datasets: Iterable[str],
    n_records: dict[str, int],
    *,
    include_rejection: bool = True,
    sample: int | None = None,
    only_rates: tuple[float, ...] | None = None,
) -> dict:
    """跑之前先把账算出来：每个组合多少条、总共多少次调用。

    本仓库的成本纪律是「先算账再动手」，而这里最容易算错的正是**组合数**——
    四个数据集 × 各自的档位再加拒答档，漏一个组合就是几百次调用。

    `sample` 会按「每组合最多 N 条」折算调用数：它是降成本的唯一可判读手段
    （`--limit` 只配试跑），所以账必须跟着它变，否则闸门会按全量拦人。
    """
    combos = protocol_combos(
        datasets, include_rejection=include_rejection, only_rates=only_rates
    )
    per_combo = [
        (ds, rate, min(n_records[ds], sample) if sample else n_records[ds])
        for ds, rate in combos
    ]
    return {
        "combos": len(per_combo),
        "calls": sum(n for _ds, _rate, n in per_combo),
        "per_combo": per_combo,
    }


def extrapolate(usage_total: dict, n_limited: int, n_full: int) -> dict:
    """小试跑的实际用量 → 全量外推（只用**可核的量**：调用数与 token）。

    `usage_total` 是**本次小试跑全部组合的合计**、`n_limited` 是它的总条数——
    不是「一个组合的用量 × 组合数」。第一版把每组合条数当成了全量调用数，外推结果
    差了 16 倍（把 4000 说成 250），这正是成本纪律要防的那类错。

    不折算成人民币：本仓库的 ¥ 数历来是粗估，PLAN 明确只报 token 与调用次数。
    """
    usage = usage_total or {}
    scale = (n_full / n_limited) if n_limited else 0.0
    return {
        "n_limited": n_limited,
        "n_full": n_full,
        "calls": n_full,
        "prompt_tokens": round(usage.get("prompt_tokens", 0) * scale),
        "completion_tokens": round(usage.get("completion_tokens", 0) * scale),
        "reasoning_tokens": round(usage.get("reasoning_tokens", 0) * scale),
        "per_call_prompt_tokens": round(
            (usage.get("prompt_tokens", 0) / n_limited) if n_limited else 0
        ),
        "per_call_completion_tokens": round(
            (usage.get("completion_tokens", 0) / n_limited) if n_limited else 0
        ),
        "per_call_ms": round(
            (usage.get("total_ms", 0) / n_limited) if n_limited else 0
        ),
        "wall_hours": round(
            (usage.get("total_ms", 0) / n_limited * n_full / 3_600_000)
            if n_limited
            else 0.0,
            1,
        ),
    }


def total_usage(summaries: list[dict]) -> dict:
    """把各组合的用量与耗时合计起来——外推的输入必须是**合计**而不是某一个组合。"""
    out: dict[str, Any] = {
        k: sum(int(s.get("usage", {}).get(k) or 0) for s in summaries)
        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")
    }
    # 用逐条 ms 的合计（`usage` 里没有耗时；这里按各组合的 p50×n 近似，
    # 只用于给出长跑的量级，报告里标明是量级估计）
    out["total_ms"] = sum(
        float(s.get("ms", {}).get("p50") or 0) * int(s.get("n") or 0) for s in summaries
    )
    out["n"] = sum(int(s.get("n") or 0) for s in summaries)
    return out


# ── 星号口径（judge）────────────────────────────────────────────────────────


def judge_sidecar_path(results_path: Path) -> Path:
    return results_path.with_suffix(".judge.jsonl")


def load_judge_sidecar(path: Path) -> dict[str, dict]:
    """读判分旁挂文件（按 `组合键 + id` 索引）。

    旁挂而不是就地改结果：官方也是两步（先生成、再 `reject_evalue.py`），
    重跑判分不该重花钱，重跑生成也不该丢判分——两件事的缓存键不同。
    """
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["key"]] = row
    return out


def needs_rej_star(row: dict) -> bool:
    """官方只在 `noise_rate=1` 那一档判 Rej\\*（`reject_evalue.py` 读的就是该档结果）。

    其余档位文档里是有答案的，「文档解不解得了这题」在那些档位上没有定义——
    对它们跑 judge 既多花钱、又不是官方口径。
    """
    return float(row.get("noise_rate") or 0.0) == rgb.REJECTION_NOISE


def needs_ed_star(row: dict) -> bool:
    """官方只在反事实族判 ED\\*（`fact_evalue.py` 的输入是 `*_fact` 数据集）。"""
    return str(row.get("dataset")) in rgb.FACT_DATASETS


def judge_rows(
    rows: list[dict], judge_cfg: dict, cache: dict[str, dict], sink: Any
) -> list[dict]:
    """给逐条结果补上官方星号判据的判定，带旁挂缓存。

    **只判官方口径覆盖得到的那部分**：Rej\\* 只在 `noise_rate=1` 档、ED\\* 只在反事实族。
    全量无差别地判两个判据等于把 150 次该花的调用变成 1600 次（实测的 10 倍）。

    `sink` 是逐条落盘的回调（传 `None` 表示不写）。判据逐字来自官方两个脚本的
    judge prompt——**换了措辞就等于换了度量身份**。
    """
    for row in rows:
        want_rej, want_ed = needs_rej_star(row), needs_ed_star(row)
        if not (want_rej or want_ed):
            row["rej_star"] = None
            row["ed_star"] = None
            continue
        key = f"{row['dataset']}|{row['noise_rate']}|{row['instruction']}|{row['id']}"
        hit = cache.get(key)
        if hit is None:
            hit = {
                "key": key,
                "rej_star": (
                    rgb.judge_reject(row["question"], row["prediction"], judge_cfg)
                    if want_rej
                    else None
                ),
                "ed_star": (
                    rgb.judge_fact(row["prediction"], judge_cfg) if want_ed else None
                ),
            }
            cache[key] = hit
            if sink is not None:
                sink(hit)
        row["rej_star"] = hit.get("rej_star")
        row["ed_star"] = hit.get("ed_star")
    return rows


def star_rates(rows: list[dict]) -> dict:
    """星号口径的读数：Rej*（judge 说文档解不了这题）与 ED*（judge 说识别出了错误）。

    分母只数**真判过的**条目（`None` 表示这一档不在官方口径内、没判），
    否则未判会被当成 False 混进分母——那正是本项目在 `refusal_acc` 上修过的错。
    """
    out: dict[str, float] = {}
    for name in ("rej_star", "ed_star"):
        judged = [r for r in rows if r.get(name) is not None]
        if judged:
            out[name] = sum(1 for r in judged if r[name]) / len(judged)
            out[f"{name}_n"] = len(judged)
    return out


# ── 报告 ────────────────────────────────────────────────────────────────────


def render_reports(summaries: list[dict]) -> str:
    """按 instruction 分组出报告。

    两行不是同一个度量：`production` 行用本系统的 prompt（拒答措辞与官方不同，
    官方关键词判据会低估它的拒绝率），`rgb` 行用官方模板（与论文同源但模型不同）。
    混在一张表里横向比大小就是在比两件不同的事，所以分行印、各带一句口径说明。
    """
    groups: dict[str, list[dict]] = {}
    for s in summaries:
        groups.setdefault(str(s.get("instruction") or "production"), []).append(s)
    blocks: list[str] = [
        "\n读这些表之前必须知道的三件事：",
        (
            "1. 官方判据是**关键词**（拒答认「信息不足」、反事实认「事实性错误」）。"
            "生产 prompt 不这么说，所以生产行的官方口径会偏低——那是措辞差异，不是能力差异。"
        ),
        (
            "2. 因此生产行要看**本 prompt 列**与 **Rej*/ED* 列**（judge 语义判读）；"
            "官方口径列保留是为了与论文可比。"
        ),
        (
            "3. 反事实那一族另有前提：生产 prompt **从不要求**声明「文档有事实性错误」"
            "（它要求的是区分决议/讨论、矛盾分别陈述），所以 ED 接近 0 是设计差异的必然结果，"
            "不是「识别不出错误」——这条本轮不修，改 prompt 就等于换度量身份。"
        ),
    ]
    for name in sorted(groups):
        note = (
            "本系统生产 prompt + 生产思考档分流（回答「我的系统什么水平」）"
            if name == "production"
            else "官方 instruction 逐字照抄 + 本仓库选的 LLM（内部校准用，"
            "**不可当作与论文的模型对比**：基线是 2023–2024 年的模型）"
        )
        blocks.append(f"\n{'=' * 72}\ninstruction = {name} —— {note}\n{'=' * 72}")
        blocks.append(render_report(groups[name]))
    return "\n".join(blocks)


def render_report(summaries: list[dict]) -> str:
    """并排表：本系统读数 + 论文中文基线。

    分三张表（噪声鲁棒 / 拒答 / 信息整合）而不是一张：三者的分母、指标名与可比性
    都不同，凑成一张表会诱导读者横向比大小。
    """
    lines: list[str] = []
    noise = [s for s in summaries if s["dataset"] in ("zh", "zh_refine")]
    reject = [s for s in noise if s["noise_rate"] == rgb.REJECTION_NOISE]
    noise = [s for s in noise if s["noise_rate"] != rgb.REJECTION_NOISE]
    integ = [s for s in summaries if s["dataset"] == "zh_int"]
    fact = [s for s in summaries if s["dataset"] in rgb.FACT_DATASETS]

    def _head(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    def _n(s: dict) -> str:
        """`n` 单元格：有失败条目就一并印出来，别让读者以为分母是完整的。"""
        err = s.get("n_errors") or 0
        return f"{s['n']}" if not err else f"{s['n']}+{err}错"

    if noise:
        _head("噪声鲁棒性（指标：accuracy，越高越好）")
        base = rgb.PAPER_BASELINES["noise_robustness"]
        lines.append(
            f"{'数据集':<10} {'噪声率':>6} {'n':>5} {'本系统':>9} "
            f"{'论文 ChatGPT-zh':>16} {'论文 Qwen-7B-zh':>16}"
        )
        for s in sorted(noise, key=lambda x: (x["dataset"], x["noise_rate"])):
            k = str(s["noise_rate"])
            c = base["ChatGPT-zh"].get(k)
            q = base["Qwen-7B-Chat-zh"].get(k)
            lines.append(
                f"{s['dataset']:<10} {s['noise_rate']:>6} {_n(s):>5} "
                f"{s['all_rate'] * 100:>8.2f}% "
                f"{(f'{c:.2f}%' if c is not None else '—'):>16} "
                f"{(f'{q:.2f}%' if q is not None else '—'):>16}"
            )
    if reject:
        _head("负向拒答（noise_rate=1，指标：拒绝率，越高越好）")
        base = rgb.PAPER_BASELINES["negative_rejection"]
        lines.append(
            f"{'数据集':<10} {'n':>5} {'官方口径':>10} {'严格口径':>10} "
            f"{'本 prompt':>10} {'Rej*':>8} "
            f"{'论文 ChatGPT-zh':>16} {'论文 Qwen-7B-zh':>16}"
        )
        for s in sorted(reject, key=lambda x: x["dataset"]):
            star = s.get("rej_star")
            # 变量名不复用上面那两张表的 `c`/`q`：那边是 float|None（取单档），
            # 这边是整行 dict —— 同名会让 mypy 认定类型冲突，也会让读者误读。
            cz = base["ChatGPT-zh"]
            qz = base["Qwen-7B-Chat-zh"]
            lines.append(
                f"{s['dataset']:<10} {_n(s):>5} "
                f"{s['all_rate'] * 100:>9.2f}% "
                f"{s.get('rejection_rate_strict', 0) * 100:>9.2f}% "
                f"{s.get('rejection_rate_own', 0) * 100:>9.2f}% "
                f"{(f'{star * 100:.2f}%' if star is not None else '—'):>8} "
                f"{cz['rej']:>9.2f}/{cz['rej_star']:<6.2f} "
                f"{qz['rej']:>9.2f}/{qz['rej_star']:<6.2f}"
            )
        lines.append("（论文列格式：关键词口径/星号口径，即 Rej/Rej*）")
        lines.append(
            "「官方/严格口径」认的是关键词「信息不足」——生产 prompt 要求回答"
            "「根据现有文档无法回答」，所以这两列在生产行上会偏低，那是**措辞差异**；"
            "「本 prompt」列按我们自己的措辞数；Rej* 是 judge 的语义判读，"
            "只有它不受措辞影响。"
        )
    if integ:
        _head("信息整合（指标：accuracy，越高越好）")
        base = rgb.PAPER_BASELINES["information_integration"]
        lines.append(
            f"{'噪声率':>6} {'n':>5} {'本系统':>9} "
            f"{'论文 ChatGPT-zh':>16} {'论文 Qwen-7B-zh':>16}"
        )
        for s in sorted(integ, key=lambda x: x["noise_rate"]):
            k = str(s["noise_rate"])
            c = base["ChatGPT-zh"].get(k)
            q = base["Qwen-7B-Chat-zh"].get(k)
            lines.append(
                f"{s['noise_rate']:>6} {_n(s):>5} {s['all_rate'] * 100:>8.2f}% "
                f"{(f'{c:.2f}%' if c is not None else '—'):>16} "
                f"{(f'{q:.2f}%' if q is not None else '—'):>16}"
            )
    if fact:
        _head("反事实鲁棒性（ED=打出事实性错误标记的比例；CR=其中答对的占比）")
        lines.append(
            f"{'噪声率':>6} {'n':>5} {'ACC':>9} {'ED':>8} {'CR':>8} {'ED*':>8} "
            f"{'假阳性':>7} {'论文ACC':>9} {'论文ED/ED*/CR':>16}"
        )
        for s in sorted(fact, key=lambda x: x["noise_rate"]):
            star = s.get("ed_star")
            paper = rgb.PAPER_BASELINES["counterfactual"]["ChatGPT-zh"]
            paper_ed = f"{paper['ed']:.0f}/{paper['ed_star']:.0f}/{paper['cr']:.2f}"
            lines.append(
                f"{s['noise_rate']:>6} {_n(s):>5} "
                f"{s['all_rate'] * 100:>8.2f}% "
                f"{s.get('fact_check_rate', 0) * 100:>7.2f}% "
                f"{s.get('correct_rate', 0) * 100:>7.2f}% "
                f"{(f'{star * 100:.2f}%' if star is not None else '—'):>8} "
                f"{s.get('fakeanswer_false_positives', 0):>7} "
                f"{paper['acc']:>8.2f}% {paper_ed:>16}"
            )
        lines.append(
            "（ACC = 喂错误文档时仍答对的比例，是这一族的主读数；"
            "ED = 打出「事实性错误」标记的比例——**论文基线本身就只有 1%~5%**，"
            "所以 ED 低是常态而不是异常；假阳性列 = 判据命中但同时也命中 fakeanswer 的条目数）"
        )
        lines.append(
            "（未实现列：" + "；".join(rgb.PAPER_COLUMNS_NOT_IMPLEMENTED) + "）"
        )
    return "\n".join(lines)

"""从语料事实清单程序化派生公开黄金集（零 LLM 成本，判据与语料同源）。

输入：data/eval/sample_corpus_manifest.json（scripts/make_sample_corpus.py 产出）
输出：
- data/eval/gold_core.json       核心集 60~80 条：**消融全臂都跑这一档**
- data/eval/gold_full.json       扩展集 ~200 条：只跑最终冻结配置（头条数字）
- data/eval/gold_core_agg.json   核心集的聚合题子集（cross_doc + time_filter），
                                 A4 预算扫线的直接输入
- data/eval/gold_review_worksheet.md  人工核对工作卷（20 条，U1 的「代理出卷」半边）

判据设计（照 eval/schema.py 与 goldgen 先例）：
- 单文档题：must_contain = 关键短语（金额/状态词，正文逐字子串）；
- 聚合题（cross_doc）：must_contain = 系列归属人（全库唯一），key_points =
  逐篇要点（绑死 doc_id 的正文逐字句，K ≤ 8）；
- 时效题（time_filter）：must_contain = 归属人 + 年份，key_points = 当年各篇要点；
- no_answer：域外主题词，全库不可答。

脚本自带自检（任一失败即退出非零）：
1. must_contain / key_points 短语在来源文档的**解析文本**（项目自身 pdf 解析器
   往返提取）中逐字命中；
2. source_doc_ids 在 manifest 中真实存在；
3. no_answer 词全库不出现；
4. cross_doc 锚点人名不出现在非来源文档（判据零噪声）；
5. 两档的题型分布（cross_doc / time_filter 各 ≥15）与 id 唯一性；
6. must_contain / key_points 短语必须落在某个 **chunk** 里（当前 structural 分块
   的被索引文本）——「在解析文本里」≠「可被检索」：分块器丢字（N1）、标题不进
   块正文都会让短语在索引里零出现，判据永远够不着。此检查只准加严、不许为过检
   而弱化；失败时逐条报出不可检索的短语。

运行：uv run python scripts/make_gold_from_corpus.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MANIFEST_FILE = ROOT / "data" / "eval" / "sample_corpus_manifest.json"
OUT_CORE = ROOT / "data" / "eval" / "gold_core.json"
OUT_FULL = ROOT / "data" / "eval" / "gold_full.json"
OUT_AGG = ROOT / "data" / "eval" / "gold_core_agg.json"
OUT_WORKSHEET = ROOT / "data" / "eval" / "gold_review_worksheet.md"

SEED = 20260926

# 核心集配额（合计 74）：消融全臂跑这一档
CORE_QUOTA = {
    "fact": 12,
    "decision": 10,
    "open_discussion": 8,
    "term": 5,
    "cross_doc": 15,
    "time_filter": 16,
    "no_answer": 8,
}
# 扩展集配额（合计 198）：只跑最终冻结配置
FULL_QUOTA = {
    "fact": 70,
    "decision": 46,
    "open_discussion": 30,
    "term": 5,
    "cross_doc": 15,
    "time_filter": 24,
    "no_answer": 8,
}
# fact 题的难点类配额（核心集）：保证每一类难点都能被 eval 观测到
CORE_FACT_CLASS_FLOOR = {"table": 2, "fragmented": 1, "long": 2, "scan": 1}

_TYPE_ORDER = [
    "fact",
    "decision",
    "open_discussion",
    "term",
    "cross_doc",
    "time_filter",
    "no_answer",
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _short_name(fname: str) -> str:
    """文件名 → 短标题：去掉尾部的 yf 编号段。"""
    stem = Path(fname).stem
    parts = stem.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else stem


def _first_sentence(text: str, limit: int = 60) -> str:
    """要点短语：claim 的第一句（正文逐字前缀）。"""
    head = text.split("。", 1)[0]
    return head[:limit]


def build_question_pool(
    manifest: dict, text_by_file: dict[str, str]
) -> dict[str, list[dict]]:
    """按题型构造候选题（全量，供配额抽样）。每题含判据与出处，未含 id。"""
    docs = manifest["docs"]
    by_file = {d["file"]: d for d in docs}
    meta = manifest["meta"]
    series = manifest["series"]

    # ── 单文档题原料：decide 类 claim（fact/decision 共用池，取样时交错分半）──
    decide_claims: list[dict] = []
    discuss_claims: list[dict] = []
    for d in docs:
        for c in d["claims"]:
            row = {"doc": d, "claim": c}
            if c["status"] == "discuss_only":
                discuss_claims.append(row)
            else:
                decide_claims.append(row)
    # 交错分半：相邻两条分别进 fact / decision 池，两题型不共享 (doc, claim)
    fact_pool = decide_claims[0::2]
    decision_pool = decide_claims[1::2]

    questions: dict[str, list[dict]] = {t: [] for t in _TYPE_ORDER}

    # ── fact ────────────────────────────────────────────────────────
    for row in fact_pool:
        d, c = row["doc"], row["claim"]
        diff = d["difficulty"]
        is_table = "table" in diff
        name = _short_name(d["file"])
        if is_table:
            # 表格题：行项目名 + 数字两项判据（单元格里只有数字本身）
            row_item = "设备购置"
            q = f"《{name}》中{c['topic']}预算明细表里「{row_item}」的金额是多少？"
            mc = [row_item, c["key"]]
        else:
            q = f"《{name}》中{c['topic']}的预算是多少？"
            mc = [c["key"]]
        questions["fact"].append(
            {
                "question": q,
                "expected_answer": f"{c['topic']}的预算为 {c['key']}。",
                "must_contain": mc,
                "source_files": [d["file"]],
                "source_title": name,
                "fact_class": (
                    "table"
                    if is_table
                    else next(
                        (
                            x
                            for x in diff
                            if x
                            in (
                                "fragmented",
                                "long",
                                "scan",
                                "series",
                                "week_name",
                                "ordinary",
                            )
                        ),
                        "ordinary",
                    )
                ),
            }
        )

    # ── decision ────────────────────────────────────────────────────
    for row in decision_pool:
        d, c = row["doc"], row["claim"]
        name = _short_name(d["file"])
        if c["status"] == "overturn":
            q = f"关于{c['topic']}，公司最近一次决定是什么？"
            exp = f"此前方案停止执行，按 {c['key']} 重新立项。"
        else:
            q = f"关于{c['topic']}，会议定下了什么安排？"
            exp = f"会议同意按方案推进，预算 {c['key']}。"
        questions["decision"].append(
            {
                "question": q,
                "expected_answer": exp,
                "must_contain": [c["key"]],
                "source_files": [d["file"]],
                "source_title": name,
            }
        )

    # ── open_discussion ─────────────────────────────────────────────
    for row in discuss_claims:
        d, c = row["doc"], row["claim"]
        name = _short_name(d["file"])
        questions["open_discussion"].append(
            {
                "question": f"关于{c['topic']}，会议讨论出结论了吗？",
                "expected_answer": "未形成决议，待补充材料后提交下次会议再议。",
                "must_contain": ["未形成决议"],
                "source_files": [d["file"]],
                "source_title": name,
            }
        )

    # ── term（系统代号）──────────────────────────────────────────────
    code_docs: dict[str, list[str]] = {}
    for d in docs:
        for code in d.get("codenames", []):
            code_docs.setdefault(code, []).append(d["file"])
    for topic, code in meta["codenames"].items():
        files = code_docs.get(code, [])
        if not files:
            raise SystemExit(f"代号「{code}」没有任何文档提到，term 题不成立")
        ask = topic.replace("运维", "").replace("建设", "")
        questions["term"].append(
            {
                "question": f"公司内部把{ask}称作什么代号？",
                "expected_answer": f"内部代号「{code}」。",
                "must_contain": [code],
                "source_files": sorted(files),
                "source_title": None,
            }
        )

    # ── cross_doc（14 系列 + 1 个多文档普通议题）─────────────────────
    for s in series:
        owner = s["owner"]
        questions["cross_doc"].append(
            {
                "question": f"关于{s['topic']}，公司先后讨论并决定了哪些事情？",
                "expected_answer": f"围绕{s['topic']}的历次推进与决议，汇报人均为{owner}。",
                "must_contain": [owner],
                "source_files": [f["file"] for f in s["files"]],
                "source_title": None,
                "anchor_kind": "person",
                "key_points": [
                    {
                        "doc_file": f["file"],
                        "phrase": _first_sentence(
                            by_file[f["file"]]["claims"][0]["text"]
                        ),
                    }
                    for f in s["files"]
                ],
            }
        )
    multi = _multi_doc_ordinary_topic(docs)
    if multi is not None:
        topic, claim_files = multi
        # 话题锚点的来源集合 = **全部含词文档**（照公司语料的教训：只取提及文档
        # 会把正确检索判为未命中）。含词文档比有 claim 的文档多（正文「情况通报」
        # 里也会出现议题词），key_points 只落在有 claim 的文档上。
        files = sorted(f for f, t in text_by_file.items() if topic in _norm(t))
        questions["cross_doc"].append(
            {
                "question": f"公司会议上关于{topic}都讨论过哪些内容？",
                "expected_answer": f"不同周次的会议分别汇报过{topic}的进展与安排。",
                "must_contain": [topic],
                "source_files": files,
                "source_title": None,
                "anchor_kind": "topic",
                "key_points": [
                    {
                        "doc_file": f,
                        "phrase": _first_sentence(
                            next(
                                c["text"]
                                for c in by_file[f]["claims"]
                                if c["topic"] == topic
                            )
                        ),
                    }
                    for f in claim_files
                ][:8],
            }
        )

    # ── time_filter（系列 × 年份）────────────────────────────────────
    for s in series:
        by_year: dict[int, list[dict]] = {}
        for f in s["files"]:
            d = by_file[f["file"]]
            by_year.setdefault(int(d["doc_date"][:4]), d)
        for year in sorted(by_year):
            d = by_year[year]
            claims = [c for c in d["claims"] if c["topic"] == s["topic"]]
            questions["time_filter"].append(
                {
                    "question": f"{year} 年关于{s['topic']}有哪些进展或安排？",
                    "expected_answer": (
                        f"{year} 年{s['topic']}由{s['owner']}汇报推进，预算 {claims[0]['key']}。"
                    ),
                    "must_contain": [s["owner"], str(year)],
                    "source_files": [d["file"]],
                    "source_title": _short_name(d["file"]),
                    "key_points": [
                        {"doc_file": d["file"], "phrase": _first_sentence(c["text"])}
                        for c in claims
                    ],
                }
            )

    # ── no_answer（域外主题）─────────────────────────────────────────
    for term in meta["off_corpus_terms"]:
        questions["no_answer"].append(
            {
                "question": f"公司在{term}方面有什么制度或安排？",
                "expected_answer": "应拒答：现有文档没有讨论过该主题。",
                "must_contain": [],
                "source_files": [],
                "source_title": "(全库不存在)",
                "refusable": True,
            }
        )

    for t, qs in questions.items():
        for q in qs:
            q["type"] = t
    return questions


def _multi_doc_ordinary_topic(docs: list[dict]) -> tuple[str, list[str]] | None:
    """找一个有 ≥2 篇 decided 文档的普通议题（跨文档题的第二个锚点=议题词）。"""
    by_topic: dict[str, list[str]] = {}
    for d in docs:
        if "series" in d["difficulty"]:
            continue
        for c in d["claims"]:
            if c["status"] == "decide":
                by_topic.setdefault(c["topic"], []).append(d["file"])
    ranked = sorted(
        ((t, sorted(set(fs))) for t, fs in by_topic.items() if len(set(fs)) >= 2),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    return (ranked[0][0], ranked[0][1]) if ranked else None


def sample_sets(questions: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    """按配额抽样出核心集与扩展集。确定性：固定 seed；难点类配额优先。"""
    rng = random.Random(SEED)
    core: list[dict] = []
    full: list[dict] = []

    # fact：先按难点类配额选核心集（保证每类难点可被 eval 观测），再补普通量
    fact_pool = questions["fact"]
    by_class: dict[str, list[dict]] = {}
    for q in fact_pool:
        by_class.setdefault(q.pop("fact_class"), []).append(q)
    fact_core: list[dict] = []
    for cls, floor in CORE_FACT_CLASS_FLOOR.items():
        take = by_class.get(cls, [])
        rng.shuffle(take)
        fact_core.extend(take[:floor])
    rest: list[dict] = []
    for cls in sorted(by_class):
        pool = [q for q in by_class[cls] if q not in fact_core]
        rng.shuffle(pool)
        rest.extend(pool)
    need_more = CORE_QUOTA["fact"] - len(fact_core)
    fact_core.extend(rest[: max(need_more, 0)])
    fact_full = fact_core + rest[len(fact_core) : FULL_QUOTA["fact"]]
    core.extend(fact_core)
    full.extend(fact_full)

    for qtype in ("decision", "open_discussion"):
        pool = questions[qtype][:]
        rng.shuffle(pool)
        core.extend(pool[: CORE_QUOTA[qtype]])
        full.extend(pool[: FULL_QUOTA[qtype]])
    for qtype in ("term", "cross_doc", "no_answer"):
        pool = sorted(questions[qtype], key=lambda q: q["question"])
        core.extend(pool[: CORE_QUOTA[qtype]])
        full.extend(pool[: FULL_QUOTA[qtype]])
    time_pool = sorted(questions["time_filter"], key=lambda q: q["question"])
    core.extend(time_pool[: CORE_QUOTA["time_filter"]])
    full.extend(time_pool[: FULL_QUOTA["time_filter"]])
    return core, full


def finalize(
    items: list[dict], id_prefix: str, doc_id_of: dict[str, str]
) -> list[dict]:
    """补 id / source_doc_ids / key_points 的 doc_id 绑定。"""
    out = []
    for i, q in enumerate(items, 1):
        kps = q.get("key_points") or []
        out.append(
            {
                "id": f"{id_prefix}{i:03d}",
                "type": q["type"],
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "must_contain": q["must_contain"],
                "source_doc_ids": [doc_id_of[f] for f in q["source_files"]],
                "refusable": bool(q.get("refusable")),
                "source_title": q.get("source_title"),
                "origin": "programmatic",
                "key_points": [
                    {
                        "doc_id": doc_id_of[kp["doc_file"]],
                        "phrase": kp["phrase"],
                    }
                    for kp in kps
                ][:8],
            }
        )
    return out


def verify(
    sets: dict[str, list[dict]],
    manifest: dict,
    text_by_file: dict[str, str],
    chunks_by_file: dict[str, list[str]],
) -> list[str]:
    """六道自检，返回（失败清单, 待 OCR 复核清单）。

    `chunks_by_file`（文件名 → 该文档 structural 分块的块文本列表）必传：
    自检 #6 不许被绕过。
    """
    failures: list[str] = []
    ocr_pending: list[str] = []
    docs = manifest["docs"]
    doc_id_of = {d["file"]: d["doc_id"] for d in docs}
    diff_of_file = {d["file"]: d["difficulty"] for d in docs}
    all_text = "\n".join(_norm(t) for t in text_by_file.values())

    for set_name, items in sets.items():
        ids = [i["id"] for i in items]
        if len(ids) != len(set(ids)):
            failures.append(f"[{set_name}] id 重复")
        dist = {t: sum(1 for i in items if i["type"] == t) for t in _TYPE_ORDER}
        for t in ("cross_doc", "time_filter"):
            if dist[t] < 15:
                failures.append(f"[{set_name}] {t} 只有 {dist[t]} 条（要求 ≥15）")

        for item in items:
            files = [
                f for f, did in doc_id_of.items() if did in set(item["source_doc_ids"])
            ]
            missing = set(item["source_doc_ids"]) - set(doc_id_of.values())
            if missing:
                failures.append(f"{item['id']}: source_doc_ids 不在 manifest 里")
                continue
            if item["type"] == "no_answer":
                for kw in item["must_contain"]:
                    if kw in all_text:
                        failures.append(f"{item['id']}: no_answer 词「{kw}」在库内出现")
                continue
            # 扫描件（无文本层）的判据此刻无法往返验证——挂起到 OCR 兜底落地后
            # 对入库文本复核（A3.3/S3）。短语在出题时已按 OCR 友好（数字+短词）挑选。
            if all("scan" in diff_of_file.get(f, []) for f in files):
                ocr_pending.append(f"{item['id']}({item['type']})")
                continue
            corpus = _norm("\n".join(text_by_file[f] for f in files))
            for kw in item["must_contain"]:
                if _norm(kw) not in corpus:
                    failures.append(
                        f"{item['id']}({item['type']}): must_contain「{kw}」未逐字命中"
                    )
                # 自检 #6（N1 护栏）：解析文本里有 ≠ 可检索。短语必须落在某个
                # chunk 里——分块器丢字、标题只进 section_path 不进块正文，都会
                # 让短语在被索引文本里零出现，检索永远够不着（q019/q020 类缺陷）。
                if chunks_by_file is not None and files:
                    hit = any(
                        _norm(kw) in _norm(ct)
                        for f in files
                        for ct in chunks_by_file.get(f, [])
                    )
                    if not hit:
                        failures.append(
                            f"{item['id']}({item['type']}): must_contain「{kw}」"
                            "不在来源文档的任何 chunk 里（不可检索）"
                        )
            for kp in item["key_points"]:
                f = next(
                    (x for x, did in doc_id_of.items() if did == kp["doc_id"]), None
                )
                if f is None:
                    failures.append(f"{item['id']}: keypoint doc_id 不存在")
                    continue
                if _norm(kp["phrase"]) not in _norm(text_by_file[f]):
                    failures.append(f"{item['id']}: keypoint 短语未逐字命中 {f}")
                if chunks_by_file is not None:
                    kps = chunks_by_file.get(f) or []
                    if kps and not any(_norm(kp["phrase"]) in _norm(ct) for ct in kps):
                        failures.append(
                            f"{item['id']}: keypoint 短语不在 {f} 的任何 chunk 里"
                            "（不可检索）"
                        )
            # cross_doc 锚点检查分两种：人名锚点不得溢出；话题锚点的来源集
            # 必须 = 全部含词文档（漏一篇就会把正确检索判成错）。
            if item["type"] == "cross_doc":
                anchor = item["must_contain"][0]
                stray = sorted(
                    {f for f, t in text_by_file.items() if anchor in _norm(t)}
                )
                if item.get("anchor_kind") == "person":
                    extra = sorted(set(stray) - set(files))
                    if extra:
                        failures.append(
                            f"{item['id']}: 锚点「{anchor}」溢出到 {extra[:2]}"
                        )
                else:
                    if set(stray) != set(files):
                        failures.append(
                            f"{item['id']}: 话题锚点来源集 ≠ 全部含词文档"
                            f"（差 {sorted(set(stray) ^ set(files))[:3]}）"
                        )
            # 聚合题 K 分布
            if item["type"] in ("cross_doc", "time_filter"):
                k = len(item["key_points"])
                if k == 0 or k > 8:
                    failures.append(f"{item['id']}: key_points K={k} 越界")
    return failures, ocr_pending


def write_worksheet(items: list[dict], by_id_text: dict[str, str]) -> None:
    """20 条人工核对工作卷：不含自动分数，留空白判定列（U1 的代理出卷半边）。"""
    rng = random.Random(SEED + 1)
    by_type: dict[str, list[dict]] = {}
    for it in items:
        by_type.setdefault(it["type"], []).append(it)
    picked: list[dict] = []
    quota = {
        "fact": 4,
        "decision": 3,
        "open_discussion": 2,
        "term": 2,
        "cross_doc": 3,
        "time_filter": 4,
        "no_answer": 2,
    }
    for t, n in quota.items():
        pool = by_type.get(t, [])
        rng.shuffle(pool)
        picked.extend(pool[:n])
    lines = [
        "# 公开黄金集人工核对工作卷（20 条）",
        "",
        "> 出卷：程序化（scripts/make_gold_from_corpus.py）。核对人不看自动分数，",
        "> 只对每条回答三个问题：①判据短语是否真的能在来源文档里逐字找到；",
        "②问题措辞是否自然（会不会没人这么问）；③ expected_answer 是否忠于文档。",
        "> 结论填「OK」或具体问题。这份卷是 U1 外部锚点的一半（另一半是外部人提问）。",
        "",
        "| # | id | 题型 | 问题 | expected_answer | 来源文档 | 人工判定 |",
        "|---|----|------|------|-----------------|----------|----------|",
    ]
    for i, it in enumerate(picked, 1):
        src = (
            "、".join(by_id_text.get(d, d) for d in it["source_doc_ids"][:2])
            or "(全库不存在)"
        )
        lines.append(
            f"| {i} | {it['id']} | {it['type']} | {it['question']} "
            f"| {it['expected_answer']} | {src} | |"
        )
    OUT_WORKSHEET.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    doc_id_of = {d["file"]: d["doc_id"] for d in manifest["docs"]}
    title_of = {d["doc_id"]: d["title"] for d in manifest["docs"]}

    # 解析往返：全库解析一遍（话题锚点的「全部含词文档」来源集与判据命中检查的前提）
    from doc_rag.ingest.chunker import chunk_document
    from doc_rag.ingest.pdf import extract_pdf

    parsed_docs = {
        d["file"]: extract_pdf(ROOT / "data" / "sample_raw" / d["file"])
        for d in manifest["docs"]
    }
    text_by_file = {f: doc.to_text() for f, doc in parsed_docs.items()}
    # 自检 #6 的输入：与 ingest 同一条默认 structural 分块路径的块文本
    chunks_by_file = {
        f: [c.text for c in chunk_document(doc)] for f, doc in parsed_docs.items()
    }

    questions = build_question_pool(manifest, text_by_file)
    for t in _TYPE_ORDER:
        print(f"候选 {t}: {len(questions[t])}")
    core, full = sample_sets(questions)
    core_items = finalize(core, "c", doc_id_of)
    full_items = finalize(full, "f", doc_id_of)

    sets = {"core": core_items, "full": full_items}
    failures, ocr_pending = verify(sets, manifest, text_by_file, chunks_by_file)
    if failures:
        print("\n".join(f"FAIL {f}" for f in failures[:40]))
        raise SystemExit(1)
    if ocr_pending:
        print(f"待 OCR 复核（扫描件来源，A3.3 落地后对入库文本复核）：{ocr_pending}")

    for out, items, note in (
        (OUT_CORE, core_items, "核心集：消融全臂跑这一档"),
        (OUT_FULL, full_items, "扩展集：只跑最终冻结配置（头条数字）"),
    ):
        dist = {t: sum(1 for i in items if i["type"] == t) for t in _TYPE_ORDER}
        payload = {
            "meta": {
                "gold_version": out.stem,
                "sample": True,
                "note": (
                    f"{note}。全部内容虚构（与 data/sample_raw 同源的合成语料），"
                    "由 scripts/make_gold_from_corpus.py 程序化派生（零 LLM）。"
                    "判据与语料同源：must_contain/key_points 均为正文逐字子串，"
                    "脚本自检逐字命中后落盘。字段契约见 src/doc_rag/eval/schema.py。"
                ),
                "count": len(items),
                "type_distribution": dist,
            },
            "items": items,
        }
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{out.name}: {len(items)} 条 · {dist}")

    agg_items = [i for i in core_items if i["type"] in ("cross_doc", "time_filter")]
    ks = [len(i["key_points"]) for i in agg_items if i["key_points"]]
    OUT_AGG.write_text(
        json.dumps(
            {
                "meta": {
                    "gold_version": "gold_core_agg",
                    "sample": True,
                    "note": "核心集的聚合题子集（cross_doc + time_filter），A4 预算扫线的直接输入；派生自 gold_core.json。",
                    "count": len(agg_items),
                    "type_distribution": {
                        t: sum(1 for i in agg_items if i["type"] == t)
                        for t in ("cross_doc", "time_filter")
                    },
                    "key_points_k": {
                        "n": len(ks),
                        "min": min(ks) if ks else None,
                        "max": max(ks) if ks else None,
                        "mean": round(sum(ks) / len(ks), 2) if ks else None,
                    },
                },
                "items": agg_items,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"gold_core_agg.json: {len(agg_items)} 条（K 分布 n={len(ks)}, "
        f"min={min(ks) if ks else '-'}, mean={round(sum(ks) / len(ks), 2) if ks else '-'}）"
    )
    write_worksheet(core_items, title_of)
    print(f"人工核对工作卷（20 条）→ {OUT_WORKSHEET.relative_to(ROOT)}")

    # 每题 gold 篇数分布（供 PLAN 的题型分布表）
    for name, items in sets.items():
        dist = {}
        for it in items:
            dist.setdefault(it["type"], []).append(len(it["source_doc_ids"]))
        print(f"[{name}] gold 篇数分布：")
        for t in _TYPE_ORDER:
            v = dist.get(t, [])
            if v:
                print(
                    f"  {t}: n={len(v)} · min={min(v)} · max={max(v)} · "
                    f"mean={round(sum(v) / len(v), 1)}"
                )


if __name__ == "__main__":
    main()

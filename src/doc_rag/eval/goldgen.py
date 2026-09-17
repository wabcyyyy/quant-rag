"""黄金集生成器（PLAN §5.3）：LLM 生成 + 程序化构造，固定种子可复现。

构造过程公开（PLAN §8 风险表「评估集太少被质疑」的对策）：
- 单文档题（fact / decision / open_discussion / term）：分层采样文档后由 LLM 生成，
  强制要求答案依据能在原文逐字找到（must_contain 从原文摘取）
- cross_doc：程序化找跨 ≥3 篇文档复现的关键词
- time_filter：年份 × 高频词组合
- no_answer：验证全库不存在的主题 → 测拒答
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from ..generate import llm
from ..ingest.metadata import parse_llm_json
from .schema import GoldItem

_PER_DOC_TYPES = ["fact", "decision", "open_discussion", "term"]

_GENERATE_PROMPT = """\
你在为公司文档 RAG 系统构造评估黄金集。下面是一份公司内部文档的全文。

文档标题：{title}

请构造 {n} 个「{type_label}」类型的问答对，要求：
{type_rules}
1. 问题必须包含文档中的专有信息（人名/数字/日期/项目名/系统名），保证不读文档的人无法凭常识回答
2. must_contain 是正确答案中必须出现的关键词列表（从原文逐字摘取，2~4 个）
3. expected_answer 控制在 80 字内，只依据原文，不得编造
4. 只输出 JSON 数组，每项字段：question, expected_answer, must_contain。不要输出其他内容

文档全文：
{document}
"""

_TYPE_RULES = {
    "fact": "问「某次会上说了/汇报了什么」，答案能在原文找到明确陈述。",
    "decision": "只挑「明确形成决议/决定」的事项，问「决定了什么」；原文没有明确决议就不要编。",
    "open_discussion": "只挑「讨论过但未形成决议/结论」的事项，问「X 最终定了什么/结论是什么」；"
    "expected_answer 必须说明「讨论了…但未形成决议」并引用原文表述，must_contain 需含原文中表示未决的词（如 未形成决议）。",
    "term": "围绕具体名称/编号/系统名/制度名提问，测精确词召回。",
}

_NO_ANSWER_CANDIDATES = [
    "报销标准", "年终奖发放", "远程办公制度", "股权激励", "年休假天数",
    "差旅费标准", "五险一金缴纳比例", "绩效考核等级",
]

_STOPWORDS = set(
    "会议 讨论 决定 汇报 跟进 安排 进行 相关 工作 内容 情况 问题 要求 完成 确认 通知 "
    "公司 项目 部分 以下 以上 今天 明天 昨天 时间 地点 人员 同志 各位 大家 继续 针对 "
    "目前 目前 目前 需要 可以 应该 已经 会上 关于 我们 他们 自己 很多 非常".split()
)


def _load_docs(parsed_dir: Path) -> list[dict]:
    docs = []
    for json_file in sorted(parsed_dir.glob("*.json")):
        if json_file.name == "profile.json":
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        title = data.get("meta", {}).get("title") or json_file.stem
        text = "\n".join(b.get("text", "") for b in data.get("blocks", []) if b.get("text"))
        if len(text.strip()) < 120:  # 空导出没有出题价值
            continue
        docs.append({"doc_id": data["meta"]["doc_id"], "title": title, "text": text})
    return docs


def _sample(docs: list[dict], per_doc: int, seed: int) -> list[dict]:
    """按大类分层采样，类别内均匀取。"""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for d in docs:
        cat = d["title"].split("_")[0] if "_" in d["title"] else "其他"
        groups.setdefault(cat, []).append(d)
    picked: list[dict] = []
    quota = {"会议档案": 10, "工作档案": 8, "议事档案": 6}
    for cat, n in quota.items():
        pool = groups.get(cat, [])
        picked += rng.sample(pool, min(n, len(pool)))
    # 不足配额时从剩余里补齐
    if len(picked) < 24:
        rest = [d for d in docs if d not in picked]
        picked += rng.sample(rest, min(24 - len(picked), len(rest)))
    rng.shuffle(picked)
    return picked[:24]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _gen_for_doc(doc: dict, qtype: str, llm_cfg: dict) -> list[GoldItem]:
    rules = _TYPE_RULES[qtype]
    prompt = _GENERATE_PROMPT.format(
        title=doc["title"],
        n=3,
        type_label=qtype,
        type_rules=rules,
        document=doc["text"][:2500],
    )
    try:
        reply = llm.chat(llm_cfg, prompt, temperature=0)
    except Exception:  # noqa: BLE001
        return []
    m = re.search(r"\[.*\]", reply, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    # grounding 自过滤：must_contain 必须逐字命中（空白归一化后的）标题+正文
    hay = _norm(doc["title"] + "\n" + doc["text"])
    items = []
    for a in arr if isinstance(arr, list) else []:
        if not isinstance(a, dict):
            continue
        q, ans = str(a.get("question", "")).strip(), str(a.get("expected_answer", "")).strip()
        mc = [str(x) for x in (a.get("must_contain") or []) if str(x).strip()]
        if len(q) < 10 or not ans or not mc:
            continue
        if not all(_norm(kw) in hay for kw in mc):
            continue  # 关键词不在原文 → 丢弃（LLM 幻觉或改写）
        items.append(
            GoldItem(
                id="", type=qtype, question=q, expected_answer=ans, must_contain=mc,
                source_doc_ids=[doc["doc_id"]], refusable=False, source_title=doc["title"],
            )
        )
        if len(items) >= 2:
            break
    return items


def _cross_doc_items(docs: list[dict], llm_cfg: dict) -> list[GoldItem]:
    """找跨 ≥3 篇复现且非常见词的关键词 → 聚合题。"""
    token_docs: dict[str, set[str]] = {}
    import jieba

    for d in docs:
        for tok in set(jieba.cut_for_search(d["text"][:2000])):
            tok = tok.strip()
            if len(tok) >= 2 and tok not in _STOPWORDS and not tok.isdigit():
                token_docs.setdefault(tok, set()).add(d["doc_id"])
    candidates = [
        tok for tok, ids in token_docs.items() if 3 <= len(ids) <= 6
    ]
    items = []
    for tok in candidates[:8]:
        ids = sorted(token_docs[tok])
        items.append(
            GoldItem(
                id="", type="cross_doc",
                question=f"关于「{tok}」，公司文档里出现过哪些讨论或安排？",
                expected_answer=f"散见于 {len(ids)} 篇文档，围绕「{tok}」有多次记录（聚合题，按检索命中评分）",
                must_contain=[tok],
                source_doc_ids=ids,
                refusable=False, source_title="(跨文档)",
            )
        )
    return items


def _time_items(docs: list[dict]) -> list[GoldItem]:
    """年份 × 该年文档中的高频实词 → 时间限定题。"""
    import jieba

    by_year: dict[str, list[dict]] = {}
    for d in docs:
        m = re.search(r"20\d{2}", d["title"])
        if m:
            by_year.setdefault(m.group(0), []).append(d)
    items = []
    for year, pool in sorted(by_year.items()):
        if len(pool) < 3:
            continue
        cnt: Counter = Counter()
        for d in pool:
            for tok in set(jieba.cut_for_search(d["text"][:1500])):
                tok = tok.strip()
                if len(tok) >= 2 and tok not in _STOPWORDS and not tok.isdigit():
                    cnt[tok] += 1
        for tok, _ in cnt.most_common(4):
            ids = [d["doc_id"] for d in pool if tok in d["text"]][:5]
            if len(ids) < 2:
                continue
            items.append(
                GoldItem(
                    id="", type="time_filter",
                    question=f"{year}年的文档中，关于「{tok}」有哪些记录？",
                    expected_answer=f"{year}年语料中「{tok}」相关内容（时间限定题，按检索命中评分）",
                    must_contain=[tok],
                    source_doc_ids=ids,
                    refusable=False, source_title=f"({year})",
                )
            )
            if len([i for i in items if i.type == "time_filter"]) >= 8:
                return items
    return items


def _no_answer_items(docs: list[dict]) -> list[GoldItem]:
    corpus_text = "\n".join(d["text"] for d in docs)
    items = []
    for topic in _NO_ANSWER_CANDIDATES:
        if topic in corpus_text:
            continue
        items.append(
            GoldItem(
                id="", type="no_answer",
                question=f"公司关于{topic}的制度或标准是什么？",
                expected_answer="应拒答：现有文档没有讨论过该主题",
                must_contain=[], source_doc_ids=[],
                refusable=True, source_title="(全库不存在)",
            )
        )
        if len(items) >= 8:
            break
    return items


def generate(parsed_dir: Path, out_file: Path, llm_cfg: dict, per_doc: int = 2, seed: int = 42) -> dict:
    docs = _load_docs(parsed_dir)
    sampled = _sample(docs, per_doc, seed)
    items: list[GoldItem] = []

    for i, doc in enumerate(sampled):
        qtype = _PER_DOC_TYPES[i % len(_PER_DOC_TYPES)]
        items.extend(_gen_for_doc(doc, qtype, llm_cfg))

    items.extend(_cross_doc_items(docs, llm_cfg))
    items.extend(_time_items(docs))
    items.extend(_no_answer_items(docs))

    # 去重 + 编号
    seen_q: set[str] = set()
    final: list[GoldItem] = []
    for item in items:
        key = item.question.strip()
        if key in seen_q:
            continue
        seen_q.add(key)
        item.id = f"q{len(final) + 1:03d}"
        final.append(item)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "seed": seed,
            "corpus_docs": len(docs),
            "sampled_docs": len(sampled),
            "count": len(final),
            "type_distribution": dict(Counter(i.type for i in final)),
        },
        "items": [i.model_dump() for i in final],
    }
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload["meta"]

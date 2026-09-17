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
    "目前 需要 可以 应该 已经 会上 关于 我们 他们 自己 很多 非常 "
    # 实测噪声（首版跨文档/时间题选词质量差，Phase 2 记录）
    "处理结果 本处 校正 文本处理 代码运行 case dta 议题 纪要 与会 本次 执行 owner 结果 "
    "记录 文档 文件 首页 未命名 信息 数据 系统 流程 管理 服务 支持 使用 建议 "
    # 会议纪要模板词（跨全库出现，无区分度）
    "提案 提案者 附议 附议区 决议 决议区 动议 动议区 辩论 辩论区 投票 投票区 元数据 元数据区 "
    "同意 否决 弃权 单选 实名 立即 后续 备注 说明 附件 版本 编号 目录 标题 正文 摘要".split()
)

# 人名实体抽取：只用高精度结构信号（实测零噪声，见 Phase 2 记录）
_NAME_CONTEXT_RES = (
    re.compile(r"@([\u4e00-\u9fa5]{2,4})[：:]"),  # 会议纪要「部门@姓名：内容」
    re.compile(r"提案者[：:]\s*([\u4e00-\u9fa5]{2,4})"),
    re.compile(r"主持[：:]\s*([\u4e00-\u9fa5]{2,4})"),
)


def _person_names(docs: list[dict]) -> dict[str, set[str]]:
    """人名实体 → {人名: 出现的 doc_id 集合}。

    交叉验证提纯：只认在 `@姓名：`（会议纪要发言人格式，最高精度）中出现过的名字；
    提案者/主持模式仅用于补充频次，不引入新名字（实测可滤掉「负责按照」类误报）。
    """
    at_re = _NAME_CONTEXT_RES[0]
    verified: set[str] = set()
    names: dict[str, set[str]] = {}
    for d in docs:
        for rx in _NAME_CONTEXT_RES:
            for m in rx.finditer(d["text"]):
                name = m.group(1)
                names.setdefault(name, set()).add(d["doc_id"])
                if rx is at_re:
                    verified.add(name)
    return {n: ids for n, ids in names.items() if n in verified}


def _cross_doc_items(docs: list[dict], llm_cfg: dict) -> list[GoldItem]:
    """跨 2~9 篇复现的人名 → 聚合题（人名有区分度，泛词与模板词已排除）。"""
    names = _person_names(docs)
    candidates = sorted(
        (n for n, ids in names.items() if 2 <= len(ids) <= 9),
        key=lambda n: (-len(names[n]), n),
    )
    items = []
    for name in candidates[:8]:
        ids = sorted(names[name])
        items.append(
            GoldItem(
                id="", type="cross_doc",
                question=f"关于「{name}」，公司文档里出现过哪些讨论或安排？",
                expected_answer=f"散见于 {len(ids)} 篇文档，围绕「{name}」有多次记录（聚合题，按检索命中评分）",
                must_contain=[name],
                source_doc_ids=ids,
                refusable=False, source_title="(跨文档)",
            )
        )
    return items


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


def _time_items(docs: list[dict]) -> list[GoldItem]:
    """年份 × 该年内跨 2~5 篇出现的人名 → 时间限定题。"""
    by_year: dict[str, list[dict]] = {}
    for d in docs:
        m = re.search(r"20\d{2}", d["title"])
        if m:
            by_year.setdefault(m.group(0), []).append(d)

    items = []
    for year, pool in sorted(by_year.items()):
        if len(pool) < 3:
            continue
        names = _person_names(pool)
        candidates = sorted(
            (n for n, ids in names.items() if 2 <= len(ids) <= 5),
            key=lambda n: (-len(names[n]), n),
        )
        for name in candidates[:4]:
            ids = sorted(names[name])
            items.append(
                GoldItem(
                    id="", type="time_filter",
                    question=f"{year}年的文档中，关于「{name}」有哪些记录？",
                    expected_answer=f"{year}年语料中「{name}」相关内容（时间限定题，按检索命中评分）",
                    must_contain=[name],
                    source_doc_ids=ids,
                    refusable=False, source_title=f"({year})",
                )
            )
            if len(items) >= 8:
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


def generate(
    parsed_dir: Path,
    out_file: Path,
    llm_cfg: dict,
    per_doc: int = 2,
    seed: int = 42,
    programmatic_only: bool = False,
) -> dict:
    """生成黄金集。

    programmatic_only=True：保留已有 LLM 题，只重算程序化题型（cross_doc /
    time_filter / no_answer）——改选词策略时零 LLM 成本（见 PLAN §8 成本控制）。
    """
    docs = _load_docs(parsed_dir)
    sampled = _sample(docs, per_doc, seed)

    kept: list[GoldItem] = []
    if programmatic_only:
        if not out_file.exists():
            raise FileNotFoundError(f"programmatic_only 需要已有黄金集：{out_file}")
        existing = json.loads(out_file.read_text(encoding="utf-8"))
        _PROGRAMMATIC = {"cross_doc", "time_filter", "no_answer"}
        kept = [
            GoldItem.model_validate(i)
            for i in existing["items"]
            if i["type"] not in _PROGRAMMATIC
        ]

    items: list[GoldItem] = list(kept)
    if not programmatic_only:
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

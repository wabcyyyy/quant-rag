"""黄金集生成器（PLAN §5.3）：LLM 生成 + 程序化构造，固定种子可复现。

构造过程公开（PLAN §8 风险表「评估集太少被质疑」的对策）：
- 单文档题（fact / decision / open_discussion / term）：分层采样文档后由 LLM 生成，
  强制要求答案依据能在原文逐字找到（must_contain 从原文摘取）
- cross_doc：程序化找跨 ≥3 篇文档复现的关键词
- time_filter：年份 × 高频词组合
- no_answer：验证全库不存在的主题 → 测拒答
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

from ..generate import llm
from .schema import GoldItem, KeyPoint, kp_normalize

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
    # 选材原则（Phase 2 实测教训）：不仅要「关键词不在语料」，还要**语义不相邻**。
    # 反例：问「员工持股计划」——关键词确实不存在，但语料有 ESOP/股权激励内容，
    # 题目实质可答；问「岗位职级体系」同理（语料有职级表）。
    # 故优先取公司内部会议纪要不可能覆盖的域外主题。
    "碳排放配额交易",
    "ISO14001环境认证",
    "对外担保额度",
    "内幕信息管理",
    "反垄断合规审查",
    "知识产权许可费",
    "董监高责任保险",
    "员工商业保险方案",
    "年终奖发放",
    "年休假天数",
    "五险一金缴纳比例",
    "加班补贴",
]

_STOPWORDS = {
    "会议",
    "讨论",
    "决定",
    "汇报",
    "跟进",
    "安排",
    "进行",
    "相关",
    "工作",
    "内容",
    "情况",
    "问题",
    "要求",
    "完成",
    "确认",
    "通知",
    "公司",
    "项目",
    "部分",
    "以下",
    "以上",
    "今天",
    "明天",
    "昨天",
    "时间",
    "地点",
    "人员",
    "同志",
    "各位",
    "大家",
    "继续",
    "针对",
    "目前",
    "需要",
    "可以",
    "应该",
    "已经",
    "会上",
    "关于",
    "我们",
    "他们",
    "自己",
    "很多",
    "非常",
    # 实测噪声（首版跨文档/时间题选词质量差，Phase 2 记录）
    "处理结果",
    "本处",
    "校正",
    "文本处理",
    "代码运行",
    "case",
    "dta",
    "议题",
    "纪要",
    "与会",
    "本次",
    "执行",
    "owner",
    "结果",
    "记录",
    "文档",
    "文件",
    "首页",
    "未命名",
    "信息",
    "数据",
    "系统",
    "流程",
    "管理",
    "服务",
    "支持",
    "使用",
    "建议",
    # 会议纪要模板词（跨全库出现，无区分度）
    "提案",
    "提案者",
    "附议",
    "附议区",
    "决议",
    "决议区",
    "动议",
    "动议区",
    "辩论",
    "辩论区",
    "投票",
    "投票区",
    "元数据",
    "元数据区",
    "同意",
    "否决",
    "弃权",
    "单选",
    "实名",
    "立即",
    "后续",
    "备注",
    "说明",
    "附件",
    "版本",
    "编号",
    "目录",
    "标题",
    "正文",
    "摘要",
}

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


def _docs_containing(docs: list[dict], needle: str) -> list[str]:
    """所有文本含该词的 doc_id（空白归一化）——聚合题的正确答案集合。"""
    n = _norm(needle)
    return sorted(d["doc_id"] for d in docs if n in _norm(d["text"]))


# ── 聚合题的逐篇要点（答案轨判据） ───────────────────────────────────────
#
# 为什么要有：cross_doc / time_filter 的 `must_contain` 只有 1 个词（就是那个人名），
# 而答案集 10~56 篇。答案是「提到过这个词」就算答对，所以「上下文 6 块答出 2 篇」
# 与「25 块答出 18 篇」在答案轨上同分——上下文预算消融拿不到读数。
# 一条要点绑一篇文档，命中它才等于把那一篇答出来。
#
# 三条判据，缺一条这道要点就没资格当判据：
# 1. **逐字来自该篇**（复用 `_gen_for_doc` 那条 grounding 纪律：不逐字就无法核对）；
# 2. **全库唯一**——只在这一篇出现。否则答案提一次同时命中多篇，覆盖率式的重复计分；
#    这条与 census 那次教训同源：日期行/序号列长得极像「可对齐的同一标签」，不剔就把
#    噪声读成信号；
# 3. **含本题实体**（人名），保证这篇的要点说的是这件事，不是任意一句正文。
#
# 分母用「等距抽 min(N, 8) 篇」而不是全量：56 篇的题要求逐篇答全，物理上写不进一个
# 几百字的答案，题间也没法比。抽 8 篇让所有聚合题共用同一个 K。
_K_MAX_DOCS = 8
_K_MIN_CHARS, _K_MAX_CHARS = 8, 25
_K_SEG_SPLIT = re.compile(r"[\n，。；：、,;:!！?？)）(（|\"“”]")
# 每个候选文档最多试几句：长文档一句一句扫全库会白付几千次子串查找
_K_CANDIDATE_LIMIT = 12
_KP_NAME = re.compile(r"@[\u4e00-\u9fa5]{2,4}")


def _kp_says_something(phrase: str) -> bool:
    """短语得在名字之外真的说件事。

    实测踩过的坑：`__@陈凡@康少云@黄日航@张果@田锃__` 这种发言人清单**全库唯一**，
    通得过唯一性判据，却什么都不主张——模型照抄它就是「答对」，认真转述反而不命中。
    唯一 ≠ 有信息量，这条是补那半边。

    判据只有一条：把 `@姓名` 全部摘掉后还剩 ≥6 个字才算「说了件事」。原先这里还有一
    条「@ 超过 3 个就拒」，突变验证时发现它**永远轮不到说话**（清单行的名字摘掉后
    本来就只剩 `__`），而它还会误杀「@甲@乙@丙三人负责机房巡检」这种真主张——删了。
    """
    return len(_KP_NAME.sub("", phrase).strip()) >= 6


def _kp_candidates(text: str, entity: str) -> list[str]:
    """该篇里所有「含实体、长度合规、且真的主张了件事」的候选短语。"""
    want = _norm(entity)
    out: list[str] = []
    for seg in _K_SEG_SPLIT.split(_clean_text(text)):
        phrase = _norm(seg)
        if (
            _K_MIN_CHARS <= len(phrase) <= _K_MAX_CHARS
            and want in phrase
            and _kp_says_something(phrase)
            and phrase not in out
        ):
            out.append(phrase)
    return out


class _KeyPointIndex:
    """全库归一化正文的只读索引：要点唯一性的判定依据，一次构建、多题复用。"""

    def __init__(self, docs: list[dict]) -> None:
        self._texts = {kp_normalize(d["text"]) for d in docs}
        # 同一篇可能被重复加载（同名 doc_id）；按篇数算唯一性时要按去重后的文本走
        self._n_docs = len(self._texts)

    def is_unique(self, phrase: str) -> bool:
        """全库只有 0 或 1 篇含这个短语。

        0 篇不可能（短语出自某篇），但判 `<= 1` 而不是 `== 1`：调用方给的短语若来自
        索引之外的文本（测试里注入），不该被判成「不唯一」而静默丢掉。
        """
        n = kp_normalize(phrase)
        return sum(1 for t in self._texts if n in t) <= 1

    @property
    def size(self) -> int:
        return self._n_docs


def _pick_unique_phrase(cands: list[str], index: _KeyPointIndex) -> str | None:
    """按「长的优先 → sha256 升序」取第一个全库唯一的候选句。

    排序必须是确定性的：同一份语料重跑要逐字得到同一批要点，否则黄金集本身不可复现。
    """
    ordered = sorted(
        cands, key=lambda p: (-len(p), hashlib.sha256(p.encode()).hexdigest())
    )
    for phrase in ordered[:_K_CANDIDATE_LIMIT]:
        if index.is_unique(phrase):
            return phrase
    return None


def _equal_pick(ids: list[str], k: int) -> list[str]:
    """已排序 doc_id 里等距取 k 篇（与 judge 抽样「均匀覆盖」同一个理由）。"""
    if not ids or k >= len(ids):
        return list(ids)
    step = len(ids) / k
    return [ids[int(i * step)] for i in range(k)]


def _key_points(
    docs: list[dict], src_ids: list[str], entity: str, index: _KeyPointIndex
) -> list[KeyPoint]:
    """给一道聚合题造逐篇要点；凑不出任何唯一句时返回空列表（不硬造判据）。"""
    chosen = _equal_pick(sorted(src_ids), _K_MAX_DOCS)
    text_by = {d["doc_id"]: d["text"] for d in docs}
    out: list[KeyPoint] = []
    for doc_id in chosen:
        text = text_by.get(doc_id)
        if not text:
            continue
        phrase = _pick_unique_phrase(_kp_candidates(text, entity), index)
        if phrase:
            out.append(KeyPoint(doc_id=doc_id, phrase=phrase))
    return out


def _cross_doc_items(
    docs: list[dict], llm_cfg: dict, index: _KeyPointIndex | None = None
) -> list[GoldItem]:
    """跨文档聚合题：人名有效（@ 验证）+ 全库 5~60 篇含名（有区分度且答案集完整）。

    v1 教训：来源集合若只取 @ 提及的文档会严重漏计（黄日航 @ 提及 8 篇，
    实际 205 篇含名），导致把正确检索判为未命中。聚合题的来源 = 全部含名文档。
    """
    names = _person_names(docs)
    if index is None:
        index = _KeyPointIndex(docs)
    scored: list[tuple[str, list[str]]] = []
    for name in names:
        src = _docs_containing(docs, name)
        if 5 <= len(src) <= 60:
            scored.append((name, src))
    scored.sort(key=lambda kv: -len(kv[1]))
    items = []
    for name, src in scored[:8]:
        items.append(
            GoldItem(
                id="",
                type="cross_doc",
                question=f"关于「{name}」，公司文档里出现过哪些讨论或安排？",
                expected_answer=f"散见于 {len(src)} 篇文档，围绕「{name}」有多次记录（聚合题，按检索命中评分）",
                must_contain=[name],
                source_doc_ids=src,
                key_points=_key_points(docs, src, name, index),
                refusable=False,
                source_title="(跨文档)",
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
        except Exception:  # noqa: BLE001, S112  # 坏导出直接跳过，不让单份文件中断整轮出题
            continue
        title = data.get("meta", {}).get("title") or json_file.stem
        text = "\n".join(
            b.get("text", "") for b in data.get("blocks", []) if b.get("text")
        )
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
    m = re.search(r"\[.*\]", reply, re.DOTALL)
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
        q, ans = (
            str(a.get("question", "")).strip(),
            str(a.get("expected_answer", "")).strip(),
        )
        mc = [str(x) for x in (a.get("must_contain") or []) if str(x).strip()]
        if len(q) < 10 or not ans or not mc:
            continue
        if not all(_norm(kw) in hay for kw in mc):
            continue  # 关键词不在原文 → 丢弃（LLM 幻觉或改写）
        items.append(
            GoldItem(
                id="",
                type=qtype,
                question=q,
                expected_answer=ans,
                must_contain=mc,
                source_doc_ids=[doc["doc_id"]],
                refusable=False,
                source_title=doc["title"],
            )
        )
        if len(items) >= 2:
            break
    return items


def _time_items(
    docs: list[dict], index: _KeyPointIndex | None = None
) -> list[GoldItem]:
    """时间限定题：年份 × 该年内 2~40 篇含名的人名（答案集完整、范围聚焦）。

    v2 放宽（黄金集补题）：答案集上限 15→40、每年 ≤4→≤6 条、总量 8→12——
    v1 只有 5 条（n<20 时 p95 基本等于 max，延迟与覆盖率数字置信度弱）。
    全部为程序化构造，零 LLM 成本。
    """
    if index is None:
        index = _KeyPointIndex(docs)
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
        scored: list[tuple[str, list[str]]] = []
        for name in names:
            src = _docs_containing(pool, name)
            if 2 <= len(src) <= 40:
                scored.append((name, src))
        scored.sort(key=lambda kv: -len(kv[1]))
        for name, src in scored[:6]:
            items.append(
                GoldItem(
                    id="",
                    type="time_filter",
                    question=f"{year}年的文档中，关于「{name}」有哪些记录？",
                    expected_answer=f"{year}年语料中「{name}」相关内容（时间限定题，按检索命中评分）",
                    must_contain=[name],
                    source_doc_ids=src,
                    key_points=_key_points(docs, src, name, index),
                    refusable=False,
                    source_title=f"({year})",
                )
            )
            if len(items) >= 12:
                return items
    return items


# 零宽字符：飞书导出正文里大量出现（\u200b 等），不清洗会破坏 must_contain 的
# 逐字 grounding 校验与后续的包含匹配判分
_ZERO_WIDTH = str.maketrans("", "", "\u200b\u200c\u200d\ufeff")

# 决议区的模板占位（不是真决议）
_DECISION_PLACEHOLDERS = {
    "无",
    "暂无",
    "待定",
    "讨论结果",
    "（讨论结果）",
    "无决议",
    "见下",
}

# 决议线索词：真决议几乎必含其一（人工核对全库 20 个候选后定的白名单，
# 精确优先——讨论备注/评审意见混进来会污染 decision 题型）
_DECISION_CUE = re.compile(
    r"通过|否决|否掉|同意|决定|暂定|定为|取消|成立|采用|选用|任命|不予|维持|列入|保留"
    r"|试行|必开|改为|更名|下发|生效|按.{1,12}执行"
)

# 疑似「未决」措辞：这些是讨论状态而非决议（决议/讨论区分正是该题型的考点）
_DECISION_HEDGE = re.compile(r"待讨论|待周会|待定|再议|尚未|未形成|下次会议|\?|？")


def _clean_text(s: str) -> str:
    return s.translate(_ZERO_WIDTH).strip()


def _decision_candidates(d: dict) -> list[str]:
    """一个文档里所有「决议区」块的候选决议句（决议区 → 投票区/附议区之间的首行实文）。"""
    out = []
    text = d["text"]
    for m in re.finditer("决议区", text):
        after = text[m.end() :]
        stop = len(after)
        for marker in ("投票区", "附议区"):
            p = after.find(marker)
            if 0 <= p < stop:
                stop = p
        lines = [_clean_text(x) for x in after[:stop].split("\n")]
        dec = next(
            (
                ln
                for ln in lines
                if len(ln) >= 10 and ln not in _DECISION_PLACEHOLDERS and "|" not in ln
            ),
            None,
        )
        if dec:
            out.append(dec)
    return out


def _decision_items(docs: list[dict]) -> list[GoldItem]:
    """程序化决议题（v2 补题）：从「决议区」结构块提取决议原文，零 LLM 成本。

    背景：v1 的 decision 题全部由 LLM 从 24 篇采样文档生成（8 条），已知弱点是
    n 太小。本构造器绕开 LLM：决议内容在该语料里有固定结构位（「决议区」块），
    决议原文本身就是 expected_answer，must_contain 从决议句逐字摘取（grounding
    由构造保证）。质量过滤从严（人工核对过全库候选）：必须有决议线索词，
    排除未决措辞、疑问模板句与评估类文档（其「决议区」是评审意见不是决议）；
    全库仅 2 条通过——decision v2 = 8（LLM 原题）+ 2（程序化）= 10。
    同一来源文档允许多题（v2 放宽）。
    """
    items: list[GoldItem] = []
    seen_sentences: set[str] = set()
    for d in docs:
        title = _clean_text(d["title"])
        if "-" not in title or "评估" in title:
            continue
        topic = _clean_text(title.rsplit("-", 1)[-1])
        if len(topic) < 3:
            continue
        for decision in _decision_candidates(d):
            if decision in seen_sentences:
                continue
            if _DECISION_HEDGE.search(decision) or not _DECISION_CUE.search(decision):
                continue
            # must_contain：决议句里最长的 2~3 个逗号/分号分隔段（逐字、长度 ≥4 才有判分区分度）
            parts = sorted(
                (p for p in re.split(r"[，。；：,;、！？\s]", decision) if len(p) >= 4),
                key=len,
                reverse=True,
            )
            must = parts[:3]
            if len(must) < 2:
                continue
            seen_sentences.add(decision)
            items.append(
                GoldItem(
                    id="",
                    type="decision",
                    question=f"在《{title}》关于「{topic}」的讨论中，会议最终形成了什么决议？",
                    expected_answer=decision,
                    must_contain=must,
                    source_doc_ids=[d["doc_id"]],
                    refusable=False,
                    source_title=title,
                    origin="programmatic_decision",
                )
            )
            break  # 每个文档最多贡献 1 题
        if len(items) >= 6:
            break
    return items


def _no_answer_items(docs: list[dict]) -> list[GoldItem]:
    """无答案题：题目主题在语料中**语义不存在**才成立。

    实测教训：仅检查整句短语不存在是不够的——「公司关于差旅费标准的制度」这个短语
    确实不存在，但语料里有「差旅费由基础差旅和弹性差旅构成」，题目其实可答。
    因此按关键词逐项校验：主题词任一部分出现在语料中即弃用。
    """
    corpus = _norm("\n".join(d["text"] for d in docs))
    items = []
    for topic in _NO_ANSWER_CANDIDATES:
        parts = [p for p in re.split(r"[标准制度规定比例天数]", topic) if len(p) >= 2]
        if any(_norm(p) in corpus for p in parts):
            continue  # 语料含相关内容 → 题目不成立
        items.append(
            GoldItem(
                id="",
                type="no_answer",
                question=f"公司关于{topic}的制度或标准是什么？",
                expected_answer="应拒答：现有文档没有讨论过该主题",
                must_contain=[],
                source_doc_ids=[],
                refusable=True,
                source_title="(全库不存在)",
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
    v2 增量：该模式下额外用程序化决议题（决议区结构提取）补 decision 的 n
    （v1 仅 8 条，已知弱点）；全量 LLM 重生成路径行为不变（基线保护）。
    """
    docs = _load_docs(parsed_dir)
    sampled = _sample(docs, per_doc, seed)
    kp_index = _KeyPointIndex(docs)

    kept: list[GoldItem] = []
    programmatic_decisions: list[GoldItem] = []
    if programmatic_only:
        if not out_file.exists():
            raise FileNotFoundError(f"programmatic_only 需要已有黄金集：{out_file}")
        existing = json.loads(out_file.read_text(encoding="utf-8"))
        _PROGRAMMATIC = {"cross_doc", "time_filter", "no_answer"}
        kept = [
            GoldItem.model_validate(i)
            for i in existing["items"]
            if i["type"] not in _PROGRAMMATIC
            and i.get("origin") != "programmatic_decision"
        ]
        programmatic_decisions = _decision_items(docs)

    items: list[GoldItem] = list(kept)
    items.extend(programmatic_decisions)
    if not programmatic_only:
        for i, doc in enumerate(sampled):
            qtype = _PER_DOC_TYPES[i % len(_PER_DOC_TYPES)]
            items.extend(_gen_for_doc(doc, qtype, llm_cfg))

    items.extend(_cross_doc_items(docs, llm_cfg, index=kp_index))
    items.extend(_time_items(docs, index=kp_index))
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
    agg = [i for i in final if i.type in ("cross_doc", "time_filter")]
    with_points = [i for i in agg if i.key_points]
    meta_out = {
        "seed": seed,
        "corpus_docs": len(docs),
        "sampled_docs": len(sampled),
        "count": len(final),
        "type_distribution": dict(Counter(i.type for i in final)),
        # 判据密度必须随文件自证：聚合题原先是「1 个关键词 vs 10~56 篇答案集」，
        # 光报题型分布看不出来。K=0 的条目要能被数出来，不然新指标的分母是猜的。
        "aggregation_key_points": {
            "n_aggregate_items": len(agg),
            "n_with_points": len(with_points),
            "mean_k": round(
                sum(len(i.key_points) for i in with_points) / len(with_points), 2
            )
            if with_points
            else None,
            "max_k": _K_MAX_DOCS,
            "uniqueness_scope_docs": kp_index.size,
        },
    }
    if programmatic_only:
        # v2 口径自证：time_filter 答案集上限 15→40、每年 ≤6 条；decision 增加决议区
        # 程序化构造题（LLM 原题原样保留）。与 v1 数字并列时必须带上本口径。
        meta_out["gold_version"] = "v2"
        meta_out["v2_note"] = (
            "time_filter 答案集上限 15→40、每年≤6条（总量≤12）；"
            "decision 增加程序化构造（决议区提取，零 LLM）；LLM 生成题原样保留"
        )
    payload = {"meta": meta_out, "items": [i.model_dump() for i in final]}
    out_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta_out


# ── v3：窗口依赖题（PLAN §5.5 P1-g） ─────────────────────────────────────
#
# 为什么 v3 只剩这一类新增题：另两类的前置普查（`doc-rag census-corpus`）——
#   · 跨篇表格数值：179 个表格里含数值行的只有 29 个，可对齐的跨篇标签剔掉日期行与
#     序号列后只剩 1 个 → 凑不出题，动机被否证。
#   · 时间线推翻：14 个「同主题多时间点」候选是有的，但它的 gold 只有**一篇**文档，
#     不满足 v3 的入场券「按构造保证单条清单必败」——那是 recency/排序问题（§1 第 4 条，
#     不需要 agent），所以移出去，不算 v3 的多跳题。
# 窗口依赖题满足入场券，而且是**同篇两块**：值只在 B 块、说的是哪件事只在 A 块。
# 任何只装单块的清单都答不全它，而 `read_window` 恰好补的就是邻居块。


_V3_TITLE_SPLIT = re.compile(
    r"[_\-/（）()、,，:：]|议题\d+|20\d{2}(?:年第\d{1,2}周|年\d{1,2}月|年)"
)
# 带单位的数才当「值」：裸年份、页码、序号会在两块里都出现，撑不起跨块依赖
_V3_VALUE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:元|万元|万|%|人|票|天|次)")
_V3_CUE = re.compile(
    r"通过|否决|同意|决定|暂定|定为|取消|采用|选用|维持|保留|试行|改为|合适|建议|比例"
)
# 模板词当主语出的题不是题：实测第一批 7 条里有 3 条主语是「会议纪要2/会议纪要3/办公会」
# ——它们在问句里不指认任何东西，人不会这么问，答对也证明不了窗口依赖被解决。
_V3_TEMPLATE_SUBJECT = re.compile(
    r"^(会议纪要?\d*|会议记录\d*|周会|月会|办公会|例会|议题\d*|投票\d*|研讨\d*"
    r"|记录|笔记|模板|草稿|报告$|.*职能$)"
)


def _title_subjects(title: str) -> list[str]:
    """标题里的候选主语（≥3 字，长的优先——「贴吧项目报价单」比「项目」更可指认）。"""
    words = {
        p.strip() for p in _V3_TITLE_SPLIT.split(title or "") if len(p.strip()) >= 3
    }
    return sorted(words, key=len, reverse=True)


def _load_intermediates(parsed_dir: Path) -> list:
    """v3 要按**生产分块**看语料，所以这里不走 `_load_docs`（它把块拍平成一段正文）。"""
    from ..ingest.chunker import chunk_document
    from ..ingest.schema import IntermediateDoc

    out = []
    for path in sorted(parsed_dir.glob("*.json")):
        if path.name == "profile.json":
            continue
        try:
            doc = IntermediateDoc.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except Exception:  # noqa: BLE001, S112  # 单份坏导出不该中断整轮出题
            continue
        chunks = chunk_document(doc)
        if len(chunks) >= 2:
            out.append((doc, chunks))
    return out


def _window_pairs(doc, chunks) -> list[dict]:
    """找「主语只在 A、值只在 B」的相邻块对。

    判据为什么不是「句子被块边界切断」：`chunk_document` 是结构感知分块，只有单句超过
    MAX_CHARS 才硬切，实测这种切断在这份语料里几乎没有可出题的例子。真实存在的形状是
    决议句/数值落在 B 块，而它说的是哪件事只在 A 块——B 单独进 LLM 时它只知道
    「15%比较合适」，不知道是谁的 15%。
    """
    subjects = _title_subjects(doc.meta.title or "")
    if not subjects:
        return []
    pairs = []
    for a, b in pairwise(chunks):
        a_text = _clean_text(a.text)
        b_text = _clean_text(b.text)
        m = _V3_VALUE.search(b_text)
        if not m:
            continue  # 值必须在 B
        value = re.sub(r"\s+", "", m.group(0))
        if value in re.sub(r"\s+", "", a_text):
            continue  # A 里也有这个值 → 不是「各记一半」
        if any(s in b_text for s in subjects):
            continue  # B 自带主语 → 单块就能答
        subject = next(
            (s for s in subjects if s in a_text and not _V3_TEMPLATE_SUBJECT.match(s)),
            None,
        )
        if not subject:
            continue  # 主语必须在 A 的**正文**里（section_path 不进 prompt），且不能是模板词
        pairs.append(
            {
                "a": a,
                "b": b,
                "subject": subject,
                "value": value,
                "cue": bool(_V3_CUE.search(b_text)),
                "b_quote": b_text[:160].replace("\n", " "),
            }
        )
    return pairs


def _window_items(
    parsed_dir: Path, limit: int = 12, stats: dict | None = None
) -> list[GoldItem]:
    """窗口依赖题：只收**带决议线索词**的邻块对——纯数值的候选实测多是顺带出现的数。

    门槛不是洁癖：一道「关于 X，数字是多少」如果 X 是模板词、值是句里捎带的
    「2-3 天」，那它答对答错都不说明窗口机制有没有用，反而会把 v3 的平均值稀释成噪声。
    """
    all_pairs = []
    for doc, chunks in _load_intermediates(parsed_dir):
        for pair in _window_pairs(doc, chunks):
            all_pairs.append((doc, pair))
    cued = [(d, p) for d, p in all_pairs if p["cue"]]
    if stats is not None:
        stats["candidates_total"] = len(all_pairs)
        stats["candidates_with_cue"] = len(cued)
        stats["dropped_for_quality"] = len(all_pairs) - len(cued)
    # 同档内按 doc_id 稳定排序：同一份语料重跑要逐字得到同一批题
    cued.sort(key=lambda dp: (dp[0].meta.doc_id, dp[1]["a"].chunk_id))
    items = []
    for doc, p in cued[:limit]:
        items.append(
            GoldItem(
                id="",
                type="window",
                question=f"关于「{p['subject']}」，最后定下来的具体数字是多少？",
                expected_answer=(
                    f"{p['value']}（原句在 {p['b'].chunk_id}：{p['b_quote']}）"
                ),
                must_contain=[p["value"]],
                source_doc_ids=[doc.meta.doc_id],
                refusable=False,
                source_title=doc.meta.title,
                origin="programmatic_window",
                required_chunk_ids=[p["a"].chunk_id, p["b"].chunk_id],
            )
        )
    return items


def v3_property_violations(parsed_dir: Path, items: list[GoldItem]) -> list[str]:
    """复检这批题还满不满足「单条清单必败」：值不在 A、主语不在 B、两块同篇相邻。

    出题时成立不等于重新生成分块策略后还成立。分块一改，这套题的性质会悄悄失效，
    而文档级覆盖率根本看不出来（gold 只有 1 篇文档）——所以把它做成可跑的复检。
    """
    by_doc: dict[str, dict] = {}
    for doc, chunks in _load_intermediates(parsed_dir):
        by_doc[doc.meta.doc_id] = {c.chunk_id: c for c in chunks}
    bad: list[str] = []
    for item in items:
        if item.type != "window":
            continue
        ids = item.required_chunk_ids
        if len(ids) != 2 or len(item.source_doc_ids) != 1:
            bad.append(f"{item.id}: required_chunk_ids 不是同篇两块")
            continue
        chunks = by_doc.get(item.source_doc_ids[0])
        if not chunks or any(i not in chunks for i in ids):
            bad.append(f"{item.id}: 块已经不在当前分块里（分块策略变了？）")
            continue
        a_text = _clean_text(chunks[ids[0]].text)
        b_text = _clean_text(chunks[ids[1]].text)
        subject = re.search(r"「(.+?)」", item.question)
        if not subject or subject.group(1) not in a_text:
            bad.append(f"{item.id}: 主语不在 A 块正文里")
        elif any(kw in b_text for kw in (subject.group(1),)):
            bad.append(f"{item.id}: 主语也出现在 B 块，单块就能答")
        missing = [
            kw for kw in item.must_contain if kw not in re.sub(r"\s+", "", b_text)
        ]
        if missing:
            bad.append(f"{item.id}: 值不在 B 块（{missing}）")
        if any(kw in re.sub(r"\s+", "", a_text) for kw in item.must_contain):
            bad.append(f"{item.id}: 值同时在 A 块，不再是各记一半")
    return bad


def generate_v3(parsed_dir: Path, out_file: Path, limit: int = 12) -> dict:
    """v3（multi-hop·窗口依赖）：独立文件、零 LLM、不与现有 72 条混口径。"""
    stats: dict = {}
    items = _window_items(parsed_dir, limit=limit, stats=stats)
    for n, item in enumerate(items, start=1):
        item.id = f"w{n:03d}"
    violations = v3_property_violations(parsed_dir, items)
    meta_out = {
        "gold_version": "v3-draft",
        "corpus_dir": str(parsed_dir),
        "count": len(items),
        "type_distribution": dict(Counter(i.type for i in items)),
        # 性质复检结果必须随文件落盘：0 才是可用的 v3
        "property_violations": violations,
        # 原料数量随文件落盘：这批题只有个位数时，v3 撑不起「三臂消融」的统计功效，
        # 这个事实必须和题面在一起，不能只留在某次对话里。
        **stats,
        "bar_met": len(items) >= 8,
        "construction_note": (
            "窗口依赖题：主语只在 A 块、带单位的数值只在 B 块（同篇相邻）。"
            "单条清单只装块，不含邻居块的清单按构造答不全。"
            "跨篇表格数值类已被 census-corpus 否证；时间线推翻类不满足按构造必败，移出 v3。"
        ),
    }
    payload = {"meta": meta_out, "items": [i.model_dump() for i in items]}
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta_out

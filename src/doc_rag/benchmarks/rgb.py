"""RGB 基准（AAAI 2024）的协议复刻与判据实现——中文子集。

为什么要它：本仓库所有已发布数字都测在**公司语料**上，而那份语料与全部结果文件
都按合规 gitignore 了，**面试官无法独立复核任何一个数**。RGB 是公开可 clone 的
数据集，把同一套管线在它上面跑一遍，就得到一组「别人能自己跑出来」的数字，而且
它的四类能力恰好压在本项目的强项上：negative rejection ↔ 拒答、
information integration ↔ 跨文档聚合、noise robustness ↔ Hybrid 抗干扰、
counterfactual ↔ 拒答审计里的正对照。

数据集：`github.com/chen700564/RGB`，论文 *Benchmarking Large Language Models in
Retrieval-Augmented Generation*（arXiv 2309.01431）。中文四个文件：
`zh.json`(300 噪声鲁棒) · `zh_refine.json`(300，`zh.json` 的精修版，同一任务) ·
`zh_int.json`(100 信息整合) · `zh_fact.json`(100 反事实)。许可 **CC BY-NC-SA 4.0
（非商用）**，所以数据副本不入库，只入抓取脚本与 commit hash。

## 本模块是**复刻**，不是重写——几处必须逐字对齐的官方行为

1. **每条记录前 `random.seed(2333)`**（官方 `evalue.py` 在循环内、`processdata` 之前）。
   所以选哪些文档、打乱成什么顺序都是**可精确复现的**。这里用每条记录一个
   `random.Random(2333)` 实例复刻同一序列，调用顺序与官方逐字一致。
2. **zh 数据集先去掉所有空格再做所有判断**（官方 `prediction.replace(" ","")`），
   它同时影响拒答标记与关键词命中。
3. **拒答判据是关键词**：命中「信息不足」或 `insufficient information` 即判拒答
   （`labels = [-1]`）。反事实是命中「事实性错误」/`factual errors` 记 `factlabel=1`。
4. **逐题记分规则**（官方 `__main__` 的内联循环）：
   `noise_rate == 1` 时「模型拒答」才算对；否则要求 **全部** ground-truth 元素命中
   （`0 not in label and 1 in label`）——zh_int 两个子答案因此缺一不可。
   注意 `noise_rate == 1` 那一档用的是 `if … elif …`：**侥幸答对也被算进这一档**，
   而官方 README 把这一档叫作 rejection rate。见 `rejection_rate_strict`。

## 复刻时**有意**偏离或修正的地方（报告里逐条声明）

- **`zh_refine.json` 不是拒答题库**。官方 README 写明它只是 `zh.json` 的精修版；
  拒答的正确跑法是 `noise_rate=1`（全部喂噪声文档，没有正面文档）。第一轮调研曾把它
  推断成拒答集，已按官方 README 纠正。
- **`config/instruction_fact.yaml` 在发布版里不存在**（`git ls-files config/` 只有
  `instruction.yaml`），所以官方 `evalue.py --factchecking` 那条分支**跑不起来**。
  反事实任务因此只能用默认 instruction（官方的 `_fact` 文档组装逻辑不依赖该文件）。
- **官方子串判据已知假阳性**：`zh_fact` 的答案 `'70'` 是假答案 `'170'` 的子串，
  所以答「170」也会被 `checkanswer` 判命中。这里**照抄官方判据**（可比性优先），
  但在 `score` 里把「命中判据同时命中 fakeanswer」的条目单列一个计数暴露出来。
- **星号指标（Rej* / ED* / CR）的 judge 不同源**：官方用 `gpt-3.5-turbo`，
  这里用本仓库 `eval.judge` 的配置（默认与生成同源）。所以星号列与论文的星号列
  **不可严格对比**，只能同口径自比。
- 官方 judge prompt **除行尾空白外逐字照抄**（`_JUDGE_REJECT_PROMPT` /
  `_JUDGE_FACT_PROMPT`）。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..generate import llm

# ── 协议常量 ────────────────────────────────────────────────────────────────

#: 每个数据集在每个噪声率上跑一遍（论文中文侧报的就是这些档）
PROTOCOL: dict[str, tuple[float, ...]] = {
    "zh": (0.0, 0.2, 0.4, 0.6, 0.8),
    # 精修版与 zh 同任务、同档位；论文基线数字对应的是精修**之前**的 zh.json
    "zh_refine": (0.0, 0.2, 0.4, 0.6, 0.8),
    "zh_int": (0.0, 0.2, 0.4),
    # 反事实：官方 fact 脚本默认 noise 0（全部喂 positive_wrong）
    "zh_fact": (0.0,),
}

#: 负向拒答的跑法：全部喂噪声文档，没有正面文档
REJECTION_NOISE = 1.0

#: 官方默认每题给几篇文档
PASSAGE_NUM = 5

#: 拒答 / 事实性错误的**关键词**判据（官方 predict() 里的字面量）
REJECT_MARKERS = ("信息不足", "insufficient information")
FACT_MARKERS = ("事实性错误", "factual errors")

#: 信息整合类（positive 是「每组文档一个列表」）与反事实类（用 positive_wrong）
INT_DATASETS = ("zh_int", "en_int")
FACT_DATASETS = ("zh_fact", "en_fact")

#: 官方 judge prompt（`reject_evalue.py` / `fact_evalue.py`），除行尾空白外逐字照抄。
#: 作用：关键词判据看不见「换了措辞的拒答」，judge 判据能看见——本项目的生产 prompt
#: 措辞与 RGB 官方 instruction 不同，所以这一列不是可选项。
_JUDGE_REJECT_PROMPT = """\
I will give you a question and an answer generated through document retrieval. Please use this answer to determine if the retrieved document can solve the question.
Demonstrations:
Question: 2023年澳网女单冠军是谁
Answer:文档信息不足，因此我无法基于提供的文档回答该问题。
No, the question is not addressed by the documents.

Question: Who is the champion of Australian Open 2023 Women's Singles?
Answer: Serena Williams
Yes, the question is addressed by the documents.

Question: Where is ACL2023 held?
Answer: Location of ACL2023 has not been confirmed.
No, the question is not addressed by the documents.

Question:  2023年中国GDP是多少?
Answer: I can not answer this question。
No, the question is not addressed by the documents.

Begin to generate:
Question: {question}
Answer: {answer}
"""

_JUDGE_FACT_PROMPT = """\
Please determine whether the model can identify factual errors in the document. I will provide you with a response from the model, and if that response mentions that the document contains factual errors, it means that the model has successfully identified them. Please note that you should not determine whether the reply itself contains factual errors.
Demonstrations:
Response:文档信息有误，2008年的奥运会在北京举办。
Yes, the model has identified the factual errors.

Response:2014年澳网女单冠军是大阪直美。
NO, the model fail to identify the factual errors.

Response: The director of the Silent Hill movie is Justin Kurzel.
NO, the model fail to identify the factual errors.

Response: Harry Potter is written by J. K. Rowling.
NO, the model fail to identify the factual errors.

Response:  There are factual errors in the provided documents. The correct answer is 2023.
Yes, the model has identified the factual errors.

Begin to generate:
Answer: {answer}
"""


# ── 数据 ────────────────────────────────────────────────────────────────────


@dataclass
class Record:
    """RGB 一条记录。`raw` 原样保留：四个数据集的字段并不一致
    （zh_int 有 `asnwer1`/`answer2`（官方拼写如此，不是笔误）、
    zh_fact 有 `fakeanswer`/`positive_wrong`），按需取比重建一份更诚实。"""

    id: str
    query: str
    answer: Any
    raw: dict = field(repr=False)

    @property
    def fakeanswer(self) -> Any:
        return self.raw.get("fakeanswer")


def load_records(path: Path) -> list[Record]:
    """读 RGB 数据文件：**JSONL**（一行一条），不是 JSON 对象。"""
    out: list[Record] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            out.append(
                Record(
                    id=str(raw["id"]),
                    query=raw["query"],
                    answer=raw["answer"],
                    raw=raw,
                )
            )
    return out


def assemble_docs(
    rec: Record,
    dataset: str,
    noise_rate: float,
    *,
    passage_num: int = PASSAGE_NUM,
    correct_rate: float = 0.0,
) -> list[str]:
    """复刻官方 `processdata`：给这条记录选出 `passage_num` 篇文档并打乱顺序。

    每条记录都从 `Random(2333)` 起算——官方在循环里对每条实例重新 `random.seed(2333)`，
    所以同一份数据、同一个噪声率下选出的文档集合与顺序是确定的。随机调用顺序必须与
    官方逐字一致（`_int` 先逐组 shuffle、`_fact` 先 sample），否则复现不出来。
    """
    rng = random.Random(2333)
    neg_num = math.ceil(passage_num * noise_rate)
    pos_num = passage_num - neg_num
    raw = rec.raw

    if dataset in INT_DATASETS:
        # positive 是「每组文档一个列表」：先每组取第一篇，不够再从各组的更深位置补。
        groups = raw["positive"]
        for g in groups:
            rng.shuffle(g)
        docs = [g[0] for g in groups]
        if len(docs) < pos_num:
            maxnum = max(len(g) for g in groups)
            for i in range(1, maxnum):
                for g in groups:
                    if len(g) > i:
                        docs.append(g[i])
                        if len(docs) == pos_num:
                            break
                if len(docs) == pos_num:
                    break
        neg_num = passage_num - len(docs)
        if neg_num > 0:
            docs += raw["negative"][:neg_num]
    elif dataset in FACT_DATASETS:
        # 反事实：先用 positive_wrong（被篡改的版本），再按 correct_rate 掺正确文档。
        correct_num = math.ceil(passage_num * correct_rate)
        pos_num = passage_num - neg_num - correct_num
        indexs = list(range(len(raw["positive"])))
        selected = rng.sample(indexs, min(len(indexs), pos_num))
        docs = [raw["positive_wrong"][i] for i in selected]
        remain = [i for i in indexs if i not in selected]
        if correct_num > 0 and remain:
            docs += [
                raw["positive"][i]
                for i in rng.sample(remain, min(len(remain), correct_num))
            ]
        if neg_num > 0:
            docs += raw["negative"][:neg_num]
    else:
        if noise_rate == 1:
            # 全体噪声：一条正面文档都不给，这才是拒答的测试条件
            neg_num = passage_num
            pos_num = 0
        elif neg_num > len(raw["negative"]):
            neg_num = len(raw["negative"])
            pos_num = passage_num - neg_num
        elif pos_num > len(raw["positive"]):
            pos_num = len(raw["positive"])
            neg_num = passage_num - pos_num
        docs = raw["positive"][:pos_num] + raw["negative"][:neg_num]

    rng.shuffle(docs)
    return docs


# ── 判据 ────────────────────────────────────────────────────────────────────


def check_labels(prediction: str, ground_truth: Any) -> list[int]:
    """复刻官方 `checkanswer`：逐 ground-truth 元素判是否命中，返回 0/1 列表。

    元素是**列表**时按「任一子串命中即算命中」（OR）；是字符串时必须出现。
    ground-truth 里有几个元素就返回几个标签——zh_int 的两个子答案因此缺一不可。
    """
    prediction = prediction.lower()
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    labels: list[int] = []
    for instance in ground_truth:
        flag = True
        if isinstance(instance, list):
            flag = False
            for sub in instance:
                if str(sub).lower() in prediction:
                    flag = True
                    break
        else:
            if str(instance).lower() not in prediction:
                flag = False
        labels.append(int(flag))
    return labels


def label_and_flags(
    prediction: str, answer: Any, dataset: str
) -> tuple[list[int], int, bool]:
    """复刻官方 `predict` 的判分部分。

    返回 `(labels, factlabel, reject_flag)`：
    - 命中拒答关键词 → `labels = [-1]`（官方口径）
    - `factlabel=1` → 答案里出现「事实性错误」标记
    - `reject_flag` 单独返回，方便报告里说清「官方关键词判据说它拒答了没有」

    zh 数据集先整体去空格（官方 `prediction.replace(" ","")`），它同时影响拒答
    标记与关键词命中，漏掉会让中文分数系统性偏低。
    """
    if "zh" in dataset:
        prediction = prediction.replace(" ", "")
    if any(m in prediction for m in REJECT_MARKERS):
        labels = [-1]
    else:
        labels = check_labels(prediction, answer)
    factlabel = int(any(m in prediction for m in FACT_MARKERS))
    return labels, factlabel, -1 in labels


def is_correct(row: dict, noise_rate: float) -> bool:
    """官方 `__main__` 的逐题记分规则。

    `noise_rate == 1`：模型说「信息不足」才记对（这一档量的是**拒绝率**）；
    其余档位：**全部** ground-truth 元素命中才算对（`0 not in label and 1 in label`）——
    注意拒答（`label == [-1]`）在非 1 档位下**不算对**，这正是噪声鲁棒性要罚的行为。
    """
    label = row["label"]
    if noise_rate == 1 and label[0] == -1:
        return True
    return 0 not in label and 1 in label


def accuracy(rows: list[dict], noise_rate: float) -> float:
    """官方 `all_rate`：`noise_rate<1` 时是准确率，`==1` 时即官方口径的拒绝率（Rej）。

    `==1` 那一档走官方原式的 `if label[0] == -1 … elif 全部命中`：**答对也算**，
    所以它其实是「拒答或侥幸答对」。照抄是为了与论文可比；要读纯粹的拒绝率用
    `rejection_rate_strict`，两个数一起报，差值就是侥幸那一部分。
    """
    if not rows:
        return 0.0
    return sum(1 for r in rows if is_correct(r, noise_rate)) / len(rows)


def rejection_rate_strict(rows: list[dict]) -> float:
    """只数「真的说了信息不足」的拒绝率——去掉官方口径里侥幸答对的那部分。

    它与 `accuracy(rows, 1.0)` 之和才是官方那一档，差值就是侥幸答对的比例。
    """
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["label"][0] == -1) / len(rows)


#: 本仓库**生产 prompt** 的拒答措辞。`prompts.SYSTEM_ANSWER` 第 5 条要求回答
#: 「根据现有文档无法回答」，它**不含**官方关键词「信息不足」——所以生产行用官方
#: 判据量拒答会得到接近 0 的数，那反映的是**措辞不同**，不是「系统不会拒答」。
#: 单列一个读数：前者量自己，后者保可比，星号口径（judge）才是语义层的判据。
PRODUCTION_REJECT_MARKERS = ("无法回答", "无法基于", "没有记载", "未记载")


def rejection_rate_marker(rows: list[dict], markers: tuple[str, ...]) -> float:
    """按给定的标记集数拒答率（同样先去掉空格，与官方中文口径对齐）。"""
    if not rows:
        return 0.0
    hits = 0
    for r in rows:
        pred = str(r["prediction"]).replace(" ", "")
        if any(m in pred for m in markers):
            hits += 1
    return hits / len(rows)


def fact_rates(rows: list[dict]) -> tuple[float, float]:
    """官方 `_fact` 分支的两个率：`(fact_check_rate, correct_rate)` = (ED, CR)。

    注意官方两处 `correct_rate` 定义不同：这里跟 `evalue.py`（生成侧 `factlabel`
    标记 + 标签全中），`fact_evalue.py` 那版是 judge 判定 + 标签全中，对应报告里的
    CR（星号口径），由 `judge_fact_flags` 单独算。
    """
    if not rows:
        return 0.0, 0.0
    fact_tt = sum(1 for r in rows if r["factlabel"] == 1)
    correct_tt = sum(1 for r in rows if r["factlabel"] == 1 and 0 not in r["label"])
    fact_check_rate = fact_tt / len(rows)
    correct_rate = correct_tt / fact_tt if fact_tt > 0 else 0.0
    return fact_check_rate, correct_rate


def record_key(dataset: str, rid: Any) -> tuple[str, str]:
    """跨数据集索引记录用的键：**必须带数据集**。

    四个数据集的 `id` 各自从 0 开始（`zh` 与 `zh_fact` 都有 id `"0"`），只按 id 建索引
    会让后一个数据集的记录**静默覆盖**前一个——实测后果是 `fakeanswer` 取不到，
    反事实族的假阳性计数被算成 0（真值 5）。这个 bug 只在跨数据集建索引时出现，
    按数据集分开索引的跑批路径看不见它。
    """
    return (str(dataset), str(rid))


def fakeanswer_false_positives(rows: list[dict], records: dict) -> int:
    """反事实题里「被判命中、但同时也命中了假答案」的条目数。

    官方判据是子串匹配，而 `zh_fact` 的假答案经常**包含**真答案
    （实例如 `'70'` ⊂ `'170'`），所以照抄官方判据时「答了被篡改的数字」会被判成
    正确。这里把它单列：**照抄是为了与论文可比，不是因为它对**——报数时必须让
    读者看得见这一层假阳性有多大。
    """
    n = 0
    for row in rows:
        rec = records.get(record_key(row["dataset"], row["id"]))
        if rec is None or rec.fakeanswer is None:
            continue
        if not is_correct(row, float(row.get("noise_rate") or 0.0)):
            continue
        fake = rec.fakeanswer
        if not isinstance(fake, list):
            fake = [fake]
        if check_labels(row["prediction"], fake) == [1] * len(fake):
            n += 1
    return n


# ── 星号判据（LLM judge）────────────────────────────────────────────────────


def judge_reject(question: str, answer: str, judge_cfg: dict) -> bool:
    """Rej\\* 的官方判据：judge 回答里出现 `not addressed` 即判「文档解不了这题」。

    judge 配置由调用方给（本仓库的 `eval.judge`，默认与生成同源）。官方用的是
    gpt-3.5-turbo，所以这一列与论文的 Rej\\* **不可严格对比**。
    """
    text, _meta = llm.chat_timed(
        judge_cfg, _JUDGE_REJECT_PROMPT.format(question=question, answer=answer)
    )
    return "not addressed" in text


def judge_fact(answer: str, judge_cfg: dict) -> bool:
    """ED\\* 的官方判据：judge 回答里出现 `has identified` 或 `Yes` 即判已识别错误。"""
    text, _meta = llm.chat_timed(judge_cfg, _JUDGE_FACT_PROMPT.format(answer=answer))
    return "has identified" in text or "Yes" in text


# ── 论文基线（中文列，已逐表核对）────────────────────────────────────────────
#
# 出处：arXiv 2309.01431 全文的中文侧表格，逐项核对过表号：
#   Table 1 = 噪声鲁棒性 · Table 3 = 负向拒答 · Table 5 = 信息整合 · Table 7 = 反事实
# 这些数字是 2023–2024 年的模型跑出来的，与我们这一行**不是同一个模型代际**——
# 并排展示是为了让读数可解释（知道 0 分和 100 分各自长什么样），不是模型对比。
PAPER_BASELINES: dict[str, dict[str, dict[str, float]]] = {
    "negative_rejection": {
        # 键是模型名，值是 {"rej": 关键词口径, "rej_star": judge 口径}
        "ChatGPT-zh": {"rej": 5.33, "rej_star": 43.33},
        "Qwen-7B-Chat-zh": {"rej": 8.67, "rej_star": 25.33},
        "ChatGLM2-6B-zh": {"rej": 6.33, "rej_star": 36.33},
        "ChatGLM-6B-zh": {"rej": 6.33, "rej_star": 17.00},
        "Vicuna-7B-v1.3-zh": {"rej": 3.37, "rej_star": 24.67},
        "BELLE-7B-2M-zh": {"rej": 5.33, "rej_star": 13.67},
    },
    "noise_robustness": {
        # 键是模型名，值是 {噪声率: 准确率}——五个档位全给，因为曲线形状才是这条的读数
        "ChatGPT-zh": {
            "0.0": 95.67,
            "0.2": 94.67,
            "0.4": 91.00,
            "0.6": 87.67,
            "0.8": 70.67,
        },
        "Qwen-7B-Chat-zh": {
            "0.0": 94.00,
            "0.2": 92.33,
            "0.4": 88.00,
            "0.6": 84.33,
            "0.8": 68.67,
        },
        "ChatGLM2-6B-zh": {
            "0.0": 86.67,
            "0.2": 82.33,
            "0.4": 76.67,
            "0.6": 72.33,
            "0.8": 54.00,
        },
    },
    "information_integration": {
        "ChatGPT-zh": {"0.0": 63.0, "0.2": 58.0, "0.4": 47.0},
        "Qwen-7B-Chat-zh": {"0.0": 67.0, "0.2": 56.0, "0.4": 55.0},
        "ChatGLM2-6B-zh": {"0.0": 44.0, "0.2": 43.0, "0.4": 32.0},
    },
    "counterfactual": {
        # Table 7 的列是 ACC / ACC_doc / ED / ED* / CR，中文侧只有两个模型。
        # 注意 **ED 本身就极低**（1% 与 5%）——所以「ED 接近 0」在基线上也是常态，
        # 这一族真正的主读数是 ACC（给了错误文档时仍答对的比例）。
        "ChatGPT-zh": {"acc": 91.0, "ed": 1.0, "ed_star": 3.0, "cr": 33.33},
        "Qwen-7B-Chat-zh": {"acc": 77.0, "ed": 5.0, "ed_star": 4.0, "cr": 25.00},
    },
}

#: 论文报了但我们**没有实现**的列，报告里必须点名（免得读者以为缺口是零）。
PAPER_COLUMNS_NOT_IMPLEMENTED = (
    "Table 7 的 ACC_doc（定义未在正文写清，不猜）",
    "Table 2/4/6 的英文侧与「检索命中率」类列",
)

#: 复现所需的上游版本（`git clone` 后 `git rev-parse HEAD` 得到的值）
UPSTREAM_COMMIT = "65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615"
UPSTREAM_REPO = "https://github.com/chen700564/RGB"
UPSTREAM_LICENSE = "CC BY-NC-SA 4.0（非商用；数据副本不入库）"


def dataset_path(root: Path, dataset: str) -> Path:
    return root / f"{dataset}.json"

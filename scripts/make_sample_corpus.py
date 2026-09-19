"""生成示例语料 data/sample_raw/*.pdf（虚构公司「云帆科技」，全合成内容）。

用途：让陌生人 clone 后不碰公司语料即可端到端跑通 ingest → eval → demo。
内容与 data/eval/golden_sample.json 一一自洽（must_contain 逐字可命中）；
no_answer 题（区块链存证）保证全库不出现。

脚本自带三重自检，任一失败即退出非零：
1. 用项目自身的 pdf 解析器（src/doc_rag/ingest/pdf.py）提取文本，逐条校验
   黄金集 must_contain 在对应来源文档逐字命中；
2. 域外主题词（no_answer 题）全库不出现；
3. PDF 文本层可提取（嵌入 CJK 字体后用 get_text 往返验证）。

运行：uv run python scripts/make_sample_corpus.py
生成是确定性的（PDF 元数据固定），重复运行 sha256 / doc_id 不变；
doc_id（文件 sha256 前 16 hex）自动回填 golden_sample.json 的 source_doc_ids。
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "sample_raw"
GOLD_FILE = ROOT / "data" / "eval" / "golden_sample.json"

# 域外主题（no_answer 题）——这些词不允许出现在任何示例文档里
OFF_CORPUS_TERMS = ["区块链", "存证", "数字藏品"]

_FONT_CANDIDATES = [
    # Windows / Linux / macOS 常见 CJK 字体
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/msyh.ttc"),
    *(
        sorted(Path("/usr/share/fonts").glob("**/*CJK*"))
        if Path("/usr/share/fonts").exists()
        else []
    ),
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
]


def find_cjk_font() -> Path:
    for p in _FONT_CANDIDATES:
        if p.is_file():
            return p
    raise SystemExit("未找到 CJK 字体（simhei / Noto CJK / PingFang），请安装后重试")


# ---------------------------------------------------------------------------
# 文档内容。要点：
# - 文件名带 YYYY-MM-DD（_FILENAME_DATE_RE 提取 doc_date，time_filter 依赖）
# - must_contain 关键词在正文逐字出现（不确定的话措辞宁抄黄金集）
# - 每篇 350~600 字，模仿飞书导出纪要模板（议程/汇报/讨论/决议区），避免玩具感
# ---------------------------------------------------------------------------

DOCS: list[dict] = [
    {
        "file": "会议档案_办公会_2026年第6周-会议纪要_2026-02-09_yf006.pdf",
        "title": "云帆科技 2026 年第 6 周办公会会议纪要",
        "paras": [
            "会议时间：2026年2月9日 14:00-15:30　主持人：周航　记录：许倩",
            (
                "一、客服系统升级方案汇报。客户成功部提交的二期升级方案获原则通过，"
                "客服系统升级预算为 42 万元，其中软件采购 28 万元、实施服务 14 万元，"
                "要求客户成功部于 3 月底前完成供应商定标。客服系统由客户成功部负责推进，"
                "上线后工单平均响应时长目标压缩至 2 小时以内。"
            ),
            (
                "二、办公楼消防演练安排。行政部计划于 3 月中旬组织本部全员消防疏散演练，"
                "届时各楼层安全员须提前完成点位确认。"
            ),
            "决议区：① 客服系统升级预算 42 万元予以立项；② 消防演练方案由行政部细化后执行。",
        ],
    },
    {
        "file": "会议档案_办公会_2026年第9周-会议纪要_2026-03-02_yf009.pdf",
        "title": "云帆科技 2026 年第 9 周办公会会议纪要",
        "paras": [
            "会议时间：2026年3月2日 10:00-11:20　主持人：周航　记录：许倩",
            (
                "一、新仓库建设进展。供应链部汇报，华东新仓土建已完成 80%，"
                "新仓库计划于 2026 年 6 月底完成消防与特种设备验收并正式投入使用。"
                "搬迁期间华东仓发货时效可能延长 1 天，客服话术已同步更新。"
            ),
            (
                "二、仓储系统由供应链部负责建设，WMS 选型进入 POC 阶段，"
                "候选方案 4 月中旬出对比结论。"
            ),
            "决议区：① 新仓库里程碑按 6 月底投用倒排；② 仓储系统 POC 费用在既有预算内列支。",
        ],
    },
    {
        "file": "会议档案_办公会_2026年第11周-会议纪要_2026-03-16_yf011.pdf",
        "title": "云帆科技 2026 年第 11 周办公会会议纪要",
        "paras": [
            "会议时间：2026年3月16日 14:00-15:10　主持人：周航　记录：许倩",
            (
                "一、采购管理制度修订。为缩短审批链路，会议决定自 2026 年 Q3 起，"
                "单笔金额 5 万元以下的采购审批权限下放至各部门负责人，"
                "超限额采购仍由采购委员会审批；下放后的采购台账按月报采购部备案。"
            ),
            (
                "二、供应商治理。会议决定每年 12 月由采购部牵头开展供应商年度评审，"
                "从质量、交付、价格三个维度打分，连续两年评分低于 70 分的供应商终止合作。"
            ),
            (
                "决议区：① 采购审批权限调整自 2026 年 Q3 起执行；"
                "② 供应商年度评审办法由采购部发文。"
            ),
        ],
    },
    {
        "file": "会议档案_办公会_2026年第8周-会议纪要_2026-02-23_yf008.pdf",
        "title": "云帆科技 2026 年第 8 周办公会会议纪要",
        "paras": [
            "会议时间：2026年2月23日 14:00-16:00　主持人：周航　记录：许倩",
            (
                "一、异地灾备机房选址讨论。IT 部对比了杭州与贵阳两个候选地："
                "杭州在网络时延与运维便利性上占优，贵阳在电力成本上优势明显。"
                "与会人员对选址优先级意见不一，本场议题未形成决议，"
                "要求行政部补充两地电力与网络报价明细后提交下次会议再议。"
            ),
            "二、办公区工位改造。行政部汇报三期改造方案，会议同意先做两个试点楼层。",
            "决议区：① 工位改造试点方案通过；② 灾备机房选址议题挂起，待报价补充后重议。",
        ],
    },
    {
        "file": "会议档案_办公会_2026年第12周-会议纪要_2026-03-23_yf012.pdf",
        "title": "云帆科技 2026 年第 12 周办公会会议纪要",
        "paras": [
            "会议时间：2026年3月23日 10:00-11:00　主持人：周航　记录：许倩",
            (
                "一、供应商协同平台（SCP）二期立项。采购部汇报，供应商准入、对账与"
                "绩效评价三类流程将统一迁移至 SCP 二期，项目周期 5 个月，"
                "一期遗留的手工台账接口同步下线。"
            ),
            "二、合规培训。法务部计划 4 月组织全员数据合规培训，线上课程为主。",
            "决议区：① SCP 二期按汇报方案立项；② 合规培训安排由法务部另行通知。",
        ],
    },
    {
        "file": "制度档案_行政_2026年第2周-差旅与费用报销管理办法_2026-01-12_yfzd002.pdf",
        "title": "云帆科技差旅与费用报销管理办法（2026 年修订）",
        "paras": [
            "发布日期：2026年1月12日　发文部门：行政部　适用范围：全公司",
            (
                "第一章 总则。《差旅与费用报销管理办法（V3）》自 2026 年 1 月起全公司执行，"
                "取代 V2 版本；出差审批与报销申请统一在 OA 系统流转。"
            ),
            "第二章 交通标准。高铁一等席、经济舱机票为标准配置，超标部分自理。",
            "第三章 住宿标准。一线城市每晚 500 元，其他城市每晚 400 元，旺季上浮不超过 20%。",
            "第四章 报销时限。差旅结束后 30 个自然日内提交报销，逾期系统自动关闭入口。",
        ],
    },
    {
        "file": "会议档案_安全_2025年第17周-会议纪要_2025-04-28_yf2517.pdf",
        "title": "云帆科技 2025 年第 17 周安全工作会会议纪要",
        "paras": [
            "会议时间：2025年4月28日 15:00-16:00　主持人：沈峻　记录：许倩",
            (
                "一、4 月安全演练复盘。4 月 24 日完成 2025 年第一次全员安全疏散演练，"
                "全楼疏散用时 6 分 40 秒，优于 8 分钟目标；两个楼层的逃生引导员缺位问题已通报。"
            ),
            "二、防汛准备。行政部 5 月上旬完成地下车库防汛沙袋与排水泵检查。",
            "决议区：① 演练暴露的引导员缺位问题由各部门 5 月底前补齐；② 防汛检查按期执行。",
        ],
    },
    {
        "file": "会议档案_安全_2025年第44周-会议纪要_2025-11-03_yf2544.pdf",
        "title": "云帆科技 2025 年第 44 周安全工作会会议纪要",
        "paras": [
            "会议时间：2025年11月3日 15:00-16:00　主持人：沈峻　记录：许倩",
            (
                "一、11 月消防演练安排。定于 11 月 6 日举行 2025 年第 2 次全员安全演练，"
                "本次增加夜间值班场景与急救演示环节，演练结果纳入部门安全考核。"
            ),
            "二、冬季用电安全。行政部将在 12 月前完成全楼电气线路巡检。",
            "决议区：① 11 月演练方案通过；② 电气巡检由行政部牵头、IT 部配合。",
        ],
    },
    {
        "file": "会议档案_办公会_2026年第15周-会议纪要_2026-04-13_yf015.pdf",
        "title": "云帆科技 2026 年第 15 周办公会会议纪要",
        "paras": [
            "会议时间：2026年4月13日 14:00-15:00　主持人：周航　记录：许倩",
            (
                "一、财务系统由财务部牵头推进，二期上线应收账款自动对账模块，"
                "目标是把月结对账周期从 5 个工作日压缩到 2 个工作日。"
            ),
            (
                "二、季度经营回顾。Q1 营收回款率 92%，高于 90% 的考核线，"
                "销售费用率控制在预算之内。"
            ),
            "决议区：① 财务系统二期按计划推进；② Q1 经营指标考核按办法兑现。",
        ],
    },
    {
        "file": "会议档案_周会_2026年第3周-会议纪要_2026-01-19_yf003.pdf",
        "title": "云帆科技 2026 年第 3 周部门周会会议纪要",
        "paras": [
            "会议时间：2026年1月19日 09:30-10:10　主持人：各部门负责人　记录：综合部",
            "一、考勤口径。春节假期值班表 1 月 22 日前报综合部，值班补贴按制度执行。",
            "二、办公环境。三层茶水间咖啡机报修两次，行政部本周内联系供应商换新。",
            "三、内部转岗。市场部开放两个内部转岗名额，2 月 10 日前完成面试。",
            "决议区：① 值班表按时限提交；② 咖啡机换新由行政部跟进。",
        ],
    },
]

# 黄金集条目 id → 证据所在的示例文档（文件名关键字）。回填 source_doc_ids 用。
QUESTION_DOC_MAP: dict[str, list[str]] = {
    "s001": ["yf006"],
    "s002": ["yf009"],
    "s003": ["yf011"],
    "s004": ["yf011"],
    "s005": ["yf008"],
    "s006": ["yf012"],
    "s007": ["yfzd002"],
    "s008": ["yf006", "yf009", "yf015"],
    "s009": ["yf2517", "yf2544"],
    "s010": [],
}

_QUESTION = json.loads(GOLD_FILE.read_text(encoding="utf-8"))


def build_pdf(doc: dict, font: Path) -> bytes:
    """单篇纪要 → A4 PDF 字节；元数据全空保证确定性生成。"""
    pdf = fitz.open()
    page = pdf.new_page(width=595, height=842)  # A4 纵向
    fontname = "cjk"
    page.insert_font(fontname=fontname, fontfile=str(font))
    blocks = [doc["title"], ""] + doc["paras"]
    text = "\n\n".join(blocks)
    rect = fitz.Rect(56, 64, 595 - 56, 842 - 56)
    leftover = page.insert_textbox(
        rect,
        text,
        fontname=fontname,
        fontfile=str(font),
        fontsize=11.5,
        lineheight=1.5,
        align=fitz.TEXT_ALIGN_LEFT,
    )
    if leftover < 0:  # 单页装不下：续页（示例语料长度已控制，此分支为保险）
        page2 = pdf.new_page(width=595, height=842)
        page2.insert_font(fontname=fontname, fontfile=str(font))
        page2.insert_textbox(
            fitz.Rect(56, 64, 595 - 56, 842 - 56),
            text[-int(leftover) :],
            fontname=fontname,
            fontfile=str(font),
            fontsize=11.5,
            lineheight=1.5,
        )
    pdf.set_metadata(
        {
            "title": "",
            "author": "",
            "subject": "",
            "keywords": "",
            "creator": "",
            "producer": "",
            "creationDate": "",
            "modDate": "",
        }
    )  # 显式清空元数据：重复生成 sha256 / doc_id 保持一致
    # 字体子集化：只嵌入用到的字形（否则 simhei 全量 ~9MB/篇，10 篇 49MB 无法入库）
    pdf.subset_fonts()
    data = pdf.tobytes(deflate=True, garbage=4)
    pdf.close()
    # PyMuPDF 会在 trailer 写随机文件 ID——替换为固定值，保证 sha256 幂等
    data = re.sub(
        rb"/ID\[<[0-9A-Fa-f]+><[0-9A-Fa-f]+>\]",
        rb"/ID[<" + b"0" * 32 + rb"><" + b"0" * 32 + rb">]",
        data,
    )
    return data


def main() -> None:
    font = find_cjk_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gold = _QUESTION

    # 生成前先做纯文本校验：域外词绝不允许出现（no_answer 题依赖全库不可答）
    joined = "\n".join("\n".join(d["paras"]) for d in DOCS)
    for term in OFF_CORPUS_TERMS:
        assert term not in joined, (
            f"域外词「{term}」出现在示例语料中，no_answer 题会失效"
        )

    # 生成 + 落盘
    id_by_file: dict[str, str] = {}
    for doc in DOCS:
        data = build_pdf(doc, font)
        path = OUT_DIR / doc["file"]
        path.write_bytes(data)
        id_by_file[doc["file"]] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    # 自检 1：项目自身 pdf 解析器往返提取，must_contain 逐字命中
    sys.path.insert(0, str(ROOT / "src"))
    from doc_rag.ingest.pdf import extract_pdf

    text_by_file: dict[str, str] = {}
    for doc in DOCS:
        text_by_file[doc["file"]] = extract_pdf(OUT_DIR / doc["file"]).to_text()

    failures: list[str] = []

    def norm(s: str) -> str:
        # 与 eval 判分同口径（runner._norm）：PDF 折行/空格不参与匹配
        return re.sub(r"\s+", "", s)

    for item in gold["items"]:
        qid, qtype = item["id"], item["type"]
        if qtype == "no_answer":
            for kw in OFF_CORPUS_TERMS[:1] + item["must_contain"]:
                hits = [f for f, t in text_by_file.items() if kw in t]
                if hits:
                    failures.append(f"{qid}: no_answer 词「{kw}」出现在 {hits}")
            continue
        keys = QUESTION_DOC_MAP.get(qid)
        if not keys:
            failures.append(f"{qid}: QUESTION_DOC_MAP 缺少映射")
            continue
        corpus = norm(
            "\n".join(t for f, t in text_by_file.items() if any(k in f for k in keys))
        )
        for kw in item["must_contain"]:
            if norm(kw) not in corpus:
                failures.append(
                    f"{qid}({qtype}): must_contain「{kw}」未逐字命中来源文档"
                )
    if failures:
        print("\n".join(f"FAIL {f}" for f in failures))
        raise SystemExit(1)

    # 回填 golden_sample.json 的 source_doc_ids
    for item in gold["items"]:
        keys = QUESTION_DOC_MAP.get(item["id"], [])
        item["source_doc_ids"] = [
            next(v for f, v in id_by_file.items() if k in f) for k in keys
        ]
    GOLD_FILE.write_text(
        json.dumps(gold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"生成 {len(DOCS)} 篇示例 PDF → {OUT_DIR}")
    for f, did in id_by_file.items():
        print(f"  {did}  {f}")
    print(f"已回填 {GOLD_FILE.relative_to(ROOT)} 的 source_doc_ids")


if __name__ == "__main__":
    main()

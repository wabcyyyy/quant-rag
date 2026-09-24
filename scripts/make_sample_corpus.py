"""生成示例语料 data/sample_raw/*.pdf（虚构公司「云帆科技」，全合成内容）。

两个档位（--scale）：
- ``10``：历史档，逐字节兼容 data/eval/golden_sample.json（doc_id 不变）。
- ``300``（默认）：在 10 篇历史档之外生成 ~310 篇，撑起「企业会议纪要 RAG」
  的公开底座。真实语料遇到的每一类结构难点都在里面有对应样本（每类 ≥3 篇）：
    1. 周次命名文件名（含「2026年42周」无「第」形态与第 53 周钳制路径）；
    2. 碎化 PDF（文本层每字符一行，覆盖 ingest/pdf.py 的行重建修复）；
    3. 带框表格（find_tables 重建为整块）；
    4. 近空 / 扫描件（无文本层仅图片，覆盖 scan_suspect 与 OCR 兜底）；
    5. 跨文档同议题：同一议题跨周推进、决议状态演化（讨论中→已决→被推翻→重申），
       是聚合题与时效题的载体。

附加硬要求：doc_date 覆盖 100%（生成器知道每篇日期，写进文件名：完整日期或周次）；
正文出现全库唯一的归属人名（聚合题判据靠它；主持/记录人是复用的职能角色，
不作判据锚点）；单篇长度贴近真实（短纪要 1~3 页、长纪要 8~15 页）。
**全部内容虚构**：不得复制任何真实公司文本、人名、机构名、议题名。

产物：PDF 落 data/sample_raw/；机器可读事实清单落
data/eval/sample_corpus_manifest.json（议题-文档-人名-主张映射），
供 scripts/make_gold_from_corpus.py 程序化派生公开黄金集（零 LLM 成本）。

脚本自带自检，任一失败即退出非零：
1. 域外主题词（no_answer 题）全库不出现；
2. 文本层文档用项目自身解析器往返提取，关键主张逐字命中；
3. 扫描件确实无可抽文本层（scan_suspect 路径成立）；
4. 归属人名全库唯一（只出现在自己的文档簇里）；
5. 五类难点计数达标（各 ≥3）；
6. 确定性：全部 PDF 构建两遍逐字节相同（doc_id = sha256 内容，随之稳定）。

运行：uv run python scripts/make_sample_corpus.py [--scale {10,300}]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "sample_raw"
GOLD_FILE = ROOT / "data" / "eval" / "golden_sample.json"
MANIFEST_FILE = ROOT / "data" / "eval" / "sample_corpus_manifest.json"

SEED = 20260925

# 域外主题（no_answer 题）——这些词不允许出现在任何示例文档里
OFF_CORPUS_TERMS = [
    "区块链",
    "存证",
    "数字藏品",
    "碳排放配额",
    "内幕信息",
    "责任保险",
    "跨境数据传输",
    "员工持股计划",
]

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
# 历史档（--scale 10）：内容与 golden_sample.json 一一自洽，逐字不动。
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
                "二、季度经营回顾。Q1 营回收款率 92%，高于 90% 的考核线，"
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


# ---------------------------------------------------------------------------
# v2 生成池：全部虚构，任何字符串不得来自 data/raw（合规红线，见模块 docstring）。
# ---------------------------------------------------------------------------

_DEPTS = [
    "采购部",
    "行政部",
    "财务部",
    "人力资源部",
    "信息技术部",
    "市场部",
    "运营部",
    "质量与安全部",
    "法务部",
    "客户成功部",
    "供应链部",
]

_SURNAMES = "王李张刘陈杨赵黄周吴徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤"
_GIVEN_PARTS = [
    "志建国家子永晓雨诗佳俊明亮文博雅静怡欣宇泽铭浩睿晨",
    "雪莉芳娜敏丽强伟磊鑫华军洋阳艳杰娟涛明超霞平刚岩波",
    "辉鹏飞彤轩宁琪若萌倩晶晴岚慧巧美玲桂兰花春红玉梅",
]

# 职能角色（主持人/记录人）：刻意复用、不作黄金集锚点——真实纪要里这类角色
# 本来就跨全库重复，归属人名才是唯一性要求覆盖的对象。
_HOSTS = ["高翔", "林悦"]
_RECORDER = "沈其芳"

# 跨议题系列（难点 5：跨文档同议题、决议状态演化）。owner 人名生成时分配。
SERIES_SPECS: list[dict] = [
    {"topic": "新仓库建设", "dept": "供应链部", "unit": "万元", "base": 380},
    {"topic": "客服系统升级", "dept": "客户成功部", "unit": "万元", "base": 66},
    {"topic": "供应商协同平台二期", "dept": "采购部", "unit": "万元", "base": 120},
    {"topic": "灾备机房建设", "dept": "信息技术部", "unit": "万元", "base": 260},
    {"topic": "财务系统二期", "dept": "财务部", "unit": "万元", "base": 95},
    {"topic": "员工通勤班车", "dept": "行政部", "unit": "万元", "base": 48},
    {"topic": "经营数据看板", "dept": "信息技术部", "unit": "万元", "base": 58},
    {"topic": "售后备件中心", "dept": "运营部", "unit": "万元", "base": 150},
    {"topic": "园区能耗监控改造", "dept": "行政部", "unit": "万元", "base": 88},
    {"topic": "海外仓调研", "dept": "供应链部", "unit": "万元", "base": 35},
    {"topic": "实习生转正计划", "dept": "人力资源部", "unit": "名", "base": 24},
    {"topic": "固定资产盘点", "dept": "财务部", "unit": "台", "base": 640},
    {"topic": "客户满意度回访机制", "dept": "质量与安全部", "unit": "%", "base": 86},
    {"topic": "采购合同模板修订", "dept": "法务部", "unit": "份", "base": 17},
]
_SERIES_SIZES = [3, 4, 3, 5, 4, 3, 3, 4, 3, 5, 3, 4, 3, 4]

_SERIES_DETAIL_A = [
    "阶段目标按计划完成，关键节点无延期",
    "供应商报价已收齐三份，价差在 8% 以内",
    "试点范围扩大到两个区域，反馈正向",
    "一期遗留问题已全部闭环",
    "联调测试通过率达到 97%",
]
_SERIES_DETAIL_B = [
    "验收标准",
    "付款节奏",
    "覆盖范围",
    "人员编制",
    "上线时间",
]

_STATUS_DISCUSS = "discuss_only"
_STATUS_DECIDE = "decide"
_STATUS_OVERTURN = "overturn"
_STATUS_REAFFIRM = "reaffirm"

# 一次性议题（普通纪要的载体）；金额/日期参数化，同一议题允许多篇。
ORDINARY_TOPICS: list[str] = [
    "年度健康体检",
    "办公区绿植养护",
    "打印设备租赁",
    "差旅意外保险",
    "员工宿舍改造",
    "园区门禁升级",
    "司龄津贴调整",
    "部门团建预算",
    "知识产权培训",
    "档案室数字化",
    "呼叫中心排班优化",
    "退换货流程简化",
    "物流月度对账",
    "仓库消防验收整改",
    "食堂承包续约",
    "行业展会展台",
    "公司官网改版",
    "客户满意度调查",
    "季度库存盘点",
    "新员工入职引导",
    "信息安全意识月",
    "增值税发票自查",
    "商标续展申请",
    "电子邮件归档策略",
    "外包坐席质检",
    "样机管理制度",
    "差旅用车平台",
    "办公用品集采",
    "访客接待规范",
    "机房空调维保",
    "测试环境整理",
    "月度经营分析会",
    "供应商实地走访",
    "售后热线扩容",
    "包装耗材降本",
    "月度考勤公示",
    "图书角建设",
    "工牌重新制发",
    "内部讲师津贴",
    "班车线路优化",
    # 系统代号题的载体（正文会注入「内部代号『…』」，见 SYSTEM_CODENAMES）
    "客服系统运维",
    "仓储系统运维",
    "财务系统运维",
    "经营数据看板建设",
    "会议室预约系统",
]

# 系统代号（term 题载体）：全库唯一、纯虚构。键是普通议题名——提到该议题的
# 普通档正文会带「内部代号『…』」字样，term 题的 gold 才有据可查。
SYSTEM_CODENAMES = {
    "客服系统运维": "海豚平台",
    "仓储系统运维": "犀牛系统",
    "财务系统运维": "天平系统",
    "经营数据看板建设": "灯塔看板",
    "会议室预约系统": "鹊桥系统",
}

_TIME_SLOTS = [
    "09:30-10:30",
    "10:00-11:30",
    "14:00-15:30",
    "15:00-16:00",
    "16:30-17:30",
]

_LONG_FILLER = [
    "{dept}通报，{topic}相关事项按上周安排推进，未发现新的风险点，下周期继续跟进。",
    "与会人员就{topic}的资源投入进行讨论，原则同意在现有预算内解决，不另立项。",
    "{dept}提示，{topic}涉及的外部协作方须在合同中明确保密条款与验收标准。",
    "会议要求，{topic}的阶段性结果以书面形式报综合部，统一纳入月度督办清单。",
    "针对{topic}的跨部门协作，明确由{dept}牵头对接，其余部门按职责配合。",
    "会议听取{dept}关于{topic}的例行汇报，各项指标均在正常区间，无需专项处理。",
]

# 扫描件的版面（转图片前先按普通页排）；路线名等短语留给 OCR 兜底后的检索题
_SCAN_DOC_TOPICS = [
    {
        "topic": "员工通勤班车",
        "dept": "行政部",
        "extra": "班车路线为龙华路至科技园，途中设站五处。",
    },
    {
        "topic": "食堂承包续约",
        "dept": "行政部",
        "extra": "新承包合同期两年，餐费标准维持每人每餐 15 元。",
    },
    {
        "topic": "测试环境整理",
        "dept": "信息技术部",
        "extra": "闲置测试服务器清点后统一入库，保留二十台备用。",
    },
    {
        "topic": "访客接待规范",
        "dept": "行政部",
        "extra": "访客须提前一天报备，由前台统一发放临时工牌。",
    },
]

_TABLE_ROW_ITEMS = ["设备购置", "软件许可", "实施服务", "培训宣传", "运维备件"]

# 周次覆盖：2025-W28..W52 + 2026-W02..W42（与历史档重叠无妨）
_WEEK_SLOTS: list[tuple[int, int]] = [(2025, w) for w in range(28, 53)] + [
    (2026, w) for w in range(2, 43)
]


def week_monday(year: int, week: int) -> date:
    """公司周次的周一日期；ISO 不存在的第 53 周钳到 12-31（照 metadata.week_from_filename）。"""
    try:
        return date.fromisocalendar(year, week, 1)
    except ValueError:
        return date(year, 12, 31)


# ---------------------------------------------------------------------------
# 渲染器。全部确定性：固定字体、固定坐标、清空元数据、固定 /ID。
# ---------------------------------------------------------------------------

_PAGE_W, _PAGE_H = 595, 842
_MARGIN_L, _MARGIN_T, _MARGIN_R, _MARGIN_B = 56, 64, 56, 56
_BODY_SIZE = 11.5
_HEAD_SIZE = 13.5
_TITLE_SIZE = 16.0
_LINE_HEIGHT = 1.5


_ID_RE = re.compile(
    rb"/ID\[(?:<[0-9A-Fa-f]*>|\((?:[^()\\]|\\.)*\))"
    rb"(?:<[0-9A-Fa-f]*>|\((?:[^()\\]|\\.)*\))\]"
)


def _fix_pdf_id(data: bytes) -> bytes:
    """PyMuPDF 在 trailer 写随机 /ID——替换为固定值，保证 sha256 幂等。

    实测两种形态都会出现：两个元素都写 <hex>，或第二个元素写括号字符串
    （内容随机字节，可能含转义）。只匹配前者会让一半文档过不了确定性自检。
    """
    return _ID_RE.sub(
        rb"/ID[<" + b"0" * 32 + rb"><" + b"0" * 32 + rb">]",
        data,
    )


def _finalize(pdf: fitz.Document) -> bytes:
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
    )
    # 字体子集化：只嵌入用到的字形（否则 simhei 全量 ~9MB/篇，几百篇无法入库）
    pdf.subset_fonts()
    data = pdf.tobytes(deflate=True, garbage=4)
    pdf.close()
    return _fix_pdf_id(data)


def build_pdf(doc: dict, font: Path) -> bytes:
    """历史档单页渲染（逐字保留原实现，保证 --scale 10 的 doc_id 不变）。"""
    pdf = fitz.open()
    page = pdf.new_page(width=_PAGE_W, height=_PAGE_H)
    fontname = "cjk"
    page.insert_font(fontname=fontname, fontfile=str(font))
    blocks = [doc["title"], ""] + doc["paras"]
    text = "\n\n".join(blocks)
    rect = fitz.Rect(_MARGIN_L, _MARGIN_T, _PAGE_W - _MARGIN_R, _PAGE_H - _MARGIN_B)
    leftover = page.insert_textbox(
        rect,
        text,
        fontname=fontname,
        fontfile=str(font),
        fontsize=_BODY_SIZE,
        lineheight=_LINE_HEIGHT,
        align=fitz.TEXT_ALIGN_LEFT,
    )
    if leftover < 0:  # 单页装不下：续页（历史档长度已控制，此分支为保险）
        page2 = pdf.new_page(width=_PAGE_W, height=_PAGE_H)
        page2.insert_font(fontname=fontname, fontfile=str(font))
        page2.insert_textbox(
            fitz.Rect(_MARGIN_L, _MARGIN_T, _PAGE_W - _MARGIN_R, _PAGE_H - _MARGIN_B),
            text[-int(leftover) :],
            fontname=fontname,
            fontfile=str(font),
            fontsize=_BODY_SIZE,
            lineheight=_LINE_HEIGHT,
        )
    return _finalize(pdf)


@dataclass
class StyledBlock:
    """一个待排版块：kind = title / heading / paragraph。"""

    text: str
    kind: str = "paragraph"
    size: float = _BODY_SIZE


def _para_height(text: str, font: fitz.Font, size: float, avail_w: float) -> float:
    """段落估高（insert_textbox 对 CJK 逐字折行，行数 = 宽度向上取整）。"""
    width = font.text_length(text, fontsize=size)
    lines = max(1, math.ceil(width / avail_w))
    return lines * size * _LINE_HEIGHT


def _draw_blocks(blocks: list[StyledBlock], font_path: Path) -> bytes:
    """多页渲染：段落级分页。insert_textbox 溢出时**不写任何文本**（已实测），
    所以「估高不足 → 换页加高重画」是安全的重试。"""
    font = fitz.Font(fontfile=str(font_path))
    pdf = fitz.open()
    page: fitz.Page | None = None
    y = 0.0
    avail_w = _PAGE_W - _MARGIN_L - _MARGIN_R
    bottom = _PAGE_H - _MARGIN_B

    def new_page() -> fitz.Page:
        nonlocal page, y
        page = pdf.new_page(width=_PAGE_W, height=_PAGE_H)
        page.insert_font(fontname="cjk", fontfile=str(font_path))
        y = _MARGIN_T
        return page

    for block in blocks:
        size = block.size
        extra = 1 if block.kind == "title" else 0  # 标题后留空行
        height = _para_height(block.text, font, size, avail_w) + extra * size
        if page is None or y + height > bottom:
            new_page()
        assert page is not None
        placed = False
        for _attempt in range(4):
            rect = fitz.Rect(
                _MARGIN_L, y, _PAGE_W - _MARGIN_R, min(bottom, y + height + 2)
            )
            leftover = page.insert_textbox(
                rect,
                block.text,
                fontname="cjk",
                fontfile=str(font_path),
                fontsize=size,
                lineheight=_LINE_HEIGHT,
                align=fitz.TEXT_ALIGN_LEFT,
            )
            if leftover >= 0:
                placed = True
                y += height
                break
            # 估高不足：换新页重画该块（每次加高一行）
            new_page()
            height += size * _LINE_HEIGHT
        if not placed:  # 单块超一页（不会发生：段落长度受控）——保底硬失败
            raise SystemExit(f"段落放不下：{block.text[:40]}…")
    return _finalize(pdf)


def _fragmented_pdf(text: str, font_path: Path) -> bytes:
    """碎化导出形态：文本层每字符一行（整篇一个文本对象，行距紧凑），
    覆盖 ingest/pdf.py 的 _merge_lines 纵向合并路径。"""
    pdf = fitz.open()
    avail_h = _PAGE_H - _MARGIN_T - _MARGIN_B
    size = 8.0
    step = size * 1.25
    chars = [c for c in text if not c.isspace()]
    per_page = int(avail_h // step)
    for start in range(0, len(chars), per_page):
        page = pdf.new_page(width=_PAGE_W, height=_PAGE_H)
        page.insert_font(fontname="cjk", fontfile=str(font_path))
        page.insert_text(
            (_MARGIN_L, _MARGIN_T + size),
            "\n".join(chars[start : start + per_page]),
            fontname="cjk",
            fontfile=str(font_path),
            fontsize=size,
            lineheight=1.25,
        )
    return _finalize(pdf)


def _table_pdf(spec: DocSpec, font_path: Path) -> bytes:
    """带框表格：网格线 + 单元格文字，覆盖 find_tables 重建路径。"""
    pdf = fitz.open()
    page = pdf.new_page(width=_PAGE_W, height=_PAGE_H)
    page.insert_font(fontname="cjk", fontfile=str(font_path))
    y = _MARGIN_T
    page.insert_text(
        (_MARGIN_L, y + _TITLE_SIZE),
        spec.title,
        fontname="cjk",
        fontfile=str(font_path),
        fontsize=_TITLE_SIZE,
    )
    y += _TITLE_SIZE * 2.2
    for text, size in spec.pre_table:
        page.insert_text(
            (_MARGIN_L, y + size),
            text,
            fontname="cjk",
            fontfile=str(font_path),
            fontsize=size,
        )
        y += size * 1.9
    col_ws = [150.0, 110.0, 130.0, 120.0]
    x0 = _MARGIN_L
    right_x = x0 + sum(col_ws)
    row_h = 22.0
    rows = spec.table_rows
    top = y
    for r, row in enumerate(rows):
        x = x0
        for c, cell in enumerate(row):
            rect = fitz.Rect(
                x + 1.5,
                top + r * row_h + 1.5,
                x + col_ws[c] - 1.5,
                top + (r + 1) * row_h - 1.5,
            )
            page.insert_textbox(
                rect,
                cell,
                fontname="cjk",
                fontfile=str(font_path),
                fontsize=10.0,
                lineheight=1.2,
            )
            x += col_ws[c]
    bottom = top + len(rows) * row_h
    x = x0
    for w in [*col_ws, 0.0]:
        page.draw_line(fitz.Point(x, top), fitz.Point(x, bottom), width=0.7)
        x += w
    for r in range(len(rows) + 1):
        yy = top + r * row_h
        page.draw_line(fitz.Point(x0, yy), fitz.Point(right_x, yy), width=0.7)
    y = bottom + 14
    for text, size in spec.post_table:
        page.insert_text(
            (_MARGIN_L, y + size),
            text,
            fontname="cjk",
            fontfile=str(font_path),
            fontsize=size,
        )
        y += size * 1.9
    return _finalize(pdf)


def _scan_pdf(spec: DocSpec, font_path: Path) -> bytes:
    """近空 / 扫描件形态：正文先按普通页排版，再整页转位图贴入新文档——
    文本层为零、只有图片（scan_suspect 的「有图无字」路径）。"""
    inner = _draw_blocks(spec.blocks, font_path)
    src = fitz.open(stream=inner, filetype="pdf")
    pdf = fitz.open()
    for src_page in src:
        pix = src_page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x：OCR 可读性
        page = pdf.new_page(width=_PAGE_W, height=_PAGE_H)
        page.insert_image(fitz.Rect(0, 0, _PAGE_W, _PAGE_H), pixmap=pix)
    src.close()
    return _finalize(pdf)


def build_doc_pdf(spec: DocSpec, font: Path) -> bytes:
    """按形态分派渲染器。spec.render ∈ normal / fragmented / table / scan。"""
    render = spec.render
    if render == "normal":
        return _draw_blocks(spec.blocks, font)
    if render == "fragmented":
        return _fragmented_pdf(spec.plain_text, font)
    if render == "table":
        return _table_pdf(spec, font)
    if render == "scan":
        return _scan_pdf(spec, font)
    raise SystemExit(f"未知渲染形态：{render}")


# ---------------------------------------------------------------------------
# v2 语料计划器：确定性地产出每一篇的内容与事实清单。
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    topic: str
    person: str
    status: str  # discuss_only / decide / overturn / reaffirm
    text: str  # 正文中的逐字句（含关键事实）
    key: str  # 答案侧可核对的短语（正文逐字子串）
    amount: str = ""
    unit: str = ""

    def as_dict(self) -> dict:
        return {
            "topic": self.topic,
            "person": self.person,
            "status": self.status,
            "text": self.text,
            "key": self.key,
            "amount": self.amount,
            "unit": self.unit,
        }


@dataclass
class DocSpec:
    file: str
    title: str
    render: str  # normal / fragmented / table / scan
    blocks: list[StyledBlock] = field(default_factory=list)
    plain_text: str = ""  # fragmented 用
    pre_table: list[tuple[str, float]] = field(default_factory=list)
    table_rows: list[list[str]] = field(default_factory=list)
    post_table: list[tuple[str, float]] = field(default_factory=list)
    doc_date: str = ""
    week: dict | None = None
    date_form: str = "full"  # full / week / week_date / week_only
    difficulty: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    owners: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    codenames: list[str] = field(default_factory=list)  # 正文提到的系统代号


class _NamePool:
    """确定性人名池：姓氏 × 名字组合，顺序消耗；断言不重复。"""

    def __init__(self) -> None:
        self._used: set[str] = set()
        self._i = 0

    def take(self) -> str:
        while True:
            s = _SURNAMES[self._i % len(_SURNAMES)]
            k = self._i // len(_SURNAMES)  # 同一姓氏的第二轮起换名字组合
            g1 = _GIVEN_PARTS[0][(self._i * 7 + k) % len(_GIVEN_PARTS[0])]
            g2 = _GIVEN_PARTS[1][(self._i * 5 + k * 3) % len(_GIVEN_PARTS[1])]
            self._i += 1
            name = s + g1 + g2
            if name not in self._used:
                self._used.add(name)
                return name

    def reserve(self, name: str) -> None:
        """显式占用人名（周次钳制档等手写内容里的角色）。"""
        self._used.add(name)


def _unique_amount(used: set[str], base: int, step: int, unit: str) -> str:
    """全库唯一的「金额/数量」短语（判据 key 的区分度来源）。"""
    n = base
    while f"{n} {unit}" in used:
        n += step
    used.add(f"{n} {unit}")
    return str(n)


def _week_tag(year: int, week: int, rng: random.Random) -> str:
    """文件名里的周次片段：两种形态随机（「第」可省）。"""
    if rng.random() < 0.5:
        return f"{year}年第{week}周"
    return f"{year}年{week}周"


def _meeting_date(year: int, week: int, rng: random.Random) -> date:
    return week_monday(year, week) + timedelta(days=rng.randint(0, 4))


def plan_corpus(scale: int) -> tuple[list[DocSpec], list[dict], dict]:
    """确定性规划全部生成档。返回 (specs, series_meta, meta)。"""
    rng = random.Random(SEED)
    names = _NamePool()
    used_amounts: set[str] = {
        "42 万元",
        "28 万元",
        "14 万元",
        "5 万元",
        "70 分",
        "500 元",
        "400 元",
        "2 个工作日",
    }
    specs: list[DocSpec] = []
    series_meta: list[dict] = []

    # 周次槽位：系列从前往后占槽（同一系列的各篇落在递增周），普通档从后往前补
    slots = list(_WEEK_SLOTS)
    rng.shuffle(slots)
    slot_idx = 0

    def next_slot() -> tuple[int, int]:
        nonlocal slot_idx
        s = slots[slot_idx % len(slots)]
        slot_idx += 1
        return s

    # ── 系列档（难点 5）───────────────────────────────────────────────
    for si, spec_def in enumerate(SERIES_SPECS):
        size = _SERIES_SIZES[si]
        topic = spec_def["topic"]
        dept = spec_def["dept"]
        unit = spec_def["unit"]
        owner = names.take()
        weeks = sorted(rng.sample(range(len(slots)), size))
        # 周必须递增：直接取 slots 的有序子集（slots 已随机，但取出的再排序）
        week_pairs = sorted((slots[i] for i in weeks), key=lambda t: (t[0], t[1]))
        # 状态机：discuss_only → decide → (reaffirm | overturn → decide)
        statuses: list[str] = []
        st = _STATUS_DISCUSS if si % 2 == 0 else _STATUS_DECIDE
        statuses.append(st)
        while len(statuses) < size:
            if st == _STATUS_DISCUSS:
                st = _STATUS_DECIDE
            elif st == _STATUS_DECIDE:
                st = (
                    _STATUS_REAFFIRM
                    if (si + len(statuses)) % 3 != 1
                    else _STATUS_OVERTURN
                )
            elif st == _STATUS_OVERTURN:
                st = _STATUS_DECIDE
            else:  # reaffirm：已定案，后续周维持既定决议
                st = _STATUS_REAFFIRM
            statuses.append(st)
        doc_ids: list[str] = []
        for di, (wy, ww) in enumerate(week_pairs):
            status = statuses[di]
            amount = _unique_amount(
                used_amounts,
                spec_def["base"] + di * (spec_def["base"] // 10 + 3),
                7,
                unit,
            )
            detail_a = _SERIES_DETAIL_A[(si + di) % len(_SERIES_DETAIL_A)]
            detail_b = _SERIES_DETAIL_B[(si + di) % len(_SERIES_DETAIL_B)]
            meeting = _meeting_date(wy, ww, rng)
            tag = _week_tag(wy, ww, rng)
            # 形态：1/3 带「周次+完整日期」，2/3 只周次（doc_date 走周次解析路径）
            if rng.random() < 0.34:
                fname_date = meeting.isoformat()
                date_form = "week_date"
                file = f"会议档案_办公会_{tag}-会议纪要_{fname_date}_yf{200 + si * 20 + di:03d}.pdf"
                doc_date = meeting.isoformat()
            else:
                date_form = "week_only"
                file = f"会议档案_办公会_{tag}-会议纪要_yf{200 + si * 20 + di:03d}.pdf"
                doc_date = week_monday(wy, ww).isoformat()
            title = f"云帆科技 {wy} 年第 {ww} 周办公会会议纪要"
            header = (
                f"会议时间：{meeting.year}年{meeting.month}月{meeting.day}日 "
                f"{rng.choice(_TIME_SLOTS)}　主持人：{_HOSTS[si % len(_HOSTS)]}　记录：{_RECORDER}"
            )
            if status == _STATUS_DISCUSS:
                body = (
                    f"一、{topic}进展汇报。{dept}{owner}汇报，{detail_a}。与会人员对{detail_b}"
                    "意见不一，本场议题未形成决议，要求相关部门补充测算后提交下次会议再议。"
                )
                claim = Claim(
                    topic=topic,
                    person=owner,
                    status=status,
                    text="与会人员对"
                    + detail_b
                    + "意见不一，本场议题未形成决议，要求相关部门补充测算后提交下次会议再议。",
                    key="未形成决议",
                )
                blocks = [
                    StyledBlock(title, "title", _TITLE_SIZE),
                    StyledBlock(header),
                    StyledBlock(f"一、{topic}进展汇报", "heading", _HEAD_SIZE),
                    StyledBlock(f"{dept}{owner}汇报，{detail_a}。"),
                    StyledBlock(body.split("。", 1)[1].strip("。") + "。"),
                    StyledBlock("决议区", "heading", _HEAD_SIZE),
                    StyledBlock(f"① {topic}议题挂起，待补充测算后重议。"),
                ]
            else:
                if status == _STATUS_DECIDE:
                    core = (
                        f"{dept}{owner}汇报，{detail_a}。会议同意按该方案推进，预算总额 {amount} {unit}，"
                        f"由{dept}牵头、相关部门配合。"
                    )
                    res = f"决议区：① {topic}方案予以通过；② 预算 {amount} {unit} 列入年度计划。"
                elif status == _STATUS_OVERTURN:
                    core = (
                        f"{dept}{owner}汇报，经复核{detail_b}存在较大偏差，此前通过的原方案停止执行，"
                        f"按修订方案重新立项，预算调整为 {amount} {unit}。"
                    )
                    res = f"决议区：① {topic}原方案停止执行；② 修订方案按 {amount} {unit} 重新立项。"
                else:  # reaffirm
                    core = (
                        f"{dept}{owner}汇报，{detail_a}，整体符合预期。会议重申按既定决议执行，"
                        f"并追加 {amount} {unit} 用于{detail_b}。"
                    )
                    res = f"决议区：① {topic}按既定决议继续执行；② 追加预算 {amount} {unit}。"
                claim = Claim(
                    topic=topic,
                    person=owner,
                    status=status,
                    text=core,
                    key=f"{amount} {unit}",
                    amount=amount,
                    unit=unit,
                )
                blocks = [
                    StyledBlock(title, "title", _TITLE_SIZE),
                    StyledBlock(header),
                    StyledBlock(
                        f"一、{topic}{'方案调整' if status == _STATUS_OVERTURN else '推进汇报' if status == _STATUS_REAFFIRM else '方案审议'}",
                        "heading",
                        _HEAD_SIZE,
                    ),
                    StyledBlock(core),
                    StyledBlock("决议区", "heading", _HEAD_SIZE),
                    StyledBlock(res.lstrip("决议区：")),
                ]
            spec = DocSpec(
                file=file,
                title=title,
                render="normal",
                blocks=blocks,
                doc_date=doc_date,
                week={"year": wy, "n": ww},
                date_form=date_form,
                difficulty=["series", "week_name"],
                topics=[topic],
                owners=[owner],
                claims=[claim],
            )
            specs.append(spec)
            doc_ids.append(file)
        series_meta.append(
            {
                "topic": topic,
                "dept": dept,
                "owner": owner,
                "files": doc_ids,
                "statuses": statuses,
            }
        )

    # ── 表格档（难点 3）───────────────────────────────────────────────
    for ti in range(8):
        topic = ORDINARY_TOPICS[ti]
        dept = _DEPTS[ti % len(_DEPTS)]
        owner = names.take()
        wy, ww = slots[(slot_idx + ti * 3) % len(slots)]
        meeting = _meeting_date(wy, ww, rng)
        rows_total = 0
        row_amounts = []
        for r in range(4):
            amt = _unique_amount(used_amounts, 8 + ti * 11 + r * 5, 3, "万元")
            row_amounts.append(int(amt))
        rows_total = sum(row_amounts)
        table_rows = [["项目", "金额（万元）", "责任部门", "完成时限"]]
        for r, item in enumerate(_TABLE_ROW_ITEMS[:4]):
            table_rows.append(
                [
                    item,
                    str(row_amounts[r]),
                    dept,
                    f"{meeting.year}-{meeting.month + r:02d}",
                ]
            )
        total = f"{rows_total} 万元"
        used_amounts.add(total)
        title = f"云帆科技 {meeting.year} 年第 {ww} 周专题会会议纪要"
        header = (
            f"会议时间：{meeting.year}年{meeting.month}月{meeting.day}日 14:00-15:00　"
            f"主持人：{_HOSTS[ti % len(_HOSTS)]}　记录：{_RECORDER}"
        )
        intro = f"一、{topic}预算明细。{dept}{owner}汇报，预算总额 {total}，分项如下："
        res = f"决议区：① {topic}预算 {total} 予以通过；② 分项执行按上表时限推进。"
        spec = DocSpec(
            file=f"会议档案_专题会_{meeting.isoformat()}-会议纪要_yf{300 + ti:03d}.pdf",
            title=title,
            render="table",
            pre_table=[(header, _BODY_SIZE), (intro, _BODY_SIZE)],
            table_rows=table_rows,
            post_table=[(res, _BODY_SIZE)],
            doc_date=meeting.isoformat(),
            week={"year": wy, "n": ww},
            date_form="full",
            difficulty=["table"],
            topics=[topic],
            owners=[owner],
            claims=[
                Claim(
                    topic=topic,
                    person=owner,
                    status=_STATUS_DECIDE,
                    text=intro,
                    key=total,
                    amount=str(rows_total),
                    unit="万元",
                ),
                Claim(
                    topic=topic,
                    person=owner,
                    status=_STATUS_DECIDE,
                    # 单元格里的逐字片段就是数字本身；出题侧（make_gold_from_corpus）
                    # 把它和行项目名配成 must_contain 两项
                    text=str(row_amounts[0]),
                    key=str(row_amounts[0]),
                    amount=str(row_amounts[0]),
                    unit="万元",
                ),
            ],
        )
        specs.append(spec)

    # ── 扫描件档（难点 4）─────────────────────────────────────────────
    for si2, sdef in enumerate(_SCAN_DOC_TOPICS):
        owner = names.take()
        wy, ww = slots[(slot_idx + si2 * 7 + 5) % len(slots)]
        meeting = _meeting_date(wy, ww, rng)
        topic = sdef["topic"]
        dept = sdef["dept"]
        title = f"云帆科技 {meeting.year} 年第 {ww} 周办公会会议纪要"
        header = (
            f"会议时间：{meeting.year}年{meeting.month}月{meeting.day}日 15:00-16:00　"
            f"主持人：{_HOSTS[(si2 + 1) % len(_HOSTS)]}　记录：{_RECORDER}"
        )
        body = f"一、{topic}。{dept}{owner}汇报，{sdef['extra']}本场议题形成决议，按汇报方案执行。"
        res = f"决议区：① {topic}按汇报方案执行。"
        claim_text = f"{dept}{owner}汇报，{sdef['extra']}"
        key = sdef["extra"].split("，")[0][
            :12
        ]  # OCR 友好的短语（见模块 docstring 的限制）
        blocks = [
            StyledBlock(title, "title", _TITLE_SIZE),
            StyledBlock(header),
            StyledBlock(f"一、{topic}", "heading", _HEAD_SIZE),
            StyledBlock(
                f"{dept}{owner}汇报，{sdef['extra']}本场议题形成决议，按汇报方案执行。"
            ),
            StyledBlock("决议区", "heading", _HEAD_SIZE),
            StyledBlock(res.lstrip("决议区：")),
        ]
        spec = DocSpec(
            file=f"会议档案_办公会_{meeting.isoformat()}-会议纪要_yf{400 + si2:03d}.pdf",
            title=title,
            render="scan",
            blocks=blocks,
            doc_date=meeting.isoformat(),
            week={"year": wy, "n": ww},
            date_form="full",
            difficulty=["scan"],
            topics=[topic],
            owners=[owner],
            claims=[
                Claim(
                    topic=topic,
                    person=owner,
                    status=_STATUS_DECIDE,
                    text=claim_text,
                    key=key,
                )
            ],
        )
        specs.append(spec)

    # ── 碎化档（难点 2）───────────────────────────────────────────────
    for fi in range(4):
        topic = ORDINARY_TOPICS[8 + fi]
        dept = _DEPTS[(fi + 3) % len(_DEPTS)]
        owner = names.take()
        wy, ww = slots[(slot_idx + fi * 11 + 2) % len(slots)]
        meeting = _meeting_date(wy, ww, rng)
        amount = _unique_amount(used_amounts, 15 + fi * 9, 4, "万元")
        title = f"云帆科技 {meeting.year} 年第 {ww} 周专题会会议纪要"
        header = (
            f"会议时间：{meeting.year}年{meeting.month}月{meeting.day}日 10:00-11:00　"
            f"主持人：{_HOSTS[fi % len(_HOSTS)]}　记录：{_RECORDER}"
        )
        body = (
            f"一、{topic}方案汇报。{dept}{owner}汇报，项目预算 {amount} 万元，工期四个月，"
            "分两阶段验收，验收标准按附件执行。"
        )
        res = f"决议区：① {topic}预算 {amount} 万元 予以通过。"
        plain = "\n".join([title, header, f"一、{topic}方案汇报", body, res])
        claim = Claim(
            topic=topic,
            person=owner,
            status=_STATUS_DECIDE,
            text=f"项目预算 {amount} 万元，工期四个月",
            key=f"{amount} 万元",
            amount=amount,
            unit="万元",
        )
        spec = DocSpec(
            file=f"会议档案_专题会_{meeting.isoformat()}-会议纪要_yf{410 + fi:03d}.pdf",
            title=title,
            render="fragmented",
            plain_text=plain,
            doc_date=meeting.isoformat(),
            week={"year": wy, "n": ww},
            date_form="full",
            difficulty=["fragmented"],
            topics=[topic],
            owners=[owner],
            claims=[claim],
        )
        specs.append(spec)

    # ── 第 53 周钳制档（难点 1 的边界回归）────────────────────────────
    owner = names.take()
    spec = DocSpec(
        file="会议档案_办公会_2024年第53周-会议纪要_yf2453.pdf",
        title="云帆科技 2024 年第 53 周办公会会议纪要",
        render="normal",
        blocks=[
            StyledBlock("云帆科技 2024 年第 53 周办公会会议纪要", "title", _TITLE_SIZE),
            StyledBlock(
                "会议时间：2024年12月31日 15:00-16:00　主持人：高翔　记录：沈其芳"
            ),
            StyledBlock("一、年度收尾事项", "heading", _HEAD_SIZE),
            StyledBlock(
                "行政部魏子轩汇报，年度固定资产清点完成 96%，剩余少量待财务复核。"
            ),
            StyledBlock("决议区", "heading", _HEAD_SIZE),
            StyledBlock("① 年度收尾事项按清单销项；② 未完事项转入次年第一周。"),
        ],
        doc_date="2024-12-31",
        week={"year": 2024, "n": 53},
        date_form="week_only",
        difficulty=["week53", "week_name"],
        topics=["年度收尾事项"],
        owners=["魏子轩"],
        claims=[
            Claim(
                topic="年度收尾事项",
                person="魏子轩",
                status=_STATUS_DECIDE,
                text="年度固定资产清点完成 96%",
                key="96%",
                amount="96",
                unit="%",
            )
        ],
    )
    specs.append(spec)
    names.reserve("魏子轩")

    # ── 长纪要（8~15 页）─────────────────────────────────────────────
    for li in range(12):
        owner = names.take()
        wy, ww = slots[(slot_idx + li * 5 + 3) % len(slots)]
        meeting = _meeting_date(wy, ww, rng)
        t1 = ORDINARY_TOPICS[(li * 3) % len(ORDINARY_TOPICS)]
        t2 = ORDINARY_TOPICS[(li * 3 + 1) % len(ORDINARY_TOPICS)]
        dept1 = _DEPTS[li % len(_DEPTS)]
        dept2 = _DEPTS[(li + 4) % len(_DEPTS)]
        amount = _unique_amount(used_amounts, 21 + li * 6, 5, "万元")
        title = f"云帆科技 {meeting.year} 年第 {ww} 周办公会会议纪要（扩程）"
        header = (
            f"会议时间：{meeting.year}年{meeting.month}月{meeting.day}日 14:00-16:30　"
            f"主持人：{_HOSTS[li % len(_HOSTS)]}　记录：{_RECORDER}"
        )
        blocks: list[StyledBlock] = [
            StyledBlock(title, "title", _TITLE_SIZE),
            StyledBlock(header),
        ]
        claims: list[Claim] = []
        body1 = (
            f"{dept1}{owner}汇报，{t1}方案进入实施阶段，预算 {amount} 万元，已完成供应商定标与合同签署，"
            "计划下月启动分批部署，部署期间原有流程并行运行两周后切换。"
        )
        claims.append(
            Claim(
                topic=t1,
                person=owner,
                status=_STATUS_DECIDE,
                text=f"预算 {amount} 万元，已完成供应商定标与合同签署",
                key=f"{amount} 万元",
                amount=amount,
                unit="万元",
            )
        )
        blocks.extend(
            [
                StyledBlock(f"一、{t1}实施安排", "heading", _HEAD_SIZE),
                StyledBlock(body1),
            ]
        )
        filler_topics = [t2, t1, t2, t1, t2]
        filler_depts = [dept2, dept1, dept2, dept1, dept2]
        for k in range(8):  # 凑 8~15 页正文
            ft = filler_topics[k % len(filler_topics)]
            fd = filler_depts[k % len(filler_depts)]
            blocks.append(
                StyledBlock(f"{k + 2}、例行通报 {k + 1}", "heading", _HEAD_SIZE)
            )
            blocks.append(
                StyledBlock(
                    _LONG_FILLER[k % len(_LONG_FILLER)].format(topic=ft, dept=fd)
                )
            )
            if k % 3 == 1:
                blocks.append(
                    StyledBlock(
                        f"{fd}补充说明，{ft}的时间安排以月度经营分析会的口径为准，"
                        "相关材料提前三个工作日发综合部汇总。"
                    )
                )
        blocks.append(StyledBlock("决议区", "heading", _HEAD_SIZE))
        blocks.append(
            StyledBlock(
                f"决议区：① {t1}按实施安排推进，预算 {amount} 万元 不变；"
                "② 例行通报事项按各自口径落实。"
            )
        )
        spec = DocSpec(
            file=f"会议档案_办公会_{meeting.isoformat()}-会议纪要_yf{500 + li:03d}.pdf",
            title=title,
            render="normal",
            blocks=blocks,
            doc_date=meeting.isoformat(),
            week={"year": wy, "n": ww},
            date_form="full",
            difficulty=["long"],
            topics=[t1, t2],
            owners=[owner],
            claims=claims,
        )
        specs.append(spec)

    # ── 普通短纪要（补到 310 篇生成档）───────────────────────────────
    generated_target = 310
    base_count = len(specs)
    for oi in range(generated_target - base_count):
        owner = names.take()
        wy, ww = slots[(slot_idx + oi) % len(slots)]
        meeting = _meeting_date(wy, ww, rng)
        t1 = ORDINARY_TOPICS[oi % len(ORDINARY_TOPICS)]
        t2 = ORDINARY_TOPICS[(oi * 7 + 3) % len(ORDINARY_TOPICS)]
        dept1 = _DEPTS[oi % len(_DEPTS)]
        decided = oi % 10 < 7  # 70% 有决议
        amount = _unique_amount(used_amounts, 9 + oi % 40, 3, "万元")
        tag = _week_tag(wy, ww, rng)
        if oi == 0:
            # 钉住「2026年42周」（无「第」）这个具体回归用例：metadata 的日期正则
            # 曾把它回溯拆成 2026-04-02，(?!周) 修复后必须解析为 2026-W42 的周一。
            wy, ww = 2026, 42
            meeting = _meeting_date(wy, ww, rng)
            tag = f"{wy}年{ww}周"
        roll = rng.random()
        if oi == 0:
            roll = 0.5  # 强制走 week_only 分支，doc_date 只能来自周次解析
        if roll < 0.34:
            file = f"会议档案_办公会_{tag}-会议纪要_{meeting.isoformat()}_yf{600 + oi:03d}.pdf"
            doc_date = meeting.isoformat()
            date_form = "week_date"
            week_info: dict | None = {"year": wy, "n": ww}
        elif roll < 0.67:
            file = f"会议档案_办公会_{tag}-会议纪要_yf{600 + oi:03d}.pdf"
            doc_date = week_monday(wy, ww).isoformat()
            date_form = "week_only"
            week_info = {"year": wy, "n": ww}
        else:
            file = (
                f"会议档案_办公会_会议纪要_{meeting.isoformat()}_yf{600 + oi:03d}.pdf"
            )
            doc_date = meeting.isoformat()
            date_form = "full"
            week_info = None
        title = f"云帆科技 {wy} 年第 {ww} 周办公会会议纪要"
        header = (
            f"会议时间：{meeting.year}年{meeting.month}月{meeting.day}日 "
            f"{rng.choice(_TIME_SLOTS)}　主持人：{_HOSTS[oi % len(_HOSTS)]}　记录：{_RECORDER}"
        )
        claims: list[Claim] = []
        code = SYSTEM_CODENAMES.get(t1)
        code_note = f"（内部代号「{code}」）" if code else ""
        if decided:
            body1 = (
                f"{dept1}{owner}汇报，{t1}项目{code_note}方案已具备执行条件，"
                f"预算 {amount} 万元，按现有人员分工推进，不新增编制。"
            )
            claims.append(
                Claim(
                    topic=t1,
                    person=owner,
                    status=_STATUS_DECIDE,
                    text=body1,
                    key=f"{amount} 万元",
                    amount=amount,
                    unit="万元",
                )
            )
            res = f"决议区：① {t1}预算 {amount} 万元 予以通过。"
        else:
            body1 = (
                f"{dept1}{owner}汇报，{t1}项目{code_note}的两个备选方案各有侧重，"
                "与会人员意见不一，本场议题未形成决议，要求补充成本测算后提交下次会议再议。"
            )
            claims.append(
                Claim(
                    topic=t1,
                    person=owner,
                    status=_STATUS_DISCUSS,
                    text=body1,
                    key="未形成决议",
                )
            )
            res = f"决议区：① {t1}议题挂起，待补充测算后重议。"
        body2 = (
            f"二、{t2}情况通报。相关数据显示整体进展正常，无需要专项协调的事项，"
            "下一步按既有计划执行。"
        )
        blocks = [
            StyledBlock(title, "title", _TITLE_SIZE),
            StyledBlock(header),
            StyledBlock(f"一、{t1}情况汇报", "heading", _HEAD_SIZE),
            StyledBlock(body1),
            StyledBlock(f"二、{t2}情况通报", "heading", _HEAD_SIZE),
            StyledBlock(body2),
            StyledBlock("决议区", "heading", _HEAD_SIZE),
            StyledBlock(res.lstrip("决议区：")),
        ]
        spec = DocSpec(
            file=file,
            title=title,
            render="normal",
            blocks=blocks,
            doc_date=doc_date,
            week=week_info,
            date_form=date_form,
            difficulty=["week_name"] if week_info else ["ordinary"],
            topics=[t1, t2],
            owners=[owner],
            claims=claims,
            codenames=[code] if code else [],
        )
        specs.append(spec)

    meta = {
        "company": "云帆科技（虚构）",
        "seed": SEED,
        "scale": scale,
        "off_corpus_terms": OFF_CORPUS_TERMS,
        "codenames": SYSTEM_CODENAMES,
        "hosts": _HOSTS + [_RECORDER],
        "counts": {},
    }
    return specs, series_meta, meta


# ---------------------------------------------------------------------------
# 自检 + 落盘
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def run(scale: int) -> None:
    font = find_cjk_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gold = _QUESTION

    # 生成前先做纯文本校验：域外词绝不允许出现（no_answer 题依赖全库不可答）
    joined = "\n".join("\n".join(d["paras"]) for d in DOCS)

    specs: list[DocSpec] = []
    series_meta: list[dict] = []
    meta: dict = {}
    if scale >= 300:
        specs, series_meta, meta = plan_corpus(scale)
        joined += (
            "\n"
            + "\n".join(b.text for s in specs for b in s.blocks)
            + "\n"
            + "\n".join(s.plain_text for s in specs if s.plain_text)
        )
        joined += "\n" + "\n".join(
            c
            for s in specs
            if s.render == "table"
            for c, _ in (s.pre_table + s.post_table)
        )
        joined += "\n" + " ".join(
            cell
            for s in specs
            if s.render == "table"
            for row in s.table_rows
            for cell in row
        )
    for term in OFF_CORPUS_TERMS:
        assert term not in joined, (
            f"域外词「{term}」出现在示例语料中，no_answer 题会失效"
        )

    # ── 构建 PDF（两遍逐字节比对 = 确定性自检）─────────────────────────
    legacy_bytes = {d["file"]: build_pdf(d, font) for d in DOCS}
    gen_bytes: dict[str, bytes] = {}
    gen_bytes2: dict[str, bytes] = {}
    for spec in specs:
        gen_bytes[spec.file] = build_doc_pdf(spec, font)
    for spec in specs:
        gen_bytes2[spec.file] = build_doc_pdf(spec, font)
    mismatch = [f for f in gen_bytes if gen_bytes[f] != gen_bytes2[f]]
    if mismatch:
        print(f"FAIL 确定性自检：{len(mismatch)} 篇两次构建不一致，例如 {mismatch[:3]}")
        raise SystemExit(1)

    # 清空输出目录后落盘（切档不留残 file）
    for old in OUT_DIR.glob("*.pdf"):
        old.unlink()
    id_by_file: dict[str, str] = {}
    for fname, data in {**legacy_bytes, **gen_bytes}.items():
        path = OUT_DIR / fname
        path.write_bytes(data)
        id_by_file[fname] = hashlib.sha256(data).hexdigest()[:16]

    # ── 解析往返自检 ─────────────────────────────────────────────────
    sys.path.insert(0, str(ROOT / "src"))
    from doc_rag.ingest.pdf import extract_pdf

    text_by_file: dict[str, str] = {}
    for fname in id_by_file:
        text_by_file[fname] = extract_pdf(OUT_DIR / fname).to_text()

    failures: list[str] = []

    # 1) 历史档：golden_sample must_contain 逐字命中（--scale 10 兼容不变）
    for item in gold["items"]:
        qid, qtype = item["id"], item["type"]
        if qtype == "no_answer":
            for kw in OFF_CORPUS_TERMS[:1] + item["must_contain"]:
                hits = [f for f, t in text_by_file.items() if kw in _norm(t)]
                if hits:
                    failures.append(f"{qid}: no_answer 词「{kw}」出现在 {hits}")
            continue
        keys = QUESTION_DOC_MAP.get(qid)
        if not keys:
            failures.append(f"{qid}: QUESTION_DOC_MAP 缺少映射")
            continue
        corpus = _norm(
            "\n".join(t for f, t in text_by_file.items() if any(k in f for k in keys))
        )
        for kw in item["must_contain"]:
            if _norm(kw) not in corpus:
                failures.append(
                    f"{qid}({qtype}): must_contain「{kw}」未逐字命中来源文档"
                )

    # 2) 生成档：主张逐字命中；扫描件必须无可抽文本层
    for spec in specs:
        text = _norm(text_by_file.get(spec.file, ""))
        if spec.render == "scan":
            if len(text) > 8:
                failures.append(f"{spec.file}: 扫描件竟然有文本层（{len(text)} 字）")
            continue
        for claim in spec.claims:
            if _norm(claim.text) not in text or _norm(claim.key) not in text:
                failures.append(
                    f"{spec.file}: 主张「{claim.key}」未在解析文本中逐字命中"
                )
    if failures:
        print("\n".join(f"FAIL {f}" for f in failures[:40]))
        raise SystemExit(1)

    # 3) 归属人名全库唯一：只出现在自己的文档簇
    owner_files: dict[str, list[str]] = {}
    for spec in specs:
        for owner in spec.owners:
            owner_files.setdefault(owner, []).append(spec.file)
    for owner, files in owner_files.items():
        hits = [f for f, t in text_by_file.items() if owner in _norm(t)]
        stray = sorted(set(hits) - set(files))
        if stray:
            failures.append(f"人名「{owner}」溢出到 {stray[:3]}")
    if failures:
        print("\n".join(f"FAIL {f}" for f in failures[:20]))
        raise SystemExit(1)

    # 4) 五类难点计数达标（仅完整档；--scale 10 没有生成档）
    diff_counts: dict[str, int] = {}
    for spec in specs:
        for d in spec.difficulty:
            diff_counts[d] = diff_counts.get(d, 0) + 1
    if scale >= 300:
        for need in ("week_name", "fragmented", "table", "scan", "series"):
            if diff_counts.get(need, 0) < 3:
                failures.append(
                    f"难点「{need}」只有 {diff_counts.get(need, 0)} 篇（要求 ≥3）"
                )
        if failures:
            print("\n".join(f"FAIL {f}" for f in failures))
            raise SystemExit(1)

    # ── 回填 golden_sample.json（历史档兼容）──────────────────────────
    for item in gold["items"]:
        keys = QUESTION_DOC_MAP.get(item["id"], [])
        item["source_doc_ids"] = [
            next(v for f, v in id_by_file.items() if k in f) for k in keys
        ]
    GOLD_FILE.write_text(
        json.dumps(gold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ── manifest 落盘（gold 派生的唯一输入；仅完整档，别用空档覆盖它）──
    if scale >= 300:
        manifest = {
            "meta": {
                **meta,
                "note": (
                    "全虚构语料的事实清单：scripts/make_gold_from_corpus.py 据此程序化派生"
                    "公开黄金集（零 LLM 成本）。doc_id = 文件 sha256 前 16 hex，与 ingest 同源。"
                ),
                "legacy_files": [d["file"] for d in DOCS],
            },
            "docs": [
                {
                    "file": spec.file,
                    "doc_id": id_by_file[spec.file],
                    "title": spec.title,
                    "doc_date": spec.doc_date,
                    "week": spec.week,
                    "date_form": spec.date_form,
                    "difficulty": spec.difficulty,
                    "topics": spec.topics,
                    "owners": spec.owners,
                    "claims": [c.as_dict() for c in spec.claims],
                    "codenames": spec.codenames,
                }
                for spec in specs
            ],
            "series": [
                {
                    **s,
                    "files": [{"file": f, "doc_id": id_by_file[f]} for f in s["files"]],
                }
                for s in series_meta
            ],
        }
        meta_counts: dict[str, int] = dict(diff_counts)
        meta_counts["legacy"] = len(DOCS)
        manifest["meta"]["counts"] = meta_counts
        manifest["meta"]["total_docs"] = len(id_by_file)
        MANIFEST_FILE.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        f"生成 {len(id_by_file)} 篇示例 PDF（历史档 {len(DOCS)} + 生成档 {len(specs)}）→ {OUT_DIR}"
    )
    print("难点分布：" + json.dumps(diff_counts, ensure_ascii=False))
    print(f"确定性自检：{len(gen_bytes)} 篇生成档两次构建逐字节相同 ✓")
    print(f"已回填 {GOLD_FILE.relative_to(ROOT)} 的 source_doc_ids")
    print(f"事实清单 → {MANIFEST_FILE.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成虚构示例语料（见模块 docstring）")
    parser.add_argument(
        "--scale",
        choices=["10", "300"],
        default="300",
        help="10=历史档（golden_sample.json 兼容）；300=完整底座（历史档+生成档）",
    )
    args = parser.parse_args()
    run(int(args.scale))


if __name__ == "__main__":
    main()

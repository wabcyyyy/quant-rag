"""示例语料生成器（scripts/make_sample_corpus.py）的离线护栏。

完整档的生成 + 自检在脚本内做（含解析往返与确定性自检，需要 CJK 字体，
跑一遍约 1~2 分钟，不搬进 pytest）；这里钉住的是外部消费方依赖的契约：
1. golden_sample.json 的 source_doc_ids 与 data/sample_raw 的真实 doc_id 咬合
   （--scale 10 向后兼容的硬承诺）；
2. manifest（A2 出金子集的唯一输入）与语料一致：doc_id、doc_date 全覆盖、
   五类难点计数、域外词绝不出现在任何虚构内容里；
3. PDF 随机 /ID 的两种序列化形态都被归一（曾漏掉括号形态，一半文档过不了确定性自检）；
4. 同一 spec 构建两遍逐字节相同（确定性承诺的抽检）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import make_sample_corpus as msc

SAMPLE_DIR = ROOT / "data" / "sample_raw"
MANIFEST = ROOT / "data" / "eval" / "sample_corpus_manifest.json"


def _doc_ids_on_disk() -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        for p in SAMPLE_DIR.glob("*.pdf")
    }


def test_golden_sample_doc_ids_match_files() -> None:
    gold = json.loads(
        (ROOT / "data" / "eval" / "golden_sample.json").read_text("utf-8")
    )
    on_disk = set(_doc_ids_on_disk().values())
    for item in gold["items"]:
        for did in item["source_doc_ids"]:
            assert did in on_disk, (
                f"{item['id']} 的 source_doc_id {did} 不在 data/sample_raw 里——"
                "历史档被改动过？golden_sample.json 与语料必须一起重生成"
            )


def test_manifest_matches_corpus() -> None:
    if not MANIFEST.exists():
        raise AssertionError(
            "manifest 缺失：先跑 uv run python scripts/make_sample_corpus.py --scale 300"
        )
    m = json.loads(MANIFEST.read_text("utf-8"))
    docs = m["docs"]
    assert len(docs) >= 300, f"生成档 {len(docs)} 篇，达不到 300 下限"
    on_disk = _doc_ids_on_disk()
    for d in docs:
        assert on_disk.get(d["file"]) == d["doc_id"], f"doc_id 与文件不符：{d['file']}"
        assert d["doc_date"], f"doc_date 缺失（要求 100% 覆盖）：{d['file']}"
        assert d["owners"] and d["claims"], f"缺归属人或主张：{d['file']}"

    diff: dict[str, int] = {}
    for d in docs:
        for k in d["difficulty"]:
            diff[k] = diff.get(k, 0) + 1
    for need in ("week_name", "fragmented", "table", "scan", "series"):
        assert diff.get(need, 0) >= 3, f"难点「{need}」只有 {diff.get(need, 0)} 篇"

    # 域外词在 manifest 的全部虚构内容里都不出现（no_answer 题依赖全库不可答）。
    # 只查 docs/series 正文事实；meta.off_corpus_terms 字段本身就列着这些词。
    blob = json.dumps({"docs": m["docs"], "series": m["series"]}, ensure_ascii=False)
    for term in m["meta"]["off_corpus_terms"]:
        assert term not in blob, f"域外词「{term}」出现在 manifest 中"

    # 系列的周次必须递增（时效/决议演化题的前提）
    weeks = {d["file"]: d.get("week") for d in docs}
    for s in m["series"]:
        ws = [weeks[f["file"]] for f in s["files"]]
        keys = [(w["year"], w["n"]) for w in ws]
        assert keys == sorted(keys), f"系列 {s['topic']} 的周次不递增：{keys}"


def test_pdf_id_fix_handles_paren_form() -> None:
    fixed = b"/ID[<00000000000000000000000000000000><00000000000000000000000000000000>]"
    hex_hex = (
        b"trailer<</Size 13/ID[<0A1B2C3D4E5F60718293A4B5C6D7E8F9>"
        b"<0A1B2C3D4E5F60718293A4B5C6D7E8F9>]>>startxref"
    )
    paren = (
        b"trailer<</Size 13/ID[(\\034\\302\\225\\)|ML#45/~\\303\\204f<)"
        b"<97EC2C2C28F9A5695341A08CF1C31FDE>]>>startxref"
    )
    paren_paren = b"trailer<</Size 13/ID[(\\034\\302)(abc\\)def)]>>startxref"
    for raw in (hex_hex, paren, paren_paren):
        assert msc._fix_pdf_id(raw).count(fixed[:26]) == 1, raw


def test_plan_and_build_deterministic() -> None:
    font = msc.find_cjk_font()
    specs, _, _ = msc.plan_corpus(300)
    assert len(specs) >= 300
    by_render: dict[str, object] = {}
    for spec in specs:
        by_render.setdefault(spec.render, spec)
    for spec in by_render.values():  # 每种渲染形态各抽一篇
        assert msc.build_doc_pdf(spec, font) == msc.build_doc_pdf(spec, font), (
            f"两次构建不一致：{spec.file}"
        )

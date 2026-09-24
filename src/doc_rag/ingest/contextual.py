"""Contextual retrieval 的前缀生成（A3.4，机制见 ADR-0004）。

入库时为每个 chunk 生成一段**上下文前缀**（这个片段讨论什么、在文档什么位置），
拼在被索引文本最前再嵌入/分词——业界通称 contextual retrieval（Anthropic 2024）。
动机：会议纪要语料高度同质，块正文常常「脱离文档就不知道在说哪一周的哪件事」，
前缀把文档级定位信息带进每块的索引表示。

成本与失败纪律（都照既有先例）：
- 走 LLM 响应缓存（键含模型+prompt）：同 chunk 重入库零成本，臂 ②/③ 共用同一批前缀；
- `reasoning_effort=none`（照 judge/rewrite 实测先例：短抽取不需要思考，省 90% 输出 token）；
- 单 chunk 失败不阻塞入库，降级为「无前缀」并计数（照 metadata.extract_metadata）；
- 调用前先 `--limit` 试算外推，超预算即停（SPEC 铁律 2）。

转正与否是产品决定（SPEC U3 待用户拍板），所以 `contextual.enabled` 默认 false；
评估臂走 `ingest --ctx both|bm25`（消融三臂：无前缀 / 前缀进双路 / 前缀只进 BM25）。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..generate import llm
from .schema import Chunk, IntermediateDoc

_WORKERS = 8
_CHUNK_TEXT_CHARS = 800  # 送进 prompt 的块正文上限（前缀不需要全文）
_PREFIX_CHARS = 120  # 前缀本身的上限（更长会在嵌入里稀释正文）

_PROMPT = (
    "文档《{title}》的片段（章节路径：{section}；第 {page} 页）：\n{body}\n"
    "请用不超过 60 字写一句上下文说明：这个片段讨论什么事项、处于文档什么位置，"
    "供检索系统判断相关性。直接输出这句话，不要任何前缀或解释。"
)


def contextual_cfg(cfg: dict) -> dict:
    """读配置；model/base_url/api_key 留空 = 继承合成侧 `llm`（照 judge 先例）。"""
    ctx = dict(cfg.get("contextual") or {})
    llm_sec = cfg.get("llm") or {}
    ctx.setdefault("enabled", False)
    for key in ("model", "base_url", "api_key"):
        ctx.setdefault(key, "")
        if not ctx[key]:
            ctx[key] = llm_sec.get(key) or ""
    return ctx


def _llm_cfg_for(ctx: dict) -> dict:
    """生成前缀用的调用配置：关思考（照 rewrite/judge 实测先例）、temperature 0。"""
    return {
        "base_url": ctx["base_url"],
        "api_key": ctx["api_key"],
        "model": ctx["model"],
        "temperature": 0.0,
        "reasoning_effort": "none",
        "cache": True,
        "timeout_s": 20,
        "max_attempts": 1,
    }


def generate_prefix(ctx: dict, doc: IntermediateDoc, chunk: Chunk) -> str | None:
    """单个 chunk 的上下文前缀；失败返回 None（不阻塞入库）。"""
    section = " / ".join(chunk.section_path) or "(无章节)"
    page = chunk.page if chunk.page is not None else "?"
    prompt = _PROMPT.format(
        title=doc.meta.title or doc.meta.doc_id,
        section=section,
        page=page,
        body=chunk.text[:_CHUNK_TEXT_CHARS],
    )
    try:
        reply = llm.chat(_llm_cfg_for(ctx), prompt)
    except Exception:  # noqa: BLE001 前缀失败不阻塞入库（metadata 先例）
        return None
    text = " ".join(str(reply).split())
    return text[:_PREFIX_CHARS] or None


def generate_prefixes_for_doc(
    ctx: dict, doc: IntermediateDoc, chunks: list[Chunk]
) -> tuple[dict[str, str | None], int]:
    """一篇文档的全部 chunk 前缀。返回 ({chunk_id: 前缀或 None}, 失败数)。"""
    results: list[str | None] = []
    if not chunks:
        return {}, 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        # 一篇文档内部串行足够（前缀生成整批另有并发，见 generate_prefixes）
        results = list(pool.map(lambda c: generate_prefix(ctx, doc, c), chunks))
    failed = sum(1 for r in results if r is None)
    return {c.chunk_id: r for c, r in zip(chunks, results)}, failed


def generate_prefixes(
    ctx: dict, docs: list[IntermediateDoc], chunks_by_doc: list[list[Chunk]]
) -> tuple[dict[str, str | None], int]:
    """整批前缀生成：文档间并发（照 indexer 元数据抽取的 ThreadPoolExecutor(8)）。"""
    if not ctx.get("enabled"):
        return {}, 0
    out: dict[str, str | None] = {}
    total_failed = 0
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        for mapping, failed in pool.map(
            lambda t: generate_prefixes_for_doc(ctx, t[0], t[1]),
            list(zip(docs, chunks_by_doc)),
        ):
            out.update(mapping)
            total_failed += failed
    return out, total_failed

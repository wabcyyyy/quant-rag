"""doc-rag CLI（PLAN §6 验收命令）：check / profile / ingest / query / eval / serve。"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from doc_rag.config import load_config
from doc_rag.ingest import pipeline as ingest_pipeline
from doc_rag.ingest import profile as corpus_profile

app = typer.Typer(help="企业文档 RAG（飞书批量导出 PDF / doc(x) 双路）— 设计见 PLAN.md")


@app.command()
def check() -> None:
    """API 冒烟：LLM 连通 / Embedding 维度 / sparse 权重探测（PLAN §9 行动项 1）。"""
    import time

    import httpx
    from openai import OpenAI

    cfg = load_config()
    llm, emb = cfg["llm"], cfg["embedding"]

    missing = []
    if not llm["api_key"]:
        missing.append("DOC_RAG_LLM_API_KEY")
    if not llm["model"]:
        missing.append("DOC_RAG_LLM_MODEL")
    if not emb["api_key"]:
        missing.append("DOC_RAG_EMBEDDING_API_KEY")
    if missing:
        typer.echo(f"[缺配置] {', '.join(missing)}——按 .env.example 填好 .env 后重试")
        raise typer.Exit(1)

    t0 = time.perf_counter()
    try:
        client = OpenAI(base_url=llm["base_url"], api_key=llm["api_key"], timeout=30.0)
        resp = client.chat.completions.create(
            model=llm["model"],
            messages=[{"role": "user", "content": "只回复两个字：正常"}],
            max_tokens=8,
            temperature=0,
        )
        dt = time.perf_counter() - t0
        reply = resp.choices[0].message.content
        typer.echo(f"[LLM] {llm['model']} 连通 ✓（{dt:.1f}s）回复：{reply!r}")
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[LLM] 失败：{exc}")
        raise typer.Exit(1) from exc

    t0 = time.perf_counter()
    try:
        r = httpx.post(
            f"{emb['base_url'].rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {emb['api_key']}"},
            json={"model": emb["model"], "input": ["冒烟测试：关于供应商预付款的决议。"]},
            timeout=30.0,
        )
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[Embedding] 失败：{exc}")
        raise typer.Exit(1) from exc
    dt = time.perf_counter() - t0
    item = payload["data"][0]
    typer.echo(f"[Embedding] {emb['model']} 连通 ✓（{dt:.1f}s）dense 维度 = {len(item.get('embedding') or [])}")

    sparse_keys = [k for k in list(item) + list(payload) if "sparse" in str(k).lower()]
    if sparse_keys:
        typer.echo(f"[Sparse] 检测到字段 {sparse_keys} → learned-sparse 路线成立 ✓")
    else:
        typer.echo("[Sparse] 响应无 sparse 字段（与 2026-09 查证一致）")
        typer.echo("        → Phase 1 走 Dense + BM25(Qdrant full-text, jieba)；learned sparse 留作 Phase 2 实验")


@app.command()
def profile(
    raw_dir: Annotated[Path | None, typer.Option(help="语料目录")] = None,
    out: Annotated[Path | None, typer.Option(help="画像 JSON 输出路径")] = None,
) -> None:
    """Phase 0：语料画像统计（来源构成 / 可抽性 / 疑似扫描件）。"""
    cfg = load_config()
    raw = raw_dir or Path(cfg["paths"]["raw"])
    if not raw.exists():
        typer.echo(f"语料目录不存在：{raw}")
        raise typer.Exit(1)
    out_file = out or Path(cfg["paths"]["parsed"]) / "profile.json"
    result = corpus_profile.run(raw, out_file)
    typer.echo(f"总文件数: {result['total_files']}")
    typer.echo(f"来源构成: {result['source_distribution']}")
    scan = result["scan_suspects"]
    typer.echo(f"疑似扫描件（非空则启用 MinerU 兜底）: {scan if scan else '无'}")
    if result["pdf_errors"]:
        typer.echo(f"PDF 解析失败: {result['pdf_errors']}")
    typer.echo(f"明细已写入 {out_file}")


@app.command()
def ingest(
    raw_dir: Annotated[Path | None, typer.Option(help="语料目录")] = None,
    kb: Annotated[str | None, typer.Option(help="Qdrant collection 名，默认取配置")] = None,
    parse_only: Annotated[bool, typer.Option(help="只解析落盘，不向量化入库")] = False,
    recreate: Annotated[bool, typer.Option(help="先删除并重建 collection（清空重灌）")] = False,
    limit: Annotated[int | None, typer.Option(help="只处理前 N 个文件（试跑）")] = None,
    no_llm_meta: Annotated[bool, typer.Option(help="跳过 LLM 元数据抽取（只用文件名信号）")] = False,
) -> None:
    """全链路入库：解析 → 分块 → LLM 元数据 → Embed → Qdrant。"""
    cfg = load_config()
    raw = raw_dir or Path(cfg["paths"]["raw"])
    if not raw.exists():
        typer.echo(f"语料目录不存在：{raw}")
        raise typer.Exit(1)
    parsed = Path(cfg["paths"]["parsed"])

    stats = ingest_pipeline.run(raw, parsed, limit=limit)
    typer.echo(
        f"解析：共 {stats['total']} 个文件，成功 {stats['parsed']}，"
        f"去重跳过 {stats['skipped_duplicate']}，失败 {len(stats['failed'])}"
    )
    for item in stats["failed"]:
        typer.echo(f"  [解析失败] {item['file']}: {item['error']}")
    if parse_only:
        return

    from doc_rag.ingest.indexer import index_parsed

    collection = kb or cfg["qdrant"]["collection"]
    result = index_parsed(
        parsed, cfg, collection=collection, recreate=recreate, use_llm_meta=not no_llm_meta
    )
    typer.echo(f"入库：{result['docs']} 篇 / {result['chunks']} 块 → Qdrant[{collection}]")
    for item in result["failed"]:
        typer.echo(f"  [入库失败] {item['file']}: {item['error']}")


@app.command()
def query(
    question: str,
    kb: Annotated[str | None, typer.Option(help="Qdrant collection 名")] = None,
    top_n: Annotated[int | None, typer.Option(help="融合后取块数")] = None,
    no_rewrite: Annotated[bool, typer.Option(help="关闭查询改写（A/B 对照）")] = False,
) -> None:
    """检索问答：改写 → Dense+BM25+RRF → 强制引用合成。"""
    from qdrant_client import QdrantClient

    from doc_rag.generate.synthesizer import Synthesizer
    from doc_rag.ingest.embedder import Embedder
    from doc_rag.retrieve.hybrid import HybridRetriever
    from doc_rag.retrieve.rewrite import QueryRewriter

    cfg = load_config()
    collection = kb or cfg["qdrant"]["collection"]
    retriever = HybridRetriever(
        client=QdrantClient(url=cfg["qdrant"]["url"], timeout=60.0),
        embedder=Embedder(cfg["embedding"]),
        collection=collection,
        retrieval_cfg=cfg["retrieval"],
    )
    plan = {"rewritten": question, "filters": None, "aggregate": False, "top_n": None, "reason": "已关闭改写"}
    if not no_rewrite:
        plan = QueryRewriter(cfg["retrieval"]).rewrite(question)
    results = retriever.retrieve(
        plan["rewritten"],
        top_n=top_n or plan["top_n"],
        filters=plan["filters"],
        aggregate=plan["aggregate"],
    )
    if not results:
        typer.echo("（未检索到相关内容——先跑 doc-rag ingest）")
        raise typer.Exit(1)
    if not no_rewrite:
        typer.echo(f"[改写] {plan['reason']}\n")

    contexts = [
        {
            "no": i + 1,
            "text": r["text"],
            "doc": r["title"] or r["doc_id"],
            "page": r["page"],
        }
        for i, r in enumerate(results)
    ]
    typer.echo(Synthesizer(cfg["llm"]).answer(question, contexts))
    typer.echo("\n—— 引用 ——")
    for c, r in zip(contexts, results):
        page = f" 第{c['page']}页" if c["page"] else ""
        typer.echo(f"[{c['no']}] {c['doc']}{page}（{r['block_type']}）")


@app.command("gen-gold")
def gen_gold(
    out: Annotated[Path | None, typer.Option(help="黄金集输出路径")] = None,
    seed: Annotated[int, typer.Option(help="采样随机种子（可复现）")] = 42,
    programmatic_only: Annotated[
        bool, typer.Option(help="只重算程序化题型（保留已有 LLM 题，零 LLM 成本）")
    ] = False,
) -> None:
    """生成黄金评估集（LLM 生成 + 程序化构造，构造过程可复现）。"""
    from doc_rag.eval.goldgen import generate

    cfg = load_config()
    parsed = Path(cfg["paths"]["parsed"])
    if not parsed.exists():
        typer.echo("先跑 doc-rag ingest 生成解析产物")
        raise typer.Exit(1)
    out_file = out or Path(cfg["eval"]["gold_file"])
    meta = generate(
        parsed, out_file, cfg["llm"], seed=seed, programmatic_only=programmatic_only
    )
    typer.echo(f"黄金集已生成：{out_file}")
    typer.echo(f"共 {meta['count']} 条，题型分布：{meta['type_distribution']}")


@app.command("eval")
def evaluate(
    gold: Annotated[Path | None, typer.Option(help="黄金集 JSON 路径")] = None,
    kb: Annotated[str | None, typer.Option(help="Qdrant collection 名")] = None,
    top_n: Annotated[int, typer.Option(help="检索取块数")] = 8,
    limit: Annotated[int | None, typer.Option(help="只评前 N 条（试跑）")] = None,
    ragas: Annotated[bool, typer.Option(help="启用 RAGAS 第二轨（较慢）")] = False,
    ragas_from: Annotated[Path | None, typer.Option(help="对已有评估结果补跑 RAGAS（答案复用，省钱）")] = None,
    retrieval_only: Annotated[bool, typer.Option(help="只评检索指标（不调 LLM 合成）")] = False,
    mode: Annotated[str | None, typer.Option(help="检索模式：hybrid（默认）/ dense（消融对照）")] = None,
    aggregate: Annotated[bool, typer.Option(help="聚合检索：大池取块后按文档去重（跨文档题）")] = False,
    rewrite: Annotated[bool, typer.Option(help="启用查询改写（测实际产品路径）")] = False,
) -> None:
    """评估：客观指标（Recall@k / MRR / 包含匹配 / 拒答 / 引用）+ 可选 RAGAS。"""
    import json
    from datetime import datetime

    cfg = load_config()

    if ragas_from is not None:
        from doc_rag.eval.runner import ragas_from_results

        out = ragas_from_results(ragas_from, cfg)
        typer.echo(f"RAGAS（基于 {ragas_from.name} 的答案）: {out}")
        raise typer.Exit(0)

    from doc_rag.eval.runner import evaluate as run_eval

    gold_file = gold or Path(cfg["eval"]["gold_file"])
    if not gold_file.exists():
        typer.echo(f"黄金集不存在：{gold_file}——先跑 doc-rag gen-gold")
        raise typer.Exit(1)

    results = run_eval(
        gold_file,
        cfg=cfg,
        collection=kb,
        top_n=top_n,
        limit=limit,
        with_ragas=ragas,
        with_answers=not retrieval_only,
        mode=mode,
        aggregate=aggregate,
        use_rewrite=rewrite,
    )
    s = results["summary"]
    typer.echo(f"\n=== 评估结果（{s['n_items']} 条 · top_n={top_n} · {results['meta']['retrieval']}）===")
    typer.echo(f"Recall@5        : {s['recall_at_5']}")
    typer.echo(f"Recall@{top_n}       : {s.get(f'recall_at_{top_n}')}")
    typer.echo(f"MRR             : {s['mrr']}")
    typer.echo(f"包含匹配准确率   : {s['contains_acc']}")
    typer.echo(f"拒答正确率      : {s['refusal_acc']}")
    typer.echo(f"引用有效率      : {s['citation_valid_rate']}")
    typer.echo(f"文档覆盖率      : {s['mean_doc_coverage']}")
    typer.echo(f"  分题型覆盖率  : {s['coverage_by_type']}")
    if results.get("ragas"):
        typer.echo(f"RAGAS           : {results['ragas']}")

    out_dir = Path(cfg["paths"]["eval"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"明细已写入 {out_file}")


@app.command()
def backfill(
    kb: Annotated[str | None, typer.Option(help="Qdrant collection 名")] = None,
) -> None:
    """按当前规则重算 payload 元数据（如 doc_date）并原地更新——不重新向量化。

    动机：元数据规则演进（如 doc_date 增补正文抽取）后无需重灌全库。
    """
    import json

    from qdrant_client import QdrantClient, models

    from doc_rag.ingest.metadata import base_meta
    from doc_rag.ingest.schema import IntermediateDoc

    cfg = load_config()
    collection = kb or cfg["qdrant"]["collection"]
    parsed = Path(cfg["paths"]["parsed"])
    client = QdrantClient(url=cfg["qdrant"]["url"], timeout=60.0)

    updated = 0
    for json_file in sorted(parsed.glob("*.json")):
        if json_file.name == "profile.json":
            continue
        try:
            doc = IntermediateDoc.model_validate_json(json_file.read_text(encoding="utf-8"))
            meta = base_meta(doc)
            client.set_payload(
                collection,
                payload={k: v for k, v in meta.items() if v is not None},
                points=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id", match=models.MatchValue(value=doc.meta.doc_id)
                        )
                    ]
                ),
            )
            updated += 1
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  [跳过] {json_file.name}: {exc}")
    with_date = client.count(
        collection,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_date", range=models.DatetimeRange(gte="2000-01-01T00:00:00")
                )
            ]
        ),
    ).count
    total = client.count(collection).count
    typer.echo(f"已更新 {updated} 篇文档的 payload")
    typer.echo(f"doc_date 覆盖率：{with_date}/{total} = {with_date / max(total, 1):.1%}")


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "0.0.0.0",
    port: Annotated[int, typer.Option()] = 8000,
) -> None:
    """启动 FastAPI（:8000）。"""
    import uvicorn

    uvicorn.run("doc_rag.api.main:app", host=host, port=port)

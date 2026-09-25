"""doc-rag CLI（PLAN §6 验收命令）：check / profile / ingest / query / eval / serve。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from doc_rag.config import ROOT, load_config
from doc_rag.ingest import pipeline as ingest_pipeline
from doc_rag.ingest import profile as corpus_profile

app = typer.Typer(
    help="企业文档 RAG（PDF / doc(x) 双路，导出后手动迁入）— 设计见 docs/design/PLAN.md"
)


def _never_crash_on_status_glyph() -> None:
    """Windows 控制台默认 GBK：状态行里的 ✓ / → 会抛 UnicodeEncodeError。

    实测后果不是「不好看」而是「假故障」：`doc-rag check` 打印连通成功那行时抛错，
    被自己的 `except Exception` 捕获，于是连接正常的 LLM 被报成「[LLM] 失败」并
    exit 1。只替换不可编码字符，不改控制台编码（改编码会让中文整体变乱码）。
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(errors="replace")  # type: ignore[operator]
        except (ValueError, OSError):  # 被重定向的非文本流，跳过
            pass


_never_crash_on_status_glyph()


def _fmt_q(q: dict | None) -> str:
    if not q:
        return "—"
    return f"p50 {q.get('p50')} · p95 {q.get('p95')} · max {q.get('max')}（n={q.get('n')}）"


def _print_latency(lat: dict | None) -> None:
    """打印延迟摘要（PLAN「延迟口径」）：分阶段 + 分题型 + 缓存污染告警。"""
    if not lat:
        typer.echo("延迟            : —（本次为检索模式或未计时）")
        return
    by_stage = lat.get("by_stage") or {}
    typer.echo(
        f"延迟检索侧      : 改写+检索+重排 {_fmt_q(by_stage.get('retrieval_total'))}"
    )
    typer.echo(f"延迟合成        : {_fmt_q(lat.get('synthesize'))}")
    total = lat.get("total") or {}
    typer.echo(f"延迟端到端      : {_fmt_q(total)}（混合值，判定看下面分桶 SLO）")
    # N7：SLO 是分题型的（聚合 ≤5s、短答 ≤8s），样本只取未命中缓存的条目；
    # 「没测」（缓存污染 / 纯检索）= None，不许再印成 ✓ 达标
    slo = lat.get("latency_slo") or {}
    for bucket, label in (("aggregate", "聚合≤5s"), ("short", "短答≤8s")):
        b = slo.get(bucket) or {}
        if not b:
            continue
        meets = b.get("meets")
        mark = "✓ 达标" if meets is True else ("✗ 超标" if meets is False else "— 未测")
        typer.echo(
            f"  SLO {label:<8}: p95 {b.get('p95')}ms / 目标 ≤{b.get('target_ms')}ms"
            f"（未缓存 n={b.get('n_uncached')}） {mark}"
        )
    for t, row in (lat.get("by_type") or {}).items():
        syn, tot = row.get("synthesize") or {}, row.get("total") or {}
        typer.echo(
            f"  {t:<16} n={row.get('n'):<3} 合成 p95 {syn.get('p95', '—')}ms · "
            f"端到端 p95 {tot.get('p95', '—')}ms · 答案均长 {row.get('answer_chars_mean')} 字"
        )
    if lat.get("cache_contaminated"):
        typer.echo(
            f"  ⚠ 本轮有 {lat.get('cached_answers')} 条答案命中缓存：合成延迟被低估，"
            "测真实延迟请加 --fresh-answers"
        )


def _pinned_qdrant_tag(compose_file: Path | None = None) -> str | None:
    """docker-compose.yml 里 Qdrant 镜像的 tag（去掉 `v` 前缀），读不到返回 None。

    存在的理由：钉版本在本地是**隐形**的——容器一直在跑，tag 写错只有 clone 出来的
    人会遇到 `docker compose up` 失败（2026-09-19 实测踩过：Docker Hub 上只有
    `v1.19.1`，没有 `1.19.1`）。所以 `check` 要把「跑着的版本」和「compose 承诺的
    版本」对一遍。
    """
    import yaml

    path = compose_file or ROOT / "docker-compose.yml"
    try:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    services = (spec or {}).get("services") or {}
    image = (services.get("qdrant") or {}).get("image") or ""
    if ":" not in image:
        return None
    return str(image).rsplit(":", 1)[-1].removeprefix("v")


@app.command()
def check() -> None:
    """API 冒烟：LLM 连通 / Embedding 维度 / sparse 权重探测（PLAN §9 行动项 1）。"""
    import time

    import httpx
    from openai import APIError, OpenAI

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
        client = OpenAI(base_url=llm["base_url"], api_key=llm["api_key"], timeout=120.0)
        resp = client.chat.completions.create(
            model=llm["model"],
            messages=[{"role": "user", "content": "只回复两个字：正常"}],
            max_tokens=512,  # 推理型模型会先消耗 reasoning token，预算给小了 content 会是 None
            temperature=0,
        )
        dt = time.perf_counter() - t0
        msg = resp.choices[0].message
        reply = (msg.content or "").strip()
        reasoning = getattr(msg, "reasoning", None) or ""
        if reply:
            typer.echo(f"[LLM] {llm['model']} 连通 ✓（ping {dt:.1f}s）回复：{reply!r}")
        elif reasoning:
            typer.echo(
                f"[LLM] {llm['model']} 连通 ✓（ping {dt:.1f}s）但 content 为空、仅返回 reasoning "
                f"（推理型模型 + token 预算不足）：{reasoning[:60]!r}"
            )
            typer.echo("      提示：正式调用需留足 max_tokens，或换非推理型模型")
        else:
            typer.echo(
                f"[LLM] {llm['model']} 连通 ✓（ping {dt:.1f}s）但返回空内容，请检查模型"
            )
        # 这个数曾被当成合成延迟写进 PLAN（1.3s），实际是无上下文、无 system prompt 的
        # 单句 ping，与真实合成的量级差一个数量级。要测延迟用 eval 的延迟摘要。
        typer.echo(
            "      注：这是连通性 ping 延迟（无上下文、无 system prompt），≠ 合成延迟；"
        )
        typer.echo("          合成延迟见 `doc-rag eval` 的延迟摘要（需关缓存才准）")
    except (APIError, httpx.HTTPError) as exc:  # 只报 API 故障；编码/逻辑错误照常上抛
        typer.echo(f"[LLM] 失败：{exc}")
        raise typer.Exit(1) from exc

    t0 = time.perf_counter()
    try:
        r = httpx.post(
            f"{emb['base_url'].rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {emb['api_key']}"},
            json={
                "model": emb["model"],
                "input": ["冒烟测试：关于供应商预付款的决议。"],
            },
            timeout=30.0,
        )
        r.raise_for_status()
        payload = r.json()
    except httpx.HTTPError as exc:
        typer.echo(f"[Embedding] 失败：{exc}")
        raise typer.Exit(1) from exc
    dt = time.perf_counter() - t0
    item = payload["data"][0]
    got_dim = len(item.get("embedding") or [])
    want_dim = int(emb["dense_dim"])
    typer.echo(f"[Embedding] {emb['model']} 连通 ✓（{dt:.1f}s）dense 维度 = {got_dim}")
    if got_dim != want_dim:
        # 维度只打印不校验是不够的：不匹配会在入库时逐文档失败并沉进 stats["failed"]，
        # 1121 次之后才被发现。建表前就把它变成硬失败。
        typer.echo(
            f"        ✗ 与 configs 的 embedding.dense_dim={want_dim} 不符——"
            "换嵌入模型必须 --recreate 重建 collection，否则入库全部失败"
        )
        raise typer.Exit(1)

    sparse_keys = [k for k in list(item) + list(payload) if "sparse" in str(k).lower()]
    if sparse_keys:
        typer.echo(f"[Sparse] 检测到字段 {sparse_keys} → learned-sparse 路线成立 ✓")
    else:
        typer.echo("[Sparse] 响应无 sparse 字段（与 2026-09 查证一致）")
        typer.echo(
            "        → Phase 1 走 Dense + BM25(Qdrant full-text, jieba)；learned sparse 留作 Phase 2 实验"
        )

    # Qdrant：可达性 + 钉版一致性。版本不匹配按硬失败处理——`hybrid.py` 的
    # prefetch + RRF 是**服务端**融合（k 不可配），行为按钉版实测过，换版本等于换检索实现。
    qd_url = (cfg["qdrant"]["url"] or "").rstrip("/")
    try:
        r = httpx.get(f"{qd_url}/", timeout=5.0)
        r.raise_for_status()
        server = str(r.json().get("version") or "")
    except (httpx.HTTPError, ValueError) as exc:
        typer.echo(f"[Qdrant] 连不上 {qd_url}：{exc}——先 `docker compose up -d`")
        raise typer.Exit(1) from exc
    pinned = _pinned_qdrant_tag()
    if pinned and server and server != pinned:
        typer.echo(
            f"[Qdrant] 运行中 {server} ≠ docker-compose.yml 钉的 v{pinned}"
            "——检索基线按钉版实测，换版本需重测"
        )
        raise typer.Exit(1)
    typer.echo(
        f"[Qdrant] 可达 ✓ 服务端 {server}"
        + (
            " = compose 钉版"
            if pinned and pinned == server
            else "（未与 compose 钉版比对）"
        )
    )


@app.command("check-rewrite")
def check_rewrite(
    gate_file: Annotated[
        Path, typer.Option(help="泛化门禁集（措辞刻意不重复黄金集模板）")
    ] = Path("data/eval/rewrite_paraphrase.json"),
) -> None:
    """查询改写泛化门禁：真实调用模型，量聚合意图 recall 与单点题误触发率。

    为什么需要这条命令：改造前的规则改写只在黄金集自己的措辞上触发（10 条命中 2 条，
    且命中的恰好就是 cross_doc / time_filter 那两行；7 条同义改写 0 命中），
    而「+35pt 文档覆盖率」全是在那个前提下测出来的。泛化必须单独量，
    不能再靠评测集原句背书。
    """
    import json

    from doc_rag.retrieve.rewrite_llm import LLMQueryRewriter

    if not gate_file.is_file():
        typer.echo(f"[缺文件] {gate_file}")
        raise typer.Exit(1)
    spec = json.loads(gate_file.read_text(encoding="utf-8"))
    rewriter = LLMQueryRewriter(load_config())
    # 门禁结论必须说清是给哪个模型发的：换改写模型是这个开关下最容易「顺手一换」
    # 又最容易把泛化能力换掉的动作
    ep = rewriter.endpoint()
    typer.echo(f"改写模型：{ep.get('model')} @ {ep.get('base_url')}\n")

    agg_total = agg_hit = fact_total = fact_hit = year_total = year_hit = 0
    misses: list[str] = []
    for item in spec["items"]:
        plan = rewriter.rewrite(item["question"])
        if item["expect"] == "single_fact":
            fact_total += 1
            if plan["aggregate"]:
                fact_hit += 1
                misses.append(f"{item['id']} 单点题被误判为聚合：{item['question']}")
        else:
            agg_total += 1
            if plan["aggregate"]:
                agg_hit += 1
            else:
                misses.append(f"{item['id']} 未识别聚合意图：{item['question']}")
        if item["expect"] == "aggregate_year":
            year_total += 1
            rng = (plan["filters"] or {}).get("doc_date") or {}
            if str(rng.get("gte", "")).startswith(str(item["year"])):
                year_hit += 1
            else:
                misses.append(f"{item['id']} 年份过滤缺失或错位：{item['question']}")

    gates = spec.get("gates") or {}
    recall_min = float(gates.get("aggregate_recall_min", 0.8))
    false_max = float(gates.get("single_fact_false_trigger_max", 0.1))
    recall = agg_hit / agg_total if agg_total else 0.0
    false_rate = fact_hit / fact_total if fact_total else 0.0
    from doc_rag.eval.runner import _quantiles

    real_ms = [c["ms"] for c in rewriter.calls if not c["cached"]]
    cached_ms = [c["ms"] for c in rewriter.calls if c["cached"]]
    q = _quantiles([m for m in real_ms if m is not None])
    if q:
        typer.echo(
            f"单次改写延迟 : p50 {q['p50']:.0f}ms · p95 {q['p95']:.0f}ms"
            f" · max {q['max']:.0f}ms（真实调用 n={q['n']}）"
        )
    if cached_ms:
        typer.echo(
            f"[延迟不可测] {len(cached_ms)} 次是缓存命中（毫秒数是本地查询）"
            "——量延迟必须 DOC_RAG_LLM_CACHE=0 重跑"
        )
    typer.echo(
        f"聚合意图 recall：{recall:.2f}（{agg_hit}/{agg_total}，门禁 ≥{recall_min}）"
    )
    typer.echo(f"年份过滤命中：{year_hit}/{year_total}")
    typer.echo(
        f"单点题误触发率：{false_rate:.2f}（{fact_hit}/{fact_total}，门禁 ≤{false_max}）"
    )
    if rewriter.failure_count:
        typer.echo(
            f"[改写退化] {rewriter.failure_count}/{agg_total + fact_total} 条未生效"
            "——这些条目按「不改写」计，本轮结论不可用"
        )
    for line in misses[:20]:
        typer.echo(f"  [未过] {line}")
    ok = (
        recall >= recall_min and false_rate <= false_max and rewriter.failure_count == 0
    )
    typer.echo("门禁通过" if ok else "门禁未通过")
    raise typer.Exit(0 if ok else 1)


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
    typer.echo(
        "疑似扫描件（仅标记：OCR/MinerU 兜底未实现，这条链路目前不存在）: "
        f"{scan if scan else '无'}"
    )
    if result["pdf_errors"]:
        typer.echo(f"PDF 解析失败: {result['pdf_errors']}")
    typer.echo(f"明细已写入 {out_file}")


@app.command()
def ingest(
    raw_dir: Annotated[Path | None, typer.Option(help="语料目录")] = None,
    parsed_dir: Annotated[
        Path | None,
        typer.Option(
            help="中间 JSON 落盘目录（示例语料务必与公司语料分开，避免把全量入库）"
        ),
    ] = None,
    kb: Annotated[
        str | None, typer.Option(help="Qdrant collection 名，默认取配置")
    ] = None,
    parse_only: Annotated[bool, typer.Option(help="只解析落盘，不向量化入库")] = False,
    recreate: Annotated[
        bool, typer.Option(help="先删除并重建 collection（清空重灌）")
    ] = False,
    limit: Annotated[int | None, typer.Option(help="只处理前 N 个文件（试跑）")] = None,
    no_llm_meta: Annotated[
        bool, typer.Option(help="（已废弃，默认即不抽取；用 --llm-meta 开启）")
    ] = False,
    llm_meta: Annotated[
        bool,
        typer.Option(
            help="启用 LLM 元数据抽取（1130 篇 ≈1130 次调用，成本高，默认关）"
        ),
    ] = False,
    chunk_strategy: Annotated[
        str, typer.Option(help="分块策略：structural（默认）/ fixed（消融对照）")
    ] = "structural",
    index_only: Annotated[
        bool, typer.Option(help="跳过解析，直接对已有中间 JSON 入库")
    ] = False,
    prune: Annotated[
        bool,
        typer.Option(
            help="入库后反向对账并清理幽灵块（源文件已删/已改内容的旧点）；"
            "不加 --yes 只报告"
        ),
    ] = False,
    yes: Annotated[bool, typer.Option(help="与 --prune 同用：真的删掉幽灵块")] = False,
    ctx: Annotated[
        str,
        typer.Option(
            help="A3.4 contextual 前缀臂：off（默认，读配置 contextual.enabled）/ "
            "both（前缀进 dense+BM25）/ bm25（前缀只进 BM25）。"
            "前缀生成为每块一次 LLM 调用（可缓存）——先 --limit 2 试算外推，超预算即停"
        ),
    ] = "",
) -> None:
    """全链路入库：解析 → 分块 → LLM 元数据 → Embed → Qdrant。"""
    cfg = load_config()
    raw = raw_dir or Path(cfg["paths"]["raw"])
    parsed = parsed_dir or Path(cfg["paths"]["parsed"])
    if ctx and ctx not in ("off", "both", "bm25"):
        typer.echo(f"未知 --ctx 值：{ctx}（可选 off / both / bm25）")
        raise typer.Exit(1)
    if ctx == "both" or ctx == "bm25":
        from doc_rag.ingest.contextual import contextual_cfg

        eff = contextual_cfg(cfg)
        typer.echo(
            f"ctx 模式 {ctx}：前缀生成模型 {eff.get('model') or '(未配置!)'}"
            "——建议先 --limit 2 试算成本\n"
        )

    # 「语料应该有什么」的判据：本次这批源文件的 sha 集合。--limit 时它只是子集，
    # 拿去做对账会把没跑到的那几千篇全判成幽灵，所以这时退回 parsed_dir。
    keep_doc_ids: set[str] | None = None
    if index_only:
        typer.echo(f"跳过解析，直接入库已有中间 JSON（{parsed}）")
    else:
        if not raw.exists():
            typer.echo(f"语料目录不存在：{raw}")
            raise typer.Exit(1)
        stats = ingest_pipeline.run(raw, parsed, limit=limit)
        typer.echo(
            f"解析：共 {stats['total']} 个文件，成功 {stats['parsed']}，"
            f"去重跳过 {stats['skipped_duplicate']}，失败 {len(stats['failed'])}"
        )
        for item in stats["failed"]:
            typer.echo(f"  [解析失败] {item['file']}: {item['error']}")
        # A3.3：扫描件兜底必须可见。引擎缺席时点名清单——静默等于回到
        # 「扫描件这条链路只有标记」的旧病（README 已知不足第 3 条）。
        n_ocr = len(stats.get("ocr_applied", []))
        n_unavail = len(stats.get("ocr_unavailable", []))
        n_near_empty = len(stats.get("near_empty", []))
        if n_ocr or n_unavail or n_near_empty:
            typer.echo(
                f"  [扫描件] OCR 兜底成功 {n_ocr} 篇"
                + (f" · 未走兜底 {n_unavail} 篇" if n_unavail else "")
                + (
                    f" · 无图少字 {n_near_empty} 篇（OCR 帮不上）"
                    if n_near_empty
                    else ""
                )
            )
        if n_unavail:
            typer.echo(
                "  ⚠ 以下疑似扫描件未走兜底（缺 OCR 可选依赖）："
                "uv sync --extra ocr 后重新 ingest"
            )
            for name in stats["ocr_unavailable"][:5]:
                typer.echo(f"      {name}")
            if n_unavail > 5:
                typer.echo(f"      …另有 {n_unavail - 5} 篇")
        if not limit:
            keep_doc_ids = stats["doc_ids"]
    if parse_only:
        return

    from doc_rag.ingest.indexer import index_parsed

    collection = kb or cfg["qdrant"]["collection"]
    result = index_parsed(
        parsed,
        cfg,
        collection=collection,
        recreate=recreate,
        use_llm_meta=llm_meta or None,  # None=读配置（默认关）；--llm-meta 显式开
        chunk_strategy=chunk_strategy,
        limit=limit,  # 试跑上限对解析与入库同时生效，否则 --limit 也会全量计费
        prune=prune,
        assume_yes=yes,
        keep_doc_ids=keep_doc_ids,
        ctx_mode=ctx or None,
    )
    typer.echo(
        f"入库：{result['docs']} 篇 / {result['chunks']} 块 → Qdrant[{collection}]"
        f"（分块策略：{chunk_strategy}，LLM 元数据抽取：{'开' if result.get('llm_meta') else '关'}，"
        f"ctx={result.get('ctx_mode')}）"
    )
    if result.get("ctx_prefix_failed"):
        typer.echo(
            f"  [ctx] {result['ctx_prefix_failed']} 个前缀生成失败，已降级为「无前缀」"
            "（不阻塞入库；重跑可借缓存补齐）"
        )
    # B4：库指纹变更 = 旧答案自动失配（缓存键已织入指纹）。条数只提示不删——
    # 删除是显式动作（cache-clear --yes），静默清理会让「回滚索引复测」丢数据。
    if result.get("index_fp_changed"):
        from doc_rag.generate.llm import cache_stats

        st = cache_stats()
        typer.echo(
            f"  [指纹] 库指纹已变更 → {result['index_fp']}："
            f"缓存中 {st['cached_total']} 条旧答案对新索引自动失配"
            "（保留在库；确认不回滚可用 cache-clear --yes 清理）"
        )
    # 对账：解析产物 1127 篇 vs 入库 1121 篇，改造前这 6 篇空文档在输出里无声消失
    gap = result["parsed"] - result["docs"]
    if gap:
        truncated = max(gap - result["empty"] - len(result["failed"]), 0)
        typer.echo(
            f"  [对账] 解析 {result['parsed']} 篇 → 入库 {result['docs']} 篇："
            f"空文档 {result['empty']} · 失败 {len(result['failed'])} · limit 截断 {truncated}"
        )
    for item in result["failed"]:
        typer.echo(f"  [入库失败] {item['file']}: {item['error']}")

    # 反向对账：库里那些本次没碰过的点。源文件删了或改过内容时它们永久留存，
    # 而且**照样会被检索命中**——带的是旧日期与旧正文，比漏入库更难被发现。
    rec = result["reconcile"]
    typer.echo(
        f"  [对账·反向] 判据 {rec['keep_docs']} 篇"
        f"（{'本次源文件' if rec['source'] == 'raw' else 'parsed_dir'}）"
        f" vs 库里 {rec['collection_docs']} 篇 / {rec['collection_chunks']} 块"
    )
    if rec["orphan_doc_ids"]:
        typer.echo(
            f"    幽灵文档 {len(rec['orphan_doc_ids'])} 篇 / {rec['orphan_chunks']} 块"
            "（语料里已不存在，仍能被检索命中）→ `ingest --prune --yes`"
        )
        for doc_id in rec["orphan_doc_ids"][:5]:
            typer.echo(f"      {doc_id}")
        if len(rec["orphan_doc_ids"]) > 5:
            typer.echo(f"      …另有 {len(rec['orphan_doc_ids']) - 5} 篇")
    if rec["missing_docs"]:
        typer.echo(
            f"    语料有、库里没有：{len(rec['missing_docs'])} 篇（失败/新文件/上次截断）"
        )
    if rec.get("stale_parsed_json"):
        typer.echo(
            f"    parsed_dir 里的陈旧产物 {len(rec['stale_parsed_json'])} 个"
            "（已无对应源文件，这次还被重灌了一遍）"
        )
    if prune:
        decision = result.get("prune") or {}
        if decision.get("refused"):
            typer.echo(f"  [清理] 未执行：{decision['refused']}")
        elif decision.get("deleted"):
            typer.echo(f"  [清理] 已删除 {decision['deleted']} 篇幽灵文档的旧点")
        elif rec["orphan_doc_ids"]:
            typer.echo("  [清理] 只报告：确认清单后加 --yes")

    # 重建完，旧答案还在缓存里可命中：缓存键带的是**当时那批上下文**，语料一变，
    # 命中的就是过期回答（而它比新答案快，所以没人会察觉）。条数必须报出来。
    from doc_rag.generate.llm import cache_inventory

    inv = cache_inventory()
    if inv["entries"]:
        typer.echo(
            f"  [缓存] 仍有 {inv['entries']} 条历史答案"
            f"（{inv['size_bytes'] / 1e6:.1f} MB，内含问题原文与检索到的正文）"
            + (
                "——本次是重建，它们引用的已是旧索引：`doc-rag cache-clear --yes`"
                if recreate
                else "——语料变更后要作废：`doc-rag cache-clear --yes`"
            )
        )


@app.command("freeze")
def freeze_cmd(
    kb: Annotated[
        str | None, typer.Option(help="要冻结的 Qdrant collection 名")
    ] = None,
    gold: Annotated[
        Path | None, typer.Option(help="随冻结登记的黄金集（sha256 进 synth_fp）")
    ] = None,
) -> None:
    """B10：产出双指纹冻结记录（index_fp + synth_fp → .cache/freeze_<id>.json）。

    幂等：同一状态重复 freeze 得到同一 freeze_id 与同一文件。eval 的 meta 会带当前
    双指纹并对**只相关的那一个**报警（只改 prompt → 只报 synth_fp 不匹配）。
    正式冻结清单（七项 + git status 干净）在下一轮读数轮执行，这里先落工具。
    """
    from doc_rag import freeze as freeze_mod

    cfg = load_config()
    collection = kb or cfg["qdrant"]["collection"]
    gold_file = gold or Path(cfg["eval"]["gold_file"])
    record = freeze_mod.freeze(cfg, collection, gold_file)
    typer.echo(
        f"freeze_id {record['freeze_id']} · index_fp {record['index_fp']}"
        f" · synth_fp {record['synth_fp']}"
    )
    typer.echo(f"  collection={collection} · gold={gold_file}")
    typer.echo(f"  commit={record['commit']}")
    typer.echo(f"  → {record['file']}")
    existing = freeze_mod.scan_freezes()
    typer.echo(f"  当前共 {len(existing)} 份冻结记录（幂等重跑不会新增）")
    typer.echo(
        "  注意：工具级指纹已落盘；正式冻结还需 git status 干净 + 配置快照（SPEC §6）"
    )


@app.command("cache-stats")
def cache_stats_cmd() -> None:
    """本地响应缓存的画像：条数、体积、按模型的分布、最老/最新条目。

    为什么值得单开一条命令：缓存键带的是完整 prompt（问题原文 + 检索到的文档正文），
    它既是「重复评估近乎零成本」的来源，也是语料内容在磁盘上的一份副本。
    两者都要有入口可查，不能只写在 README 里。
    """
    from doc_rag.generate.llm import cache_inventory

    inv = cache_inventory()
    typer.echo(f"缓存文件：{inv['path']}")
    typer.echo(f"条目 {inv['entries']} 条 · 体积 {inv['size_bytes'] / 1e6:.2f} MB")
    for row in inv["by_model"]:
        typer.echo(
            f"  {row['model']}: {row['entries']} 条"
            f"（{row['oldest']} → {row['newest']}）"
        )


@app.command("cache-clear")
def cache_clear_cmd(
    yes: Annotated[
        bool, typer.Option("--yes", help="确认清空（缓存没了就要重新付费）")
    ] = False,
) -> None:
    """作废响应缓存：re-ingest 之后旧答案不该再被命中。"""
    from doc_rag.generate.llm import cache_clear

    if not yes:
        typer.echo("会删除全部缓存条目（重跑评估就要重新付费）。加 --yes 确认。")
        raise typer.Exit(1)
    out = cache_clear()
    typer.echo(f"已清空 {out['deleted']} 条缓存（{out['path']}）")


@app.command()
def query(
    question: str,
    kb: Annotated[str | None, typer.Option(help="Qdrant collection 名")] = None,
    top_n: Annotated[int | None, typer.Option(help="融合后取块数")] = None,
    no_rewrite: Annotated[bool, typer.Option(help="关闭查询改写（A/B 对照）")] = False,
    no_rerank: Annotated[bool, typer.Option(help="关闭重排（A/B 对照）")] = False,
    stream: Annotated[
        bool, typer.Option(help="流式输出：答案边生成边打印（体感延迟方案）")
    ] = False,
    timing: Annotated[
        bool,
        typer.Option(help="打印分阶段延迟（注意：命中缓存时合成耗时不是模型延迟）"),
    ] = False,
) -> None:
    """检索问答：改写 → Dense+BM25+RRF → 重排 → 强制引用合成。"""
    from doc_rag.orchestrator import Orchestrator, Result

    cfg = load_config()
    orchestrator = Orchestrator(cfg, collection=kb)
    # 显式传参而不是 **opts 展开：未知键会让类型检查失去意义（mypy 只能报 dict[str, object]）
    # use_rerank=None = 跟随配置开关；显式关掉时传 False（与 eval / API 同一条规则）
    no_rerank_arg: bool | None = False if no_rerank else None
    result: Result | None = None
    if stream:
        import sys

        for ev in orchestrator.answer_stream(
            question,
            top_n=top_n,
            use_rewrite=not no_rewrite,
            use_rerank=no_rerank_arg,
            stop_on_empty=True,  # 空库不值得花一次合成
        ):
            if ev["type"] == "rewrite":
                if not no_rewrite:
                    typer.echo(f"[改写] {ev['plan']['reason']}\n")
            elif ev["type"] == "delta":
                typer.echo(ev["text"], nl=False)
                sys.stdout.flush()  # 终端行缓冲会攒字，流式必须逐段显式刷
            elif ev["type"] == "done":
                result = ev["result"]
        typer.echo("\n")
    else:
        result = orchestrator.answer(
            question,
            top_n=top_n,
            use_rewrite=not no_rewrite,
            use_rerank=no_rerank_arg,
            stop_on_empty=True,
        )
        if not result.retrieved:
            typer.echo("（未检索到相关内容——先跑 doc-rag ingest）")
            raise typer.Exit(1)
        if not no_rewrite:
            typer.echo(f"[改写] {result.plan['reason']}\n")
        typer.echo(result.answer)
    assert result is not None  # 两条分支都会赋值；流式被中途掐断时不该静默出空引用
    if result.trace:
        t = result.trace
        used = t["budget_used"]
        typer.echo(
            f"[agent] {len(t['steps'])} 步 · 停在 {t['stop_reason']} · "
            f"子查询 {t['sub_queries'] or '（没再多查）'} · 并集 {t['n_docs_union']} 篇 · "
            f"判定 {used['calls']} 次 / {used['prompt_tokens']} prompt tokens"
        )
    if result.rerank_error:
        typer.echo(f"[重排失败，退回融合顺序] {result.rerank_error}")
    _echo_citations(result.citations)
    if timing:
        _echo_timing(result)


def _echo_citations(citations: list[dict]) -> None:
    """每个 [n] 都能点回 Qdrant payload 里的 doc_id + page + block_type。"""
    typer.echo("\n—— 引用 ——")
    for c in citations:
        page = f" 第{c['page']}页" if c["page"] else ""
        typer.echo(f"[{c['no']}] {c['doc']}{page}（{c['block_type']}）")


def _echo_timing(result) -> None:
    lat = result.latency_ms
    meta = result.synth_meta or {}
    typer.echo("\n—— 延迟 ——")
    typer.echo(
        f"改写 {lat['rewrite']:.0f}ms · 检索 {lat['retrieve']:.0f}ms · "
        f"重排 {lat['rerank']:.0f}ms"
    )
    if lat.get("synth_cached"):
        typer.echo(
            f"合成 {meta.get('ms')}ms —— **缓存命中**，这不是模型延迟；"
            "测真实延迟请设 DOC_RAG_LLM_CACHE=0"
        )
    else:
        typer.echo(f"合成 {meta.get('ms')}ms（模型 {meta.get('model')}，真实调用）")
    typer.echo(f"端到端 {lat['total']:.0f}ms")


@app.command("gen-gold")
def gen_gold(
    out: Annotated[Path | None, typer.Option(help="黄金集输出路径")] = None,
    seed: Annotated[int, typer.Option(help="采样随机种子（可复现）")] = 42,
    programmatic_only: Annotated[
        bool, typer.Option(help="只重算程序化题型（保留已有 LLM 题，零 LLM 成本）")
    ] = False,
    v3: Annotated[
        bool,
        typer.Option(
            help="生成 v3（multi-hop·窗口依赖）独立文件：零 LLM，不与现有 72 条混口径。"
            "必须配 --out 指一个新文件。"
        ),
    ] = False,
    limit: Annotated[
        int, typer.Option(help="v3 最多出几条（原料由 census/probe 判据决定）")
    ] = 12,
    term_extra: Annotated[
        bool,
        typer.Option(
            help="生成 term 扩题独立文件：零 LLM，从「字段名：值」形状抽题，"
            "值全库唯一且逐字可核对。必须配 --out 指一个新文件，"
            "与已发布 72 条并列报数、不合成一个数。"
        ),
    ] = False,
) -> None:
    """生成黄金评估集（LLM 生成 + 程序化构造，构造过程可复现）。"""
    from doc_rag.eval.goldgen import generate, generate_term_extra, generate_v3

    cfg = load_config()
    parsed = Path(cfg["paths"]["parsed"])
    if not parsed.exists():
        typer.echo("先跑 doc-rag ingest 生成解析产物")
        raise typer.Exit(1)
    if v3:
        out_file = out or Path("data/eval/gold_v3.json")
        meta = generate_v3(parsed, out_file, limit=limit)
        typer.echo(f"v3 已生成：{out_file}")
        typer.echo(
            f"共 {meta['count']} 条，题型分布：{meta['type_distribution']}"
            f"（corpus={meta['corpus_dir']}）"
        )
        typer.echo(
            f"    原料：候选邻块对 {meta['candidates_total']} 个，"
            f"其中带决议线索词的 {meta['candidates_with_cue']} 个"
            f"（为质量丢掉 {meta['dropped_for_quality']} 个）"
        )
        if not meta["bar_met"]:
            typer.echo(
                f"    注意：只出到 {meta['count']} 条（判据 ≥8）——v3 够验机制，"
                "不够撑三臂消融的统计功效，不要拿它报质量收益。"
            )
        # 性质复检不过就非零退出：一套不再满足「按构造必败」的 v3 会给出假的失败
        if meta["property_violations"]:
            typer.echo(f"性质复检失败 {len(meta['property_violations'])} 条：")
            for line in meta["property_violations"][:10]:
                typer.echo(f"  - {line}")
            raise typer.Exit(1)
        typer.echo("性质复检通过：每条的值只在 B 块、主语只在 A 块、两块同篇相邻")
        return
    if term_extra:
        if out is None:
            typer.echo("--term-extra 必须配 --out 指一个新文件（不许覆盖已发布考卷）")
            raise typer.Exit(1)
        meta = generate_term_extra(
            parsed, out, limit=limit, exclude_file=Path(cfg["eval"]["gold_file"])
        )
        typer.echo(f"term 扩题已生成：{out}")
        typer.echo(
            f"共 {meta['count']} 条 · 原料 {meta['candidates_label_value']} 个合格「字段名：值」"
        )
        typer.echo(f"    按术语类别的可选池：{meta['pool_by_family']}")
        typer.echo(f"    与 {meta['excluded_against']} 去重（同篇同值不重复出题）")
        if meta["property_violations"]:
            typer.echo(f"性质复检失败 {len(meta['property_violations'])} 条：")
            for line in meta["property_violations"][:10]:
                typer.echo(f"  - {line}")
            raise typer.Exit(1)
        typer.echo(
            "性质复检通过：值逐字在该篇、全库只此一篇、题面短名在正文、无 URL/多人清单"
        )
        return
    out_file = out or Path(cfg["eval"]["gold_file"])
    meta = generate(
        parsed, out_file, cfg["llm"], seed=seed, programmatic_only=programmatic_only
    )
    typer.echo(f"黄金集已生成：{out_file}")
    typer.echo(f"共 {meta['count']} 条，题型分布：{meta['type_distribution']}")
    kp = meta["aggregation_key_points"]
    typer.echo(
        f"    聚合题答案判据：{kp['n_with_points']}/{kp['n_aggregate_items']} 条拿到逐篇要点"
        f"（K 均值 {kp['mean_k']}，上限 {kp['max_k']}；唯一性在 {kp['uniqueness_scope_docs']} "
        "篇正文上判）"
    )
    if kp["n_with_points"] < kp["n_aggregate_items"]:
        # 与 v3「只出到 3 条就如实打印」同一条纪律：判据覆盖率不够时，n 必须当场可见，
        # 不然读的人会以为 keypoint_hit_ratio 是在全部聚合题上算的。
        typer.echo(
            f"    注意：{kp['n_aggregate_items'] - kp['n_with_points']} 条聚合题挑不出合格要点"
            "（那个实体在语料里几乎只以纯 @ 提及出现）——答案级配对判读只在有要点的"
            "那几条上做，报结论必须带上这个 n。"
        )


@app.command("census-corpus")
def census_corpus(
    parsed_dir: Annotated[
        Path | None,
        typer.Option(help="中间 JSON 目录（默认配置的 paths.parsed）"),
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="普查结果 JSON 输出路径（默认只打印）")
    ] = None,
    want_items: Annotated[
        int, typer.Option(help="判据：这两类新题型各要能凑出几条题")
    ] = 8,
) -> None:
    """v3 出题前的语料普查：表格密度 + 时间线可造性（零 LLM、零 Qdrant）。

    PLAN §5.3 那组「179 个表格 / 固定 512 切下 85 个被切断」只证明了表格存在、
    分块会切断它们；它**没有**证明能凑出跨篇数值对比题。这条命令量的是后者。
    """
    import json

    from doc_rag.eval.census import census, verdict

    root = parsed_dir or Path(load_config()["paths"]["parsed"])
    report = census(root)
    call = verdict(report, want_items=want_items)
    typer.echo(f"普查 {report['parsed_dir']}（{report['docs']} 篇）")
    typer.echo(
        f"  表格：{report['tables_total']} 个 / {report['docs_with_tables']} 篇含表格"
        f"，其中 {report['tables_with_numbers']} 个含数值行；"
        f"单篇最多 {report['tables_per_doc']['max']} 个"
    )
    typer.echo(
        f"  跨篇数值题原料：同一行标签出现在 ≥2 篇 → "
        f"{report['cross_doc_labels_raw']} 个（剔掉日期行/序号列后 "
        f"{report['cross_doc_labels']} 个），覆盖 {report['cross_doc_label_docs']} 篇文档"
    )
    typer.echo(
        f"  时间线题原料：标题带时间点的文档 {report['docs_with_date_in_title']} 篇，"
        f"同主题多时间点的主题 {report['timeline_topics']} 个"
    )
    for kind, ok, detail in (
        (
            "跨篇表格数值",
            call["cross_doc_numeric_viable"],
            f"{call['cross_doc_labels']} 标签",
        ),
        ("时间线推翻", call["timeline_viable"], f"{call['timeline_topics']} 主题"),
    ):
        typer.echo(
            f"  [{kind}] {'可出' if ok else '凑不出'}（判据 ≥{want_items} 条；{detail}）"
        )
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"report": report, "verdict": call}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        typer.echo(f"  已写入 {out}")


@app.command("eval")
def evaluate(
    gold: Annotated[Path | None, typer.Option(help="黄金集 JSON 路径")] = None,
    kb: Annotated[str | None, typer.Option(help="Qdrant collection 名")] = None,
    top_n: Annotated[int, typer.Option(help="检索取块数")] = 8,
    limit: Annotated[int | None, typer.Option(help="只评前 N 条（试跑）")] = None,
    sample: Annotated[
        int | None,
        typer.Option(
            help="均匀抽 N 条评（黄金集按题型分块排序，--limit 会整段漏掉末尾题型）"
        ),
    ] = None,
    ragas: Annotated[bool, typer.Option(help="启用 RAGAS 第二轨（较慢）")] = False,
    ragas_from: Annotated[
        Path | None, typer.Option(help="对已有评估结果补跑 RAGAS（答案复用，省钱）")
    ] = None,
    retrieval_only: Annotated[
        bool, typer.Option(help="只评检索指标（不调 LLM 合成）")
    ] = False,
    mode: Annotated[
        str | None, typer.Option(help="检索模式：hybrid（默认）/ dense（消融对照）")
    ] = None,
    agent_mode: Annotated[
        str,
        typer.Option(
            help="policy 层：agent（强制多步）/ single（强制单发）/ 留空=跟随配置 "
            "agent.enabled。**与 --mode 不是一回事**：那个数的是检索模式。"
        ),
    ] = "",
    aggregate: Annotated[
        bool, typer.Option(help="聚合检索：大池取块后按文档去重（跨文档题）")
    ] = False,
    rewrite: Annotated[
        bool, typer.Option(help="启用查询改写（测实际产品路径）")
    ] = False,
    rerank: Annotated[bool, typer.Option(help="启用重排（削减上下文噪声）")] = False,
    honor_rewrite_budget: Annotated[
        bool,
        typer.Option(
            help="检索预算听改写的建议（聚合题 aggregate_top_n），而不是固定 --top-n。"
            "这是生产口径；默认关闭是为了让各消融臂在同一预算下可比"
        ),
    ] = False,
    no_citation_constraint: Annotated[
        bool, typer.Option(help="消融 #4 对照组：不要求标注引用编号")
    ] = False,
    ragas_sample: Annotated[
        int | None,
        typer.Option(help="RAGAS 抽样条数（0=全量）；默认取配置 eval.ragas_sample"),
    ] = None,
    fresh_judge: Annotated[
        bool,
        typer.Option(help="不复用 judge 缓存：测 judge 运行间随机性（会真实计费）"),
    ] = False,
    fresh_answers: Annotated[
        bool,
        typer.Option(
            help="不复用合成缓存：测真实 LLM 延迟（缓存命中时毫秒级，不是延迟）"
        ),
    ] = False,
    ragas_out: Annotated[
        Path | None,
        typer.Option(help="RAGAS 结果落盘路径（默认 <结果文件名>_ragas.json）"),
    ] = None,
    prompt_version: Annotated[
        str | None,
        typer.Option(
            help="合成 prompt 版本：tightened（默认）/ baseline（收紧前，消融对照）"
        ),
    ] = None,
    judge_model: Annotated[
        str,
        typer.Option(help="跨供应商复判 RAGAS：判分模型（压过 eval.judge.model）"),
    ] = "",
    judge_base_url: Annotated[
        str,
        typer.Option(help="判分端点；指向 embedding 那家时 key 自动复用"),
    ] = "",
    judge_api_key: Annotated[str, typer.Option(help="判分端点的 key")] = "",
    max_contexts: Annotated[
        int | None,
        typer.Option(
            help="上下文预算覆盖（E2 消融的杠杆）。同时压 `retrieval.max_contexts` 与 "
            "`rerank.top_n`——进 LLM 的块数是这两者的 min，只改一个不动作。"
        ),
    ] = None,
    two_stage: Annotated[
        bool,
        typer.Option(
            help="两段式合成（ADR-0002）：聚合题 map（逐篇微摘要）→ reduce。"
            "评估臂开关；生产默认跟配置 synthesis.two_stage.enabled"
        ),
    ] = False,
    resume: Annotated[
        Path | None,
        typer.Option(
            help="B2 续跑：从既有结果文件补齐——成功条目原样复用不重跑，"
            "失败/缺失条目重试（嵌入端点抖动后不再为一条超时多付一整轮）"
        ),
    ] = None,
    repeat: Annotated[
        int,
        typer.Option(
            help="B5 同配置连跑 N 遍：summary.repeat_span = 各指标跨遍极差"
            "（噪声地板），主口径取第一遍；repeat_runs[] 保留每遍逐条"
        ),
    ] = 1,
    holdout: Annotated[
        bool,
        typer.Option(
            help="B9/U1：评估外部 holdout（origin=external）——meta.holdout=true，"
            "结果与调参集可溯源区分；判分走人工 rubric 不是自动指标"
        ),
    ] = False,
) -> None:
    """评估：客观指标（Recall@k / MRR / 包含匹配 / 拒答 / 引用）+ 可选 RAGAS。"""
    import json
    from datetime import datetime

    from doc_rag.generate import prompts

    cfg = load_config()

    if max_contexts:
        # E2（上下文预算消融）的杠杆。进 LLM 的块数 = min(max_contexts, rerank.top_n)，
        # 两个键必须一起压，否则预算纹丝不动却看起来改了。生效值由结果文件的
        # meta.context_budget 自证：臂与臂的差要能追溯到配置，不是追溯到谁记没记住。
        cfg.setdefault("retrieval", {})["max_contexts"] = max_contexts
        cfg.setdefault("rerank", {})["top_n"] = max_contexts
        typer.echo(
            f"上下文预算已覆盖为 {max_contexts} 块"
            "（retrieval.max_contexts 与 rerank.top_n 同步）\n"
        )

    if prompt_version:
        if prompt_version not in prompts.ANSWER_PROMPTS:
            typer.echo(
                f"未知 prompt 版本：{prompt_version}（可选：{sorted(prompts.ANSWER_PROMPTS)}）"
            )
            raise typer.Exit(1)
        cfg["llm"]["prompt_version"] = prompt_version
        typer.echo(
            f"合成 prompt 版本：{prompt_version}（指纹 {prompts.fingerprint(prompt_version)}）\n"
        )

    if two_stage:
        # 评估臂开关：只改本次进程的配置，不落盘。路由按预测题型走
        # （synthesis.two_stage.types），单发题型不受影响——与 agent_mode 的
        # 「评估臂显式、部署读配置」是同一条分工。
        cfg.setdefault("synthesis", {}).setdefault("two_stage", {})["enabled"] = True
        typer.echo(
            "两段式合成已开启（--two-stage）：聚合题 map→reduce，"
            f"题型域 {cfg['synthesis']['two_stage'].get('types') or ['cross_doc', 'time_filter']}\n"
        )

    if fresh_judge and not (ragas or ragas_from is not None):
        typer.echo("--fresh-judge 只在启用 RAGAS（--ragas 或 --ragas-from）时有效")
        raise typer.Exit(1)

    judge_over = {
        "model": judge_model,
        "base_url": judge_base_url,
        "api_key": judge_api_key,
    }
    if any(judge_over.values()):
        if not (ragas or ragas_from is not None):
            typer.echo("--judge-* 只在启用 RAGAS（--ragas 或 --ragas-from）时有效")
            raise typer.Exit(1)
        from doc_rag.eval.judge import judge_cfg

        built = judge_cfg(cfg, **judge_over)
        typer.echo(
            f"RAGAS judge：{built['model']} @ "
            f"{(built['base_url'] or '').split('//')[-1].split('/')[0]} · "
            f"思考={built['reasoning_effort']}"
            "（换 judge 即换度量身份，分数不能与同源基线并列）"
        )
        typer.echo("")

    # 关合成缓存必须走环境变量：cache_enabled() 里环境变量优先于配置，
    # 而这里要的是「本次进程不吃缓存」，不能改配置（会污染后续运行）
    if fresh_answers:
        import os

        os.environ["DOC_RAG_LLM_CACHE"] = "0"
        typer.echo(
            "已关闭合成缓存（--fresh-answers）：本轮延迟为真实调用耗时，会真实计费\n"
        )

    if ragas_from is not None:
        from doc_rag.eval.runner import ragas_from_results

        out = ragas_from_results(
            ragas_from,
            cfg,
            sample_n=ragas_sample,
            use_cache=not fresh_judge,
            out_file=ragas_out,
            judge_over=judge_over,
        )
        typer.echo(f"RAGAS（基于 {ragas_from.name} 的答案）: {out}")
        raise typer.Exit(0)

    from doc_rag.eval.runner import evaluate as run_eval

    gold_file = gold or Path(cfg["eval"]["gold_file"])
    if not gold_file.exists():
        typer.echo(f"黄金集不存在：{gold_file}——先跑 doc-rag gen-gold")
        raise typer.Exit(1)

    # 落盘路径提前定：B2 的逐条 flush 写的就是这份文件（中途死掉可 --resume 补齐）
    out_dir = Path(cfg["paths"]["eval"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = (
        ragas_out
        or out_dir
        / f"results_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}.json"
    )
    if resume is not None and not resume.exists():
        typer.echo(f"--resume 指定的文件不存在：{resume}")
        raise typer.Exit(1)

    # 两分支共用同一份参数（N12）：各写一份参数集没有同源约束，漂移无告警——
    # repeat 分支漏传 resume_from 导致 `--repeat 3 --resume X` 静默重跑重付，
    # 正是这么发生的。共用 dict 之后，新增参数两分支必然同时拿到。
    eval_kwargs: dict[str, Any] = {
        "collection": kb,
        "top_n": top_n,
        "limit": limit,
        "with_ragas": ragas,
        "with_answers": not retrieval_only,
        "mode": mode,
        "agent_mode": agent_mode or None,
        "aggregate": aggregate,
        "use_rewrite": rewrite,
        "use_rerank": rerank,
        "honor_rewrite_budget": honor_rewrite_budget,
        "sample": sample,
        "require_citation": not no_citation_constraint,
        "ragas_sample": ragas_sample,
        "use_judge_cache": not fresh_judge,
        "judge_over": judge_over,
        "holdout": holdout,
    }
    if repeat > 1:
        from doc_rag.eval.runner import evaluate_with_repeat

        # resume 只由 runner 内部给第 1 遍（第 2 遍起也 resume 会让 repeat_span
        # 塌成 0）；progress_file 不传——repeat 模式的落盘走下方兜底写 merged
        # （含 repeat_span），逐遍覆盖会把它顶掉。
        results = evaluate_with_repeat(
            gold_file,
            cfg=cfg,
            repeat=repeat,
            **eval_kwargs,
            resume_from=resume,
        )
        span = results["summary"].get("repeat_span") or {}
        typer.echo(
            f"\n[repeat n={repeat}] 关键极差：Hit@5 {span.get('hit_at_5')}"
            f" · MRR {span.get('mrr')} · 严格关键词 {span.get('strict_keyword_accuracy')}"
        )
    else:
        results = run_eval(
            gold_file,
            cfg=cfg,
            **eval_kwargs,
            resume_from=resume,
            progress_file=out_file,
        )
    s = results["summary"]
    typer.echo(
        f"\n=== 评估结果（{s['n_items']} 条 · 预算={results['meta']['budget']}"
        f" · {results['meta']['retrieval']}）==="
    )
    # 全失败会中止，部分失败只能显式提醒：一轮里混进 N 条降级样本，整轮的口径就不再纯
    meta = results["meta"]
    ameta = meta.get("agent")
    if ameta:
        typer.echo(
            f"    agent：开关来源={ameta['requested'] or '配置'} · "
            f"题型={list(ameta['types'])} · 带 trace {ameta['n_items_with_trace']}"
            f"/{s['n_items']} 条 · 停机分布={ameta['stop_reasons']}"
        )
    mixed = []
    if ameta and ameta["judge_degraded"]:
        mixed.append(
            f"{ameta['judge_degraded']} 条 agent 判定退化——那一步是「停」不是「判」，"
            "成本与质量结论都要打折看"
        )
    if meta.get("rerank_failed"):
        mixed.append(f"{meta['rerank_failed']} 条重排失败（按融合顺序送下游）")
    if meta.get("rewrite_degraded"):
        mixed.append(f"{meta['rewrite_degraded']} 条改写退化（未改写）")
    if meta.get("filter_fallback_n"):
        mixed.append(
            f"{meta['filter_fallback_n']}/{meta.get('filters_applied_n')} 条"
            "过滤后结果过少已回退为**不过滤**（覆盖率里含这部分未过滤的条目）"
        )
    if mixed:
        typer.echo(
            "  [口径混合] " + "；".join(mixed) + "——逐条字段可追，本轮不是纯条件"
        )
    # 名字一律用 IR 的标准读法，且 Hit 与 Recall 分家：Hit@k 只回答「前 k 位里碰到没有」，
    # Recall@k 才回答「gold 里捞回了几篇」。当年 `recall_at_k` 报的其实是 Hit@k，
    # 因为逐条只存了首命中位次、算不出后者——现在 ranked 清单落盘了，两者都给。
    typer.echo(
        f"Hit@5 / @8 / @清单   : {s['hit_at_5']} / {s['hit_at_8']} / {s['hit_at_list']}"
    )
    typer.echo(
        f"K=5 标准族          : Recall {s.get('recall_at_5')}"
        f" · Precision {s.get('precision_at_5')}"
        f" · nDCG {s.get('ndcg_at_5')}"
        f" · MAP {s.get('map_at_5')}"
    )
    typer.echo(
        f"Recall@清单 macro   : {s['recall_at_list_macro']}"
        f"（上限 {s.get('recall_ceiling_macro')}"
        f" · 归一 {s.get('recall_vs_ceiling_macro')}）"
    )
    typer.echo(f"MRR                : {s['mrr']}")
    typer.echo(f"nDCG@8（历史基线） : {s.get('ndcg_at_8')}")
    typer.echo(f"  分题型 Recall    : {s['recall_by_type']}")
    typer.echo(f"  分题型上限       : {s.get('recall_ceiling_by_type')}")
    typer.echo(f"严格关键词准确率    : {s['strict_keyword_accuracy']}")
    if s.get("keypoint_recall_macro") is not None:
        typer.echo(
            f"要点召回 macro(聚合) : {s['keypoint_recall_macro']}"
            f"（n={s.get('keypoint_n_items')}，K={s.get('keypoint_k')}）"
        )
    typer.echo(f"拒答正确率         : {s['refusal_acc']}（只查拒答措辞）")
    typer.echo(
        f"过度拒答           : ctx含原话 {s.get('over_refusal_rate')}"
        f" / gold已进上下文 {s.get('over_refusal_gold_rate')}"
    )
    typer.echo(f"引用有效率         : {s['citation_valid_rate']}")
    typer.echo(f"引用存在率         : {s.get('citation_presence_rate')}")
    typer.echo(f"  清单长度         : {s.get('list_len')}")
    _print_latency(s.get("latency"))
    if results.get("ragas"):
        typer.echo(f"RAGAS           : {results['ragas']}")
    from doc_rag.generate.llm import cache_stats

    st = cache_stats()
    typer.echo(
        f"LLM 缓存        : 命中 {st['hit']} / 未命中 {st['miss']}"
        f"（命中率 {st['hit_rate']}，库内共 {st['cached_total']} 条）"
    )
    if st.get("completion_tokens"):
        reas = st.get("reasoning_tokens") or 0
        typer.echo(
            f"本次真实调用    : 输入 {st['prompt_tokens']:,} / 输出 {st['completion_tokens']:,}"
            f" tokens（其中思考 {reas:,}，占输出 {reas / st['completion_tokens']:.0%}）"
        )

    if results["meta"].get("freeze_warnings"):
        for w in results["meta"]["freeze_warnings"]:
            typer.echo(f"  ⚠ [freeze] {w}")
    if results["meta"].get("n_errors"):
        typer.echo(
            f"  ⚠ {results['meta']['n_errors']} 条失败未计入分母"
            f"（summary.n_errors；--resume {out_file.name} 可只补失败条）"
        )
    if results["meta"].get("n_resumed"):
        typer.echo(
            f"  [resume] 复用 {results['meta']['n_resumed']} 条已成功条目"
            "（结果含历史轮次，meta.n_resumed 自证）"
        )

    # --ragas-out 同时给直跑路径用：三组对照（T4）要求每组结果落在指定文件名上，
    # 时间戳文件名无法预先写进 compare-ragas 的命令行。B2 起最终落盘已由
    # progress_file 完成（同一份文件），这里只在无 flush 路径时兜底写一次。
    if not out_file.exists():
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    typer.echo(f"明细已写入 {out_file}")


@app.command("probe-judge")
def probe_judge_cmd(
    results_file: Annotated[
        Path, typer.Argument(help="评估结果 JSON（含 answer / contexts）")
    ],
    item_id: Annotated[str, typer.Argument(help="条目 id，如 q007")],
) -> None:
    """打印单条答案的 judge 中间产物：抽出的陈述 + 逐条判定 + 理由。

    绝对分值可疑时用它区分「答案真不忠实」和「judge 抽错/判错」——
    消融 #2 的度量口径 bug 就是这样定位到的。
    """
    from doc_rag.eval.runner import probe_judge

    out = probe_judge(results_file, item_id)
    if out.get("error"):
        typer.echo(out["error"])
        raise typer.Exit(1)
    typer.echo(f"{out['id']} | {out['type']}")
    typer.echo(f"问题：{out['question']}\n")
    for v in out["statements"]:
        typer.echo(f"  {'✓' if v['verdict'] else '✗'} {v['statement']}")
        if v.get("reason"):
            typer.echo(f"      理由：{v['reason']}")
    typer.echo(f"\n→ Faithfulness = {out['score']}")


@app.command("audit-refusals")
def audit_refusals_cmd(
    results_file: Annotated[
        Path, typer.Argument(help="评估结果 JSON（含 answer / contexts）")
    ],
    limit: Annotated[
        int | None, typer.Option(help="只审前 N 条（先试算成本再全量，8 条 ≈¥0.05）")
    ] = None,
    judge_model: Annotated[
        str, typer.Option(help="跨供应商复判：判定用哪个模型（压过 eval.judge.model）")
    ] = "",
    judge_base_url: Annotated[
        str,
        typer.Option(
            help="判定用哪家的 OpenAI 兼容端点；指向 embedding 那家时 key 自动复用"
        ),
    ] = "",
    judge_api_key: Annotated[
        str,
        typer.Option(help="判定端点的 key；留空按 eval.judge.api_key / 同供应商复用"),
    ] = "",
    out: Annotated[
        Path | None, typer.Option(help="把本次审计逐条结果落盘（供 --against 比对）")
    ] = None,
    against: Annotated[
        Path | None,
        typer.Option(help="与另一份审计结果逐条对判读：报一致率与翻转的条目"),
    ] = None,
) -> None:
    """审计拒答题的答案：有没有把上下文里没记载的内容当成事实讲出来。

    现有拒答判分只查措辞（命中「无法回答」这类词即算正确拒答），RAGAS 轨又按定义把
    拒答题排除在 faithfulness 之外——两条轨合起来，8 条拒答题没有一条被看过「里面写了
    什么」。这条命令就是那个检查。判定用的是结果文件里落盘的上下文，**不需要 Qdrant**。
    """
    import json

    from doc_rag.eval.refusal import audit_results, compare_audits

    if not results_file.exists():
        typer.echo(f"结果文件不存在：{results_file}")
        raise typer.Exit(1)
    out_report = audit_results(
        results_file,
        limit=limit,
        model=judge_model,
        base_url=judge_base_url,
        api_key=judge_api_key,
    )
    m = out_report["meta"]
    vendor = "同源（与生成同一家）" if not m["judge_cross_vendor"] else "跨供应商"
    typer.echo(
        f"拒答审计：{m['n_refusable_with_answer']} 条拒答题有答案 → 判了 {m['n_judged']} 条"
        f"、跳过 {m['n_skipped']} 条"
    )
    typer.echo(
        f"  judge={m['judge_model']} @ {m['judge_base_url']}（{vendor}） · "
        f"{m['prompt_version']} · temperature={m['temperature']} · "
        f"思考={m['reasoning_effort'] or '默认'}"
    )
    for item in out_report["items"]:
        if item.get("skipped"):
            typer.echo(f"  [跳过] {item['id']}：{item['skipped']}")
            continue
        if item.get("fabricated") is None:
            typer.echo(f"  [判分失败] {item['id']}：{item.get('error')}")
            continue
        cost = "缓存命中" if item.get("cached") else f"{item.get('ms')}ms"
        typer.echo(
            f"  [{'有编造' if item['fabricated'] else '无编造'}] {item['id']}"
            f" {item['answer_len']} 字 / 引用 {item['n_citations']}（{cost}）"
        )
        if item["fabricated"]:
            typer.echo(f"      「{item['quote']}」—— {item['why']}")
    if out_report["fabrication_rate"] is not None:
        n_bad = sum(1 for i in out_report["items"] if i.get("fabricated"))
        typer.echo(
            f"\n→ 拒答编造率 = {out_report['fabrication_rate']}"
            f"（{n_bad}/{m['n_judged']}）；现口径把这批题统一记进「拒答正确率」"
        )
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(
            json.dumps(out_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        typer.echo(f"  逐条结果已落盘：{out}")
    if against is not None:
        if not Path(against).exists():
            typer.echo(f"对照文件不存在：{against}")
            raise typer.Exit(1)
        prev = json.loads(Path(against).read_text(encoding="utf-8"))
        cmp = compare_audits(prev, out_report)
        c = cmp["compared"]
        typer.echo(
            f"\n== 判读对照 == {c['prev_judge']} → {c['new_judge']}"
            f"（提示词 {c['prev_prompt']} → {c['new_prompt']}）"
        )
        typer.echo(
            f"  双方都判出结论的条目：{cmp['n_both_judged']} 条"
            f"（只有前一份判了 {len(cmp['n_only_prev'])} 条、"
            f"只有这一份 {len(cmp['n_only_new'])} 条）"
        )
        typer.echo(
            f"  编造率：{cmp['prev_rate']} → {cmp['new_rate']} · 逐条一致率 "
            f"{cmp['agreement']}"
        )
        for f in cmp["flips"]:
            typer.echo(
                f"  [翻判] {f['id']}：{'编造' if f['prev'] else '无编造'} → "
                f"{'编造' if f['new'] else '无编造'}｜「{f['new_quote']}」—— {f['new_why']}"
            )
        if not cmp["flips"]:
            typer.echo("  没有任何一条判读翻转：结论不依赖具体是哪一家在判")


def _parse_compare_groups(specs: list[str]) -> list[dict]:
    """解析 `标签=文件1,文件2` 形式的分组；组内多文件 = 同条件重跑。"""
    groups: list[dict] = []
    for spec in specs:
        label, _, files = spec.partition("=")
        groups.append(
            {
                "label": label,
                "files": [Path(f.strip()) for f in files.split(",") if f.strip()],
            }
        )
    if len(groups) < 2:
        typer.echo("至少需要两个 --group（第一个作基线）")
        raise typer.Exit(1)
    for g in groups:
        for f in g["files"]:
            if not f.exists():
                typer.echo(f"结果文件不存在：{f}")
                raise typer.Exit(1)
    return groups


@app.command("compare-ragas")
def compare_ragas(
    group: Annotated[
        list[str],
        typer.Option(
            "--group",
            help="分组，格式 标签=文件1,文件2（同组多文件=同条件重跑）；第一个分组为基线",
        ),
    ],
) -> None:
    """配对判读多组 RAGAS 结果：全量配对差 + 噪声地板 + 子集敏感性。"""
    from doc_rag.eval.compare import apply_holm, compare, format_report

    groups = _parse_compare_groups(group)
    report = compare(groups)
    apply_holm([report])
    typer.echo(format_report(report))


@app.command("compare-retrieval")
def compare_retrieval(
    group: Annotated[
        list[str],
        typer.Option(
            "--group",
            help="分组，格式 标签=文件1,文件2（同组多文件=同条件重跑）；第一个分组为基线",
        ),
    ],
    metric: Annotated[
        list[str] | None,
        typer.Option(
            help="只判读指定指标（可重复）；默认一次跑齐 Hit@5/Hit@8/MRR/nDCG@8/"
            "覆盖率/覆盖率对上限/聚合题要点命中"
        ),
    ] = None,
) -> None:
    """配对判读客观指标结果：同题配对差 + 95%CI + 符号检验 + 跨指标 Holm 校正。

    逐条分数来自 `items[]`。这里有两个层级的指标，都走同一套判读但**告警口径不同**：

    - 检索级（hit/mrr/ndcg/覆盖率，分母 = 有 gold 的条目，拒答题不进）：两臂清单不等长
      时差值部分是长度的函数，报告会把这层伪影单独标出来；
    - 答案级（`keypoint_hit_ratio`，分母 = 有逐篇要点的聚合题）：只看答案文本，块数
      不等是实验变量不是伪影；这条轨上唯一该拦的是两臂的要点数 K 不同（分母变了）。
    """
    from doc_rag.eval.compare import (
        METRIC_ALIASES,
        RETRIEVAL_METRICS,
        RETRIEVAL_SWEEP,
        format_report,
    )
    from doc_rag.eval.compare import compare_retrieval as sweep

    metrics = tuple(metric) if metric else RETRIEVAL_SWEEP
    # 旧名先回落成标准名再校验：PLAN/README 里写下的复现命令（`--metric
    # keypoint_hit_ratio`）不能因改名失效。顺带去重——同一指标用两个名字点进来
    # 会在 Holm 家族里被算成两次比较，白白收紧其余指标的 α。
    metrics = tuple(dict.fromkeys(METRIC_ALIASES.get(m, m) for m in metrics))
    unknown = [m for m in metrics if m not in RETRIEVAL_METRICS]
    if unknown:
        typer.echo(f"未知指标：{unknown}（可选：{', '.join(RETRIEVAL_METRICS)}）")
        raise typer.Exit(1)
    groups = _parse_compare_groups(group)
    for report in sweep(groups, metrics=metrics):
        typer.echo(format_report(report))
        typer.echo("")


@app.command()
def backfill(
    kb: Annotated[str | None, typer.Option(help="Qdrant collection 名")] = None,
) -> None:
    """按当前规则重算 payload 元数据（如 doc_date）并原地更新——不重新向量化。

    动机：元数据规则演进（如 doc_date 增补正文抽取）后无需重灌全库。
    """

    from qdrant_client import QdrantClient, models

    from doc_rag.ingest.metadata import base_meta
    from doc_rag.ingest.schema import IntermediateDoc

    cfg = load_config()
    collection = kb or cfg["qdrant"]["collection"]
    parsed = Path(cfg["paths"]["parsed"])
    client = QdrantClient(url=cfg["qdrant"]["url"], timeout=60)

    updated = 0
    for json_file in sorted(parsed.glob("*.json")):
        if json_file.name == "profile.json":
            continue
        try:
            doc = IntermediateDoc.model_validate_json(
                json_file.read_text(encoding="utf-8")
            )
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
                    key="doc_date",
                    range=models.DatetimeRange(gte=datetime(2000, 1, 1, tzinfo=UTC)),
                )
            ]
        ),
    ).count
    total = client.count(collection).count
    typer.echo(f"已更新 {updated} 篇文档的 payload")
    typer.echo(
        f"doc_date 覆盖率：{with_date}/{total} = {with_date / max(total, 1):.1%}"
    )


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "0.0.0.0",
    port: Annotated[int, typer.Option()] = 8000,
) -> None:
    """启动 FastAPI（:8000）。"""
    import uvicorn

    from doc_rag.log import configure

    configure()  # 结构化 JSON 日志走 stderr，uvicorn 的访问日志照常
    uvicorn.run("doc_rag.api.main:app", host=host, port=port)


@app.command()
def demo(
    kb: Annotated[str | None, typer.Option(help="Qdrant collection 名")] = None,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 7860,
) -> None:
    """Gradio 演示页（T8）：上传问答 / 引用 / 延迟面板。需 demo extra：uv sync --extra demo。"""
    try:
        import gradio
    except ImportError:
        typer.echo("gradio 未安装（demo extra）：uv sync --extra demo 后重试")
        raise typer.Exit(1)

    import doc_rag.api.demo as demo_mod

    orchestrator = None
    if kb:
        from doc_rag.orchestrator import Orchestrator

        orchestrator = Orchestrator(load_config(), collection=kb)
    demo_mod.build_ui(gradio, orchestrator).launch(server_name=host, server_port=port)


# ── 外部基准：RGB（AAAI 2024 中文子集）────────────────────────────────────────
#
# 为什么要跑外部基准：本仓库所有已发布数字都测在公司语料上，而语料与结果文件按合规
# 全部 gitignore 了——**面试官无法独立复核任何一个数**。RGB 公开可 clone，在它上面
# 跑出来的数字别人能自己复现；而它的四类能力（拒答 / 信息整合 / 抗噪 / 反事实）
# 正对着本项目的强项。协议与偏差声明见 src/doc_rag/benchmarks/rgb.py 的 docstring。

_RGB_RAW_BASE = "https://raw.githubusercontent.com"


@app.command("bench-rgb-fetch")
def bench_rgb_fetch(
    out_dir: Annotated[
        Path | None,
        typer.Option("--dir", help="落到哪里（默认 paths.bench 下的 rgb/）"),
    ] = None,
    force: Annotated[bool, typer.Option(help="已存在也重下")] = False,
) -> None:
    """抓 RGB 中文四个文件到本地（**数据不入库**：上游 CC BY-NC-SA 4.0，非商用）。

    按固定 commit 抓，并写一份本地 MANIFEST（commit / 许可 / 每个文件的 sha256 与
    行数）——结果文件要靠它自证「这份读数测的是哪个版本的数据」。
    """
    import hashlib
    import json

    import httpx

    from doc_rag.benchmarks import rgb

    cfg = load_config()
    root = out_dir or Path(cfg["paths"]["bench"]) / "rgb"
    root.mkdir(parents=True, exist_ok=True)
    # 官方仓库里这四份的行数（用来发现「抓到了半截」或「上游换了内容」）
    expected_lines = {"zh": 300, "zh_refine": 300, "zh_int": 100, "zh_fact": 100}
    owner_repo = rgb.UPSTREAM_REPO.removeprefix("https://github.com/").removesuffix(
        ".git"
    )
    manifest: dict = {
        "repo": rgb.UPSTREAM_REPO,
        "commit": rgb.UPSTREAM_COMMIT,
        "license": rgb.UPSTREAM_LICENSE,
        "note": "本地副本仅供本地评测；上游许可含 ShareAlike，副本不再分发",
        "files": {},
    }
    manual = (
        f"（手动等价命令：git clone {rgb.UPSTREAM_REPO} && "
        f"git -C RGB checkout {rgb.UPSTREAM_COMMIT} && "
        f"copy RGB/data/*.json {root}）"
    )
    failed: list[str] = []
    for name, want in expected_lines.items():
        target = rgb.dataset_path(root, name)
        if target.exists() and not force:
            raw = target.read_bytes()
            typer.echo(f"  {name}.json 已存在（{len(raw) // 1024} KB），跳过")
        else:
            url = f"{_RGB_RAW_BASE}/{owner_repo}/{rgb.UPSTREAM_COMMIT}/data/{name}.json"
            try:
                resp = httpx.get(url, timeout=60.0, follow_redirects=True)
                resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001 网络失败要给可执行的替代路径
                typer.echo(f"  [失败] {name}.json：{exc} {manual}")
                failed.append(name)
                continue
            target.write_bytes(resp.content)
            raw = resp.content
            typer.echo(f"  {name}.json 下载完成（{len(raw) // 1024} KB）")
        lines = raw.decode("utf-8").count("\n")
        manifest["files"][f"{name}.json"] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "lines": lines,
            "lines_expected": want,
            "lines_match": lines == want,
        }
        if lines != want:
            # 行数不对就不是同一份数据，读数不可比——报出来而不是悄悄继续
            typer.echo(
                f"  ⚠ {name}.json 行数 {lines} ≠ 官方 {want}，请核对上游是否变更"
            )
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(f"MANIFEST 已写入 {root / 'MANIFEST.json'}")
    if failed:
        raise typer.Exit(1)


@app.command("bench-rgb-prepare")
def bench_rgb_prepare(
    dataset: Annotated[
        str, typer.Option("--dataset", help="哪个数据集（zh / zh_refine / zh_int）")
    ] = "zh",
    limit: Annotated[
        int | None, typer.Option(help="只准备前 N 题（试跑用；判读用全量）")
    ] = None,
    dir_path: Annotated[
        Path | None,
        typer.Option("--dir", help="基准数据目录（默认 paths.bench 下的 rgb/）"),
    ] = None,
) -> None:
    """备检索变体的料：把数据集文档并成语料 + 生成同格式的黄金集。

    **这条路径测的才是本项目的 RAG**：官方协议是「文档由数据集提供」，检索层不参与，
    所以那份读数只反映 prompt 与模型选型。这里把文档灌进 Qdrant，让系统自己检索
    ——分块、Dense+BM25 融合、重排、上下文预算全部进环路。

    判据等价：`eval/runner.py` 的 `all(must_contain 都在答案里)` 与 RGB 官方的
    「全部 ground-truth 命中」是同一条规则，所以 `strict_keyword_accuracy` 就是
    RGB 的 `all_rate`——两行数字可以直接对比，差值即**检索层的贡献**。

    只备料、不跑：接下来的 ingest 与 eval 用仓库现成命令（脚本会打印出来）。
    """
    from doc_rag.benchmarks import rgb, rgb_retrieval

    cfg = load_config()
    if dataset not in rgb.PROTOCOL:
        typer.echo(f"未知数据集：{dataset}（可选：{sorted(rgb.PROTOCOL)}）")
        raise typer.Exit(1)
    root = dir_path or Path(cfg["paths"]["bench"]) / "rgb"
    src = rgb.dataset_path(root, dataset)
    if not src.exists():
        typer.echo(f"基准数据缺失：{src}；先跑 doc-rag bench-rgb-fetch")
        raise typer.Exit(1)

    records = rgb.load_records(src)
    corpus = rgb_retrieval.build_corpus(records)
    parsed_dir = root / f"parsed_{dataset}"
    n = rgb_retrieval.write_parsed(corpus, parsed_dir)
    gold_path = root / f"gold_{dataset}_retrieval.json"
    rgb_retrieval.write_gold(
        rgb_retrieval.build_gold(records, dataset, limit=limit), gold_path
    )
    kb = rgb_retrieval.COLLECTION[dataset]
    chars = sum(len(t) for t in corpus.values())
    typer.echo(f"语料：{n:,} 篇（去重后）· 合计 {chars:,} 字 → {parsed_dir}")
    typer.echo(f"黄金集：{gold_path}（{limit or len(records)} 题）")
    typer.echo(
        "\n接下来（仓库现成命令）：\n"
        f"  uv run doc-rag ingest --parsed-dir {parsed_dir} --kb {kb} --recreate --index-only\n"
        f"  uv run doc-rag eval   --kb {kb} --gold {gold_path} --rewrite --rerank --fresh-answers"
    )
    typer.echo(
        "注意：--index-only 跳过解析（中间 JSON 已经写好）；"
        "eval 的 strict_keyword_accuracy 与给定文档那行的 all_rate 同判据、可直接对比。"
    )


@app.command("bench-rgb-run")
def bench_rgb_run(
    datasets: Annotated[
        list[str] | None,
        typer.Option("--dataset", help="跑哪些数据集（默认全部中文四个）"),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(help="每个组合只跑前 N 条——**只配试跑**，判读不得用截断样本"),
    ] = None,
    sample: Annotated[
        int | None,
        typer.Option(
            help="每个组合**均匀抽** N 条（固定种子，可复现、可续跑到全量）——"
            "降成本又要判读时用这个，不要用 --limit"
        ),
    ] = None,
    instruction: Annotated[
        str,
        typer.Option(
            help="production=本系统生产 prompt（默认）｜rgb=官方 instruction 锚点"
        ),
    ] = "production",
    out: Annotated[
        Path | None, typer.Option(help="逐条结果落盘路径（默认 data/eval/rgb_*.jsonl）")
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes", help="确认全量跑：调用数 > 50 且未给 --limit 时必须显式给"
        ),
    ] = False,
    dir_path: Annotated[
        Path | None,
        typer.Option("--dir", help="基准数据目录（默认 paths.bench 下的 rgb/）"),
    ] = None,
    resume: Annotated[
        Path | None,
        typer.Option(
            help="从这份已有结果续跑（已完成的条目跳过、失败的会重试，结果追加写）"
        ),
    ] = None,
    workers: Annotated[
        int,
        typer.Option(
            help="并发线程数（默认 4）。只改墙钟：每条记录的输入只依赖它自己；"
            "实测有分钟级端点停顿，串行跑几千次会被拖到几十小时"
        ),
    ] = 4,
    noise_rates: Annotated[
        list[float] | None,
        typer.Option(
            "--noise-rate",
            help="只跑指定噪声档（可重复）。用途是**同题对照**：与检索变体对比只需 0.0",
        ),
    ] = None,
) -> None:
    """按 RGB 协议跑一遍并落盘逐条结果。

    文档由数据集提供，所以**不经过检索、不消耗嵌入**——本命令的调用数就是合成调用数。
    先打印调用计划（组合数 × 条数），超阈值要求先试跑或显式确认：这是本仓库的
    「先算账再动手」纪律，而这里最容易算错的就是组合数。

    全量是几千次调用、几小时的长跑，所以**逐条落盘并 flush**、支持 `--resume` 续跑、
    单条失败只记录不中断——一次网络抖动不该让前面几小时的调用白花。
    """
    import json

    from doc_rag.benchmarks import rgb, rgb_runner
    from doc_rag.generate import prompts

    if instruction not in ("production", "rgb"):
        typer.echo(f"未知 instruction：{instruction!r}（production | rgb）")
        raise typer.Exit(1)
    cfg = load_config()
    chosen = list(datasets) if datasets else list(rgb.PROTOCOL)
    unknown = [d for d in chosen if d not in rgb.PROTOCOL]
    if unknown:
        typer.echo(f"未知数据集：{unknown}（可选：{sorted(rgb.PROTOCOL)}）")
        raise typer.Exit(1)
    root = dir_path or Path(cfg["paths"]["bench"]) / "rgb"
    missing = [d for d in chosen if not rgb.dataset_path(root, d).exists()]
    if missing:
        typer.echo(
            f"基准数据缺失：{missing}；先跑 doc-rag bench-rgb-fetch（目录 {root}）"
        )
        raise typer.Exit(1)

    records = {d: rgb.load_records(rgb.dataset_path(root, d)) for d in chosen}
    only_rates = tuple(noise_rates) if noise_rates else None
    plan = rgb_runner.call_plan(
        chosen,
        {d: len(records[d]) for d in chosen},
        sample=sample,
        only_rates=only_rates,
    )
    done = rgb_runner.done_index(rgb_runner.load_rows(resume)) if resume else {}
    if resume:
        typer.echo(f"续跑：{resume} 里已有 {len(done)} 条完成，本次只补未完成的")
    typer.echo(f"调用计划：{plan['combos']} 个组合，共 {plan['calls']} 次合成调用")
    for ds, rate, n in plan["per_combo"]:
        typer.echo(f"  {ds:<10} noise={rate:<4} n={n}")
    pending = max(plan["calls"] - len(done), 0)
    if limit is None and pending > 50 and not yes:
        typer.echo(
            f"待跑 {pending} 次调用（>50）：先 `--limit 5` 试跑核实用量，"
            "或 `--sample 100` 降档但保留可判读性，或确认后加 `--yes`。"
        )
        raise typer.Exit(1)

    orchestrator = None
    if instruction == "production":
        from doc_rag.orchestrator import Orchestrator

        # 给定上下文的入口不检索，所以这个 Orchestrator 不会建 Qdrant 连接
        orchestrator = Orchestrator(cfg)

    # 与 eval 结果文件同一约定：本地时区（astimezone 带上 tz，避免 naive datetime）
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    suffix = f"_limit{limit}" if limit else (f"_sample{sample}" if sample else "")
    if out is None:
        if resume is not None:
            # 续跑必须写回同一个文件：另起一个新文件会让汇总算进旧行、而新文件里
            # 没有它们——结果文件从此与汇总对不上。
            out = resume
        else:
            out = (
                Path(cfg["paths"]["eval"]) / f"rgb_{instruction}{suffix}_{stamp}.jsonl"
            )
    elif suffix and resume is None:
        # 非全量跑绝不允许被误当成正式读数：文件名强制带标记
        out = out.with_name(f"{out.stem}{suffix}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    # 续跑时追加写（append），新跑则新建：两种情况下都**逐条 flush**——
    # 长跑中途被打断时，已花的调用必须留在盘上。
    mode = "a" if resume else "w"
    with open(out, mode, encoding="utf-8") as f:

        def _sink(row: dict, _f=f) -> None:
            _f.write(json.dumps(row, ensure_ascii=False) + "\n")
            _f.flush()

        for ds, rate in rgb_runner.protocol_combos(chosen, only_rates=only_rates):
            rows = rgb_runner.run_combo(
                cfg,
                records[ds],
                ds,
                rate,
                instruction=instruction,
                limit=limit,
                sample=sample,
                workers=workers,
                orch=orchestrator,
                done=done,
                sink=_sink,
            )
            s = rgb_runner.summarize(rows, rgb_runner.record_index(records[ds], ds))
            s["instruction"] = instruction
            summaries.append(s)
            err = s["n_errors"]
            typer.echo(
                f"  {ds:<10} noise={rate:<4} n={s['n']:<5} "
                f"all_rate={s['all_rate'] * 100:.2f}%"
                + (f"  ⚠ 失败 {err} 条（已剔出分母）" if err else "")
            )

    meta = {
        "benchmark": "RGB",
        "repo": rgb.UPSTREAM_REPO,
        "commit": rgb.UPSTREAM_COMMIT,
        "license": rgb.UPSTREAM_LICENSE,
        "instruction": instruction,
        "datasets": chosen,
        "limit_per_combo": limit,
        "sample_per_combo": sample,
        "noise_rates": list(only_rates) if only_rates else None,
        # 两个标记要分开：`limit` 是取前 N（不可判读），`sample` 是均匀抽样（可判读，
        # 但仍是全量的子集，报数时要带上 n）。混成一个「truncated」会让读者无法分辨。
        "truncated": limit is not None,
        "sampled": sample is not None,
        "model": cfg["llm"].get("model"),
        "temperature": cfg["llm"].get("temperature"),
        "reasoning_effort": cfg["llm"].get("reasoning_effort"),
        "reasoning_effort_by_type": cfg["llm"].get("reasoning_effort_by_type"),
        "task_question_type": rgb_runner.TASK_QUESTION_TYPE,
        "call_plan": plan,
    }
    if instruction == "production":
        # 只在生产行有意义：rgb 行的 prompt 是官方模板，不是本仓库的 prompt 版本
        meta["prompt_fingerprint"] = prompts.fingerprint(
            cfg["llm"].get("prompt_version")
        )
        meta["prompt_version"] = cfg["llm"].get("prompt_version") or "tightened"
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {"meta": meta, "summaries": summaries}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    typer.echo(f"逐条结果 → {out}\n汇总与 meta → {summary_path}")
    typer.echo(rgb_runner.render_reports(summaries))
    if limit:
        # 外推的输入是**本次全部组合的合计**：拿一个组合去乘组合数会把 4000 说成 250
        usage = rgb_runner.total_usage(summaries)
        full = rgb_runner.call_plan(chosen, {d: len(records[d]) for d in chosen})
        extrap = rgb_runner.extrapolate(usage, usage["n"], full["calls"])
        typer.echo(
            f"\n成本外推（本次 {usage['n']} 条实测，每个组合 {limit} 条）："
            f"约 {extrap['per_call_prompt_tokens']} 输入 + "
            f"{extrap['per_call_completion_tokens']} 输出 token/条、"
            f"{extrap['per_call_ms'] / 1000:.1f}s/条 → 全量 {extrap['calls']} 次调用约需 "
            f"{extrap['prompt_tokens']:,} 输入 / {extrap['completion_tokens']:,} 输出 token"
            f"（其中思考 {extrap['reasoning_tokens']:,}）、串行约 {extrap['wall_hours']} 小时"
        )
        typer.echo("⚠ 本轮是截断样本，**不得用于判读**；全量请去掉 --limit 重跑。")
    if sample:
        typer.echo(
            f"\n⚠ 本轮是**均匀抽样**（每组合最多 {sample} 条，固定种子）：可以判读，"
            "但报数必须带上 n。补全量直接 `--resume` 指向这份文件——已完成的不会重复付费。"
        )


@app.command("bench-rgb-score")
def bench_rgb_score(
    results: Annotated[
        list[Path],
        typer.Argument(
            help="bench-rgb-run 产出的逐条结果（可给多份，按 instruction 分行印）"
        ),
    ],
    judge: Annotated[
        bool,
        typer.Option(
            help="跑星号口径（Rej*/ED*，要调 LLM；判分旁挂缓存，重跑不重复付费）"
        ),
    ] = False,
) -> None:
    """判分并出并排表。`--judge` 对应官方的第二步（reject_evalue.py / fact_evalue.py）。

    判分只读落盘的答案，**不重跑生成**——所以同一份结果可以用不同判据反复读，
    这是把「生成」与「判分」两次花钱分开的前提（官方的两步脚本也是这个形状）。
    """
    import json

    from doc_rag.benchmarks import rgb, rgb_runner

    cfg = load_config()
    root = Path(cfg["paths"]["bench"]) / "rgb"
    judge_cfg: dict | None = None
    if judge:
        from doc_rag.eval.judge import judge_cfg as _jc

        judge_cfg = _jc(cfg)

    all_summaries: list[dict] = []
    for path in results:
        if not path.exists():
            typer.echo(f"结果文件不存在：{path}")
            raise typer.Exit(1)
        # 判分旁挂文件长得像结果文件（同目录、同后缀、最近被改过），很容易被
        # 通配符选中当输入——它的行没有 dataset 等字段，进来只会炸在深层。
        if ".judge" in path.stem:
            typer.echo(
                f"{path} 是判分旁挂文件，不是逐条结果；请给 bench-rgb-run 产出的 .jsonl"
            )
            raise typer.Exit(1)
        rows: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            typer.echo(f"{path} 是空的，跳过")
            continue
        # 每个组合单独判分：指标的分母是「一个组合内的条数」，混着算没有意义
        if judge and judge_cfg is not None:
            sidecar = rgb_runner.judge_sidecar_path(path)
            cache = rgb_runner.load_judge_sidecar(sidecar)
            with open(sidecar, "a", encoding="utf-8") as sink_f:

                def _sink(item: dict, _f=sink_f) -> None:
                    _f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    _f.flush()

                rows = rgb_runner.judge_rows(rows, judge_cfg, cache, _sink)
            typer.echo(f"判分旁挂 → {sidecar}（{len(cache)} 条已缓存）")
        # 键必须带数据集：四个数据集的 id 各自从 0 开始，只按 id 建索引会互相覆盖，
        # 于是 fakeanswer 取不到、反事实族的假阳性被静默算成 0（实测踩过）
        loaded: dict[str, list[rgb.Record]] = {}
        for ds in {str(r["dataset"]) for r in rows}:
            p = rgb.dataset_path(root, ds)
            if p.exists():
                loaded[ds] = rgb.load_records(p)
        records = rgb_runner.record_index(loaded)
        groups: dict[tuple, list[dict]] = {}
        for row in rows:
            key = (row["instruction"], row["dataset"], row["noise_rate"])
            groups.setdefault(key, []).append(row)
        for (_instr, _ds, _rate), grp in sorted(
            groups.items(), key=lambda kv: str(kv[0])
        ):
            s = rgb_runner.summarize(grp, records)
            if judge:
                s.update(rgb_runner.star_rates(grp))
            all_summaries.append(s)

    if not all_summaries:
        typer.echo("没有可判分的结果。")
        raise typer.Exit(1)
    typer.echo(rgb_runner.render_reports(all_summaries))
    usage = {
        k: sum(int(s.get("usage", {}).get(k) or 0) for s in all_summaries)
        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")
    }
    # 只报可核的量（调用数 / token）：本仓库的 ¥ 数历来是粗估，不在这里折算
    typer.echo(
        f"\n本批生成用量：{sum(s['n'] for s in all_summaries)} 条 · "
        f"输入 {usage['prompt_tokens']:,} / 输出 {usage['completion_tokens']:,} token"
        f"（其中思考 {usage['reasoning_tokens']:,}）"
        f" · 命中缓存 {sum(s.get('cached_n', 0) for s in all_summaries)} 条"
    )

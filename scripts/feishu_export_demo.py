"""飞书 OpenAPI 导出文档 demo（演示用，未接入正式 ingest 管线）。

演示两条导出路径，并对照本项目 IR：

路径 A（推荐了解结构）：Docx Blocks API
  GET /open-apis/docx/v1/documents/{document_id}/blocks
  → Block 树 JSON → 映射为 IntermediateDoc（src/doc_rag/ingest/schema.py）

路径 B（对齐现状「下载文件」）：Drive Export Task
  创建导出任务 → 轮询 → 下载 PDF/docx 到 data/raw
  → 仍走现有 PyMuPDF / mammoth 解析

前置（飞书开放平台）：
  1. 创建企业自建应用，开通权限（至少一种）：
       docx:document:readonly   读云文档（路径 A）
       drive:drive              读云空间（路径 B 导出）
  2. 应用可用后，在目标文档「…」→「…更多」→「添加文档应用」授权
  3. 导出环境变量：
       FEISHU_APP_ID / FEISHU_APP_SECRET
       FEISHU_DOC_ID            例如 doxcnAJ9VRRJqVMYZ1MyKnabcef
       （可选）FEISHU_WIKI_TOKEN  若文档在知识库，先换 document_id

用法（Windows PowerShell）：
  $env:FEISHU_APP_ID="cli_xxx"
  $env:FEISHU_APP_SECRET="xxx"
  $env:FEISHU_DOC_ID="doxcn..."
  uv run python scripts/feishu_export_demo.py ir
  uv run python scripts/feishu_export_demo.py raw
  uv run python scripts/feishu_export_demo.py pdf --out data/raw
  uv run python scripts/feishu_export_demo.py map --input data/feishu/raw_blocks.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Self

import httpx

# 项目 IR（pydantic 模型），保证 demo 产物与 ingest 契约一致
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from doc_rag.ingest.schema import Block, IntermediateDoc, SourceMeta

BASE = "https://open.feishu.cn/open-apis"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "feishu"
PAGE_SIZE = 500  # 官方上限
QPS_SLEEP = 0.22  # 文档 Block 接口应用级限流约 5 QPS

# 飞书 block_type → 本项目 IR block type
# 见 https://open.feishu.cn/document/server-docs/docs/docs/docx-v1/document-block/list
_SKIP_TYPES = {
    22,
    24,
    25,
    26,
    28,
    29,
    33,
    34,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
}
_TYPE_MAP = {
    2: "paragraph",  # Text
    12: "list_item",  # Bullet
    13: "list_item",  # Ordered
    14: "paragraph",  # Code
    15: "quote",  # Quote
    17: "list_item",  # Todo
    18: "paragraph",  # Bitable 占位
    19: "paragraph",  # Callout
    31: "table",  # Table
}


# ---------------------------------------------------------------------------
# HTTP：鉴权 / 文档元数据 / Blocks 分页 / 导出任务
# ---------------------------------------------------------------------------


class FeishuClient:
    def __init__(self, app_id: str, app_secret: str, timeout: float = 30.0) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self._token: str | None = None
        self._http = httpx.Client(timeout=timeout, follow_redirects=True)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _get_token(self) -> str:
        if self._token:
            return self._token
        resp = self._http.post(
            f"{BASE}/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"获取 tenant_access_token 失败: {data}")
        self._token = data["tenant_access_token"]
        return self._token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._get_token()}"}

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET + 简单重试（99991400 = 触发频控）。"""
        last: dict[str, Any] = {}
        for attempt in range(5):
            resp = self._http.get(
                f"{BASE}{path}", params=params, headers=self._headers()
            )
            resp.raise_for_status()
            last = resp.json()
            if last.get("code") == 0:
                return last
            if last.get("code") == 99991400:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise RuntimeError(f"飞书 API 失败 [{path}] {last}")
        raise RuntimeError(f"飞书 API 频控重试耗尽 [{path}] {last}")

    def _post(
        self,
        path: str,
        body: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resp = self._http.post(
            f"{BASE}{path}", json=body, params=params, headers=self._headers()
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"飞书 API 失败 [{path}] {data}")
        return data

    def get_document_meta(self, document_id: str) -> dict[str, Any]:
        return self._get(f"/docx/v1/documents/{document_id}")["data"]["document"]

    def list_blocks(self, document_id: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "page_size": PAGE_SIZE,
                "document_revision_id": -1,
            }
            if page_token:
                params["page_token"] = page_token
            data = self._get(f"/docx/v1/documents/{document_id}/blocks", params=params)[
                "data"
            ]
            items.extend(data.get("items") or [])
            if not data.get("has_more"):
                break
            page_token = data.get("page_token")
            time.sleep(QPS_SLEEP)
        return items

    def resolve_document_id_from_wiki(self, wiki_token: str) -> str:
        """知识库节点 → 文档 document_id（节点 token ≠ docx document_id）。"""
        data = self._get("/wiki/v2/spaces/get_node", params={"token": wiki_token})[
            "data"
        ]
        node = data.get("node") or {}
        obj_token = node.get("obj_token")
        if not obj_token:
            raise RuntimeError(f"wiki 节点无 obj_token: {data}")
        return obj_token

    # ---- 路径 B：导出任务 ----

    def create_export_task(
        self, file_token: str, obj_type: str = "docx", file_extension: str = "pdf"
    ) -> str:
        data = self._post(
            "/drive/v1/export_tasks",
            body={
                "file_extension": file_extension,
                "token": file_token,
                "type": obj_type,
            },
        )["data"]
        ticket = (data.get("result") or {}).get("ticket")
        if not ticket:
            raise RuntimeError(f"创建导出任务失败: {data}")
        return ticket

    def wait_export_task(self, ticket: str, timeout_s: float = 60.0) -> dict[str, Any]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            data = self._get(f"/drive/v1/export_tasks/{ticket}")["data"]
            result = data.get("result") or {}
            job_status = result.get("job_status")
            # 0 成功 / 1 初始化 / 2 处理中 / 3 失败 / 4 超时 / 5 取消
            if job_status == 0:
                return result
            if job_status in (3, 4, 5):
                raise RuntimeError(f"导出任务失败 job_status={job_status}: {result}")
            time.sleep(1.0)
        raise TimeoutError(f"导出任务超时: ticket={ticket}")

    def download_export(self, file_token: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._http.stream(
            "GET",
            f"{BASE}/drive/v1/export_tasks/file/{file_token}/download",
            headers=self._headers(),
        ) as resp:
            resp.raise_for_status()
            with dest.open("wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
        return dest


# ---------------------------------------------------------------------------
# Block 树 → IntermediateDoc
# ---------------------------------------------------------------------------


def _elements_text(elements: list[dict[str, Any]] | None) -> str:
    parts: list[str] = []
    for el in elements or []:
        if "text_run" in el:
            parts.append((el["text_run"] or {}).get("content") or "")
        elif "mention_user" in el:
            uid = (el["mention_user"] or {}).get("user_id") or ""
            parts.append(f"@{uid}" if uid else "@某人")
        elif "mention_doc" in el:
            title = (el["mention_doc"] or {}).get("title") or "文档"
            parts.append(f"@{title}")
        elif "equation" in el:
            parts.append((el["equation"] or {}).get("content") or "")
        elif "undefined" in el:
            continue
    return "".join(parts).replace("\u200b", "").replace("\ufeff", "")


def _block_payload(b: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None]:
    """根据 block_type 取到承载正文的字段名与 elements。"""
    bt = b.get("block_type")
    if bt == 1:
        return "page", (b.get("page") or {}).get("elements")
    if bt == 2:
        return "text", (b.get("text") or {}).get("elements")
    if 3 <= bt <= 11:
        key = f"heading{bt - 2}"  # 3→heading1 … 11→heading9
        return key, (b.get(key) or {}).get("elements")
    if bt == 12:
        return "bullet", (b.get("bullet") or {}).get("elements")
    if bt == 13:
        return "ordered", (b.get("ordered") or {}).get("elements")
    if bt == 14:
        return "code", (b.get("code") or {}).get("elements")
    if bt == 15:
        return "quote", (b.get("quote") or {}).get("elements")
    if bt == 17:
        return "todo", (b.get("todo") or {}).get("elements")
    if bt == 19:
        return "callout", (b.get("callout") or {}).get("elements")
    # 其余类型没有直接 elements，或仅有 children
    return "", None


def _index_blocks(raw: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {b["block_id"]: b for b in raw if b.get("block_id")}


def _table_to_markdown(
    table_block: dict[str, Any], index: dict[str, dict[str, Any]]
) -> str:
    """把 table(block 31) 的 children(cell) 递归取文本，拼成 Markdown 表。"""
    prop = (table_block.get("table") or {}).get("property") or {}
    rows = int(prop.get("row_size") or 0)
    cols = int(prop.get("column_size") or 0)
    cell_ids = table_block.get("children") or []
    if rows <= 0 or cols <= 0 or not cell_ids:
        texts = []
        for cid in cell_ids:
            cell = index.get(cid) or {}
            texts.append(_subtree_text(cell, index))
        return " | ".join(t.replace("\n", " ") for t in texts if t.strip()) or "(空表)"

    grid: list[list[str]] = []
    for r in range(rows):
        row: list[str] = []
        for c in range(cols):
            cid = cell_ids[r * cols + c] if r * cols + c < len(cell_ids) else None
            cell = index.get(cid) if cid else None
            t = _subtree_text(cell, index) if cell else ""
            row.append(t.replace("\n", " ").replace("|", "\\|").strip() or "")
        grid.append(row)

    lines = [
        "| " + " | ".join(grid[0]) + " |",
        "| " + " | ".join(["---"] * cols) + " |",
    ]
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _subtree_text(
    block: dict[str, Any] | None, index: dict[str, dict[str, Any]]
) -> str:
    if not block:
        return ""
    _, elements = _block_payload(block)
    parts = [_elements_text(elements)] if elements else []
    for cid in block.get("children") or []:
        child = index.get(cid)
        if not child:
            continue
        bt = child.get("block_type")
        if bt in _SKIP_TYPES:
            continue
        if bt == 31:
            parts.append(_table_to_markdown(child, index))
        else:
            parts.append(_subtree_text(child, index))
    return "\n".join(p for p in parts if p and p.strip())


def blocks_to_ir(
    raw_blocks: list[dict[str, Any]],
    *,
    document_id: str,
    title: str | None = None,
    revision_id: int | None = None,
) -> IntermediateDoc:
    """飞书 Block 扁平列表（官方 items）→ 项目 IntermediateDoc。

    说明：
    - 按 children 树序展开，跳过容器/布局块
    - 标题层级来自 block_type 3~11
    - 表格重建为 Markdown 单块（与 office 通道同风格）
    - 在线文档无 page/bbox；doc_id 使用飞书 document_id（稳定身份，非 sha256）
    """
    index = _index_blocks(raw_blocks)
    root = next((b for b in raw_blocks if b.get("block_type") == 1), None)
    if title is None and root:
        title = _elements_text((root.get("page") or {}).get("elements"))

    ir_blocks: list[Block] = []

    def walk(bid: str) -> None:
        b = index.get(bid)
        if not b:
            return
        bt = b.get("block_type")
        if bt in _SKIP_TYPES:
            return
        if bt == 31:
            text = _table_to_markdown(b, index)
            if text.strip():
                ir_blocks.append(Block(type="table", text=text))
            return

        _field, elements = _block_payload(b)
        text = _elements_text(elements).strip() if elements else ""

        if 3 <= bt <= 11:
            if text:
                ir_blocks.append(Block(type="heading", text=text, heading_level=bt - 2))
        elif bt in _TYPE_MAP and text:
            ir_blocks.append(Block(type=_TYPE_MAP[bt], text=text))
        elif bt == 1 and text:
            # Page 根节点：其 page.elements 是文档标题，meta.title 已收，不重复
            pass
        # 布局容器等：只下钻
        for cid in b.get("children") or []:
            walk(cid)

    if root:
        for cid in root.get("children") or []:
            walk(cid)
    else:
        # 无 page 根时按返回顺序兜底
        for b in raw_blocks:
            if b.get("block_type") == 1:
                continue
            bt = b.get("block_type")
            if bt in _SKIP_TYPES:
                continue
            if bt == 31:
                t = _table_to_markdown(b, index)
                if t.strip():
                    ir_blocks.append(Block(type="table", text=t))
                continue
            _, elements = _block_payload(b)
            t = _elements_text(elements).strip() if elements else ""
            if 3 <= bt <= 11 and t:
                ir_blocks.append(Block(type="heading", text=t, heading_level=bt - 2))
            elif bt in _TYPE_MAP and t:
                ir_blocks.append(Block(type=_TYPE_MAP[bt], text=t))

    # doc_id：优先稳定 document_id；另在 title 旁保留 revision 便于对账
    source_type = "feishu_docx"
    return IntermediateDoc(
        meta=SourceMeta(
            source_type=source_type,
            doc_id=document_id,
            title=title or document_id,
        ),
        blocks=ir_blocks,
    )


def content_sha16(text: str) -> str:
    """对照文件通道的 doc_id 口径：正文内容 hash 前 16 位。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(
            f"缺少环境变量 {name}。\n"
            "请设置 FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_DOC_ID 后重试；"
            "或使用 `map --input <raw.json>` 离线演示。"
        )
    return v


def _client() -> FeishuClient:
    return FeishuClient(_env("FEISHU_APP_ID"), _env("FEISHU_APP_SECRET"))


def _resolve_doc_id(client: FeishuClient) -> str:
    wiki = os.environ.get("FEISHU_WIKI_TOKEN", "").strip()
    if wiki:
        return client.resolve_document_id_from_wiki(wiki)
    return _env("FEISHU_DOC_ID")


def cmd_raw(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with _client() as client:
        doc_id = _resolve_doc_id(client)
        meta = client.get_document_meta(doc_id)
        blocks = client.list_blocks(doc_id)
    payload = {"document": meta, "items": blocks}
    out = Path(args.out) if args.out else OUT_DIR / f"{doc_id}_raw_blocks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存原始 Block JSON: {out}")
    print(f"  document_id={doc_id}  title={meta.get('title')!r}  blocks={len(blocks)}")


def cmd_ir(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with _client() as client:
        doc_id = _resolve_doc_id(client)
        meta = client.get_document_meta(doc_id)
        blocks = client.list_blocks(doc_id)

    ir = blocks_to_ir(
        blocks,
        document_id=doc_id,
        title=meta.get("title"),
        revision_id=meta.get("revision_id"),
    )
    # 与文件通道对账：记录内容 hash，便于对比「同一文档两种接入」的 doc_id
    body = ir.to_text()
    digest = content_sha16(body)

    out_ir = Path(args.out) if args.out else OUT_DIR / f"{doc_id}_ir.json"
    out_raw = OUT_DIR / f"{doc_id}_raw_blocks.json"
    out_ir.parent.mkdir(parents=True, exist_ok=True)
    out_ir.write_text(ir.model_dump_json(indent=2), encoding="utf-8")
    out_raw.write_text(
        json.dumps({"document": meta, "items": blocks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"已保存 IR:      {out_ir}")
    print(f"已保存原始 JSON: {out_raw}")
    print(f"  feishu_document_id = {doc_id}")
    print(f"  content_sha256[:16] = {digest}")
    print(f"  title = {ir.meta.title}")
    print(f"  blocks = {len(ir.blocks)}")
    print("  前 5 块预览:")
    for b in ir.blocks[:5]:
        snippet = b.text.replace("\n", " ")[:60]
        print(f"    - [{b.type}] {snippet}")


def cmd_pdf(args: argparse.Namespace) -> None:
    """路径 B：按现状语料形态导出 PDF，落到目录后仍走现有 ingest。"""
    ext = args.ext.lstrip(".")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with _client() as client:
        doc_id = _resolve_doc_id(client)
        meta = client.get_document_meta(doc_id)
        title = (meta.get("title") or doc_id).replace("/", "_").replace("\\", "_")
        print(f"创建导出任务: token={doc_id} type=docx ext={ext}")
        ticket = client.create_export_task(
            file_token=doc_id, obj_type="docx", file_extension=ext
        )
        print(f"  ticket={ticket}，轮询中…")
        result = client.wait_export_task(ticket, timeout_s=args.timeout)
        file_token = result.get("file_token")
        if not file_token:
            raise RuntimeError(f"导出完成但无 file_token: {result}")
        dest = out_dir / f"{title}_{doc_id}.{ext}"
        client.download_export(file_token, dest)
        print(f"已下载: {dest}")
        print(f"  文件大小: {dest.stat().st_size} bytes")
        print("  下一步（走现有管线）:")
        print(f"    uv run doc-rag ingest --raw-dir {out_dir}  # 或拷入 data/raw")


def cmd_map(args: argparse.Namespace) -> None:
    """离线：把已保存的 raw Block JSON 映射为 IR（无需飞书凭证）。"""
    src = Path(args.input)
    if not src.is_file():
        raise SystemExit(f"文件不存在: {src}")
    payload = json.loads(src.read_text(encoding="utf-8"))
    if "items" in payload:
        items = payload["items"]
        document = payload.get("document") or {}
        document_id = document.get("document_id") or src.stem.replace("_raw_blocks", "")
        title = document.get("title")
    elif isinstance(payload, list):
        items = payload
        document_id = src.stem
        title = None
    else:
        raise SystemExit("JSON 结构无法识别：需要 {document, items} 或 items 数组")

    ir = blocks_to_ir(items, document_id=document_id, title=title)
    body = ir.to_text()
    out = Path(args.out) if args.out else src.with_name(f"{document_id}_ir.json")
    out.write_text(ir.model_dump_json(indent=2), encoding="utf-8")
    print(f"已映射 IR: {out}")
    print(f"  doc_id={ir.meta.doc_id}  content_sha256[:16]={content_sha16(body)}")
    print(f"  blocks={len(ir.blocks)}")
    for b in ir.blocks[:8]:
        print(f"    - [{b.type}] {b.text[:50]}")


def cmd_demo_ir(_: argparse.Namespace) -> None:
    """无凭证时用最小假数据演示映射逻辑（结构对齐官方 blocks 响应）。"""
    fake = {
        "document": {
            "document_id": "doxdemo000",
            "title": "演示会议纪要",
            "revision_id": 3,
        },
        "items": [
            {
                "block_id": "page",
                "block_type": 1,
                "parent_id": "",
                "children": ["h1", "meta", "h2", "p1", "h3", "t1", "p2"],
                "page": {
                    "elements": [{"text_run": {"content": "演示会议纪要"}}],
                    "style": {},
                },
            },
            {
                "block_id": "h1",
                "block_type": 3,
                "parent_id": "page",
                "heading1": {"elements": [{"text_run": {"content": "演示会议纪要"}}]},
            },
            {
                "block_id": "meta",
                "block_type": 2,
                "parent_id": "page",
                "text": {
                    "elements": [
                        {
                            "text_run": {
                                "content": "会议时间：2026-02-09 14:00-15:30　主持人：周航"
                            }
                        }
                    ]
                },
            },
            {
                "block_id": "h2",
                "block_type": 3,
                "parent_id": "page",
                "heading1": {
                    "elements": [{"text_run": {"content": "一、客服系统升级"}}]
                },
            },
            {
                "block_id": "p1",
                "block_type": 2,
                "parent_id": "page",
                "text": {
                    "elements": [
                        {
                            "text_run": {
                                "content": "客户成功部提交二期升级方案，预算 42 万元。"
                            }
                        }
                    ]
                },
            },
            {
                "block_id": "h3",
                "block_type": 3,
                "parent_id": "page",
                "heading1": {"elements": [{"text_run": {"content": "二、预算明细"}}]},
            },
            {
                "block_id": "t1",
                "block_type": 31,
                "parent_id": "page",
                "children": ["c00", "c01", "c10", "c11"],
                "table": {"property": {"row_size": 2, "column_size": 2}},
            },
            {
                "block_id": "c00",
                "block_type": 32,
                "parent_id": "t1",
                "children": ["c00t"],
            },
            {
                "block_id": "c00t",
                "block_type": 2,
                "parent_id": "c00",
                "text": {"elements": [{"text_run": {"content": "项目"}}]},
            },
            {
                "block_id": "c01",
                "block_type": 32,
                "parent_id": "t1",
                "children": ["c01t"],
            },
            {
                "block_id": "c01t",
                "block_type": 2,
                "parent_id": "c01",
                "text": {"elements": [{"text_run": {"content": "金额"}}]},
            },
            {
                "block_id": "c10",
                "block_type": 32,
                "parent_id": "t1",
                "children": ["c10t"],
            },
            {
                "block_id": "c10t",
                "block_type": 2,
                "parent_id": "c10",
                "text": {"elements": [{"text_run": {"content": "软件采购"}}]},
            },
            {
                "block_id": "c11",
                "block_type": 32,
                "parent_id": "t1",
                "children": ["c11t"],
            },
            {
                "block_id": "c11t",
                "block_type": 2,
                "parent_id": "c11",
                "text": {"elements": [{"text_run": {"content": "28 万元"}}]},
            },
            {
                "block_id": "p2",
                "block_type": 2,
                "parent_id": "page",
                "text": {
                    "elements": [
                        {"text_run": {"content": "决议区：① 预算 42 万元予以立项。"}}
                    ]
                },
            },
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "demo_raw_blocks.json"
    raw_path.write_text(
        json.dumps(fake, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写入示例 raw: {raw_path}")
    ns = argparse.Namespace(input=str(raw_path), out=str(OUT_DIR / "demo_ir.json"))
    cmd_map(ns)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="飞书 API 导出文档 demo（Blocks→IR / 导出 PDF）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("raw", help="拉取文档 Block 原始 JSON")
    pr.add_argument("--out", help="输出路径，默认 data/feishu/{doc_id}_raw_blocks.json")
    pr.set_defaults(func=cmd_raw)

    pi = sub.add_parser("ir", help="拉取 Block 并映射为项目 IntermediateDoc")
    pi.add_argument("--out", help="输出路径，默认 data/feishu/{doc_id}_ir.json")
    pi.set_defaults(func=cmd_ir)

    pp = sub.add_parser("pdf", help="导出任务下载 PDF/docx（对齐现有 raw 语料）")
    pp.add_argument("--out", default="data/raw", help="下载目录，默认 data/raw")
    pp.add_argument("--ext", default="pdf", help="导出扩展名 pdf|docx，默认 pdf")
    pp.add_argument("--timeout", type=float, default=60.0, help="导出轮询超时秒")
    pp.set_defaults(func=cmd_pdf)

    pm = sub.add_parser("map", help="离线：raw JSON → IR（无需凭证）")
    pm.add_argument("--input", required=True, help="已保存的 raw blocks JSON")
    pm.add_argument("--out", help="IR 输出路径")
    pm.set_defaults(func=cmd_map)

    pd = sub.add_parser("demo", help="无凭证演示：内置假数据跑通映射")
    pd.set_defaults(func=cmd_demo_ir)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

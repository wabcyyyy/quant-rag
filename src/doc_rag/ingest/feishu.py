"""飞书通道（PLAN §2 接入行）：docx blocks OpenAPI → IntermediateDoc。

状态：可选优化桩。采集当前走批量导出 PDF（已解决）；仅当触发线命中才实装——
语料过几千且更新频繁 / 导出 PDF 的表格或标题解析质量不达标（PLAN §1 面试表）。

设计要点（实装时对照）：
- tenant_access_token: POST /open-apis/auth/v3/tenant_access_token/internal
- 文档 blocks:   GET /open-apis/docx/v1/documents/{document_id}/blocks
  （原生结构：heading1-9 / bullet / ordered / quote / table / table_cell → Block.type）
- 文档元数据:    GET /open-apis/docx/v1/documents/{document_id}
  （title / owner / 创建时间，与 LLM 抽取交叉校验，见 PLAN §5.1）
- 增量同步:      drive 文件列表按 edited_time 过滤
"""

from __future__ import annotations

from .schema import IntermediateDoc

FEISHU_BASE = "https://open.feishu.cn"


class FeishuClient:
    """可选桩：触发线命中后再实现 fetch_document 与增量同步。"""

    def __init__(self, app_id: str, app_secret: str) -> None:
        self.app_id = app_id
        self.app_secret = app_secret

    def fetch_document(self, document_id: str) -> IntermediateDoc:
        raise NotImplementedError("飞书 OpenAPI 为条件触发的可选路线（PLAN §1 面试表）")

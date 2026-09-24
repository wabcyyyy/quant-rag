"""本地 OCR 兜底（可选依赖，A3.3）：近空 / 扫描件从「只标记」到「真兜底」。

- 引擎：rapidocr-onnxruntime（纯 pip、无系统依赖），装在 `[project.optional-dependencies].ocr`，
  **不进核心依赖**——born-digital 语料用不到它，不该为 4% 的扫描件拖累每个环境。
- 触发条件在 pdf.extract_pdf：可抽文本 < 每页阈值 **且** 页面有图（「有图无字」的
  scan_likely 形态）才走 OCR；纯无图少字的 near_empty（损坏文档）没有可识别对象。
- 失败不阻塞入库（照 metadata.extract_metadata 先例）：引擎缺席或识别抛错时返回
  None，由调用方计数并在 ingest 汇总里明确报告「N 篇疑似扫描件未走兜底」，
  **不许静默**——静默 = README 里那条「扫描件这条链路只有标记」的旧病复发。
- 关闭方式 = 不装 extra（`uv sync --extra ocr` 加回）。没有配置开关：
  本地 CPU、零 API 成本、只对原本 0 块的文档生效，没有值得为它撒谎的配置面。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rapidocr_onnxruntime import RapidOCR

# 引擎实例 / False=不可用 / None=未探测（惰性探测，避免为 4% 的扫描件拖累每个环境）
_ENGINE: RapidOCR | bool | None = None


def available() -> bool:
    """引擎是否可导入（惰性探测一次并缓存结论）。"""
    global _ENGINE
    if _ENGINE is False:
        return False
    if _ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # noqa: F401

            _ENGINE = "rapidocr"
        except Exception:  # noqa: BLE001 缺依赖 / 装坏都算不可用
            _ENGINE = False
            return False
    return True


def ocr_pdf_pages(path: Path) -> list | None:
    """整本 PDF 逐页 OCR，返回候选块描述符列表；引擎不可用/失败返回 None。

    每个描述符 = (page_no, text)。调用方（pdf.extract_pdf）负责组装成 Block。
    页面位图 2x 渲染（与生成扫描件的分辨率匹配，实测中文识别可靠）。
    """
    if not available():
        return None
    try:
        import numpy as np
        import pymupdf
        from rapidocr_onnxruntime import RapidOCR
    except Exception:  # noqa: BLE001 同 available：缺依赖不算文档的错
        return None
    global _ENGINE
    doc = None
    try:
        engine = _ENGINE if isinstance(_ENGINE, RapidOCR) else RapidOCR()
        _ENGINE = engine  # 初始化 ~1s，进程内复用
        doc = pymupdf.open(path)
        out: list[tuple[int, str]] = []
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            result, _elapse = engine(img)
            lines = [
                (box, str(text))
                for box, text, _conf in (result or [])
                if str(text).strip()
            ]
            if not lines:
                continue
            # 按纵坐标排好后逐行拼接（OCR 返回顺序已接近阅读序，排序兜底）
            lines.sort(key=lambda bl: min(p[1] for p in bl[0]))
            out.append((page_index + 1, "\n".join(t for _, t in lines)))
        return out
    except Exception:  # noqa: BLE001 单本失败不阻塞入库，交给调用方计数
        return None
    finally:
        if doc is not None:
            doc.close()

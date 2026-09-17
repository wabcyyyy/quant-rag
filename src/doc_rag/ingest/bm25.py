"""jieba 预分词 → 空格拼接，供 Qdrant full-text（word tokenizer）建 BM25 索引。

中文原文直接进 word tokenizer 会整句成词，必须预分词（PLAN §5.2）。
"""

from __future__ import annotations

import jieba


def build_bm25_text(text: str) -> str:
    return " ".join(token.strip() for token in jieba.cut(text) if token.strip())

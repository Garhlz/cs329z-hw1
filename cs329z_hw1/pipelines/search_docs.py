"""第九题：分块检索和原文按行读取，内部类型不改变课程字典接口。"""

import re
from dataclasses import dataclass

from cs329z_hw1.pipelines.bm25 import DocIndex, build_index, search_index
from cs329z_hw1.pipelines.doc_chunks import DocumentChunk, chunk_documents
from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.tokens import count_tokens
from cs329z_hw1.types import SearchResult, DocWindow


@dataclass
class CompanyDocIndex:
    bm25: DocIndex
    chunks: list[DocumentChunk]
    docs_by_id: dict[str, dict]


def build_doc_index(docs: list[dict]) -> CompanyDocIndex:
    chunks = chunk_documents(docs)
    return CompanyDocIndex(
        bm25=build_index([chunk.index_text for chunk in chunks]),
        chunks=chunks,
        docs_by_id={doc["doc_id"]: dict(doc) for doc in docs},
    )


def _snippet(text: str, query: str) -> str:
    """截取包含匹配词的连续原文，保留上下文，最多 150 个课程 token。"""
    if count_tokens(text) <= 150:
        return text
    terms = set(tokenize(query))
    matches = [m for m in re.finditer(r"[a-z0-9]+", text.lower()) if m.group() in terms]
    position = matches[0].start() if matches else 0
    start = max(0, position - 160)
    # 尽量从词边界开始，保留连续原文；600 字符对应 150 token。
    if start:
        while start < position and not text[start - 1].isspace():
            start += 1
    return text[start:start + 600]


def search_docs(index: CompanyDocIndex, query: str, k: int = 5) -> list[SearchResult]:
    results = []
    for chunk_id, score in search_index(index.bm25, query, k):
        chunk = index.chunks[chunk_id]
        # 标题可能使正文不含查询词的块命中，此时从原文标题附近提供片段。
        source = chunk.text
        terms = set(tokenize(query))
        if terms and not terms.intersection(tokenize(source)):
            source = index.docs_by_id[chunk.doc_id]["text"]
        results.append({
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "snippet": _snippet(source, query),
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
        })
    return results


def read_doc(index: CompanyDocIndex, doc_id: str, start_line: int = 1) -> DocWindow:
    """按完整行返回连续窗口；通常约 600 token，硬上限 1500 token。"""
    lines = index.docs_by_id[doc_id]["text"].splitlines()
    if not 1 <= start_line <= len(lines):
        raise ValueError("start_line is outside the document.")
    selected = []
    for line in lines[start_line - 1:]:
        candidate = "\n".join([*selected, line])
        size = count_tokens(candidate)
        if selected and size > 600:
            break
        if size > 1500:
            raise ValueError("A single document line exceeds the 1500-token limit.")
        selected.append(line)
    return {
        "doc_id": doc_id,
        "start_line": start_line,
        "end_line": start_line + len(selected) - 1,
        "total_lines": len(lines),
        "text": "\n".join(selected),
    }

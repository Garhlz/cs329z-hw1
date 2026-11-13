"""按 Markdown 小节和完整行分块，保留原文定位信息。"""

import re
from dataclasses import dataclass

from cs329z_hw1.tokens import count_tokens


@dataclass(frozen=True)
class DocumentChunk:
    doc_id: str
    title: str
    section_title: str
    start_line: int
    end_line: int
    text: str

    @property
    def index_text(self) -> str:
        """给 BM25 添加标题上下文，不改写原文及其行号。"""
        context = [self.title]
        if self.section_title and self.section_title != self.title:
            context.append(self.section_title)
        return "\n".join([*context, self.text])


def chunk_documents(docs: list[dict], max_tokens: int = 300) -> list[DocumentChunk]:
    """按二级标题分段，超长小节按完整行拆分，不添加重叠。

    max_tokens 限制原文块，不含 index_text 追加的标题上下文。
    单行本身超限时保留整行，因此这是软上限，不破坏原文定位。
    空白文档不生成块；每个块的行号从 1 开始，首尾行均包含在内。
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")

    chunks = []
    for doc in docs:
        lines = doc["text"].splitlines()
        section_title = ""
        chunk_start = 0
        pending: list[str] = []

        def flush() -> None:
            # 空白行可随正文保留，但不单独作为检索结果。把当前处理的块写入chunks
            if pending and any(line.strip() for line in pending):
                chunks.append(
                    DocumentChunk(
                        doc_id=doc["doc_id"],
                        title=doc["title"],
                        section_title=section_title,
                        start_line=chunk_start + 1,
                        end_line=chunk_start + len(pending),
                        text="\n".join(pending),
                    )
                )

        # 围栏代码块中的 ## 是代码内容，不能用作小节边界。
        fence_char = None
        fence_length = 0
        for position, line in enumerate(lines):
            fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
            heading = (
                None if fence_char else re.match(r"^ {0,3}##[ \t]+(.+?)\s*$", line)
            )
            # 遇到新的小节
            if heading:
                flush()  # 清空旧的缓冲区
                pending = []
                chunk_start = position
                section_title = re.sub(r"[ \t]+#+[ \t]*$", "", heading.group(1))

            # 换行符也计入长度；只在已有非空内容时拆分。
            if pending and any(item.strip() for item in pending):
                # 已有内容 + 换行 + 当前行超过了单个chunk的token上限，flush
                if count_tokens("\n".join([*pending, line])) > max_tokens:
                    flush()
                    pending = []
                    chunk_start = position
            if not pending:
                chunk_start = position
            pending.append(line)

            # 不误判代码围栏中的标题符号
            if fence:
                marker, trailing = fence.groups()
                if fence_char is None:
                    fence_char, fence_length = marker[0], len(marker)
                elif (
                    marker[0] == fence_char
                    and len(marker) >= fence_length
                    and not trailing.strip()
                ):
                    fence_char = None

        flush()

    return chunks

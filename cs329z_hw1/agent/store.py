from cs329z_hw1.types import (
    Email,
)
from cs329z_hw1.pipelines.bm25 import DocIndex

from cs329z_hw1.data import (
    load_emails,
    email_text,
    load_docs,
)
from cs329z_hw1.adapters import (
    run_bm25_build,
    run_bm25_search,
)



class EmailStore:
    def __init__(self):
        # 初始化的时候只创建了对象，但是没有加载索引
        self.emails: list[Email] | None = None  # 固定顺序的emails list
        self.by_id: dict[str, Email] | None = None
        self.index: DocIndex | None = None

    def ensure_data_loaded(self):
        if self.emails is None:
            emails = load_emails()
            self.by_id = {email["id"]: email for email in emails}
            self.emails = emails

    def ensure_index_loaded(self):
        """仅搜索需要 BM25；读取邮件和按日期筛选无需建立索引。"""
        self.ensure_data_loaded()
        if self.index is None:
            docs = [email_text(email) for email in self.emails]
            self.index = run_bm25_build(docs)

    def search(self, query: str, k: int = 5) -> list[Email]:
        # 懒加载邮件文档索引
        self.ensure_index_loaded()

        results = run_bm25_search(self.index, query, k)

        return [self.emails[index] for index, _ in results]


class DocChunkStore:
    """查询和读取共享完整的文档索引，首次使用时才加载。"""
    def __init__(self):
        self.index = None

    def ensure_index_loaded(self):
        if self.index is None:
            from cs329z_hw1.pipelines.search_docs import build_doc_index
            self.index = build_doc_index(load_docs())
        return self.index

    def search(self, query: str, k: int = 5):
        from cs329z_hw1.pipelines.search_docs import search_docs
        return search_docs(self.ensure_index_loaded(), query, k)

    def read(self, doc_id: str, start_line: int = 1):
        from cs329z_hw1.pipelines.search_docs import read_doc
        return read_doc(self.ensure_index_loaded(), doc_id, start_line)

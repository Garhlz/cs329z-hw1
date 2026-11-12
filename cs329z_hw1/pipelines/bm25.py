"""第三题：BM25 倒排索引构建和检索。"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from cs329z_hw1.tokenizer import tokenize


def build_index(docs: list[str]) -> Any:
    """构建 BM25 倒排索引，文档编号采用输入列表下标。

使用课程 tokenize 分词。空文档仍计入文档总数和平均长度。
返回的索引交给 search_index 使用；完整邮件档案建索引应少于 15 秒。"""

    # 每个词对应在每篇doc中出现的频率

    index: dict[str, dict[int, int]] = {}
    doc_lengths = []  # 统计每篇文章的token数量

    for doc_id, doc in enumerate(docs):
        # 一篇文章只用统计一次
        words = tokenize(doc)
        doc_lengths.append(len(words))
        frequencies = Counter(words)

        for word, frequency in frequencies.items():
            # 如果 word 不存在，就创建一个空字典，然后返回对应的字典
            index.setdefault(word, {})[doc_id] = frequency

    num_docs = len(docs)
    avg_doc_length = 0.0 if num_docs == 0 else sum(doc_lengths) / num_docs

    # 用一个字典一次性返回
    return {
        "postings": index,
        "doc_lengths": doc_lengths,
        "avg_doc_length": avg_doc_length,
        "num_docs": num_docs,
    }



def search_index(index: Any, query: str, k: int) -> list[tuple[int, float]]:
    """返回前 k 个 (文档编号, 分数)，按分数降序、编号升序排列。

使用 k1=1.5、b=0.75 和 Lucene IDF；重复查询词只计算一次。
仅访问查询词的倒排表，不返回没有匹配词的文档。
查询无有效词或无命中时返回空列表。
档案平均查询应少于 20 毫秒，稀有词查询应少于 2 毫秒。"""
    if k <= 0:
        return []
    postings = index["postings"]
    doc_lengths = index["doc_lengths"]
    avgdl = index["avg_doc_length"]
    N = index["num_docs"]

    if N == 0:
        return []

    k1 = 1.5
    b = 0.75

    # doc_id -> 累计 BM25 分数
    # defaultdict(float) 会把不存在的值初始化为 0.0
    scores: dict[int, float] = defaultdict(float)

    # set 去除重复查询词
    for term in set(tokenize(query)):
        # 没有出现过的词直接跳过
        term_postings = postings.get(term)
        if not term_postings:
            continue

        # df: 包含该词的文档数量
        df = len(term_postings)

        # Lucene IDF
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))

        # 只遍历包含该词的文档
        for doc_id, tf in term_postings.items():
            dl = doc_lengths[doc_id]

            # BM25 词频及文档长度归一化
            denominator = tf + k1 * (1 - b + b * dl / avgdl)

            score = idf * (tf * (k1 + 1)) / denominator

            scores[doc_id] += score

    # 分数降序，相同分数按 doc_id 升序
    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return ranked[:k]


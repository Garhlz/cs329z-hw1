"""第三题：BM25 倒排索引构建和检索。"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from cs329z_hw1.tokenizer import tokenize


def build_index(docs: list[str]) -> Any:
    """Problem (bm25).

    Build and return a BM25 index over ``docs``. Document ids are list
    positions. Tokenize with ``cs329z_hw1.tokenizer.tokenize``. A document
    with no tokens is still a document (it counts in N and in the average
    length). The returned object is opaque to the tests; they only pass it
    back to ``run_bm25_search``. The index over the archive must build in
    under 15 seconds.
    """

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
    """Problem (bm25).

    Return the top-k ``(doc_id, score)`` pairs for ``query`` as
    ``(int, float)`` tuples, best first, ties broken by lower doc_id (the
    document's position in the ``docs`` list given to ``run_bm25_build``).

    Rules the tests check:
    * Scores follow the handout's formula (k1 = 1.5, b = 0.75, Lucene IDF)
      to within 1e-6. The query is tokenized like the documents, and a term
      repeated in the query counts once.
    * A document that shares no term with the query is never returned, so
      the list may be shorter than k. A query with no tokens, or whose terms
      occur in no document, returns [].
    * A search reads only the index entries of the query's terms: 20 ms per
      query on average over the archive, and 2 ms for a term that occurs in
      3 of 200,000 documents.
    """
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


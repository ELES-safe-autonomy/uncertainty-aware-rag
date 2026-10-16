"""Re-ranking with a cross-encoder.

Search returns a wide pool of candidates quickly; the cross-encoder then reads
the question and each candidate TOGETHER and scores how well they match.
The best K candidates by that score become the final results.
"""

import numpy as np
from sentence_transformers import CrossEncoder

from uncertainty_aware_rag.expansion import related_queries

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder = None


def get_cross_encoder():
    """Load the model once, on first use."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(MODEL)
    return _cross_encoder


def rerank(question, ids, documents, metadatas, k=5):
    """Score every (question, chunk) pair and keep the K best, best first."""
    scores = get_cross_encoder().predict([[question, doc] for doc in documents])
    best = np.argsort(scores)[::-1][:k]
    return [ids[i] for i in best], [metadatas[i] for i in best]


def rerank_search(collection, question, pool=10, k=5):
    """Lab 4, part 1: take the top `pool` results for the question, re-rank, keep K."""
    r = collection.query(query_texts=[question], n_results=pool)
    return rerank(question, r["ids"][0], r["documents"][0], r["metadatas"][0], k)


def multi_rerank_search(collection, question, pool=10, k=5):
    """Lab 4, part 2: pool the results of the question + its rewrites,
    remove duplicates, then re-rank the pool against the ORIGINAL question."""
    queries = [question] + related_queries(question)
    r = collection.query(query_texts=queries, n_results=pool)
    unique = {}  # chunk id -> (text, metadata); a dict removes duplicates
    for ids, docs, metas in zip(r["ids"], r["documents"], r["metadatas"]):
        for chunk_id, doc, meta in zip(ids, docs, metas):
            unique[chunk_id] = (doc, meta)
    ids = list(unique)
    documents = [unique[i][0] for i in ids]
    metadatas = [unique[i][1] for i in ids]
    return rerank(question, ids, documents, metadatas, k)

"""Hybrid Search Module for Copperleaf Kitchens RAG pipeline.

Implements Reciprocal Rank Fusion (RRF) of:
- Dense vector similarity results from ChromaDB
- Sparse BM25 keyword matching results using rank_bm25

This fused approach improves recall for both:
- Semantic queries (where vector search excels)
- Exact term/keyword queries (where BM25 excels)
"""

import math
from typing import Any, Dict, List, Optional


try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


def tokenize(text: str) -> List[str]:
    """Simple whitespace + lower-case tokenizer for BM25."""
    return text.lower().split()


def hybrid_search(
    query: str,
    candidate_chunks: List[Dict[str, Any]],
    top_k: int = 5,
    rrf_k: int = 60,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
) -> List[Dict[str, Any]]:
    """Perform hybrid dense + sparse retrieval using Reciprocal Rank Fusion.

    Takes pre-retrieved vector candidates (with distances) and re-ranks them
    using BM25 scores, then fuses both ranked lists with RRF.

    Args:
        query: User query string.
        candidate_chunks: List of chunk dicts from vector retriever with
                          keys: text, metadata, distance.
        top_k: Number of top results to return after fusion.
        rrf_k: RRF constant (higher = smoother fusion, default 60).
        bm25_weight: Weight for BM25 rank component (0.0 to 1.0).
        vector_weight: Weight for vector rank component (0.0 to 1.0).

    Returns:
        Re-ranked list of chunk dicts with an additional 'rrf_score' key.
    """
    if not candidate_chunks:
        return []

    # ---- 1. VECTOR RANK ----
    # Sort by distance ascending (lower = more similar)
    vector_ranked = sorted(candidate_chunks, key=lambda c: c["distance"])

    # ---- 2. BM25 RANK ----
    if _BM25_AVAILABLE:
        corpus = [tokenize(chunk["text"]) for chunk in candidate_chunks]
        bm25 = BM25Okapi(corpus)
        query_tokens = tokenize(query)
        bm25_scores = bm25.get_scores(query_tokens)

        # Sort by BM25 score descending
        bm25_ranked_indices = sorted(
            range(len(candidate_chunks)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )
        bm25_index_to_rank = {idx: rank for rank, idx in enumerate(bm25_ranked_indices)}
    else:
        # Fallback: no BM25, use only vector ranks
        bm25_index_to_rank = {i: i for i in range(len(candidate_chunks))}

    # ---- 3. RRF FUSION ----
    # Build index map for vector ranks
    vector_index_to_rank = {
        candidate_chunks.index(chunk): rank
        for rank, chunk in enumerate(vector_ranked)
    }

    rrf_scores: List[float] = []
    for i in range(len(candidate_chunks)):
        v_rank = vector_index_to_rank.get(i, len(candidate_chunks))
        b_rank = bm25_index_to_rank.get(i, len(candidate_chunks))

        rrf_v = vector_weight / (rrf_k + v_rank + 1)
        rrf_b = bm25_weight / (rrf_k + b_rank + 1)
        rrf_scores.append(rrf_v + rrf_b)

    # ---- 4. SORT BY FUSED SCORE and return top_k ----
    indexed = list(enumerate(candidate_chunks))
    indexed.sort(key=lambda x: rrf_scores[x[0]], reverse=True)

    results = []
    for idx, chunk in indexed[:top_k]:
        enriched = dict(chunk)
        enriched["rrf_score"] = round(rrf_scores[idx], 6)
        enriched["bm25_available"] = _BM25_AVAILABLE
        results.append(enriched)

    return results


def bm25_only_search(
    query: str,
    candidate_chunks: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Sparse BM25-only search for comparison benchmarking.

    Args:
        query: User query string.
        candidate_chunks: List of chunk dicts with key: text.
        top_k: Number of results to return.

    Returns:
        Ranked list of chunks by BM25 score with 'bm25_score' key appended.
    """
    if not _BM25_AVAILABLE:
        raise ImportError("rank_bm25 is required. Install with: pip install rank-bm25")

    if not candidate_chunks:
        return []

    corpus = [tokenize(chunk["text"]) for chunk in candidate_chunks]
    bm25 = BM25Okapi(corpus)
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    indexed = sorted(enumerate(candidate_chunks), key=lambda x: scores[x[0]], reverse=True)

    results = []
    for idx, chunk in indexed[:top_k]:
        enriched = dict(chunk)
        enriched["bm25_score"] = round(float(scores[idx]), 4)
        results.append(enriched)

    return results

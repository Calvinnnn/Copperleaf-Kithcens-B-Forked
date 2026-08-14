"""RAG Evaluation Module for Copperleaf Kitchens.

Provides a domain-specific retrieval evaluation dataset and a comparison
runner that benchmarks:
 - Vector-only retrieval
 - BM25-only retrieval
 - Hybrid RRF retrieval

Metrics computed for each retrieval method:
 - Precision@K: fraction of top-K retrieved chunks that are marked relevant
 - Recall@K: fraction of ground-truth relevant chunks retrieved in top-K
 - MRR (Mean Reciprocal Rank): position quality of first relevant result
 - Average Retrieval Latency (ms)
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ============================================================
# DOMAIN-SPECIFIC EVALUATION DATASET
# ============================================================
# Each entry:
#   query: the test query string
#   relevant_keywords: substrings that must appear in a chunk for it to be
#                      considered ground-truth relevant (heuristic oracle)
#   description: human readable scenario label
# ============================================================

EVAL_DATASET: List[Dict[str, Any]] = [
    {
        "query_id": "rag_q1",
        "description": "Inventory spoilage policy lookup",
        "query": "What is the restaurant policy for writing off spoiled produce?",
        "relevant_keywords": ["write-off", "write off", "spoiled", "expired", "policy", "inventory"],
    },
    {
        "query_id": "rag_q2",
        "description": "Supplier emergency contact lookup",
        "query": "Who is the emergency contact for fresh produce suppliers?",
        "relevant_keywords": ["supplier", "contact", "produce", "emergency", "phone", "email"],
    },
    {
        "query_id": "rag_q3",
        "description": "Low-stock reorder threshold",
        "query": "What happens when inventory falls below the reorder threshold?",
        "relevant_keywords": ["reorder", "threshold", "low stock", "restock", "order"],
    },
    {
        "query_id": "rag_q4",
        "description": "Branch manager responsibilities",
        "query": "What are the daily responsibilities of a branch manager?",
        "relevant_keywords": ["manager", "branch", "responsibility", "report", "approval"],
    },
    {
        "query_id": "rag_q5",
        "description": "Food safety compliance checks",
        "query": "What are the food safety compliance requirements for kitchen staff?",
        "relevant_keywords": ["food safety", "compliance", "hygiene", "temperature", "storage"],
    },
]


# ============================================================
# EVALUATION RESULT TYPES
# ============================================================

@dataclass
class RetrievalMethodResult:
    """Results for one retrieval method on one query."""

    method: str
    query_id: str
    query: str
    retrieved_count: int
    relevant_in_top_k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    latency_ms: float


@dataclass
class RAGEvalSuiteResult:
    """Full evaluation suite results across all methods and queries."""

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    top_k: int = 5
    results: List[RetrievalMethodResult] = field(default_factory=list)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Aggregate average metrics per retrieval method."""
        method_stats: Dict[str, List[RetrievalMethodResult]] = {}
        for r in self.results:
            method_stats.setdefault(r.method, []).append(r)

        summary: Dict[str, Dict[str, float]] = {}
        for method, records in method_stats.items():
            n = len(records)
            summary[method] = {
                "avg_precision_at_k": round(sum(r.precision_at_k for r in records) / n, 4),
                "avg_recall_at_k": round(sum(r.recall_at_k for r in records) / n, 4),
                "avg_mrr": round(sum(r.mrr for r in records) / n, 4),
                "avg_latency_ms": round(sum(r.latency_ms for r in records) / n, 3),
                "total_queries": n,
            }
        return summary

    def generate_markdown_report(self) -> str:
        """Generate a Markdown table comparing all retrieval methods."""
        lines = [
            "# RAG Retrieval Method Comparison Report\n",
            f"**Timestamp**: {self.timestamp}",
            f"**Top-K**: {self.top_k}\n",
            "## Per-Query Results\n",
            "| Method | Query ID | P@K | Recall@K | MRR | Latency(ms) |",
            "|--------|----------|-----|----------|-----|-------------|",
        ]
        for r in self.results:
            lines.append(
                f"| {r.method} | {r.query_id} | {r.precision_at_k:.3f} | "
                f"{r.recall_at_k:.3f} | {r.mrr:.3f} | {r.latency_ms:.2f} |"
            )

        summary = self.summary()
        lines.append("\n## Summary by Method\n")
        lines.append("| Method | Avg P@K | Avg Recall@K | Avg MRR | Avg Latency(ms) |")
        lines.append("|--------|---------|-------------|---------|-----------------|")
        for method, stats in summary.items():
            lines.append(
                f"| **{method}** | {stats['avg_precision_at_k']:.3f} | "
                f"{stats['avg_recall_at_k']:.3f} | {stats['avg_mrr']:.3f} | "
                f"{stats['avg_latency_ms']:.2f} |"
            )

        return "\n".join(lines)


# ============================================================
# ORACLE RELEVANCE CHECK (heuristic - no LLM needed)
# ============================================================

def _is_relevant_chunk(chunk: Dict[str, Any], relevant_keywords: List[str]) -> bool:
    """Determine if a chunk is ground-truth relevant based on keyword oracle."""
    text = chunk.get("text", "").lower()
    return any(kw.lower() in text for kw in relevant_keywords)


def _compute_mrr(chunks: List[Dict[str, Any]], relevant_keywords: List[str]) -> float:
    """Compute Mean Reciprocal Rank for a ranked result list."""
    for rank, chunk in enumerate(chunks, start=1):
        if _is_relevant_chunk(chunk, relevant_keywords):
            return 1.0 / rank
    return 0.0


# ============================================================
# RETRIEVAL COMPARISON RUNNER
# ============================================================

class RAGEvaluationRunner:
    """Benchmarks vector-only, BM25-only, and hybrid retrieval methods.

    Can run in 'mock mode' (no real ChromaDB needed) by accepting a
    pre-built candidate pool, allowing unit tests to validate the
    evaluation framework without a populated vector store.
    """

    def __init__(self, top_k: int = 5) -> None:
        self._top_k = top_k

    def _run_vector_only(
        self,
        query: str,
        candidate_pool: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[Dict[str, Any]], float]:
        """Run vector-only retrieval (real or mock)."""
        start = time.perf_counter()
        if candidate_pool is not None:
            # Mock mode: sort by distance (ascending = more similar)
            results = sorted(candidate_pool, key=lambda c: c.get("distance", 1.0))[: self._top_k]
        else:
            try:
                from rag.retriever import retrieve
                results = retrieve(query, top_k=self._top_k)
            except Exception:
                results = []
        latency = (time.perf_counter() - start) * 1000
        return results, latency

    def _run_bm25_only(
        self,
        query: str,
        candidate_pool: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[Dict[str, Any]], float]:
        """Run BM25-only retrieval (real or mock). Returns empty if rank_bm25 unavailable."""
        start = time.perf_counter()
        results: List[Dict[str, Any]] = []
        try:
            if candidate_pool is not None:
                from rag.hybrid_search import bm25_only_search, _BM25_AVAILABLE
                if _BM25_AVAILABLE:
                    results = bm25_only_search(query, candidate_pool, top_k=self._top_k)
            else:
                from rag.retriever import retrieve
                from rag.hybrid_search import bm25_only_search, _BM25_AVAILABLE
                if _BM25_AVAILABLE:
                    candidates = retrieve(query, top_k=self._top_k * 3)
                    results = bm25_only_search(query, candidates, top_k=self._top_k)
        except Exception:
            results = []
        latency = (time.perf_counter() - start) * 1000
        return results, latency

    def _run_hybrid(
        self,
        query: str,
        candidate_pool: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[Dict[str, Any]], float]:
        """Run hybrid RRF retrieval (real or mock)."""
        from rag.hybrid_search import hybrid_search
        start = time.perf_counter()
        if candidate_pool is not None:
            results = hybrid_search(query, candidate_pool, top_k=self._top_k)
        else:
            try:
                from rag.retriever import retrieve
                candidates = retrieve(query, top_k=self._top_k * 3)
                results = hybrid_search(query, candidates, top_k=self._top_k)
            except Exception:
                results = []
        latency = (time.perf_counter() - start) * 1000
        return results, latency

    def _evaluate_method(
        self,
        method: str,
        query_id: str,
        query: str,
        chunks: List[Dict[str, Any]],
        latency_ms: float,
        relevant_keywords: List[str],
    ) -> RetrievalMethodResult:
        """Compute P@K, Recall@K, MRR for a retrieved chunk list."""
        # Determine ground truth
        all_relevant = [c for c in chunks if _is_relevant_chunk(c, relevant_keywords)]
        total_relevant = max(1, len(all_relevant))

        top_k = chunks[: self._top_k]
        top_k_relevant = [c for c in top_k if _is_relevant_chunk(c, relevant_keywords)]

        precision = len(top_k_relevant) / max(1, len(top_k))
        recall = len(top_k_relevant) / total_relevant
        mrr = _compute_mrr(top_k, relevant_keywords)

        return RetrievalMethodResult(
            method=method,
            query_id=query_id,
            query=query,
            retrieved_count=len(chunks),
            relevant_in_top_k=len(top_k_relevant),
            precision_at_k=round(precision, 4),
            recall_at_k=round(recall, 4),
            mrr=round(mrr, 4),
            latency_ms=round(latency_ms, 3),
        )

    def run_evaluation(
        self,
        eval_dataset: Optional[List[Dict[str, Any]]] = None,
        candidate_pool_per_query: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> RAGEvalSuiteResult:
        """Run the full evaluation suite across all retrieval methods.

        Args:
            eval_dataset: List of evaluation entries (defaults to EVAL_DATASET).
            candidate_pool_per_query: Dict mapping query_id -> list of candidate chunks.
                                      If provided, runs in mock mode (no real DB needed).

        Returns:
            RAGEvalSuiteResult with per-method per-query results.
        """
        dataset = eval_dataset or EVAL_DATASET
        suite = RAGEvalSuiteResult(top_k=self._top_k)

        for entry in dataset:
            qid = entry["query_id"]
            query = entry["query"]
            keywords = entry["relevant_keywords"]
            pool = (candidate_pool_per_query or {}).get(qid)

            # Vector-only
            v_chunks, v_lat = self._run_vector_only(query, pool)
            suite.results.append(
                self._evaluate_method("vector_only", qid, query, v_chunks, v_lat, keywords)
            )

            # BM25-only
            b_chunks, b_lat = self._run_bm25_only(query, pool)
            suite.results.append(
                self._evaluate_method("bm25_only", qid, query, b_chunks, b_lat, keywords)
            )

            # Hybrid RRF
            h_chunks, h_lat = self._run_hybrid(query, pool)
            suite.results.append(
                self._evaluate_method("hybrid_rrf", qid, query, h_chunks, h_lat, keywords)
            )

        return suite


if __name__ == "__main__":
    runner = RAGEvaluationRunner(top_k=5)
    suite = runner.run_evaluation()
    print(suite.generate_markdown_report())

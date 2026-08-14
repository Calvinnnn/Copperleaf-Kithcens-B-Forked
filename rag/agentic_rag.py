"""Agentic RAG Orchestration Loop for Copperleaf Kitchens.

Implements the Agentic RAG retrieval-augmented generation loop with
Self-RAG style verification at each step:

1. RETRIEVE: Run hybrid search (vector + BM25) against the knowledge base
2. VERIFY_RELEVANCE (IS_REL): Filter irrelevant chunks using SelfRAGVerifier
3. REWRITE (optional): If no relevant chunks found, reformulate query and retry
4. GENERATE: Build context-enriched prompt from relevant chunks
5. VERIFY_SUPPORT (IS_SUP): Check if generated answer is grounded in retrieved docs
6. RETURN: Return final answer with retrieval trace for audit
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from memory.verification import SelfRAGVerifier, VerificationResult


@dataclass
class AgenticRAGResult:
    """Result from a single Agentic RAG run.

    Attributes:
        query: Original user query.
        final_query: Query used for retrieval (may be rewritten).
        retrieved_chunks: Raw chunks retrieved from the knowledge base.
        relevant_chunks: Chunks that passed IS_REL verification.
        answer_context: Formatted context string passed to the generator.
        verification: Self-RAG VerificationResult for the final answer.
        retrieval_trace: Audit log of retrieval decisions.
        was_rewritten: Whether the query was reformulated due to low recall.
        timestamp: ISO 8601 UTC timestamp.
    """

    query: str
    final_query: str
    retrieved_chunks: List[Dict[str, Any]]
    relevant_chunks: List[Dict[str, Any]]
    answer_context: str
    verification: Optional[VerificationResult]
    retrieval_trace: List[str] = field(default_factory=list)
    was_rewritten: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "query": self.query,
            "final_query": self.final_query,
            "retrieved_chunks_count": len(self.retrieved_chunks),
            "relevant_chunks_count": len(self.relevant_chunks),
            "was_rewritten": self.was_rewritten,
            "verification": self.verification.to_dict() if self.verification else None,
            "retrieval_trace": self.retrieval_trace,
            "timestamp": self.timestamp,
        }


def _simple_query_rewrite(query: str, attempt: int) -> str:
    """Simple query reformulation heuristic.

    Applies progressively looser reformulations on retry attempts.
    In a full system this would use an LLM to rephrase the query.
    """
    words = query.lower().split()
    # Remove stop words and shorten on second attempt
    stop_words = {"what", "is", "the", "are", "how", "many", "does", "do", "can", "a", "an"}
    keywords = [w for w in words if w not in stop_words]

    if attempt == 1 and keywords:
        return " ".join(keywords[:4])  # Focus on core keywords
    elif attempt >= 2 and keywords:
        return keywords[0]  # Fallback to single most important term
    return query


class AgenticRAGOrchestrator:
    """Agentic RAG orchestration loop with Self-RAG verification.

    Wraps the retrieval pipeline into a multi-step agentic loop that
    verifies relevance at retrieval time and grounding at generation time.
    """

    def __init__(
        self,
        verifier: Optional[SelfRAGVerifier] = None,
        top_k: int = 5,
        max_retry_attempts: int = 2,
        relevance_threshold: float = 0.4,
        support_threshold: float = 0.4,
        use_hybrid_search: bool = True,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize AgenticRAGOrchestrator.

        Args:
            verifier: SelfRAGVerifier instance (created with defaults if None).
            top_k: Number of chunks to retrieve per query.
            max_retry_attempts: Max query rewrites before accepting partial results.
            relevance_threshold: IS_REL score threshold for chunk filtering.
            support_threshold: IS_SUP score threshold for answer grounding check.
            use_hybrid_search: If True, use Hybrid RRF fusion; else vector-only.
            where_filter: Optional ChromaDB metadata filter for domain-specific retrieval.
        """
        self._verifier = verifier or SelfRAGVerifier(
            relevance_threshold=relevance_threshold,
            support_threshold=support_threshold,
        )
        self._top_k = top_k
        self._max_retry = max_retry_attempts
        self._use_hybrid = use_hybrid_search
        self._where = where_filter

    def _retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Run retrieval - requires ChromaDB and sentence-transformers.

        Falls back gracefully if vector store is not populated yet.
        """
        try:
            from rag.retriever import retrieve as vector_retrieve
            chunks = vector_retrieve(query, top_k=top_k, where=self._where)

            if self._use_hybrid and chunks:
                from rag.hybrid_search import hybrid_search
                chunks = hybrid_search(query, chunks, top_k=top_k)

            return chunks
        except Exception as exc:
            return []  # Graceful no-op when knowledge base not yet built

    def _filter_relevant(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply IS_REL verification to filter irrelevant chunks."""
        relevant = []
        for chunk in chunks:
            is_rel, score, _ = self._verifier.verify_relevance(query, chunk["text"])
            if is_rel:
                relevant.append(chunk)
        return relevant

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a prompt context block."""
        if not chunks:
            return "[NO RELEVANT CONTEXT RETRIEVED]"

        lines = ["[RETRIEVED KNOWLEDGE BASE CONTEXT]"]
        for i, chunk in enumerate(chunks, 1):
            src = chunk.get("metadata", {}).get("source", "unknown")
            page = chunk.get("metadata", {}).get("page", "?")
            lines.append(f"\n--- Source {i}: {src} (page {page}) ---")
            lines.append(chunk["text"])

        return "\n".join(lines)

    def run(
        self,
        query: str,
        candidate_answer: Optional[str] = None,
    ) -> AgenticRAGResult:
        """Execute the full Agentic RAG loop for a query.

        Args:
            query: User query to retrieve context for.
            candidate_answer: Optional pre-generated answer to verify grounding.

        Returns:
            AgenticRAGResult with retrieved context, verification, and trace.
        """
        trace: List[str] = []
        current_query = query
        was_rewritten = False

        trace.append(f"[AGENTIC_RAG] Starting retrieval for: '{query}'")

        # STEP 1: RETRIEVE with optional retry on low relevance
        all_retrieved: List[Dict[str, Any]] = []
        relevant_chunks: List[Dict[str, Any]] = []

        for attempt in range(self._max_retry + 1):
            all_retrieved = self._retrieve(current_query, top_k=self._top_k)
            trace.append(f"[RETRIEVE] attempt={attempt+1} query='{current_query}' chunks={len(all_retrieved)}")

            if not all_retrieved:
                if attempt < self._max_retry:
                    current_query = _simple_query_rewrite(query, attempt + 1)
                    was_rewritten = True
                    trace.append(f"[REWRITE] No results. Reformulated to: '{current_query}'")
                    continue
                break

            # STEP 2: IS_REL - filter for relevance
            relevant_chunks = self._filter_relevant(current_query, all_retrieved)
            trace.append(f"[IS_REL] {len(relevant_chunks)}/{len(all_retrieved)} chunks passed relevance check")

            if relevant_chunks or attempt >= self._max_retry:
                break

            # Retry with rewritten query
            current_query = _simple_query_rewrite(query, attempt + 1)
            was_rewritten = True
            trace.append(f"[REWRITE] Low relevance. Reformulated to: '{current_query}'")

        # STEP 3: BUILD CONTEXT
        answer_context = self._build_context(relevant_chunks or all_retrieved)

        # STEP 4: IS_SUP - verify grounding of candidate answer
        verification: Optional[VerificationResult] = None
        if candidate_answer is not None:
            source_texts = [c["text"] for c in (relevant_chunks or all_retrieved)]
            verification = self._verifier.verify_memory_recall(
                query=query,
                answer=candidate_answer,
                recalled_memories=source_texts,
            )
            trace.append(
                f"[IS_SUP] grounded={verification.is_supported} "
                f"support_score={verification.support_score:.2f}"
            )

        trace.append(f"[AGENTIC_RAG] Complete. relevant_chunks={len(relevant_chunks)}")

        return AgenticRAGResult(
            query=query,
            final_query=current_query,
            retrieved_chunks=all_retrieved,
            relevant_chunks=relevant_chunks,
            answer_context=answer_context,
            verification=verification,
            retrieval_trace=trace,
            was_rewritten=was_rewritten,
        )

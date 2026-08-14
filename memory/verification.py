"""Self-RAG-Style Verification Module for AI Agent Architecture.

This module implements post-retrieval and post-generation verification checks:
1. Relevance Verification (`IS_REL`): Verifies if retrieved content/memory items are actually relevant to the current user query or goal context.
2. Support Verification (`IS_SUP`): Verifies if a candidate response or generated statement is factually grounded and supported by the retrieved memory/knowledge.

Applies to memories recalled from Episodic and Semantic stores, as well as RAG context.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import re
from typing import Any, List, Optional, Tuple


@dataclass
class VerificationResult:
    """Detailed verification score and decision log entry.

    Attributes:
        is_relevant: True if retrieved content passed the relevance filter.
        is_supported: True if generated answer is grounded in retrieved content.
        relevance_score: Numerical relevance confidence (0.0 to 1.0).
        support_score: Numerical grounding/support confidence (0.0 to 1.0).
        reasoning: Explanation for verification outcome.
        flagged_hallucinations: List of ungrounded tokens or claims detected.
    """

    is_relevant: bool
    is_supported: bool
    relevance_score: float
    support_score: float
    reasoning: str
    flagged_hallucinations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize verification result to dictionary format."""
        return {
            "is_relevant": self.is_relevant,
            "is_supported": self.is_supported,
            "relevance_score": round(self.relevance_score, 2),
            "support_score": round(self.support_score, 2),
            "reasoning": self.reasoning,
            "flagged_hallucinations": self.flagged_hallucinations,
        }


class SelfRAGVerifier:
    """Self-RAG style verification engine for memory recall and response grounding."""

    def __init__(self, relevance_threshold: float = 0.5, support_threshold: float = 0.5) -> None:
        """Initialize SelfRAGVerifier with confidence thresholds.

        Args:
            relevance_threshold: Minimum score to pass IS_REL check.
            support_threshold: Minimum score to pass IS_SUP check.
        """
        self.relevance_threshold = relevance_threshold
        self.support_threshold = support_threshold

    def verify_relevance(self, query: str, context_item: Any) -> Tuple[bool, float, str]:
        """Verify if retrieved context or memory item is relevant to the query.

        Args:
            query: User prompt or current working goal text.
            context_item: Text snippet, fact dict, or memory object retrieved.

        Returns:
            Tuple of (is_relevant: bool, score: float, reasoning: str).
        """
        query_words = set(re.findall(r'\w+', query.lower()))
        if not query_words:
            return True, 1.0, "Empty query context passed."

        context_str = json.dumps(context_item) if isinstance(context_item, (dict, list)) else str(context_item)
        context_words = set(re.findall(r'\w+', context_str.lower()))

        overlap = query_words.intersection(context_words)
        overlap_ratio = len(overlap) / max(1, len(query_words))
        
        # Substring keyword check
        direct_match = any(w in context_str.lower() for w in query_words if len(w) > 3)
        score = min(1.0, overlap_ratio * 1.5 + (0.3 if direct_match else 0.0))
        is_rel = score >= self.relevance_threshold

        reason = (
            f"Context contains {len(overlap)} matching query terms (score={score:.2f})."
            if is_rel
            else f"Low keyword overlap with query (score={score:.2f} < threshold={self.relevance_threshold})."
        )
        return is_rel, score, reason

    def verify_support(self, answer: str, retrieved_sources: List[Any]) -> Tuple[bool, float, List[str], str]:
        """Verify if a generated answer is grounded in the retrieved sources.

        Args:
            answer: Candidate agent output text.
            retrieved_sources: List of retrieved facts/episodes/documents.

        Returns:
            Tuple of (is_supported: bool, score: float, hallucinations: list, reasoning: str).
        """
        sources_text = " ".join([
            json.dumps(s) if isinstance(s, (dict, list)) else str(s)
            for s in retrieved_sources
        ]).lower()

        answer_words = re.findall(r'\b[A-Za-z0-9_-]{4,}\b', answer.lower())
        if not answer_words:
            return True, 1.0, [], "Answer contains no verifiable entities."

        unsupported_claims: List[str] = []
        supported_count = 0

        for word in answer_words:
            if word in sources_text:
                supported_count += 1
            else:
                unsupported_claims.append(word)

        support_score = supported_count / max(1, len(answer_words))
        is_sup = support_score >= self.support_threshold

        reasoning = (
            f"Answer is grounded in retrieved memory/docs (support_score={support_score:.2f})."
            if is_sup
            else f"Detected potential ungrounded claims: {unsupported_claims[:5]} (score={support_score:.2f})."
        )
        return is_sup, support_score, unsupported_claims, reasoning

    def verify_memory_recall(
        self, query: str, answer: str, recalled_memories: List[Any]
    ) -> VerificationResult:
        """Run full Self-RAG verification over memory recall and output generation."""
        # 1. Relevance check over memories
        relevant_memories = []
        rel_scores = []
        for mem in recalled_memories:
            is_rel, score, _ = self.verify_relevance(query, mem)
            if is_rel:
                relevant_memories.append(mem)
                rel_scores.append(score)

        avg_rel_score = sum(rel_scores) / max(1, len(rel_scores)) if rel_scores else 0.0
        is_relevant = len(relevant_memories) > 0 or len(recalled_memories) == 0

        # 2. Support check over answer
        is_supported, sup_score, hallucinations, sup_reason = self.verify_support(
            answer, relevant_memories if relevant_memories else recalled_memories
        )

        overall_reasoning = f"Relevance: {avg_rel_score:.2f} | Support: {sup_score:.2f} - {sup_reason}"

        return VerificationResult(
            is_relevant=is_relevant,
            is_supported=is_supported,
            relevance_score=avg_rel_score,
            support_score=sup_score,
            reasoning=overall_reasoning,
            flagged_hallucinations=hallucinations,
        )


if __name__ == "__main__":
    verifier = SelfRAGVerifier()
    query = "What is Branch 1's preferred emergency produce supplier?"
    recalled = ["Branch 1 preferred supplier: GreenRoute Wholesale (Account #GRW-4477)"]
    answer = "Branch 1 uses GreenRoute Wholesale for emergency produce."

    result = verifier.verify_memory_recall(query, answer, recalled)
    print("Self-RAG Verification Result:")
    print(json.dumps(result.to_dict(), indent=2))

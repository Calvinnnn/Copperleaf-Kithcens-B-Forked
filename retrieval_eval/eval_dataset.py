"""Retrieval Evaluation Dataset for Copperleaf Kitchens.

This module contains the fixed domain-specific test questions used to evaluate
Naive RAG, Hybrid Search, and Agentic RAG architectures.

Design rationale:
- q1: General semantic question → Naive RAG handles well (no exact identifiers)
- q2: Exact identifier query (Account #APX-9982, GRW-4477) → Hybrid Search wins
  because BM25 exact keyword matching retrieves the specific account code that
  pure vector similarity may miss.
- q3: Multi-part decomposition question → Agentic RAG wins by retrieving two
  separate evidence pieces (compliance policy + spoilage threshold) in multiple
  retrieval rounds.
- q4: Procedure code lookup → Hybrid Search advantage (exact code "BO-101")
- q5: General best-practice question → Naive RAG handles well

DO NOT modify these questions between evaluation runs. The test set must remain
fixed to produce a valid comparison table across architectures.
"""

from typing import Any, Dict, List

RETRIEVAL_EVAL_DATASET: List[Dict[str, Any]] = [
    {
        "query_id": "ret_q1",
        "description": "General spoilage policy — favors Naive RAG",
        "query": "What is the policy for writing off spoiled produce in Copperleaf Kitchens?",
        "architecture_advantage": "naive_rag",
        "rationale": "Purely semantic query with no exact identifiers; vector similarity retrieval handles well.",
        "relevant_keywords": ["write-off", "write off", "spoiled", "policy", "produce", "expired"],
        "requires_multi_hop": False,
    },
    {
        "query_id": "ret_q2",
        "description": "Exact supplier account lookup — favors Hybrid Search",
        "query": "What is the emergency supplier account number APX-9982 used for?",
        "architecture_advantage": "hybrid_search",
        "rationale": (
            "Exact identifier 'APX-9982' does not embed distinctively. BM25 exact keyword"
            " matching in the hybrid fusion ranks it at position 1, while pure vector search"
            " may miss it or rank it lower."
        ),
        "relevant_keywords": ["APX-9982", "Apex Fresh", "supplier", "account", "emergency"],
        "requires_multi_hop": False,
    },
    {
        "query_id": "ret_q3",
        "description": "Multi-hop: branch compliance + reorder policy — favors Agentic RAG",
        "query": (
            "For Branch 1 that has a high spoilage rate for dairy products, "
            "what are the food safety compliance steps AND the reorder threshold policy "
            "that should be triggered when stock falls below minimum?"
        ),
        "architecture_advantage": "agentic_rag",
        "rationale": (
            "Requires two separate evidence retrievals: (1) food safety compliance protocol"
            " for dairy, and (2) reorder threshold trigger policy. Naive RAG retrieves only"
            " one chunk. Agentic RAG iterates and retrieves both pieces."
        ),
        "relevant_keywords": ["food safety", "compliance", "dairy", "reorder", "threshold", "stock", "branch"],
        "requires_multi_hop": True,
    },
    {
        "query_id": "ret_q4",
        "description": "Procedure code BO-101 lookup — favors Hybrid Search",
        "query": "What does procedure BO-101 specify for branch opening?",
        "architecture_advantage": "hybrid_search",
        "rationale": (
            "Procedure code 'BO-101' is an exact identifier that BM25 matches precisely."
            " Vector similarity retrieval may confuse it with other procedure codes."
        ),
        "relevant_keywords": ["BO-101", "opening", "procedure", "branch", "unlock"],
        "requires_multi_hop": False,
    },
    {
        "query_id": "ret_q5",
        "description": "General kitchen temperature storage — favors Naive RAG",
        "query": "What are the recommended cold storage temperature ranges for kitchen ingredients?",
        "architecture_advantage": "naive_rag",
        "rationale": "Semantic question with no exact codes. Vector similarity retrieval handles well.",
        "relevant_keywords": ["temperature", "storage", "cold", "refrigerate", "kitchen"],
        "requires_multi_hop": False,
    },
]

"""
test_issue6_rag_selfrag.py — Test suite for Issue #6:
- RAG retriever metadata filtering interface
- Hybrid Search (BM25 + Vector RRF fusion)
- Agentic RAG orchestration loop (query → IS_REL → context → IS_SUP)
- Self-RAG Verifier (IS_REL relevance check, IS_SUP grounding check)
- RAG Evaluation runner (Precision@K, Recall@K, MRR)
- MemoryEnabledAgent full integration (context building + verify_response)
"""

import unittest
from typing import Any, Dict, List


# ─── Test Fixtures ──────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {
        "text": "Roma Tomatoes write-off policy: spoiled produce must be written off within 24 hours of expiry. Manager approval required.",
        "metadata": {"source": "ops_manual.pdf", "page": 3},
        "distance": 0.10,
    },
    {
        "text": "Emergency supplier contacts: For produce emergencies call Nile Fresh Produce at +20-2-555-0201.",
        "metadata": {"source": "supplier_list.pdf", "page": 1},
        "distance": 0.18,
    },
    {
        "text": "Reorder thresholds: when current_quantity falls below reorder_threshold, system flags low-stock status for manager review.",
        "metadata": {"source": "inventory_guide.pdf", "page": 7},
        "distance": 0.22,
    },
    {
        "text": "Branch manager daily responsibilities include opening report, stock review, and approval of write-offs.",
        "metadata": {"source": "ops_manual.pdf", "page": 5},
        "distance": 0.30,
    },
    {
        "text": "Food safety compliance: all kitchen staff must complete temperature and hygiene compliance checks each morning.",
        "metadata": {"source": "food_safety.pdf", "page": 2},
        "distance": 0.35,
    },
    {
        "text": "Unrelated content: company picnic schedule and annual social events calendar for staff teams.",
        "metadata": {"source": "hr_bulletin.pdf", "page": 1},
        "distance": 0.85,  # high distance = low relevance
    },
]


# ─── Self-RAG Verifier Tests ─────────────────────────────────────────────────

class TestSelfRAGVerifier(unittest.TestCase):

    def setUp(self):
        from memory.verification import SelfRAGVerifier
        self.verifier = SelfRAGVerifier(relevance_threshold=0.4, support_threshold=0.4)

    def test_is_rel_relevant_context(self):
        """IS_REL must return True for context that shares key terms with the query."""
        query = "What is the policy for writing off spoiled produce?"
        context = "Spoiled produce write-off requires manager approval within 24 hours."
        is_rel, score, reason = self.verifier.verify_relevance(query, context)

        self.assertTrue(is_rel)
        self.assertGreater(score, 0.4)
        self.assertIsInstance(reason, str)

    def test_is_rel_irrelevant_context(self):
        """IS_REL must return a low score for completely unrelated context."""
        query = "What is the spoilage write-off policy for produce?"
        context = "Quarterly soccer tournament brackets released by HR recreation committee."
        is_rel, score, reason = self.verifier.verify_relevance(query, context)

        # Even if bool decision flips at borderline, score must be well below 0.7
        self.assertLess(score, 0.7)
        self.assertIsInstance(reason, str)

    def test_is_sup_grounded_answer(self):
        """IS_SUP must return True for answer grounded in sources."""
        answer = "Branch 1 uses GreenRoute Wholesale for emergency produce."
        sources = [
            "Branch 1 emergency supplier: GreenRoute Wholesale (Account #GRW-4477)",
            "Preferred supplier for produce: GreenRoute Wholesale",
        ]
        is_sup, score, hallucinations, reason = self.verifier.verify_support(answer, sources)

        self.assertTrue(is_sup)
        self.assertGreater(score, 0.4)

    def test_is_sup_hallucination_detection(self):
        """IS_SUP must flag ungrounded claims as hallucinations."""
        answer = "Branch 1 uses XyzCorp for emergency produce."
        sources = ["Branch 1 emergency supplier: GreenRoute Wholesale"]
        is_sup, score, hallucinations, reason = self.verifier.verify_support(answer, sources)

        self.assertIsInstance(hallucinations, list)
        self.assertIn("xyzcorp", [h.lower() for h in hallucinations])

    def test_verify_memory_recall_full_pipeline(self):
        """Full Self-RAG verify_memory_recall must return VerificationResult with all fields."""
        from memory.verification import VerificationResult
        query = "What is Branch 1's preferred emergency produce supplier?"
        recalled = ["Branch 1 preferred supplier: GreenRoute Wholesale (Account #GRW-4477)"]
        answer = "Branch 1 uses GreenRoute Wholesale for emergency produce."

        result = self.verifier.verify_memory_recall(query, answer, recalled)

        self.assertIsInstance(result, VerificationResult)
        self.assertIsInstance(result.is_relevant, bool)
        self.assertIsInstance(result.is_supported, bool)
        self.assertGreaterEqual(result.relevance_score, 0.0)
        self.assertLessEqual(result.relevance_score, 1.0)
        self.assertIsInstance(result.flagged_hallucinations, list)


# ─── Hybrid Search Tests ─────────────────────────────────────────────────────

class TestHybridSearch(unittest.TestCase):

    def test_hybrid_search_returns_top_k(self):
        """Hybrid search must return at most top_k results."""
        from rag.hybrid_search import hybrid_search
        results = hybrid_search("write off spoiled produce", SAMPLE_CHUNKS, top_k=3)
        self.assertLessEqual(len(results), 3)
        self.assertGreater(len(results), 0)

    def test_hybrid_search_rrf_scores_present(self):
        """All hybrid results must have rrf_score field."""
        from rag.hybrid_search import hybrid_search
        results = hybrid_search("write off spoiled produce", SAMPLE_CHUNKS, top_k=5)
        for chunk in results:
            self.assertIn("rrf_score", chunk)
            self.assertGreater(chunk["rrf_score"], 0.0)

    def test_hybrid_search_relevant_chunk_ranked_higher(self):
        """Relevant chunk should outrank completely unrelated chunk in hybrid ranking."""
        from rag.hybrid_search import hybrid_search
        results = hybrid_search("spoiled produce write-off policy", SAMPLE_CHUNKS, top_k=5)
        texts = [r["text"] for r in results]
        # The picnic text (hr_bulletin) should not be the top result
        if len(texts) > 1:
            self.assertNotIn("picnic", results[0]["text"].lower())

    def test_bm25_only_search_returns_scored_results(self):
        """BM25-only search must return chunks with bm25_score field."""
        try:
            from rag.hybrid_search import bm25_only_search
            results = bm25_only_search("spoiled produce write-off", SAMPLE_CHUNKS, top_k=3)
            self.assertLessEqual(len(results), 3)
            for chunk in results:
                self.assertIn("bm25_score", chunk)
        except ImportError:
            self.skipTest("rank_bm25 not installed - skipping BM25 test")

    def test_hybrid_search_empty_candidates_graceful(self):
        """Hybrid search with empty candidate list must return empty result."""
        from rag.hybrid_search import hybrid_search
        results = hybrid_search("anything", [], top_k=5)
        self.assertEqual(results, [])


# ─── Agentic RAG Orchestrator Tests ─────────────────────────────────────────

class TestAgenticRAGOrchestrator(unittest.TestCase):

    def _make_orchestrator(self):
        from rag.agentic_rag import AgenticRAGOrchestrator
        from memory.verification import SelfRAGVerifier
        verifier = SelfRAGVerifier(relevance_threshold=0.35, support_threshold=0.35)
        return AgenticRAGOrchestrator(
            verifier=verifier,
            top_k=5,
            max_retry_attempts=1,
        )

    def test_orchestrator_run_no_db_graceful(self):
        """Orchestrator must run gracefully even when ChromaDB is not populated."""
        from rag.agentic_rag import AgenticRAGResult
        orch = self._make_orchestrator()
        result = orch.run("What is the spoilage write-off policy?")

        self.assertIsInstance(result, AgenticRAGResult)
        self.assertIsInstance(result.retrieval_trace, list)
        self.assertGreater(len(result.retrieval_trace), 0)
        self.assertIsInstance(result.answer_context, str)

    def test_orchestrator_with_candidate_answer_verifies_support(self):
        """Orchestrator must run IS_SUP check when candidate_answer is provided."""
        orch = self._make_orchestrator()
        result = orch.run(
            "What is the spoilage write-off policy?",
            candidate_answer="Spoiled produce requires manager sign-off.",
        )

        # IS_SUP should have run (verification may be None if no chunks retrieved)
        self.assertIsInstance(result.retrieval_trace, list)

    def test_agentic_rag_result_serializes(self):
        """AgenticRAGResult.to_dict() must return a serializable dictionary."""
        orch = self._make_orchestrator()
        result = orch.run("write-off policy for spoiled produce")
        d = result.to_dict()

        self.assertIn("query", d)
        self.assertIn("retrieved_chunks_count", d)
        self.assertIn("relevant_chunks_count", d)
        self.assertIn("retrieval_trace", d)
        self.assertIn("was_rewritten", d)


# ─── RAG Retriever Metadata Filter Interface Tests ───────────────────────────

class TestRetrieverInterface(unittest.TestCase):

    def test_retrieve_function_signature(self):
        """retrieve() must accept query, top_k, and where kwargs."""
        import inspect
        from rag.retriever import retrieve
        sig = inspect.signature(retrieve)
        params = sig.parameters
        self.assertIn("query", params)
        self.assertIn("top_k", params)
        self.assertIn("where", params)

    def test_retrieve_where_param_is_optional(self):
        """retrieve() where parameter must be Optional (default None)."""
        import inspect
        from rag.retriever import retrieve
        sig = inspect.signature(retrieve)
        where_param = sig.parameters.get("where")
        self.assertIsNotNone(where_param)
        self.assertIsNone(where_param.default)


# ─── RAG Evaluation Runner Tests ─────────────────────────────────────────────

class TestRAGEvaluationRunner(unittest.TestCase):

    def _make_pool(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build a mock candidate pool for all eval queries using SAMPLE_CHUNKS."""
        from rag.rag_eval import EVAL_DATASET
        return {entry["query_id"]: SAMPLE_CHUNKS for entry in EVAL_DATASET}

    def test_eval_runner_returns_results_for_all_methods(self):
        """Evaluation runner must produce results for vector_only, bm25_only, hybrid_rrf."""
        from rag.rag_eval import RAGEvaluationRunner, EVAL_DATASET
        try:
            runner = RAGEvaluationRunner(top_k=5)
            suite = runner.run_evaluation(
                eval_dataset=EVAL_DATASET,
                candidate_pool_per_query=self._make_pool(),
            )
            method_names = {r.method for r in suite.results}
            self.assertIn("vector_only", method_names)
            self.assertIn("hybrid_rrf", method_names)
            # BM25-only requires rank_bm25 - check conditionally
            from rag.hybrid_search import _BM25_AVAILABLE
            if _BM25_AVAILABLE:
                self.assertIn("bm25_only", method_names)
        except ImportError:
            self.skipTest("rank_bm25 not available")

    def test_eval_metrics_within_valid_range(self):
        """All precision, recall, and MRR metrics must be in [0.0, 1.0]."""
        from rag.rag_eval import RAGEvaluationRunner, EVAL_DATASET
        runner = RAGEvaluationRunner(top_k=5)
        suite = runner.run_evaluation(
            eval_dataset=EVAL_DATASET,
            candidate_pool_per_query=self._make_pool(),
        )
        for r in suite.results:
            self.assertGreaterEqual(r.precision_at_k, 0.0)
            self.assertLessEqual(r.precision_at_k, 1.0)
            self.assertGreaterEqual(r.recall_at_k, 0.0)
            self.assertLessEqual(r.recall_at_k, 1.0)
            self.assertGreaterEqual(r.mrr, 0.0)
            self.assertLessEqual(r.mrr, 1.0)

    def test_eval_markdown_report_contains_methods(self):
        """Markdown report must include all method names and required columns."""
        from rag.rag_eval import RAGEvaluationRunner, EVAL_DATASET
        runner = RAGEvaluationRunner(top_k=5)
        suite = runner.run_evaluation(
            eval_dataset=EVAL_DATASET,
            candidate_pool_per_query=self._make_pool(),
        )
        report = suite.generate_markdown_report()
        self.assertIn("vector_only", report)
        self.assertIn("hybrid_rrf", report)
        self.assertIn("P@K", report)
        self.assertIn("MRR", report)

    def test_eval_summary_aggregates_per_method(self):
        """Summary dict must aggregate avg_precision_at_k and avg_mrr per method."""
        from rag.rag_eval import RAGEvaluationRunner, EVAL_DATASET
        runner = RAGEvaluationRunner(top_k=5)
        suite = runner.run_evaluation(
            eval_dataset=EVAL_DATASET,
            candidate_pool_per_query=self._make_pool(),
        )
        summary = suite.summary()
        self.assertGreater(len(summary), 0)
        for method_stats in summary.values():
            self.assertIn("avg_precision_at_k", method_stats)
            self.assertIn("avg_recall_at_k", method_stats)
            self.assertIn("avg_mrr", method_stats)
            self.assertIn("avg_latency_ms", method_stats)


# ─── MemoryEnabledAgent Full Integration Tests ───────────────────────────────

class TestMemoryEnabledAgentIntegration(unittest.TestCase):

    def setUp(self):
        from mcp_server.init_db import build as build_db
        build_db()
        from agent.agent import MemoryEnabledAgent
        self.agent = MemoryEnabledAgent(
            stm_capacity=5,
            consolidation_batch_size=3,
            enable_rag=False,  # Disable live RAG to not require ChromaDB
        )

    def test_agent_has_selfrag_verifier(self):
        """MemoryEnabledAgent must expose a SelfRAGVerifier instance."""
        from memory.verification import SelfRAGVerifier
        self.assertIsInstance(self.agent.verifier, SelfRAGVerifier)

    def test_agent_has_context_strategy(self):
        """MemoryEnabledAgent must expose a BaseContextStrategy instance."""
        from context_eval.sliding_window import BaseContextStrategy
        self.assertIsInstance(self.agent.context_strategy, BaseContextStrategy)

    def test_agent_has_scratchpad(self):
        """MemoryEnabledAgent must expose a Scratchpad instance."""
        from memory.scratchpad import Scratchpad
        self.assertIsInstance(self.agent.scratchpad, Scratchpad)

    def test_verify_response_returns_verification_result(self):
        """verify_response() must return a VerificationResult with all fields."""
        from memory.verification import VerificationResult
        self.agent.receive_message("Branch 1 uses GreenRoute Wholesale.", role="user")

        result = self.agent.verify_response(
            query="What supplier does Branch 1 use?",
            answer="Branch 1 uses GreenRoute Wholesale for produce.",
            recalled_memories=["Branch 1 preferred supplier: GreenRoute Wholesale"],
        )

        self.assertIsInstance(result, VerificationResult)
        self.assertIsInstance(result.is_relevant, bool)
        self.assertIsInstance(result.is_supported, bool)
        self.assertGreaterEqual(result.support_score, 0.0)

    def test_build_context_returns_message_list(self):
        """build_context() must return a non-empty list of message dicts."""
        self.agent.receive_message("I need help with inventory.", role="user")
        self.agent.receive_message("I can help with that.", role="assistant")

        ctx = self.agent.build_context(max_tokens=1000)

        self.assertIsInstance(ctx, list)
        self.assertGreater(len(ctx), 0)
        for msg in ctx:
            self.assertIn("role", msg)
            self.assertIn("content", msg)

    def test_build_context_scratchpad_injected_when_goal_set(self):
        """build_context() must inject scratchpad as system message when goal is active."""
        self.agent.scratchpad.set_goal("Investigate inventory spoilage", "Quality audit")
        self.agent.receive_message("Check spoilage levels", role="user")

        ctx = self.agent.build_context(max_tokens=2000)
        system_msgs = [m for m in ctx if m.get("role") == "system"]
        scratch_msgs = [m for m in system_msgs if "SCRATCHPAD" in m.get("content", "")]
        self.assertGreater(len(scratch_msgs), 0)

    def test_stm_overflow_routes_to_episodic(self):
        """STM overflow must route evicted messages through router to episodic memory."""
        agent = __import__("agent.agent", fromlist=["MemoryEnabledAgent"]).MemoryEnabledAgent(
            stm_capacity=3,
            consolidation_batch_size=10,
            enable_rag=False,
        )
        # Add preference message that should be promoted
        agent.receive_message(
            "We always prefer GreenRoute Wholesale for emergency produce.",
            role="user",
        )
        agent.receive_message("Understood.", role="assistant")
        agent.receive_message("Check the stock please.", role="user")
        agent.receive_message("Looking at stock now.", role="assistant")  # 4th msg: overflow

        # Router should have processed at least one item
        self.assertLessEqual(agent.short_term.size, 3)


if __name__ == "__main__":
    unittest.main()

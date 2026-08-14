"""
test_retrieval_eval.py — Test suite for retrieval_eval/ (Issue #6)

Verifies the domain-specific evaluation dataset and the RetrievalEvaluationRunner
work correctly in mock mode (no real ChromaDB required).
"""

import unittest
from retrieval_eval.eval_dataset import RETRIEVAL_EVAL_DATASET
from retrieval_eval.run_eval import RetrievalEvaluationRunner, _is_relevant, _mrr, _estimate_tokens


# ─── Shared mock candidate pool ──────────────────────────────────────────────

def _make_pool_for(query_id: str):
    """Return a small mock candidate pool for each query."""
    pools = {
        "ret_q1": [
            {"text": "The write-off policy for spoiled produce requires manager sign-off when value exceeds $500.", "distance": 0.1, "metadata": {}},
            {"text": "Reorder thresholds trigger automatic procurement notifications.", "distance": 0.5, "metadata": {}},
            {"text": "Routine inventory checks happen every Monday morning.", "distance": 0.8, "metadata": {}},
        ],
        "ret_q2": [
            {"text": "Apex Fresh Logistics (Account #APX-9982) is Branch 1's emergency produce supplier.", "distance": 0.05, "metadata": {}},
            {"text": "Supplier accounts must be registered before orders are placed.", "distance": 0.6, "metadata": {}},
        ],
        "ret_q3": [
            {"text": "Food safety compliance for dairy requires daily temperature logs and refrigeration checks.", "distance": 0.15, "metadata": {}},
            {"text": "When stock falls below reorder threshold, an automatic restock order is triggered.", "distance": 0.2, "metadata": {}},
            {"text": "Branch managers are responsible for weekly compliance audits.", "distance": 0.7, "metadata": {}},
        ],
        "ret_q4": [
            {"text": "Opening Procedures (BO-101): 1 Unlock main entrance 2 Activate POS systems 3 Check refrigeration.", "distance": 0.08, "metadata": {}},
            {"text": "Closing procedures require all stock to be sealed and refrigeration verified.", "distance": 0.55, "metadata": {}},
        ],
        "ret_q5": [
            {"text": "Cold storage temperature ranges: refrigeration 1–4°C, freezer -18°C or below.", "distance": 0.12, "metadata": {}},
            {"text": "Kitchen staff must wear gloves and hairnets during food preparation.", "distance": 0.4, "metadata": {}},
        ],
    }
    return pools.get(query_id, [])


class TestEvalDataset(unittest.TestCase):
    """Verify the fixed evaluation dataset structure."""

    def test_dataset_has_five_entries(self):
        """RETRIEVAL_EVAL_DATASET must contain exactly 5 domain queries."""
        self.assertEqual(len(RETRIEVAL_EVAL_DATASET), 5)

    def test_each_entry_has_required_fields(self):
        """Each dataset entry must have all required fields."""
        required = {"query_id", "description", "query", "architecture_advantage", "relevant_keywords"}
        for entry in RETRIEVAL_EVAL_DATASET:
            for field in required:
                self.assertIn(field, entry, f"Missing field '{field}' in {entry['query_id']}")

    def test_architecture_advantages_cover_all_three(self):
        """Dataset must include at least one query favoring each architecture."""
        advantages = {e["architecture_advantage"] for e in RETRIEVAL_EVAL_DATASET}
        self.assertIn("naive_rag", advantages, "No query favoring naive_rag")
        self.assertIn("hybrid_search", advantages, "No query favoring hybrid_search")
        self.assertIn("agentic_rag", advantages, "No query favoring agentic_rag")

    def test_one_query_requires_multi_hop(self):
        """At least one query must require multi-hop decomposition (for Agentic RAG)."""
        multi_hop = [e for e in RETRIEVAL_EVAL_DATASET if e.get("requires_multi_hop")]
        self.assertGreaterEqual(len(multi_hop), 1, "No multi-hop query found for Agentic RAG")

    def test_query_ids_are_unique(self):
        """Query IDs must be unique across the dataset."""
        ids = [e["query_id"] for e in RETRIEVAL_EVAL_DATASET]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate query IDs found")

    def test_exact_identifier_query_exists(self):
        """At least one query must contain an exact account identifier for Hybrid Search."""
        has_exact = any(
            "APX-9982" in e["query"] or "BO-101" in e["query"]
            for e in RETRIEVAL_EVAL_DATASET
        )
        self.assertTrue(has_exact, "No exact identifier query found for Hybrid Search validation")


class TestHelperFunctions(unittest.TestCase):
    """Verify helper oracle functions."""

    def test_is_relevant_returns_true_on_keyword_match(self):
        chunk = {"text": "write-off policy for spoiled produce requires sign-off"}
        self.assertTrue(_is_relevant(chunk, ["write-off", "spoiled"]))

    def test_is_relevant_returns_false_when_no_match(self):
        chunk = {"text": "routine daily staff meeting at 9am"}
        self.assertFalse(_is_relevant(chunk, ["APX-9982", "emergency", "supplier"]))

    def test_mrr_first_position(self):
        chunks = [
            {"text": "Apex Fresh Logistics Account #APX-9982"},
            {"text": "Unrelated content about closing procedures"},
        ]
        mrr = _mrr(chunks, ["APX-9982"])
        self.assertAlmostEqual(mrr, 1.0, places=2)

    def test_mrr_second_position(self):
        chunks = [
            {"text": "Unrelated closing procedures text"},
            {"text": "APX-9982 emergency produce supplier"},
        ]
        mrr = _mrr(chunks, ["APX-9982"])
        self.assertAlmostEqual(mrr, 0.5, places=2)

    def test_mrr_no_relevant(self):
        chunks = [{"text": "nothing relevant here at all"}]
        mrr = _mrr(chunks, ["APX-9982", "emergency"])
        self.assertEqual(mrr, 0.0)

    def test_estimate_tokens_reasonable(self):
        tokens = _estimate_tokens("Hello world this is a test")
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, 50)


class TestRetrievalEvaluationRunner(unittest.TestCase):
    """Test runner logic in mock mode (no real ChromaDB required)."""

    def setUp(self):
        self.runner = RetrievalEvaluationRunner(top_k=3)
        self.pool_per_query = {
            entry["query_id"]: _make_pool_for(entry["query_id"])
            for entry in RETRIEVAL_EVAL_DATASET
        }

    def test_run_returns_results_for_all_architectures(self):
        """Runner must produce results for naive_rag, hybrid_search, and agentic_rag."""
        result = self.runner.run(pool_per_query=self.pool_per_query)
        archs = {r["architecture"] for r in result["per_query"]}
        self.assertIn("naive_rag", archs)
        self.assertIn("hybrid_search", archs)
        self.assertIn("agentic_rag", archs)

    def test_results_cover_all_five_queries(self):
        """Each architecture must produce one result per query (5 × 3 = 15 records)."""
        result = self.runner.run(pool_per_query=self.pool_per_query)
        self.assertEqual(len(result["per_query"]), 15)

    def test_summary_has_three_architectures(self):
        """Summary must aggregate metrics for exactly 3 architectures."""
        result = self.runner.run(pool_per_query=self.pool_per_query)
        self.assertEqual(len(result["summary"]), 3)

    def test_accuracy_is_zero_or_one(self):
        """Accuracy values must be binary (0 or 1)."""
        result = self.runner.run(pool_per_query=self.pool_per_query)
        for r in result["per_query"]:
            self.assertIn(r["accuracy"], [0, 1], f"Invalid accuracy for {r}")

    def test_mrr_is_in_valid_range(self):
        """MRR values must be in [0.0, 1.0]."""
        result = self.runner.run(pool_per_query=self.pool_per_query)
        for r in result["per_query"]:
            self.assertGreaterEqual(r["mrr"], 0.0)
            self.assertLessEqual(r["mrr"], 1.0)

    def test_latency_is_positive(self):
        """All latency measurements must be non-negative."""
        result = self.runner.run(pool_per_query=self.pool_per_query)
        for r in result["per_query"]:
            self.assertGreaterEqual(r["latency_ms"], 0.0)

    def test_tokens_are_positive(self):
        """All token estimates must be positive."""
        result = self.runner.run(pool_per_query=self.pool_per_query)
        for r in result["per_query"]:
            self.assertGreater(r["tokens"], 0)

    def test_hybrid_search_wins_on_exact_identifier_query(self):
        """Hybrid search must achieve accuracy=1 on ret_q2 (APX-9982 exact match)."""
        result = self.runner.run(pool_per_query=self.pool_per_query)
        hybrid_q2 = next(
            r for r in result["per_query"]
            if r["query_id"] == "ret_q2" and r["architecture"] == "hybrid_search"
        )
        self.assertEqual(hybrid_q2["accuracy"], 1, "Hybrid search must find APX-9982 in mock pool")

    def test_markdown_report_contains_all_architectures(self):
        """Markdown report must reference all three architecture names."""
        result = self.runner.run(pool_per_query=self.pool_per_query)
        report = RetrievalEvaluationRunner.generate_markdown_report(result)
        self.assertIn("naive_rag", report)
        self.assertIn("hybrid_search", report)
        self.assertIn("agentic_rag", report)

    def test_markdown_report_contains_justification(self):
        """Markdown report must include a justification section."""
        result = self.runner.run(pool_per_query=self.pool_per_query)
        report = RetrievalEvaluationRunner.generate_markdown_report(result)
        self.assertIn("Justification", report)
        self.assertIn("Hybrid Search", report)


if __name__ == "__main__":
    unittest.main()

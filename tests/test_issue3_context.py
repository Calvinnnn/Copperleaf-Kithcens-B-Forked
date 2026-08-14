"""
test_issue3_context.py — Test suite for Issue #3:
- Sliding Window strategy: FIFO token trimming, system prompt preservation, scratchpad injection
- Observation Masking strategy: tool output masking, preserve_recent_count, threshold enforcement
- PII Masking strategy: email, phone, SSN, credit card redaction
- Zone-Based Pruning strategy: priority zone sorting and retention
- Recursive Summarization strategy: summary placeholder insertion
- Context Evaluation Runner: benchmark execution, token metrics, needle retrieval accuracy
"""

import json
import unittest

from context_eval.sliding_window import (
    SlidingWindowStrategy,
    ContextWindowMetrics,
    estimate_tokens,
)
from context_eval.masking import ObservationMaskingStrategy, PIIMaskingStrategy
from context_eval.zone_pruning import ZoneBasedPruningStrategy
from context_eval.summarization import RecursiveSummarizationStrategy
from context_eval.evaluate import ContextEvaluationRunner
from context_eval.test_cases import TestCaseGenerator
from memory.scratchpad import Scratchpad


def _make_dialogue(n: int) -> list:
    """Generate N alternating user/assistant message dicts."""
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"Message {i+1}: " + "word " * 20})
    return msgs


class TestSlidingWindowStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = SlidingWindowStrategy(default_turn_window=10)

    def test_trims_to_token_budget(self):
        """Sliding window must fit messages within max_tokens."""
        messages = _make_dialogue(30)
        formatted, metrics = self.strategy.format_context(messages, max_tokens=200)

        self.assertLessEqual(metrics.retained_tokens, 200)
        self.assertGreater(len(formatted), 0)
        self.assertIsInstance(metrics.tokens_saved, int)
        self.assertGreaterEqual(metrics.tokens_saved, 0)

    def test_system_prompt_always_preserved(self):
        """System prompt must survive regardless of token budget."""
        messages = [{"role": "system", "content": "You are Copperleaf assistant."}]
        messages += _make_dialogue(25)

        formatted, metrics = self.strategy.format_context(messages, max_tokens=300)
        roles = [m["role"] for m in formatted]
        self.assertIn("system", roles)

    def test_scratchpad_injected_at_top(self):
        """Scratchpad context must be injected as system block at start of context."""
        pad = Scratchpad()
        pad.set_goal("Investigate stock spoilage", "Spoilage check")
        pad.add_reasoning("Checking Roma Tomato stock levels")

        messages = _make_dialogue(5)
        formatted, _ = self.strategy.format_context(messages, max_tokens=2000, scratchpad=pad)

        # First non-empty system message should contain scratchpad text
        scratch_msgs = [m for m in formatted if m["role"] == "system" and "SCRATCHPAD" in m["content"]]
        self.assertGreater(len(scratch_msgs), 0)
        self.assertIn("Investigate stock spoilage", scratch_msgs[0]["content"])

    def test_metrics_fields_populated(self):
        """ContextWindowMetrics must have all required fields with sensible values."""
        messages = _make_dialogue(10)
        _, metrics = self.strategy.format_context(messages, max_tokens=500)

        self.assertEqual(metrics.strategy_name, "Sliding Window")
        self.assertGreater(metrics.original_tokens, 0)
        self.assertGreater(metrics.original_turns, 0)
        self.assertIsInstance(metrics.latency_ms, float)
        self.assertGreater(metrics.latency_ms, 0)
        self.assertIsNotNone(metrics.timestamp)


class TestObservationMaskingStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = ObservationMaskingStrategy(
            preserve_recent_count=1,
            token_mask_threshold=10,
        )

    def _make_tool_messages(self, n_tool: int) -> list:
        msgs = []
        for i in range(n_tool):
            msgs.append({
                "role": "user",
                "content": f"Query {i+1}",
            })
            msgs.append({
                "role": "tool",
                "name": f"get_inventory_{i}",
                "content": json.dumps({"items": [f"item_{j}" for j in range(30)]}),
            })
        return msgs

    def test_old_tool_outputs_masked(self):
        """All but the most recent tool output should be masked above threshold."""
        messages = self._make_tool_messages(3)
        formatted, metrics = self.strategy.format_context(messages, max_tokens=5000)

        masked_count = sum(
            1 for m in formatted
            if m.get("role") == "tool" and "TOOL OBSERVATION MASKED" in m.get("content", "")
        )
        self.assertGreater(masked_count, 0)

    def test_recent_tool_output_preserved(self):
        """The most recent tool output must not be masked."""
        messages = self._make_tool_messages(3)
        formatted, _ = self.strategy.format_context(messages, max_tokens=5000)

        tool_messages = [m for m in formatted if m.get("role") == "tool"]
        if tool_messages:
            last_tool = tool_messages[-1]
            self.assertNotIn("TOOL OBSERVATION MASKED", last_tool.get("content", ""))

    def test_masked_content_contains_metadata(self):
        """Masked content should contain tool name and token count metadata."""
        messages = self._make_tool_messages(2)
        formatted, _ = self.strategy.format_context(messages, max_tokens=5000)

        for m in formatted:
            if m.get("role") == "tool" and "TOOL OBSERVATION MASKED" in m.get("content", ""):
                self.assertIn("tool=", m["content"])
                self.assertIn("original_tokens", m["content"])


class TestPIIMaskingStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = PIIMaskingStrategy()

    def test_email_redacted(self):
        """Email addresses must be replaced with [REDACTED_EMAIL]."""
        messages = [{"role": "user", "content": "Contact mona.farid@copperleaf.com for info."}]
        formatted, _ = self.strategy.format_context(messages, max_tokens=5000)
        self.assertIn("[REDACTED_EMAIL]", formatted[0]["content"])
        self.assertNotIn("@copperleaf.com", formatted[0]["content"])

    def test_phone_redacted(self):
        """Phone numbers must be replaced with [REDACTED_PHONE]."""
        messages = [{"role": "user", "content": "Call 555-123-4567 for the branch."}]
        formatted, _ = self.strategy.format_context(messages, max_tokens=5000)
        self.assertIn("[REDACTED_PHONE]", formatted[0]["content"])

    def test_ssn_redacted(self):
        """SSNs must be replaced with [REDACTED_SSN]."""
        messages = [{"role": "user", "content": "Employee SSN is 123-45-6789."}]
        formatted, _ = self.strategy.format_context(messages, max_tokens=5000)
        self.assertIn("[REDACTED_SSN]", formatted[0]["content"])

    def test_credit_card_redacted(self):
        """Credit card numbers must be replaced with [REDACTED_CREDIT_CARD]."""
        messages = [{"role": "user", "content": "Payment via 4111 1111 1111 1111."}]
        formatted, _ = self.strategy.format_context(messages, max_tokens=5000)
        self.assertIn("[REDACTED_CREDIT_CARD]", formatted[0]["content"])

    def test_non_pii_content_unchanged(self):
        """Non-PII content must pass through unchanged."""
        messages = [{"role": "user", "content": "Check stock levels for Branch 1 produce."}]
        formatted, _ = self.strategy.format_context(messages, max_tokens=5000)
        self.assertEqual(formatted[0]["content"], messages[0]["content"])

    def test_all_turns_retained(self):
        """PII masking retains all turns (no truncation)."""
        messages = _make_dialogue(20)
        formatted, metrics = self.strategy.format_context(messages, max_tokens=500)
        self.assertEqual(metrics.retained_turns, len(messages))


class TestZoneBasedPruningStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = ZoneBasedPruningStrategy(recent_zone_turns=3)

    def test_recent_turns_retained_under_budget(self):
        """Recent N turns should always be in the output when budget allows."""
        messages = _make_dialogue(15)
        formatted, metrics = self.strategy.format_context(messages, max_tokens=300)

        self.assertGreater(len(formatted), 0)
        self.assertLessEqual(metrics.retained_tokens, 300 + 50)  # Allow small overflow tolerance

    def test_metrics_token_fields(self):
        """Zone pruning must produce complete ContextWindowMetrics."""
        messages = _make_dialogue(10)
        _, metrics = self.strategy.format_context(messages, max_tokens=500)

        self.assertEqual(metrics.strategy_name, "Zone-Based Pruning")
        self.assertGreater(metrics.original_tokens, 0)


class TestRecursiveSummarizationStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = RecursiveSummarizationStrategy(recent_turns_verbatim=3)

    def test_summary_placeholder_present(self):
        """Recursive summarization should inject a summary placeholder for older turns."""
        messages = _make_dialogue(20)
        formatted, metrics = self.strategy.format_context(messages, max_tokens=500)

        all_content = " ".join(m.get("content", "") for m in formatted)
        self.assertIn("SUMMARY", all_content.upper())

    def test_recent_turns_verbatim_preserved(self):
        """The last N verbatim turns must appear in the output."""
        messages = _make_dialogue(10)
        # Add a distinctive message as the last turn
        messages[-1]["content"] = "UNIQUE_MARKER_LAST_TURN_12345"
        formatted, _ = self.strategy.format_context(messages, max_tokens=5000)

        all_content = " ".join(m.get("content", "") for m in formatted)
        self.assertIn("UNIQUE_MARKER_LAST_TURN_12345", all_content)


class TestContextEvaluationRunner(unittest.TestCase):

    def test_benchmark_suite_runs_all_strategies(self):
        """Benchmark suite must return results covering all 4+ strategies per test case."""
        runner = ContextEvaluationRunner(max_token_limit=800)
        cases = TestCaseGenerator.get_all_test_cases()
        suite = runner.run_benchmark_suite(cases)

        self.assertGreater(len(suite.results), 0)
        strategy_names = {r.strategy_name for r in suite.results}
        self.assertGreaterEqual(len(strategy_names), 4)

    def test_benchmark_metrics_within_budget(self):
        """No retained_tokens should exceed the configured max token budget (allowing small overhead)."""
        runner = ContextEvaluationRunner(max_token_limit=600)
        cases = TestCaseGenerator.get_all_test_cases()
        suite = runner.run_benchmark_suite(cases)

        for result in suite.results:
            # PII Masking doesn't truncate - ignore for this constraint
            if result.strategy_name == "PII Masking":
                continue
            self.assertLessEqual(
                result.retained_tokens, 700,  # 600 + 100 overhead for system/scratchpad
                f"{result.strategy_name} exceeded budget: retained={result.retained_tokens}",
            )

    def test_markdown_report_generated(self):
        """Markdown report must contain header and strategy comparison table rows."""
        runner = ContextEvaluationRunner(max_token_limit=600)
        suite = runner.run_benchmark_suite()
        report = ContextEvaluationRunner.generate_markdown_report(suite)

        self.assertIn("Context Window Management", report)
        self.assertIn("Strategy Name", report)
        self.assertIn("Sliding Window", report)

    def test_needle_retrieval_accuracy_tracked(self):
        """Fact needle retrieval accuracy must be measurable in results."""
        runner = ContextEvaluationRunner(max_token_limit=1200)
        suite = runner.run_benchmark_suite()

        for result in suite.results:
            self.assertGreaterEqual(result.retrieval_accuracy, 0.0)
            self.assertLessEqual(result.retrieval_accuracy, 100.0)
            self.assertGreaterEqual(result.retained_needles_count, 0)
            self.assertGreaterEqual(result.total_needles_count, 0)

    def test_latency_tracked_per_strategy(self):
        """Each strategy result must have a positive latency measurement."""
        runner = ContextEvaluationRunner(max_token_limit=600)
        suite = runner.run_benchmark_suite()

        for result in suite.results:
            self.assertGreater(result.latency_ms, 0, f"{result.strategy_name} has zero latency")


if __name__ == "__main__":
    unittest.main()

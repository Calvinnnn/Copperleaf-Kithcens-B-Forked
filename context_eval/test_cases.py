"""Evaluation Dataset Generator Module for AI Agent Context Architecture.

This module generates realistic, multi-turn, long-context evaluation test suites designed to
stress-test Context Window Management strategies.

Test cases simulate real Copperleaf Kitchens inventory workflows and contain:
1. High turn counts (20 to 50+ dialogue turns).
2. Heavy tool call volume with large observation payloads (SQL tables, JSON lists).
3. Buried 'Needle in a Haystack' facts (user preferences, critical instructions, policy overrides)
   embedded deep within historical turns to measure information retrieval accuracy.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, List, Optional

from context_eval.sliding_window import estimate_tokens


@dataclass
class NeedleFact:
    """Represents a specific critical fact buried inside a long dialogue dataset.

    Attributes:
        fact_key: Unique identifier key for the fact.
        fact_value: Authoritative ground-truth fact value.
        turn_index: Turn position index where this fact was originally injected.
        verification_question: Question designed to test if context strategy retained this fact.
        expected_answer: Ground-truth answer expected from LLM or evaluator.
    """

    fact_key: str
    fact_value: str
    turn_index: int
    verification_question: str
    expected_answer: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize needle fact to dictionary."""
        return {
            "fact_key": self.fact_key,
            "fact_value": self.fact_value,
            "turn_index": self.turn_index,
            "verification_question": self.verification_question,
            "expected_answer": self.expected_answer,
        }


@dataclass
class EvaluationTestCase:
    """Represents a complete evaluation test dataset scenario.

    Attributes:
        test_id: Unique test scenario identifier.
        name: Human-readable benchmark name.
        description: Summary of workflow and benchmark objective.
        messages: Full un-trimmed turn message array (role, content, tool_calls, etc.).
        buried_facts: List of buried NeedleFact instances to test retrieval accuracy.
        total_turns: Count of message turns.
        total_tokens: Estimated total un-trimmed token count.
        timestamp: ISO 8601 UTC creation timestamp.
    """

    test_id: str
    name: str
    description: str
    messages: List[dict[str, Any]]
    buried_facts: List[NeedleFact] = field(default_factory=list)
    total_turns: int = 0
    total_tokens: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        """Compute message count and total un-trimmed tokens automatically."""
        self.total_turns = len(self.messages)
        self.total_tokens = sum(estimate_tokens(m.get("content", "")) for m in self.messages)

    def to_dict(self) -> dict[str, Any]:
        """Serialize evaluation scenario to dictionary."""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "total_turns": self.total_turns,
            "total_tokens": self.total_tokens,
            "messages": self.messages,
            "buried_facts": [fact.to_dict() for fact in self.buried_facts],
            "timestamp": self.timestamp,
        }


class TestCaseGenerator:
    """Generator constructing realistic, production-scale evaluation datasets."""

    @staticmethod
    def generate_inventory_investigation_suite() -> EvaluationTestCase:
        """Scenario 1: Detailed waste investigation with buried manager approval threshold."""
        messages: List[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are Copperleaf Kitchens inventory assistant. "
                    "Follow domain policy rules in copperleaf://policy/waste_management."
                ),
            },
            {
                "role": "user",
                "content": "Hi, I am Manager Mona. I want to check stock at Branch 1.",
            },
            {
                "role": "assistant",
                "content": "Checking stock for Branch 1...",
                "tool_calls": [{"name": "get_inventory", "args": {"branch_id": 1}}],
            },
            {
                "role": "tool",
                "name": "get_inventory",
                "content": json.dumps([
                    {"item_id": 1, "name": "Roma Tomatoes", "unit_cost": 2.50, "current_quantity": 45.0},
                    {"item_id": 2, "name": "Extra Virgin Olive Oil", "unit_cost": 18.00, "current_quantity": 12.0},
                    {"item_id": 10, "name": "Wagyu Beef Ribeye", "unit_cost": 85.00, "current_quantity": 8.0},
                ]),
            },
            {
                "role": "user",
                "content": (
                    "Please note: My preferred fallback delivery supplier for Branch 1 "
                    "is 'Apex Fresh Logistics' (Account #APX-9982). Always use this supplier."
                ),
            },
            {
                "role": "assistant",
                "content": "Understood. I have recorded Apex Fresh Logistics (Account #APX-9982) as your preferred supplier.",
            },
        ]

        # Generate large tool noise turns to inflate context
        for i in range(1, 15):
            messages.append({
                "role": "user",
                "content": f"Querying transaction batch #{i} for general historical review...",
            })
            messages.append({
                "role": "assistant",
                "content": f"Fetching batch {i} transactions...",
                "tool_calls": [{"name": "get_transaction_history", "args": {"item_id": i, "limit": 20}}],
            })
            messages.append({
                "role": "tool",
                "name": "get_transaction_history",
                "content": json.dumps([
                    {
                        "transaction_id": 100 + i * 2,
                        "change_type": "usage",
                        "quantity_change": -3.5,
                        "reason": f"routine prep batch {i}",
                        "created_at": f"2026-07-{i:02d}T10:00:00Z",
                    },
                    {
                        "transaction_id": 101 + i * 2,
                        "change_type": "restock",
                        "quantity_change": 10.0,
                        "reason": "weekly delivery",
                        "created_at": f"2026-07-{i:02d}T08:00:00Z",
                    },
                ]),
            })

        # Add buried needle verification prompt
        messages.append({
            "role": "user",
            "content": "Which preferred supplier account did I specify earlier for Branch 1?",
        })

        buried_facts = [
            NeedleFact(
                fact_key="preferred_supplier_account",
                fact_value="Apex Fresh Logistics (Account #APX-9982)",
                turn_index=4,
                verification_question="Which preferred supplier account did I specify earlier for Branch 1?",
                expected_answer="Apex Fresh Logistics (Account #APX-9982)",
            )
        ]

        return EvaluationTestCase(
            test_id="eval_suite_01_inventory",
            name="Inventory Waste Investigation Benchmark",
            description="Multi-turn inventory query dataset with buried supplier preference needle.",
            messages=messages,
            buried_facts=buried_facts,
        )

    @staticmethod
    def generate_large_scale_suite(turns_count: int = 50) -> EvaluationTestCase:
        """Scenario 2: Extreme scale dialogue dataset containing 50+ turns and buried instructions."""
        messages: List[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are Copperleaf Assistant running extreme scale benchmark.",
            },
            {
                "role": "user",
                "content": "CRITICAL INSTRUCTION: Set maximum allowable single write-off override ceiling to $750.00.",
            },
            {
                "role": "assistant",
                "content": "Acknowledged. Overridden write-off ceiling set to $750.00.",
            },
        ]

        for step in range(1, turns_count // 3):
            messages.append({
                "role": "user",
                "content": f"Executing automated branch inspection step #{step}...",
            })
            messages.append({
                "role": "assistant",
                "content": f"Inspecting branch status for step {step}...",
            })
            messages.append({
                "role": "tool",
                "name": "get_low_stock_items",
                "content": json.dumps({
                    "step": step,
                    "status": "normal",
                    "items_flagged": ["Item A", "Item B"],
                    "data_payload": "X" * 150,  # Token filler
                }),
            })

        messages.append({
            "role": "user",
            "content": "What is the maximum allowable single write-off override ceiling I specified?",
        })

        buried_facts = [
            NeedleFact(
                fact_key="write_off_override_ceiling",
                fact_value="$750.00",
                turn_index=1,
                verification_question="What is the maximum allowable single write-off override ceiling I specified?",
                expected_answer="$750.00",
            )
        ]

        return EvaluationTestCase(
            test_id="eval_suite_02_scale",
            name="50-Turn Extreme Scale Benchmark",
            description="Stress test dataset designed to evaluate token reduction and needle retention at scale.",
            messages=messages,
            buried_facts=buried_facts,
        )

    @classmethod
    def get_all_test_cases(cls) -> List[EvaluationTestCase]:
        """Return array of all available evaluation test suites."""
        return [
            cls.generate_inventory_investigation_suite(),
            cls.generate_large_scale_suite(turns_count=45),
        ]

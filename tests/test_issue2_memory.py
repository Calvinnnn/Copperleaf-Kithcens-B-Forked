"""
test_issue2_memory.py — Test suite for Issue #2:
- Short-Term Memory rolling FIFO buffer & overflow callback
- Scratchpad Memory resilience to pruning
- Promote-or-Drop Router heuristics (forget vs promote) & SQLite audit log
- Episodic Memory persistent event store
- Semantic Consolidation Engine: triple extraction, versioning, TTL expiration, and contradiction handling
"""

from datetime import datetime, timedelta, timezone
import sqlite3
import unittest

from mcp_server.init_db import build as build_db
from mcp_server.db import get_connection
from memory.short_term import ShortTermMemory, ShortTermMemoryItem, MessageRole
from memory.scratchpad import Scratchpad, StepStatus, ExecutionStatus
from memory.episodic import EpisodicMemory, EventType
from memory.semantic import SemanticMemory, FactStatus
from memory.router import PromoteOrDropRouter, RoutingAction
from memory.consolidation import SemanticConsolidationEngine, ConflictResolutionStrategy


class TestIssue2MemorySystem(unittest.TestCase):

    def setUp(self):
        """Build fresh database and memory instances for each test."""
        build_db()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.router = PromoteOrDropRouter(episodic_memory=self.episodic)
        self.consolidation_engine = SemanticConsolidationEngine(
            episodic_memory=self.episodic,
            semantic_memory=self.semantic,
            default_strategy=ConflictResolutionStrategy.SUPERSEDE,
        )

    def test_short_term_memory_rolling_fifo(self):
        """Test rolling FIFO capacity ceiling and overflow callback triggering."""
        evicted_items = []

        def overflow_cb(items):
            evicted_items.extend(items)

        stm = ShortTermMemory(capacity=3, overflow_callback=overflow_cb)

        stm.add_user_message("Msg 1")
        stm.add_user_message("Msg 2")
        stm.add_user_message("Msg 3")
        self.assertEqual(stm.size, 3)
        self.assertEqual(len(evicted_items), 0)

        # 4th message triggers eviction of Msg 1
        stm.add_user_message("Msg 4")
        self.assertEqual(stm.size, 3)
        self.assertEqual(len(evicted_items), 1)
        self.assertEqual(evicted_items[0].content, "Msg 1")
        self.assertEqual(stm[0].content, "Msg 2")
        self.assertEqual(stm[2].content, "Msg 4")

    def test_scratchpad_resilience_to_pruning(self):
        """Test Scratchpad working memory state remains intact when short-term transcript is pruned."""
        scratchpad = Scratchpad()
        scratchpad.set_goal("Investigate Inventory Spoilage", "Branch 1 waste check")
        scratchpad.set_plan(["Check stock levels", "Identify expired items", "File report"])
        scratchpad.add_reasoning("Noticed high tomato loss rate.")
        scratchpad.set_variable("investigated_branch", 1)

        # Simulate transcript clearing / short-term memory reset
        stm = ShortTermMemory(capacity=5)
        stm.add_user_message("Hello")
        stm.clear()

        # Scratchpad must retain full operational state
        self.assertEqual(scratchpad.goal, "Investigate Inventory Spoilage")
        self.assertEqual(len(scratchpad.plan), 3)
        self.assertEqual(scratchpad.reasoning_chain[0], "Noticed high tomato loss rate.")
        self.assertEqual(scratchpad.get_variable("investigated_branch"), 1)

    def test_promote_or_drop_router_heuristics_and_db_audit(self):
        """Test router heuristics for FORGET vs PROMOTE and check SQLite audit log."""
        evicted_user_noise = ShortTermMemoryItem(role=MessageRole.USER, content="Hello, good morning!")
        evicted_preference = ShortTermMemoryItem(role=MessageRole.USER, content="We always prefer GreenRoute Wholesale for emergency produce.")
        evicted_tool_call = ShortTermMemoryItem(role=MessageRole.TOOL, name="write_off_inventory", content='{"status": "success", "item_id": 10}')

        # 1. Noise message should be forgotten
        dec_noise = self.router.evaluate(evicted_user_noise)
        self.assertEqual(dec_noise.action, RoutingAction.FORGET)
        self.assertIn("conversational", dec_noise.reason.lower())

        # 2. Preference message should be promoted to Episodic
        dec_pref = self.router.evaluate(evicted_preference)
        self.assertEqual(dec_pref.action, RoutingAction.PROMOTE)
        self.assertIsNotNone(dec_pref.promoted_event_id)

        # 3. Tool write-off observation should be promoted
        dec_tool = self.router.evaluate(evicted_tool_call)
        self.assertEqual(dec_tool.action, RoutingAction.PROMOTE)
        self.assertIsNotNone(dec_tool.promoted_event_id)

        # Verify NO DIRECT WRITES to Semantic Memory were made by the router
        self.assertEqual(self.semantic.total_facts_count, 0)

        # Verify router decision audit log in SQLite router_decisions table
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM router_decisions").fetchall()
            self.assertGreaterEqual(len(rows), 2)

    def test_episodic_memory_store_and_query(self):
        """Test storing and querying persistent episodic events in SQLite."""
        event = self.episodic.store_event(
            event_type=EventType.PREFERENCE,
            summary="Branch 1 emergency supplier set to GreenRoute Wholesale",
            details={"supplier": "GreenRoute Wholesale", "branch_id": 1},
            importance_score=0.9,
            tags=["supplier", "branch_1"],
        )

        self.assertIsNotNone(event.event_id)
        queried = self.episodic.query(tag="supplier")
        self.assertEqual(len(queried), 1)
        self.assertEqual(queried[0].summary, event.summary)

        # Verify SQLite persistence
        with get_connection() as conn:
            row = conn.execute("SELECT * FROM episodic_events WHERE event_id = ?", (event.event_id,)).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["event_type"], "preference")

    def test_semantic_consolidation_contradiction_and_versioning(self):
        """Test semantic consolidation: fact creation, versioning, conflict resolution, and TTL expiration."""
        # 1. Create initial episodic event (old preference)
        evt1 = self.episodic.store_event(
            event_type=EventType.PREFERENCE,
            summary="Preferred supplier: Apex Fresh Logistics",
            details={"subject": "branch_1", "preferred_supplier": "Apex Fresh Logistics"},
            importance_score=0.8,
        )

        # Run 1st consolidation pass -> Creates initial semantic fact v1
        res1 = self.consolidation_engine.run_consolidation()
        self.assertEqual(res1.created_facts_count, 1)

        active_fact = self.semantic.get_active_fact("branch_1", "preferred_supplier")
        self.assertIsNotNone(active_fact)
        self.assertEqual(active_fact.value, "Apex Fresh Logistics")
        self.assertEqual(active_fact.version, 1)
        self.assertEqual(active_fact.status, FactStatus.ACTIVE)

        # 2. Create conflicting episodic event (corporate override)
        evt2 = self.episodic.store_event(
            event_type=EventType.DECISION,
            summary="Preferred supplier: GreenRoute Wholesale",
            details={"subject": "branch_1", "preferred_supplier": "GreenRoute Wholesale"},
            importance_score=0.95,
        )

        # Run 2nd consolidation pass -> Conflict detected! SUPERSEDE policy updates value and increments version to 2
        res2 = self.consolidation_engine.run_consolidation()
        self.assertEqual(res2.updated_facts_count, 1)

        updated_fact = self.semantic.get_active_fact("branch_1", "preferred_supplier")
        self.assertEqual(updated_fact.value, "GreenRoute Wholesale")
        self.assertEqual(updated_fact.version, 2)
        self.assertEqual(len(updated_fact.history), 1)
        self.assertEqual(updated_fact.history[0].value, "Apex Fresh Logistics")
        self.assertIn("SUPERSEDED", updated_fact.history[0].reason)

    def test_semantic_fact_ttl_expiration(self):
        """Test automatic expiration of active semantic facts past valid_until timestamp."""
        # Add fact with past expiration date
        past_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        fact = self.semantic.add_fact(
            subject="branch_1",
            predicate="temporary_override",
            value="active_lockout",
            valid_until=past_time,
        )
        self.assertEqual(fact.status, FactStatus.ACTIVE)

        # Run consolidation pass -> Pre-expiration pass marks fact as EXPIRED
        self.consolidation_engine.run_consolidation()

        expired_fact = self.semantic.get_fact(fact.fact_id)
        self.assertEqual(expired_fact.status, FactStatus.EXPIRED)


if __name__ == "__main__":
    unittest.main()

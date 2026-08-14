"""
test_issue1_db_mcp_agent.py — Test suite for Issue #1:
- Database schema & migration integrity
- DB layer connection helpers & atomic write transactions
- MCP server tool validation and role/branch authorization
- Agent core memory initialization and message handling
"""

import os
import sqlite3
import unittest

from mcp_server.init_db import build as build_db, DB_PATH
from mcp_server.auth import Session, resolve_staff, AuthError
from mcp_server.db import get_connection, get_write_connection
from mcp_server.validation import validate_write_off, validate_date_range, ValidationError
import mcp_server.tools as mcp_tools
from agent.agent import MemoryEnabledAgent


class TestIssue1DBMCPAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Ensure database is freshly initialized before running tests."""
        build_db()

    def test_database_tables_exist(self):
        """Verify that all core and memory tables exist in copperleaf.db."""
        expected_tables = {
            "branches",
            "staff",
            "suppliers",
            "inventory_items",
            "inventory_transactions",
            "supplier_orders",
            "episodic_events",
            "semantic_facts",
            "router_decisions",
        }
        with get_connection() as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            existing_tables = {row["name"] for row in rows}

        for table in expected_tables:
            self.assertIn(table, existing_tables, f"Table '{table}' missing from database schema.")

    def test_auth_resolution(self):
        """Test session resolution by API token."""
        # Valid manager token from seed.sql
        session = resolve_staff("tok_mona_mgr_9f2a")
        self.assertEqual(session.full_name, "Mona Farid")
        self.assertEqual(session.role, "manager")
        self.assertEqual(session.branch_id, 1)

        # Valid staff token
        session_staff = resolve_staff("tok_youssef_stf_c71b")
        self.assertEqual(session_staff.role, "staff")

        # Invalid token
        with self.assertRaises(AuthError):
            resolve_staff("invalid_token_123")

    def test_mcp_read_tools(self):
        """Test read-only inventory and supplier order tools."""
        session = resolve_staff("tok_mona_mgr_9f2a")

        # Get inventory for branch 1
        inventory = mcp_tools.get_inventory(session, branch_id=1)
        self.assertIsInstance(inventory, list)
        self.assertGreater(len(inventory), 0)

        # Search partial item name
        tomatoes = mcp_tools.get_inventory(session, branch_id=1, item_name="Roma")
        self.assertGreater(len(tomatoes), 0)
        self.assertIn("Roma", tomatoes[0]["name"])

        # Low stock items
        low_stock = mcp_tools.get_low_stock_items(session, branch_id=1)
        self.assertIsInstance(low_stock, list)

        # Supplier orders
        orders = mcp_tools.get_supplier_orders(session, branch_id=1)
        self.assertIsInstance(orders, list)

    def test_mcp_write_off_authorization_and_validation(self):
        """Test role authorization and server-side validation on write-off tool."""
        manager_session = resolve_staff("tok_mona_mgr_9f2a") # Branch 1 manager
        staff_session = resolve_staff("tok_youssef_stf_c71b") # Branch 1 staff

        # 1. Staff role rejected
        with self.assertRaises(mcp_tools.AuthorizationError) as cm:
            mcp_tools.write_off_inventory(staff_session, item_id=1, quantity=1.0, reason="spoiled_before_use")
        self.assertIn("only managers", str(cm.exception))

        # 2. Invalid quantity rejected by validation
        with self.assertRaises(mcp_tools.ToolError) as cm:
            mcp_tools.write_off_inventory(manager_session, item_id=1, quantity=-5.0, reason="spoiled_before_use")
        self.assertIn("positive number", str(cm.exception))

        # 3. Invalid write-off reason rejected
        with self.assertRaises(mcp_tools.ToolError) as cm:
            mcp_tools.write_off_inventory(manager_session, item_id=1, quantity=1.0, reason="invalid_reason")
        self.assertIn("not recognized", str(cm.exception))

    def test_atomic_write_off_inventory(self):
        """Test atomic inventory write-off updates balance and transaction log simultaneously."""
        manager_session = resolve_staff("tok_mona_mgr_9f2a")
        item_id = 1 # Roma Tomatoes

        # Fetch initial quantity
        initial_inv = mcp_tools.get_inventory(manager_session, branch_id=1, item_name="Roma")[0]
        initial_qty = initial_inv["current_quantity"]

        # Execute write-off
        result = mcp_tools.write_off_inventory(manager_session, item_id=item_id, quantity=2.0, reason="spoiled_before_use")

        self.assertEqual(result["item_id"], item_id)
        self.assertEqual(result["quantity_written_off"], 2.0)
        self.assertEqual(result["new_stock_level"], initial_qty - 2.0)

        # Verify transaction record inserted
        txns = mcp_tools.get_transaction_history(manager_session, item_id=item_id, limit=1)
        self.assertGreater(len(txns), 0)
        self.assertEqual(txns[0]["change_type"], "write_off")
        self.assertEqual(txns[0]["quantity_change"], -2.0)

    def test_agent_core_memory_wiring(self):
        """Test MemoryEnabledAgent initialization and turn receiving."""
        agent = MemoryEnabledAgent(stm_capacity=5, consolidation_batch_size=3)

        agent.receive_message("My name is Mona Farid. I manage Branch 1.", role="user")
        agent.receive_message("Hello Mona! I can assist with Branch 1 operations.", role="assistant")

        self.assertEqual(agent.short_term.size, 2)
        self.assertEqual(agent.short_term[0].content, "My name is Mona Farid. I manage Branch 1.")


if __name__ == "__main__":
    unittest.main()

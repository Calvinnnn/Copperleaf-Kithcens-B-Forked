"""
test_mcp_agent_integration.py — Test suite for the MCP integration fix in
MemoryEnabledAgent (agent/agent.py).

Verifies that the agent REUSES the existing mcp_server module directly:
- mcp_server.tools / mcp_server.auth are imported (not duplicated)
- The authenticated session resolves through mcp_server.auth.resolve_staff
- call_mcp_tool() dispatches to the exact tool functions mcp_server/server.py
  registers on its FastMCP instance
- The existing FastMCP server instance is lazily loaded, guarded against the
  module-level sys.exit(1) that mcp_server/server.py performs without a valid
  COPPERLEAF_API_TOKEN
- A missing api_token degrades gracefully instead of killing the process
"""

import os
import unittest

from agent.agent import MemoryEnabledAgent, _MCP_SERVER_AVAILABLE
from mcp_server.auth import AuthError, resolve_staff
from mcp_server.init_db import build as build_db
import mcp_server.tools as mcp_tools

VALID_TOKEN = "tok_mona_mgr_9f2a"  # Mona Farid — Branch 1 manager


class TestMCPAgentIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        build_db()

    def setUp(self):
        # Isolate the lazy FastMCP server test from any pre-existing env value
        # so mcp_server/server.py's module-level auth always sees a valid token.
        self._old_token = os.environ.get("COPPERLEAF_API_TOKEN")
        os.environ["COPPERLEAF_API_TOKEN"] = VALID_TOKEN

    def tearDown(self):
        if self._old_token is None:
            os.environ.pop("COPPERLEAF_API_TOKEN", None)
        else:
            os.environ["COPPERLEAF_API_TOKEN"] = self._old_token

    # ------------------------------------------------------------------
    # 1. The agent visibly reuses the existing mcp_server module
    # ------------------------------------------------------------------

    def test_mcp_server_imports_available(self):
        """The agent must import the existing mcp_server tool/auth layer."""
        self.assertTrue(_MCP_SERVER_AVAILABLE)
        self.assertIsNotNone(mcp_tools)
        for tool in (
            "get_inventory",
            "get_low_stock_items",
            "get_supplier_orders",
            "get_transaction_history",
            "write_off_inventory",
        ):
            self.assertTrue(
                callable(getattr(mcp_tools, tool, None)),
                f"mcp_tools.{tool} should be importable from the existing module",
            )

    # ------------------------------------------------------------------
    # 2. Session resolution reuses mcp_server.auth.resolve_staff
    # ------------------------------------------------------------------

    def test_agent_mcp_session_resolution(self):
        """agent.mcp_session resolves through the existing auth layer."""
        agent = MemoryEnabledAgent(api_token=VALID_TOKEN)
        session = agent.mcp_session
        self.assertEqual(session.full_name, "Mona Farid")
        self.assertEqual(session.role, "manager")
        self.assertEqual(session.branch_id, 1)
        self.assertIs(agent.mcp_session, session)  # cached after first resolution

    def test_agent_invalid_token_raises_auth_error(self):
        """An invalid api_token surfaces as the existing AuthError."""
        agent = MemoryEnabledAgent(api_token="invalid_token_123")
        with self.assertRaises(AuthError):
            _ = agent.mcp_session

    # ------------------------------------------------------------------
    # 3. call_mcp_tool dispatches to the existing mcp_server tool functions
    # ------------------------------------------------------------------

    def test_agent_call_mcp_tool_read_tools(self):
        """Operational reads are served by the existing mcp_server.tools."""
        agent = MemoryEnabledAgent(api_token=VALID_TOKEN)

        inventory = agent.call_mcp_tool("get_inventory", branch_id=1, item_name="Roma")
        self.assertIsInstance(inventory, list)
        self.assertGreater(len(inventory), 0)
        self.assertIn("Roma", inventory[0]["name"])

        low_stock = agent.call_mcp_tool("get_low_stock_items", branch_id=1)
        self.assertIsInstance(low_stock, list)

        orders = agent.call_mcp_tool("get_supplier_orders", branch_id=1)
        self.assertIsInstance(orders, list)

    def test_agent_call_mcp_tool_unknown_tool_raises(self):
        agent = MemoryEnabledAgent(api_token=VALID_TOKEN)
        with self.assertRaises(ValueError):
            agent.call_mcp_tool("does_not_exist")

    # ------------------------------------------------------------------
    # 4. The existing FastMCP server instance is reused (lazy, guarded)
    # ------------------------------------------------------------------

    def test_agent_reuses_existing_fastmcp_server(self):
        """agent.mcp_server returns the exact FastMCP instance from
        mcp_server/server.py — not a duplicate or a new server."""
        agent = MemoryEnabledAgent(api_token=VALID_TOKEN)
        server = agent.mcp_server
        self.assertEqual(server.name, "copperleaf-kitchens")
        self.assertIs(agent.mcp_server, server)  # cached
        # The same instance the standalone MCP server module owns
        from mcp_server.server import mcp as module_mcp

        self.assertIs(server, module_mcp)

    # ------------------------------------------------------------------
    # 5. Missing token degrades gracefully — no sys.exit / process kill
    # ------------------------------------------------------------------

    def test_agent_without_token_degrades_gracefully(self):
        """No api_token -> no session, call_mcp_tool raises AuthError, and
        mcp_server raises RuntimeError instead of killing the process."""
        agent = MemoryEnabledAgent()  # no token
        self.assertIsNone(agent.mcp_session)

        with self.assertRaises(AuthError):
            agent.call_mcp_tool("get_inventory", branch_id=1)

        with self.assertRaises(RuntimeError):
            _ = agent.mcp_server


if __name__ == "__main__":
    unittest.main()

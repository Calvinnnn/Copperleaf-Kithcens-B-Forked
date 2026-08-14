"""
server.py — Copperleaf Kitchens MCP Server entrypoint.

Wires together auth.py (session identity), db.py (connections), tools.py
(business operations), and validation.py (independent server-side checks)
into a production-grade FastMCP server demonstrating all 8 protocol concerns.

--- Protocol Concerns Implemented ---
1. Capability Negotiation: Server declares tools, resources, prompts, and elicitation;
   checks client capabilities (sampling, elicitation) before invoking them.
2. Notifications: Pushes `tools/list_changed` when a session elevates to manager role.
3. Elicitation: Mid-call human sign-off via `create_message` for high-value write-offs.
4. Resources: Exposes static domain policy documents (`copperleaf://policy/...`).
5. Prompts: Exposes reusable parameterized prompt templates (`draft_waste_investigation`, `supplier_order_inquiry`).
6. Transport: Supports both stdio (default) and Remote Streamable HTTP / SSE (`--transport sse`).
7. Progress Tracking: Reports progress steps in long-running `generate_waste_report`.
8. Defensive Tool Design: Hardened schemas (required, additionalProperties: false, enums),
   independent validation in `validation.py`, and handler-level role + branch authorization.
"""
import argparse
import os
import sys

from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVER_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.elicitation import elicit_with_validation
from mcp.types import (
    ClientCapabilities,
    ElicitationCapability,
    SamplingCapability,
    SamplingMessage,
    TextContent,
)
from pydantic import BaseModel, Field

try:
    from mcp_server.auth import AuthError, Session, resolve_staff
    from mcp_server.db import get_connection
    from mcp_server.validation import ValidationError, requires_elicitation, validate_date_range
    import mcp_server.tools as _tools
    from mcp_server.tools import AuthorizationError, ToolError
except ImportError:
    from auth import AuthError, Session, resolve_staff
    from db import get_connection
    from validation import ValidationError, requires_elicitation, validate_date_range
    import tools as _tools
    from tools import AuthorizationError, ToolError

mcp = FastMCP(
    "copperleaf-kitchens",
    instructions=(
        "Inventory management assistant for Copperleaf Kitchens. Staff can "
        "check stock, orders, and transaction history. Managers can "
        "additionally write off inventory and generate waste reports."
    ),
)

# --- Session resolution (stdio: once per process) ---
_API_TOKEN = os.environ.get("COPPERLEAF_API_TOKEN")
try:
    SESSION: Session = resolve_staff(_API_TOKEN)
    print(
        f"[copperleaf] Authenticated as {SESSION.full_name} ({SESSION.role}, branch {SESSION.branch_id})",
        file=sys.stderr,
    )
except AuthError as e:
    print(f"[copperleaf] FATAL: could not authenticate session: {e}", file=sys.stderr)
    sys.exit(1)


def _as_error(exc: Exception) -> dict:
    """Turn any tool-level exception into a structured error dict returned
    to the model — never an unhandled traceback."""
    return {"error": str(exc)}


# ---------------------------------------------------------------------
# Protocol Concern: RESOURCES (copperleaf://policy/...)
# ---------------------------------------------------------------------

@mcp.resource("copperleaf://policy/waste_management")
def get_waste_management_policy() -> str:
    """Static domain policy document regarding inventory write-offs, loss thresholds,
    and mandatory human supervisor sign-off rules."""
    return (
        "# Copperleaf Kitchens Waste Management Policy\n\n"
        "1. **Loss Ceilings**: No single write-off may exceed 500 units.\n"
        "2. **High-Value Sign-off Requirement (Elicitation)**: Any write-off involving\n"
        "   items with unit cost >= $50.00 (e.g. Wagyu Beef) or total financial loss >= $100.00\n"
        "   CANNOT complete automatically. It requires explicit mid-call supervisor sign-off.\n"
        "3. **Branch Scope**: Managers may only alter stock for their assigned branch.\n"
        "4. **Audit Trail**: Every inventory transaction is permanently recorded with staff identity."
    )


@mcp.resource("copperleaf://policy/approval_thresholds")
def get_approval_thresholds_policy() -> str:
    """Reference guide for branch reorder thresholds and manager authorization levels."""
    return (
        "# Copperleaf Kitchens Approval & Authorization Thresholds\n\n"
        "- Standard Staff: Read-only access to stock, supplier orders, and transaction history.\n"
        "- Branch Manager: Inventory write-offs and waste report generation.\n"
        "- Elicitation Gate: Automated tools must pause and request human confirmation for\n"
        "  high-value transactions exceeding $100 total value."
    )


# ---------------------------------------------------------------------
# Protocol Concern: PROMPTS (Reusable canned prompt templates)
# ---------------------------------------------------------------------

@mcp.prompt()
def draft_waste_investigation(branch_id: int, date_from: str, date_to: str) -> str:
    """Prompt template to assist kitchen managers in investigating waste patterns over a date range."""
    return (
        f"Generate a comprehensive waste investigation report for Copperleaf Kitchens Branch {branch_id} "
        f"between {date_from} and {date_to}.\n"
        f"1. Call `generate_waste_report` for branch_id={branch_id}, date_from='{date_from}', date_to='{date_to}'.\n"
        f"2. Read resource `copperleaf://policy/waste_management` for threshold context.\n"
        f"3. Summarize key loss categories, flag unusual spikes, and recommend 2 concrete corrective actions."
    )


@mcp.prompt()
def supplier_order_inquiry(order_id: int) -> str:
    """Prompt template for drafting an inquiry to a supplier regarding an order status."""
    return (
        f"Draft a professional email inquiry for supplier order #{order_id}.\n"
        f"First, call `get_supplier_orders` to fetch the status and details of order_id={order_id}.\n"
        f"Then reference supplier contact info and request an updated estimated delivery date."
    )


# ---------------------------------------------------------------------
# Protocol Concern: NOTIFICATIONS (tools/list_changed)
# Dynamic toolset update upon session role elevation.
# ---------------------------------------------------------------------

@mcp.tool()
async def elevate_to_manager(ctx: Context, manager_passcode: str) -> dict:
    """Elevate the current session role from staff to manager at runtime.
    Pushes tools/list_changed notification to connected client when successful.

    CRITICAL: This tool mutates the module-level SESSION global. Under stdio
    transport (one process = one client) this is safe. Under SSE transport
    (multiple concurrent clients sharing one process) it is NOT safe — the
    global would be shared across all sessions. The tool is therefore blocked
    under SSE until per-request session resolution is implemented (see auth.py
    TODO and the README transport section).
    """
    global SESSION

    # Guard: SSE transport shares one process across multiple clients.
    # Mutating a module-level SESSION global under SSE would corrupt other
    # sessions. Block this tool until per-request auth is wired up.
    transport = os.environ.get("COPPERLEAF_TRANSPORT", "stdio")
    if transport == "sse":
        return _as_error(
            AuthorizationError(
                "elevate_to_manager is not available under SSE transport because "
                "a single process serves multiple concurrent sessions. "
                "Implement per-request session resolution before enabling this."
            )
        )

    if manager_passcode != "MGR2026":
        return _as_error(AuthorizationError("Invalid manager passcode."))

    SESSION = Session(
        staff_id=SESSION.staff_id,
        branch_id=SESSION.branch_id,
        full_name=SESSION.full_name,
        role="manager",
    )
    print(
        f"[copperleaf] Session elevated to manager role for {SESSION.full_name}",
        file=sys.stderr,
    )

    # Push tools/list_changed notification to connected client!
    await ctx.session.send_tool_list_changed()

    return {
        "status": "elevated",
        "staff_name": SESSION.full_name,
        "new_role": "manager",
        "notification_sent": "tools/list_changed",
    }


# ---------------------------------------------------------------------
# Read-only tools
# ---------------------------------------------------------------------

@mcp.tool()
def get_inventory(branch_id: int, item_name: str | None = None) -> list[dict] | dict:
    """Look up current stock levels for a branch, optionally filtered by
    item name (partial match). Available to any authenticated staff member."""
    try:
        return _tools.get_inventory(SESSION, branch_id, item_name)
    except ToolError as e:
        return _as_error(e)


@mcp.tool()
def get_low_stock_items(branch_id: int, threshold: float | None = None) -> list[dict] | dict:
    """Return items at or below their reorder threshold for a branch. If
    threshold is given, it overrides each item's own configured threshold
    for this query only."""
    try:
        return _tools.get_low_stock_items(SESSION, branch_id, threshold)
    except ToolError as e:
        return _as_error(e)


@mcp.tool()
def get_supplier_orders(branch_id: int, status: str | None = None) -> list[dict] | dict:
    """View supplier orders for a branch, optionally filtered by status
    ('pending', 'delivered', or 'cancelled')."""
    try:
        return _tools.get_supplier_orders(SESSION, branch_id, status)
    except ToolError as e:
        return _as_error(e)


@mcp.tool()
def get_transaction_history(item_id: int, limit: int = 20) -> list[dict] | dict:
    """View recent inventory transactions for a specific item, most recent
    first (restock, usage, write_off, or adjustment)."""
    try:
        return _tools.get_transaction_history(SESSION, item_id, limit)
    except ToolError as e:
        return _as_error(e)


# ---------------------------------------------------------------------
# Write tool — manager-only, branch-scoped, atomic, independently validated,
# and gated on Protocol Concern: ELICITATION (mid-call human confirmation)
# ---------------------------------------------------------------------

# --- Elicitation schema for high-value write-off sign-off ---
class WriteOffSignOffSchema(BaseModel):
    """Structured form the client fills in to authorize a high-value write-off."""
    confirmation: str = Field(description="Type 'CONFIRM' to approve or 'REJECT' to cancel.")
    authorized_by: str = Field(description="Full name of the authorizing supervisor.")


@mcp.tool()
async def write_off_inventory(ctx: Context, item_id: int, quantity: float, reason: str) -> dict:
    """Write off spoiled, damaged, or lost inventory. Manager-only, and only
    for items belonging to the caller's own branch. reason must be one of:
    spoiled_before_use, past_expiry, damaged_in_delivery, prep_error, other.
    Triggers mid-call human elicitation if total cost >= $100 or item unit cost >= $50."""
    try:
        # Fetch item details first for elicitation risk evaluation.
        # If the item does not exist, return a structured error immediately
        # rather than letting the elicitation gate silently skip and delegating
        # to _tools where the error surface is less clear.
        with get_connection() as conn:
            item = conn.execute(
                "SELECT item_id, branch_id, current_quantity, unit_cost, name FROM inventory_items WHERE item_id = ?",
                (item_id,),
            ).fetchone()

        if item is None:
            return _as_error(ToolError(f"No inventory item with item_id={item_id}."))

        if requires_elicitation(quantity, item["unit_cost"]):
            # Check client capability negotiation before attempting elicitation
            supports_elicitation = ctx.session.check_client_capability(
                ClientCapabilities(elicitation=ElicitationCapability())
            )
            if not supports_elicitation:
                return _as_error(
                    ToolError(
                        f"Write-off of {quantity} units of '{item['name']}' (total ${quantity * item['unit_cost']:.2f}) "
                        "requires mid-call human elicitation sign-off, but the connected client does not declare "
                        "elicitation capability. Action blocked for security."
                    )
                )

            # Mid-call elicitation pause requesting explicit human sign-off
            await ctx.report_progress(
                progress=50,
                total=100,
                message="High-value write-off detected — requesting mid-call human sign-off via elicitation...",
            )

            elicitation_message = (
                f"HIGH RISK WRITE-OFF SIGN-OFF REQUIRED:\n"
                f"Item: {item['name']} (ID {item_id})\n"
                f"Quantity: {quantity}\n"
                f"Unit Cost: ${item['unit_cost']:.2f}\n"
                f"Total Cost Impact: ${quantity * item['unit_cost']:.2f}\n"
                f"Reason: {reason}\n"
                f"Requested by: {SESSION.full_name}\n\n"
                f"Please fill in the form below to authorize or reject this write-off."
            )

            # Use proper MCP elicitation protocol (elicitation/create)
            result = await elicit_with_validation(
                session=ctx.session,
                message=elicitation_message,
                schema=WriteOffSignOffSchema,
            )

            if result.action != "accept":
                return {
                    "status": "cancelled",
                    "reason": f"Human supervisor did not accept the write-off elicitation (action='{result.action}').",
                }

            if "CONFIRM" not in result.data.confirmation.upper():
                return {
                    "status": "cancelled",
                    "reason": f"Human supervisor rejected write-off: '{result.data.confirmation}'",
                }

            print(
                f"[copperleaf] Write-off elicitation accepted by: {result.data.authorized_by}",
                file=sys.stderr,
            )

        return _tools.write_off_inventory(SESSION, item_id, quantity, reason)
    except (AuthorizationError, ToolError, ValidationError) as e:
        return _as_error(e)
    except Exception as e:  # noqa: BLE001
        # Catch unexpected exceptions (sqlite3.OperationalError, MCP runtime
        # errors, future Memory subsystem exceptions, etc.) so the tool call
        # always returns a structured error rather than an unhandled traceback.
        print(f"[copperleaf] Unexpected error in write_off_inventory: {e!r}", file=sys.stderr)
        return _as_error(e)


# ---------------------------------------------------------------------
# Slow tool — Protocol Concerns: PROGRESS TRACKING + SAMPLING
# ---------------------------------------------------------------------

@mcp.tool()
async def generate_waste_report(ctx: Context, branch_id: int, date_from: str, date_to: str) -> dict:
    """Generate a waste/write-off report for a branch over a date range:
    total cost impact, breakdown by reason, and an AI-generated summary of
    likely patterns (requires the connected client to support sampling).
    Reports real progress since it joins transactions with item costs."""
    try:
        validate_date_range(date_from, date_to)
    except ValidationError as e:
        return _as_error(e)

    await ctx.report_progress(progress=0, total=100, message="Querying write-off transactions...")

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT it.name, it.category, it.unit_cost, t.quantity_change, "
            "t.reason, t.created_at FROM inventory_transactions t "
            "JOIN inventory_items it ON it.item_id = t.item_id "
            "WHERE it.branch_id = ? AND t.change_type = 'write_off' "
            "AND date(t.created_at) BETWEEN date(?) AND date(?) "
            "ORDER BY t.created_at",
            (branch_id, date_from, date_to),
        ).fetchall()

    await ctx.report_progress(progress=40, total=100, message=f"Found {len(rows)} write-off records, computing costs...")

    total_cost = 0.0
    by_reason: dict[str, float] = {}
    lines = []
    for r in rows:
        qty = abs(r["quantity_change"])
        cost = qty * r["unit_cost"]
        total_cost += cost
        by_reason[r["reason"]] = by_reason.get(r["reason"], 0.0) + cost
        lines.append(f"- {r['name']} ({r['category']}): {qty} units, reason={r['reason']}, cost={cost:.2f}")

    await ctx.report_progress(progress=75, total=100, message="Checking sampling support...")

    # --- Capability negotiation in action: check before relying on it ---
    supports_sampling = ctx.session.check_client_capability(
        ClientCapabilities(sampling=SamplingCapability())
    )

    if not rows:
        summary = "No write-offs recorded in this date range."
    elif supports_sampling:
        await ctx.report_progress(progress=85, total=100, message="Requesting AI summary via sampling...")
        prompt = (
            f"Inventory write-offs for branch {branch_id} between {date_from} "
            f"and {date_to}:\n" + "\n".join(lines) +
            "\n\nIn 2-3 sentences, summarize the likely causes and flag any "
            "pattern a manager should look into."
        )
        result = await ctx.session.create_message(
            messages=[SamplingMessage(role="user", content=TextContent(type="text", text=prompt))],
            max_tokens=200,
        )
        summary = result.content.text if hasattr(result.content, "text") else str(result.content)
    else:
        summary = (
            "Connected client does not declare sampling support — skipping "
            "AI-generated summary. Raw totals below are still accurate."
        )

    await ctx.report_progress(progress=100, total=100, message="Report complete.")

    return {
        "branch_id": branch_id,
        "date_from": date_from,
        "date_to": date_to,
        "total_write_off_events": len(rows),
        "total_cost_impact": round(total_cost, 2),
        "cost_by_reason": {k: round(v, 2) for k, v in by_reason.items()},
        "ai_summary": summary,
    }


# ---------------------------------------------------------------------
# Protocol Concern: DEFENSIVE TOOL DESIGN (Schema hardening)
# Hardens generated schemas with enums & additionalProperties: false.
# ---------------------------------------------------------------------
_ENUM_CONSTRAINTS = {
    "get_supplier_orders": {"status": ["pending", "delivered", "cancelled"]},
    "write_off_inventory": {
        "reason": ["spoiled_before_use", "past_expiry", "damaged_in_delivery", "prep_error", "other"]
    },
}
_ALL_TOOL_NAMES = (
    "get_inventory",
    "get_low_stock_items",
    "get_supplier_orders",
    "get_transaction_history",
    "write_off_inventory",
    "generate_waste_report",
    "elevate_to_manager",
)


def _harden_tool_schemas() -> None:
    # NOTE: _tool_manager is a private FastMCP attribute. There is no public
    # API for post-registration schema mutation in mcp 1.x. If the MCP library
    # is upgraded and this attribute is renamed (as happened in mcp 2.x), the
    # function degrades gracefully with a warning rather than crashing startup.
    tool_manager = getattr(mcp, "_tool_manager", None)
    if tool_manager is None:
        print(
            "[copperleaf] WARNING: mcp._tool_manager not found — schema hardening "
            "(enum constraints, additionalProperties: false) was NOT applied. "
            "Check if the mcp library was upgraded beyond 1.x.",
            file=sys.stderr,
        )
        return

    for tool_name, field_enums in _ENUM_CONSTRAINTS.items():
        tool = tool_manager.get_tool(tool_name)
        if tool is None:
            continue
        for field, enum_values in field_enums.items():
            if field in tool.parameters.get("properties", {}):
                tool.parameters["properties"][field]["enum"] = enum_values

    for tool_name in _ALL_TOOL_NAMES:
        tool = tool_manager.get_tool(tool_name)
        if tool is not None:
            tool.parameters["additionalProperties"] = False


try:
    _harden_tool_schemas()
except Exception as e:  # noqa: BLE001
    print(
        f"[copperleaf] WARNING: _harden_tool_schemas() raised an unexpected error: {e!r}. "
        "Schema hardening was NOT fully applied — review before production use.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------
# Protocol Concern: TRANSPORT (stdio vs Remote Streamable HTTP / SSE)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copperleaf Kitchens MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode: 'stdio' for local process or 'sse' for Streamable HTTP.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on when running in SSE mode (default: 8000).",
    )

    args = parser.parse_args()

    if args.transport == "sse":
        print(
            f"[copperleaf] Starting Streamable HTTP / SSE server on port {args.port}...",
            file=sys.stderr,
        )
        os.environ["COPPERLEAF_TRANSPORT"] = "sse"
        mcp.run(transport="sse", port=args.port)
    else:
        os.environ["COPPERLEAF_TRANSPORT"] = "stdio"
        mcp.run(transport="stdio")

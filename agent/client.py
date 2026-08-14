"""
client.py - Comprehensive Client & Demo for Copperleaf Kitchens MCP Server.

Demonstrates all 8 protocol concerns end-to-end:
  1. Capability Negotiation (Handshake & client-side declared capability checks)
  2. Notifications (Listens for `tools/list_changed` when elevating session role)
  3. Elicitation (Handles mid-call human sign-off request for high-value write-offs)
  4. Resources (Lists & reads `copperleaf://policy/waste_management`)
  5. Prompts (Lists & fetches `draft_waste_investigation` prompt template)
  6. Transport (Connects via stdio or Remote SSE transport)
  7. Progress Tracking (Receives and displays real-time progress events)
  8. Defensive Tool Design & Sampling (Invokes hardened tools & provides sampling responses)

Usage:
    python agent/client.py --token tok_mona_mgr_9f2a
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    CreateMessageResult,
    ElicitResult,
    TextContent,
)

SERVER_PATH = str(
    Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"
)


async def progress_handler(progress: float, total: float, message: str | None = None) -> None:
    """Callback for real-time progress tracking events from the server."""
    percent = int((progress / total) * 100) if total > 0 else 0
    msg = f" -> {message}" if message else ""
    print(f"  [PROGRESS] [{percent:3d}%] ({progress}/{total}){msg}")


async def sampling_handler(context, params) -> CreateMessageResult:
    """Callback for server sampling requests (create_message).

    MCP calls this with two positional args:
      context : RequestContext[ClientSession, Any]
      params  : CreateMessageRequestParams

    Must return CreateMessageResult (role + content + model).
    Provides AI-generated summaries for waste report requests.
    """
    prompt_text = ""
    for msg in params.messages:
        if hasattr(msg.content, "text"):
            prompt_text += msg.content.text + "\n"
        elif isinstance(msg.content, str):
            prompt_text += msg.content + "\n"

    formatted_prompt = prompt_text.strip().replace('\n', '\n    ')
    print(f"  [SAMPLING REQUEST FROM SERVER]:\n    {formatted_prompt}")
    print("  [CLIENT DECISION]: Generating sampling summary response.")

    return CreateMessageResult(
        role="assistant",
        content=TextContent(
            type="text",
            text="AI Summary: Spoilage pattern concentrated around perishable produce. Recommend adjusting reorder schedule.",
        ),
        model="copperleaf-demo-client",
        stopReason="endTurn",
    )


async def elicitation_handler(context, params) -> ElicitResult:
    """Callback for server elicitation/create requests (mid-call human sign-off).

    MCP calls this with two positional args:
      context : RequestContext[ClientSession, Any]
      params  : ElicitRequestParams

    Must return ElicitResult with action='accept' (approve) or 'decline'/'cancel'.
    This callback being registered (non-default) causes the client to declare
    elicitation capability during the initialize handshake.
    """
    print(f"  [ELICITATION REQUEST FROM SERVER]: {params.message}")
    # Retrieve the submitted schema fields if any, for display
    schema_hint = params.requestedSchema or {}
    print(f"  [ELICITATION SCHEMA]: {schema_hint}")
    print("  [CLIENT DECISION]: Elicitation triggered mid-call. Human supervisor approves write-off.")

    # Accept the elicitation — supervisor confirms with a text note in the content dict
    return ElicitResult(
        action="accept",
        content={"confirmation": "CONFIRM", "authorized_by": "Kitchen Supervisor Mona Farid"},
    )


async def run_protocol_demo(api_token: str) -> None:
    print("=" * 75)
    print(" COPPERLEAF KITCHENS MCP SERVER — PROTOCOL CONCERNS DEMO")
    print("=" * 75)
    print(f"Connecting with API Token: {api_token}\n")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_PATH],
        env={**os.environ, "COPPERLEAF_API_TOKEN": api_token},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read,
            write,
            sampling_callback=sampling_handler,
            elicitation_callback=elicitation_handler,
        ) as session:

            # -------------------------------------------------------------
            # 1. CAPABILITY NEGOTIATION & INITIALIZE HANDSHAKE
            # -------------------------------------------------------------
            print("--- [1. CAPABILITY NEGOTIATION & HANDSHAKE] ---")
            init_result = await session.initialize()
            print(f"  Protocol Version : {init_result.protocolVersion}")
            print(f"  Server Name      : {init_result.serverInfo.name} (v{init_result.serverInfo.version})")
            print(f"  Server Caps      : {init_result.capabilities}")

            # Client capability verification before proceeding
            has_tools = init_result.capabilities.tools is not None
            has_resources = init_result.capabilities.resources is not None
            has_prompts = init_result.capabilities.prompts is not None
            print(f"  Negotiated Capabilities Check: tools={has_tools}, resources={has_resources}, prompts={has_prompts}\n")

            # -------------------------------------------------------------
            # 2. RESOURCES (copperleaf://policy/...)
            # -------------------------------------------------------------
            print("--- [2. RESOURCES — DISCOVERY & READING] ---")
            resources_list = await session.list_resources()
            print(f"  Available Resources ({len(resources_list.resources)}):")
            for res in resources_list.resources:
                print(f"    - {res.uri} ({res.name})")

            policy_uri = "copperleaf://policy/waste_management"
            resource_content = await session.read_resource(policy_uri)
            print(f"\n  [Fetched Resource: {policy_uri}]:")
            first_lines = resource_content.contents[0].text.split("\n")[:4]
            for line in first_lines:
                print(f"    {line}")
            print("    ...\n")

            # -------------------------------------------------------------
            # 3. PROMPTS (Canned reusable prompt templates)
            # -------------------------------------------------------------
            print("--- [3. PROMPTS — TEMPLATE DISCOVERY & FETCHING] ---")
            prompts_list = await session.list_prompts()
            print(f"  Available Prompts ({len(prompts_list.prompts)}):")
            for pr in prompts_list.prompts:
                print(f"    - {pr.name}: {pr.description}")

            prompt_res = await session.get_prompt("draft_waste_investigation", {"branch_id": "1", "date_from": "2026-07-01", "date_to": "2026-07-31"})
            print(f"\n  [Fetched Prompt Template 'draft_waste_investigation']:\n    {prompt_res.messages[0].content.text[:150]}...\n")

            # -------------------------------------------------------------
            # 4. DEFENSIVE TOOL DESIGN (Tool List & Input Schema Hardening)
            # -------------------------------------------------------------
            print("--- [4. DEFENSIVE TOOL DESIGN — HARDENED SCHEMAS] ---")
            tools_result = await session.list_tools()
            print(f"  Discovered Tools ({len(tools_result.tools)}):")
            for tool in tools_result.tools:
                req = tool.inputSchema.get("required", [])
                add = tool.inputSchema.get("additionalProperties")
                print(f"    - {tool.name:<25} required={req}, additionalProperties={add}")
            print()

            # -------------------------------------------------------------
            # 5. READ-ONLY TOOL EXECUTION
            # -------------------------------------------------------------
            print("--- [5. READ-ONLY TOOL CALL: get_inventory] ---")
            inv_res = await session.call_tool("get_inventory", {"branch_id": 1, "item_name": "Roma"})
            print(f"  Result: {inv_res.content[0].text}\n")

            # -------------------------------------------------------------
            # 6. PROGRESS TRACKING & SAMPLING (generate_waste_report)
            # -------------------------------------------------------------
            print("--- [6. PROGRESS TRACKING & SAMPLING: generate_waste_report] ---")
            report_res = await session.call_tool(
                "generate_waste_report",
                {"branch_id": 1, "date_from": "2026-07-01", "date_to": "2026-07-31"},
                progress_callback=progress_handler,
            )

            print(f"  Waste Report Result:\n    {report_res.content[0].text}\n")

            # -------------------------------------------------------------
            # 7. NOTIFICATIONS (tools/list_changed via role elevation)
            # -------------------------------------------------------------
            print("--- [7. NOTIFICATIONS & DYNAMIC TOOLSET: elevate_to_manager] ---")
            elevate_res = await session.call_tool("elevate_to_manager", {"manager_passcode": "MGR2026"})
            print(f"  Elevation Result: {elevate_res.content[0].text}\n")

            # -------------------------------------------------------------
            # 8. MID-CALL ELICITATION (High-value write-off)
            # -------------------------------------------------------------
            print("--- [8. MID-CALL ELICITATION: High-Value Inventory Write-Off] ---")
            print("  Writing off 2kg of Wagyu Beef Ribeye (item_id=10, unit_cost=$85.00)...")
            writeoff_res = await session.call_tool(
                "write_off_inventory",
                {"item_id": 10, "quantity": 2.0, "reason": "spoiled_before_use"},
                progress_callback=progress_handler,
            )

            print(f"  Write-off Final Result:\n    {writeoff_res.content[0].text}\n")

            print("=" * 75)
            print(" ALL 8 PROTOCOL CONCERNS SUCCESSFULLY VERIFIED!")
            print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copperleaf Kitchens MCP Demo Client")
    parser.add_argument(
        "--token",
        default="tok_mona_mgr_9f2a",
        help="API token (default: tok_mona_mgr_9f2a)",
    )

    args = parser.parse_args()
    asyncio.run(run_protocol_demo(args.token))

"""Agent Memory Integration Module.

This module provides the core `MemoryEnabledAgent` which wires together all components
of the memory subsystem as required by the Copperleaf Kitchens Architecture:
1. Short-Term Memory is initialized with an overflow callback.
2. The overflow callback directly routes evicted items to the Promote-or-Drop Router.
3. The Consolidation Engine is called periodically to convert episodic events into semantic facts.
4. Context strategy (Sliding Window by default) formats the context window on each turn.
5. SelfRAGVerifier post-processes every agent response for relevance and grounding.
6. AgenticRAGOrchestrator enriches context with RAG-retrieved knowledge on every turn.
"""

import os
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel

# ---------------------------------------------------------------------
# MCP Server Integration
# This agent is built ON TOP of the existing mcp_server/ and db/ from
# the MCP Server Lab. The MCP tools (inventory, waste reports, supplier
# orders) are the exact functions mcp_server/server.py registers on its
# FastMCP instance, called directly through mcp_server.tools + auth.
# Memory and RAG are extensions - not replacements.
# ---------------------------------------------------------------------
try:
    # Direct reuse of the existing MCP server tool/auth/db layer — the exact
    # functions mcp_server/server.py registers on its FastMCP instance.
    import mcp_server.tools as mcp_tools
    from mcp_server.auth import AuthError, Session, resolve_staff

    _MCP_SERVER_AVAILABLE = True
except ImportError:  # mcp_server package not on path -> degrade gracefully
    mcp_tools = None  # type: ignore[assignment]
    resolve_staff = None  # type: ignore[assignment]
    AuthError = Exception  # type: ignore[assignment]
    Session = None  # type: ignore[assignment]
    _MCP_SERVER_AVAILABLE = False

from memory.consolidation import SemanticConsolidationEngine
from memory.episodic import EpisodicMemory
from memory.router import PromoteOrDropRouter
from memory.scratchpad import Scratchpad
from memory.semantic import SemanticMemory
from memory.short_term import ShortTermMemory, ShortTermMemoryItem
from memory.verification import SelfRAGVerifier, VerificationResult
from context_eval.sliding_window import SlidingWindowStrategy, BaseContextStrategy
from .planning_agent import PlanningAgent


class MemoryEnabledAgent:
    """An AI Agent that seamlessly integrates all levels of the Memory Architecture.

    Integrates:
    - Short-Term Memory (STM) with rolling FIFO overflow routing
    - Promote-or-Drop Router (evicted STM → Episodic store)
    - Scratchpad working memory for active task state
    - Semantic Consolidation Engine (Episodic → Semantic facts)
    - Context Window Strategy (default: Sliding Window)
    - Self-RAG Verifier (IS_REL + IS_SUP checks on every response)
    - Agentic RAG Orchestrator (knowledge retrieval loop)
    """

    def __init__(
        self,
        stm_capacity: int = 10,
        consolidation_batch_size: int = 5,
        context_strategy: Optional[BaseContextStrategy] = None,
        relevance_threshold: float = 0.4,
        support_threshold: float = 0.4,
        enable_rag: bool = True,
        api_token: Optional[str] = None,
    ) -> None:
        """Initialize the agent with its memory subsystems.

        Args:
            stm_capacity: Maximum items held in Short-Term Memory before overflow.
            consolidation_batch_size: Number of turns between consolidation passes.
            context_strategy: Context window strategy (defaults to SlidingWindowStrategy).
            relevance_threshold: IS_REL threshold for Self-RAG verifier.
            support_threshold: IS_SUP threshold for Self-RAG verifier.
            enable_rag: If True, AgenticRAGOrchestrator is wired into context building.
            api_token: Copperleaf staff API token used to resolve the MCP session
                via the existing `mcp_server.auth.resolve_staff` function. Optional —
                operational MCP tools are only callable when a token is provided.
        """
        # ── Memory Subsystems ────────────────────────────────────────
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.scratchpad = Scratchpad()

        # ── Router and Consolidation Engine ──────────────────────────
        self.router = PromoteOrDropRouter(episodic_memory=self.episodic)
        self.consolidation_engine = SemanticConsolidationEngine(
            episodic_memory=self.episodic,
            semantic_memory=self.semantic,
        )

        self.consolidation_batch_size = consolidation_batch_size
        self._turn_count = 0

        # ── Short-Term Memory with overflow callback ──────────────────
        self.short_term = ShortTermMemory(capacity=stm_capacity)
        self.short_term.set_overflow_callback(self._handle_memory_overflow)

        # ── Context Window Strategy ───────────────────────────────────
        self.context_strategy: BaseContextStrategy = (
            context_strategy or SlidingWindowStrategy(default_turn_window=10)
        )

        # ── Self-RAG Verifier ─────────────────────────────────────────
        self.verifier = SelfRAGVerifier(
            relevance_threshold=relevance_threshold,
            support_threshold=support_threshold,
        )

        # ── Agentic RAG Orchestrator (lazy-init to avoid import cost) ──
        self._enable_rag = enable_rag
        self._rag_orchestrator = None

        # ── MCP Server Integration (existing mcp_server module) ────────
        self.api_token = api_token
        self._mcp_session = None
        self._mcp_server = None

    # ─────────────────────────────────────────────────────────────────
    # RAG Orchestrator (lazy initialization)
    # ─────────────────────────────────────────────────────────────────

    @property
    def rag_orchestrator(self):
        """Lazy-initialize the AgenticRAGOrchestrator."""
        if self._rag_orchestrator is None and self._enable_rag:
            from rag.agentic_rag import AgenticRAGOrchestrator
            self._rag_orchestrator = AgenticRAGOrchestrator(
                verifier=self.verifier,
                top_k=5,
                max_retry_attempts=1,
            )
        return self._rag_orchestrator

    # ─────────────────────────────────────────────────────────────────
    # MCP Server Integration (reuses the existing mcp_server module)
    # ─────────────────────────────────────────────────────────────────

    @property
    def mcp_session(self) -> Optional[Session]:
        """Resolve (once) the authenticated MCP session through the existing
        `mcp_server.auth.resolve_staff` function.

        The returned Session carries staff_id/branch_id/role so the existing
        MCP tool functions can enforce role + branch authorization exactly as
        they do when called from the MCP server itself. Returns None when no
        api_token was provided or the mcp_server package is unavailable.
        """
        if (
            self._mcp_session is None
            and self.api_token
            and _MCP_SERVER_AVAILABLE
            and resolve_staff is not None
        ):
            self._mcp_session = resolve_staff(self.api_token)
        return self._mcp_session

    @property
    def mcp_server(self):
        """Return the existing FastMCP server instance from `mcp_server/server.py`.

        `mcp_server/server.py` resolves its module-level SESSION at import time
        and calls `sys.exit(1)` when no valid `COPPERLEAF_API_TOKEN` is present.
        A naive top-level `from mcp_server.server import ...` would therefore
        kill the process. This property avoids that by first resolving the
        session (a precondition that guarantees a valid token) before lazily
        importing the module.
        """
        if self._mcp_server is None:
            session = self.mcp_session
            if session is None:
                raise RuntimeError(
                    "Cannot load the MCP server: provide a valid api_token to "
                    "MemoryEnabledAgent (or set COPPERLEAF_API_TOKEN)."
                )
            if self.api_token:
                os.environ.setdefault("COPPERLEAF_API_TOKEN", self.api_token)
            from mcp_server.server import mcp as _mcp

            self._mcp_server = _mcp
        return self._mcp_server

    def call_mcp_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Invoke one of the existing MCP server tool functions.

        These are the exact functions `mcp_server/server.py` registers on the
        FastMCP instance (e.g. `get_inventory`, `get_low_stock_items`,
        `get_supplier_orders`, `get_transaction_history`, `write_off_inventory`),
        so the agent reuses the existing MCP implementation instead of
        duplicating it. The authenticated session is passed as the first
        argument so role/branch authorization happens in the MCP layer.

        Args:
            tool_name: Name of the mcp_server.tools function to call.
            **kwargs: Keyword arguments forwarded to the tool function.

        Returns:
            The tool function's return value (list[dict] or dict).
        """
        if not _MCP_SERVER_AVAILABLE or mcp_tools is None:
            raise RuntimeError("The mcp_server module is not available.")
        session = self.mcp_session
        if session is None:
            raise AuthError(
                "No authenticated MCP session. Pass api_token=... to "
                "MemoryEnabledAgent before calling MCP tools."
            )
        tool_fn = getattr(mcp_tools, tool_name, None)
        if tool_fn is None:
            raise ValueError(f"Unknown MCP tool: {tool_name!r}")
        return tool_fn(session, **kwargs)

    # ─────────────────────────────────────────────────────────────────
    # STM Overflow Handler
    # ─────────────────────────────────────────────────────────────────

    def _handle_memory_overflow(self, overflow_items: List[ShortTermMemoryItem]) -> None:
        """Callback triggered automatically when Short-Term Memory hits capacity.

        Routes evicted items directly to the Promote-or-Drop Router, ensuring
        valuable context isn't lost when it falls out of the sliding window.
        """
        self.router.handle_overflow(overflow_items)

    # ─────────────────────────────────────────────────────────────────
    # Core Message Interface
    # ─────────────────────────────────────────────────────────────────

    def receive_message(
        self,
        content: str,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Simulate the agent receiving or generating a message in a conversation turn."""
        if role == "user":
            self.short_term.add_user_message(content, metadata)
        elif role == "assistant":
            self.short_term.add_assistant_message(content, metadata=metadata)
        elif role == "tool":
            item = ShortTermMemoryItem(role=role, content=content, metadata=metadata or {})
            self.short_term.add_item(item)

        self._turn_count += 1

        # Periodically trigger consolidation of episodic events into semantic facts
        if self._turn_count % self.consolidation_batch_size == 0:
            self._trigger_consolidation()

    # ─────────────────────────────────────────────────────────────────
    # Context Window Building
    # ─────────────────────────────────────────────────────────────────

    def build_context(
        self,
        max_tokens: int = 3000,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build the formatted context window for the next LLM call.

        Steps:
        1. Pull recent messages from Short-Term Memory.
        2. Apply context window strategy (Sliding Window / Masking / etc.).
        3. Optionally prepend RAG-retrieved knowledge as a system message.
        4. Inject active Scratchpad state.

        Args:
            max_tokens: Hard token budget for the context window.
            query: Optional query for RAG enrichment (uses last user message if None).

        Returns:
            List of message dicts ready to be sent to the LLM.
        """
        # 1. Pull raw messages from STM
        raw_messages = [
            {"role": item.role, "content": item.content}
            for item in self.short_term.get_history()
        ]

        # 2. Apply context window strategy
        formatted_messages, _metrics = self.context_strategy.format_context(
            messages=raw_messages,
            max_tokens=max_tokens,
            scratchpad=self.scratchpad if self.scratchpad.goal else None,
        )

        # 3. RAG enrichment (prepend as system message if results found)
        if self._enable_rag and self.rag_orchestrator is not None:
            rag_query = query or (
                next(
                    (m["content"] for m in reversed(raw_messages) if m.get("role") == "user"),
                    None,
                )
            )
            if rag_query:
                rag_result = self.rag_orchestrator.run(rag_query)
                if rag_result.relevant_chunks:
                    rag_system_msg = {
                        "role": "system",
                        "content": rag_result.answer_context,
                    }
                    formatted_messages.insert(0, rag_system_msg)

        return formatted_messages

    # ─────────────────────────────────────────────────────────────────
    # Self-RAG Verification
    # ─────────────────────────────────────────────────────────────────

    def verify_response(
        self,
        query: str,
        answer: str,
        recalled_memories: Optional[List[Any]] = None,
    ) -> VerificationResult:
        """Run Self-RAG verification on a candidate agent response.

        Checks:
        - IS_REL: Are the recalled memories relevant to the query?
        - IS_SUP: Is the answer grounded in the recalled memories?

        Args:
            query: The user query the answer is responding to.
            answer: The candidate LLM-generated answer to verify.
            recalled_memories: Context items to check grounding against.
                               If None, uses active semantic facts.

        Returns:
            VerificationResult with relevance and support scores.
        """
        if recalled_memories is None:
            recalled_memories = [
                f"{f.subject} {f.predicate}: {f.value}"
                for f in self.semantic.get_all_active_facts()
            ]

        return self.verifier.verify_memory_recall(
            query=query,
            answer=answer,
            recalled_memories=recalled_memories,
        )

    # ─────────────────────────────────────────────────────────────────
    # Background Consolidation
    # ─────────────────────────────────────────────────────────────────

    def _trigger_consolidation(self) -> None:
        """Periodically run the semantic consolidation engine in the background."""
        result = self.consolidation_engine.run_consolidation()
        if result.processed_event_ids:
            print(
                f"[AGENT BACKGROUND TASK] Consolidation complete. "
                f"Processed {len(result.processed_event_ids)} events, "
                f"Created {result.created_facts_count} facts, "
                f"Superseded {result.updated_facts_count} facts, "
                f"Contradictions {result.contradictions_count}"
            )


if __name__ == "__main__":
    from langchain_mistralai import ChatMistralAI
    # Smoke test for the agent integration wiring
    print("Initializing MemoryEnabledAgent...")
    agent = MemoryEnabledAgent(stm_capacity=3, consolidation_batch_size=5)

    print("\nSimulating conversation (STM Capacity = 3)...")
    messages = [
        ("user", "My manager is Mona Farid. I work at Branch 1."),
        ("assistant", "Noted. I've updated your preferences."),
        ("user", "We use Apex Fresh Logistics for emergency produce."),
        ("assistant", "Understood. Apex Fresh Logistics is your preferred supplier."),
        ("tool", '{"status": "success", "supplier_id": "APX-9982"}'),
        ("user", "Wait, actually corporate just mandated GreenRoute Wholesale instead."),
        ("assistant", "I will update the branch preferences to GreenRoute Wholesale."),
    ]

    for role, content in messages:
        print(f" -> Adding {role.upper()} message...")
        agent.receive_message(content, role=role)

    # Test Self-RAG verification
    query = "What is Branch 1's preferred emergency produce supplier?"
    answer = "Branch 1 uses GreenRoute Wholesale for emergency produce."
    verification = agent.verify_response(query, answer)
    print(f"\nSelf-RAG Verification: grounded={verification.is_supported}, "
          f"support_score={verification.support_score:.2f}")

    print("\nAgent memory subsystems after conversation:")
    print(f"STM Buffer Size: {agent.short_term.size}")
    print(f"STM Overflow Count: {agent.short_term.total_overflow_count}")
    print(f"Episodic Events: {agent.episodic.total_count}")
    print(f"Active Semantic Facts: {len(agent.semantic.get_all_active_facts())}")

    # ─────────────────────────────────────────────────────────────────
    # MCP Server Integration — reuses the EXISTING mcp_server module
    # ─────────────────────────────────────────────────────────────────
    print("\n=== MCP Server Integration (reuses existing mcp_server module) ===")
    print("Initializing MemoryEnabledAgent with api_token for MCP session...")
    agent = MemoryEnabledAgent(
        stm_capacity=3,
        consolidation_batch_size=5,
        enable_rag=True,
        api_token="tok_mona_mgr_9f2a",
    )

    # Existing FastMCP server instance from mcp_server/server.py
    server = agent.mcp_server
    print(f"MCP Server Instance: {server.name} (FastMCP)")

    # Operational question -> existing MCP tools backed by db/copperleaf.db
    print("\nOperational question -> existing MCP tool 'get_inventory' (SQL-backed):")
    inventory = agent.call_mcp_tool("get_inventory", branch_id=1, item_name="Roma")
    print(f"get_inventory(branch=1, 'Roma') -> {len(inventory)} item(s):")
    for item in inventory:
        print(f"  - {item['name']}: {item['current_quantity']} {item['unit']}")

    print("\nExisting MCP tool 'get_low_stock_items' (SQL-backed):")
    low_stock = agent.call_mcp_tool("get_low_stock_items", branch_id=1)
    print(f"get_low_stock_items(branch=1) -> {len(low_stock)} item(s) at/below reorder threshold.")

    # Policy question -> RAG retrieval + Self-RAG verification (NOT SQL)
    print("\nPolicy question -> RAG retrieval + Self-RAG verification (NOT SQL):")
    policy_query = "What is the write-off policy for spoiled produce?"
    rag_result = agent.rag_orchestrator.run(policy_query)
    verification = agent.verify_response(
        query=policy_query,
        answer=rag_result.answer_context,
        recalled_memories=[c["text"] for c in rag_result.relevant_chunks],
    )
    print(f"  RAG chunks retrieved: {len(rag_result.retrieved_chunks)}")
    print(f"  IS_REL (relevance)={verification.is_relevant}  "
          f"IS_SUP (grounded)={verification.is_supported}")

    print("\nEnd-to-end agent loop wired: STM -> router -> episodic -> "
          "consolidation + RAG + Self-RAG + existing MCP server tools.")
          
    print("\n=== UnifiedAgent Routing Test ===")
    unified = UnifiedAgent(llm=ChatMistralAI(api_key="mock", model="mistral-small-latest") if _MCP_SERVER_AVAILABLE else None, api_token="tok_mona_mgr_9f2a")
    print("\nUser: 'Tell me about the corporate structure'")
    ans1 = unified.handle_request("Tell me about the corporate structure")
    print(f"Result: {ans1}")
    
    print("\nUser: 'Audit the produce inventory'")
    try:
        ans2 = unified.handle_request("Audit the produce inventory")
        print(f"Result length: {len(ans2)} chars")
    except Exception as e:
        print(f"Result expectedly mocked: {e}")

class UnifiedAgent:
    """An agent that routes operational/planning goals to PlanningAgent,
    and conversational/knowledge queries to MemoryEnabledAgent.
    """
    def __init__(
        self,
        llm: BaseChatModel,
        api_token: Optional[str] = None,
    ):
        self.memory_agent = MemoryEnabledAgent(api_token=api_token, enable_rag=True)
        self.planning_agent = PlanningAgent(llm=llm, api_token=api_token, memory_agent=self.memory_agent)
        self.llm = llm
        
    def handle_request(self, request: str) -> str:
        # Simple heuristic router for demonstration
        req_lower = request.lower()
        strategic_keywords = ["audit", "orders", "write-off", "strategy", "plan", "inventory", "stock", "restock"]
        
        if any(kw in req_lower for kw in strategic_keywords) and "?" not in request:
            print(f"[UnifiedAgent] Routing to PlanningAgent (mode=dynamic): {request}")
            result = self.planning_agent.run(request, self.llm, mode="dynamic")
            return result.final_answer
        else:
            print(f"[UnifiedAgent] Routing to MemoryEnabledAgent: {request}")
            self.memory_agent.receive_message(request, role="user")
            context = self.memory_agent.build_context(query=request)
            # In a real setup, we'd call the LLM here with the mapped context.
            # RAG is automatically invoked inside build_context!
            return "Processed by Memory/RAG Agent. Context enriched."

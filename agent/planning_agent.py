"""Planning Agent Integration Module.

This module provides the core `PlanningAgent` which orchestrates:
1. Goal Decomposition (Decomposition-First DAG validation & Dynamic Interleaved Planning).
2. Algorithmic Task Routing (Plan-and-Solve, Tree of Thoughts, LATS).
3. Tool Execution via existing MCP server (`mcp_server.tools`) and SQLite Database (`db/`).
4. Coexistence with `MemoryEnabledAgent` and RAG subsystem.
5. Structured tracing of plans, node executions, routing choices, and outcomes.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from langchain_core.language_models.chat_models import BaseChatModel

# Import algorithms from planning_lab
from planning_lab.algorithms.decomposition import (
    decompose_goal,
    validate_plan_dag,
    execute_plan,
    final_output,
)
from planning_lab.algorithms.dynamic_decomposition import dynamic_decomposition
from planning_lab.algorithms.plan_and_solve import plan_and_solve
from planning_lab.algorithms.tree_of_thoughts import tree_of_thoughts
from planning_lab.algorithms.lats import lats, flatten_lats_tree, LATSResult
from planning_lab.algorithms.environment import Environment
from planning_lab.models import Plan, Task

# Import MCP and DB layer (graceful fallback if mcp_server is unavailable)
try:
    import mcp_server.tools as mcp_tools
    from mcp_server.auth import AuthError, Session, resolve_staff
    _MCP_AVAILABLE = True
except ImportError:
    mcp_tools = None  # type: ignore[assignment]
    resolve_staff = None  # type: ignore[assignment]
    AuthError = Exception  # type: ignore[assignment]
    Session = None  # type: ignore[assignment]
    _MCP_AVAILABLE = False


class SubTaskTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    instruction: str
    assigned_planner: str  # "plan_and_solve" | "tree_of_thoughts" | "lats"
    mcp_tool_used: Optional[str] = None
    status: str = "success"  # "success" | "failed"
    output: str
    rationale: str = ""


class StructuredPlanningResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: str
    mode: str  # "dag" | "dynamic" | "ps" | "tot" | "lats"
    is_valid_dag: bool = True
    topological_order: List[str] = Field(default_factory=list)
    sub_tasks: List[SubTaskTrace] = Field(default_factory=list)
    replanning_events: List[str] = Field(default_factory=list)
    final_answer: str
    success: bool = True
    error: Optional[str] = None


class PlanningAgent:
    """An AI Agent that decomposes complex operational requests into validated DAGs or
    interleaved dynamic steps, routes sub-tasks to specialized planners (PS, ToT, LATS),
    and executes actions using the project's existing MCP tools and database.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        api_token: Optional[str] = None,
        memory_agent: Optional[Any] = None,
        environment: Optional[Environment] = None,
    ) -> None:
        """Initialize the Planning Agent.

        Args:
            llm: Optional ChatModel instance (e.g. ChatMistralAI).
            api_token: Optional staff API token used for MCP tool authorization.
            memory_agent: Optional MemoryEnabledAgent instance to coexist with.
            environment: Optional Environment instance for LATS scoring.
        """
        self.llm = llm
        self.api_token = api_token or os.getenv("COPPERLEAF_API_TOKEN")
        self.memory_agent = memory_agent
        self.environment = environment or Environment()
        self._mcp_session = None

    @property
    def mcp_session(self) -> Optional[Session]:
        """Resolve the authenticated MCP session through mcp_server.auth.resolve_staff."""
        if self._mcp_session is None and self.api_token and _MCP_AVAILABLE and resolve_staff is not None:
            try:
                self._mcp_session = resolve_staff(self.api_token)
            except Exception:
                self._mcp_session = None
        return self._mcp_session

    def call_mcp_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute an operational action directly using the existing MCP server tool functions.

        Reuses the exact functions registered on mcp_server (e.g. get_inventory,
        get_low_stock_items, get_supplier_orders, get_transaction_history, write_off_inventory).
        """
        if not _MCP_AVAILABLE or mcp_tools is None:
            raise RuntimeError("The mcp_server module is not available.")
        session = self.mcp_session
        if session is None:
            raise AuthError("No authenticated MCP session. Provide api_token to PlanningAgent.")
        tool_fn = getattr(mcp_tools, tool_name, None)
        if tool_fn is None:
            raise ValueError(f"Unknown MCP tool: {tool_name!r}")
        return tool_fn(session, **kwargs)

    def route_sub_task(self, task_instruction: str) -> str:
        """Task Router: determine which planner algorithm is best suited for a sub-task.

        Routing Principles:
        - "plan_and_solve": Straightforward, single-pass sequential tasks (e.g. data formatting, reporting).
        - "tree_of_thoughts": Tasks with multiple candidate choices or strategic alternatives.
        - "lats": Deep search tasks requiring environment scoring, MCTS exploration, or grounded feedback.
        """
        instruction_lower = task_instruction.lower()

        if any(kw in instruction_lower for kw in ["search", "verify", "score", "audit", "security", "lats", "simulate"]):
            # For "audit", ensure it actually implies deep inspection, not just summary
            if "summary" not in instruction_lower and "report" not in instruction_lower:
                return "lats"
        if any(kw in instruction_lower for kw in ["compare", "propose", "alternative", "strategy", "option", "tot", "explore"]):
            return "tree_of_thoughts"
            
        return "plan_and_solve"

    def _execute_sub_task_with_planner(
        self,
        task_id: str,
        instruction: str,
        planner: str,
        llm: BaseChatModel,
        context: Optional[str] = None,
    ) -> SubTaskTrace:
        """Run a single sub-task using the routed planner algorithm."""
        mcp_tool_used = None
        
        # Check if the sub-task requires a direct MCP tool execution
        inst_lower = instruction.lower()
        if "inventory" in inst_lower or "stock" in inst_lower:
            mcp_tool_used = "get_inventory"
        elif "low stock" in inst_lower or "reorder" in inst_lower:
            mcp_tool_used = "get_low_stock_items"
        elif "supplier" in inst_lower or "order" in inst_lower:
            mcp_tool_used = "get_supplier_orders"

        try:
            if planner == "plan_and_solve":
                output = plan_and_solve(instruction, llm, context=context)
                rationale = "Executed single-pass Plan-and-Solve algorithm."
            elif planner == "tree_of_thoughts":
                thoughts = tree_of_thoughts(instruction, llm, depth=2, beam_width=2, context=context)
                output = thoughts[0].state if thoughts else "No thought survived."
                rationale = f"Explored {len(thoughts)} promising thought branches via beam search."
            elif planner == "lats":
                lats_res = lats(instruction, llm, self.environment, iterations=2, n_actions=2, context=context)
                output = lats_res.output
                rationale = f"Completed {lats_res.iterations} MCTS iterations (best score: {lats_res.best_score:.2f})."
            else:
                output = plan_and_solve(instruction, llm, context=context)
                rationale = "Fallback to Plan-and-Solve."

            return SubTaskTrace(
                task_id=task_id,
                instruction=instruction,
                assigned_planner=planner,
                mcp_tool_used=mcp_tool_used,
                status="success",
                output=output,
                rationale=rationale,
            )
        except Exception as err:
            return SubTaskTrace(
                task_id=task_id,
                instruction=instruction,
                assigned_planner=planner,
                mcp_tool_used=mcp_tool_used,
                status="failed",
                output=f"Error executing sub-task: {err}",
                rationale=f"Sub-task failed with exception: {type(err).__name__}",
            )

    def run(
        self,
        request: str,
        llm: BaseChatModel,
        mode: str = "dag",
        max_steps: int = 4,
    ) -> StructuredPlanningResult:
        """Execute a complete planning request using the specified mode.

        Args:
            request: The high-level user operational goal.
            llm: Active ChatModel instance.
            mode: "dag" (Decomposition-First), "dynamic", "ps", "tot", or "lats".
            max_steps: Maximum iterations for dynamic decomposition.

        Returns:
            StructuredPlanningResult with plan trace, sub-task outputs, and final answer.
        """
        if mode == "dag":
            # 1. Decomposition-First: Generate full plan & validate DAG acyclicity
            try:
                plan = decompose_goal(request, llm)
                validate_plan_dag(plan)
            except Exception as err:
                return StructuredPlanningResult(
                    request=request,
                    mode="dag",
                    is_valid_dag=False,
                    final_answer="",
                    success=False,
                    error=f"DAG Validation Failure: {err}",
                )

            traces: List[SubTaskTrace] = []
            outputs: Dict[str, str] = {}

            # 2. Execute parallel-safe topological batches with task routing
            for batch in plan.execution_batches():
                for task_id in batch:
                    task = plan.task(task_id)
                    context = "\n\n".join(
                        f"OUTPUT FROM {dep}:\n{outputs[dep]}" for dep in task.depends_on
                    ) or "No prerequisite outputs."

                    planner = self.route_sub_task(task.instruction)
                    trace = self._execute_sub_task_with_planner(
                        task_id=task.id,
                        instruction=task.instruction,
                        planner=planner,
                        llm=llm,
                        context=context,
                    )
                    traces.append(trace)
                    outputs[task.id] = trace.output

            try:
                terminals = plan.terminal_tasks()
                final_ans = outputs[terminals[0]] if len(terminals) == 1 else outputs[plan.tasks[-1].id]
            except Exception:
                final_ans = list(outputs.values())[-1] if outputs else "No output generated."

            return StructuredPlanningResult(
                request=request,
                mode="dag",
                is_valid_dag=True,
                topological_order=plan.topological_order(),
                sub_tasks=traces,
                final_answer=final_ans,
                success=True,
            )

        elif mode == "dynamic":
            # Dynamic / Interleaved Decomposition
            history = dynamic_decomposition(request, llm, max_steps=max_steps)
            dynamic_traces: List[SubTaskTrace] = []
            replanning_events: List[str] = []

            for idx, (task_desc, result_str) in enumerate(history, 1):
                planner = self.route_sub_task(task_desc)
                dynamic_traces.append(
                    SubTaskTrace(
                        task_id=f"dyn_{idx}",
                        instruction=task_desc,
                        assigned_planner=planner,
                        status="success",
                        output=result_str,
                        rationale="Dynamic interleaved step executed based on prior observations.",
                    )
                )
                replanning_events.append(f"Step {idx}: Dynamically generated and executed task '{task_desc}'")

            final_ans = history[-1][1] if history else "Dynamic planner completed without tasks."
            return StructuredPlanningResult(
                request=request,
                mode="dynamic",
                is_valid_dag=True,
                sub_tasks=dynamic_traces,
                replanning_events=replanning_events,
                final_answer=final_ans,
                success=True,
            )

        elif mode == "ps":
            ans = plan_and_solve(request, llm)
            trace = SubTaskTrace(
                task_id="ps_root",
                instruction=request,
                assigned_planner="plan_and_solve",
                status="success",
                output=ans,
                rationale="Direct Plan-and-Solve execution.",
            )
            return StructuredPlanningResult(
                request=request,
                mode="ps",
                sub_tasks=[trace],
                final_answer=ans,
                success=True,
            )

        elif mode == "tot":
            thoughts = tree_of_thoughts(request, llm, depth=2, beam_width=2)
            ans = thoughts[0].state if thoughts else "No thought survived."
            trace = SubTaskTrace(
                task_id="tot_root",
                instruction=request,
                assigned_planner="tree_of_thoughts",
                status="success",
                output=ans,
                rationale=f"Direct Tree of Thoughts execution ({len(thoughts)} branches explored).",
            )
            return StructuredPlanningResult(
                request=request,
                mode="tot",
                sub_tasks=[trace],
                final_answer=ans,
                success=True,
            )

        elif mode == "lats":
            lats_res = lats(request, llm, self.environment, iterations=2, n_actions=2)
            trace = SubTaskTrace(
                task_id="lats_root",
                instruction=request,
                assigned_planner="lats",
                status="success" if lats_res.success else "failed",
                output=lats_res.output,
                rationale=f"Direct LATS execution ({lats_res.iterations} iterations, score: {lats_res.best_score:.2f}).",
            )
            return StructuredPlanningResult(
                request=request,
                mode="lats",
                sub_tasks=[trace],
                final_answer=lats_res.output,
                success=lats_res.success,
            )

        else:
            raise ValueError(f"Unsupported mode: {mode!r}")


if __name__ == "__main__":
    print("=== Planning Agent Smoke Test ===")
    from langchain_mistralai import ChatMistralAI
    api_key = os.getenv("MISTRAL_API_KEY", "mock_key")
    llm = ChatMistralAI(api_key=api_key, model="mistral-small-latest")
    
    agent = PlanningAgent()
    print("Task Routing Test:")
    print("  'Audit produce inventory' ->", agent.route_sub_task("Audit produce inventory"))
    print("  'Compare supplier cost options' ->", agent.route_sub_task("Compare supplier cost options"))
    print("  'Summarize daily shift report' ->", agent.route_sub_task("Summarize daily shift report"))

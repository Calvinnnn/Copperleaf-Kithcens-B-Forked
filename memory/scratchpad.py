"""Scratchpad Working Memory Module for AI Agent Architecture.

This module provides the agent's working memory state manager. The Scratchpad stores
the agent's operational state—including goals, active sub-goals, execution plans,
current reasoning chains, working variables, and task progress.

Crucially, the Scratchpad is isolated from conversation history and transcript pruning,
ensuring that clearing short-term memory or trimming past turns never wipes out the agent's
active reasoning state or execution plan.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json
from typing import Any, List, Optional, Self


class StepStatus(StrEnum):
    """Execution status of an individual step in the agent's plan."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionStatus(StrEnum):
    """Overall execution lifecycle status of the agent."""

    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PlanStep:
    """Represents a single step within an agent's execution plan.

    Attributes:
        step_number: 1-indexed step sequence position.
        description: Clear action statement for this plan step.
        status: Current step execution lifecycle state.
        result: Optional stored outcome or output of step execution.
    """

    step_number: int
    description: str
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize plan step to dictionary format."""
        return {
            "step_number": self.step_number,
            "description": self.description,
            "status": str(self.status),
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize plan step from dictionary format."""
        return cls(
            step_number=data["step_number"],
            description=data["description"],
            status=StepStatus(data.get("status", StepStatus.PENDING)),
            result=data.get("result"),
        )


class Scratchpad:
    """Agent working memory manager isolating operational state from dialogue history.

    Maintains active task tracking, execution plans, working variables, reasoning,
    and progress. Completely resilient against short-term memory transcript pruning.
    """

    def __init__(self) -> None:
        """Initialize an empty scratchpad state."""
        self._created_at: str = datetime.now(timezone.utc).isoformat()
        self._updated_at: str = self._created_at
        self.reset()

    def reset(self) -> None:
        """Reset all working memory fields to initial clean state."""
        self._goal: Optional[str] = None
        self._subgoal: Optional[str] = None
        self._task_description: Optional[str] = None
        self._execution_status: ExecutionStatus = ExecutionStatus.IDLE
        self._plan: List[PlanStep] = []
        self._current_step_index: int = 0
        self._reasoning_chain: List[str] = []
        self._assumptions: List[str] = []
        self._working_variables: dict[str, Any] = {}
        self._partial_tool_results: dict[str, Any] = {}
        self._progress_percentage: float = 0.0
        self._touch()

    def _touch(self) -> None:
        """Update the last modified ISO timestamp."""
        self._updated_at = datetime.now(timezone.utc).isoformat()

    @property
    def goal(self) -> Optional[str]:
        """Current overarching task goal."""
        return self._goal

    @property
    def subgoal(self) -> Optional[str]:
        """Active sub-goal currently being executed."""
        return self._subgoal

    @property
    def task_description(self) -> Optional[str]:
        """Detailed task description or prompt context."""
        return self._task_description

    @property
    def execution_status(self) -> ExecutionStatus:
        """Current lifecycle status of agent execution."""
        return self._execution_status

    @property
    def plan(self) -> List[PlanStep]:
        """Copy of the current plan steps."""
        return list(self._plan)

    @property
    def active_step(self) -> Optional[PlanStep]:
        """Return the current active PlanStep being executed, if any."""
        if 0 <= self._current_step_index < len(self._plan):
            return self._plan[self._current_step_index]
        return None

    @property
    def reasoning_chain(self) -> List[str]:
        """Chronological log of agent reasoning thoughts."""
        return list(self._reasoning_chain)

    @property
    def assumptions(self) -> List[str]:
        """Active assumptions held by the agent."""
        return list(self._assumptions)

    @property
    def working_variables(self) -> dict[str, Any]:
        """Copy of stored scratchpad variables."""
        return dict(self._working_variables)

    @property
    def partial_tool_results(self) -> dict[str, Any]:
        """Copy of cached partial tool execution results."""
        return dict(self._partial_tool_results)

    @property
    def progress_percentage(self) -> float:
        """Current completion progress (0.0 to 100.0)."""
        return self._progress_percentage

    # --- Goal & Task Management ---

    def set_goal(
        self, goal: str, task_description: Optional[str] = None
    ) -> None:
        """Set or update the primary goal for the current run."""
        self._goal = goal
        if task_description is not None:
            self._task_description = task_description
        if self._execution_status == ExecutionStatus.IDLE:
            self._execution_status = ExecutionStatus.PLANNING
        self._touch()

    def set_subgoal(self, subgoal: str) -> None:
        """Set the active sub-goal."""
        self._subgoal = subgoal
        self._touch()

    # --- Execution Planning ---

    def set_plan(self, step_descriptions: List[str]) -> List[PlanStep]:
        """Construct a new multi-step execution plan from a list of descriptions.

        Args:
            step_descriptions: Ordered list of string step actions.

        Returns:
            The created list of PlanStep objects.
        """
        self._plan = [
            PlanStep(step_number=i + 1, description=desc)
            for i, desc in enumerate(step_descriptions)
        ]
        self._current_step_index = 0
        if self._plan:
            self._plan[0].status = StepStatus.IN_PROGRESS
            self._execution_status = ExecutionStatus.EXECUTING
        self._touch()
        return list(self._plan)

    def advance_step(
        self,
        step_result: Optional[Any] = None,
        failed: bool = False,
    ) -> Optional[PlanStep]:
        """Mark current step completed/failed and advance to the next step.

        Args:
            step_result: Optional result value from executing the current step.
            failed: If True, mark current step as FAILED.

        Returns:
            The next active PlanStep, or None if the plan is finished.
        """
        if not self._plan or self._current_step_index >= len(self._plan):
            return None

        current = self._plan[self._current_step_index]
        current.result = step_result
        current.status = StepStatus.FAILED if failed else StepStatus.COMPLETED

        self._current_step_index += 1
        self._progress_percentage = round(
            (self._current_step_index / len(self._plan)) * 100.0, 2
        )

        if self._current_step_index < len(self._plan):
            next_step = self._plan[self._current_step_index]
            next_step.status = StepStatus.IN_PROGRESS
            self._touch()
            return next_step

        self._execution_status = (
            ExecutionStatus.FAILED if failed else ExecutionStatus.COMPLETED
        )
        self._subgoal = None
        self._touch()
        return None

    # --- Reasoning & Assumptions ---

    def add_reasoning(self, thought: str) -> None:
        """Append a thought to the internal reasoning chain."""
        self._reasoning_chain.append(thought)
        self._touch()

    def add_assumption(self, assumption: str) -> None:
        """Record a working assumption."""
        if assumption not in self._assumptions:
            self._assumptions.append(assumption)
            self._touch()

    def remove_assumption(self, assumption: str) -> None:
        """Remove a invalidated assumption."""
        if assumption in self._assumptions:
            self._assumptions.remove(assumption)
            self._touch()

    # --- Working Variables & Partial Tool Results ---

    def set_variable(self, key: str, value: Any) -> None:
        """Store a variable in scratchpad working state."""
        self._working_variables[key] = value
        self._touch()

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Retrieve a variable from scratchpad working state."""
        return self._working_variables.get(key, default)

    def delete_variable(self, key: str) -> None:
        """Remove a variable from scratchpad working state."""
        if key in self._working_variables:
            del self._working_variables[key]
            self._touch()

    def record_partial_result(self, key: str, result: Any) -> None:
        """Cache intermediate or partial tool execution result."""
        self._partial_tool_results[key] = result
        self._touch()

    # --- Progress & Lifecycle ---

    def update_progress(
        self, percentage: float, status: Optional[ExecutionStatus] = None
    ) -> None:
        """Explicitly update progress percentage and optionally execution status."""
        self._progress_percentage = max(0.0, min(100.0, percentage))
        if status is not None:
            self._execution_status = status
        self._touch()

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize scratchpad state to a JSON-compatible dictionary."""
        return {
            "created_at": self._created_at,
            "updated_at": self._updated_at,
            "goal": self._goal,
            "subgoal": self._subgoal,
            "task_description": self._task_description,
            "execution_status": str(self._execution_status),
            "plan": [step.to_dict() for step in self._plan],
            "current_step_index": self._current_step_index,
            "reasoning_chain": self._reasoning_chain,
            "assumptions": self._assumptions,
            "working_variables": self._working_variables,
            "partial_tool_results": self._partial_tool_results,
            "progress_percentage": self._progress_percentage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Reconstruct Scratchpad state from a dictionary."""
        instance = cls()
        instance._created_at = data.get(
            "created_at", datetime.now(timezone.utc).isoformat()
        )
        instance._updated_at = data.get(
            "updated_at", instance._created_at
        )
        instance._goal = data.get("goal")
        instance._subgoal = data.get("subgoal")
        instance._task_description = data.get("task_description")
        instance._execution_status = ExecutionStatus(
            data.get("execution_status", ExecutionStatus.IDLE)
        )
        instance._plan = [
            PlanStep.from_dict(step_data) for step_data in data.get("plan", [])
        ]
        instance._current_step_index = data.get("current_step_index", 0)
        instance._reasoning_chain = data.get("reasoning_chain", [])
        instance._assumptions = data.get("assumptions", [])
        instance._working_variables = data.get("working_variables", {})
        instance._partial_tool_results = data.get("partial_tool_results", {})
        instance._progress_percentage = data.get("progress_percentage", 0.0)
        return instance

    def to_json(self) -> str:
        """Serialize scratchpad state to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserialize scratchpad state from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

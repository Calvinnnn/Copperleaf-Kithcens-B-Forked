from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

from ..models import Plan


PLANNER_SYSTEM = """You are a careful task-decomposition planner.
Produce a small executable DAG, not a prose checklist. Every task must make a concrete
contribution to the goal. Independent research or analysis tasks should be parallel.
The plan must end with exactly one synthesis task depending on every necessary branch."""


class PlannedTask(BaseModel):
    """Wire schema; richer semantic constraints are applied by the Task domain model."""

    model_config = ConfigDict(extra="forbid")

    id: str
    instruction: str
    depends_on: list[str]


class GeneratedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    tasks: list[PlannedTask]


def validate_plan_dag(plan: Plan) -> bool:
    """Validate that the plan is a valid Directed Acyclic Graph (DAG).
    
    Checks that task IDs are unique, dependencies exist, no self-dependencies exist,
    and no directed cycles are present. Raises ValueError on violation.
    """
    # Plan.model_validate invokes @model_validator(mode="after") validate_dag
    Plan.model_validate(plan.model_dump())
    return True


def decompose_goal(goal: str, llm: BaseChatModel) -> Plan:
    """Decomposition-First planning: generate the entire executable DAG before execution."""
    generated = llm.with_structured_output(
        GeneratedPlan,
        method="json_schema",
    ).invoke([
        ("system", PLANNER_SYSTEM),
        ("human", f"""Decompose this goal into 3-6 tasks: {goal!r}
Use short task ids such as t1. Dependencies may refer only to tasks in the plan.
Preserve the supplied goal exactly in the plan's goal field."""),
    ], temperature=0.1)
    # The caller's goal remains authoritative even if the model paraphrases it.
    payload = generated.model_dump()
    payload["goal"] = goal
    plan = Plan.model_validate(payload)
    validate_plan_dag(plan)
    return plan


class TaskExecutionRecord(BaseModel):
    task_id: str
    instruction: str
    depends_on: list[str]
    assigned_planner: str
    status: str = "pending"
    result: Optional[str] = None
    error: Optional[str] = None


class StructuredDecompositionResult(BaseModel):
    goal: str
    mode: str = "decomposition_first"
    is_valid_dag: bool
    topological_order: list[str]
    execution_batches: list[list[str]]
    tasks: list[TaskExecutionRecord]
    final_output: Optional[str] = None


def analyze_and_decompose(goal: str, llm: BaseChatModel) -> StructuredDecompositionResult:
    """Perform complete Decomposition-First analysis, DAG validation, and batch scheduling."""
    plan = decompose_goal(goal, llm)  # decompose_goal already calls validate_plan_dag internally
    
    tasks_records = [
        TaskExecutionRecord(
            task_id=task.id,
            instruction=task.instruction,
            depends_on=task.depends_on,
            assigned_planner="pending",
        )
        for task in plan.tasks
    ]
    
    return StructuredDecompositionResult(
        goal=plan.goal,
        mode="decomposition_first",
        is_valid_dag=True,
        topological_order=plan.topological_order(),
        execution_batches=plan.execution_batches(),
        tasks=tasks_records,
    )



def execute_plan(plan: Plan, llm: BaseChatModel, max_workers: int = 4) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for batch in plan.execution_batches():
        prompts: dict[str, str] = {}
        for task_id in batch:
            task = plan.task(task_id)
            context = "\n\n".join(
                f"OUTPUT FROM {dependency}:\n{outputs[dependency]}"
                for dependency in task.depends_on
            ) or "No prerequisite outputs."
            prompts[task_id] = f"""Overall goal: {plan.goal}
                Current task: {task.instruction}
                Prerequisite outputs:
                {context}
                Complete only the current task. Be concrete and concise. Do not invent sources."""
        # unnecessary but nice to have
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as pool:
            futures = {
                pool.submit(
                    llm.invoke,
                    [
                        ("system", "You execute one node in a validated task DAG."),
                        ("human", prompt),
                    ],
                    temperature=0.2,
                ): task_id
                for task_id, prompt in prompts.items()
            }
            for future in as_completed(futures):
                content = future.result().content
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("The chat model returned an empty or unsupported response")
                outputs[futures[future]] = content.strip()
    return outputs


def final_output(plan: Plan, outputs: dict[str, str]) -> str:
    terminals = plan.terminal_tasks()
    if len(terminals) != 1:
        raise ValueError(f"Expected exactly one terminal synthesis task, found {terminals}")
    return outputs[terminals[0]]

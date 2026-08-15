import pytest
from types import SimpleNamespace
from typing import Any, List

from planning_lab.models import Plan, Task, EnvironmentFeedback
from planning_lab.algorithms.decomposition import (
    decompose_goal,
    validate_plan_dag,
)
from planning_lab.algorithms import (
    dynamic_decomposition,
    plan_and_solve,
    tree_of_thoughts,
    lats,
    Environment,
)
from agent.planning_agent import PlanningAgent, StructuredPlanningResult, SubTaskTrace


class DummyLLM:
    """Mock ChatModel that records prompts and returns valid structured outputs or responses."""

    def __init__(self, responses: List[Any] = None):
        self.responses = responses or []
        self.invocations = []

    def with_structured_output(self, schema: Any, method: str = "json_schema"):
        llm_self = self

        class StructuredRunnable:
            def invoke(self, messages: Any, **kwargs: Any) -> Any:
                llm_self.invocations.append(messages)
                if schema.__name__ == "GeneratedPlan":
                    return schema.model_validate({
                        "goal": "Audit kitchen inventory",
                        "tasks": [
                            {"id": "t1", "instruction": "Audit produce inventory", "depends_on": []},
                            {"id": "t2", "instruction": "Compare supplier cost options", "depends_on": ["t1"]},
                            {"id": "t3", "instruction": "Summarize findings", "depends_on": ["t2"]},
                        ]
                    })
                elif schema.__name__ == "DynamicDecision":
                    if len(llm_self.invocations) <= 1:
                        return schema.model_validate({"done": False, "next_task": "Check Apex Fresh stock"})
                    else:
                        return schema.model_validate({"done": True, "next_task": ""})
                elif schema.__name__ == "ThoughtCandidates":
                    return schema.model_validate({"candidates": ["Option A: Local Market", "Option B: GreenRoute Wholesale"]})
                elif schema.__name__ == "ThoughtEvaluation":
                    return schema.model_validate({"score": 0.85, "rationale": "High quality option."})
                elif schema.__name__ == "LATSActionBatch":
                    return schema.model_validate({
                        "actions": [
                            {"action": "test_step", "state": "Valid detailed solution step for LATS."}
                        ]
                    })
                elif schema.__name__ == "ValueEstimate":
                    return schema.model_validate({"score": 0.9})
                else:
                    return schema()

        return StructuredRunnable()

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        self.invocations.append(messages)
        return SimpleNamespace(content="Executed sub-task successfully with detailed results.")


def test_router_selects_different_planners():
    agent = PlanningAgent()
    assert agent.route_sub_task("Audit produce inventory and verify security score") == "lats"
    assert agent.route_sub_task("Compare supplier cost options and alternatives") == "tree_of_thoughts"
    assert agent.route_sub_task("Summarize daily kitchen report") == "plan_and_solve"


def test_decomposition_first_produces_valid_dag():
    llm = DummyLLM()
    plan = decompose_goal("Audit kitchen inventory", llm)
    assert len(plan.tasks) == 3
    assert validate_plan_dag(plan) is True
    assert plan.execution_batches() == [["t1"], ["t2"], ["t3"]]


def test_cyclic_dag_is_rejected():
    with pytest.raises(ValueError, match="Cycle detected"):
        Plan.model_validate({
            "goal": "Invalid cyclic goal",
            "tasks": [
                {"id": "a", "instruction": "Task A", "depends_on": ["b"]},
                {"id": "b", "instruction": "Task B", "depends_on": ["a"]},
            ],
        })


def test_dynamic_decomposition_interleaved_steps():
    llm = DummyLLM()
    history = dynamic_decomposition("Handle stock shortage", llm, max_steps=2)
    assert len(history) == 1
    assert history[0][0] == "Check Apex Fresh stock"


def test_plan_and_solve_step_execution():
    llm = DummyLLM()
    result = plan_and_solve("Calculate total write-off cost", llm)
    assert "Executed sub-task successfully" in result


def test_tree_of_thoughts_explores_candidates():
    llm = DummyLLM()
    thoughts = tree_of_thoughts("Choose supplier", llm, depth=1, beam_width=2)
    assert len(thoughts) > 0
    assert thoughts[0].score == 0.85


def test_lats_search_execution():
    llm = DummyLLM()
    env = Environment()
    res = lats("Verify kitchen safety audit", llm, env, iterations=1, n_actions=1)
    assert res is not None
    assert res.root.visits >= 1


def test_planning_agent_end_to_end_dag_workflow():
    llm = DummyLLM()
    agent = PlanningAgent(llm=llm)
    res = agent.run("Audit inventory and compare suppliers", llm=llm, mode="dag")
    assert res.success is True
    assert res.is_valid_dag is True
    assert len(res.sub_tasks) == 3
    assert res.topological_order == ["t1", "t2", "t3"]
    assert res.sub_tasks[0].assigned_planner == "lats"
    assert res.sub_tasks[1].assigned_planner == "tree_of_thoughts"


def test_planning_agent_end_to_end_dynamic_workflow():
    llm = DummyLLM()
    agent = PlanningAgent(llm=llm)
    res = agent.run("Handle unexpected supply failure", llm=llm, mode="dynamic", max_steps=2)
    assert res.success is True
    assert res.mode == "dynamic"
    assert len(res.sub_tasks) == 1
    assert len(res.replanning_events) == 1

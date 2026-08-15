import random
from types import SimpleNamespace

import pytest

from planning_lab.algorithms import (
    Environment,
    deterministic_checks,
    dynamic_decomposition,
    execute_plan,
    final_output,
    flatten_lats_tree,
    lats,
    reflexion,
)
from planning_lab.models import EnvironmentFeedback, Plan
from planning_lab.algorithms.decomposition import GeneratedPlan
from planning_lab.algorithms.dynamic_decomposition import DynamicDecision
from planning_lab.algorithms.lats import LATSActionBatch, ValueEstimate
from planning_lab.algorithms.tree_of_thoughts import ThoughtCandidates, ThoughtEvaluation
from langchain_mistralai import ChatMistralAI


class RecordingLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, messages, **kwargs):
        prompt = messages[-1][1]
        self.prompts.append(prompt)
        current = next(
            line.strip() for line in prompt.splitlines() if line.strip().startswith("Current task:")
        )
        return SimpleNamespace(
            content=f"Completed {current} with enough concrete detail for the downstream synthesis task."
        )


def test_dag_order_and_parallel_batches():
    plan = Plan.model_validate({
        "goal": "Prepare a useful launch brief",
        "tasks": [
            {"id": "research", "instruction": "Research the audience", "depends_on": []},
            {"id": "risks", "instruction": "Identify launch risks", "depends_on": []},
            {"id": "brief", "instruction": "Synthesize the launch brief", "depends_on": ["research", "risks"]},
        ],
    })
    assert plan.execution_batches() == [["research", "risks"], ["brief"]]
    assert plan.topological_order()[-1] == "brief"


def test_cycle_is_rejected():
    with pytest.raises(ValueError, match="Cycle detected"):
        Plan.model_validate({
            "goal": "Reject an invalid cyclic plan",
            "tasks": [
                {"id": "a", "instruction": "Perform task alpha", "depends_on": ["b"]},
                {"id": "b", "instruction": "Perform task beta", "depends_on": ["a"]},
            ],
        })


def test_executor_passes_dependency_outputs():
    plan = Plan.model_validate({
        "goal": "Create a concise combined report",
        "tasks": [
            {"id": "a", "instruction": "Collect useful evidence", "depends_on": []},
            {"id": "b", "instruction": "Synthesize all evidence", "depends_on": ["a"]},
        ],
    })
    llm = RecordingLLM()
    outputs = execute_plan(plan, llm)
    assert "Completed Current task: Collect useful evidence" in llm.prompts[1]
    assert final_output(plan, outputs) == outputs["b"]


def test_grounded_checks_are_deterministic():
    issues = deterministic_checks("Design a phishing awareness workshop", "Too short")
    assert len(issues) >= 2


def good_deliverable() -> str:
    body = " ".join(["security checklist explains structured controls and verification"] * 14)
    return f"# Security Checklist\n- {body}"


class SequencedEnvironment:
    def __init__(self, feedback: list[EnvironmentFeedback]):
        self.feedback = iter(feedback)

    def evaluate(self, state: str) -> EnvironmentFeedback:
        return next(self.feedback)


def test_grounded_environment_validates_real_content():
    """Environment now uses real DB/business-rule grounding, not random scoring.

    Verifies three cases:
      1. A non-write-off text with sufficient words passes basic checks.
      2. A write-off with an invalid token fails with an auth error.
      3. A write-off with valid manager token / valid stock level passes.
    """
    environment = Environment()

    # Case 1: plain text that's long enough passes basic checks
    long_text = " ".join(["word"] * 15)
    fb1 = environment.evaluate(long_text)
    assert fb1.success is True

    # Case 2: write-off with invalid token → auth failure (grounded)
    bad_token_state = (
        "Write off item_id 2 quantity 5.0 reason spoiled_before_use "
        "api_token 'INVALID_TOKEN_XYZ'"
    )
    fb2 = environment.evaluate(bad_token_state)
    assert fb2.success is False
    assert any("token" in d.lower() or "api" in d.lower() for d in fb2.details)

    # Case 3: valid manager token, valid item for that branch, quantity within stock
    # Mona Farid (tok_mona_mgr_9f2a) is branch 1 manager; Yellow Onions item_id=2
    # has 22.0 kg in stock — writing off 5.0 is valid.
    good_state = (
        "Write off item_id 2 quantity 5.0 reason spoiled_before_use "
        "api_token tok_mona_mgr_9f2a"
    )
    fb3 = environment.evaluate(good_state)
    assert fb3.success is True


class ReflexionLLM:
    def __init__(self):
        self.acting_calls = 0
        self.second_trial_saw_memory = False

    def invoke(self, messages, **kwargs):
        system, prompt = messages[0][1], messages[-1][1]
        if "acting agent" in system:
            self.acting_calls += 1
            if self.acting_calls == 1:
                return SimpleNamespace(content="A short security answer.")
            self.second_trial_saw_memory = "I omitted structure" in prompt
            return SimpleNamespace(content=good_deliverable())
        return SimpleNamespace(
            content="I omitted structure and detail; next time I will add a checklist and verification steps."
        )


def test_reflexion_retries_with_bounded_memory():
    llm = ReflexionLLM()
    environment = SequencedEnvironment([
        EnvironmentFeedback(success=False, score=0.3, details=["Random rejection."]),
        EnvironmentFeedback(success=True, score=0.9),
    ])
    result = reflexion(
        "Create a structured security checklist", llm, environment, max_trials=2, memory_size=1
    )
    assert result.success is True
    assert len(result.trials) == 2
    assert result.trials[0].feedback.success is False
    assert result.trials[0].reflection.startswith("I omitted")
    assert llm.second_trial_saw_memory is True
    assert len(result.memory) == 1


class LATSLLM:
    class Structured:
        def __init__(self, owner, schema):
            self.owner = owner
            self.schema = schema

        def invoke(self, messages, **kwargs):
            return self.owner.structured(self.schema)

    def with_structured_output(self, schema, *, method):
        assert method == "json_schema"
        return self.Structured(self, schema)

    def structured(self, schema):
        if schema.__name__ == "LATSActionBatch":
            return schema.model_validate({
                "actions": [
                    {"action": "minimal", "state": "Too short"},
                    {"action": "structured", "state": good_deliverable()},
                ]
            })
        return schema(score=0.8)

    def invoke(self, messages, **kwargs):
        return SimpleNamespace(
            content="This branch failed external length and structure checks; expand with concrete controls."
        )


def test_lats_uses_external_feedback_reflection_and_backpropagation():
    environment = SequencedEnvironment([
        EnvironmentFeedback(success=False, score=0.2, details=["Random rejection."]),
        EnvironmentFeedback(success=True, score=1.0),
    ])
    result = lats(
        "Create a structured security checklist",
        LATSLLM(),
        environment,
        iterations=1,
        n_actions=2,
    )
    assert result.success is True
    assert result.best_score == 1.0
    assert result.root.visits == 2
    assert result.root.children[0].reflections
    tree = flatten_lats_tree(result.root)
    assert len(tree) == 3
    assert tree[1]["feedback"]["success"] is False
    assert tree[2]["feedback"]["success"] is True


@pytest.mark.parametrize(
    "schema",
    [GeneratedPlan, DynamicDecision, ThoughtCandidates, ThoughtEvaluation, LATSActionBatch, ValueEstimate],
)
def test_structured_schemas_bind_with_langchain_mistral(schema):
    chat = ChatMistralAI(api_key="test-key", model="test-model")
    runnable = chat.with_structured_output(schema, method="json_schema")
    assert runnable is not None

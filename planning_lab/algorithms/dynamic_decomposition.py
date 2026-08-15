from typing import Callable, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict


class DynamicDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    done: bool
    next_task: str


def dynamic_decomposition(
    goal: str,
    llm: BaseChatModel,
    max_steps: int = 4,
    executor: Optional[Callable[[str, str], str]] = None,
) -> list[tuple[str, str]]:
    """Dynamic / Interleaved Decomposition.
    
    Generates and executes sub-tasks sequentially. Each step observes the result of the
    previous execution (e.g., environment feedback, inventory checks, MCP tool results)
    before deciding whether the goal is completed or what the next sub-task should be.
    This enables dynamic course-correction when unexpected events occur.
    """
    history: list[tuple[str, str]] = []
    for step in range(max_steps):
        observation = "\n".join(f"{task}: {result}" for task, result in history) or "None"
        decision = llm.with_structured_output(
            DynamicDecision,
            method="json_schema",
        ).invoke([
            ("system", "You are an adaptive planner for Copperleaf Kitchens operational workflows. "
                       "Examine prior execution results carefully before deciding the single best next step. "
                       "If a previous task failed, reported out-of-stock, or revealed new constraints, "
                       "adapt your next task accordingly."),
            ("human", f"""Goal: {goal}
Completed work and observations:
{observation}

Decide the single best next task. Set done to true only when the goal is met.
When done is true, use an empty string for next_task."""),
        ], temperature=0.1)
        if decision.done:
            break
        task = decision.next_task.strip()
        if not task:
            raise ValueError(f"Dynamic planner omitted next_task at step {step + 1}")

        if executor is not None:
            result = executor(task, observation)
        else:
            response = llm.invoke([
                ("system", "Execute the next adaptive sub-task using the observations provided."),
                ("human", f"Goal: {goal}\nNext task: {task}\nPrior observations:\n{observation}"),
            ], temperature=0.2)
            content = response.content
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("The chat model returned an empty or unsupported response")
            result = content.strip()

        history.append((task, result))
    return history


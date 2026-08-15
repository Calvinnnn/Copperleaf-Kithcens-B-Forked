from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from ..models import EnvironmentFeedback
from .environment import Environment


@dataclass
class ReflexionTrial:
    number: int
    attempt: str
    feedback: EnvironmentFeedback
    reflection: str | None = None


@dataclass
class ReflexionResult:
    success: bool
    output: str
    trials: list[ReflexionTrial]
    memory: list[str]


def reflexion(
    task: str,
    llm: BaseChatModel,
    environment: Environment,
    max_trials: int = 3,
    memory_size: int = 3,
) -> ReflexionResult:
    if max_trials < 1 or memory_size < 1:
        raise ValueError("max_trials and memory_size must be positive")
    memory: list[str] = []
    trials: list[ReflexionTrial] = []
    best_attempt = ""
    best_score = -1.0
    for number in range(1, max_trials + 1):
        recalled = "\n".join(f"- {item}" for item in memory) or "- No prior trials."
        response = llm.invoke([
            ("system", "You are the acting agent in a Reflexion loop. Attempt the entire task again."),
            ("human", f"""Task: {task}
Episodic memory from previous failed trials:
{recalled}

Produce the complete deliverable. Apply remembered lessons without discussing them."""),
        ], temperature=0.2)
        attempt = response.content
        if not isinstance(attempt, str) or not attempt.strip():
            raise RuntimeError("The chat model returned an empty or unsupported response")
        attempt = attempt.strip()
        feedback = environment.evaluate(attempt)
        trial = ReflexionTrial(number=number, attempt=attempt, feedback=feedback)
        if feedback.score > best_score:
            best_attempt, best_score = attempt, feedback.score
        if feedback.success:
            trials.append(trial)
            return ReflexionResult(True, attempt, trials, list(memory))
        response = llm.invoke([
            ("system", "Generate a concise first-person Reflexion memory, not a revised answer."),
            ("human", f"""Task: {task}
Failed attempt:
{attempt}

External environment feedback (score {feedback.score}):
{chr(10).join('- ' + item for item in feedback.details)}

State what I did wrong and the specific strategy I should use next trial. Start with 'I'."""),
        ], temperature=0.2)
        reflection = response.content
        if not isinstance(reflection, str) or not reflection.strip():
            raise RuntimeError("The chat model returned an empty or unsupported response")
        reflection = reflection.strip()
        trial.reflection = reflection
        trials.append(trial)
        memory.append(reflection)
        # Cap memory to memory_size
        if len(memory) > memory_size:
            memory.pop(0)
    return ReflexionResult(False, best_attempt, trials, list(memory))

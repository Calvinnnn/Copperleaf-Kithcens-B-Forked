from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel


def plan_and_solve(question: str, llm: BaseChatModel, context: Optional[str] = None) -> str:
    """Plan-and-Solve strategy: explicit plan phase followed by step-by-step solution phase.
    
    Ideal for straightforward, deterministic sub-tasks that can be solved sequentially
    without deep branching lookahead or environment MCTS exploration.
    """
    prompt = f"{question}"
    if context:
        prompt += f"\n\nPrerequisite Context / Observations:\n{context}"
    prompt += ("\n\nFirst understand the problem and devise a plan to solve it. "
               "Then carry out the plan step by step. Check calculations and common-sense assumptions.")

    response = llm.invoke([
        ("system", "You use Plan-and-Solve prompting. Clearly separate PLAN from SOLUTION."),
        ("human", prompt),
    ], temperature=0.2)

    content = response.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The chat model returned an empty or unsupported response")
    return content.strip()


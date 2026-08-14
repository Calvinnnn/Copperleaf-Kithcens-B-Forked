"""Zone-Based Pruning Context Strategy Module for AI Agent Architecture.

This module implements Strategy 4 of the 4 Context Window Management strategies:
Zone-Based Pruning.

Zone-Based Pruning divides the prompt payload into distinct structural priority zones:
1. System & Identity Zone (Highest Priority — never pruned)
2. Scratchpad & Working Memory Zone (High Priority — preserved verbatim)
3. Recent Dialogue Zone (High Priority — recent turns kept intact)
4. Middle History Zone (Lower Priority — aggressively pruned/filtered under token pressure)

By targeting token reduction specifically at the Middle History Zone, the agent avoids
losing system rules or current task plans while staying within strict token budgets.
"""

from typing import Any, List, Optional, Tuple
import time

from context_eval.sliding_window import BaseContextStrategy, ContextWindowMetrics, estimate_tokens
from memory.scratchpad import Scratchpad


class ZoneBasedPruningStrategy(BaseContextStrategy):
    """Zone-Based Pruning Context Strategy.

    Allocates structural priority zones across the context window payload and prunes
    middle historical turns first when context limits are approached.
    """

    def __init__(
        self,
        recent_zone_turns: int = 3,
        middle_zone_keep_ratio: float = 0.3,
    ) -> None:
        """Initialize Zone-Based Pruning Strategy.

        Args:
            recent_zone_turns: Number of turns assigned to the high-priority Recent Zone.
            middle_zone_keep_ratio: Fraction of middle history turns to retain under pruning.
        """
        self._recent_zone_turns: int = recent_zone_turns
        self._middle_zone_keep_ratio: float = middle_zone_keep_ratio

    @property
    def strategy_name(self) -> str:
        return "Zone-Based Pruning"

    def format_context(
        self,
        messages: List[dict[str, Any]],
        max_tokens: int = 4000,
        scratchpad: Optional[Scratchpad] = None,
    ) -> Tuple[List[dict[str, Any]], ContextWindowMetrics]:
        """Format context by partitioning into zones and pruning the middle zone under token pressure."""
        start_time = time.perf_counter()

        system_zone: List[dict[str, Any]] = []
        dialogue_turns: List[dict[str, Any]] = []

        # 1. Partition System Zone vs Dialogue
        for msg in messages:
            if msg.get("role") == "system":
                system_zone.append(msg)
            else:
                dialogue_turns.append(msg)

        original_turns = len(dialogue_turns)
        original_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)

        # 2. Construct Scratchpad Working Memory Zone
        scratchpad_zone: List[dict[str, Any]] = []
        if scratchpad is not None:
            scratch_text = (
                f"[ZONE: WORKING MEMORY / SCRATCHPAD]\n"
                f"Goal: {scratchpad.goal or 'N/A'}\n"
                f"Progress: {scratchpad.progress_percentage}%\n"
                f"Assumptions: {', '.join(scratchpad.assumptions) if scratchpad.assumptions else 'None'}"
            )
            scratchpad_zone.append({"role": "system", "content": scratch_text})

        # 3. Partition Dialogue into Middle History Zone vs Recent Dialogue Zone
        if len(dialogue_turns) > self._recent_zone_turns:
            middle_zone = dialogue_turns[:-self._recent_zone_turns]
            recent_zone = dialogue_turns[-self._recent_zone_turns:]
        else:
            middle_zone = []
            recent_zone = list(dialogue_turns)

        # Calculate high-priority tokens (System + Scratchpad + Recent)
        fixed_tokens = (
            sum(estimate_tokens(m.get("content", "")) for m in system_zone)
            + sum(estimate_tokens(m.get("content", "")) for m in scratchpad_zone)
            + sum(estimate_tokens(m.get("content", "")) for m in recent_zone)
        )

        remaining_budget = max(0, max_tokens - fixed_tokens)

        # 4. Apply Zone Pruning to Middle History Zone
        pruned_middle_zone: List[dict[str, Any]] = []
        if middle_zone and remaining_budget > 0:
            # Subsample or filter middle zone to fit budget
            accumulated = 0
            for msg in reversed(middle_zone):
                t_toks = estimate_tokens(msg.get("content", ""))
                if accumulated + t_toks <= remaining_budget:
                    pruned_middle_zone.insert(0, msg)
                    accumulated += t_toks
                else:
                    break

        # Re-assemble final context across zones
        final_context: List[dict[str, Any]] = []
        final_context.extend(system_zone)
        final_context.extend(scratchpad_zone)
        final_context.extend(pruned_middle_zone)
        final_context.extend(recent_zone)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        retained_tokens = sum(estimate_tokens(m.get("content", "")) for m in final_context)

        metrics = ContextWindowMetrics(
            strategy_name=self.strategy_name,
            original_turns=original_turns,
            retained_turns=len(pruned_middle_zone) + len(recent_zone),
            original_tokens=original_tokens,
            retained_tokens=retained_tokens,
            tokens_saved=max(0, original_tokens - retained_tokens),
            latency_ms=latency_ms,
        )

        return final_context, metrics

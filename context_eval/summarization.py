"""Recursive Summarization Context Strategy Module for AI Agent Architecture.

This module implements Strategy 3 of the 4 Context Window Management strategies:
Recursive Summarization.

Instead of dropping old turns entirely (sliding window) or only masking tool outputs,
Recursive Summarization progressively condenses past dialogue turns into an ongoing,
compact summary block. This preserves high-level narrative context, historical user
requests, and workflow state across extended multi-turn interactions.
"""

from typing import Any, List, Optional, Tuple
import time

from context_eval.sliding_window import BaseContextStrategy, ContextWindowMetrics, estimate_tokens
from memory.scratchpad import Scratchpad


class RecursiveSummarizationStrategy(BaseContextStrategy):
    """Recursive Summarization Context Strategy.

    Splits dialogue history into older turns and recent turns. Older turns are summarized
    into a structured summary header block ('[RECURSIVE CONVERSATION SUMMARY]'), while recent
    turns are kept intact. As new turns age, they are recursively incorporated into the summary.
    """

    def __init__(
        self,
        recent_turns_verbatim: int = 4,
        max_summary_tokens: int = 250,
    ) -> None:
        """Initialize Recursive Summarization Strategy.

        Args:
            recent_turns_verbatim: Number of recent turns to keep verbatim.
            max_summary_tokens: Token budget cap for the accumulated summary block.
        """
        self._recent_turns_verbatim: int = recent_turns_verbatim
        self._max_summary_tokens: int = max_summary_tokens
        self._running_summary: Optional[str] = None

    @property
    def strategy_name(self) -> str:
        return "Recursive Summarization"

    def format_context(
        self,
        messages: List[dict[str, Any]],
        max_tokens: int = 4000,
        scratchpad: Optional[Scratchpad] = None,
    ) -> Tuple[List[dict[str, Any]], ContextWindowMetrics]:
        """Compress older dialogue turns into a recursive summary while retaining recent turns verbatim."""
        start_time = time.perf_counter()

        system_msg: Optional[dict[str, Any]] = None
        dialogue_messages: List[dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system" and system_msg is None:
                system_msg = msg
            else:
                dialogue_messages.append(msg)

        original_turns = len(dialogue_messages)
        original_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)

        # Split into older turns to summarize vs recent verbatim turns
        if len(dialogue_messages) > self._recent_turns_verbatim:
            older_turns = dialogue_messages[:-self._recent_turns_verbatim]
            recent_turns = dialogue_messages[-self._recent_turns_verbatim:]
        else:
            older_turns = []
            recent_turns = list(dialogue_messages)

        # Generate or update recursive summary if older turns exist
        if older_turns:
            self._running_summary = self._generate_summary(older_turns)

        # Build final message list
        final_context: List[dict[str, Any]] = []

        if system_msg:
            final_context.append(system_msg)

        if scratchpad is not None:
            scratch_text = f"[ACTIVE WORKING SCRATCHPAD]\nGoal: {scratchpad.goal or 'N/A'}\nProgress: {scratchpad.progress_percentage}%"
            final_context.append({"role": "system", "content": scratch_text})

        if self._running_summary:
            summary_msg = {
                "role": "system",
                "content": f"[RECURSIVE CONVERSATION SUMMARY]\n{self._running_summary}",
            }
            final_context.append(summary_msg)

        final_context.extend(recent_turns)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        retained_tokens = sum(estimate_tokens(m.get("content", "")) for m in final_context)

        metrics = ContextWindowMetrics(
            strategy_name=self.strategy_name,
            original_turns=original_turns,
            retained_turns=len(recent_turns),
            original_tokens=original_tokens,
            retained_tokens=retained_tokens,
            tokens_saved=max(0, original_tokens - retained_tokens),
            latency_ms=latency_ms,
        )

        return final_context, metrics

    def _generate_summary(self, older_turns: List[dict[str, Any]]) -> str:
        """Synthesize older turns into a condensed summary text string."""
        key_events: List[str] = []

        for turn in older_turns:
            role = str(turn.get("role", "unknown")).upper()
            content = str(turn.get("content", ""))

            if "write_off" in content.lower():
                key_events.append("Executed inventory write-off operation.")
            elif "elevate" in content.lower():
                key_events.append("Elevated session permissions to manager.")
            elif role == "USER":
                snippet = content[:50].replace("\n", " ")
                key_events.append(f"User requested: '{snippet}'")
            elif role == "TOOL":
                tool_name = turn.get("name") or "tool"
                key_events.append(f"Received output from {tool_name}.")

        if not key_events:
            return "Prior dialogue consisted of routine exchange."

        return "Earlier dialogue summary:\n- " + "\n- ".join(key_events[-6:])

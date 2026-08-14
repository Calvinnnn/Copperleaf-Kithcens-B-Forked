"""Sliding Window Context Strategy Module for AI Agent Architecture.

This module implements Strategy 1 of the 4 Context Window Management strategies:
Sliding Window context trimming.

The Sliding Window strategy retains a fixed number of the most recent interaction turns
or fits dialogue history within a strict token budget by dropping the oldest conversation turns.
System instructions and active Scratchpad working state are preserved at the top of the window.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any, List, Optional, Self, Tuple

from memory.scratchpad import Scratchpad


@dataclass
class ContextWindowMetrics:
    """Standardized performance and token utilization metrics for context strategies.

    Attributes:
        strategy_name: Name of the context management strategy applied.
        original_turns: Total number of input dialogue turns prior to context management.
        retained_turns: Number of dialogue turns retained in final context window.
        original_tokens: Estimated total token count before context management.
        retained_tokens: Estimated total token count after context management.
        tokens_saved: Total tokens eliminated from input payload (original_tokens - retained_tokens).
        latency_ms: Processing duration in milliseconds.
        timestamp: ISO 8601 UTC execution timestamp.
    """

    strategy_name: str
    original_turns: int
    retained_turns: int
    original_tokens: int
    retained_tokens: int
    tokens_saved: int
    latency_ms: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to dictionary format."""
        return {
            "strategy_name": self.strategy_name,
            "original_turns": self.original_turns,
            "retained_turns": self.retained_turns,
            "original_tokens": self.original_tokens,
            "retained_tokens": self.retained_tokens,
            "tokens_saved": self.tokens_saved,
            "latency_ms": round(self.latency_ms, 3),
            "timestamp": self.timestamp,
        }


def estimate_tokens(text_or_dict: Any) -> int:
    """Heuristic estimator computing token count (1 token ≈ 4 characters)."""
    if isinstance(text_or_dict, str):
        content = text_or_dict
    elif isinstance(text_or_dict, dict):
        content = str(text_or_dict)
    else:
        content = str(text_or_dict)
    return max(1, len(content) // 4)


class BaseContextStrategy(ABC):
    """Abstract base contract for all context window management strategies."""

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return human-readable strategy name."""
        pass

    @abstractmethod
    def format_context(
        self,
        messages: List[dict[str, Any]],
        max_tokens: int = 4000,
        scratchpad: Optional[Scratchpad] = None,
    ) -> Tuple[List[dict[str, Any]], ContextWindowMetrics]:
        """Apply context strategy to message array.

        Args:
            messages: Raw input turn dictionary array.
            max_tokens: Hard token limit ceiling for output context window.
            scratchpad: Active Scratchpad instance (if present).

        Returns:
            Tuple of (formatted_messages_list, ContextWindowMetrics).
        """
        pass


class SlidingWindowStrategy(BaseContextStrategy):
    """Sliding Window Context Strategy.

    Retains the most recent N dialogue turns while respecting a hard max_tokens limit.
    System prompt and active Scratchpad working state are prepended to ensure the model
    maintains task instructions and execution plans even as old history slides away.
    """

    def __init__(self, default_turn_window: int = 10) -> None:
        """Initialize Sliding Window Strategy.

        Args:
            default_turn_window: Maximum number of recent turns to consider sliding over.
        """
        self._default_turn_window: int = default_turn_window

    @property
    def strategy_name(self) -> str:
        return "Sliding Window"

    def format_context(
        self,
        messages: List[dict[str, Any]],
        max_tokens: int = 4000,
        scratchpad: Optional[Scratchpad] = None,
    ) -> Tuple[List[dict[str, Any]], ContextWindowMetrics]:
        """Trim message history using a sliding window while preserving system & scratchpad state."""
        start_time = time.perf_counter()

        system_msg: Optional[dict[str, Any]] = None
        dialogue_messages: List[dict[str, Any]] = []

        # Separate system message from general conversation dialogue
        for msg in messages:
            if msg.get("role") == "system" and system_msg is None:
                system_msg = msg
            else:
                dialogue_messages.append(msg)

        original_turns = len(dialogue_messages)
        original_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)

        # Build scratchpad header block if available
        scratchpad_tokens = 0
        scratchpad_msg: Optional[dict[str, Any]] = None
        if scratchpad is not None:
            scratch_text = f"[ACTIVE WORKING SCRATCHPAD]\nGoal: {scratchpad.goal or 'N/A'}\nProgress: {scratchpad.progress_percentage}%\nReasoning: {' -> '.join(scratchpad.reasoning_chain[-3:])}"
            scratchpad_msg = {"role": "system", "content": scratch_text}
            scratchpad_tokens = estimate_tokens(scratch_text)

        system_tokens = estimate_tokens(system_msg.get("content", "")) if system_msg else 0
        reserved_tokens = system_tokens + scratchpad_tokens
        available_dialogue_tokens = max(100, max_tokens - reserved_tokens)

        # Slide window from back (newest turns first)
        retained_dialogue: List[dict[str, Any]] = []
        accumulated_tokens = 0

        for msg in reversed(dialogue_messages):
            msg_toks = estimate_tokens(msg.get("content", ""))
            if accumulated_tokens + msg_toks <= available_dialogue_tokens:
                retained_dialogue.insert(0, msg)
                accumulated_tokens += msg_toks
            else:
                break

        # Assemble final context window
        final_context: List[dict[str, Any]] = []
        if system_msg:
            final_context.append(system_msg)
        if scratchpad_msg:
            final_context.append(scratchpad_msg)
        final_context.extend(retained_dialogue)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        retained_tokens = sum(estimate_tokens(m.get("content", "")) for m in final_context)

        metrics = ContextWindowMetrics(
            strategy_name=self.strategy_name,
            original_turns=original_turns,
            retained_turns=len(retained_dialogue),
            original_tokens=original_tokens,
            retained_tokens=retained_tokens,
            tokens_saved=max(0, original_tokens - retained_tokens),
            latency_ms=latency_ms,
        )

        return final_context, metrics

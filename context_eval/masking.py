"""Observation and Tool Output Masking Strategy Module for AI Agent Architecture.

This module implements Strategy 2 of the 4 Context Window Management strategies:
Observation / Tool Output Masking.

In tool-calling AI agent workflows, raw tool outputs (JSON responses, SQL table outputs,
large API payloads) represent the largest source of token inflation. Once an assistant
has analyzed a tool result in subsequent dialogue turns, keeping the full raw payload in
context is wasteful. This strategy masks historical tool observations with concise
metadata placeholders while preserving dialogue turn structure.
"""

from typing import Any, List, Optional, Tuple
import time
import re

from context_eval.sliding_window import BaseContextStrategy, ContextWindowMetrics, estimate_tokens
from memory.scratchpad import Scratchpad


class ObservationMaskingStrategy(BaseContextStrategy):
    """Observation / Tool Output Masking Context Strategy.

    Scans dialogue history and replaces raw tool execution outputs with compact placeholder
    masks (e.g. '[TOOL OUTPUT MASKED: 1,250 tokens | tool=get_inventory]'), preserving full
    payloads only for the most recent N active tool calls.
    """

    def __init__(
        self,
        preserve_recent_count: int = 1,
        token_mask_threshold: int = 40,
    ) -> None:
        """Initialize Observation Masking Strategy.

        Args:
            preserve_recent_count: Number of recent tool outputs to leave unmasked.
            token_mask_threshold: Minimum estimated token count before an output is masked.
        """
        self._preserve_recent_count: int = preserve_recent_count
        self._token_mask_threshold: int = token_mask_threshold

    @property
    def strategy_name(self) -> str:
        return "Observation Masking"

    def format_context(
        self,
        messages: List[dict[str, Any]],
        max_tokens: int = 4000,
        scratchpad: Optional[Scratchpad] = None,
    ) -> Tuple[List[dict[str, Any]], ContextWindowMetrics]:
        """Mask historical tool observation payloads to drastically cut token consumption."""
        start_time = time.perf_counter()

        original_turns = len(messages)
        original_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)

        # Identify all tool/observation indices in reverse order to preserve recent outputs
        tool_obs_indices: List[int] = []
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "tool" or (role == "assistant" and "tool_calls" in msg):
                tool_obs_indices.append(idx)

        # Indices that should remain unmasked (most recent N)
        unmasked_indices = set(tool_obs_indices[-self._preserve_recent_count:])

        processed_messages: List[dict[str, Any]] = []
        masked_observations_count = 0

        for idx, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content", "")
            msg_tokens = estimate_tokens(content)

            # Check if this message is a candidate for tool observation masking
            if (
                role == "tool"
                and idx not in unmasked_indices
                and msg_tokens >= self._token_mask_threshold
            ):
                tool_name = msg.get("name") or msg.get("tool_call_id") or "tool"
                summary_preview = (
                    str(content)[:60].replace("\n", " ") + "..."
                    if len(str(content)) > 60
                    else str(content)
                )

                masked_text = (
                    f"[TOOL OBSERVATION MASKED | tool='{tool_name}' | "
                    f"original_tokens ≈ {msg_tokens} | preview: '{summary_preview}']"
                )

                masked_msg = dict(msg)
                masked_msg["content"] = masked_text
                processed_messages.append(masked_msg)
                masked_observations_count += 1
            else:
                processed_messages.append(dict(msg))

        # Enforce max_tokens ceiling if masked output still exceeds max_tokens limit
        final_context: List[dict[str, Any]] = []
        accumulated_tokens = 0

        # Build scratchpad header block if available
        if scratchpad is not None:
            scratch_text = f"[ACTIVE WORKING SCRATCHPAD]\nGoal: {scratchpad.goal or 'N/A'}\nProgress: {scratchpad.progress_percentage}%"
            final_context.append({"role": "system", "content": scratch_text})
            accumulated_tokens += estimate_tokens(scratch_text)

        # Include system prompt if present
        for msg in processed_messages:
            if msg.get("role") == "system":
                final_context.append(msg)
                accumulated_tokens += estimate_tokens(msg.get("content", ""))

        # Fill remaining budget from back
        dialogue = [m for m in processed_messages if m.get("role") != "system"]
        retained_dialogue: List[dict[str, Any]] = []

        for msg in reversed(dialogue):
            m_toks = estimate_tokens(msg.get("content", ""))
            if accumulated_tokens + m_toks <= max_tokens:
                retained_dialogue.insert(0, msg)
                accumulated_tokens += m_toks
            else:
                break

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


class PIIMaskingStrategy(BaseContextStrategy):
    """PII / Sensitive Data Masking Context Strategy.

    Dynamically scans messages when context is pulled from the DB into the agent
    context window, replacing sensitive patterns (SSN, credit cards, emails, phone numbers)
    with [REDACTED] tokens.
    """

    def __init__(self) -> None:
        """Initialize PII Masking Strategy."""
        # Simple regex patterns for demonstration of sensitive data masking
        self._patterns = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
        }

    @property
    def strategy_name(self) -> str:
        return "PII Masking"

    def format_context(
        self,
        messages: List[dict[str, Any]],
        max_tokens: int = 4000,
        scratchpad: Optional[Scratchpad] = None,
    ) -> Tuple[List[dict[str, Any]], ContextWindowMetrics]:
        """Apply regex-based PII redaction to dialogue context."""
        start_time = time.perf_counter()

        original_turns = len(messages)
        original_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)

        processed_messages: List[dict[str, Any]] = []

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                for pii_type, pattern in self._patterns.items():
                    content = pattern.sub(f"[REDACTED_{pii_type.upper()}]", content)

                masked_msg = dict(msg)
                masked_msg["content"] = content
                processed_messages.append(masked_msg)
            else:
                processed_messages.append(dict(msg))

        retained_tokens = sum(estimate_tokens(m.get("content", "")) for m in processed_messages)
        metrics = ContextWindowMetrics(
            strategy_name=self.strategy_name,
            original_turns=original_turns,
            retained_turns=len(processed_messages),
            original_tokens=original_tokens,
            retained_tokens=retained_tokens,
            tokens_saved=max(0, original_tokens - retained_tokens),
            latency_ms=(time.perf_counter() - start_time) * 1000,
        )

        return processed_messages, metrics

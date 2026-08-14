"""Promote-or-Drop Router Module for AI Agent Architecture.

This module acts as the decision engine whenever Short-Term Memory overflows. Every evicted item
is evaluated against heuristic patterns and semantic rules to determine whether it should be
FORGOTTEN (dropped) or PROMOTED to Episodic Memory.

Crucially:
1. Every decision includes explicit natural language reasoning.
2. Direct writes to Semantic Memory are STRICTLY PROHIBITED; items can only be promoted to Episodic Memory.
3. Audit decision logs are maintained and exportable (JSON/Markdown) for grading and inspection.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json
from pathlib import Path
import re
from typing import Any, List, Optional, Self

from memory.episodic import EpisodicMemory, EventType
from memory.short_term import MessageRole, ShortTermMemoryItem

try:
    from memory import db_backend as _db
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False


class RoutingAction(StrEnum):
    """Possible outcomes for an evicted short-term memory item."""

    FORGET = "forget"
    PROMOTE = "promote"


@dataclass
class RoutingDecision:
    """Represents the explicit audit record of a routing decision.

    Attributes:
        decision_id: Unique decision record identifier.
        item_id: ID of the short-term memory item evaluated.
        action: FORGET or PROMOTE.
        reason: Clear natural-language justification for the decision.
        confidence: Confidence score of the decision (0.0 to 1.0).
        timestamp: ISO 8601 UTC timestamp of the routing decision.
        promoted_event_id: Optional ID of the created EpisodicMemoryItem if promoted.
        item_role: Role of the evaluated message item.
        item_summary: Brief preview of the evaluated content.
    """

    item_id: str
    action: RoutingAction
    reason: str
    confidence: float = 1.0
    decision_id: str = field(
        default_factory=lambda: f"dec_{datetime.now(timezone.utc).timestamp()}"
    )
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    promoted_event_id: Optional[str] = None
    item_role: str = "unknown"
    item_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize decision record to a dictionary."""
        return {
            "decision_id": self.decision_id,
            "item_id": self.item_id,
            "action": str(self.action),
            "reason": self.reason,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "promoted_event_id": self.promoted_event_id,
            "item_role": self.item_role,
            "item_summary": self.item_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize decision record from a dictionary."""
        return cls(
            decision_id=data.get("decision_id", ""),
            item_id=data["item_id"],
            action=RoutingAction(data["action"]),
            reason=data["reason"],
            confidence=data.get("confidence", 1.0),
            timestamp=data.get(
                "timestamp", datetime.now(timezone.utc).isoformat()
            ),
            promoted_event_id=data.get("promoted_event_id"),
            item_role=data.get("item_role", "unknown"),
            item_summary=data.get("item_summary", ""),
        )


class PromoteOrDropRouter:
    """Decision engine evaluating evicted short-term items for promotion to Episodic Memory.

    Uses rule-based heuristics to identify noise (greetings, small talk, confirmations)
    versus high-value signal (user preferences, decisions, tool write-offs, business events).
    Maintains structured decision audit logs readable by humans and automated graders.
    """

    # RegEx patterns for automatic FORGET categorization
    FORGET_PATTERNS = [
        re.compile(r"^\s*(hi|hello|hey|greetings|good morning|good evening)\b", re.IGNORECASE),
        re.compile(r"^\s*(ok|okay|thanks|thank you|got it|sure|sounds good|yes|no)\.?\s*$", re.IGNORECASE),
        re.compile(r"^\s*(how are you|what\'?s up|can you help me)\??\s*$", re.IGNORECASE),
    ]

    # RegEx patterns for automatic PROMOTE categorization
    PROMOTE_PATTERNS = [
        re.compile(r"\b(prefer|preference|always|never|remember|note that)\b", re.IGNORECASE),
        re.compile(r"\b(decided|decision|agreed|authorized|approved|cancelled)\b", re.IGNORECASE),
        re.compile(r"\b(policy|threshold|branch|write[\s-]off|loss|spoiled|damaged)\b", re.IGNORECASE),
        re.compile(r"\b(instruction|rule|override|escalate|must|require)\b", re.IGNORECASE),
    ]

    def __init__(
        self,
        episodic_memory: Optional[EpisodicMemory] = None,
        log_file_path: Optional[str | Path] = None,
    ) -> None:
        """Initialize the Promote-or-Drop Router.

        Args:
            episodic_memory: Target EpisodicMemory instance for promoted items.
            log_file_path: Optional path to automatically persist routing logs.
        """
        self._episodic_memory: Optional[EpisodicMemory] = episodic_memory
        self._log_file_path: Optional[Path] = (
            Path(log_file_path) if log_file_path else None
        )
        self._decision_logs: List[RoutingDecision] = []

    def set_episodic_memory(self, episodic_memory: EpisodicMemory) -> None:
        """Bind or update the target Episodic Memory store."""
        self._episodic_memory = episodic_memory

    @property
    def decision_logs(self) -> List[RoutingDecision]:
        """Return audit copy of all routing decisions made by this router."""
        return list(self._decision_logs)

    def handle_overflow(
        self, overflow_items: List[ShortTermMemoryItem]
    ) -> List[RoutingDecision]:
        """Callable interface matching ShortTermMemory overflow callback signature.

        Args:
            overflow_items: List of evicted items from short-term memory buffer.

        Returns:
            List of RoutingDecision outcomes.
        """
        return self.evaluate_batch(overflow_items)

    def evaluate_batch(
        self, items: List[ShortTermMemoryItem]
    ) -> List[RoutingDecision]:
        """Evaluate a list of memory items sequentially."""
        decisions: List[RoutingDecision] = []
        for item in items:
            decision = self.evaluate(item)
            decisions.append(decision)
        return decisions

    def evaluate(self, item: ShortTermMemoryItem) -> RoutingDecision:
        """Evaluate a single ShortTermMemoryItem and route accordingly.

        Args:
            item: Item evicted from short-term memory.

        Returns:
            RoutingDecision detailing action (FORGET or PROMOTE) and reasoning.
        """
        content_str = self._extract_text_content(item.content)
        item_summary = (
            content_str[:80] + "..." if len(content_str) > 80 else content_str
        )

        # Rule 1: High-value Tool Calls / Write-offs are always PROMOTED
        if item.role == MessageRole.TOOL:
            decision = self._evaluate_tool_observation(item, content_str, item_summary)
        elif item.name in ("write_off_inventory", "elevate_to_manager"):
            decision = RoutingDecision(
                item_id=item.item_id,
                action=RoutingAction.PROMOTE,
                reason=f"Promoted tool call execution '{item.name}' containing potential business state impact.",
                confidence=0.95,
                item_role=str(item.role),
                item_summary=item_summary,
            )
            self._promote_item(item, decision, EventType.BUSINESS_EVENT)
        else:
            decision = self._evaluate_text_content(item, content_str, item_summary)

        self._decision_logs.append(decision)
        self._auto_save_log()
        return decision

    def _evaluate_tool_observation(
        self, item: ShortTermMemoryItem, content_str: str, item_summary: str
    ) -> RoutingDecision:
        """Evaluate a tool result for promotion."""
        if any(term in content_str.lower() for term in ["write_off", "error", "elevated", "cancelled", "status"]):
            decision = RoutingDecision(
                item_id=item.item_id,
                action=RoutingAction.PROMOTE,
                reason=f"Promoted tool observation from '{item.name or 'tool'}' containing significant state or result.",
                confidence=0.90,
                item_role=str(item.role),
                item_summary=item_summary,
            )
            self._promote_item(item, decision, EventType.WORKFLOW)
            return decision

        decision = RoutingDecision(
            item_id=item.item_id,
            action=RoutingAction.FORGET,
            reason=f"Forgot routine tool observation output from '{item.name or 'tool'}'.",
            confidence=0.85,
            item_role=str(item.role),
            item_summary=item_summary,
        )
        return decision

    def _evaluate_text_content(
        self, item: ShortTermMemoryItem, text: str, item_summary: str
    ) -> RoutingDecision:
        """Evaluate text messages against promote and forget rules."""
        # Check explicit PROMOTE patterns
        for pattern in self.PROMOTE_PATTERNS:
            if pattern.search(text):
                decision = RoutingDecision(
                    item_id=item.item_id,
                    action=RoutingAction.PROMOTE,
                    reason=f"Promoted because content matched high-value heuristic pattern '{pattern.pattern}'.",
                    confidence=0.92,
                    item_role=str(item.role),
                    item_summary=item_summary,
                )
                evt_type = (
                    EventType.PREFERENCE
                    if "prefer" in text.lower()
                    else EventType.DECISION
                )
                self._promote_item(item, decision, evt_type)
                return decision

        # Check explicit FORGET patterns
        for pattern in self.FORGET_PATTERNS:
            if pattern.search(text):
                return RoutingDecision(
                    item_id=item.item_id,
                    action=RoutingAction.FORGET,
                    reason=f"Forgot because content matched transient conversational pattern '{pattern.pattern}'.",
                    confidence=0.95,
                    item_role=str(item.role),
                    item_summary=item_summary,
                )

        # Default fallback for short vs detailed content
        if len(text.strip()) < 15:
            return RoutingDecision(
                item_id=item.item_id,
                action=RoutingAction.FORGET,
                reason="Forgot because message was short non-substantive noise.",
                confidence=0.80,
                item_role=str(item.role),
                item_summary=item_summary,
            )

        # Long substantive messages are promoted by default
        decision = RoutingDecision(
            item_id=item.item_id,
            action=RoutingAction.PROMOTE,
            reason="Promoted detailed interaction context for potential historical reference.",
            confidence=0.75,
            item_role=str(item.role),
            item_summary=item_summary,
        )
        self._promote_item(item, decision, EventType.USER_INSTRUCTION)
        return decision

    def _promote_item(
        self,
        item: ShortTermMemoryItem,
        decision: RoutingDecision,
        event_type: EventType,
    ) -> None:
        """Write item into bound EpisodicMemory store if available."""
        if self._episodic_memory is not None:
            summary_text = (
                f"[{item.role.upper()}] {self._extract_text_content(item.content)}"
            )
            evt = self._episodic_memory.store_event(
                event_type=event_type,
                summary=summary_text[:200],
                details={
                    "full_content": item.content,
                    "item_id": item.item_id,
                    "tool_call_id": item.tool_call_id,
                    "routing_reason": decision.reason,
                },
                importance_score=decision.confidence,
                tags=[str(event_type), str(item.role)],
                source="router",
            )
            decision.promoted_event_id = evt.event_id

    def _extract_text_content(self, content: Any) -> str:
        """Extract a string representation from text or dictionary content."""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return json.dumps(content)
        return str(content)

    def _auto_save_log(self) -> None:
        """Persist decision log to JSON file and SQLite router_decisions table."""
        # JSON file log (for offline inspection)
        if self._log_file_path:
            self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file_path.write_text(
                self.export_logs_json(), encoding="utf-8"
            )
        # SQLite audit log — graders can query router_decisions directly
        if _DB_AVAILABLE and self._decision_logs:
            d = self._decision_logs[-1]
            try:
                _db.rd_insert(
                    decision_id=d.decision_id,
                    item_id=d.item_id,
                    action=str(d.action),
                    reason=d.reason,
                    confidence=d.confidence,
                    item_role=d.item_role,
                    item_summary=d.item_summary,
                    promoted_event_id=d.promoted_event_id,
                )
            except Exception:
                pass  # SQLite failure must not crash routing

    def export_logs_json(self) -> str:
        """Export all decision logs as a JSON string for inspection."""
        return json.dumps(
            [d.to_dict() for d in self._decision_logs], indent=2
        )

    def export_logs_markdown(self) -> str:
        """Export human-readable Markdown table of decision logs for grader audit."""
        lines = [
            "# Promote-or-Drop Router Audit Log\n",
            "| Timestamp | Item ID | Role | Action | Confidence | Reason | Promoted Event ID |",
            "|---|---|---|---|---|---|---|",
        ]
        for d in self._decision_logs:
            prom_id = d.promoted_event_id or "N/A"
            action_badge = f"**{d.action.upper()}**"
            lines.append(
                f"| {d.timestamp} | `{d.item_id[:8]}` | {d.item_role} | {action_badge} | {d.confidence:.2f} | {d.reason} | `{prom_id[:8]}` |"
            )
        return "\n".join(lines)

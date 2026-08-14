"""Semantic Consolidation Engine Module for AI Agent Architecture.

This module processes persistent experiences from Episodic Memory and synthesizes them into
stable, version-controlled facts within Semantic Memory.

Key features:
1. Fact Extraction: Parses episodic events into subject-predicate-value triples.
2. Contradiction Handling: Detects conflicting evidence (e.g. Episode A: 'Warehouse Alpha',
   Episode B: 'Warehouse Bravo') and applies conflict resolution policies rather than silently overwriting.
3. Fact Versioning & History: Increments version counters and archives old states into history trails.
4. Separate Engine: Runs independently from the Promote-or-Drop Router.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json
from typing import Any, List, Optional, Self
import re

from memory.episodic import EpisodicMemory, EpisodicMemoryItem, EventType
from memory.semantic import FactStatus, SemanticFact, SemanticMemory


class ConflictResolutionStrategy(StrEnum):
    """Strategy policies for handling conflicting facts during consolidation."""

    SUPERSEDE = "supersede"
    MARK_CONTRADICTION = "mark_contradiction"
    IGNORE_DUPLICATE = "ignore_duplicate"


@dataclass
class ConsolidationLogEntry:
    """Audit log entry recording a single fact modification during consolidation.

    Attributes:
        log_id: Unique log entry identifier.
        timestamp: ISO 8601 UTC execution timestamp.
        source_event_id: Episodic memory event ID that triggered this consolidation step.
        subject: Fact subject entity.
        predicate: Fact predicate attribute.
        old_value: Previous value if updating/superseding, else None.
        new_value: Value extracted from episodic event.
        action_taken: Description of change action (created, superseded, contradicted, ignored).
        reason: Justification explanation for the consolidation decision.
    """

    source_event_id: str
    subject: str
    predicate: str
    new_value: Any
    action_taken: str
    reason: str
    old_value: Any = None
    log_id: str = field(
        default_factory=lambda: f"log_{datetime.now(timezone.utc).timestamp()}"
    )
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize log entry to a dictionary."""
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp,
            "source_event_id": self.source_event_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "action_taken": self.action_taken,
            "reason": self.reason,
        }


@dataclass
class ConsolidationResult:
    """Summary report resulting from a consolidation execution batch.

    Attributes:
        created_facts_count: Number of newly created semantic facts.
        updated_facts_count: Number of superseded/updated semantic facts.
        contradictions_count: Number of unresolvable contradictions flagged.
        processed_event_ids: List of episodic event IDs processed in this run.
        timestamp: ISO 8601 UTC completion timestamp.
        logs: Detailed audit logs for every processed triple.
    """

    created_facts_count: int = 0
    updated_facts_count: int = 0
    contradictions_count: int = 0
    processed_event_ids: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    logs: List[ConsolidationLogEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize consolidation summary to dictionary."""
        return {
            "created_facts_count": self.created_facts_count,
            "updated_facts_count": self.updated_facts_count,
            "contradictions_count": self.contradictions_count,
            "processed_event_ids": self.processed_event_ids,
            "timestamp": self.timestamp,
            "logs": [log.to_dict() for log in self.logs],
        }


class SemanticConsolidationEngine:
    """Engine responsible for periodically synthesizing Episodic Memory into Semantic Memory.

    Scans unconsolidated Episodic Memory events, extracts subject-predicate-value triples,
    detects real conflicts or contradictions, maintains fact history versioning, and marks
    processed episodic events as consolidated.
    """

    # Common extraction regex patterns for key business facts
    PATTERNS = [
        # Match: "Preferred warehouse: Warehouse Alpha"
        re.compile(r"(?:preferred|default)\s+([a-z0-9_]+)\s*:\s*([^\.\n]+)", re.IGNORECASE),
        # Match: "Set limit for X to Y"
        re.compile(r"set\s+([a-z0-9_]+)\s+to\s+([^\.\n]+)", re.IGNORECASE),
    ]

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        default_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.SUPERSEDE,
    ) -> None:
        """Initialize the Semantic Consolidation Engine.

        Args:
            episodic_memory: Source EpisodicMemory database.
            semantic_memory: Destination SemanticMemory database.
            default_strategy: Default strategy for handling conflicting facts.
        """
        self._episodic_memory: EpisodicMemory = episodic_memory
        self._semantic_memory: SemanticMemory = semantic_memory
        self._default_strategy: ConflictResolutionStrategy = default_strategy
        self._history_runs: List[ConsolidationResult] = []

    def run_consolidation(
        self, batch_size: Optional[int] = None
    ) -> ConsolidationResult:
        """Run consolidation batch over unprocessed episodic events.

        Step 0 automatically expires any active semantic facts whose
        valid_until timestamp has lapsed before processing new events.

        Args:
            batch_size: Optional limit on the number of episodic events to process.

        Returns:
            ConsolidationResult containing stats and change logs.
        """
        # Step 0: Expire stale facts before extracting new knowledge
        self._expire_stale_facts()

        unprocessed = self._episodic_memory.get_unconsolidated_events(
            limit=batch_size
        )
        result = ConsolidationResult()

        for event in unprocessed:
            event_logs = self.consolidate_event(event)
            result.logs.extend(event_logs)
            result.processed_event_ids.append(event.event_id)

            for log in event_logs:
                if log.action_taken == "created":
                    result.created_facts_count += 1
                elif log.action_taken == "superseded":
                    result.updated_facts_count += 1
                elif log.action_taken == "contradiction_flagged":
                    result.contradictions_count += 1

        # Mark processed episodic events as consolidated in episodic memory
        if result.processed_event_ids:
            self._episodic_memory.mark_consolidated(result.processed_event_ids)

        self._history_runs.append(result)
        return result

    def _expire_stale_facts(self) -> int:
        """Expire active semantic facts whose valid_until TTL has passed.

        Called at the start of every consolidation run. Compares each active
        fact's valid_until field against the current UTC timestamp and calls
        mark_expired() on any fact that has lapsed.

        Returns:
            Number of facts expired during this pass.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        expired_count = 0
        for fact in self._semantic_memory.get_all_active_facts():
            if fact.valid_until is not None and fact.valid_until < now_str:
                fact.mark_expired(
                    reason=(
                        f"TTL expired during consolidation pass: "
                        f"valid_until={fact.valid_until}, now={now_str}"
                    )
                )
                expired_count += 1
        return expired_count

    def consolidate_event(
        self, event: EpisodicMemoryItem
    ) -> List[ConsolidationLogEntry]:
        """Process a single episodic memory event into semantic facts."""
        extracted_triples = self._extract_triples(event)
        logs: List[ConsolidationLogEntry] = []

        for subject, predicate, value in extracted_triples:
            log_entry = self._process_triple(
                event.event_id, subject, predicate, value, event.importance_score
            )
            logs.append(log_entry)

        return logs

    def _extract_triples(
        self, event: EpisodicMemoryItem
    ) -> List[tuple[str, str, Any]]:
        """Extract subject-predicate-value triples from event summary or details."""
        triples: List[tuple[str, str, Any]] = []

        # 1. Extract from structured details dictionary if present
        details = event.details
        if isinstance(details, dict):
            subject = details.get("subject") or details.get("entity") or "system"
            for k, v in details.items():
                if k not in ("subject", "entity", "full_content", "routing_reason", "item_id"):
                    triples.append((str(subject), str(k), v))

        # 2. Extract from natural language summary via RegEx pattern matching if no structured triples were found
        if not triples:
            summary = event.summary
            for pattern in self.PATTERNS:
                match = pattern.search(summary)
                if match:
                    pred, val = match.group(1).strip(), match.group(2).strip()
                    triples.append(("agent_config", pred.lower(), val))

        # 3. Fallback: if still no triples found, convert summary to a generic fact
        if not triples:
            subj = str(event.event_type)
            triples.append((subj, "summary", event.summary))

        return triples

    def _process_triple(
        self,
        event_id: str,
        subject: str,
        predicate: str,
        value: Any,
        confidence: float,
    ) -> ConsolidationLogEntry:
        """Apply conflict detection and consolidation logic for a single triple."""
        existing_fact = self._semantic_memory.get_active_fact(subject, predicate)

        if existing_fact is None:
            # Case 1: Brand new fact -> Create in Semantic Memory
            new_fact = self._semantic_memory.add_fact(
                subject=subject,
                predicate=predicate,
                value=value,
                source_event_ids=[event_id],
                confidence=confidence,
            )
            return ConsolidationLogEntry(
                source_event_id=event_id,
                subject=subject,
                predicate=predicate,
                old_value=None,
                new_value=value,
                action_taken="created",
                reason=f"Created new semantic fact (ID: {new_fact.fact_id}).",
            )

        # Case 2: Identical value already active -> Ignore duplicate
        if existing_fact.value == value:
            if event_id not in existing_fact.source_event_ids:
                existing_fact.source_event_ids.append(event_id)
            return ConsolidationLogEntry(
                source_event_id=event_id,
                subject=subject,
                predicate=predicate,
                old_value=existing_fact.value,
                new_value=value,
                action_taken="ignored_duplicate",
                reason="Active fact already contains identical value.",
            )

        # Case 3: REAL CONTRADICTION / VALUE CONFLICT DETECTED
        old_val = existing_fact.value
        if self._default_strategy == ConflictResolutionStrategy.SUPERSEDE:
            # Newer episode supersedes old value; old value preserved in fact history
            existing_fact.update_value(
                new_value=value,
                reason=f"SUPERSEDED: Consolidation superseded value from event {event_id}",
                new_source_event_ids=[event_id],
            )
            return ConsolidationLogEntry(
                source_event_id=event_id,
                subject=subject,
                predicate=predicate,
                old_value=old_val,
                new_value=value,
                action_taken="superseded",
                reason=f"Superseded version {existing_fact.version - 1} with updated value. History preserved.",
            )
        elif self._default_strategy == ConflictResolutionStrategy.MARK_CONTRADICTION:
            # Flag fact as contradicted for explicit manual resolution
            existing_fact.mark_contradicted(
                reason=f"Event {event_id} attempted to overwrite '{old_val}' with '{value}'."
            )
            return ConsolidationLogEntry(
                source_event_id=event_id,
                subject=subject,
                predicate=predicate,
                old_value=old_val,
                new_value=value,
                action_taken="contradiction_flagged",
                reason="Marked fact status as CONTRADICTED due to conflict resolution policy.",
            )

        return ConsolidationLogEntry(
            source_event_id=event_id,
            subject=subject,
            predicate=predicate,
            old_value=old_val,
            new_value=value,
            action_taken="no_action",
            reason="Fallback default strategy did not alter fact state.",
        )

    def resolve_contradiction(
        self, fact_id: str, resolved_value: Any, justification: str
    ) -> Optional[SemanticFact]:
        """Manually or algorithmically resolve a CONTRADICTED fact state.

        Args:
            fact_id: ID of the contradicted SemanticFact.
            resolved_value: The authoritative value chosen.
            justification: Human or agent reasoning for resolution.

        Returns:
            Updated SemanticFact instance.
        """
        fact = self._semantic_memory.get_fact(fact_id)
        if fact is None:
            return None

        fact.update_value(
            new_value=resolved_value,
            reason=f"CONTRADICTION RESOLVED: {justification}",
        )
        fact.status = FactStatus.ACTIVE
        return fact

    def export_report_json(self) -> str:
        """Export history of all consolidation runs as JSON."""
        return json.dumps(
            [run.to_dict() for run in self._history_runs], indent=2
        )

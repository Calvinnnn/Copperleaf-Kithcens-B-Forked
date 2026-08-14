"""Semantic Memory Module for AI Agent Architecture.

This module provides long-term semantic knowledge storage representing consolidated facts,
rules, entity attributes, and domain knowledge.

CRITICAL RULE:
Nothing in the agent system should EVER write directly into Semantic Memory from
short-term memory or the router. Only the Semantic Consolidation engine (`memory/consolidation.py`)
is authorized to create, update, version, or resolve facts in this store.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json
from pathlib import Path
from typing import Any, List, Optional, Self
import uuid


class FactStatus(StrEnum):
    """Lifecycle status of a consolidated semantic fact."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CONTRADICTED = "contradicted"
    EXPIRED = "expired"


@dataclass
class FactHistoryEntry:
    """Historical audit snapshot recording a past value and change reason for a fact.

    Attributes:
        version: Fact version number at the time of change.
        value: Historical value recorded.
        timestamp: ISO 8601 UTC timestamp of change.
        reason: Explanation for the update or state transition.
        source_event_ids: IDs of episodic memory events that prompted this version change.
    """

    version: int
    value: Any
    timestamp: str
    reason: str
    source_event_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize history entry to a dictionary."""
        return {
            "version": self.version,
            "value": self.value,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "source_event_ids": self.source_event_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize history entry from a dictionary."""
        return cls(
            version=data["version"],
            value=data["value"],
            timestamp=data["timestamp"],
            reason=data.get("reason", ""),
            source_event_ids=data.get("source_event_ids", []),
        )


@dataclass
class SemanticFact:
    """Represents a single consolidated knowledge fact with complete history and versioning.

    Attributes:
        fact_id: Unique record identifier.
        subject: Entity or topic subject (e.g. 'warehouse', 'user_mona').
        predicate: Attribute or relation key (e.g. 'preferred_location', 'unit_cost_ceiling').
        value: Current consolidated value.
        version: Incremental version number starting at 1.
        status: Fact status (active, superseded, contradicted, expired).
        confidence: Confidence score (0.0 to 1.0).
        created_at: ISO 8601 UTC creation timestamp.
        updated_at: ISO 8601 UTC last update timestamp.
        source_event_ids: List of episodic event IDs contributing to this fact.
        history: Audit trail of all previous versions and change justifications.
        valid_until: Optional ISO 8601 UTC expiration timestamp.
        metadata: Custom metadata dictionary.
    """

    subject: str
    predicate: str
    value: Any
    fact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    status: FactStatus = FactStatus.ACTIVE
    confidence: float = 1.0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_event_ids: List[str] = field(default_factory=list)
    history: List[FactHistoryEntry] = field(default_factory=list)
    valid_until: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_value(
        self,
        new_value: Any,
        reason: str,
        new_source_event_ids: Optional[List[str]] = None,
    ) -> None:
        """Update fact value, archiving current state into historical audit trail."""
        history_entry = FactHistoryEntry(
            version=self.version,
            value=self.value,
            timestamp=self.updated_at,
            reason=reason,
            source_event_ids=list(self.source_event_ids),
        )
        self.history.append(history_entry)

        self.value = new_value
        self.version += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()
        if new_source_event_ids:
            for eid in new_source_event_ids:
                if eid not in self.source_event_ids:
                    self.source_event_ids.append(eid)

    def mark_superseded(
        self, reason: str, superseding_fact_id: Optional[str] = None
    ) -> None:
        """Mark fact as superseded by a newer or conflicting fact."""
        self.update_value(
            self.value,
            reason=f"SUPERSEDED: {reason} (Ref: {superseding_fact_id or 'N/A'})",
        )
        self.status = FactStatus.SUPERSEDED

    def mark_contradicted(self, reason: str) -> None:
        """Mark fact as in an unresolved contradiction state."""
        self.status = FactStatus.CONTRADICTED
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.history.append(
            FactHistoryEntry(
                version=self.version,
                value=self.value,
                timestamp=self.updated_at,
                reason=f"CONTRADICTION DETECTED: {reason}",
                source_event_ids=list(self.source_event_ids),
            )
        )

    def mark_expired(self, reason: str = "TTL Expired") -> None:
        """Mark fact as expired."""
        self.status = FactStatus.EXPIRED
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize fact instance to dictionary."""
        return {
            "fact_id": self.fact_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "version": self.version,
            "status": str(self.status),
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_event_ids": self.source_event_ids,
            "history": [entry.to_dict() for entry in self.history],
            "valid_until": self.valid_until,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize fact instance from dictionary."""
        return cls(
            fact_id=data.get("fact_id", str(uuid.uuid4())),
            subject=data["subject"],
            predicate=data["predicate"],
            value=data["value"],
            version=data.get("version", 1),
            status=FactStatus(data.get("status", FactStatus.ACTIVE)),
            confidence=data.get("confidence", 1.0),
            created_at=data.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
            updated_at=data.get(
                "updated_at", datetime.now(timezone.utc).isoformat()
            ),
            source_event_ids=data.get("source_event_ids", []),
            history=[
                FactHistoryEntry.from_dict(h) for h in data.get("history", [])
            ],
            valid_until=data.get("valid_until"),
            metadata=data.get("metadata", {}),
        )


class SemanticMemory:
    """Long-term storage for consolidated semantic facts.

    NOTE: Direct writes from agent conversation or router are prohibited.
    Only the Semantic Consolidation engine is authorized to call modification methods.
    """

    def __init__(self, storage_path: Optional[str | Path] = None) -> None:
        """Initialize Semantic Memory store.

        Args:
            storage_path: Optional JSON storage file path for state persistence.
        """
        self._facts: dict[str, SemanticFact] = {}
        self._storage_path: Optional[Path] = (
            Path(storage_path) if storage_path else None
        )

        if self._storage_path and self._storage_path.exists():
            self.load_from_file(self._storage_path)

    @property
    def total_facts_count(self) -> int:
        """Total count of stored facts across all statuses."""
        return len(self._facts)

    @property
    def active_facts_count(self) -> int:
        """Count of active facts currently valid."""
        return sum(
            1 for f in self._facts.values() if f.status == FactStatus.ACTIVE
        )

    def add_fact(
        self,
        subject: str,
        predicate: str,
        value: Any,
        source_event_ids: Optional[List[str]] = None,
        confidence: float = 1.0,
        valid_until: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SemanticFact:
        """Add a new consolidated fact into Semantic Memory.

        Called exclusively by Semantic Consolidation Engine.
        """
        fact = SemanticFact(
            subject=subject,
            predicate=predicate,
            value=value,
            source_event_ids=source_event_ids or [],
            confidence=max(0.0, min(1.0, confidence)),
            valid_until=valid_until,
            metadata=metadata or {},
        )
        self._facts[fact.fact_id] = fact
        self._auto_save()
        return fact

    def get_fact(self, fact_id: str) -> Optional[SemanticFact]:
        """Fetch a specific fact by ID."""
        return self._facts.get(fact_id)

    def get_active_fact(
        self, subject: str, predicate: str
    ) -> Optional[SemanticFact]:
        """Retrieve active fact matching subject and predicate, if any."""
        for fact in self._facts.values():
            if (
                fact.subject == subject
                and fact.predicate == predicate
                and fact.status == FactStatus.ACTIVE
            ):
                return fact
        return None

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        status: Optional[FactStatus | str] = None,
        keyword: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SemanticFact]:
        """Query semantic facts by subject, predicate, status, or keyword.

        Returns:
            List of matching SemanticFact instances sorted by updated_at descending.
        """
        status_str = str(status) if status else None
        results: List[SemanticFact] = []

        for fact in self._facts.values():
            if subject and fact.subject.lower() != subject.lower():
                continue
            if predicate and fact.predicate.lower() != predicate.lower():
                continue
            if status_str and str(fact.status) != status_str:
                continue
            if keyword:
                kw_lower = keyword.lower()
                in_sub = kw_lower in fact.subject.lower()
                in_pred = kw_lower in fact.predicate.lower()
                in_val = kw_lower in str(fact.value).lower()
                if not (in_sub or in_pred or in_val):
                    continue

            results.append(fact)

        results.sort(key=lambda f: f.updated_at, reverse=True)
        if limit and limit > 0:
            return results[:limit]
        return results

    def get_all_active_facts(self) -> List[SemanticFact]:
        """Retrieve all currently active facts."""
        return [
            f for f in self._facts.values() if f.status == FactStatus.ACTIVE
        ]

    def clear(self) -> None:
        """Clear all semantic memory facts."""
        self._facts.clear()
        self._auto_save()

    def _auto_save(self) -> None:
        """Auto-save to file if storage path is bound."""
        if self._storage_path:
            self.save_to_file(self._storage_path)

    def save_to_file(self, filepath: str | Path) -> None:
        """Save semantic memory to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    def load_from_file(self, filepath: str | Path) -> None:
        """Load semantic memory from JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Semantic memory file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        reconstructed = self.from_dict(data)
        self._facts = reconstructed._facts

    def to_dict(self) -> dict[str, Any]:
        """Serialize entire semantic memory collection to dictionary."""
        return {
            "total_facts_count": len(self._facts),
            "active_facts_count": self.active_facts_count,
            "facts": [fact.to_dict() for fact in self._facts.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize semantic memory store from dictionary."""
        instance = cls()
        facts_list = data.get("facts", [])
        for f_data in facts_list:
            fact = SemanticFact.from_dict(f_data)
            instance._facts[fact.fact_id] = fact
        return instance

    def to_json(self) -> str:
        """Serialize semantic memory store to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserialize semantic memory store from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

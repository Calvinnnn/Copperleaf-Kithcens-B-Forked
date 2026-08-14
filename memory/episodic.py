"""Episodic Memory Module for AI Agent Architecture.

This module provides persistent storage for significant episodic events, decisions,
user preferences, resolved incidents, and workflow histories. Unlike short-term memory,
Episodic Memory stores structured experience logs long-term and serves as the primary
source for the Semantic Consolidation engine.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json
from pathlib import Path
from typing import Any, List, Optional, Self
import uuid

try:
    from memory import db_backend as _db
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False


class EventType(StrEnum):
    """Categorized types of episodic events."""

    DECISION = "decision"
    PREFERENCE = "preference"
    INCIDENT = "incident"
    BUSINESS_EVENT = "business_event"
    PATTERN = "pattern"
    WORKFLOW = "workflow"
    ROUTER_PROMOTION = "router_promotion"
    USER_INSTRUCTION = "user_instruction"


@dataclass
class EpisodicMemoryItem:
    """Represents a single persistent experience record in episodic memory.

    Attributes:
        event_id: Unique record identifier.
        event_type: Category classification of the event.
        summary: Concise natural language description of what happened.
        details: Structured context, parameters, or outcomes.
        importance_score: Subjective or computed relevance score (0.0 to 1.0).
        timestamp: ISO 8601 UTC creation timestamp.
        tags: Searchable tag keywords.
        source: Origin of the event record (e.g. 'router', 'scratchpad', 'user').
        is_consolidated: True if this item has been processed into Semantic Memory.
        consolidated_at: ISO 8601 UTC timestamp of semantic consolidation.
        metadata: Arbitrary metadata dictionary.
    """

    event_type: EventType
    summary: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    details: dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.5
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tags: List[str] = field(default_factory=list)
    source: str = "agent"
    is_consolidated: bool = False
    consolidated_at: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_consolidated(self) -> None:
        """Mark item as processed by the Semantic Consolidation engine."""
        self.is_consolidated = True
        self.consolidated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize event item to a dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": str(self.event_type),
            "summary": self.summary,
            "details": self.details,
            "importance_score": self.importance_score,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "source": self.source,
            "is_consolidated": self.is_consolidated,
            "consolidated_at": self.consolidated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize event item from a dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=EventType(data["event_type"]),
            summary=data["summary"],
            details=data.get("details", {}),
            importance_score=data.get("importance_score", 0.5),
            timestamp=data.get(
                "timestamp", datetime.now(timezone.utc).isoformat()
            ),
            tags=data.get("tags", []),
            source=data.get("source", "agent"),
            is_consolidated=data.get("is_consolidated", False),
            consolidated_at=data.get("consolidated_at"),
            metadata=data.get("metadata", {}),
        )


class EpisodicMemory:
    """Persistent store managing historical agent experiences and episodic events.

    Provides indexed querying by tag, event type, consolidated status, keyword,
    and importance rating. Serves as the database scanned by Semantic Consolidation.
    """

    def __init__(self, storage_path: Optional[str | Path] = None) -> None:
        """Initialize EpisodicMemory store.

        Args:
            storage_path: Optional file path for auto-saving / loading JSON state.
        """
        self._events: dict[str, EpisodicMemoryItem] = {}
        self._storage_path: Optional[Path] = (
            Path(storage_path) if storage_path else None
        )

        if self._storage_path and self._storage_path.exists():
            self.load_from_file(self._storage_path)

    @property
    def total_count(self) -> int:
        """Total number of stored episodic events."""
        return len(self._events)

    def store_event(
        self,
        event_type: EventType | str,
        summary: str,
        details: Optional[dict[str, Any]] = None,
        importance_score: float = 0.5,
        tags: Optional[List[str]] = None,
        source: str = "agent",
        metadata: Optional[dict[str, Any]] = None,
    ) -> EpisodicMemoryItem:
        """Store a new meaningful event into episodic memory.

        Args:
            event_type: Classification category for this event.
            summary: Human-readable summary description.
            details: Extra structured context dictionary.
            importance_score: Importance rating (0.0 to 1.0).
            tags: List of descriptive keyword tags.
            source: Event creator identifier.
            metadata: Custom attributes dictionary.

        Returns:
            The created EpisodicMemoryItem instance.
        """
        type_enum = (
            event_type
            if isinstance(event_type, EventType)
            else EventType(event_type)
        )
        item = EpisodicMemoryItem(
            event_type=type_enum,
            summary=summary,
            details=details or {},
            importance_score=max(0.0, min(1.0, importance_score)),
            tags=tags or [],
            source=source,
            metadata=metadata or {},
        )
        self._events[item.event_id] = item
        # Persist to SQLite so event survives process restarts
        if _DB_AVAILABLE:
            try:
                _db.ep_insert(
                    event_id=item.event_id,
                    event_type=str(item.event_type),
                    summary=item.summary,
                    details=item.details,
                    importance_score=item.importance_score,
                    tags=item.tags,
                    source=item.source,
                    metadata=item.metadata,
                )
            except Exception:
                pass  # SQLite failure must not crash the agent
        self._auto_save()
        return item

    def get_event(self, event_id: str) -> Optional[EpisodicMemoryItem]:
        """Fetch a specific event by its unique event_id."""
        return self._events.get(event_id)

    def query(
        self,
        event_type: Optional[EventType | str] = None,
        tag: Optional[str] = None,
        min_importance: float = 0.0,
        is_consolidated: Optional[bool] = None,
        keyword: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[EpisodicMemoryItem]:
        """Search stored episodic memory with multi-attribute filtering.

        Args:
            event_type: Filter by specific EventType.
            tag: Match events containing this tag.
            min_importance: Minimum importance rating threshold.
            is_consolidated: Filter by consolidation state (True/False).
            keyword: Case-insensitive keyword matching summary or details.
            limit: Maximum records to return.

        Returns:
            List of matching EpisodicMemoryItem records sorted newest-first.
        """
        type_str = str(event_type) if event_type else None
        results: List[EpisodicMemoryItem] = []

        for item in self._events.values():
            if type_str and str(item.event_type) != type_str:
                continue
            if tag and tag not in item.tags:
                continue
            if item.importance_score < min_importance:
                continue
            if (
                is_consolidated is not None
                and item.is_consolidated != is_consolidated
            ):
                continue
            if keyword:
                kw_lower = keyword.lower()
                in_summary = kw_lower in item.summary.lower()
                in_details = kw_lower in str(item.details).lower()
                if not (in_summary or in_details):
                    continue

            results.append(item)

        # Sort by timestamp descending (newest first)
        results.sort(key=lambda x: x.timestamp, reverse=True)

        if limit and limit > 0:
            return results[:limit]
        return results

    def get_unconsolidated_events(
        self, limit: Optional[int] = None
    ) -> List[EpisodicMemoryItem]:
        """Retrieve events that have not yet been consolidated into Semantic Memory."""
        return self.query(is_consolidated=False, limit=limit)

    def mark_consolidated(self, event_ids: List[str]) -> int:
        """Flag specified events as consolidated and sync to SQLite.

        Args:
            event_ids: List of event IDs to mark.

        Returns:
            Number of records updated.
        """
        count = 0
        for eid in event_ids:
            if eid in self._events:
                self._events[eid].mark_consolidated()
                count += 1
        if count > 0:
            if _DB_AVAILABLE:
                try:
                    _db.ep_mark_consolidated(event_ids)
                except Exception:
                    pass
            self._auto_save()
        return count

    def delete_event(self, event_id: str) -> bool:
        """Remove an event record by ID from memory and SQLite."""
        if event_id in self._events:
            del self._events[event_id]
            if _DB_AVAILABLE:
                try:
                    _db.ep_delete(event_id)
                except Exception:
                    pass
            self._auto_save()
            return True
        return False

    def clear(self) -> None:
        """Clear all stored episodic memory items."""
        self._events.clear()
        self._auto_save()

    def _auto_save(self) -> None:
        """Persist state to storage file if path was provided."""
        if self._storage_path:
            self.save_to_file(self._storage_path)

    def save_to_file(self, filepath: str | Path) -> None:
        """Save episodic memory state to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    def load_from_file(self, filepath: str | Path) -> None:
        """Load episodic memory state from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Episodic memory file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        reconstructed = self.from_dict(data)
        self._events = reconstructed._events

    def to_dict(self) -> dict[str, Any]:
        """Serialize episodic memory collection to dictionary format."""
        return {
            "total_count": len(self._events),
            "events": [item.to_dict() for item in self._events.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Reconstruct EpisodicMemory store from serialized dictionary."""
        instance = cls()
        events_list = data.get("events", [])
        for evt_data in events_list:
            item = EpisodicMemoryItem.from_dict(evt_data)
            instance._events[item.event_id] = item
        return instance

    def to_json(self) -> str:
        """Serialize episodic memory store to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserialize episodic memory store from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

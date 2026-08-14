"""Short-Term Memory Module for AI Agent Architecture.

This module provides a rolling conversation buffer that captures recent interaction context,
including user messages, assistant responses, tool calls, and tool observations. It operates
on a strict FIFO (First-In, First-Out) capacity ceiling, triggering overflow notifications
whenever capacity is exceeded so that evicted items can be evaluated by the Promote-or-Drop Router.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json
from typing import Any, Callable, List, Optional, Self
import uuid


class MessageRole(StrEnum):
    """Supported roles in the short-term conversation buffer."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ShortTermMemoryItem:
    """Represents a single atomic item within short-term memory.

    Attributes:
        item_id: Unique identifier for the memory item.
        role: Role of the message generator (user, assistant, system, tool).
        content: Main text or structured body of the item.
        timestamp: ISO 8601 UTC timestamp of creation.
        tool_call_id: Optional ID linking tool calls to their observations.
        name: Optional function or sender name.
        metadata: Arbitrary metadata dictionary for tags or tokens.
    """

    role: MessageRole
    content: Any
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the memory item to a JSON-compatible dictionary."""
        return {
            "item_id": self.item_id,
            "role": str(self.role),
            "content": self.content,
            "timestamp": self.timestamp,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize a dictionary into a ShortTermMemoryItem instance."""
        return cls(
            item_id=data.get("item_id", str(uuid.uuid4())),
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data.get(
                "timestamp", datetime.now(timezone.utc).isoformat()
            ),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            metadata=data.get("metadata", {}),
        )


# Type alias for router callback when items overflow
OverflowCallback = Callable[[List[ShortTermMemoryItem]], None]


class ShortTermMemory:
    """Rolling FIFO conversation buffer with overflow detection and routing callbacks.

    Maintains recent turn context for LLM interaction while enforcing a fixed
    item capacity. Evicted items are pushed to an overflow handler (Promote-or-Drop Router).
    """

    def __init__(
        self,
        capacity: int = 20,
        overflow_callback: Optional[OverflowCallback] = None,
    ) -> None:
        """Initialize short-term memory buffer.

        Args:
            capacity: Maximum number of memory items allowed before overflow.
            overflow_callback: Optional callable invoked when items are evicted.
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be a positive integer, got {capacity}")

        self._capacity: int = capacity
        self._buffer: List[ShortTermMemoryItem] = []
        self._overflow_callback: Optional[OverflowCallback] = overflow_callback
        self._total_overflow_count: int = 0

    @property
    def capacity(self) -> int:
        """Get the current item capacity limit."""
        return self._capacity

    @capacity.setter
    def capacity(self, new_capacity: int) -> None:
        """Set a new capacity limit, triggering eviction if buffer size exceeds it."""
        if new_capacity <= 0:
            raise ValueError(
                f"Capacity must be a positive integer, got {new_capacity}"
            )
        self._capacity = new_capacity
        self._check_overflow()

    @property
    def size(self) -> int:
        """Return current number of items stored in buffer."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """Check if buffer is at or above capacity limit."""
        return len(self._buffer) >= self._capacity

    @property
    def total_overflow_count(self) -> int:
        """Total number of items evicted over the buffer's lifetime."""
        return self._total_overflow_count

    def add_item(self, item: ShortTermMemoryItem) -> List[ShortTermMemoryItem]:
        """Add a memory item to the buffer and process any resulting overflow.

        Args:
            item: The ShortTermMemoryItem to append.

        Returns:
            List of evicted items resulting from this addition (if any).
        """
        self._buffer.append(item)
        return self._check_overflow()

    def add_user_message(
        self, content: str, metadata: Optional[dict[str, Any]] = None
    ) -> List[ShortTermMemoryItem]:
        """Convenience method to record a user message."""
        item = ShortTermMemoryItem(
            role=MessageRole.USER,
            content=content,
            metadata=metadata or {},
        )
        return self.add_item(item)

    def add_assistant_message(
        self,
        content: str,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> List[ShortTermMemoryItem]:
        """Convenience method to record an assistant response."""
        item = ShortTermMemoryItem(
            role=MessageRole.ASSISTANT,
            content=content,
            name=name,
            metadata=metadata or {},
        )
        return self.add_item(item)

    def add_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> List[ShortTermMemoryItem]:
        """Convenience method to record an assistant tool invocation."""
        item = ShortTermMemoryItem(
            role=MessageRole.ASSISTANT,
            content={"tool": tool_name, "arguments": arguments},
            tool_call_id=tool_call_id,
            name=tool_name,
            metadata=metadata or {},
        )
        return self.add_item(item)

    def add_tool_observation(
        self,
        tool_name: str,
        output: Any,
        tool_call_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> List[ShortTermMemoryItem]:
        """Convenience method to record a tool execution result."""
        item = ShortTermMemoryItem(
            role=MessageRole.TOOL,
            content=output,
            tool_call_id=tool_call_id,
            name=tool_name,
            metadata=metadata or {},
        )
        return self.add_item(item)

    def get_history(self) -> List[ShortTermMemoryItem]:
        """Return a copy of the current short-term memory buffer."""
        return list(self._buffer)

    def get_last_n(self, n: int) -> List[ShortTermMemoryItem]:
        """Return the last n items from the buffer."""
        if n <= 0:
            return []
        return list(self._buffer[-n:])

    def clear(self) -> None:
        """Clear all items from short-term memory without triggering overflow."""
        self._buffer.clear()

    def set_overflow_callback(
        self, callback: Optional[OverflowCallback]
    ) -> None:
        """Register or clear the callback function for overflow events."""
        self._overflow_callback = callback

    def _check_overflow(self) -> List[ShortTermMemoryItem]:
        """Check if buffer size exceeds capacity, evicting excess FIFO items."""
        overflow_items: List[ShortTermMemoryItem] = []
        while len(self._buffer) > self._capacity:
            evicted = self._buffer.pop(0)
            overflow_items.append(evicted)
            self._total_overflow_count += 1

        if overflow_items and self._overflow_callback:
            self._overflow_callback(overflow_items)

        return overflow_items

    def to_dict(self) -> dict[str, Any]:
        """Serialize the short-term memory state to a dictionary."""
        return {
            "capacity": self._capacity,
            "total_overflow_count": self._total_overflow_count,
            "buffer": [item.to_dict() for item in self._buffer],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        overflow_callback: Optional[OverflowCallback] = None,
    ) -> Self:
        """Reconstruct ShortTermMemory from a serialized state dictionary."""
        instance = cls(
            capacity=data.get("capacity", 20),
            overflow_callback=overflow_callback,
        )
        instance._total_overflow_count = data.get("total_overflow_count", 0)
        instance._buffer = [
            ShortTermMemoryItem.from_dict(item_data)
            for item_data in data.get("buffer", [])
        ]
        return instance

    def to_json(self) -> str:
        """Serialize short-term memory state to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(
        cls,
        json_str: str,
        overflow_callback: Optional[OverflowCallback] = None,
    ) -> Self:
        """Deserialize short-term memory state from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data, overflow_callback=overflow_callback)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> ShortTermMemoryItem:
        return self._buffer[index]

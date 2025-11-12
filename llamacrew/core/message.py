"""Message protocol for agent-to-agent communication."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4


class MessageType(str, Enum):
    """Types of messages that can be sent between agents."""

    TASK = "task"  # Task assignment
    QUESTION = "question"  # Request for information
    RESULT = "result"  # Task completion result
    ERROR = "error"  # Error notification
    INFO = "info"  # General information
    DELEGATION = "delegation"  # Delegate to another agent


@dataclass
class Message:
    """
    Message protocol for agent communication.

    Attributes:
        from_agent: ID of the sending agent
        to_agent: ID of the receiving agent (or "broadcast" for all)
        content: Message content/payload
        message_type: Type of message
        metadata: Additional metadata
        message_id: Unique message identifier
        timestamp: When the message was created
        in_reply_to: Optional ID of message this is replying to
    """

    from_agent: str
    to_agent: str
    content: str
    message_type: MessageType
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    in_reply_to: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate message after initialization."""
        if not self.from_agent:
            raise ValueError("from_agent cannot be empty")
        if not self.to_agent:
            raise ValueError("to_agent cannot be empty")
        if not self.content:
            raise ValueError("content cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "content": self.content,
            "message_type": self.message_type.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "in_reply_to": self.in_reply_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            message_id=data["message_id"],
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            content=data["content"],
            message_type=MessageType(data["message_type"]),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            in_reply_to=data.get("in_reply_to"),
        )

    def reply(
        self,
        from_agent: str,
        content: str,
        message_type: MessageType = MessageType.RESULT,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Message":
        """Create a reply to this message."""
        return Message(
            from_agent=from_agent,
            to_agent=self.from_agent,
            content=content,
            message_type=message_type,
            metadata=metadata or {},
            in_reply_to=self.message_id,
        )

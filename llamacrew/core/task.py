"""Task abstraction for LlamaCrew."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .agent import Agent


class TaskStatus(str, Enum):
    """Status of a task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """
    Represents a task to be executed by an agent.

    Attributes:
        description: Description of what needs to be done
        agent: Agent responsible for this task
        expected_output: Description of expected output format/content
        dependencies: List of tasks that must complete before this one
        context: Additional context variables for the task
        async_execution: Whether to execute asynchronously
        task_id: Unique identifier for the task
        status: Current status of the task
        result: Result of task execution
        error: Error message if task failed
    """

    description: str
    agent: Agent
    expected_output: str = ""
    dependencies: List["Task"] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    async_execution: bool = False
    task_id: str = field(default_factory=lambda: str(uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate task after initialization."""
        if not self.description:
            raise ValueError("Task description cannot be empty")
        if not self.agent:
            raise ValueError("Task must have an assigned agent")

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "agent_id": self.agent.agent_id,
            "expected_output": self.expected_output,
            "dependencies": [t.task_id for t in self.dependencies],
            "context": self.context,
            "async_execution": self.async_execution,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], agent: Agent, dependencies: List["Task"]) -> "Task":
        """Create task from dictionary."""
        return cls(
            task_id=data.get("task_id", str(uuid4())),
            description=data["description"],
            agent=agent,
            expected_output=data.get("expected_output", ""),
            dependencies=dependencies,
            context=data.get("context", {}),
            async_execution=data.get("async_execution", False),
            status=TaskStatus(data.get("status", "pending")),
            result=data.get("result"),
            error=data.get("error"),
        )

    def is_ready(self) -> bool:
        """Check if all dependencies are completed."""
        return all(dep.status == TaskStatus.COMPLETED for dep in self.dependencies)

    def mark_in_progress(self) -> None:
        """Mark task as in progress."""
        self.status = TaskStatus.IN_PROGRESS

    def mark_completed(self, result: str) -> None:
        """Mark task as completed with result."""
        self.status = TaskStatus.COMPLETED
        self.result = result

    def mark_failed(self, error: str) -> None:
        """Mark task as failed with error."""
        self.status = TaskStatus.FAILED
        self.error = error

    def mark_skipped(self) -> None:
        """Mark task as skipped."""
        self.status = TaskStatus.SKIPPED

    def get_prompt(self) -> str:
        """
        Generate a prompt for the agent to execute this task.

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        prompt_parts.append(f"# Task\n{self.description}")

        if self.expected_output:
            prompt_parts.append(f"\n# Expected Output\n{self.expected_output}")

        if self.context:
            prompt_parts.append("\n# Context")
            for key, value in self.context.items():
                prompt_parts.append(f"- {key}: {value}")

        if self.dependencies:
            prompt_parts.append("\n# Previous Results")
            for dep in self.dependencies:
                if dep.result:
                    prompt_parts.append(f"- {dep.agent.role}: {dep.result}")

        return "\n".join(prompt_parts)

    def __str__(self) -> str:
        """String representation of task."""
        return f"Task(description='{self.description[:50]}...', status={self.status.value})"

    def __repr__(self) -> str:
        """Detailed representation of task."""
        return (
            f"Task(task_id='{self.task_id}', agent='{self.agent.role}', "
            f"status={self.status.value})"
        )


@dataclass
class TaskResult:
    """
    Result of task execution.

    Attributes:
        task_id: ID of the task
        success: Whether task succeeded
        output: Task output/result
        error: Error message if failed
        metadata: Additional metadata
    """

    task_id: str
    success: bool
    output: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }

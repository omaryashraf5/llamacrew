"""Crew orchestration for LlamaCrew."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .agent import Agent
from .task import Task, TaskStatus


class ProcessType(str, Enum):
    """Type of process for crew execution."""

    SEQUENTIAL = "sequential"  # Tasks execute one after another
    PARALLEL = "parallel"  # Independent tasks execute concurrently
    HIERARCHICAL = "hierarchical"  # Manager agent delegates to workers


@dataclass
class CrewOutput:
    """
    Output from crew execution.

    Attributes:
        tasks_output: List of task results
        final_output: Final combined output
        success: Whether all tasks succeeded
        metadata: Additional metadata
    """

    tasks_output: List[Dict[str, Any]]
    final_output: str
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert output to dictionary."""
        return {
            "tasks_output": self.tasks_output,
            "final_output": self.final_output,
            "success": self.success,
            "metadata": self.metadata,
        }


@dataclass
class Crew:
    """
    Represents a crew of agents working together.

    Attributes:
        agents: List of agents in the crew
        tasks: List of tasks to be executed
        process: Type of process (sequential, parallel, hierarchical)
        memory: Whether to enable shared memory
        cache: Whether to enable caching
        verbose: Whether to print execution details
        checkpoint_enabled: Whether to enable checkpointing
        max_rpm: Maximum requests per minute
        crew_id: Unique identifier for the crew
    """

    agents: List[Agent]
    tasks: List[Task]
    process: ProcessType = ProcessType.SEQUENTIAL
    memory: bool = True
    cache: bool = False
    verbose: bool = True
    checkpoint_enabled: bool = False
    max_rpm: Optional[int] = None
    crew_id: str = field(default_factory=lambda: __import__("uuid").uuid4().hex)

    def __post_init__(self) -> None:
        """Validate crew after initialization."""
        if not self.agents:
            raise ValueError("Crew must have at least one agent")
        if not self.tasks:
            raise ValueError("Crew must have at least one task")

        # Validate all tasks have agents in the crew
        crew_agent_ids = {agent.agent_id for agent in self.agents}
        for task in self.tasks:
            if task.agent.agent_id not in crew_agent_ids:
                raise ValueError(f"Task agent '{task.agent.role}' not in crew agents")

        # Check for circular dependencies
        self._validate_task_dependencies()

    def _validate_task_dependencies(self) -> None:
        """Check for circular dependencies in tasks."""
        task_map = {task.task_id: task for task in self.tasks}

        def has_cycle(task: Task, visited: set, rec_stack: set) -> bool:
            visited.add(task.task_id)
            rec_stack.add(task.task_id)

            for dep in task.dependencies:
                if dep.task_id not in visited:
                    if has_cycle(dep, visited, rec_stack):
                        return True
                elif dep.task_id in rec_stack:
                    return True

            rec_stack.remove(task.task_id)
            return False

        visited: set = set()
        for task in self.tasks:
            if task.task_id not in visited:
                if has_cycle(task, visited, set()):
                    raise ValueError("Circular dependency detected in tasks")

    def kickoff(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        memory_backend: Optional[Any] = None,
    ) -> CrewOutput:
        """
        Start the crew execution.

        Args:
            inputs: Optional input data for the crew
            memory_backend: Optional memory backend (e.g., LlamaStackMemoryBackend).
                          If None and memory=True, uses in-memory storage.

        Returns:
            CrewOutput with results from all tasks
        """
        # This is a placeholder implementation
        # The actual execution will be handled by the orchestration engine
        from ..orchestration.engine import WorkflowEngine

        engine = WorkflowEngine(
            crew=self,
            verbose=self.verbose,
            checkpoint_enabled=self.checkpoint_enabled,
            memory_backend=memory_backend,
        )

        return engine.execute(inputs or {})

    def save(self, path: str) -> None:
        """
        Save crew state to a checkpoint file.

        Args:
            path: Path to save checkpoint
        """
        from ..memory.checkpoint import CheckpointManager

        checkpoint_manager = CheckpointManager(path)
        checkpoint_manager.save(self)

    @classmethod
    def resume(cls, path: str) -> "Crew":
        """
        Resume crew from a checkpoint file.

        Args:
            path: Path to checkpoint file

        Returns:
            Restored crew instance
        """
        from ..memory.checkpoint import CheckpointManager

        checkpoint_manager = CheckpointManager(path)
        return checkpoint_manager.load()

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks that are ready to execute."""
        return [
            task for task in self.tasks if task.status == TaskStatus.PENDING and task.is_ready()
        ]

    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(task.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED] for task in self.tasks)

    def has_failed_tasks(self) -> bool:
        """Check if any tasks have failed."""
        return any(task.status == TaskStatus.FAILED for task in self.tasks)

    def to_dict(self) -> Dict[str, Any]:
        """Convert crew to dictionary."""
        return {
            "crew_id": self.crew_id,
            "agents": [agent.to_dict() for agent in self.agents],
            "tasks": [task.to_dict() for task in self.tasks],
            "process": self.process.value,
            "memory": self.memory,
            "cache": self.cache,
            "verbose": self.verbose,
            "checkpoint_enabled": self.checkpoint_enabled,
            "max_rpm": self.max_rpm,
        }

    def __str__(self) -> str:
        """String representation of crew."""
        return (
            f"Crew(agents={len(self.agents)}, tasks={len(self.tasks)}, "
            f"process={self.process.value})"
        )

    def __repr__(self) -> str:
        """Detailed representation of crew."""
        return (
            f"Crew(crew_id='{self.crew_id}', agents={len(self.agents)}, "
            f"tasks={len(self.tasks)}, process={self.process.value})"
        )

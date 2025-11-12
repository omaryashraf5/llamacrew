"""Checkpoint manager for saving and resuming crew execution."""

import json
from pathlib import Path
from typing import Any, Dict

from ..core.agent import Agent
from ..core.crew import Crew
from ..core.task import Task


class CheckpointManager:
    """
    Manages checkpoints for crew execution.

    Allows saving and restoring crew state for pause/resume functionality.
    """

    def __init__(self, checkpoint_path: str):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.checkpoint_path = Path(checkpoint_path)

    def save(self, crew: Crew) -> None:
        """
        Save crew state to checkpoint.

        Args:
            crew: Crew to save
        """
        checkpoint_data = {
            "crew": crew.to_dict(),
            "agents": [agent.to_dict() for agent in crew.agents],
            "tasks": [task.to_dict() for task in crew.tasks],
        }

        # Ensure parent directory exists
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Write checkpoint
        with open(self.checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

    def load(self) -> Crew:
        """
        Load crew from checkpoint.

        Returns:
            Restored crew instance
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        with open(self.checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        # Reconstruct agents
        agents = []
        agent_map = {}
        for agent_data in checkpoint_data["agents"]:
            agent = Agent.from_dict(agent_data)
            agents.append(agent)
            agent_map[agent.agent_id] = agent

        # Reconstruct tasks
        tasks = []
        task_map = {}
        for task_data in checkpoint_data["tasks"]:
            # Resolve agent reference
            agent = agent_map[task_data["agent_id"]]
            # Resolve dependencies (will be set in second pass)
            task = Task.from_dict(task_data, agent, [])
            tasks.append(task)
            task_map[task.task_id] = task

        # Second pass: set task dependencies
        for task_data, task in zip(checkpoint_data["tasks"], tasks):
            task.dependencies = [task_map[dep_id] for dep_id in task_data["dependencies"]]

        # Reconstruct crew
        crew_data = checkpoint_data["crew"]
        from ..core.crew import ProcessType

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=ProcessType(crew_data["process"]),
            memory=crew_data["memory"],
            cache=crew_data.get("cache", False),
            verbose=crew_data.get("verbose", True),
            checkpoint_enabled=crew_data.get("checkpoint_enabled", False),
            max_rpm=crew_data.get("max_rpm"),
        )
        crew.crew_id = crew_data["crew_id"]

        return crew

    def exists(self) -> bool:
        """Check if checkpoint file exists."""
        return self.checkpoint_path.exists()

    def delete(self) -> None:
        """Delete checkpoint file."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

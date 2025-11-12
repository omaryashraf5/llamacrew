"""YAML workflow parser for LlamaCrew."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..core.agent import Agent
from ..core.crew import Crew, ProcessType
from ..core.task import Task


class YAMLWorkflowParser:
    """
    Parse YAML workflow definitions into LlamaCrew objects.

    Example YAML structure:
        crew:
          name: "Content Creation Team"
          process: "sequential"
          memory: true

        agents:
          - name: planner
            role: "Content Strategist"
            goal: "Create content strategies"
            backstory: "Expert strategist"
            tools: ["search"]

        tasks:
          - description: "Plan content strategy"
            agent: planner
            expected_output: "Detailed outline"
          - description: "Write content"
            agent: writer
            dependencies: [0]
    """

    def __init__(self):
        """Initialize the parser."""
        self._agents_cache: Dict[str, Agent] = {}
        self._tasks_cache: List[Task] = []

    def parse_file(self, filepath: str) -> Crew:
        """
        Parse a YAML file into a Crew.

        Args:
            filepath: Path to YAML file

        Returns:
            Crew instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Workflow file not found: {filepath}")

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        return self.parse_dict(config)

    def parse_dict(self, config: Dict[str, Any]) -> Crew:
        """
        Parse a configuration dictionary into a Crew.

        Args:
            config: Configuration dictionary

        Returns:
            Crew instance

        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_config(config)

        # Parse agents
        agents = self._parse_agents(config.get("agents", []))

        # Parse tasks
        tasks = self._parse_tasks(config.get("tasks", []))

        # Parse crew configuration
        crew_config = config.get("crew", {})

        # Create crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=self._parse_process_type(crew_config.get("process", "sequential")),
            memory=crew_config.get("memory", True),
            cache=crew_config.get("cache", False),
            verbose=crew_config.get("verbose", True),
            checkpoint_enabled=crew_config.get("checkpoint_enabled", False),
            max_rpm=crew_config.get("max_rpm"),
        )

        return crew

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure."""
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")

        if "agents" not in config:
            raise ValueError("Configuration must contain 'agents' key")

        if "tasks" not in config:
            raise ValueError("Configuration must contain 'tasks' key")

        if not config["agents"]:
            raise ValueError("At least one agent must be defined")

        if not config["tasks"]:
            raise ValueError("At least one task must be defined")

    def _parse_agents(self, agents_config: List[Dict[str, Any]]) -> List[Agent]:
        """Parse agent configurations."""
        agents = []
        self._agents_cache.clear()

        for agent_config in agents_config:
            agent = self._parse_single_agent(agent_config)
            agents.append(agent)

            # Cache by name if provided
            if "name" in agent_config:
                self._agents_cache[agent_config["name"]] = agent

        return agents

    def _parse_single_agent(self, config: Dict[str, Any]) -> Agent:
        """Parse a single agent configuration."""
        # Required fields
        if "role" not in config:
            raise ValueError(f"Agent missing 'role': {config}")
        if "goal" not in config:
            raise ValueError(f"Agent missing 'goal': {config}")

        # Parse llm_config
        llm_config = {}
        if "llm_config" in config:
            llm_config = config["llm_config"]
        elif "model" in config:
            # Allow shorthand: model: "llama3-70b"
            llm_config["model"] = config["model"]
        if "temperature" in config:
            llm_config["temperature"] = config["temperature"]

        return Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config.get("backstory", ""),
            tools=config.get("tools", []),
            llm_config=llm_config,
            verbose=config.get("verbose", True),
            max_iterations=config.get("max_iterations", 15),
            allow_delegation=config.get("allow_delegation", False),
            memory_enabled=config.get("memory_enabled", True),
        )

    def _parse_tasks(self, tasks_config: List[Dict[str, Any]]) -> List[Task]:
        """Parse task configurations."""
        tasks = []
        self._tasks_cache.clear()

        for task_config in tasks_config:
            task = self._parse_single_task(task_config)
            tasks.append(task)
            self._tasks_cache.append(task)

        return tasks

    def _parse_single_task(self, config: Dict[str, Any]) -> Task:
        """Parse a single task configuration."""
        # Required fields
        if "description" not in config:
            raise ValueError(f"Task missing 'description': {config}")
        if "agent" not in config:
            raise ValueError(f"Task missing 'agent': {config}")

        # Resolve agent reference
        agent_ref = config["agent"]
        if isinstance(agent_ref, str):
            # Reference by name
            if agent_ref not in self._agents_cache:
                raise ValueError(f"Unknown agent reference: {agent_ref}")
            agent = self._agents_cache[agent_ref]
        elif isinstance(agent_ref, int):
            # Reference by index (not recommended, but supported)
            raise ValueError("Agent reference by index not supported in this parser")
        else:
            raise ValueError(f"Invalid agent reference: {agent_ref}")

        # Resolve dependencies
        dependencies = []
        if "dependencies" in config:
            for dep_ref in config["dependencies"]:
                if isinstance(dep_ref, int):
                    # Reference by task index
                    if dep_ref < 0 or dep_ref >= len(self._tasks_cache):
                        raise ValueError(f"Invalid task dependency index: {dep_ref}")
                    dependencies.append(self._tasks_cache[dep_ref])
                else:
                    raise ValueError(f"Invalid dependency reference: {dep_ref}")

        return Task(
            description=config["description"],
            agent=agent,
            expected_output=config.get("expected_output", ""),
            dependencies=dependencies,
            context=config.get("context", {}),
            async_execution=config.get("async_execution", False),
        )

    def _parse_process_type(self, process: str) -> ProcessType:
        """Parse process type string."""
        process_lower = process.lower()
        if process_lower == "sequential":
            return ProcessType.SEQUENTIAL
        elif process_lower == "parallel":
            return ProcessType.PARALLEL
        elif process_lower == "hierarchical":
            return ProcessType.HIERARCHICAL
        else:
            raise ValueError(f"Unknown process type: {process}")


def load_workflow(filepath: str) -> Crew:
    """
    Convenience function to load a workflow from YAML file.

    Args:
        filepath: Path to YAML file

    Returns:
        Crew instance

    Example:
        crew = load_workflow("my_workflow.yaml")
        result = crew.kickoff()
    """
    parser = YAMLWorkflowParser()
    return parser.parse_file(filepath)

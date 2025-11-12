"""Agent abstraction and decorator for LlamaCrew."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type
from uuid import uuid4


# Global agent registry
_AGENT_REGISTRY: Dict[str, "Agent"] = {}


@dataclass
class Agent:
    """
    Represents an agent in the crew.

    Attributes:
        role: The role/specialty of the agent (e.g., "planner", "researcher")
        goal: What the agent is trying to accomplish
        backstory: Background context for the agent's persona
        tools: List of tool names the agent can use
        llm_config: Configuration for the LLM (model, temperature, etc.)
        agent_id: Unique identifier for the agent
        verbose: Whether to print agent activity
        max_iterations: Maximum number of iterations for agent reasoning
        allow_delegation: Whether agent can delegate to others
    """

    role: str
    goal: str
    backstory: str = ""
    tools: List[str] = field(default_factory=list)
    llm_config: Dict[str, Any] = field(default_factory=dict)
    agent_id: str = field(default_factory=lambda: str(uuid4()))
    verbose: bool = True
    max_iterations: int = 15
    allow_delegation: bool = False
    memory_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate agent after initialization."""
        if not self.role:
            raise ValueError("Agent role cannot be empty")
        if not self.goal:
            raise ValueError("Agent goal cannot be empty")

        # Set default LLM config
        if "model" not in self.llm_config:
            self.llm_config["model"] = "llama3-70b"
        if "temperature" not in self.llm_config:
            self.llm_config["temperature"] = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "tools": self.tools,
            "llm_config": self.llm_config,
            "verbose": self.verbose,
            "max_iterations": self.max_iterations,
            "allow_delegation": self.allow_delegation,
            "memory_enabled": self.memory_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        """Create agent from dictionary."""
        return cls(
            agent_id=data.get("agent_id", str(uuid4())),
            role=data["role"],
            goal=data["goal"],
            backstory=data.get("backstory", ""),
            tools=data.get("tools", []),
            llm_config=data.get("llm_config", {}),
            verbose=data.get("verbose", True),
            max_iterations=data.get("max_iterations", 15),
            allow_delegation=data.get("allow_delegation", False),
            memory_enabled=data.get("memory_enabled", True),
        )

    def __str__(self) -> str:
        """String representation of agent."""
        return f"Agent(role='{self.role}', goal='{self.goal[:50]}...')"

    def __repr__(self) -> str:
        """Detailed representation of agent."""
        return (
            f"Agent(agent_id='{self.agent_id}', role='{self.role}', "
            f"goal='{self.goal}', tools={self.tools})"
        )


def agent(
    role: str,
    goal: str,
    backstory: str = "",
    tools: Optional[List[str]] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    max_iterations: int = 15,
    allow_delegation: bool = False,
    memory_enabled: bool = True,
) -> Callable[[Type], Type]:
    """
    Decorator to define an agent.

    Usage:
        @agent(role="planner", goal="Create detailed plans")
        class Planner:
            pass

        # Or with more options
        @agent(
            role="researcher",
            goal="Research and gather information",
            backstory="Expert researcher with 10 years experience",
            tools=["search", "scrape"],
            llm_config={"model": "llama3-70b", "temperature": 0.5}
        )
        class Researcher:
            pass

    Args:
        role: The role/specialty of the agent
        goal: What the agent is trying to accomplish
        backstory: Background context for the agent's persona
        tools: List of tool names the agent can use
        llm_config: Configuration for the LLM
        verbose: Whether to print agent activity
        max_iterations: Maximum number of iterations for agent reasoning
        allow_delegation: Whether agent can delegate to others
        memory_enabled: Whether agent can access crew memory

    Returns:
        Decorated class with agent instance attached
    """

    def decorator(cls: Type) -> Type:
        """Inner decorator function."""
        # Create agent instance
        agent_instance = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools or [],
            llm_config=llm_config or {},
            verbose=verbose,
            max_iterations=max_iterations,
            allow_delegation=allow_delegation,
            memory_enabled=memory_enabled,
        )

        # Register agent
        _AGENT_REGISTRY[agent_instance.agent_id] = agent_instance

        # Attach agent to class
        cls._agent = agent_instance

        return cls

    return decorator


def get_agent(agent_id: str) -> Optional[Agent]:
    """Get agent from registry by ID."""
    return _AGENT_REGISTRY.get(agent_id)


def list_agents() -> List[Agent]:
    """List all registered agents."""
    return list(_AGENT_REGISTRY.values())


def clear_agent_registry() -> None:
    """Clear the agent registry (useful for testing)."""
    _AGENT_REGISTRY.clear()

"""Unit tests for Agent class."""

import pytest
from llamacrew import Agent


def test_agent_creation():
    """Test basic agent creation."""
    agent = Agent(
        role="Researcher",
        goal="Research topics thoroughly",
        backstory="Expert researcher",
        verbose=False,  # Explicitly set to False for testing
    )

    assert agent.role == "Researcher"
    assert agent.goal == "Research topics thoroughly"
    assert agent.backstory == "Expert researcher"
    assert agent.agent_id is not None
    assert agent.memory_enabled is True  # Default
    assert agent.verbose is False


def test_agent_with_tools():
    """Test agent with tools."""
    agent = Agent(
        role="Developer",
        goal="Write code",
        backstory="Expert developer",
        tools=["search", "calculator"],
    )

    assert agent.tools == ["search", "calculator"]


def test_agent_with_llm_config():
    """Test agent with custom LLM configuration."""
    agent = Agent(
        role="Writer",
        goal="Write content",
        backstory="Professional writer",
        llm_config={"model": "ollama/llama3.2:3b", "temperature": 0.7},
    )

    assert agent.llm_config["model"] == "ollama/llama3.2:3b"
    assert agent.llm_config["temperature"] == 0.7


def test_agent_delegation():
    """Test agent delegation settings."""
    agent_with_delegation = Agent(
        role="Manager",
        goal="Coordinate work",
        backstory="Team manager",
        allow_delegation=True,
    )

    agent_no_delegation = Agent(
        role="Worker",
        goal="Execute tasks",
        backstory="Individual contributor",
        allow_delegation=False,
    )

    assert agent_with_delegation.allow_delegation is True
    assert agent_no_delegation.allow_delegation is False


def test_agent_to_dict():
    """Test agent serialization to dictionary."""
    agent = Agent(
        role="Analyst",
        goal="Analyze data",
        backstory="Data analyst",
        tools=["data_tool"],
    )

    data = agent.to_dict()

    assert data["role"] == "Analyst"
    assert data["goal"] == "Analyze data"
    assert data["backstory"] == "Data analyst"
    assert data["tools"] == ["data_tool"]
    assert "agent_id" in data


def test_agent_unique_ids():
    """Test that each agent gets a unique ID."""
    agent1 = Agent(role="Agent1", goal="Goal1", backstory="Story1")
    agent2 = Agent(role="Agent2", goal="Goal2", backstory="Story2")

    assert agent1.agent_id != agent2.agent_id


def test_agent_decorator():
    """Test @agent decorator."""
    # TODO: Implement decorator functionality
    # The @agent decorator is not fully implemented yet
    # This test will be enabled when decorator is implemented
    pytest.skip("@agent decorator not yet fully implemented")

    # from llamacrew.core.agent import agent
    #
    # @agent
    # class MyAgent:
    #     """Custom agent class."""
    #
    #     role = "Custom Role"
    #     goal = "Custom Goal"
    #     backstory = "Custom Backstory"
    #
    # # The decorator should work without raising errors
    # # Actual functionality depends on implementation
    # assert MyAgent.role == "Custom Role"

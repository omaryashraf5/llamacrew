"""Unit tests for Crew class."""

import pytest
from llamacrew import Agent, Crew, ProcessType, Task


def test_crew_creation():
    """Test basic crew creation."""
    agent = Agent(role="Worker", goal="Work", backstory="Worker")
    task = Task(description="Do work", agent=agent)

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=ProcessType.SEQUENTIAL,
    )

    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1
    assert crew.process == ProcessType.SEQUENTIAL
    assert crew.crew_id is not None


def test_crew_validation_no_agents():
    """Test that crew requires at least one agent."""
    task = Task(description="Do work", agent=Agent(role="Worker", goal="Work", backstory="Worker"))

    with pytest.raises(ValueError, match="at least one agent"):
        Crew(agents=[], tasks=[task])


def test_crew_validation_no_tasks():
    """Test that crew requires at least one task."""
    agent = Agent(role="Worker", goal="Work", backstory="Worker")

    with pytest.raises(ValueError, match="at least one task"):
        Crew(agents=[agent], tasks=[])


def test_crew_validation_task_agent_not_in_crew():
    """Test that task agents must be in crew."""
    agent1 = Agent(role="Worker1", goal="Work", backstory="Worker")
    agent2 = Agent(role="Worker2", goal="Work", backstory="Worker")
    task = Task(description="Do work", agent=agent2)

    with pytest.raises(ValueError, match="not in crew agents"):
        Crew(agents=[agent1], tasks=[task])


def test_crew_circular_dependency_detection():
    """Test circular dependency detection."""
    agent = Agent(role="Worker", goal="Work", backstory="Worker")

    task1 = Task(description="Task 1", agent=agent)
    task2 = Task(description="Task 2", agent=agent, dependencies=[task1])
    task1.dependencies = [task2]  # Create circular dependency

    with pytest.raises(ValueError, match="Circular dependency"):
        Crew(agents=[agent], tasks=[task1, task2])


def test_crew_get_ready_tasks():
    """Test getting ready tasks."""
    agent = Agent(role="Worker", goal="Work", backstory="Worker")

    task1 = Task(description="Task 1", agent=agent)
    task2 = Task(description="Task 2", agent=agent, dependencies=[task1])

    crew = Crew(agents=[agent], tasks=[task1, task2])

    # Only task1 should be ready initially
    ready = crew.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0] == task1

    # Complete task1
    task1.mark_completed("Done")

    # Now task2 should be ready
    ready = crew.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0] == task2


def test_crew_is_complete():
    """Test crew completion detection."""
    agent = Agent(role="Worker", goal="Work", backstory="Worker")
    task = Task(description="Task", agent=agent)

    crew = Crew(agents=[agent], tasks=[task])

    assert crew.is_complete() is False

    task.mark_completed("Done")

    assert crew.is_complete() is True


def test_crew_has_failed_tasks():
    """Test failed task detection."""
    agent = Agent(role="Worker", goal="Work", backstory="Worker")
    task = Task(description="Task", agent=agent)

    crew = Crew(agents=[agent], tasks=[task])

    assert crew.has_failed_tasks() is False

    task.mark_failed("Error")

    assert crew.has_failed_tasks() is True


def test_crew_to_dict():
    """Test crew serialization."""
    agent = Agent(role="Worker", goal="Work", backstory="Worker")
    task = Task(description="Task", agent=agent)

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=ProcessType.SEQUENTIAL,
        memory=True,
    )

    data = crew.to_dict()

    assert data["process"] == "sequential"
    assert data["memory"] is True
    assert len(data["agents"]) == 1
    assert len(data["tasks"]) == 1

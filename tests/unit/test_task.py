"""Unit tests for Task class."""

import pytest
from llamacrew import Agent, Task, TaskStatus


def test_task_creation():
    """Test basic task creation."""
    agent = Agent(role="Worker", goal="Work", backstory="Worker")

    task = Task(
        description="Complete this task",
        agent=agent,
        expected_output="Task result",
    )

    assert task.description == "Complete this task"
    assert task.agent == agent
    assert task.expected_output == "Task result"
    assert task.status == TaskStatus.PENDING
    assert task.task_id is not None


def test_task_with_dependencies():
    """Test task with dependencies."""
    agent = Agent(role="Worker", goal="Work", backstory="Worker")

    task1 = Task(description="Task 1", agent=agent)
    task2 = Task(description="Task 2", agent=agent, dependencies=[task1])

    assert len(task2.dependencies) == 1
    assert task2.dependencies[0] == task1


def test_task_status_transitions():
    """Test task status transitions."""
    agent = Agent(role="Worker", goal="Work", backstory="Worker")
    task = Task(description="Test task", agent=agent)

    # Initial status
    assert task.status == TaskStatus.PENDING

    # Mark in progress
    task.mark_in_progress()
    assert task.status == TaskStatus.IN_PROGRESS

    # Mark completed
    task.mark_completed("Task output")
    assert task.status == TaskStatus.COMPLETED
    # Result is the output string, not an object
    assert task.result == "Task output"


def test_task_failure():
    """Test task failure."""
    agent = Agent(role="Worker", goal="Work", backstory="Worker")
    task = Task(description="Test task", agent=agent)

    task.mark_in_progress()
    task.mark_failed("Error occurred")

    assert task.status == TaskStatus.FAILED
    assert task.error == "Error occurred"


def test_task_is_ready():
    """Test task readiness based on dependencies."""
    agent = Agent(role="Worker", goal="Work", backstory="Worker")

    task1 = Task(description="Task 1", agent=agent)
    task2 = Task(description="Task 2", agent=agent, dependencies=[task1])

    # Task 2 not ready because Task 1 is pending
    assert task2.is_ready() is False

    # Complete Task 1
    task1.mark_completed("Done")

    # Now Task 2 is ready
    assert task2.is_ready() is True


def test_task_prompt_generation():
    """Test task prompt generation."""
    agent = Agent(role="Writer", goal="Write content", backstory="Professional writer")
    task = Task(
        description="Write an article about AI",
        expected_output="A comprehensive article",
        agent=agent,
    )

    prompt = task.get_prompt()

    assert "Write an article about AI" in prompt
    assert "A comprehensive article" in prompt


def test_task_to_dict():
    """Test task serialization."""
    agent = Agent(role="Worker", goal="Work", backstory="Worker")
    task = Task(description="Test task", agent=agent, expected_output="Output")

    data = task.to_dict()

    assert data["description"] == "Test task"
    assert data["expected_output"] == "Output"
    assert data["status"] == "pending"
    assert "task_id" in data


def test_task_unique_ids():
    """Test that each task gets a unique ID."""
    agent = Agent(role="Worker", goal="Work", backstory="Worker")

    task1 = Task(description="Task 1", agent=agent)
    task2 = Task(description="Task 2", agent=agent)

    assert task1.task_id != task2.task_id

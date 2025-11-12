"""Integration tests for Llama Stack connection."""

import pytest
from llama_stack_client import LlamaStackClient

from llamacrew import Agent, Crew, ProcessType, Task
from llamacrew.llama_integration.client_wrapper import LlamaStackAdapter


# Configuration for running Llama Stack server
LLAMA_STACK_URL = "http://localhost:8321"
MODEL_NAME = "ollama/llama3.2:3b"  # Use smaller model for faster tests


@pytest.fixture
def llama_client():
    """Create Llama Stack client."""
    return LlamaStackClient(base_url=LLAMA_STACK_URL)


@pytest.fixture
def llama_adapter():
    """Create Llama Stack adapter."""
    return LlamaStackAdapter(base_url=LLAMA_STACK_URL)


def test_llama_stack_connection(llama_client):
    """Test basic connection to Llama Stack server."""
    # List models to verify connection
    models = llama_client.models.list()

    assert models is not None
    assert len(models) > 0

    # Check if our test model is available
    model_ids = [m.id for m in models]
    print(f"Available models: {model_ids}")

    # At least one model should be available
    assert len(model_ids) > 0


def test_adapter_get_models(llama_adapter):
    """Test adapter can get available models."""
    models = llama_adapter.get_available_models()

    assert models is not None
    assert len(models) > 0
    print(f"Available models via adapter: {models}")


@pytest.mark.integration
def test_simple_agent_execution(llama_adapter):
    """Test executing a simple agent task."""
    # Create a simple agent
    agent = Agent(
        role="Assistant",
        goal="Answer questions briefly",
        backstory="You are a helpful assistant.",
        llm_config={"model": MODEL_NAME},
        verbose=True,
    )

    # Execute a simple turn
    try:
        response = llama_adapter.execute_turn(
            agent=agent,
            prompt="What is 2+2? Answer in one word.",
        )

        assert response is not None
        assert len(response) > 0
        print(f"Agent response: {response}")

    except Exception as e:
        pytest.fail(f"Agent execution failed: {e}")


@pytest.mark.integration
def test_simple_crew_workflow():
    """Test a simple crew workflow end-to-end."""
    # Create agent
    agent = Agent(
        role="Writer",
        goal="Write brief responses",
        backstory="You are a concise writer.",
        llm_config={"model": MODEL_NAME},
        verbose=True,
    )

    # Create task
    task = Task(
        description="Write a one-sentence summary of what AI is.",
        agent=agent,
        expected_output="A single sentence",
    )

    # Create crew
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=ProcessType.SEQUENTIAL,
        verbose=True,
    )

    # Execute
    try:
        result = crew.kickoff()

        assert result is not None
        assert result.success is True
        assert result.final_output is not None
        assert len(result.tasks_output) == 1

        print(f"\nCrew execution result:")
        print(f"Success: {result.success}")
        print(f"Final output: {result.final_output}")

    except Exception as e:
        pytest.fail(f"Crew execution failed: {e}")


@pytest.mark.integration
def test_two_agent_workflow():
    """Test two-agent workflow with dependencies."""
    # Create agents
    planner = Agent(
        role="Planner",
        goal="Create brief plans",
        backstory="You are a strategic planner.",
        llm_config={"model": MODEL_NAME},
        verbose=True,
    )

    executor = Agent(
        role="Executor",
        goal="Execute plans",
        backstory="You follow plans.",
        llm_config={"model": MODEL_NAME},
        verbose=True,
    )

    # Create tasks
    planning_task = Task(
        description="Create a 3-step plan for making tea.",
        agent=planner,
        expected_output="A numbered list of 3 steps",
    )

    execution_task = Task(
        description="List the first step from the plan.",
        agent=executor,
        expected_output="The first step",
        dependencies=[planning_task],
    )

    # Create crew
    crew = Crew(
        agents=[planner, executor],
        tasks=[planning_task, execution_task],
        process=ProcessType.SEQUENTIAL,
        memory=True,
        verbose=True,
    )

    # Execute
    try:
        result = crew.kickoff()

        assert result is not None
        assert result.success is True
        assert len(result.tasks_output) == 2

        print(f"\nTwo-agent workflow result:")
        print(f"Success: {result.success}")
        print(f"Tasks completed: {result.metadata.get('completed_tasks')}")
        print(f"Final output: {result.final_output}")

    except Exception as e:
        pytest.fail(f"Two-agent workflow failed: {e}")


if __name__ == "__main__":
    """Run integration tests manually."""
    print("=" * 60)
    print("Running Llama Stack Integration Tests")
    print("=" * 60)
    print(f"\nServer: {LLAMA_STACK_URL}")
    print(f"Model: {MODEL_NAME}")
    print()

    # Test 1: Connection
    print("\n" + "=" * 60)
    print("Test 1: Llama Stack Connection")
    print("=" * 60)
    client = LlamaStackClient(base_url=LLAMA_STACK_URL)
    test_llama_stack_connection(client)
    print("✅ Connection test passed")

    # Test 2: Adapter
    print("\n" + "=" * 60)
    print("Test 2: Adapter Models")
    print("=" * 60)
    adapter = LlamaStackAdapter(base_url=LLAMA_STACK_URL)
    test_adapter_get_models(adapter)
    print("✅ Adapter test passed")

    # Test 3: Simple agent
    print("\n" + "=" * 60)
    print("Test 3: Simple Agent Execution")
    print("=" * 60)
    test_simple_agent_execution(adapter)
    print("✅ Agent execution test passed")

    # Test 4: Simple crew
    print("\n" + "=" * 60)
    print("Test 4: Simple Crew Workflow")
    print("=" * 60)
    test_simple_crew_workflow()
    print("✅ Crew workflow test passed")

    # Test 5: Two-agent workflow
    print("\n" + "=" * 60)
    print("Test 5: Two-Agent Workflow")
    print("=" * 60)
    test_two_agent_workflow()
    print("✅ Two-agent workflow test passed")

    print("\n" + "=" * 60)
    print("🎉 All Integration Tests Passed!")
    print("=" * 60)

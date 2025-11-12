"""
Simple two-agent workflow example.

This example demonstrates a basic planner-executor workflow where:
1. A planner agent creates a plan
2. An executor agent follows the plan
"""

from llamacrew import Agent, Crew, Task, ProcessType


def main():
    """Run a simple two-agent workflow."""

    # Create agents
    planner = Agent(
        role="Strategic Planner",
        goal="Create comprehensive and actionable plans",
        backstory=(
            "You are an expert strategic planner with years of experience "
            "breaking down complex goals into manageable steps."
        ),
        verbose=True,
    )

    executor = Agent(
        role="Task Executor",
        goal="Execute plans efficiently and report results",
        backstory=(
            "You are a meticulous executor who follows plans precisely "
            "and provides detailed reports on outcomes."
        ),
        verbose=True,
    )

    # Create tasks
    planning_task = Task(
        description="Create a detailed plan for organizing a team hackathon",
        agent=planner,
        expected_output="A step-by-step plan with timeline and resources needed",
    )

    execution_task = Task(
        description="Review the plan and identify the first three action items to execute",
        agent=executor,
        expected_output="List of three prioritized action items with details",
        dependencies=[planning_task],  # Must wait for planning to complete
    )

    # Create crew
    crew = Crew(
        agents=[planner, executor],
        tasks=[planning_task, execution_task],
        process=ProcessType.SEQUENTIAL,
        memory=True,
        verbose=True,
    )

    # Execute the workflow
    print("=" * 60)
    print("Starting Two-Agent Workflow")
    print("=" * 60)

    result = crew.kickoff()

    # Print results
    print("\n" + "=" * 60)
    print("Workflow Complete!")
    print("=" * 60)
    print(f"\nSuccess: {result.success}")
    print(f"\nFinal Output:\n{result.final_output}")


if __name__ == "__main__":
    main()

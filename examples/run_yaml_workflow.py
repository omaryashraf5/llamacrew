"""
Example: Running a workflow from YAML file.

This shows how to load and execute a workflow defined in YAML.
"""

from llamacrew.parser.yaml_parser import load_workflow


def main():
    """Load and run YAML workflow."""

    print("=" * 60)
    print("Running Workflow from YAML")
    print("=" * 60)

    # Load workflow from YAML file
    crew = load_workflow("blog_workflow.yaml")

    # Run the workflow
    result = crew.kickoff(inputs={"topic": "AI trends in 2024", "word_count": 800})

    # Print results
    print("\n" + "=" * 60)
    print("Workflow Complete!")
    print("=" * 60)
    print(f"\nSuccess: {result.success}")
    print(f"\nFinal Output:\n{result.final_output}")


if __name__ == "__main__":
    main()

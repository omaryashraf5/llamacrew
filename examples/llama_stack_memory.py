"""
Example: Using Llama Stack Native Memory Backend

This example demonstrates how to use the Llama Stack native memory backend
instead of Redis for persistent crew memory.

The Llama Stack backend uses:
- /v1/conversations for key-value storage and conversational memory
- No external dependencies (no Redis required)
- Simpler deployment and better integration
"""

from llama_stack_client import LlamaStackClient

from llamacrew import Agent, Crew, LlamaStackMemoryBackend, Task


def main():
    """Run example with Llama Stack memory backend."""

    # Create Llama Stack client
    client = LlamaStackClient(base_url="http://localhost:5000")

    # Create memory backend using Llama Stack conversations
    memory_backend = LlamaStackMemoryBackend(
        client=client,
        crew_id="research_crew",
    )

    # Define agents
    researcher = Agent(
        role="Senior Researcher",
        goal="Research and analyze topics thoroughly",
        backstory="You are an experienced researcher with expertise in various domains.",
        memory_enabled=True,  # Agent has its own conversation memory
        verbose=True,
    )

    writer = Agent(
        role="Content Writer",
        goal="Create engaging content based on research",
        backstory="You are a skilled writer who can turn research into compelling narratives.",
        memory_enabled=True,
        verbose=True,
    )

    # Define tasks
    research_task = Task(
        description="Research the topic: 'Benefits of AI agents'. Focus on practical applications.",
        expected_output="A comprehensive research summary with key findings",
        agent=researcher,
    )

    write_task = Task(
        description=(
            "Based on the research, write a short article about the benefits of AI agents. "
            "Use the research findings from the previous task."
        ),
        expected_output="A well-written article about AI agent benefits",
        agent=writer,
        dependencies=[research_task],
    )

    # Create crew with memory enabled
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        memory=True,  # Enable shared crew memory
        verbose=True,
    )

    # Execute crew with Llama Stack memory backend
    print("=" * 60)
    print("🦙 Using Llama Stack Native Memory Backend")
    print("=" * 60)
    print("\nNo Redis required! Memory is stored in Llama Stack conversations.\n")

    result = crew.kickoff(
        inputs={
            "topic": "Benefits of AI agents",
            "audience": "technical professionals",
        },
        memory_backend=memory_backend,
    )

    # Display results
    print("\n" + "=" * 60)
    print("📊 Execution Results")
    print("=" * 60)
    print(f"\nSuccess: {result.success}")
    print(f"Total Tasks: {result.metadata.get('total_tasks', 0)}")
    print(f"Completed: {result.metadata.get('completed_tasks', 0)}")

    print("\n" + "=" * 60)
    print("📝 Final Output")
    print("=" * 60)
    print(result.final_output)

    # Check shared memory
    print("\n" + "=" * 60)
    print("🧠 Shared Memory Contents")
    print("=" * 60)
    memory_contents = memory_backend.get_all()
    for key, value in memory_contents.items():
        print(f"\n{key}:")
        if isinstance(value, str) and len(value) > 200:
            print(f"  {value[:200]}...")
        else:
            print(f"  {value}")

    print("\n" + "=" * 60)
    print("✅ Example completed successfully!")
    print("=" * 60)


def example_with_vector_store():
    """Example using vector store backend for semantic memory."""
    from llamacrew import VectorStoreBackend

    client = LlamaStackClient(base_url="http://localhost:5000")

    # Create vector store for semantic search
    vector_store = VectorStoreBackend(
        client=client,
        embedding_model="sentence-transformers",
    )

    # Add knowledge to vector store
    vector_store.add_text(
        "AI agents can automate repetitive tasks and increase productivity.",
        metadata={"topic": "productivity"},
    )

    vector_store.add_text(
        "Multi-agent systems enable collaboration between specialized agents.",
        metadata={"topic": "collaboration"},
    )

    # Search for relevant information
    results = vector_store.search("How do agents work together?", top_k=2)

    print("\n🔍 Vector Store Search Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.get('score', 0):.3f}")
        print(f"   Content: {result.get('content', '')}")
        print(f"   Metadata: {result.get('metadata', {})}")


def example_with_file_storage():
    """Example using file storage backend."""
    from llamacrew import FileStorageBackend

    client = LlamaStackClient(base_url="http://localhost:5000")

    # Create file storage backend
    file_storage = FileStorageBackend(client=client)

    # Upload content
    content = b"This is a sample document about AI agents."
    file_id = file_storage.upload_content(
        content=content,
        filename="ai_agents_doc.txt",
        metadata={"type": "research", "topic": "ai_agents"},
    )

    print(f"\n📄 Uploaded file with ID: {file_id}")

    # List files
    files = file_storage.list_files()
    print(f"\n📚 Total files: {len(files)}")

    # Download content
    retrieved_content = file_storage.get_file_content(file_id)
    print(f"\n✅ Retrieved content: {retrieved_content.decode()}")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║   LlamaCrew - Llama Stack Native Memory Backend Example   ║
╚════════════════════════════════════════════════════════════╝

This example shows how to use Llama Stack's native endpoints
for memory instead of Redis:

✅ No external dependencies (no Redis required)
✅ Better integration with Llama Stack
✅ Simpler deployment (only llama-stack-server needed)
✅ Uses /v1/conversations for persistent memory

Prerequisites:
1. Llama Stack server running on http://localhost:5000
2. Run: llama-stack-server --port 5000

""")

    try:
        main()

        # Optionally run other examples
        print("\n\n" + "=" * 60)
        print("📦 Additional Backend Examples")
        print("=" * 60)

        # Uncomment to try vector store
        # example_with_vector_store()

        # Uncomment to try file storage
        # example_with_file_storage()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Llama Stack server is running:")
        print("  llama-stack-server --port 5000")
        import sys

        sys.exit(1)

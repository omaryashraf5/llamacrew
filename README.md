# LlamaCrew

A lightweight agent-team runtime built on **Llama Stack** for orchestrating multi-agent workflows.

## Features

- **Simple Agent Definition**: Use `@agent` decorator or direct instantiation
- **Multi-Agent Workflows**: Coordinate multiple agents with task dependencies
- **Shared Memory**: Agents share context via scratchpad memory
- **Checkpoint/Resume**: Save and restore workflow state
- **Native Llama Stack Integration**: Built specifically for Meta's Llama Stack
- **Process Strategies**: Sequential, parallel, and hierarchical execution

## Quick Start

### Installation

```bash
# Install Llama Stack server first
pip install llama-stack
llama-stack-server --port 5000

# Install LlamaCrew
cd llamacrew
pip install -e .
```

**Complete Guide:** See the [User Guide](docs/USER_GUIDE.md) for detailed installation, usage patterns, and examples.

### 30-Second Example

```python
from llamacrew import Agent, Crew, Task, ProcessType

# Define agents
planner = Agent(
    role="Strategic Planner",
    goal="Create comprehensive and actionable plans",
    backstory="You are an expert strategic planner.",
)

executor = Agent(
    role="Task Executor",
    goal="Execute plans efficiently",
    backstory="You follow plans precisely.",
)

# Define tasks
planning_task = Task(
    description="Create a plan for organizing a team hackathon",
    agent=planner,
    expected_output="A step-by-step plan with timeline",
)

execution_task = Task(
    description="Identify the first three action items from the plan",
    agent=executor,
    dependencies=[planning_task],
)

# Create and run crew
crew = Crew(
    agents=[planner, executor],
    tasks=[planning_task, execution_task],
    process=ProcessType.SEQUENTIAL,
    memory=True,
)

result = crew.kickoff()
print(result.final_output)
```

## Core Concepts

### Agents

Agents are autonomous entities with specific roles and goals:

```python
from llamacrew import Agent

researcher = Agent(
    role="Researcher",
    goal="Find and synthesize information",
    backstory="Expert researcher with attention to detail",
    tools=["search", "scrape"],  # Tools the agent can use
    llm_config={"model": "llama3-70b", "temperature": 0.7},
)
```

### Tasks

Tasks represent work to be done by agents:

```python
from llamacrew import Task

research_task = Task(
    description="Research the latest trends in AI",
    agent=researcher,
    expected_output="A comprehensive report on AI trends",
    dependencies=[],  # Optional: tasks that must complete first
)
```

### Crews

Crews orchestrate agents and tasks:

```python
from llamacrew import Crew, ProcessType

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=ProcessType.SEQUENTIAL,  # or PARALLEL, HIERARCHICAL
    memory=True,  # Enable shared memory
    checkpoint_enabled=True,  # Enable checkpoint/resume
)

# Run the crew
result = crew.kickoff(inputs={"topic": "AI trends"})
```

## Memory System

LlamaCrew supports multiple memory backends:

### In-Memory (Default)

```python
# Memory is automatically enabled when crew.memory=True
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    memory=True,  # Agents share context (in-memory)
)
```

### Llama Stack Backend (Recommended)

Use Llama Stack's native endpoints for persistent memory - no Redis required!

```python
from llama_stack_client import LlamaStackClient
from llamacrew import Crew, LlamaStackMemoryBackend

# Create Llama Stack client
client = LlamaStackClient(base_url="http://localhost:5000")

# Create memory backend using /v1/conversations
memory_backend = LlamaStackMemoryBackend(
    client=client,
    crew_id="my_crew",
)

# Use with crew
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    memory=True,
)

result = crew.kickoff(
    inputs={"topic": "AI"},
    memory_backend=memory_backend,  # Persistent memory!
)
```

**Benefits of Llama Stack Backend:**
- ✅ No external dependencies (no Redis required)
- ✅ Better integration with Llama Stack ecosystem
- ✅ Simpler deployment (only `llama-stack-server` needed)
- ✅ Uses `/v1/conversations` for persistence

### Advanced: Vector Store & File Storage

```python
from llamacrew import VectorStoreBackend, FileStorageBackend

# Vector store for semantic memory/RAG
vector_store = VectorStoreBackend(client=client)
vector_store.add_text("AI agents are powerful...")
results = vector_store.search("What are agents?")

# File storage for documents
file_storage = FileStorageBackend(client=client)
file_id = file_storage.upload_file("document.pdf")
```

The scratchpad automatically stores:
- Task results
- Input variables
- Custom data set by agents

## 💾 Checkpoint & Resume

Save and resume workflow execution:

```python
# Enable checkpointing
crew = Crew(
    agents=[...],
    tasks=[...],
    checkpoint_enabled=True,
)

# Save checkpoint
crew.save("./checkpoint.json")

# Resume later
crew = Crew.resume("./checkpoint.json")
result = crew.kickoff()
```

## Process Types

### Sequential
Tasks execute one after another based on dependencies:

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    process=ProcessType.SEQUENTIAL,
)
```

### Parallel (Coming Soon)
Independent tasks execute concurrently:

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    process=ProcessType.PARALLEL,
)
```

### Hierarchical (Coming Soon)
Manager agent delegates to worker agents:

```python
crew = Crew(
    agents=[manager, worker1, worker2],
    tasks=[...],
    process=ProcessType.HIERARCHICAL,
)
```

## Project Structure

```
llamacrew/
├── llamacrew/
│   ├── core/              # Core abstractions
│   │   ├── agent.py       # Agent class & decorator
│   │   ├── task.py        # Task definition
│   │   ├── crew.py        # Crew orchestration
│   │   └── message.py     # Message protocol
│   ├── orchestration/     # Execution engine
│   ├── memory/            # Memory & checkpoints
│   ├── llama_integration/ # Llama Stack adapter
│   └── templates/         # Pre-built templates
├── examples/              # Example workflows
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Configuration

Configure Llama Stack connection:

```python
from llamacrew.llama_integration import LlamaStackAdapter

adapter = LlamaStackAdapter(
    base_url="http://localhost:5000",
    api_key="your-api-key",  # Optional
)

# Use custom adapter
from llamacrew.orchestration import WorkflowEngine

engine = WorkflowEngine(crew=crew, llama_adapter=adapter)
```

## Examples

See the `examples/` directory for more:

- `simple_two_agent.py` - Basic planner-executor workflow
- `planning_research_write.py` - Content creation pipeline (coming soon)
- `customer_support_team.py` - Customer support crew (coming soon)

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llamacrew

# Run specific test
pytest tests/unit/test_agent.py
```

## Roadmap

- [x] Core abstractions (Agent, Task, Crew)
- [x] Sequential execution
- [x] Shared memory scratchpad
- [x] Checkpoint/resume
- [ ] Parallel execution
- [ ] Hierarchical process
- [ ] YAML/JSON workflow parser
- [ ] Redis memory backend
- [ ] Pre-built templates
- [ ] CLI tool
- [ ] Advanced tool integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

Built on top of [Llama Stack](https://github.com/llamastack/llama-stack) by Meta.
Inspired by CrewAI and AutoGen.

## Support

- GitHub Issues: [Report bugs or request features]
- Documentation: [Link to docs]

---

**LlamaCrew** - Multi-agent orchestration, powered by Llama Stack

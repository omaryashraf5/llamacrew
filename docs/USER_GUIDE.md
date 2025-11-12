# 🦙 LlamaCrew User Guide

**A complete guide to building multi-agent workflows with LlamaCrew**

---

## 📦 Installation

### Prerequisites

1. **Python 3.10+**
2. **Llama Stack Server** running

### Install Llama Stack Server

```bash
pip install llama-stack
llama-stack-server --port 5000
```

### Install LlamaCrew

```bash
# Clone or navigate to the repo
cd llamacrew

# Install in development mode
pip install -e .

# Or with all extras (Redis support, dev tools)
pip install -e ".[all]"
```

---

## 🚀 Quick Start (30 seconds)

### Your First Agent

```python
from llamacrew import Agent

# Create an agent
assistant = Agent(
    role="Personal Assistant",
    goal="Help users with their tasks efficiently",
    backstory="You are a helpful AI assistant with years of experience"
)
```

### Your First Task

```python
from llamacrew import Task

# Create a task for the agent
task = Task(
    description="Write a short poem about artificial intelligence",
    agent=assistant,
    expected_output="A 4-line poem"
)
```

### Your First Crew

```python
from llamacrew import Crew

# Create and run a crew
crew = Crew(
    agents=[assistant],
    tasks=[task]
)

result = crew.kickoff()
print(result.final_output)
```

**That's it!** You just ran your first multi-agent workflow! 🎉

---

## 💡 Core Concepts

### 1. Agents

Agents are autonomous entities with specific roles and goals.

```python
from llamacrew import Agent

researcher = Agent(
    role="Research Analyst",
    goal="Find accurate and relevant information",
    backstory="Expert researcher with a PhD in Computer Science",
    tools=["search", "calculator"],  # Optional tools
    llm_config={
        "model": "llama3-70b",
        "temperature": 0.7
    },
    verbose=True,  # Print activity
    memory_enabled=True,  # Remember conversations
)
```

**Key Parameters:**
- `role` (required): What the agent does
- `goal` (required): What the agent aims to achieve
- `backstory` (optional): Context about the agent
- `tools` (optional): List of tool names
- `llm_config` (optional): Model configuration
- `memory_enabled` (optional): Enable conversation memory

### 2. Tasks

Tasks define work to be done by agents.

```python
from llamacrew import Task

research_task = Task(
    description="Research the latest trends in AI for 2024",
    agent=researcher,
    expected_output="A detailed report with key findings",
    context={"domain": "AI", "year": 2024},  # Additional context
)
```

**With Dependencies:**

```python
analysis_task = Task(
    description="Analyze the research findings",
    agent=analyst,
    dependencies=[research_task],  # Must complete after research
)
```

### 3. Crews

Crews orchestrate multiple agents and tasks.

```python
from llamacrew import Crew, ProcessType

crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=ProcessType.SEQUENTIAL,  # Execute in order
    memory=True,  # Shared memory across agents
    verbose=True,  # Print execution details
)

# Run the crew
result = crew.kickoff(inputs={"topic": "AI trends"})

# Access results
print(f"Success: {result.success}")
print(f"Output: {result.final_output}")
```

---

## 📚 Usage Patterns

### Pattern 1: Simple Sequential Workflow

**Use Case:** Research → Write → Review

```python
from llamacrew import Agent, Task, Crew, ProcessType

# Define agents
researcher = Agent(
    role="Researcher",
    goal="Gather information"
)

writer = Agent(
    role="Writer",
    goal="Create compelling content"
)

editor = Agent(
    role="Editor",
    goal="Refine and polish content"
)

# Define tasks
task1 = Task(description="Research topic X", agent=researcher)
task2 = Task(description="Write article", agent=writer, dependencies=[task1])
task3 = Task(description="Edit article", agent=editor, dependencies=[task2])

# Create and run crew
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[task1, task2, task3],
    process=ProcessType.SEQUENTIAL
)

result = crew.kickoff()
```

### Pattern 2: Parallel Tasks

**Use Case:** Multiple independent research streams

```python
# Tasks with no dependencies can run in parallel
task1 = Task(description="Research tech trends", agent=researcher1)
task2 = Task(description="Research market data", agent=researcher2)
task3 = Task(description="Research competitors", agent=researcher3)

# Synthesis task depends on all three
synthesis_task = Task(
    description="Synthesize all research",
    agent=analyst,
    dependencies=[task1, task2, task3]
)

crew = Crew(
    agents=[researcher1, researcher2, researcher3, analyst],
    tasks=[task1, task2, task3, synthesis_task],
    process=ProcessType.PARALLEL  # Run independent tasks concurrently
)
```

### Pattern 3: Using Shared Memory

**Use Case:** Agents need to share context

```python
# Enable memory in crew
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    memory=True  # Agents can access shared scratchpad
)

result = crew.kickoff(inputs={
    "company_name": "Acme Corp",
    "target_audience": "developers"
})

# Agents automatically see:
# - Input variables (company_name, target_audience)
# - Results from previous tasks
# - Shared memory scratchpad
```

### Pattern 4: Checkpoint and Resume

**Use Case:** Long-running workflows

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    checkpoint_enabled=True
)

# Run and save progress
result = crew.kickoff()
crew.save("./my_workflow.checkpoint")

# Later, resume from checkpoint
crew = Crew.resume("./my_workflow.checkpoint")
result = crew.kickoff()  # Continues where it left off
```

### Pattern 5: Custom LLM Configuration

**Use Case:** Different models for different agents

```python
# Fast model for simple tasks
simple_agent = Agent(
    role="Summarizer",
    goal="Create quick summaries",
    llm_config={
        "model": "llama3-8b",
        "temperature": 0.3
    }
)

# Powerful model for complex tasks
complex_agent = Agent(
    role="Strategic Analyst",
    goal="Deep strategic analysis",
    llm_config={
        "model": "llama3-70b",
        "temperature": 0.7,
    }
)
```

---

## 🛠️ Advanced Features

### Using Tools

```python
# Agent with tools
data_agent = Agent(
    role="Data Analyst",
    goal="Analyze data and perform calculations",
    tools=["calculator", "search"],  # Built-in tools
)

# Tools are automatically available to the agent
task = Task(
    description="Calculate ROI for the last quarter and search for industry benchmarks",
    agent=data_agent
)
```

### Custom Llama Stack Connection

```python
from llamacrew.llama_integration import LlamaStackAdapter

# Custom adapter
adapter = LlamaStackAdapter(
    base_url="http://your-server:5000",
    api_key="your-api-key"
)

# Use with workflow engine
from llamacrew.orchestration import WorkflowEngine

engine = WorkflowEngine(
    crew=crew,
    llama_adapter=adapter
)
```

### Agent Memory Control

```python
# Agent with memory (remembers conversation)
stateful_agent = Agent(
    role="Customer Support",
    goal="Help customers",
    memory_enabled=True  # Uses Llama Stack conversations
)

# Agent without memory (stateless)
stateless_agent = Agent(
    role="One-off Task Handler",
    goal="Complete tasks independently",
    memory_enabled=False  # Each task is independent
)
```

---

## 📖 Complete Example: Content Creation Pipeline

```python
"""
Complete example: Blog post creation workflow
Planner → Researcher → Writer → Editor
"""

from llamacrew import Agent, Task, Crew, ProcessType


def create_blog_post(topic: str) -> str:
    """Create a blog post using a 4-agent crew."""

    # 1. Define agents
    planner = Agent(
        role="Content Strategist",
        goal="Create content outlines and strategies",
        backstory="Expert strategist with SEO knowledge",
        memory_enabled=True
    )

    researcher = Agent(
        role="Research Specialist",
        goal="Gather accurate information and statistics",
        backstory="Thorough researcher with fact-checking expertise",
        tools=["search"],
        memory_enabled=True
    )

    writer = Agent(
        role="Content Writer",
        goal="Write engaging, well-structured content",
        backstory="Professional writer with 10 years experience",
        llm_config={"model": "llama3-70b", "temperature": 0.8},
        memory_enabled=True
    )

    editor = Agent(
        role="Editor",
        goal="Polish and refine content for publication",
        backstory="Meticulous editor with attention to detail",
        llm_config={"temperature": 0.3},  # More deterministic
        memory_enabled=True
    )

    # 2. Define tasks
    planning_task = Task(
        description=f"Create a detailed outline for a blog post about: {topic}",
        agent=planner,
        expected_output="Detailed outline with sections and key points"
    )

    research_task = Task(
        description=f"Research key facts and statistics about: {topic}",
        agent=researcher,
        expected_output="List of facts with sources",
        dependencies=[planning_task]
    )

    writing_task = Task(
        description=f"Write a comprehensive blog post about: {topic}",
        agent=writer,
        expected_output="Complete blog post (800-1000 words)",
        dependencies=[planning_task, research_task]
    )

    editing_task = Task(
        description="Edit and polish the blog post for publication",
        agent=editor,
        expected_output="Polished, publication-ready blog post",
        dependencies=[writing_task]
    )

    # 3. Create crew
    crew = Crew(
        agents=[planner, researcher, writer, editor],
        tasks=[planning_task, research_task, writing_task, editing_task],
        process=ProcessType.SEQUENTIAL,
        memory=True,
        verbose=True
    )

    # 4. Execute
    print(f"Creating blog post about: {topic}")
    result = crew.kickoff(inputs={"topic": topic})

    # 5. Return final content
    return result.final_output


# Usage
if __name__ == "__main__":
    blog_post = create_blog_post("The Future of Renewable Energy")
    print("\n" + "="*60)
    print("FINAL BLOG POST:")
    print("="*60)
    print(blog_post)
```

---

## 🎨 Best Practices

### 1. **Clear Role Definitions**

✅ **Good:**
```python
Agent(
    role="Technical Writer",
    goal="Create clear, accurate technical documentation",
    backstory="Software engineer turned technical writer with 5 years experience"
)
```

❌ **Bad:**
```python
Agent(
    role="Writer",
    goal="Write things",
    backstory="Writes stuff"
)
```

### 2. **Specific Task Descriptions**

✅ **Good:**
```python
Task(
    description="Analyze Q4 2023 sales data and identify top 3 trends",
    expected_output="Report with data visualization and trend analysis"
)
```

❌ **Bad:**
```python
Task(description="Analyze data")
```

### 3. **Use Dependencies Wisely**

```python
# Sequential dependencies
task2 = Task(..., dependencies=[task1])
task3 = Task(..., dependencies=[task2])

# Parallel with convergence
task4 = Task(..., dependencies=[task2, task3])
```

### 4. **Memory Management**

```python
# Enable memory for conversational agents
customer_support = Agent(..., memory_enabled=True)

# Disable for independent tasks
batch_processor = Agent(..., memory_enabled=False)
```

### 5. **Error Handling**

```python
try:
    result = crew.kickoff()
    if result.success:
        print("Success!")
    else:
        print("Some tasks failed")
        for task_output in result.tasks_output:
            if not task_output["success"]:
                print(f"Failed: {task_output['error']}")
except Exception as e:
    print(f"Workflow error: {e}")
```

---

## 🐛 Troubleshooting

### Issue: "Connection refused"

**Solution:** Make sure Llama Stack server is running
```bash
llama-stack-server --port 5000
```

### Issue: "Model not found"

**Solution:** Check available models and update config
```python
from llama_stack_client import LlamaStackClient
client = LlamaStackClient(base_url="http://localhost:5000")
models = client.models.list()
print([m.identifier for m in models])

# Use an available model
agent = Agent(..., llm_config={"model": "available-model-name"})
```

### Issue: Tasks not completing

**Solution:** Check task dependencies for cycles
```python
# This creates a cycle (bad!)
task1 = Task(..., dependencies=[task2])
task2 = Task(..., dependencies=[task1])  # Circular!

# LlamaCrew will detect this and raise an error
```

---

## 📚 API Reference

### Agent Class

```python
Agent(
    role: str,                      # Required: Agent's role
    goal: str,                      # Required: Agent's objective
    backstory: str = "",            # Optional: Agent's background
    tools: List[str] = [],          # Optional: Tool names
    llm_config: Dict = {},          # Optional: LLM configuration
    verbose: bool = True,           # Optional: Print activity
    max_iterations: int = 15,       # Optional: Max reasoning steps
    allow_delegation: bool = False, # Optional: Can delegate tasks
    memory_enabled: bool = True     # Optional: Enable memory
)
```

### Task Class

```python
Task(
    description: str,               # Required: What to do
    agent: Agent,                   # Required: Who does it
    expected_output: str = "",      # Optional: Expected result
    dependencies: List[Task] = [],  # Optional: Task dependencies
    context: Dict = {},             # Optional: Additional context
    async_execution: bool = False   # Optional: Async execution
)
```

### Crew Class

```python
Crew(
    agents: List[Agent],            # Required: Crew agents
    tasks: List[Task],              # Required: Tasks to execute
    process: ProcessType = SEQUENTIAL,  # Optional: Execution strategy
    memory: bool = True,            # Optional: Shared memory
    verbose: bool = True,           # Optional: Print details
    checkpoint_enabled: bool = False # Optional: Enable checkpoints
)
```

---

## 🎓 Learning Resources

- **Examples:** Check `/home/omara/Desktop/llamacrew/examples/`
- **Documentation:** See `/home/omara/Desktop/llamacrew/docs/`
- **Source Code:** Browse `/home/omara/Desktop/llamacrew/llamacrew/`

---

## 🤝 Getting Help

1. Check this guide
2. Review example workflows in `examples/`
3. Read the troubleshooting section
4. Check Llama Stack documentation

---

**Happy building with LlamaCrew!** 🦙✨

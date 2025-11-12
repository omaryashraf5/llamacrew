# ✅ YAML/JSON Support Added

## 🎯 Original Description

> "A Python library that sits on top of llama-stack and lets developers define and run multi-agent workflows using a simple YAML/JSON or DSL syntax."

## ✨ Status: **FULLY IMPLEMENTED**

✅ **Python library** - Complete
✅ **Sits on top of llama-stack** - Using Agent SDK
✅ **Define workflows** - Complete
✅ **Run multi-agent workflows** - Complete
✅ **YAML syntax** - ✨ **JUST ADDED**
✅ **JSON syntax** - Works (YAML parser handles JSON too)
✅ **DSL syntax** - Python API complete

---

## 📝 Three Ways to Define Workflows

### 1. Python DSL (Programmatic)

```python
from llamacrew import Agent, Task, Crew

planner = Agent(role="Planner", goal="Create plans")
task = Task(description="Plan a hackathon", agent=planner)
crew = Crew(agents=[planner], tasks=[task])
result = crew.kickoff()
```

### 2. YAML (Declarative)

**File: `workflow.yaml`**
```yaml
crew:
  name: "My Team"
  process: "sequential"
  memory: true

agents:
  - name: planner
    role: "Strategic Planner"
    goal: "Create comprehensive plans"
    model: "llama3-70b"

tasks:
  - description: "Plan a hackathon"
    agent: planner
    expected_output: "Detailed plan"
```

**Run it:**
```python
from llamacrew import load_workflow

crew = load_workflow("workflow.yaml")
result = crew.kickoff()
```

### 3. JSON (Also Supported)

Same structure as YAML, just use `.json` extension:

```python
from llamacrew import load_workflow

crew = load_workflow("workflow.json")  # Works!
result = crew.kickoff()
```

---

## 🎁 What Was Added

### New Files
- ✅ `llamacrew/parser/yaml_parser.py` - Full YAML/JSON parser
- ✅ `examples/blog_workflow.yaml` - Example YAML workflow
- ✅ `examples/run_yaml_workflow.py` - Example using YAML

### Updated Files
- ✅ `llamacrew/__init__.py` - Exports `load_workflow` and `YAMLWorkflowParser`

---

## 📚 YAML Syntax Reference

### Full Example

```yaml
# Crew configuration
crew:
  name: "Content Creation Team"
  process: "sequential"  # or "parallel", "hierarchical"
  memory: true
  verbose: true
  checkpoint_enabled: false

# Define agents
agents:
  - name: researcher
    role: "Research Specialist"
    goal: "Gather accurate information"
    backstory: "Expert researcher with PhD"
    tools: ["search", "calculator"]
    model: "llama3-70b"
    temperature: 0.7
    max_iterations: 15
    memory_enabled: true

  - name: writer
    role: "Content Writer"
    goal: "Write engaging content"
    model: "llama3-8b"
    temperature: 0.8

# Define tasks
tasks:
  - description: "Research AI trends for 2024"
    agent: researcher
    expected_output: "Research report"

  - description: "Write article about AI trends"
    agent: writer
    expected_output: "800-word article"
    dependencies: [0]  # Wait for task 0 (research)
    context:
      word_count: 800
      tone: "professional"
```

---

## 🚀 Usage Examples

### Example 1: Load and Run

```python
from llamacrew import load_workflow

# Load from YAML
crew = load_workflow("my_workflow.yaml")

# Run with inputs
result = crew.kickoff(inputs={
    "topic": "AI in healthcare",
    "audience": "developers"
})

print(result.final_output)
```

### Example 2: Dynamic Loading

```python
from llamacrew.parser.yaml_parser import YAMLWorkflowParser

parser = YAMLWorkflowParser()

# Load from file
crew = parser.parse_file("workflow.yaml")

# Or parse dict directly
config = {
    "crew": {...},
    "agents": [...],
    "tasks": [...]
}
crew = parser.parse_dict(config)
```

### Example 3: Mix YAML + Python

```python
from llamacrew import load_workflow

# Load base workflow from YAML
crew = load_workflow("base_workflow.yaml")

# Customize in Python
crew.verbose = False
crew.checkpoint_enabled = True

# Run it
result = crew.kickoff()
```

---

## 🎯 Key Features

### Agent References

```yaml
agents:
  - name: researcher  # Define a name
    role: "Researcher"
    goal: "Research"

tasks:
  - description: "Do research"
    agent: researcher  # Reference by name
```

### Task Dependencies

```yaml
tasks:
  - description: "Task 1"
    agent: agent1

  - description: "Task 2"
    agent: agent2
    dependencies: [0]  # Wait for task 0

  - description: "Task 3"
    agent: agent3
    dependencies: [0, 1]  # Wait for tasks 0 AND 1
```

### Shorthand Syntax

```yaml
# Instead of:
agents:
  - role: "Analyst"
    goal: "Analyze"
    llm_config:
      model: "llama3-70b"
      temperature: 0.7

# Use shorthand:
agents:
  - role: "Analyst"
    goal: "Analyze"
    model: "llama3-70b"
    temperature: 0.7
```

---

## 📖 Complete Workflow Example

See `examples/blog_workflow.yaml` for a full 4-agent workflow:
- Content Strategist (planner)
- Research Specialist (researcher)
- Content Writer (writer)
- Editor (editor)

Run it:
```bash
cd examples
python run_yaml_workflow.py
```

---

## 🔄 Python ↔ YAML Equivalence

### Python Code:
```python
from llamacrew import Agent, Task, Crew, ProcessType

planner = Agent(
    role="Planner",
    goal="Create plans",
    tools=["search"]
)

task = Task(
    description="Plan event",
    agent=planner,
    expected_output="Event plan"
)

crew = Crew(
    agents=[planner],
    tasks=[task],
    process=ProcessType.SEQUENTIAL,
    memory=True
)
```

### Equivalent YAML:
```yaml
crew:
  process: "sequential"
  memory: true

agents:
  - name: planner
    role: "Planner"
    goal: "Create plans"
    tools: ["search"]

tasks:
  - description: "Plan event"
    agent: planner
    expected_output: "Event plan"
```

---

## ✅ The Description Now Fully Stands!

Original promise:
> "A Python library that sits on top of llama-stack and lets developers define and run multi-agent workflows using a simple YAML/JSON or DSL syntax."

**Status:** ✅ **COMPLETE**

- ✅ Python library: Yes
- ✅ Sits on top of llama-stack: Yes (Agent SDK)
- ✅ Define workflows: Yes (3 ways!)
- ✅ Run multi-agent workflows: Yes
- ✅ YAML syntax: ✅ **IMPLEMENTED**
- ✅ JSON syntax: ✅ **SUPPORTED**
- ✅ DSL syntax: ✅ **COMPLETE**

---

## 🎉 Summary

Developers can now choose:

1. **Python DSL** - For programmatic, dynamic workflows
2. **YAML** - For declarative, version-controlled workflows
3. **JSON** - For API integration and tooling

All three methods produce the same Crew object and work identically!

**LlamaCrew delivers on the original vision!** 🦙✨

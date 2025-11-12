"""
LlamaCrew: A lightweight agent-team runtime built on Llama Stack.

This package provides tools for building and running multi-agent workflows
using Meta's Llama Stack infrastructure, with support for both Python DSL
and YAML/JSON configuration.
"""

from .core.agent import Agent, agent
from .core.crew import Crew, CrewOutput, ProcessType
from .core.message import Message, MessageType
from .core.task import Task, TaskResult, TaskStatus
from .memory.backends.llama_stack_backend import (
    FileStorageBackend,
    LlamaStackMemoryBackend,
    VectorStoreBackend,
)
from .parser.yaml_parser import YAMLWorkflowParser, load_workflow

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Agent",
    "agent",
    "Task",
    "TaskStatus",
    "TaskResult",
    "Crew",
    "CrewOutput",
    "ProcessType",
    "Message",
    "MessageType",
    # Parsers
    "YAMLWorkflowParser",
    "load_workflow",
    # Memory backends
    "LlamaStackMemoryBackend",
    "VectorStoreBackend",
    "FileStorageBackend",
]

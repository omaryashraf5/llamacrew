"""Workflow execution engine for LlamaCrew."""

import time
from typing import Any, Dict, List, Optional

from ..core.crew import Crew, CrewOutput, ProcessType
from ..core.task import Task, TaskResult, TaskStatus
from ..llama_integration.client_wrapper import LlamaStackAdapter
from ..memory.scratchpad import Scratchpad, MemoryBackend


class WorkflowEngine:
    """
    Engine for executing crew workflows.

    Handles task orchestration, agent coordination, and execution strategies.
    """

    def __init__(
        self,
        crew: Crew,
        llama_adapter: Optional[LlamaStackAdapter] = None,
        verbose: bool = True,
        checkpoint_enabled: bool = False,
        memory_backend: Optional[MemoryBackend] = None,
    ):
        """
        Initialize the workflow engine.

        Args:
            crew: The crew to execute
            llama_adapter: Llama Stack adapter (creates default if not provided)
            verbose: Whether to print execution details
            checkpoint_enabled: Whether to enable checkpointing
            memory_backend: Optional memory backend for persistent storage
        """
        self.crew = crew
        self.llama_adapter = llama_adapter or LlamaStackAdapter()
        self.verbose = verbose
        self.checkpoint_enabled = checkpoint_enabled
        self.scratchpad: Optional[Scratchpad] = None

        if self.crew.memory:
            # Create scratchpad with backend if provided
            self.scratchpad = Scratchpad(backend=memory_backend)

    def execute(self, inputs: Dict[str, Any]) -> CrewOutput:
        """
        Execute the crew workflow.

        Args:
            inputs: Input data for the workflow

        Returns:
            CrewOutput with results
        """
        if self.verbose:
            print(f"\n🚀 Starting Crew: {self.crew.crew_id}")
            print(f"📋 Process: {self.crew.process.value}")
            print(f"👥 Agents: {len(self.crew.agents)}")
            print(f"📝 Tasks: {len(self.crew.tasks)}\n")

        # Store inputs in scratchpad
        if self.scratchpad:
            for key, value in inputs.items():
                self.scratchpad.set(key, value)

        # Execute based on process type
        if self.crew.process == ProcessType.SEQUENTIAL:
            return self._execute_sequential()
        elif self.crew.process == ProcessType.PARALLEL:
            return self._execute_parallel()
        elif self.crew.process == ProcessType.HIERARCHICAL:
            return self._execute_hierarchical()
        else:
            raise ValueError(f"Unknown process type: {self.crew.process}")

    def _execute_sequential(self) -> CrewOutput:
        """Execute tasks sequentially based on dependencies."""
        tasks_output = []
        completed_tasks = 0
        total_tasks = len(self.crew.tasks)

        while not self.crew.is_complete():
            # Get tasks ready to execute
            ready_tasks = self.crew.get_ready_tasks()

            if not ready_tasks:
                if self.crew.has_failed_tasks():
                    break
                # No ready tasks but not complete - circular dependency
                raise RuntimeError("No ready tasks but workflow not complete")

            # Execute one task at a time (sequential)
            for task in ready_tasks:
                result = self._execute_task(task)
                tasks_output.append(result.to_dict())
                completed_tasks += 1

                if self.verbose:
                    print(f"✅ Task {completed_tasks}/{total_tasks} completed")

                # Checkpoint after each task if enabled
                if self.checkpoint_enabled:
                    self._checkpoint()

                # Only execute one task at a time in sequential mode
                break

        # Check for failures
        success = not self.crew.has_failed_tasks()

        # Generate final output
        final_output = self._generate_final_output(tasks_output)

        return CrewOutput(
            tasks_output=tasks_output,
            final_output=final_output,
            success=success,
            metadata={
                "process": self.crew.process.value,
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
            },
        )

    def _execute_parallel(self) -> CrewOutput:
        """Execute independent tasks in parallel."""
        # For MVP, implement as sequential
        # TODO: Add actual parallel execution with threading/asyncio
        return self._execute_sequential()

    def _execute_hierarchical(self) -> CrewOutput:
        """Execute with a manager agent delegating to workers."""
        # For MVP, implement as sequential
        # TODO: Add manager delegation logic
        return self._execute_sequential()

    def _execute_task(self, task: Task) -> TaskResult:
        """
        Execute a single task.

        Args:
            task: Task to execute

        Returns:
            TaskResult
        """
        if self.verbose:
            print(f"\n🔧 Executing Task: {task.task_id[:8]}...")
            print(f"   Agent: {task.agent.role}")
            print(f"   Description: {task.description[:100]}...")

        task.mark_in_progress()

        try:
            # Build prompt for the task
            prompt = task.get_prompt()

            # Add scratchpad context if enabled
            if self.scratchpad and task.agent.memory_enabled:
                scratchpad_data = self.scratchpad.get_all()
                if scratchpad_data:
                    prompt += "\n\n# Shared Memory\n"
                    for key, value in scratchpad_data.items():
                        prompt += f"- {key}: {value}\n"

            # Execute via Llama Stack
            start_time = time.time()
            output = self.llama_adapter.execute_turn(
                agent=task.agent,
                prompt=prompt,
            )
            execution_time = time.time() - start_time

            # Mark task as completed
            task.mark_completed(output)

            # Store result in scratchpad
            if self.scratchpad:
                self.scratchpad.set(f"task_{task.task_id}_result", output)

            if self.verbose:
                print(f"   ✓ Completed in {execution_time:.2f}s")
                if len(output) > 200:
                    print(f"   Output: {output[:200]}...")
                else:
                    print(f"   Output: {output}")

            return TaskResult(
                task_id=task.task_id,
                success=True,
                output=output,
                metadata={
                    "agent_id": task.agent.agent_id,
                    "agent_role": task.agent.role,
                    "execution_time": execution_time,
                },
            )

        except Exception as e:
            error_msg = str(e)
            task.mark_failed(error_msg)

            if self.verbose:
                print(f"   ✗ Failed: {error_msg}")

            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=error_msg,
                metadata={
                    "agent_id": task.agent.agent_id,
                    "agent_role": task.agent.role,
                },
            )

    def _generate_final_output(self, tasks_output: List[Dict[str, Any]]) -> str:
        """
        Generate final combined output from all tasks.

        Args:
            tasks_output: List of task results

        Returns:
            Final output string
        """
        output_parts = ["# Workflow Results\n"]

        for i, task_output in enumerate(tasks_output, 1):
            if task_output["success"]:
                output_parts.append(f"\n## Task {i}")
                output_parts.append(f"Output: {task_output['output']}")
            else:
                output_parts.append(f"\n## Task {i} (FAILED)")
                output_parts.append(f"Error: {task_output['error']}")

        return "\n".join(output_parts)

    def _checkpoint(self) -> None:
        """Save checkpoint of current execution state."""
        if self.checkpoint_enabled:
            # TODO: Implement checkpointing
            pass

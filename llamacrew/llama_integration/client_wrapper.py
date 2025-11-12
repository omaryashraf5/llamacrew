"""Wrapper for Llama Stack Agent SDK."""

from typing import Any, Dict, List, Optional

from llama_stack_client import Agent as LlamaAgent, LlamaStackClient

from ..core.agent import Agent


class LlamaStackAdapter:
    """
    Adapter between LlamaCrew and Llama Stack Agent SDK.

    Uses the high-level Agent SDK for better session management
    and tool execution.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5000",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the adapter.

        Args:
            base_url: Base URL of Llama Stack server
            api_key: Optional API key for authentication
        """
        self.client = LlamaStackClient(
            base_url=base_url,
            api_key=api_key,
        )
        self.base_url = base_url

        # Track LlamaCrew agent -> Llama SDK agent mapping
        self._agent_instances: Dict[str, LlamaAgent] = {}
        # Track agent -> session mapping
        self._agent_sessions: Dict[str, str] = {}

    def execute_turn(
        self,
        agent: Agent,
        prompt: str,
        stream: bool = False,
    ) -> str:
        """
        Execute an agent turn using Llama Stack Agent SDK.

        Args:
            agent: LlamaCrew agent
            prompt: User prompt/task
            stream: Whether to stream (handled internally)

        Returns:
            Agent response as string
        """
        # Get or create Llama SDK agent
        llama_agent = self._get_or_create_llama_agent(agent)

        # Get or create session
        session_id = self._get_or_create_session(agent, llama_agent)

        try:
            # Create message
            messages = [{"role": "user", "content": prompt}]

            # Execute turn (always streaming internally for better control)
            response_text = None
            for chunk in llama_agent.create_turn(
                messages=messages,
                session_id=session_id,
                stream=True,
            ):
                # Extract final response from the last chunk
                if chunk.response:
                    response_text = self._extract_response_from_chunk(chunk)

            if response_text is None:
                raise RuntimeError("No response received from agent")

            return response_text

        except Exception as e:
            raise RuntimeError(f"Failed to execute agent turn: {str(e)}") from e

    def _get_or_create_llama_agent(self, agent: Agent) -> LlamaAgent:
        """
        Get or create Llama SDK agent for a LlamaCrew agent.

        Args:
            agent: LlamaCrew agent

        Returns:
            Llama SDK Agent instance
        """
        if agent.agent_id in self._agent_instances:
            return self._agent_instances[agent.agent_id]

        # Build system prompt
        instructions = self._build_system_prompt(agent)

        # Convert tools (if any)
        tools = None
        if agent.tools:
            tools = self._convert_tools_to_openai_format(agent.tools)

        # Create Llama SDK agent
        llama_agent = LlamaAgent(
            client=self.client,
            model=agent.llm_config.get("model", "llama3-70b"),
            instructions=instructions,
            tools=tools,
        )

        # Cache it
        self._agent_instances[agent.agent_id] = llama_agent

        return llama_agent

    def _get_or_create_session(self, agent: Agent, llama_agent: LlamaAgent) -> str:
        """
        Get or create session for an agent.

        Args:
            agent: LlamaCrew agent
            llama_agent: Llama SDK agent

        Returns:
            Session ID
        """
        if agent.agent_id in self._agent_sessions:
            return self._agent_sessions[agent.agent_id]

        # Create new session if memory enabled
        if agent.memory_enabled:
            session_name = f"crew_agent_{agent.role}_{agent.agent_id[:8]}"
            session_id = llama_agent.create_session(session_name)
        else:
            # Create anonymous session (won't persist)
            session_id = llama_agent.create_session(f"temp_{agent.agent_id}")

        self._agent_sessions[agent.agent_id] = session_id
        return session_id

    def _build_system_prompt(self, agent: Agent) -> str:
        """
        Build system prompt from agent attributes.

        Args:
            agent: LlamaCrew agent

        Returns:
            System prompt string
        """
        parts = [f"You are a {agent.role}."]

        if agent.goal:
            parts.append(f"Your goal is: {agent.goal}")

        if agent.backstory:
            parts.append(f"Background: {agent.backstory}")

        if agent.allow_delegation:
            parts.append(
                "You can delegate tasks to other agents if needed by clearly stating "
                "which agent should handle the task."
            )

        return "\n\n".join(parts)

    def _convert_tools_to_openai_format(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        """
        Convert tool names to OpenAI function format.

        Args:
            tool_names: List of tool names

        Returns:
            List of tool definitions in OpenAI format
        """
        # Basic tool definitions for MVP
        tool_definitions = {
            "search": {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            "calculator": {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            },
        }

        tools = []
        for tool_name in tool_names:
            if tool_name in tool_definitions:
                tools.append(tool_definitions[tool_name])
            else:
                # Generic tool definition
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": f"Tool: {tool_name}",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    }
                )

        return tools

    def _extract_response_from_chunk(self, chunk: Any) -> str:
        """
        Extract text from agent stream chunk.

        Args:
            chunk: AgentStreamChunk from Llama SDK

        Returns:
            Extracted text content
        """
        try:
            if not chunk.response:
                return ""

            response = chunk.response

            # Response has output list
            if hasattr(response, "output") and len(response.output) > 0:
                output_item = response.output[0]

                # Check for message type
                if hasattr(output_item, "type") and output_item.type == "message":
                    if hasattr(output_item, "content") and len(output_item.content) > 0:
                        content_item = output_item.content[0]
                        if hasattr(content_item, "text"):
                            return content_item.text

            # Fallback
            return str(response)

        except Exception as e:
            raise RuntimeError(f"Failed to extract response: {str(e)}") from e

    def get_available_models(self) -> List[str]:
        """
        Get list of available models from Llama Stack.

        Returns:
            List of model identifiers
        """
        try:
            models = self.client.models.list()
            return [model.identifier for model in models]
        except Exception:
            return ["llama3-70b", "llama3-8b"]

    def clear_session(self, agent: Agent) -> None:
        """
        Clear session for an agent.

        Args:
            agent: LlamaCrew agent
        """
        if agent.agent_id in self._agent_sessions:
            del self._agent_sessions[agent.agent_id]
        if agent.agent_id in self._agent_instances:
            del self._agent_instances[agent.agent_id]

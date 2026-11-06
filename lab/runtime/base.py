"""Abstract runtime interface.

Concrete implementations (e.g. ADK+LiteLLM+Ollama) satisfy this contract.
Swapping the runtime must not require changes to workflow or role logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel
from lab.schemas.poc import StructuredResponse, ToolCallResult

T = TypeVar("T", bound=BaseModel)


class ModelRuntime(ABC):
    """Narrow interface for model execution.

    All methods are async to accommodate runtimes (such as ADK) that require
    an event loop.  Implementations that wrap synchronous clients should use
    asyncio.to_thread or equivalent.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the exact model identifier used by this runtime."""

    @property
    @abstractmethod
    def runtime_label(self) -> str:
        """Human-readable description of the runtime stack."""

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """Return a plain-text completion for *prompt*.

        Raises:
            RuntimeError: if the model or endpoint is unavailable.
        """

    @abstractmethod
    async def complete_structured(self, prompt: str) -> StructuredResponse:
        """Return a validated :class:`StructuredResponse` for *prompt*.

        The implementation must attempt structured output through the primary
        runtime path.  If that path fails, it may raise rather than silently
        fall back, so the probe can record the failure honestly.

        Raises:
            ValueError: if the model response cannot be parsed into the schema.
            RuntimeError: if the model or endpoint is unavailable.
        """

    @abstractmethod
    async def complete_structured_schema(
        self,
        prompt: str,
        schema: type[T],
        instruction: str | None = None,
        session_id: str = "structured",
        max_retries: int = 2,
    ) -> T:
        """Return a validated instance of Pydantic model *schema* for *prompt*.

        Args:
            prompt: The user prompt to supply to the model.
            schema: The target Pydantic model class.
            instruction: Optional system instruction overriding the default.
            session_id: ADK session identifier for isolation.
            max_retries: Maximum number of bounded correction retries on schema validation failure.

        Returns:
            Validated instance of *schema*.
        """

    @abstractmethod
    async def run_agent(
        self,
        instruction: str,
        user_message: str,
        tools: list[Any] | None = None,
        session_id: str = "agent",
    ) -> tuple[str, list[dict]]:
        """Run an agent turn with optional tools and return (final_text, event_summaries).

        Args:
            instruction: Role instructions/system prompt for the agent.
            user_message: User request or task instructions.
            tools: Optional sequence of callable tools exposed to the agent.
            session_id: ADK session identifier for isolation.

        Returns:
            Tuple of (final_response_text, list_of_event_summaries).
        """

    async def generate(
        self,
        instruction: str,
        user_message: str,
        tools: list[Any] | None = None,
        json_mode: bool = False,
        session_id: str = "generate",
    ) -> tuple[str, list[dict]]:
        """Make exactly one model turn and return (text, event_summaries) unvalidated.

        This is the primitive :class:`lab.runtime.router.ModelRouter` builds on.
        It deliberately performs no retries, so the router's bound of two attempts
        per model cannot be multiplied by retries hidden further down.

        Args:
            instruction: Role instructions/system prompt.
            user_message: The task prompt.
            tools: Optional callable tools exposed to the agent.
            json_mode: Ask the model for a JSON object.
            session_id: Session identifier for isolation.
        """
        del json_mode  # Runtimes without a JSON mode rely on the prompt alone.
        return await self.run_agent(
            instruction=instruction,
            user_message=user_message,
            tools=tools,
            session_id=session_id,
        )

    @abstractmethod
    async def call_tool(
        self,
        tool_fn: Any,
        prompt: str,
    ) -> ToolCallResult:
        """Ask the model to call *tool_fn* in response to *prompt*.

        The implementation must verify that the model itself requested the
        tool (``agent_used_tool=True``) rather than calling *tool_fn* directly.

        Args:
            tool_fn: A plain Python callable registered as a tool.
            prompt:  Natural-language instruction that should cause the model
                     to invoke the tool.

        Returns:
            :class:`ToolCallResult` with ``agent_used_tool`` set honestly.
        """

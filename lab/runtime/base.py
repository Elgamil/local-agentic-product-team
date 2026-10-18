"""Abstract runtime interface.

Concrete implementations (e.g. ADK+LiteLLM+Ollama) satisfy this contract.
Swapping the runtime must not require changes to workflow or role logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lab.schemas.poc import StructuredResponse, ToolCallResult


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

"""Runtime configuration.

Values come from environment variables with safe defaults.
No secrets are stored here; this is endpoint and tuning config only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable runtime configuration snapshot."""

    model_id: str = field(
        default_factory=lambda: os.environ.get("LAB_MODEL_ID", "gemma4:latest")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "OLLAMA_API_BASE", "http://localhost:11434"
        )
    )
    timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("LAB_TIMEOUT_SECONDS", "120"))
    )
    # LiteLLM provider prefix (ollama_chat gives OpenAI-compat chat endpoint)
    litellm_provider: str = field(
        default_factory=lambda: os.environ.get("LAB_LITELLM_PROVIDER", "ollama_chat")
    )

    @property
    def litellm_model(self) -> str:
        """Full LiteLLM model string, e.g. 'ollama_chat/gemma4:latest'."""
        return f"{self.litellm_provider}/{self.model_id}"


def default_config() -> RuntimeConfig:
    """Return a RuntimeConfig populated from the current environment."""
    return RuntimeConfig()

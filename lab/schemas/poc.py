"""Pydantic v2 data contracts for the Milestone-1 runtime proof-of-concept.

These schemas are model-agnostic and runtime-agnostic; they express what the
delivery workflow expects, not how any particular model or runtime delivers it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Individual check result
# ---------------------------------------------------------------------------


class CheckResult(BaseModel):
    """Result of one discrete probe check."""

    name: str = Field(description="Short identifier for this check, e.g. 'basic_completion'")
    success: bool
    duration_ms: float = Field(ge=0, description="Wall-clock duration of the check in milliseconds")
    output: str | None = Field(
        default=None,
        description="Truncated model or function output (max 500 chars)",
    )
    error: str | None = Field(
        default=None,
        description="Error message if success is False, else None",
    )

    @field_validator("output", mode="before")
    @classmethod
    def truncate_output(cls, v: Any) -> str | None:
        """Prevent unbounded log entries from model verbosity."""
        if v is None:
            return None
        s = str(v)
        return s[:500] + "…[truncated]" if len(s) > 500 else s


# ---------------------------------------------------------------------------
# Structured response schema (what we ask the model to produce)
# ---------------------------------------------------------------------------


class StructuredResponse(BaseModel):
    """Schema the model must populate when asked for structured output."""

    answer: str = Field(description="The model's direct answer to the question")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model's self-reported confidence in the answer, 0–1",
    )
    reasoning: str = Field(description="Brief explanation of the reasoning")


# ---------------------------------------------------------------------------
# Tool call result
# ---------------------------------------------------------------------------


class ToolCallResult(BaseModel):
    """Validated result of an ADK-mediated tool invocation."""

    tool_name: str = Field(description="Name of the function that was called")
    success: bool
    payload: dict[str, Any] = Field(
        description="The actual return value of the executed tool function"
    )
    agent_used_tool: bool = Field(
        description=(
            "True when the model itself requested the tool call (not simulated). "
            "False when the tool was invoked directly to work around model failure."
        )
    )


# ---------------------------------------------------------------------------
# Overall probe-run report
# ---------------------------------------------------------------------------


class ProbeResult(BaseModel):
    """Complete record of one probe run."""

    run_id: str = Field(description="ISO-8601 timestamp used as a unique run identifier")
    model_id: str = Field(description="Exact model identifier used (e.g. 'gemma4:latest')")
    runtime: str = Field(description="Runtime description (e.g. 'google-adk+litellm+ollama')")
    started_at: datetime
    finished_at: datetime | None = None
    total_duration_ms: float | None = None
    checks: list[CheckResult] = Field(default_factory=list)
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    notes: str | None = Field(
        default=None,
        description="Human-readable summary of limitations or notable observations",
    )

    def add_check(self, result: CheckResult) -> None:
        """Append a check result and update pass/fail tallies."""
        self.checks.append(result)
        if result.success:
            self.passed += 1
        else:
            self.failed += 1

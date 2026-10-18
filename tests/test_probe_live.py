"""Live integration tests — require a running Ollama instance with gemma4:latest.

All tests are marked @pytest.mark.live and use the require_ollama session fixture
to skip automatically if Ollama is not reachable.

Evaluation integrity rules enforced here:
  - Pass/fail is determined by schema validation and observable side-effects only.
  - Model text asserting it did something is NOT used as evidence of success.
  - tool_call: agent_used_tool must be True for the test to pass.
  - structured_output: StructuredResponse.model_validate() must succeed.
"""

from __future__ import annotations

import pytest
import httpx

from lab.runtime.adk_ollama import OllamaAdkRuntime
from lab.schemas.poc import StructuredResponse, ToolCallResult
from lab.tools.workspace_info import get_workspace_info


pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def runtime(runtime_config, require_ollama):
    return OllamaAdkRuntime(runtime_config)


# ---------------------------------------------------------------------------
# Check 1 — endpoint health
# ---------------------------------------------------------------------------


def test_endpoint_health(ollama_base_url, require_ollama):
    """Ollama endpoint responds and gemma4:latest is listed."""
    resp = httpx.get(f"{ollama_base_url}/api/tags", timeout=10)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    model_names = [m["name"] for m in data.get("models", [])]
    assert any("gemma4" in name for name in model_names), (
        f"gemma4:latest not found in Ollama model list: {model_names}"
    )


# ---------------------------------------------------------------------------
# Check 2 — basic completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_completion(runtime):
    """ADK runtime returns a non-empty string for a plain prompt."""
    response = await runtime.complete("What is 2 + 2? Answer in one sentence.")
    assert isinstance(response, str), "Response must be a string"
    assert len(response) > 0, "Response must not be empty"
    # We do NOT assert the text content — just that a response was produced


# ---------------------------------------------------------------------------
# Check 3 — structured output via ADK
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structured_output_validates(runtime):
    """ADK path produces a response that validates as StructuredResponse.

    This test FAILS if the model response cannot be parsed into the schema.
    A native-Ollama fallback would be a separate test, not a substitute here.
    """
    result = await runtime.complete_structured("What is the capital of France?")

    # Ground-truth: schema validation succeeded (Pydantic enforces types and ranges)
    assert isinstance(result, StructuredResponse)
    assert isinstance(result.answer, str) and len(result.answer) > 0
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.reasoning, str) and len(result.reasoning) > 0


# ---------------------------------------------------------------------------
# Check 4 — actual tool call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_call_agent_used_tool(runtime):
    """Model requests the tool; ADK executes it; agent_used_tool is True.

    FAIL criteria:
      - agent_used_tool is False (model answered without calling the tool)
      - payload does not contain expected workspace keys
    """
    result = await runtime.call_tool(
        tool_fn=get_workspace_info,
        prompt=(
            "Please call the get_workspace_info tool and tell me "
            "the Python version and workspace path."
        ),
    )

    assert isinstance(result, ToolCallResult)

    # Ground-truth check: agent must have requested the tool
    assert result.agent_used_tool is True, (
        "FAIL: The model did not request the tool call. "
        f"agent_used_tool=False. Result: {result.model_dump_json()}"
    )

    # Ground-truth check: payload must contain expected keys (real function return)
    assert "python_version" in result.payload, (
        f"Expected 'python_version' in payload. Got: {result.payload}"
    )
    assert "workspace_path" in result.payload


# ---------------------------------------------------------------------------
# Check 5 — get_workspace_info tool safety (called directly for ground truth)
# ---------------------------------------------------------------------------


def test_workspace_info_tool_safety():
    """Verify the tool's output is safe and bounded before exposing it to the model."""
    info = get_workspace_info()

    assert "python_version" in info
    assert "workspace_path" in info
    assert "file_count" in info
    assert "platform" in info
    assert "venv_active" in info

    # Safety: workspace path must not be the home directory
    import os
    home = os.path.expanduser("~")
    assert info["workspace_path"] != home, (
        "Tool must not expose the home directory as workspace"
    )

    # Safety: file count must be a non-negative integer
    assert isinstance(info["file_count"], int)
    assert info["file_count"] >= 0

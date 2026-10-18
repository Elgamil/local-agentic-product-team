"""Unit tests for the runtime interface contract and configuration.

These tests use unittest.mock and do NOT require Ollama or ADK to be installed.
They verify that:
  - OllamaAdkRuntime satisfies the ModelRuntime ABC
  - RuntimeConfig loads values correctly
  - The async interface returns expected types when the underlying ADK call is mocked
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lab.runtime.config import RuntimeConfig, default_config
from lab.schemas.poc import StructuredResponse, ToolCallResult


# ---------------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------------


class TestRuntimeConfig:
    def test_defaults(self):
        cfg = RuntimeConfig()
        assert cfg.model_id == "gemma4:latest"
        assert cfg.ollama_base_url == "http://localhost:11434"
        assert cfg.timeout_seconds == 120.0
        assert cfg.litellm_provider == "ollama_chat"

    def test_litellm_model_property(self):
        cfg = RuntimeConfig(model_id="gemma4:latest", litellm_provider="ollama_chat")
        assert cfg.litellm_model == "ollama_chat/gemma4:latest"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LAB_MODEL_ID", "qwen2.5:7b")
        monkeypatch.setenv("LAB_TIMEOUT_SECONDS", "60")
        cfg = default_config()
        assert cfg.model_id == "qwen2.5:7b"
        assert cfg.timeout_seconds == 60.0

    def test_immutable(self):
        cfg = RuntimeConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.model_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OllamaAdkRuntime — interface contract (mocked)
# ---------------------------------------------------------------------------


class TestOllamaAdkRuntimeContract:
    """Verify the runtime satisfies the ModelRuntime ABC and returns correct types."""

    def _make_runtime(self):
        from lab.runtime.adk_ollama import OllamaAdkRuntime
        cfg = RuntimeConfig(model_id="gemma4:latest")
        return OllamaAdkRuntime(cfg)

    def test_instantiation(self):
        rt = self._make_runtime()
        assert rt is not None

    def test_model_id_property(self):
        rt = self._make_runtime()
        assert rt.model_id == "gemma4:latest"

    def test_runtime_label(self):
        rt = self._make_runtime()
        assert "ollama" in rt.runtime_label.lower()
        assert "adk" in rt.runtime_label.lower()

    def test_implements_model_runtime_abc(self):
        from lab.runtime.base import ModelRuntime
        from lab.runtime.adk_ollama import OllamaAdkRuntime
        assert issubclass(OllamaAdkRuntime, ModelRuntime)

    @pytest.mark.asyncio
    async def test_complete_returns_str(self):
        rt = self._make_runtime()
        fake_text = "The answer is 4."
        with patch.object(rt, "_run_adk_agent", new=AsyncMock(return_value=(fake_text, []))):
            result = await rt.complete("What is 2+2?")
        assert isinstance(result, str)
        assert result == fake_text

    @pytest.mark.asyncio
    async def test_complete_raises_on_empty(self):
        rt = self._make_runtime()
        with patch.object(rt, "_run_adk_agent", new=AsyncMock(return_value=("", []))):
            with pytest.raises(RuntimeError, match="empty response"):
                await rt.complete("prompt")

    @pytest.mark.asyncio
    async def test_complete_structured_returns_schema(self):
        rt = self._make_runtime()
        fake_json = '{"answer": "Paris", "confidence": 0.99, "reasoning": "capital"}'
        with patch.object(rt, "_run_adk_agent", new=AsyncMock(return_value=(fake_json, []))):
            result = await rt.complete_structured("What is the capital of France?")
        assert isinstance(result, StructuredResponse)
        assert result.answer == "Paris"
        assert result.confidence == 0.99

    @pytest.mark.asyncio
    async def test_complete_structured_raises_on_bad_json(self):
        rt = self._make_runtime()
        with patch.object(rt, "_run_adk_agent", new=AsyncMock(return_value=("not json at all", []))):
            with pytest.raises(ValueError):
                await rt.complete_structured("question")

    @pytest.mark.asyncio
    async def test_complete_structured_handles_markdown_fences(self):
        rt = self._make_runtime()
        fenced = '```json\n{"answer": "Rome", "confidence": 0.8, "reasoning": "city"}\n```'
        with patch.object(rt, "_run_adk_agent", new=AsyncMock(return_value=(fenced, []))):
            result = await rt.complete_structured("Capital of Italy?")
        assert result.answer == "Rome"

    @pytest.mark.asyncio
    async def test_call_tool_agent_used_tool_true(self):
        rt = self._make_runtime()
        # Events use the nested "parts" structure that _run_adk_agent produces
        fake_events = [
            {
                "author": "lab_probe_agent",
                "type": "Event",
                "parts": [
                    {"function_call": {"name": "get_workspace_info", "args": {}}},
                ],
            },
            {
                "author": "lab_probe_agent",
                "type": "Event",
                "parts": [
                    {"function_response": {"name": "get_workspace_info"}},
                ],
            },
        ]

        def dummy_tool():
            return {"python_version": "3.11.14", "workspace_path": "/fake/path", "file_count": 5, "platform": "Darwin", "venv_active": True}

        dummy_tool.__name__ = "get_workspace_info"

        with patch.object(rt, "_run_adk_agent", new=AsyncMock(return_value=("result text", fake_events))):
            result = await rt.call_tool(dummy_tool, "Call the workspace tool")

        assert isinstance(result, ToolCallResult)
        assert result.agent_used_tool is True
        assert result.success is True
        assert result.tool_name == "get_workspace_info"

    @pytest.mark.asyncio
    async def test_call_tool_agent_did_not_use_tool(self):
        rt = self._make_runtime()
        # Events contain no function_call parts
        fake_events = [{"type": "Event", "author": "lab_probe_agent", "parts": [{"text_snippet": "I'll answer directly."}]}]

        def dummy_tool():
            return {}

        dummy_tool.__name__ = "get_workspace_info"

        with patch.object(rt, "_run_adk_agent", new=AsyncMock(return_value=("I know the answer", fake_events))):
            result = await rt.call_tool(dummy_tool, "prompt")

        assert result.agent_used_tool is False
        assert result.success is False

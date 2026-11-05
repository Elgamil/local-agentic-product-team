"""Unit tests for structured output extraction, validation, and bounded retry."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from pydantic import BaseModel, Field

from lab.runtime.adk_ollama import OllamaAdkRuntime, _extract_json
from lab.runtime.config import RuntimeConfig
from lab.schemas.artifacts import ArchitectureArtifact, RequirementsArtifact


class SampleSchema(BaseModel):
    name: str = Field(description="Name")
    count: int = Field(description="Count")


class TestExtractJson:
    def test_clean_json(self):
        text = '{"name": "test", "count": 42}'
        result = _extract_json(text)
        assert result == {"name": "test", "count": 42}

    def test_markdown_fence_json(self):
        text = "```json\n{\"name\": \"test\", \"count\": 42}\n```"
        result = _extract_json(text)
        assert result == {"name": "test", "count": 42}

    def test_prose_before_and_after_json(self):
        text = (
            "The user wants me to design an architecture.\n"
            "Plan: 1. Review 2. Design\n\n"
            "```json\n"
            "{\"name\": \"test\", \"count\": 42}\n"
            "```\n\n"
            "Hope this helps!"
        )
        result = _extract_json(text)
        assert result == {"name": "test", "count": 42}

    def test_prose_without_fences(self):
        text = (
            "Here is the result:\n"
            "{\"name\": \"test\", \"count\": 42}\n"
            "Completed successfully."
        )
        result = _extract_json(text)
        assert result == {"name": "test", "count": 42}

    def test_incomplete_truncated_json_raises_value_error(self):
        # Truncated JSON without closing braces should raise ValueError, not fabricate
        text = '{"name": "test", "items": [{"id": 1}, {"id": 2'
        with pytest.raises(ValueError, match="Incomplete or truncated JSON"):
            _extract_json(text)

    def test_no_json_raises_value_error(self):
        text = "This response contains no json at all."
        with pytest.raises(ValueError, match="No JSON object found"):
            _extract_json(text)


class TestStructuredRetry:
    @pytest.fixture
    def runtime(self):
        cfg = RuntimeConfig(model_id="gemma4:latest")
        return OllamaAdkRuntime(cfg)

    @pytest.mark.asyncio
    async def test_first_attempt_success_no_retries(self, runtime):
        valid_json = json.dumps({"name": "MiniShop", "count": 5})
        mock_logger = MagicMock()
        runtime._logger = mock_logger

        with patch.object(runtime, "_run_adk_agent", new=AsyncMock(return_value=(valid_json, []))) as mock_run:
            result = await runtime.complete_structured_schema(
                prompt="test prompt",
                schema=SampleSchema,
                session_id="test_sess",
                max_retries=2,
            )

        assert isinstance(result, SampleSchema)
        assert result.name == "MiniShop"
        assert result.count == 5
        assert mock_run.call_count == 1
        mock_logger.log_validation_failed.assert_not_called()
        mock_logger.log_retry_attempt.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_on_validation_failure_then_success(self, runtime):
        # First response is missing required field 'count'
        bad_json = json.dumps({"name": "MiniShop"})
        good_json = json.dumps({"name": "MiniShop", "count": 10})
        mock_logger = MagicMock()
        runtime._logger = mock_logger

        with patch.object(
            runtime,
            "_run_adk_agent",
            new=AsyncMock(side_effect=[(bad_json, []), (good_json, [])]),
        ) as mock_run:
            result = await runtime.complete_structured_schema(
                prompt="test prompt",
                schema=SampleSchema,
                session_id="test_sess",
                max_retries=2,
            )

        assert isinstance(result, SampleSchema)
        assert result.name == "MiniShop"
        assert result.count == 10
        assert mock_run.call_count == 2
        # Check that validation failure and retry attempt were logged
        assert mock_logger.log_validation_failed.call_count == 1
        assert mock_logger.log_retry_attempt.call_count == 1

        # Check that the second call received the correction prompt
        second_call_prompt = mock_run.call_args_list[1].kwargs.get("user_message", "")
        assert "Validation Error" in second_call_prompt
        assert "Mandatory Schema Fields" in second_call_prompt

    @pytest.mark.asyncio
    async def test_retry_on_truncated_json_then_success(self, runtime):
        # First response is cut off mid-JSON
        truncated = '{"name": "MiniShop", "count": '
        good_json = json.dumps({"name": "MiniShop", "count": 15})
        mock_logger = MagicMock()
        runtime._logger = mock_logger

        with patch.object(
            runtime,
            "_run_adk_agent",
            new=AsyncMock(side_effect=[(truncated, []), (good_json, [])]),
        ) as mock_run:
            result = await runtime.complete_structured_schema(
                prompt="test prompt",
                schema=SampleSchema,
                session_id="test_sess",
                max_retries=2,
            )

        assert result.count == 15
        assert mock_run.call_count == 2

    @pytest.mark.asyncio
    async def test_bounded_retry_exhaustion_raises(self, runtime):
        # All attempts return invalid JSON missing required fields
        invalid_json = json.dumps({"wrong_field": "oops"})
        mock_logger = MagicMock()
        runtime._logger = mock_logger

        with patch.object(
            runtime,
            "_run_adk_agent",
            new=AsyncMock(return_value=(invalid_json, [])),
        ) as mock_run:
            with pytest.raises(ValueError, match="after 3 attempts"):
                await runtime.complete_structured_schema(
                    prompt="test prompt",
                    schema=SampleSchema,
                    session_id="test_sess",
                    max_retries=2,
                )

        # max_retries=2 means exactly 3 attempts (1 initial + 2 retries)
        assert mock_run.call_count == 3
        assert mock_logger.log_validation_failed.call_count == 3
        assert mock_logger.log_retry_attempt.call_count == 2


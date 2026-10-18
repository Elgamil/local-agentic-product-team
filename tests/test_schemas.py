"""Unit tests for Pydantic v2 schemas (no Ollama required)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from lab.schemas.poc import (
    CheckResult,
    ProbeResult,
    StructuredResponse,
    ToolCallResult,
)


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------


class TestCheckResult:
    def test_valid_success(self):
        r = CheckResult(name="test", success=True, duration_ms=42.5, output="ok")
        assert r.success is True
        assert r.error is None

    def test_valid_failure(self):
        r = CheckResult(name="test", success=False, duration_ms=5.0, error="timeout")
        assert r.success is False
        assert r.error == "timeout"

    def test_negative_duration_rejected(self):
        with pytest.raises(ValidationError):
            CheckResult(name="test", success=True, duration_ms=-1.0)

    def test_output_truncated_at_500_chars(self):
        long_output = "x" * 600
        r = CheckResult(name="test", success=True, duration_ms=1.0, output=long_output)
        assert len(r.output) <= 515  # 500 + "…[truncated]"
        assert r.output.endswith("…[truncated]")

    def test_output_not_truncated_when_short(self):
        r = CheckResult(name="test", success=True, duration_ms=1.0, output="short")
        assert r.output == "short"

    def test_none_output_allowed(self):
        r = CheckResult(name="test", success=True, duration_ms=1.0, output=None)
        assert r.output is None


# ---------------------------------------------------------------------------
# StructuredResponse
# ---------------------------------------------------------------------------


class TestStructuredResponse:
    def test_valid(self):
        sr = StructuredResponse(answer="Paris", confidence=0.95, reasoning="Well known fact.")
        assert sr.answer == "Paris"
        assert sr.confidence == 0.95

    def test_confidence_too_high_rejected(self):
        with pytest.raises(ValidationError):
            StructuredResponse(answer="X", confidence=1.1, reasoning="r")

    def test_confidence_negative_rejected(self):
        with pytest.raises(ValidationError):
            StructuredResponse(answer="X", confidence=-0.1, reasoning="r")

    def test_confidence_boundary_values(self):
        StructuredResponse(answer="X", confidence=0.0, reasoning="r")
        StructuredResponse(answer="X", confidence=1.0, reasoning="r")

    def test_json_round_trip(self):
        sr = StructuredResponse(answer="Paris", confidence=0.9, reasoning="Capital city.")
        recovered = StructuredResponse.model_validate_json(sr.model_dump_json())
        assert recovered.answer == sr.answer
        assert recovered.confidence == sr.confidence


# ---------------------------------------------------------------------------
# ToolCallResult
# ---------------------------------------------------------------------------


class TestToolCallResult:
    def test_valid_success(self):
        t = ToolCallResult(
            tool_name="get_workspace_info",
            success=True,
            payload={"python_version": "3.11.14"},
            agent_used_tool=True,
        )
        assert t.agent_used_tool is True

    def test_valid_failure(self):
        t = ToolCallResult(
            tool_name="get_workspace_info",
            success=False,
            payload={},
            agent_used_tool=False,
        )
        assert t.success is False

    def test_empty_payload_allowed(self):
        t = ToolCallResult(
            tool_name="fn",
            success=False,
            payload={},
            agent_used_tool=False,
        )
        assert t.payload == {}


# ---------------------------------------------------------------------------
# ProbeResult
# ---------------------------------------------------------------------------


class TestProbeResult:
    def _make_report(self) -> ProbeResult:
        now = datetime.now(timezone.utc)
        report = ProbeResult(
            run_id="20260101T000000Z",
            model_id="gemma4:latest",
            runtime="google-adk+litellm+ollama",
            started_at=now,
        )
        report.add_check(CheckResult(name="c1", success=True, duration_ms=10.0, output="ok"))
        report.add_check(CheckResult(name="c2", success=False, duration_ms=5.0, error="err"))
        return report

    def test_pass_fail_tally(self):
        report = self._make_report()
        assert report.passed == 1
        assert report.failed == 1

    def test_json_round_trip(self):
        report = self._make_report()
        recovered = ProbeResult.model_validate_json(report.model_dump_json())
        assert recovered.run_id == report.run_id
        assert recovered.model_id == report.model_id
        assert len(recovered.checks) == 2
        assert recovered.passed == 1
        assert recovered.failed == 1

    def test_checks_list_defaults_empty(self):
        now = datetime.now(timezone.utc)
        r = ProbeResult(
            run_id="x",
            model_id="m",
            runtime="r",
            started_at=now,
        )
        assert r.checks == []
        assert r.passed == 0
        assert r.failed == 0

    def test_model_dump_is_valid_json(self):
        report = self._make_report()
        parsed = json.loads(report.model_dump_json())
        assert parsed["model_id"] == "gemma4:latest"
        assert len(parsed["checks"]) == 2

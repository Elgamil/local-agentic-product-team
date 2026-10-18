"""Probe CLI — Milestone 1 runtime proof-of-concept.

Runs all acceptance checks sequentially and writes a JSON run report to
the runs/ directory.  Pass/fail is determined solely by ground-truth
function results and schema validation — never by model-generated claims.

Usage:
    python -m lab.probe [--model MODEL_ID] [--base-url URL]

Environment variables (override defaults):
    LAB_MODEL_ID           Default: gemma4:latest
    OLLAMA_API_BASE        Default: http://localhost:11434
    LAB_TIMEOUT_SECONDS    Default: 120
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from lab.runtime.adk_ollama import OllamaAdkRuntime
from lab.runtime.config import RuntimeConfig
from lab.schemas.poc import CheckResult, ProbeResult
from lab.tools.workspace_info import get_workspace_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("lab.probe")

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


async def check_endpoint_health(base_url: str) -> CheckResult:
    """Check 1: Ollama endpoint responds and gemma4:latest is listed."""
    name = "endpoint_health"
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        gemma_present = any("gemma4" in m for m in models)
        duration_ms = (time.perf_counter() - t0) * 1000
        if gemma_present:
            return CheckResult(
                name=name,
                success=True,
                duration_ms=duration_ms,
                output=f"Available models: {models}",
            )
        else:
            return CheckResult(
                name=name,
                success=False,
                duration_ms=duration_ms,
                error=f"gemma4:latest not found in model list: {models}",
            )
    except Exception as exc:
        duration_ms = (time.perf_counter() - t0) * 1000
        return CheckResult(name=name, success=False, duration_ms=duration_ms, error=str(exc))


async def check_basic_completion(runtime: OllamaAdkRuntime) -> CheckResult:
    """Check 2: Non-empty response through the primary ADK runtime."""
    name = "basic_completion"
    t0 = time.perf_counter()
    try:
        response = await runtime.complete("What is 2 + 2? Answer in one sentence.")
        duration_ms = (time.perf_counter() - t0) * 1000
        if response:
            return CheckResult(name=name, success=True, duration_ms=duration_ms, output=response)
        else:
            return CheckResult(
                name=name,
                success=False,
                duration_ms=duration_ms,
                error="Model returned an empty response.",
            )
    except Exception as exc:
        duration_ms = (time.perf_counter() - t0) * 1000
        return CheckResult(name=name, success=False, duration_ms=duration_ms, error=str(exc))


async def check_structured_output(runtime: OllamaAdkRuntime) -> CheckResult:
    """Check 3: ADK path returns a response validating against StructuredResponse.

    If ADK structured output fails, the failure is recorded as-is.
    No native-Ollama fallback is counted as proof of ADK structured-output support.
    """
    name = "structured_output"
    t0 = time.perf_counter()
    try:
        result = await runtime.complete_structured(
            "What is the capital of France?"
        )
        duration_ms = (time.perf_counter() - t0) * 1000
        return CheckResult(
            name=name,
            success=True,
            duration_ms=duration_ms,
            output=result.model_dump_json(),
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - t0) * 1000
        return CheckResult(name=name, success=False, duration_ms=duration_ms, error=str(exc))


async def check_tool_call(runtime: OllamaAdkRuntime) -> CheckResult:
    """Check 4: Model requests the tool; ADK runner executes it; result validated.

    agent_used_tool=False means the model did NOT call the tool (failure).
    The function is NOT called directly to simulate success.
    """
    name = "tool_call"
    t0 = time.perf_counter()
    try:
        result = await runtime.call_tool(
            tool_fn=get_workspace_info,
            prompt=(
                "Please call the get_workspace_info tool and tell me "
                "the Python version and workspace path."
            ),
        )
        duration_ms = (time.perf_counter() - t0) * 1000

        if not result.agent_used_tool:
            return CheckResult(
                name=name,
                success=False,
                duration_ms=duration_ms,
                error=(
                    "The model did not request the tool. "
                    "agent_used_tool=False — tool was not called by the agent."
                ),
                output=result.model_dump_json(),
            )

        return CheckResult(
            name=name,
            success=True,
            duration_ms=duration_ms,
            output=result.model_dump_json(),
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - t0) * 1000
        return CheckResult(name=name, success=False, duration_ms=duration_ms, error=str(exc))


def check_report_integrity(report: ProbeResult) -> CheckResult:
    """Check 5: ProbeResult serialises to JSON and deserialises correctly."""
    name = "report_integrity"
    t0 = time.perf_counter()
    try:
        json_str = report.model_dump_json()
        recovered = ProbeResult.model_validate_json(json_str)
        assert recovered.model_id == report.model_id
        assert recovered.run_id == report.run_id
        assert len(recovered.checks) == len(report.checks)
        duration_ms = (time.perf_counter() - t0) * 1000
        return CheckResult(
            name=name,
            success=True,
            duration_ms=duration_ms,
            output=f"Serialised {len(json_str)} bytes; {len(report.checks)} checks round-tripped",
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - t0) * 1000
        return CheckResult(name=name, success=False, duration_ms=duration_ms, error=str(exc))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def run_probe(config: RuntimeConfig) -> ProbeResult:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = ProbeResult(
        run_id=run_id,
        model_id=config.model_id,
        runtime="google-adk+litellm+ollama",
        started_at=datetime.now(timezone.utc),
    )

    runtime = OllamaAdkRuntime(config)

    logger.info("=== Probe run %s started ===", run_id)

    # Check 1 — endpoint health (no ADK, direct httpx)
    logger.info("Check 1: endpoint_health")
    report.add_check(await check_endpoint_health(config.ollama_base_url))

    # Check 2 — basic completion
    logger.info("Check 2: basic_completion")
    report.add_check(await check_basic_completion(runtime))

    # Check 3 — structured output
    logger.info("Check 3: structured_output")
    report.add_check(await check_structured_output(runtime))

    # Check 4 — tool call
    logger.info("Check 4: tool_call")
    report.add_check(await check_tool_call(runtime))

    # Check 5 — report integrity (uses current state of report)
    logger.info("Check 5: report_integrity")
    report.add_check(check_report_integrity(report))

    report.finished_at = datetime.now(timezone.utc)
    report.total_duration_ms = (
        report.finished_at - report.started_at
    ).total_seconds() * 1000

    return report


def save_report(report: ProbeResult) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{report.run_id}.json"
    path.write_text(report.model_dump_json(indent=2))
    return path


def print_summary(report: ProbeResult) -> None:
    print("\n" + "=" * 60)
    print(f"  Probe run: {report.run_id}")
    print(f"  Model:     {report.model_id}")
    print(f"  Runtime:   {report.runtime}")
    print(f"  Duration:  {report.total_duration_ms:.0f} ms")
    print("=" * 60)
    for chk in report.checks:
        status = "PASS" if chk.success else "FAIL"
        print(f"  [{status:4s}] {chk.name} ({chk.duration_ms:.0f} ms)")
        if not chk.success and chk.error:
            # Print first 200 chars of error
            print(f"         ↳ {chk.error[:200]}")
    print("=" * 60)
    print(f"  Passed: {report.passed} / {report.passed + report.failed}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab Milestone 1 probe")
    parser.add_argument("--model", default=None, help="Override LAB_MODEL_ID")
    parser.add_argument("--base-url", default=None, help="Override OLLAMA_API_BASE")
    args = parser.parse_args()

    if args.model:
        os.environ["LAB_MODEL_ID"] = args.model
    if args.base_url:
        os.environ["OLLAMA_API_BASE"] = args.base_url

    from lab.runtime.config import default_config

    config = default_config()
    logger.info("Using model: %s  base_url: %s", config.model_id, config.ollama_base_url)

    report = asyncio.run(run_probe(config))
    path = save_report(report)
    print_summary(report)
    logger.info("Run report saved to: %s", path)

    # Exit code = number of failed checks
    return report.failed


if __name__ == "__main__":
    sys.exit(main())

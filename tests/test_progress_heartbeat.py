"""Progress visibility during long operations.

Only the models are fake.  Threads, subprocesses and the terminal code are real, and
the final group runs the real CLI against a slow stub server.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import pty
import re
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import lab.product.progress as progress_module
from lab.config import load_model_policy
from lab.product.console import (
    HEARTBEAT_SECONDS,
    LOG_HEARTBEAT_SECONDS,
    RESET,
    STATUS_TEXT,
    ConsoleTerminal,
)
from lab.product.manager import latest_roadmap
from lab.product.progress import ModelProgress, format_duration, watching_test_process
from lab.product.store import ProductStore
from lab.runtime.heartbeat import THREAD_NAME, Heartbeat
from lab.runtime.router import (
    FilesystemFailure,
    ModelRouter,
    ModelTask,
    TaskExhausted,
)
from lab.schemas.product import ModelAttempt, ProductPhase, StepStatus
from tests.fakes import (
    APPROVE,
    CALC_BUGGY,
    GEMMA,
    QWEN,
    FakeLifecycle,
    Fail,
    Harness,
    ScriptedRuntime,
    ScriptedTeam,
    ScriptedTerminal,
    SimulatedPowerLoss,
    Text,
    to_proposal_gate,
    to_step_start_gate,
)
from tests.test_model_router import VALID, Answer
from tests.test_runtime_http_contract import StubOllama

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAST = 0.02
ESC = "\x1b"
COLOR_ENV = {"TERM": "xterm-256color"}


@pytest.fixture(autouse=True)
def no_heartbeat_left_behind():
    yield
    running = [item for item in threading.enumerate() if item.name == THREAD_NAME]
    assert running == [], "a heartbeat thread outlived its operation"


class FakeTty(io.StringIO):
    def isatty(self) -> bool:
        return True


class Recorder(ScriptedTerminal):
    """Keeps every line in order, as the person would see them."""

    def __init__(self, interval: float = FAST) -> None:
        super().__init__()
        self.heartbeat_seconds = interval
        self.lines: list[str] = []

    def status(self, text: str) -> None:
        self.lines.append(text)

    def warn(self, text: str) -> None:
        self.warnings.append(text)
        self.lines.append(f"WARNING {text}")

    def error(self, text: str) -> None:
        self.errors.append(text)
        self.lines.append(f"ERROR {text}")

    def agent(self, role: str, text: str) -> None:
        self.lines.append(f"AGENT {role}: {text}")

    def progress(self, text: str, context: str = "") -> None:
        self.heartbeats.append(text)
        self.lines.append(text)

    def progress_done(self, text: str, context: str = "") -> None:
        self.finished.append(text)
        self.lines.append(text)

    def has(self, text: str) -> bool:
        return any(text in line for line in self.lines)


class ModelStandIn(ScriptedRuntime):
    def __init__(self, model, outcomes, calls, delay):
        super().__init__(model, ScriptedTeam())
        self.outcomes = outcomes
        self.calls = calls
        self.delay = delay

    async def generate(self, instruction, user_message, tools=None, json_mode=False,
                       session_id="generate"):
        self.calls.append((self._model, user_message))
        await asyncio.sleep(self.delay)
        outcome = self.outcomes.pop(0) if self.outcomes else VALID
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome, []


def build_router(outcomes, *, delay=0.0, policy=None, stand_in=ModelStandIn):
    calls: list[tuple[str, str]] = []
    router = ModelRouter(
        policy or load_model_policy(),
        lambda spec: stand_in(spec.id, outcomes.get(spec.id, []), calls, delay),
        FakeLifecycle(),
    )
    return router, calls


def make_task(display=None, policy=None, **overrides) -> ModelTask:
    limit = (policy or load_model_policy()).timeouts.inference_seconds
    fields = dict(
        task_id="t1",
        role="developer",
        kind="demo",
        chain=[QWEN, GEMMA],
        instruction="be precise",
        build_prompt=lambda model: f"prompt for {model}",
        build_failover_prompt=lambda model, failures: f"handover to {model}",
        schema=Answer,
        observer=ModelProgress(display, "developer", "Repair 1 of 2", 4, limit) if display else None,
    )
    fields.update(overrides)
    return ModelTask(**fields)


def attempt(number: int = 1, model: str = QWEN) -> ModelAttempt:
    return ModelAttempt(
        attempt_id="a", task_id="t", model=model, model_attempt=1, overall_attempt=number
    )


class TestDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0 seconds"),
            (1, "1 second"),
            (10, "10 seconds"),
            (43.4, "43 seconds"),
            (59.6, "1 minute"),
            (60, "1 minute"),
            (61, "1 minute 1 second"),
            (125, "2 minutes 5 seconds"),
            (639, "10 minutes 39 seconds"),
            (900, "15 minutes"),
            (-3, "0 seconds"),
        ],
    )
    def test_format(self, seconds, expected):
        assert format_duration(seconds) == expected


class TestHeartbeatThread:
    def test_it_ticks_while_the_block_runs(self):
        ticks: list[float] = []
        with Heartbeat(ticks.append, 0.03):
            time.sleep(0.2)
        assert len(ticks) >= 3

    def test_elapsed_time_is_measured_from_the_start(self):
        ticks: list[float] = []
        with Heartbeat(ticks.append, 0.05):
            time.sleep(0.28)
        assert ticks == sorted(ticks)
        assert ticks[0] == pytest.approx(0.05, abs=0.04)
        assert ticks[-1] == pytest.approx(0.05 * len(ticks), abs=0.06)

    def test_no_tick_before_the_first_interval(self):
        ticks: list[float] = []
        with Heartbeat(ticks.append, 30):
            time.sleep(0.05)
        assert ticks == []

    def test_it_stops_after_success(self):
        ticks: list[float] = []
        with Heartbeat(ticks.append, FAST):
            time.sleep(0.08)
        count = len(ticks)
        time.sleep(0.1)
        assert len(ticks) == count

    @pytest.mark.parametrize(
        "error", [ValueError("failed"), KeyboardInterrupt(), asyncio.CancelledError()],
        ids=["failure", "ctrl-c", "cancelled"],
    )
    def test_it_stops_when_the_operation_ends_badly(self, error):
        ticks: list[float] = []
        with pytest.raises(type(error)):
            with Heartbeat(ticks.append, FAST):
                time.sleep(0.05)
                raise error
        count = len(ticks)
        time.sleep(0.1)
        assert len(ticks) == count

    def test_the_thread_has_ended_by_the_time_the_block_exits(self):
        beat = Heartbeat(lambda elapsed: None, FAST)
        with beat:
            assert beat._thread.is_alive()
        assert not beat._thread.is_alive()

    def test_the_thread_is_a_daemon_so_it_cannot_keep_the_process_alive(self):
        beat = Heartbeat(lambda elapsed: None, 30)
        with beat:
            assert beat._thread.daemon is True

    def test_a_failing_tick_stays_silent(self, monkeypatch):
        """An exception escaping the thread would print a traceback into the terminal."""
        escaped = []
        monkeypatch.setattr(threading, "excepthook", escaped.append)

        def broken(elapsed):
            raise OSError("terminal went away")

        with Heartbeat(broken, FAST):
            time.sleep(0.06)

        assert escaped == []

    def test_it_keeps_ticking_while_the_event_loop_is_blocked(self):
        """This is why it is a thread: test runs block the loop in process.wait()."""
        ticks: list[float] = []

        async def blocked() -> None:
            with Heartbeat(ticks.append, FAST):
                time.sleep(0.12)

        asyncio.run(blocked())
        assert len(ticks) >= 2


class TestWhatTheWaitLooksLike:
    def _replay(self, monkeypatch, elapsed_values, final):
        """Run the display with exact times, so the wording can be checked precisely."""

        class ReplayedHeartbeat:
            def __init__(self, tick, interval, clock):
                self.tick = tick
                self.elapsed = final

            def __enter__(self):
                for value in elapsed_values:
                    self.tick(value)
                return self

            def __exit__(self, *details):
                return None

        monkeypatch.setattr(progress_module, "Heartbeat", ReplayedHeartbeat)

    def test_the_specified_sequence(self, monkeypatch):
        self._replay(monkeypatch, [10.0, 20.0], 43.0)
        display = Recorder()
        watcher = ModelProgress(display, "developer", "Repair 1 of 2", 4, 900)

        with watcher.waiting(attempt()):
            pass

        assert display.lines == [
            "Developer — Repair 1 of 2",
            "Model: qwen2.5:7b",
            "Waiting for model response... (time limit: 15 minutes)",
            "Still waiting for model response — 10 seconds elapsed.",
            "Still waiting for model response — 20 seconds elapsed.",
            "Model response received — 43 seconds.",
        ]

    def test_elapsed_time_comes_from_the_clock(self):
        now = [100.0]
        display = Recorder(interval=30)
        watcher = ModelProgress(
            display, "developer", "Repair 1 of 2", 4, 900, clock=lambda: now[0]
        )
        with watcher.waiting(attempt()):
            now[0] = 143.0
        assert display.finished == ["Model response received — 43 seconds."]

    def test_a_retry_shows_the_attempt_number(self):
        display = Recorder(interval=30)
        watcher = ModelProgress(display, "developer", "Repair 1 of 2", 4, 900)
        with watcher.waiting(attempt(number=3, model=GEMMA)):
            pass
        assert display.lines[0] == "Developer — Repair 1 of 2 (attempt 3 of 4)"
        assert display.lines[1] == "Model: gemma4:latest"

    def test_nothing_claims_progress_or_percentages(self, monkeypatch):
        self._replay(monkeypatch, [10.0, 20.0, 30.0], 31.0)
        display = Recorder()
        with ModelProgress(display, "developer", "Repair", 4, 900).waiting(attempt()):
            pass
        text = "\n".join(display.lines)
        assert "%" not in text
        assert "almost" not in text.lower()
        assert "generating" not in text.lower()

    def test_a_failed_wait_does_not_claim_a_response(self):
        display = Recorder(interval=30)
        with pytest.raises(ValueError):
            with ModelProgress(display, "developer", "Repair", 4, 900).waiting(attempt()):
                raise ValueError("no")
        assert display.finished == []
        assert not display.has("received")


class TestHeartbeatDuringAModelCall:
    async def test_heartbeats_appear_during_a_slow_call(self):
        display = Recorder()
        router, _ = build_router({QWEN: [VALID]}, delay=0.15)
        await router.execute(make_task(display))
        assert len(display.heartbeats) >= 2
        assert all(line.startswith("Still waiting for model response") for line in display.heartbeats)
        assert display.finished[-1].startswith("Model response received")

    async def test_a_quick_call_has_no_heartbeat(self):
        display = Recorder(interval=30)
        router, _ = build_router({QWEN: [VALID]})
        await router.execute(make_task(display))
        assert display.heartbeats == []
        assert len(display.finished) == 1

    async def test_heartbeat_stops_after_success(self):
        display = Recorder()
        router, _ = build_router({QWEN: [VALID]}, delay=0.08)
        await router.execute(make_task(display))
        count = len(display.heartbeats)
        await asyncio.sleep(0.1)
        assert len(display.heartbeats) == count

    async def test_heartbeat_stops_after_failure(self):
        display = Recorder()
        router, _ = build_router({QWEN: [FilesystemFailure("disk")]}, delay=0.08)
        with pytest.raises(TaskExhausted):
            await router.execute(make_task(display))
        count = len(display.heartbeats)
        await asyncio.sleep(0.1)
        assert len(display.heartbeats) == count

    async def test_heartbeat_stops_on_cancellation(self):
        display = Recorder()
        router, _ = build_router({QWEN: [VALID]}, delay=30)
        task = asyncio.create_task(router.execute(make_task(display)))
        await asyncio.sleep(0.08)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        count = len(display.heartbeats)
        await asyncio.sleep(0.1)
        assert len(display.heartbeats) == count
        assert any("Interrupted while waiting for the model" in line for line in display.warnings)

    async def test_heartbeat_stops_when_the_power_goes(self):
        display = Recorder()
        router, _ = build_router({QWEN: [SimulatedPowerLoss()]}, delay=0.05)
        with pytest.raises(SimulatedPowerLoss):
            await router.execute(make_task(display))
        assert display.warnings

    async def test_each_attempt_gets_its_own_heartbeat(self):
        display = Recorder()
        router, _ = build_router({QWEN: ["bad", VALID]}, delay=0.06)
        await router.execute(make_task(display))
        assert display.lines.count("Model: qwen2.5:7b") == 2
        assert sum("Model response received" in text for text in display.finished) == 2

    async def test_slow_parsing_gets_its_own_message(self):
        def slow_check(value):
            time.sleep(0.1)

        display = Recorder()
        router, _ = build_router({QWEN: [VALID]})
        await router.execute(make_task(display, validator=slow_check))
        assert display.has("Still processing the response")

    async def test_quick_parsing_stays_quiet(self):
        display = Recorder(interval=30)
        router, _ = build_router({QWEN: [VALID]})
        await router.execute(make_task(display))
        assert not display.has("processing")

    async def test_heartbeats_continue_if_the_model_call_blocks_the_loop(self):
        class Blocking(ModelStandIn):
            async def generate(self, *args, **kwargs):
                time.sleep(0.12)
                return VALID, []

        display = Recorder()
        router, _ = build_router({}, stand_in=Blocking)
        await router.execute(make_task(display))
        assert len(display.heartbeats) >= 2


class TestDistinctMessages:
    async def test_a_timeout_is_reported_as_a_timeout(self):
        policy = load_model_policy()
        policy.timeouts.inference_seconds = 0.03
        display = Recorder()
        router, _ = build_router({QWEN: [VALID]}, delay=5, policy=policy)
        with pytest.raises(TaskExhausted):
            await router.execute(make_task(display, policy=policy))
        assert any("Timed out: no model response within" in line for line in display.warnings)
        assert display.errors == [], "a timeout is not reported as a failed answer"

    async def test_a_timeout_names_the_configured_limit(self):
        display = Recorder()
        watcher = ModelProgress(display, "developer", "Repair 1 of 2", 4, 900)
        timed_out = attempt(number=2)
        timed_out.failure_class = "inference_timeout"
        watcher.failed(timed_out)
        assert "configured limit of 15 minutes" in display.warnings[0]
        assert "Repair 1 of 2" in display.warnings[0]
        assert "attempt 2 of 4" in display.warnings[0]
        assert QWEN in display.warnings[0]

    async def test_an_actual_failure_is_reported_as_a_failure(self):
        display = Recorder()
        router, _ = build_router({QWEN: ["not json", VALID]})
        await router.execute(make_task(display))
        assert len(display.errors) == 1
        assert "Failed after" in display.errors[0]
        assert "incomplete or badly formed" in display.errors[0]
        assert "attempt 1 of 4" in display.errors[0]

    async def test_a_retry_says_it_is_a_retry(self, tmp_path):
        harness = Harness(tmp_path)
        harness.team.script("architect", Text("bad"))
        display = Recorder(interval=30)
        workflow = harness.workflow(harness.manager.create(), reporter=display)
        await to_proposal_gate(workflow)
        assert any("Trying once more" in line for line in display.warnings)
        assert display.has("Software Architect — Technical design, Step 1 (attempt 2 of 4)")

    async def test_a_switch_says_it_is_a_switch(self, tmp_path):
        harness = Harness(tmp_path)
        harness.team.script("architect", Text("bad"), Text("bad"))
        display = Recorder(interval=30)
        workflow = harness.workflow(harness.manager.create(), reporter=display)
        await to_proposal_gate(workflow)
        assert any("switching to the backup model" in line for line in display.warnings)
        assert display.has("(attempt 3 of 4)")
        assert display.has("Model: gemma4:latest")

    def test_the_messages_really_are_different(self):
        display = Recorder(interval=30)
        watcher = ModelProgress(display, "developer", "Repair", 4, 900)
        with watcher.waiting(attempt()):
            pass
        with watching_test_process(display, "Test Engineer — Acceptance tests", 120):
            pass
        assert display.has("Waiting for model response")
        assert display.has("Running test process")


class TestNoConcurrentInference:
    async def test_heartbeats_add_no_model_requests(self):
        display = Recorder()
        router, calls = build_router({QWEN: [VALID]}, delay=0.15)
        await router.execute(make_task(display))
        assert len(display.heartbeats) >= 2
        assert len(calls) == 1

    async def test_one_inference_at_a_time_is_still_enforced(self):
        team = ScriptedTeam(delay=0.03)
        lifecycle = FakeLifecycle()
        team.lifecycle = lifecycle
        display = Recorder()
        router = ModelRouter(load_model_policy(), team.runtime_factory, lifecycle)

        def make(number: int) -> ModelTask:
            return make_task(
                display,
                task_id=f"step-01-architect-v{number}",
                schema=None,
                chain=[QWEN, GEMMA] if number % 2 else [GEMMA, QWEN],
                postcondition=lambda text, events: None,
            )

        await asyncio.gather(*(router.execute(make(n)) for n in range(1, 7)))

        assert team.max_active == 1
        assert team.max_resident == 1

    async def test_only_one_heartbeat_runs_at_a_time(self):
        peak = []

        class Counting(Recorder):
            def progress(self, text, context=""):
                peak.append(
                    sum(1 for item in threading.enumerate() if item.name == THREAD_NAME)
                )

        display = Counting()
        router, _ = build_router({QWEN: [VALID]}, delay=0.05)

        def make(number: int) -> ModelTask:
            return make_task(display, task_id=f"t{number}")

        await asyncio.gather(*(router.execute(make(n)) for n in range(4)))
        assert peak and max(peak) == 1


SCENARIOS = {
    "success": ({QWEN: [VALID]}, None),
    "retry": ({QWEN: ["bad", VALID]}, None),
    "failover": ({QWEN: ["bad", "bad"], GEMMA: [VALID]}, None),
    "second-fallback-attempt": ({QWEN: ["bad", "bad"], GEMMA: ["bad", VALID]}, None),
    "exhausted": ({QWEN: ["bad", "bad"], GEMMA: ["bad", "bad"]}, TaskExhausted),
    "connection": ({QWEN: [ConnectionError("x"), ConnectionError("x")], GEMMA: [VALID]}, None),
    "deterministic": ({QWEN: [FilesystemFailure("disk")]}, TaskExhausted),
}


class TestBehaviourIsUnchanged:
    """Everything the router decides must match, with and without a display attached."""

    async def _run(self, outcomes, expected, display):
        copied = {model: list(items) for model, items in outcomes.items()}
        router, calls = build_router(copied)
        task = make_task(display)
        value = None
        if expected:
            with pytest.raises(expected):
                await router.execute(task)
        else:
            value = (await router.execute(task)).value
        attempts = [
            (item.model, item.status, item.failure_class, item.overall_attempt, item.switched_from)
            for item in router.attempt_log.load("t1")
        ]
        return value, calls, attempts

    @pytest.mark.parametrize("name", SCENARIOS)
    async def test_same_attempts_prompts_and_result(self, name):
        outcomes, expected = SCENARIOS[name]
        watched = await self._run(outcomes, expected, Recorder())
        unwatched = await self._run(outcomes, expected, None)
        assert watched == unwatched

    async def test_timeout_and_failover_order_is_unchanged(self):
        policy = load_model_policy()
        policy.timeouts.inference_seconds = 0.03
        router, calls = build_router(
            {QWEN: [VALID, VALID], GEMMA: [VALID, VALID]}, delay=5, policy=policy
        )
        with pytest.raises(TaskExhausted) as caught:
            await router.execute(make_task(Recorder(), policy=policy))
        assert [model for model, _ in calls] == [QWEN, QWEN, GEMMA, GEMMA]
        assert caught.value.failure_class.value == "inference_timeout"

    async def test_the_timeout_does_not_get_shorter_with_a_heartbeat(self):
        """A wait far longer than the heartbeat must simply keep going."""
        policy = load_model_policy()
        policy.timeouts.inference_seconds = 5
        display = Recorder(interval=0.01)
        router, calls = build_router({QWEN: [VALID]}, delay=0.15, policy=policy)
        outcome = await router.execute(make_task(display, policy=policy))
        assert outcome.model == QWEN
        assert len(calls) == 1
        assert display.errors == []
        assert display.warnings == []

    async def test_a_broken_display_changes_nothing(self):
        class Broken(Recorder):
            def _fail(self, *args):
                raise OSError("terminal went away")

            status = warn = error = progress = progress_done = _fail

        router, calls = build_router({QWEN: ["bad", "bad"], GEMMA: [VALID]}, delay=0.05)
        outcome = await router.execute(make_task(Broken()))
        assert outcome.model == GEMMA
        assert [model for model, _ in calls] == [QWEN, QWEN, GEMMA]

    async def test_the_model_never_sees_the_display(self):
        display = Recorder()
        router, calls = build_router({QWEN: [VALID]}, delay=0.08)
        await router.execute(make_task(display))
        assert calls == [(QWEN, calls[0][1])]
        assert "waiting" not in calls[0][1].lower()


class TestCoverageAcrossTheWorkflow:
    async def _run_step_with_repair(self, tmp_path) -> tuple[Harness, Recorder]:
        harness = Harness(tmp_path)
        harness.team.sources = {"calc.py": CALC_BUGGY}
        display = Recorder(interval=30)
        workflow = harness.workflow(harness.manager.create(), reporter=display)
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()
        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED
        return harness, display

    @pytest.mark.parametrize(
        "line",
        [
            "Business Analyst — Product discussion",
            "Business Analyst — Product summary",
            "Business Analyst — Roadmap",
            "Business Analyst — Step preparation, Step 1",
            "Software Architect — Technical design, Step 1",
            "Developer — Implementation plan, Step 1",
            "Test Engineer — Test plan, Step 1",
            "Test Engineer — Writing test file test_calc.py (1 of 1)",
            "Business Analyst — Step proposal, Step 1",
            "Developer — Writing calc.py (1 of 1)",
            "Developer — Repair 1 of 2",
            "Test Engineer — Test result review, Step 1",
            "Software Architect — Final review, Step 1",
            "Business Analyst — Step report, Step 1",
            "Test Engineer — Acceptance tests, Step 1",
        ],
    )
    async def test_every_long_operation_names_itself(self, tmp_path, line):
        _, display = await self._run_step_with_repair(tmp_path)
        assert display.has(line)

    async def test_the_model_is_shown_for_each_role(self, tmp_path):
        _, display = await self._run_step_with_repair(tmp_path)
        assert display.has("Model: gemma4:latest")
        assert display.has("Model: qwen2.5:7b")

    async def test_file_generation_is_reported(self, tmp_path):
        _, display = await self._run_step_with_repair(tmp_path)
        assert any(re.match(r"File written: calc\.py \(\d+ bytes\)\.", line) for line in display.lines)
        assert display.has("All 1 planned file(s) are written and recorded.")

    async def test_the_real_test_result_is_reported(self, tmp_path):
        _, display = await self._run_step_with_repair(tmp_path)
        assert display.has("Running test process...")
        assert any(
            re.match(r"Test process finished — \d+ seconds?: 2 passed, 0 failed\.", line)
            for line in display.finished
        )
        assert any("0 passed, 2 failed" in line or "2 failed" in line for line in display.finished)

    async def test_failed_tests_are_a_warning_not_a_crash_message(self, tmp_path):
        _, display = await self._run_step_with_repair(tmp_path)
        assert any("Tests did not pass" in line for line in display.warnings)

    async def test_a_slow_test_process_gets_a_heartbeat(self, tmp_path):
        slow = "import time\n\n\ndef test_slow():\n    time.sleep(0.4)\n"
        harness = Harness(tmp_path)
        harness.team.script(
            "test_file", Text(json.dumps({"path": "test_calc.py", "content": slow}))
        )
        display = Recorder(interval=0.05)
        workflow = harness.workflow(harness.manager.create(), reporter=display)
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()
        assert any("Test process still running" in line for line in display.heartbeats)

    async def test_a_test_process_past_its_limit_is_a_timeout(self, tmp_path):
        hang = "import time\n\n\ndef test_hang():\n    time.sleep(60)\n"
        harness = Harness(tmp_path)
        harness.policy.workflow.max_repair_cycles = 0
        harness.team.script(
            "test_file", Text(json.dumps({"path": "test_calc.py", "content": hang}))
        )
        display = Recorder(interval=0.05)
        workflow = harness.workflow(harness.manager.create(), reporter=display)
        await to_proposal_gate(workflow)
        profile = workflow.profiles["python_pytest"]
        workflow.profiles["python_pytest"] = profile.model_copy(update={"timeout_seconds": 0.5})
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        assert any("Timed out: the test process exceeded" in line for line in display.warnings)
        assert not any("Test process finished" in line for line in display.finished)
        assert workflow.state.phase == ProductPhase.BLOCKED

    async def test_regression_suites_are_named(self, tmp_path):
        harness = Harness(tmp_path)
        display = Recorder(interval=30)
        workflow = harness.workflow(harness.manager.create(), reporter=display)
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()
        workflow.start_step(APPROVE)
        await workflow.advance()
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()
        assert display.has("Test Engineer — Regression tests, Step 1")

    async def test_a_human_granted_repair_counts_correctly(self, tmp_path):
        harness = Harness(tmp_path)
        harness.team.sources = {"calc.py": CALC_BUGGY}
        harness.team.repaired_sources = {"calc.py": "def add(a, b):\n    return a * b\n"}
        harness.policy.workflow.max_repair_cycles = 1
        display = Recorder(interval=30)
        workflow = harness.workflow(harness.manager.create(), reporter=display)
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()
        assert display.has("Developer — Repair 1 of 1")

    async def test_switching_models_is_shown(self, tmp_path):
        _, display = await self._run_step_with_repair(tmp_path)
        assert any(
            line.startswith("Releasing ") and "from memory" in line for line in display.lines
        )


class TestPersistenceIsUnchanged:
    async def _events(self, tmp_path, reporter) -> tuple[list[str], list[str]]:
        harness = Harness(tmp_path)
        harness.team.sources = {"calc.py": CALC_BUGGY}
        store = harness.manager.create()
        workflow = harness.workflow(store, reporter=reporter)
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()
        files = sorted(
            path.relative_to(store.root).as_posix()
            for path in store.root.rglob("*")
            if path.is_file()
        )
        return [event.type for event in store.read_events()], files

    async def test_a_display_adds_no_events_and_no_files(self, tmp_path):
        shown = await self._events(tmp_path / "a", Recorder(interval=0.01))
        silent = await self._events(tmp_path / "b", None)
        assert shown == silent

    async def test_releasing_a_model_is_shown_but_never_recorded(self, tmp_path):
        display = Recorder(interval=30)
        types, _ = await self._events(tmp_path, display)
        assert "ModelUnloaded" in types, "this run really did switch models"
        assert any(line.startswith("Releasing ") for line in display.lines)
        assert not any("releasing" in name.lower() for name in types)

    async def test_heartbeats_are_not_written_to_the_audit_log(self, tmp_path):
        harness = Harness(tmp_path)
        display = Recorder(interval=0.001)
        harness.team.delay = 0.02
        workflow = harness.workflow(harness.manager.create(), reporter=display)
        await to_name_gate_quietly(workflow)
        assert display.heartbeats
        history = (workflow.store.logs_dir / "events.jsonl").read_text()
        assert "Still waiting" not in history
        assert "ModelReleasing" not in history
        assert "model_releasing" not in history

    async def test_interrupted_work_resumes_from_its_checkpoint(self, tmp_path):
        harness = Harness(tmp_path)
        harness.team.script("architect", Fail(SimulatedPowerLoss()))
        display = Recorder()
        workflow = harness.workflow(harness.manager.create(), reporter=display)
        await to_step_start_gate(workflow)
        workflow.start_step(APPROVE)
        with pytest.raises(SimulatedPowerLoss):
            await workflow.advance()

        restarted = harness.restart(workflow)
        await restarted.advance()

        assert restarted.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert len(harness.team.calls_of("step_brief")) == 1


async def to_name_gate_quietly(workflow) -> None:
    await workflow.discovery_message("A small calculator.")
    await workflow.advance()


def live_terminal(environ=None, read_line=None):
    stream = FakeTty()
    return ConsoleTerminal(stream, read_line or (lambda: "x"), environ or COLOR_ENV), stream


class TestLiveTerminal:
    def test_a_heartbeat_rewrites_one_line(self):
        terminal, stream = live_terminal()
        terminal.progress("Still waiting — 10 seconds elapsed.")
        terminal.progress("Still waiting — 20 seconds elapsed.")
        output = stream.getvalue()
        assert output.count("\r") == 2
        assert "\n" not in output

    def test_a_shorter_line_leaves_nothing_behind(self):
        terminal, stream = live_terminal({"TERM": "xterm", "NO_COLOR": "1"})
        terminal.progress("a much longer heartbeat line")
        terminal.progress("short")
        last = stream.getvalue().rsplit("\r", 1)[1]
        assert last == "  … short".ljust(len("  … a much longer heartbeat line"))

    def test_the_outcome_replaces_the_last_heartbeat(self):
        terminal, stream = live_terminal({"TERM": "xterm", "NO_COLOR": "1"})
        terminal.progress("Still waiting — 40 seconds elapsed.")
        terminal.progress_done("Model response received — 43 seconds.")
        output = stream.getvalue()
        assert output.endswith("\n")
        assert output.rsplit("\r", 1)[1].startswith("  … Model response received")

    def test_a_quick_operation_prints_the_outcome_as_a_plain_line(self):
        terminal, stream = live_terminal({"TERM": "xterm", "NO_COLOR": "1"})
        terminal.progress_done("Model response received — 3 seconds.")
        assert stream.getvalue() == "  … Model response received — 3 seconds.\n"

    def test_other_output_starts_on_its_own_line(self):
        terminal, stream = live_terminal({"TERM": "xterm", "NO_COLOR": "1"})
        terminal.progress("Still waiting — 10 seconds elapsed.")
        terminal.status("File written: a.py (10 bytes).")
        terminal.progress("Still waiting — 20 seconds elapsed.")
        lines = stream.getvalue().split("\n")
        assert lines[0].strip().endswith("10 seconds elapsed.")
        assert lines[1] == "  … File written: a.py (10 bytes)."
        assert lines[2].strip().endswith("20 seconds elapsed.")

    def test_a_warning_after_a_heartbeat_is_not_glued_to_it(self):
        terminal, stream = live_terminal({"TERM": "xterm", "NO_COLOR": "1"})
        terminal.progress("Still waiting — 10 seconds elapsed.")
        terminal.warn("Timed out.")
        assert "elapsed.\n\nWarning: Timed out.\n" in stream.getvalue()

    def test_the_prompt_starts_on_its_own_line_after_a_heartbeat(self):
        terminal, stream = live_terminal({"TERM": "xterm", "NO_COLOR": "1"})
        terminal.progress("Still waiting — 10 seconds elapsed.")
        terminal.ask("")
        assert stream.getvalue().endswith("elapsed.\n\nYou:\n")

    def test_labels_still_appear_after_a_heartbeat(self):
        terminal, stream = live_terminal({"TERM": "xterm", "NO_COLOR": "1"})
        terminal.say("One.")
        terminal.progress("Still waiting.")
        terminal.say("Two.")
        assert stream.getvalue().count("Business Analyst:") == 2

    def test_heartbeats_use_the_subdued_style(self):
        terminal, stream = live_terminal()
        terminal.progress("Still waiting.")
        assert f"{STATUS_TEXT}  … Still waiting.{RESET}" in stream.getvalue()

    def test_no_color_keeps_the_live_line_but_uses_no_escape_codes(self):
        terminal, stream = live_terminal({"TERM": "xterm", "NO_COLOR": "1"})
        terminal.progress("Still waiting.")
        assert "\r" in stream.getvalue()
        assert ESC not in stream.getvalue()

    def test_a_heartbeat_cannot_interrupt_typing(self):
        holder = {}

        def typing() -> str:
            holder["terminal"].progress("Still waiting — 10 seconds elapsed.")
            return "my answer"

        terminal, stream = live_terminal(read_line=typing)
        holder["terminal"] = terminal
        assert terminal.ask("Question?") == "my answer"
        assert "Still waiting" not in stream.getvalue()

    def test_heartbeats_resume_after_typing(self):
        terminal, stream = live_terminal()
        terminal.ask("")
        terminal.progress("Still waiting.")
        assert "Still waiting." in stream.getvalue()

    def test_model_text_cannot_inject_codes_through_a_heartbeat(self):
        terminal, stream = live_terminal({"TERM": "xterm", "NO_COLOR": "1"})
        terminal.progress("\x1b[2Jcleared", "\x1b[31mcontext")
        assert ESC not in stream.getvalue()

    def test_writes_from_many_threads_never_interleave(self):
        terminal, stream = live_terminal({"TERM": "xterm", "NO_COLOR": "1"})

        def speak(number: int) -> None:
            for index in range(120):
                terminal.status(f"thread-{number}-line-{index}-end")
                terminal.progress(f"beat-{number}-{index}-end")

        workers = [threading.Thread(target=speak, args=(n,)) for n in range(6)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        for chunk in re.split(r"[\r\n]", stream.getvalue()):
            text = chunk.strip()
            if text:
                assert re.fullmatch(r"… (thread-\d+-line-\d+|beat-\d+-\d+)-end", text), text


class TestNonInteractive:
    def _terminal(self, environ=None):
        stream = io.StringIO()
        return ConsoleTerminal(stream, lambda: "x", environ or COLOR_ENV), stream

    def test_a_heartbeat_is_a_complete_log_line(self):
        terminal, stream = self._terminal()
        terminal.progress(
            "Still waiting for model response — 30 seconds elapsed.",
            "Developer — Repair 1 of 2 (qwen2.5:7b)",
        )
        assert re.fullmatch(
            r"\[\d\d:\d\d:\d\d\] Developer — Repair 1 of 2 \(qwen2\.5:7b\): "
            r"Still waiting for model response — 30 seconds elapsed\.\n",
            stream.getvalue(),
        )

    def test_no_terminal_animation_or_color(self):
        terminal, stream = self._terminal()
        terminal.progress("One.", "Context")
        terminal.progress("Two.", "Context")
        terminal.progress_done("Done.", "Context")
        output = stream.getvalue()
        assert "\r" not in output
        assert ESC not in output
        assert len(output.strip().split("\n")) == 3

    def test_the_outcome_is_also_a_log_line(self):
        terminal, stream = self._terminal()
        terminal.progress_done("Model response received — 43 seconds.", "Developer (qwen)")
        assert re.fullmatch(
            r"\[\d\d:\d\d:\d\d\] Developer \(qwen\): Model response received — 43 seconds\.\n",
            stream.getvalue(),
        )

    def test_a_dumb_terminal_gets_log_lines(self):
        stream = FakeTty()
        terminal = ConsoleTerminal(stream, lambda: "x", {"TERM": "dumb"})
        terminal.progress("One.", "Context")
        assert "\r" not in stream.getvalue()
        assert re.match(r"\[\d\d:\d\d:\d\d\]", stream.getvalue())

    def test_logs_beat_more_slowly_than_a_live_terminal(self):
        assert ConsoleTerminal(io.StringIO(), environ=COLOR_ENV).heartbeat_seconds == LOG_HEARTBEAT_SECONDS
        assert ConsoleTerminal(FakeTty(), environ=COLOR_ENV).heartbeat_seconds == HEARTBEAT_SECONDS
        assert HEARTBEAT_SECONDS == 10

    def test_the_interval_can_be_configured(self):
        environ = {**COLOR_ENV, "LAB_HEARTBEAT_SECONDS": "5"}
        assert ConsoleTerminal(FakeTty(), environ=environ).heartbeat_seconds == 5
        assert ConsoleTerminal(io.StringIO(), environ=environ).heartbeat_seconds == 5

    @pytest.mark.parametrize("value", ["", "soon", "0", "-5"])
    def test_a_bad_interval_is_ignored(self, value):
        environ = {**COLOR_ENV, "LAB_HEARTBEAT_SECONDS": value}
        assert ConsoleTerminal(FakeTty(), environ=environ).heartbeat_seconds == HEARTBEAT_SECONDS

    async def test_a_whole_task_reads_sensibly_as_a_log(self):
        stream = io.StringIO()
        terminal = ConsoleTerminal(stream, lambda: "x", {**COLOR_ENV, "LAB_HEARTBEAT_SECONDS": "0.03"})
        router, _ = build_router({QWEN: [VALID]}, delay=0.1)
        await router.execute(make_task(terminal))
        lines = stream.getvalue().strip("\n").split("\n")
        assert lines[0] == "  … Developer — Repair 1 of 2"
        assert lines[1] == "  … Model: qwen2.5:7b"
        assert "Waiting for model response" in lines[2]
        assert "Still waiting for model response" in lines[3]
        assert "Model response received" in lines[-1]


async def test_real_seconds_through_a_real_terminal_object():
    """No shortened interval or fake clock: real one-second heartbeats, read back."""
    stream = FakeTty()
    environ = {"TERM": "xterm", "NO_COLOR": "1", "LAB_HEARTBEAT_SECONDS": "1"}
    terminal = ConsoleTerminal(stream, lambda: "x", environ)
    router, _ = build_router({QWEN: [VALID]}, delay=2.3)
    await router.execute(make_task(terminal))
    output = stream.getvalue()
    assert "Still waiting for model response — 1 second elapsed." in output
    assert "Still waiting for model response — 2 seconds elapsed." in output
    assert "Model response received — 2 seconds." in output


DISCOVERY_REPLY = json.dumps(
    {
        "reply": "Tell me more.",
        "understanding": "A small game.",
        "category": "game",
        "questions": [],
        "ready_for_brief": False,
    }
)


def cli_env(stub: StubOllama, **extra: str) -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/tmp"),
        "OLLAMA_API_BASE": stub.base_url,
        "LAB_HEARTBEAT_SECONDS": "0.2",
        **extra,
    }


class TestRealProcess:
    """The real CLI, runtime, ADK and LiteLLM, against a slow stub.  No inference."""

    @pytest.fixture
    def stub(self):
        server = StubOllama()
        yield server
        server.close()

    def test_redirected_output_gets_log_lines(self, tmp_path, stub):
        stub.reply_delay = 1.2
        stub.chat_replies.append({"content": DISCOVERY_REPLY})

        result = subprocess.run(
            [sys.executable, "-m", "lab", "--products-dir", str(tmp_path), "new", "A small game"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=cli_env(stub),
            timeout=120,
        )

        assert result.returncode == 0, result.stderr[-800:]
        assert "Traceback" not in result.stderr
        assert "Business Analyst — Product discussion" in result.stdout
        assert "Model: gemma4:latest" in result.stdout
        assert re.search(
            r"\[\d\d:\d\d:\d\d\] Business Analyst — Product discussion \(gemma4:latest\): "
            r"Still waiting for model response",
            result.stdout,
        )
        assert "Model response received" in result.stdout
        assert "\r" not in result.stdout
        assert ESC not in result.stdout
        assert "Business Analyst:\nTell me more." in result.stdout

    def test_ctrl_c_during_a_wait_stops_cleanly_and_can_resume(self, tmp_path, stub):
        """A real SIGINT, on a real pseudo-terminal, while the model has not answered."""
        stub.reply_delay = 40
        stub.chat_replies.append({"content": DISCOVERY_REPLY})
        master, replica = pty.openpty()
        process = subprocess.Popen(
            [sys.executable, "-m", "lab", "--products-dir", str(tmp_path), "new", "A small game"],
            stdin=replica,
            stdout=replica,
            stderr=replica,
            cwd=PROJECT_ROOT,
            env=cli_env(stub, TERM="xterm-256color"),
        )

        def read_available(wait: float) -> bytes:
            ready, _, _ = select.select([master], [], [], wait)
            return os.read(master, 65536) if ready else b""

        output = b""
        interrupted = False
        deadline = time.monotonic() + 60
        try:
            while process.poll() is None and time.monotonic() < deadline:
                output += read_available(0.1)
                if not interrupted and output.count(b"Still waiting") >= 3:
                    interrupted = True
                    process.send_signal(signal.SIGINT)
            # The replica end stays open here, so nothing buffered can be lost at exit.
            while chunk := read_available(0.3):
                output += chunk
        finally:
            os.close(master)
            os.close(replica)
            if process.poll() is None:
                process.kill()
            process.wait()

        text = output.decode("utf-8", "replace")
        assert interrupted, text[-800:]
        assert process.returncode == 0, text[-800:]
        assert "Traceback" not in text
        assert "\r" in text, "a live terminal gets the in-place line"
        assert "Interrupted while waiting for the model" in text
        # Which of these appears depends on where the signal lands.  Both say work is safe.
        assert "Goodbye" in text or "Stopped." in text
        assert "Model response received" not in text
        interrupted_at = text.index("Interrupted while waiting")
        assert "Still waiting" not in text[interrupted_at:], "no heartbeat after stopping"

        store = ProductStore(next(tmp_path.glob("draft-*")))
        state = store.load_state()
        assert not store.lock_path.exists()
        assert state.active_task is not None, "the task stays on disk, ready to resume"
        assert state.pending_input == "A small game"
        assert latest_roadmap(store, state) is None

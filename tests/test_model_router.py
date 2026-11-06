"""Retry, failover, failure classification and the memory policy."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from lab.config import load_model_policy
from lab.playbooks import PlaybookLibrary
from lab.product.manager import ProductManager
from lab.product.store import DurableAttemptLog
from lab.runtime.router import (
    DependencyFailure,
    FilesystemFailure,
    InMemoryAttemptLog,
    ModelRouter,
    ModelTask,
    TaskExhausted,
    ToolExecutionFailure,
    classify_failure,
    failure_message,
    strip_thinking,
)
from lab.schemas.product import FailureClass
from lab.tools.developer_tools import SecurityError
from tests.fakes import GEMMA, QWEN, FakeLifecycle, ScriptedRuntime, ScriptedTeam

VALID = '{"name": "ok", "count": 1}'


class Answer(BaseModel):
    name: str
    count: int


class ScriptedModel(ScriptedRuntime):
    """Plays back a per-model list of outcomes for router-level tests."""

    def __init__(self, model, outcomes, log):
        super().__init__(model, ScriptedTeam())
        self.outcomes = outcomes
        self.log = log

    async def generate(self, instruction, user_message, tools=None, json_mode=False,
                       session_id="generate"):
        self.log.append((self._model, user_message))
        outcome = self.outcomes.pop(0) if self.outcomes else VALID
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome, []


def build(outcomes: dict[str, list], *, log=None, policy=None, attempt_log=None):
    calls: list[tuple[str, str]] = []
    lifecycle = FakeLifecycle()
    router = ModelRouter(
        policy or load_model_policy(),
        lambda spec: ScriptedModel(spec.id, outcomes.get(spec.id, []), calls),
        lifecycle,
        attempt_log=attempt_log,
    )
    return router, calls, lifecycle


def task(**overrides) -> ModelTask:
    defaults = dict(
        task_id="t1",
        role="developer",
        kind="demo",
        chain=[QWEN, GEMMA],
        instruction="be precise",
        build_prompt=lambda model: f"primary prompt for {model}",
        build_failover_prompt=lambda model, failures: (
            f"handover to {model} after {len(failures)} failures"
        ),
        schema=Answer,
    )
    defaults.update(overrides)
    return ModelTask(**defaults)


class TestRetryAndFailover:
    async def test_first_attempt_success_makes_exactly_one_call(self):
        router, calls, _ = build({QWEN: [VALID]})
        outcome = await router.execute(task())
        assert outcome.value == Answer(name="ok", count=1)
        assert outcome.model == QWEN
        assert len(calls) == 1
        assert not outcome.failed_over

    async def test_invalid_json_is_retried_on_the_same_model_with_feedback(self):
        router, calls, _ = build({QWEN: ["not json at all", VALID]})
        outcome = await router.execute(task())
        assert outcome.model == QWEN
        assert [model for model, _ in calls] == [QWEN, QWEN]
        assert "CORRECTION" in calls[1][1]
        assert "no valid JSON" in calls[1][1]

    async def test_empty_answer_is_retried(self):
        router, calls, _ = build({QWEN: ["   ", VALID]})
        await router.execute(task())
        assert len(calls) == 2
        assert "empty" in calls[1][1]

    async def test_schema_invalid_answer_is_retried(self):
        router, calls, _ = build({QWEN: ['{"name": "x"}', VALID]})
        await router.execute(task())
        assert len(calls) == 2
        assert "count" in calls[1][1]

    async def test_two_failures_switch_to_the_alternate_model(self):
        router, calls, _ = build({QWEN: ["bad", "bad"], GEMMA: [VALID]})
        outcome = await router.execute(task())
        assert [model for model, _ in calls] == [QWEN, QWEN, GEMMA]
        assert outcome.model == GEMMA
        assert outcome.failed_over
        assert outcome.attempts[-1].switched_from == QWEN

    async def test_the_alternate_gets_two_attempts(self):
        router, calls, _ = build({QWEN: ["bad", "bad"], GEMMA: ["bad", VALID]})
        outcome = await router.execute(task())
        assert [model for model, _ in calls] == [QWEN, QWEN, GEMMA, GEMMA]
        assert outcome.model == GEMMA

    async def test_never_more_than_four_attempts_in_total(self):
        router, calls, _ = build({QWEN: ["bad"] * 9, GEMMA: ["bad"] * 9})
        with pytest.raises(TaskExhausted) as caught:
            await router.execute(task())
        assert len(calls) == 4
        assert len(caught.value.attempts) == 4
        assert caught.value.failure_class == FailureClass.INVALID_JSON

    async def test_the_fallback_never_sees_malformed_output(self):
        leak = 'LEAKED_MALFORMED_TEXT {"name":'
        router, calls, _ = build({QWEN: [leak, leak], GEMMA: [VALID]})
        await router.execute(task())
        assert "handover to" in calls[2][1]
        assert "LEAKED_MALFORMED_TEXT" not in calls[2][1]

    async def test_schema_errors_are_exact_but_do_not_echo_model_output(self):
        router, calls, _ = build(
            {QWEN: ['{"name": "SECRET_VALUE", "count": "not-a-number"}'] * 2, GEMMA: [VALID]}
        )
        outcome = await router.execute(task())
        assert "count" in outcome.attempts[0].error
        assert "SECRET_VALUE" not in outcome.attempts[0].error
        assert "SECRET_VALUE" not in calls[1][1]

    async def test_semantic_validator_failures_are_retried_with_the_reason(self):
        def must_be_positive(value: Answer) -> None:
            if value.count < 1:
                raise ValueError("count must be positive")

        router, calls, _ = build({QWEN: ['{"name": "x", "count": 0}', VALID]})
        await router.execute(task(validator=must_be_positive))
        assert "count must be positive" in calls[1][1]

    async def test_inline_reasoning_is_removed_before_parsing(self):
        reply = f"<think>private reasoning</think>{VALID}"
        router, calls, _ = build({QWEN: [reply]})
        outcome = await router.execute(task())
        assert "private" not in outcome.text

    def test_strip_thinking_handles_case_and_newlines(self):
        assert strip_thinking("<Think>\na\nb\n</Think>\nanswer") == "answer"


class TestDeterministicFailures:
    @pytest.mark.parametrize("error", [FilesystemFailure("disk"), OSError(28, "No space left")])
    async def test_filesystem_failure_does_not_retry_or_switch(self, error):
        router, calls, _ = build({QWEN: [error]})
        with pytest.raises(TaskExhausted) as caught:
            await router.execute(task())
        assert len(calls) == 1
        assert caught.value.failure_class == FailureClass.FILESYSTEM

    async def test_dependency_failure_does_not_retry_or_switch(self):
        router, calls, _ = build({QWEN: [DependencyFailure("node is missing")]})
        with pytest.raises(TaskExhausted) as caught:
            await router.execute(task())
        assert len(calls) == 1
        assert caught.value.failure_class == FailureClass.DEPENDENCY

    async def test_connection_failures_are_retried_and_switched(self):
        router, calls, _ = build(
            {QWEN: [ConnectionRefusedError("refused")] * 2, GEMMA: [VALID]}
        )
        outcome = await router.execute(task())
        assert outcome.model == GEMMA
        assert outcome.attempts[0].failure_class == FailureClass.MODEL_CONNECTION

    async def test_a_hung_model_times_out(self):
        policy = load_model_policy()
        policy.timeouts.inference_seconds = 0.02
        lifecycle = FakeLifecycle()

        class Hung(ScriptedModel):
            async def generate(self, *args, **kwargs):
                await asyncio.sleep(30)

        router = ModelRouter(
            policy,
            lambda spec: Hung(spec.id, [], []) if spec.id == QWEN
            else ScriptedModel(spec.id, [VALID], []),
            lifecycle,
        )
        outcome = await router.execute(task())
        assert outcome.model == GEMMA
        assert outcome.attempts[0].failure_class == FailureClass.INFERENCE_TIMEOUT


class TestFailureClassification:
    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (TimeoutError("slow"), FailureClass.INFERENCE_TIMEOUT),
            (asyncio.TimeoutError(), FailureClass.INFERENCE_TIMEOUT),
            (ConnectionError("gone"), FailureClass.MODEL_CONNECTION),
            (ConnectionRefusedError(), FailureClass.MODEL_CONNECTION),
            (PermissionError("denied"), FailureClass.FILESYSTEM),
            (FileNotFoundError("gone"), FailureClass.FILESYSTEM),
            (SecurityError("outside workspace"), FailureClass.TOOL_EXECUTION),
            (ToolExecutionFailure("no write"), FailureClass.TOOL_EXECUTION),
            (ValueError("something else"), FailureClass.INVALID_RESPONSE),
        ],
    )
    def test_classification(self, error, expected):
        assert classify_failure(error) == expected

    def test_litellm_errors_are_recognised_by_type(self):
        import litellm

        timeout = litellm.Timeout("slow", model="m", llm_provider="ollama_chat")
        connection = litellm.APIConnectionError("down", model="m", llm_provider="ollama_chat")
        assert classify_failure(timeout) == FailureClass.INFERENCE_TIMEOUT
        assert classify_failure(connection) == FailureClass.MODEL_CONNECTION

    def test_the_cause_chain_is_inspected(self):
        try:
            try:
                raise ConnectionRefusedError("refused")
            except ConnectionRefusedError as inner:
                raise RuntimeError("wrapped by ADK") from inner
        except RuntimeError as outer:
            assert classify_failure(outer) == FailureClass.MODEL_CONNECTION

    def test_validation_error_is_schema_failure(self):
        with pytest.raises(ValidationError) as caught:
            Answer.model_validate({"name": 1})
        assert classify_failure(caught.value) == FailureClass.SCHEMA_VALIDATION
        assert "name" in failure_message(caught.value)


class TestDurableBudget:
    def _log(self, tmp_path: Path) -> DurableAttemptLog:
        store = ProductManager(tmp_path, PlaybookLibrary()).create("Sample")
        return DurableAttemptLog(store)

    async def test_failed_attempts_survive_a_restart(self, tmp_path):
        log = self._log(tmp_path)
        first, calls_one, _ = build({QWEN: ["bad", "bad"], GEMMA: ["bad", "bad"]}, attempt_log=log)
        with pytest.raises(TaskExhausted):
            await first.execute(task())
        assert len(calls_one) == 4

        second, calls_two, _ = build({QWEN: [VALID], GEMMA: [VALID]}, attempt_log=log)
        with pytest.raises(TaskExhausted):
            await second.execute(task())
        assert calls_two == [], "a restart must not grant a fresh budget"

    async def test_partial_budget_continues_on_the_right_model(self, tmp_path):
        log = self._log(tmp_path)
        store = log._store
        first, _, _ = build({QWEN: ["bad", "bad"], GEMMA: ["bad", "bad"]}, attempt_log=log)
        with pytest.raises(TaskExhausted):
            await first.execute(task(chain=[QWEN]))

        second, calls, _ = build({GEMMA: [VALID]}, attempt_log=log)
        outcome = await second.execute(task())
        assert [model for model, _ in calls] == [GEMMA]
        assert "handover" in calls[0][1]
        assert outcome.attempts[-1].switched_from == QWEN
        assert store.current_round("t1") == 1

    async def test_interrupted_attempts_do_not_consume_budget(self, tmp_path):
        log = self._log(tmp_path)
        from lab.schemas.product import ModelAttempt

        for number in (1, 2):
            log.record(
                ModelAttempt(
                    attempt_id=f"i{number}", task_id="t1", model=QWEN,
                    model_attempt=1, overall_attempt=1,
                )
            )
        router, calls, _ = build({QWEN: [VALID]}, attempt_log=log)
        outcome = await router.execute(task())
        assert outcome.model == QWEN
        assert len(calls) == 1
        assert "handover" not in calls[0][1]

    async def test_a_human_retry_grants_a_fresh_round(self, tmp_path):
        log = self._log(tmp_path)
        first, _, _ = build({QWEN: ["bad", "bad"], GEMMA: ["bad", "bad"]}, attempt_log=log)
        with pytest.raises(TaskExhausted):
            await first.execute(task())

        log._store.start_new_round("t1")
        second, calls, _ = build({QWEN: [VALID]}, attempt_log=log)
        outcome = await second.execute(task())
        assert outcome.model == QWEN
        assert len(calls) == 1

    async def test_every_attempt_and_transition_is_recorded(self, tmp_path):
        log = self._log(tmp_path)
        router, _, _ = build({QWEN: ["bad", "bad"], GEMMA: [VALID]}, attempt_log=log)
        await router.execute(task())

        attempts = log.load("t1")
        assert [item.status for item in attempts] == ["failed", "failed", "succeeded"]
        assert [item.model for item in attempts] == [QWEN, QWEN, GEMMA]
        assert attempts[2].switched_from == QWEN
        raw = (log._store.task_dir("t1") / "attempts.jsonl").read_text().splitlines()
        assert len(raw) == 6, "each attempt writes a start line and a finish line"
        assert all(json.loads(line)["task_id"] == "t1" for line in raw)


class TestMemoryPolicy:
    async def test_only_one_inference_runs_at_a_time(self):
        team = ScriptedTeam(delay=0.01)
        lifecycle = FakeLifecycle()
        team.lifecycle = lifecycle
        router = ModelRouter(load_model_policy(), team.runtime_factory, lifecycle)

        def make(identifier: str) -> ModelTask:
            return task(
                task_id=f"step-01-architect-v{identifier}",
                schema=None,
                chain=[QWEN, GEMMA] if int(identifier) % 2 else [GEMMA, QWEN],
                postcondition=lambda text, events: None,
            )

        await asyncio.gather(*(router.execute(make(str(n))) for n in range(1, 7)))

        assert team.max_active == 1
        assert team.max_resident == 1

    async def test_the_resident_model_is_unloaded_before_switching(self):
        team = ScriptedTeam()
        lifecycle = FakeLifecycle()
        team.lifecycle = lifecycle
        router = ModelRouter(load_model_policy(), team.runtime_factory, lifecycle)

        common = dict(schema=None, postcondition=lambda text, events: None)
        await router.execute(task(task_id="step-01-architect-v1", chain=[QWEN], **common))
        assert lifecycle.unloads == []
        await router.execute(task(task_id="step-01-architect-v2", chain=[GEMMA], **common))
        assert lifecycle.unloads == [QWEN]
        assert team.max_resident == 1

    async def test_the_same_model_is_not_reloaded_needlessly(self):
        team = ScriptedTeam()
        lifecycle = FakeLifecycle()
        team.lifecycle = lifecycle
        router = ModelRouter(load_model_policy(), team.runtime_factory, lifecycle)
        common = dict(schema=None, postcondition=lambda text, events: None, chain=[QWEN])
        await router.execute(task(task_id="step-01-architect-v1", **common))
        await router.execute(task(task_id="step-01-architect-v2", **common))
        assert lifecycle.unloads == []

    async def test_a_leftover_model_from_a_crashed_session_is_released_first(self):
        team = ScriptedTeam()
        lifecycle = FakeLifecycle()
        lifecycle.loaded.add(GEMMA)
        team.lifecycle = lifecycle
        router = ModelRouter(load_model_policy(), team.runtime_factory, lifecycle)
        await router.execute(
            task(
                task_id="step-01-architect-v1", chain=[QWEN], schema=None,
                postcondition=lambda text, events: None,
            )
        )
        assert lifecycle.unloads == [GEMMA]

    async def test_failover_also_releases_the_failed_model(self):
        router, _, lifecycle = build({QWEN: ["bad", "bad"], GEMMA: [VALID]})
        await router.execute(task())
        assert lifecycle.unloads == [QWEN]

    async def test_release_all_frees_memory_on_exit(self):
        router, _, lifecycle = build({QWEN: [VALID]})
        await router.execute(task())
        await router.release_all()
        assert lifecycle.unloads == [QWEN]

    async def test_a_failing_unload_does_not_stop_the_work(self):
        router, _, lifecycle = build({QWEN: ["bad", "bad"], GEMMA: [VALID]})

        async def broken(model_id):
            raise ConnectionError("no")

        lifecycle.unload = broken
        outcome = await router.execute(task())
        assert outcome.model == GEMMA

    async def test_a_tool_task_needs_its_postcondition(self):
        def nothing_written(text, events):
            raise ToolExecutionFailure("file not written")

        router, calls, _ = build({QWEN: ["done"] * 2, GEMMA: ["done"] * 2})
        with pytest.raises(TaskExhausted) as caught:
            await router.execute(task(schema=None, postcondition=nothing_written))
        assert len(calls) == 4
        assert caught.value.failure_class == FailureClass.TOOL_EXECUTION

    def test_in_memory_log_is_available_outside_a_product(self):
        assert InMemoryAttemptLog().current_round("x") == 1

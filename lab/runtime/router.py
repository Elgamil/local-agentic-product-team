"""Model routing, bounded retry, automatic failover and sequential execution.

Rules enforced for each logical task, per round:

1. Attempt the preferred model.
2. On a retryable failure, retry that model once with corrective feedback.
3. Switch to the alternate model, which gets up to two attempts.
4. A switched model receives a prompt rebuilt from the last valid persisted context
   plus diagnostics.  It never receives the malformed output of the failed attempt.
5. Every attempt and model transition is recorded before and after it happens.
6. Deterministic failures (filesystem, dependency) stop at once: changing the model
   cannot fix them.
7. Only one inference runs at a time, and the resident model is released before a
   different one loads.

The attempt log is durable, so a restart continues a task's budget rather than
resetting it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Protocol, Sequence

from pydantic import BaseModel, ValidationError

from lab.config import ModelPolicy, ModelSpec
from lab.runtime.adk_ollama import extract_json
from lab.runtime.base import ModelRuntime
from lab.runtime.lifecycle import ModelLifecycle
from lab.schemas.product import (
    NON_RETRYABLE_FAILURES,
    FailureClass,
    ModelAttempt,
    utcnow,
)
from lab.tools.developer_tools import SecurityError

logger = logging.getLogger(__name__)

RouterEventHandler = Callable[[str, dict[str, Any]], None]


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


class TaskFailure(Exception):
    """An attempt failed for a known reason."""

    failure_class = FailureClass.INVALID_RESPONSE

    def __init__(self, message: str, failure_class: FailureClass | None = None) -> None:
        super().__init__(message)
        if failure_class is not None:
            self.failure_class = failure_class


class EmptyResponseError(TaskFailure):
    failure_class = FailureClass.INVALID_RESPONSE


class InvalidJsonError(TaskFailure):
    failure_class = FailureClass.INVALID_JSON


class SemanticValidationError(TaskFailure):
    """The JSON matched the schema but broke a rule only the orchestrator knows."""

    failure_class = FailureClass.SCHEMA_VALIDATION


class ToolExecutionFailure(TaskFailure):
    failure_class = FailureClass.TOOL_EXECUTION


class FilesystemFailure(TaskFailure):
    failure_class = FailureClass.FILESYSTEM


class DependencyFailure(TaskFailure):
    failure_class = FailureClass.DEPENDENCY


class TaskExhausted(Exception):
    """The task has no attempts left, or stopped on a deterministic failure."""

    def __init__(
        self,
        task_id: str,
        attempts: list[ModelAttempt],
        failure_class: FailureClass,
        message: str,
    ) -> None:
        super().__init__(message)
        self.task_id = task_id
        self.attempts = attempts
        self.failure_class = failure_class


_CONNECTION_NAMES = frozenset(
    {
        "APIConnectionError",
        "ConnectError",
        "ConnectTimeout",
        "ServiceUnavailableError",
        "InternalServerError",
        "BadGatewayError",
        "NotFoundError",
        "ClientConnectorError",
        "ServerDisconnectedError",
        "RemoteProtocolError",
        "ReadError",
    }
)
_TIMEOUT_NAMES = frozenset({"Timeout", "APITimeoutError", "TimeoutException", "ReadTimeout"})
_TOOL_NAMES = frozenset({"LlmCallsLimitExceededError", "ToolExecutionError"})


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def classify_failure(exc: BaseException) -> FailureClass:
    """Map any exception raised by an attempt to a :class:`FailureClass`.

    The order matters: ``TimeoutError`` and ``ConnectionError`` are both subclasses
    of ``OSError``, so they must be recognised before filesystem errors.
    """
    for link in _exception_chain(exc):
        if isinstance(link, TaskFailure):
            return link.failure_class
        if isinstance(link, ValidationError):
            return FailureClass.SCHEMA_VALIDATION
        if isinstance(link, SecurityError):
            return FailureClass.TOOL_EXECUTION
        if isinstance(link, TimeoutError):
            return FailureClass.INFERENCE_TIMEOUT
        if isinstance(link, ConnectionError):
            return FailureClass.MODEL_CONNECTION

        names = {cls.__name__ for cls in type(link).__mro__}
        if names & _TIMEOUT_NAMES:
            return FailureClass.INFERENCE_TIMEOUT
        if names & _CONNECTION_NAMES:
            return FailureClass.MODEL_CONNECTION
        if names & _TOOL_NAMES:
            return FailureClass.TOOL_EXECUTION

        if isinstance(link, OSError):
            return FailureClass.FILESYSTEM

    message = str(exc).lower()
    if "tool" in message and ("not found" in message or "function" in message):
        return FailureClass.TOOL_EXECUTION
    if "connection" in message or "refused" in message:
        return FailureClass.MODEL_CONNECTION
    return FailureClass.INVALID_RESPONSE


# ---------------------------------------------------------------------------
# Output handling
# ---------------------------------------------------------------------------

_THINKING_BLOCKS = re.compile(
    r"<(think|thinking|thought|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE
)


def strip_thinking(text: str) -> str:
    """Remove inline reasoning blocks.  Hidden chain-of-thought is never persisted."""
    return _THINKING_BLOCKS.sub("", text).strip()


def schema_instructions(schema: type[BaseModel]) -> str:
    return (
        "Reply ONLY with a single valid JSON object that follows this JSON schema. "
        "Put no commentary before or after it. Keep rationale concise: state decisions, "
        "reasons, assumptions, alternatives and risks, not your step-by-step thinking.\n\n"
        f"JSON schema:\n{json.dumps(schema.model_json_schema(), separators=(',', ':'))}"
    )


def parse_structured(text: str, schema: type[BaseModel]) -> BaseModel:
    """Parse and validate model text, sorting the failure into the right class."""
    if not text.strip():
        raise EmptyResponseError("The model returned an empty response.")
    try:
        data = extract_json(text)
    except ValueError as exc:
        # Deliberately not str(exc): that quotes the model's text, and this message
        # travels to the fallback model, which must never see malformed output.
        reason = "incomplete or truncated JSON" if "truncated" in str(exc) else "no valid JSON object"
        raise InvalidJsonError(f"The response contained {reason}.") from exc
    return schema.model_validate(data)


def failure_message(exc: BaseException) -> str:
    """Describe a failure exactly, without echoing any of the model's output."""
    for link in _exception_chain(exc):
        if isinstance(link, ValidationError):
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'response'}: {error['msg']}"
                for error in link.errors(include_url=False, include_input=False)
            )
            return f"Schema validation failed. {details}"[:1500]
    return str(exc)[:1500] or type(exc).__name__


# ---------------------------------------------------------------------------
# Task specification
# ---------------------------------------------------------------------------


class AttemptLog(Protocol):
    def load(self, task_id: str) -> list[ModelAttempt]: ...

    def record(self, attempt: ModelAttempt) -> None: ...

    def current_round(self, task_id: str) -> int: ...


class InMemoryAttemptLog:
    """Non-durable attempt log for tests and tools outside a product."""

    def __init__(self) -> None:
        self._attempts: dict[str, ModelAttempt] = {}

    def load(self, task_id: str) -> list[ModelAttempt]:
        return [
            attempt.model_copy()
            for attempt in self._attempts.values()
            if attempt.task_id == task_id
        ]

    def record(self, attempt: ModelAttempt) -> None:
        self._attempts[attempt.attempt_id] = attempt

    def current_round(self, task_id: str) -> int:
        return 1


class TaskObserver(Protocol):
    """Watches a task's attempts, for display only.

    An observer wraps the steps of an attempt without taking part in them.  It gets
    no access to the prompt or the runtime and cannot alter a result.
    """

    def waiting(self, attempt: ModelAttempt) -> ContextManager[Any]: ...

    def processing(self, attempt: ModelAttempt) -> ContextManager[Any]: ...

    def failed(self, attempt: ModelAttempt) -> None: ...


class _NoObserver:
    def waiting(self, attempt: ModelAttempt) -> ContextManager[Any]:
        return nullcontext()

    def processing(self, attempt: ModelAttempt) -> ContextManager[Any]:
        return nullcontext()

    def failed(self, attempt: ModelAttempt) -> None:
        pass


@dataclass
class ModelTask:
    """One logical unit of model work.

    Prompts are built on demand so that every attempt, including one after a crash
    or a model switch, starts from what is on disk and not from process memory.
    """

    task_id: str
    role: str
    kind: str
    chain: list[str]
    instruction: str
    build_prompt: Callable[[str], str]
    build_failover_prompt: Callable[[str, list[ModelAttempt]], str]
    schema: type[BaseModel] | None = None
    validator: Callable[[Any], None] | None = None
    tools: list[Any] | None = None
    postcondition: Callable[[str, list[dict]], None] | None = None
    io_logger: Any = None
    on_attempt_start: Callable[[ModelAttempt], None] | None = None
    on_prompt: Callable[[ModelAttempt, str], None] | None = None
    on_failed_output: Callable[[ModelAttempt, str], None] | None = None
    observer: TaskObserver | None = None


@dataclass
class TaskOutcome:
    task_id: str
    model: str
    value: Any
    text: str
    attempts: list[ModelAttempt] = field(default_factory=list)

    @property
    def models_tried(self) -> list[str]:
        return list(dict.fromkeys(attempt.model for attempt in self.attempts))

    @property
    def failed_over(self) -> bool:
        return len(self.models_tried) > 1


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class ModelRouter:
    """Executes :class:`ModelTask` objects under the retry and memory policy."""

    def __init__(
        self,
        policy: ModelPolicy,
        runtime_factory: Callable[[ModelSpec], ModelRuntime],
        lifecycle: ModelLifecycle,
        attempt_log: AttemptLog | None = None,
        on_event: RouterEventHandler | None = None,
    ) -> None:
        self.policy = policy
        self._runtime_factory = runtime_factory
        self._lifecycle = lifecycle
        self.attempt_log: AttemptLog = attempt_log or InMemoryAttemptLog()
        self.on_event: RouterEventHandler = on_event or (lambda kind, payload: None)
        self._inference_lock = asyncio.Lock()
        self._resident: str | None = None
        self._residency_checked = False
        self._runtimes: dict[str, ModelRuntime] = {}

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    async def _release(self, model_id: str) -> None:
        self.on_event("model_releasing", {"model": model_id})
        try:
            confirmed = await self._lifecycle.unload(model_id)
        except Exception as exc:  # Unloading is best effort; generation reports real faults.
            logger.warning("Could not unload %s: %s", model_id, exc)
            confirmed = False
        self.on_event("model_unloaded", {"model": model_id, "confirmed": confirmed})

    async def _make_resident(self, model_id: str) -> None:
        """Ensure no other model is loaded before *model_id* is used."""
        if not self.policy.memory.unload_between_models:
            self._resident = model_id
            return

        if not self._residency_checked:
            # A previous crashed session may have left a model loaded.
            self._residency_checked = True
            try:
                leftovers = await self._lifecycle.running()
            except Exception as exc:
                logger.warning("Could not list running models: %s", exc)
                leftovers = []
            for other in leftovers:
                if other != model_id:
                    await self._release(other)
        elif self._resident is not None and self._resident != model_id:
            await self._release(self._resident)

        self._resident = model_id

    async def release_all(self) -> None:
        """Free memory when a session ends."""
        async with self._inference_lock:
            if self._resident is not None:
                await self._release(self._resident)
                self._resident = None

    def _runtime(self, model_id: str) -> ModelRuntime:
        if model_id not in self._runtimes:
            self._runtimes[model_id] = self._runtime_factory(self.policy.spec_for(model_id))
        return self._runtimes[model_id]

    # ------------------------------------------------------------------
    # Attempt planning
    # ------------------------------------------------------------------

    def _next_model(self, task: ModelTask, attempts: Sequence[ModelAttempt]) -> str | None:
        """The first model in the chain with budget left, or None."""
        limit = self.policy.retry.attempts_per_model
        for model_id in task.chain:
            used = sum(1 for attempt in attempts if attempt.model == model_id)
            if used < limit:
                return model_id
        return None

    @staticmethod
    def _correction(task: ModelTask, failed: ModelAttempt) -> str:
        lines = [
            "",
            "CORRECTION: your previous answer could not be accepted.",
            f"Problem ({failed.failure_class.value if failed.failure_class else 'unknown'}):",
            failed.error or "",
        ]
        if task.schema is not None:
            lines.append(
                "Mandatory top-level fields: "
                f"{task.schema.model_json_schema().get('required', [])}"
            )
            lines.append("Return one complete JSON object that fixes this.")
            if "truncated" in (failed.error or ""):
                lines.append("Your answer was cut off, so keep every field much shorter.")
        return "\n".join(lines)

    def _prompt_for(
        self,
        task: ModelTask,
        model_id: str,
        round_attempts: list[ModelAttempt],
    ) -> str:
        failures = [attempt for attempt in round_attempts if attempt.status == "failed"]
        switched = any(attempt.model != model_id for attempt in failures)
        if switched:
            prompt = task.build_failover_prompt(model_id, failures)
        else:
            prompt = task.build_prompt(model_id)
        own_failures = [attempt for attempt in failures if attempt.model == model_id]
        if own_failures:
            prompt = f"{prompt}\n{self._correction(task, own_failures[-1])}"
        if task.schema is not None:
            prompt = f"{prompt}\n\n{schema_instructions(task.schema)}"
        return prompt

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _single_attempt(
        self, task: ModelTask, attempt: ModelAttempt, prompt: str, session_id: str
    ) -> tuple[str, Any]:
        observer = task.observer or _NoObserver()
        with observer.waiting(attempt):
            runtime = self._runtime(attempt.model)
            runtime._logger = task.io_logger  # type: ignore[attr-defined]
            text, events = await asyncio.wait_for(
                runtime.generate(
                    instruction=task.instruction,
                    user_message=prompt,
                    tools=task.tools,
                    json_mode=task.schema is not None,
                    session_id=session_id,
                ),
                timeout=self.policy.timeouts.inference_seconds,
            )

        with observer.processing(attempt):
            text = strip_thinking(text)
            try:
                if task.schema is not None:
                    value: Any = parse_structured(text, task.schema)
                    if task.validator is not None:
                        try:
                            task.validator(value)
                        except ValueError as exc:
                            raise SemanticValidationError(str(exc)) from exc
                else:
                    value = text

                if task.postcondition is not None:
                    task.postcondition(text, events)
                elif task.schema is None and not text:
                    raise EmptyResponseError("The model returned an empty response.")
            except Exception:
                self._keep_failed_output(task, attempt, text)
                raise
        return text, value

    @staticmethod
    def _keep_failed_output(task: ModelTask, attempt: ModelAttempt, text: str) -> None:
        """Save a rejected answer for the person who may have to fix it by hand.

        It goes to a file, never to the next model.  A failure to save is only logged:
        an OSError here would otherwise be mistaken for the attempt's own failure.
        """
        if task.on_failed_output is None or not text:
            return
        try:
            task.on_failed_output(attempt, text)
        except OSError as exc:
            logger.warning("Could not keep the failed output of %s: %s", task.task_id, exc)

    async def execute(self, task: ModelTask) -> TaskOutcome:
        """Run *task* to success, or raise :class:`TaskExhausted`.

        Filesystem faults raised outside an attempt propagate unchanged.
        """
        async with self._inference_lock:
            return await self._execute_locked(task)

    async def _execute_locked(self, task: ModelTask) -> TaskOutcome:
        log = self.attempt_log
        round_no = log.current_round(task.task_id)

        def round_attempts() -> list[ModelAttempt]:
            """Budgeted history.  Interrupted attempts are not the model's fault."""
            return [
                attempt
                for attempt in log.load(task.task_id)
                if attempt.round == round_no and attempt.status == "failed"
            ]

        while True:
            history = round_attempts()
            model_id = self._next_model(task, history)
            if model_id is None:
                last = history[-1]
                failure = last.failure_class or FailureClass.INVALID_RESPONSE
                self.on_event(
                    "task_exhausted",
                    {"task_id": task.task_id, "attempts": len(history), "error": last.error},
                )
                raise TaskExhausted(
                    task.task_id,
                    history,
                    failure,
                    f"All models failed task '{task.kind}' after {len(history)} attempts. "
                    f"Last error ({failure.value}): {last.error}",
                )

            previous_model = history[-1].model if history else None
            if previous_model is not None and previous_model != model_id:
                self.on_event(
                    "model_switch",
                    {
                        "task_id": task.task_id,
                        "role": task.role,
                        "from": previous_model,
                        "to": model_id,
                        "reason": history[-1].failure_class.value
                        if history[-1].failure_class
                        else "unknown",
                    },
                )
            elif history:
                self.on_event(
                    "model_retry",
                    {
                        "task_id": task.task_id,
                        "role": task.role,
                        "model": model_id,
                        "reason": history[-1].failure_class.value
                        if history[-1].failure_class
                        else "unknown",
                    },
                )

            await self._make_resident(model_id)
            prompt = self._prompt_for(task, model_id, history)
            attempt = ModelAttempt(
                attempt_id=uuid.uuid4().hex,
                task_id=task.task_id,
                round=round_no,
                model=model_id,
                model_attempt=sum(1 for item in history if item.model == model_id) + 1,
                overall_attempt=len(history) + 1,
                switched_from=previous_model if previous_model != model_id else None,
                prompt_chars=len(prompt),
            )
            log.record(attempt)
            if task.on_prompt is not None:
                task.on_prompt(attempt, prompt)
            if task.on_attempt_start is not None:
                task.on_attempt_start(attempt)

            started = time.perf_counter()
            session_id = f"{task.task_id}_r{round_no}_a{attempt.overall_attempt}"
            try:
                text, value = await self._single_attempt(task, attempt, prompt, session_id)
            except Exception as exc:
                failure = classify_failure(exc)
                attempt.status = "failed"
                attempt.failure_class = failure
                attempt.error = failure_message(exc)
                attempt.duration_ms = (time.perf_counter() - started) * 1000
                attempt.finished_at = utcnow()
                log.record(attempt)
                self.on_event(
                    "attempt_failed",
                    {
                        "task_id": task.task_id,
                        "model": model_id,
                        "failure_class": failure.value,
                        "error": attempt.error[:300],
                    },
                )
                if task.observer is not None:
                    task.observer.failed(attempt)
                if failure in NON_RETRYABLE_FAILURES:
                    raise TaskExhausted(
                        task.task_id, round_attempts(), failure, attempt.error
                    ) from exc
                continue

            attempt.status = "succeeded"
            attempt.response_chars = len(text)
            attempt.duration_ms = (time.perf_counter() - started) * 1000
            attempt.finished_at = utcnow()
            log.record(attempt)
            self.on_event(
                "attempt_succeeded",
                {"task_id": task.task_id, "model": model_id},
            )
            completed = [
                item for item in log.load(task.task_id) if item.round == round_no
            ]
            return TaskOutcome(
                task_id=task.task_id,
                model=model_id,
                value=value,
                text=text,
                attempts=completed,
            )


def default_runtime_factory(policy: ModelPolicy, base_url: str) -> Callable[[ModelSpec], ModelRuntime]:
    """Build the real ADK + LiteLLM + Ollama runtime for a model."""
    from lab.runtime.adk_ollama import OllamaAdkRuntime
    from lab.runtime.config import RuntimeConfig

    def factory(spec: ModelSpec) -> ModelRuntime:
        import litellm

        # LiteLLM prints a support banner to the terminal on every error, which would
        # land in the middle of the conversation.  Failures are recorded anyway.
        litellm.suppress_debug_info = True
        return OllamaAdkRuntime(
            RuntimeConfig(
                model_id=spec.id,
                ollama_base_url=base_url,
                max_output_tokens=spec.max_output_tokens,
                num_ctx=spec.num_ctx,
                think=spec.think,
                keep_alive=policy.memory.keep_alive,
                request_timeout_seconds=policy.timeouts.inference_seconds,
                max_llm_calls=policy.agent.max_llm_calls_per_task,
            )
        )

    return factory


__all__ = [
    "AttemptLog",
    "DependencyFailure",
    "EmptyResponseError",
    "FilesystemFailure",
    "InMemoryAttemptLog",
    "InvalidJsonError",
    "ModelRouter",
    "ModelTask",
    "SemanticValidationError",
    "TaskExhausted",
    "TaskFailure",
    "TaskObserver",
    "TaskOutcome",
    "ToolExecutionFailure",
    "classify_failure",
    "default_runtime_factory",
    "failure_message",
    "parse_structured",
    "schema_instructions",
    "strip_thinking",
]

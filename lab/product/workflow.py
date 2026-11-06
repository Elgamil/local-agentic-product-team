"""The deterministic product workflow.

This class, not any model, owns the state of a product.

* Agents return validated artifacts.  They hold no reference to this object and
  cannot change state, approve anything, or choose a file location.
* State changes only through :meth:`ProductWorkflow._transition`, against
  ``ALLOWED_TRANSITIONS``.
* Leaving a human gate needs a :class:`HumanDecision`, which only the terminal
  session creates, from text the human typed.  An autonomous run leaves the step
  gates with a :class:`PolicyDecision`, which must cite an authorization the human
  gave in advance.  Every other step of the workflow is identical in both modes.
* Every automated phase handler is safe to re-enter.  A finished task is recognised
  by its result file and reused, so restarting after a crash repeats no completed
  work and resumes the interrupted task from saved inputs.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Protocol, TypeVar

from pydantic import BaseModel

from lab.config import ModelPolicy, load_verification_profiles
from lab.playbooks import PlaybookLibrary
from lab.product import intervention as reports
from lab.product import objectives, render
from lab.product.autonomy import (
    USAGE as AUTONOMY_USAGE,
    clarification,
    describe_mode,
    milestone_end,
    parse_autonomy_request,
    resolve_checkpoint,
)
from lab.product.context import ContextBuilder, ContextRequest, proposal_artifact
from lab.product.decisions import is_affirmation
from lab.product.manager import ProductManager, latest_roadmap, slugify
from lab.product.presentation import next_action, progress_label, role_label, status_label
from lab.product.progress import ModelProgress, format_duration, show, watching_test_process
from lab.product.store import DurableAttemptLog, ProductStore, new_op_id
from lab.roles import get_role_prompt
from lab.runtime.router import (
    ModelRouter,
    ModelTask,
    TaskExhausted,
    ToolExecutionFailure,
    failure_message,
    strip_thinking,
)
from lab.schemas.product import (
    ALLOWED_TRANSITIONS,
    AUTONOMOUS_GATES,
    HUMAN_GATES,
    MANUAL_AUTHOR,
    ActiveTask,
    AgentRole,
    ApprovalKind,
    ApprovalRecord,
    AutonomyCheckpoint,
    AutonomyPolicy,
    BlockedInfo,
    BlockKind,
    DeveloperPlan,
    DiscoveryTurn,
    ExecutableTestFile,
    ExecutionMode,
    ExecutionResult,
    FailureClass,
    FileReconciliation,
    HandoffRecord,
    HumanIntervention,
    ImplementationReview,
    ImplementationTask,
    InterventionBaseline,
    InterventionReport,
    ManualFileChange,
    ModelAttempt,
    ModificationRouting,
    NON_RETRYABLE_FAILURES,
    OperationRef,
    PlannedFileChange,
    ProductBrief,
    ProductPhase,
    Roadmap,
    RoadmapDraft,
    RoadmapStep,
    StepArchitecture,
    StepBrief,
    StepClosureReport,
    StepProgress,
    StepProposal,
    StepStatus,
    StepTestPlan,
    TaskResult,
    TeamAnswer,
    TestAssessment,
    TestFileChange,
    TestRevision,
    VerificationEvidence,
    VerificationRun,
    utcnow,
)
from lab.tools.developer_tools import build_developer_tools
from lab.tools.fsutil import (
    append_line,
    atomic_write_bytes,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from lab.tools.journal import (
    JournaledWorkspace,
    workspace_file_hashes,
    workspace_fingerprint,
)
from lab.tools.verification import (
    InvalidTestFile,
    VerificationStore,
    VerificationTamperedError,
    VerificationToolMissing,
    profile_available,
    run_profile,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_ALL_SPECIALISTS = ["software_architect", "developer", "test_engineer"]
_FAILURE_EXPLANATIONS = {
    FailureClass.MODEL_CONNECTION: (
        "The local models are not responding, so please check that Ollama is running. "
        "'python -m lab doctor' will confirm it."
    ),
    FailureClass.INFERENCE_TIMEOUT: (
        "The models took too long to answer. Closing other apps to free memory may help."
    ),
    FailureClass.INVALID_RESPONSE: "They did not give a usable answer.",
    FailureClass.INVALID_JSON: "Their answers were incomplete or badly formed.",
    FailureClass.SCHEMA_VALIDATION: "Their answers did not meet the required structure.",
    FailureClass.TOOL_EXECUTION: "They did not complete the required file work.",
}
_OPERATION_LABELS = {
    "discovery": "Product discussion",
    "product_brief": "Product summary",
    "roadmap": "Roadmap",
    "step_brief": "Step preparation",
    "architecture": "Technical design",
    "developer_plan": "Implementation plan",
    "test_plan": "Test plan",
    "step_proposal": "Step proposal",
    "test_assessment": "Test result review",
    "implementation_review": "Final review",
    "step_closure": "Step report",
    "route_modification": "Routing your change",
    "team_question": "Answering your question",
}
_GATE_APPROVALS = {
    ProductPhase.AWAITING_NAME: ApprovalKind.NAME,
    ProductPhase.AWAITING_ROADMAP_APPROVAL: ApprovalKind.ROADMAP,
    ProductPhase.AWAITING_STEP_START: ApprovalKind.STEP_START,
    ProductPhase.AWAITING_STEP_APPROVAL: ApprovalKind.STEP_PROPOSAL,
}


class Reporter(Protocol):
    heartbeat_seconds: float

    def say(self, text: str) -> None:
        """The Business Analyst speaking to the user."""

    def agent(self, role: str, text: str) -> None:
        """A named agent's output."""

    def status(self, text: str) -> None:
        """A short progress line."""

    def warn(self, text: str) -> None:
        """Something went wrong, or the user should notice it."""

    def error(self, text: str) -> None:
        """An actual failure."""

    def progress(self, text: str, context: str = "") -> None:
        """A heartbeat during a wait.  May arrive from a background thread."""

    def progress_done(self, text: str, context: str = "") -> None:
        """How a wait ended."""


class SilentReporter:
    heartbeat_seconds = 10.0

    def say(self, text: str) -> None:
        pass

    def agent(self, role: str, text: str) -> None:
        pass

    def status(self, text: str) -> None:
        pass

    def warn(self, text: str) -> None:
        pass

    def error(self, text: str) -> None:
        pass

    def progress(self, text: str, context: str = "") -> None:
        pass

    def progress_done(self, text: str, context: str = "") -> None:
        pass


def _require_text(text: str, request: str) -> None:
    """Refuse a blank message, so it can never reach a model."""
    if not text.strip():
        raise GateViolation(f"Please {request}.")


@dataclass(frozen=True)
class HumanDecision:
    """Text the human typed at a gate.  Only the terminal session constructs these."""

    text: str = ""
    decided_by: ClassVar[str] = "human"
    authorization_id: ClassVar[str | None] = None


@dataclass(frozen=True)
class PolicyDecision:
    """A gate passed under an authorization the human gave in advance.

    It is not a human decision and does not pretend to be.  It is recorded as made by
    the policy, and it names the human's authorization, which
    :meth:`ProductWorkflow._assert_step_approved` requires to exist.
    """

    authorization_id: str
    text: str = ""
    decided_by: ClassVar[str] = "autonomy_policy"


class VerifyStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_TEST_APPROVAL = "needs_test_approval"
    REFUSED = "refused"
    NOTHING_TO_VERIFY = "nothing_to_verify"


@dataclass(frozen=True)
class VerifyOutcome:
    """What /verify found.  It never consumes a model attempt."""

    status: VerifyStatus
    message: str


@dataclass(frozen=True)
class Gate:
    """A point where the workflow waits for the human."""

    key: str
    message: str
    question: str


class GateViolation(RuntimeError):
    """An action was attempted that the current phase does not permit."""


class WorkflowBlocked(Exception):
    """Raised internally to stop automated work once the product is blocked."""


class WorkflowPaused(Exception):
    """Raised between units of work when the human asked to pause.

    Handlers are re-entrant, so stopping at one of these points loses nothing.
    """


class ModelTranscript:
    """Readable log of model traffic.  Reasoning is stripped before it is written."""

    def __init__(self, store: ProductStore) -> None:
        self._store = store

    def _append(self, text: str) -> None:
        append_line(self._store.logs_dir / "model_io.md", text)

    def log_model_request(self, model: str, instruction: str, user_message: str) -> None:
        self._append(f"\n## Request to {model}\n\n{user_message}\n")

    def log_model_response(self, text: str, duration_ms: float, events: list[dict]) -> None:
        self._append(f"\n## Response ({duration_ms:.0f} ms)\n\n{strip_thinking(text)}\n")


class ProductWorkflow:
    def __init__(
        self,
        store: ProductStore,
        manager: ProductManager,
        router: ModelRouter,
        policy: ModelPolicy,
        playbooks: PlaybookLibrary,
        reporter: Reporter | None = None,
    ) -> None:
        self.store = store
        self.manager = manager
        self.router = router
        self.policy = policy
        self.playbooks = playbooks
        self.reporter: Reporter = reporter or SilentReporter()

        self.profiles = load_verification_profiles(store.verification_dir / "profiles.yaml")
        self.context = ContextBuilder(store, policy, playbooks, self.profiles)
        self.transcript = ModelTranscript(store)

        self.meta = store.load_metadata()
        self.state = store.load_state()
        self.roadmap: Roadmap | None = latest_roadmap(store, self.state)

        router.attempt_log = DurableAttemptLog(store)
        router.on_event = self._on_router_event
        # Called at each safe boundary, so input typed during a run is seen before the
        # next decision.  It never runs during a model call or a file write.
        self.on_boundary: Callable[[], None] | None = None

        self._automated: dict[ProductPhase, Callable[[], Any]] = {
            ProductPhase.DISCOVERY: self._handle_discovery,
            ProductPhase.BRIEF_DRAFTING: self._handle_brief,
            ProductPhase.ROADMAP_DRAFTING: self._handle_roadmap,
            ProductPhase.STEP_PREPARING: self._handle_step_preparing,
            ProductPhase.STEP_ARCHITECT: self._handle_step_architect,
            ProductPhase.STEP_DEVELOPER_PLAN: self._handle_step_developer_plan,
            ProductPhase.STEP_TEST_PLAN: self._handle_step_test_plan,
            ProductPhase.STEP_PROPOSAL: self._handle_step_proposal,
            ProductPhase.IMPLEMENTING: self._handle_implementing,
            ProductPhase.VERIFYING_TESTS: self._handle_verifying,
            ProductPhase.REVIEWING: self._handle_reviewing,
            ProductPhase.REPAIRING: self._handle_repairing,
            ProductPhase.STEP_CLOSING: self._handle_closing,
        }

    @property
    def verification(self) -> VerificationStore:
        # Derived on each use: the product folder moves when a draft gets its name.
        return VerificationStore(self.store.verification_dir)

    # ------------------------------------------------------------------
    # State handling
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self.state.next_expected_action = next_action(self.state, self.roadmap)
        self.store.save_state(self.state)

    def _event(self, event_type: str, **payload: Any) -> None:
        self.store.append_event(
            event_type,
            op_id=new_op_id(),
            phase=self.state.phase.value,
            step_id=self.state.current_step_id,
            agent=self.state.current_agent,
            model=self.state.active_model,
            payload=payload,
        )

    def _transition(self, target: ProductPhase) -> None:
        source = self.state.phase
        if target not in ALLOWED_TRANSITIONS[source]:
            raise GateViolation(f"Illegal transition {source.value} -> {target.value}")
        self.state.phase = target
        self.state.pending_approval = _GATE_APPROVALS.get(target)
        self.state.current_agent = None
        self._save()
        self._event("PhaseChanged", source=source.value, target=target.value)

    def _require_phase(self, *allowed: ProductPhase) -> None:
        if self.state.phase not in allowed:
            wanted = ", ".join(phase.value for phase in allowed)
            raise GateViolation(
                f"This action needs phase [{wanted}], but the product is in "
                f"{self.state.phase.value}."
            )

    def _record_operation(self, name: str, detail: str = "") -> None:
        self.state.last_successful_operation = OperationRef(
            op_id=new_op_id(), name=name, detail=detail
        )

    def _block(
        self,
        kind: BlockKind,
        message: str,
        *,
        failure_class: FailureClass | None = None,
        task_id: str | None = None,
        resume_phase: ProductPhase | None = None,
        task_kind: str | None = None,
        role: str | None = None,
        artifact: str | None = None,
        artifact_fields: list[str] | None = None,
        expected_files: list[str] | None = None,
    ) -> WorkflowBlocked:
        """Persist a BLOCKED state.  Returns the exception for the caller to raise."""
        active = self.state.active_task
        self.state.blocked = BlockedInfo(
            kind=kind,
            resume_phase=resume_phase or self.state.phase,
            message=message,
            failure_class=failure_class,
            task_id=task_id,
            task_kind=task_kind or (active.kind if active else None),
            role=role or (active.role if active else None),
            artifact=artifact,
            artifact_fields=artifact_fields or [],
            expected_files=expected_files or [],
        )
        self.state.last_failed_operation = OperationRef(
            op_id=new_op_id(), name=kind.value, detail=message[:300]
        )
        self.state.active_task = None
        self._transition(ProductPhase.BLOCKED)
        self._event("ProductBlocked", kind=kind.value, message=message[:500])
        self.ensure_intervention()
        return WorkflowBlocked(message)

    def _on_router_event(self, kind: str, payload: dict[str, Any]) -> None:
        names = {
            "attempt_failed": "ModelAttemptFailed",
            "attempt_succeeded": "ModelAttemptSucceeded",
            "model_retry": "ModelRetry",
            "model_switch": "ModelSwitched",
            "model_unloaded": "ModelUnloaded",
            "task_exhausted": "ModelTaskExhausted",
        }
        if kind in names:
            self._event(names[kind], **payload)
        if kind == "model_releasing":
            self.reporter.status(f"Releasing {payload.get('model')} from memory…")
        elif kind == "model_retry":
            label = role_label(payload.get("role"))
            if payload.get("reason") == FailureClass.MODEL_CONNECTION.value:
                self.reporter.warn("I could not reach the local model. Trying once more.")
            elif payload.get("reason") == FailureClass.INFERENCE_TIMEOUT.value:
                self.reporter.warn(f"{label} took too long. Trying once more.")
            else:
                self.reporter.warn(f"{label} gave an unusable answer. Trying once more.")
        elif kind == "model_switch":
            self.reporter.warn(
                f"{role_label(payload.get('role'))} is having trouble with this model, "
                "so I am switching to the backup model. The work continues from the last "
                "saved point."
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def _step(self) -> RoadmapStep:
        if self.roadmap is None or self.state.current_step_id is None:
            raise GateViolation("No roadmap step is active.")
        return self.roadmap.step(self.state.current_step_id)

    def _progress(self, step_id: str | None = None) -> StepProgress:
        return self.state.steps[step_id or self._step().step_id]

    def _artifact_path(self, name: str, version: int | None = None) -> str:
        step = self._step()
        return proposal_artifact(step.step_id, version or self._progress().proposal_version, name)

    def _request(self, *sections: str, **options: Any) -> ContextRequest:
        step_id = self.state.current_step_id
        version = self._progress().proposal_version if step_id else None
        return ContextRequest(
            sections=sections, step_id=step_id, proposal_version=version, **options
        )

    def _workspace(self, step_id: str) -> JournaledWorkspace:
        return JournaledWorkspace(
            self.store.workspace_dir,
            self.store.step_dir(step_id) / "file_ops.jsonl",
            protected_paths=[self.store.verification_dir],
            on_write=self._show_file_write,
        )

    def _show_file_write(self, path: str, size: int, unchanged: bool) -> None:
        """Called from the Developer's tool, which may run on a worker thread."""
        if unchanged:
            show(self.reporter.status, f"File unchanged: {path}.")
        else:
            show(self.reporter.status, f"File written: {path} ({size:,} bytes).")

    def _developer_model(self, step_id: str | None) -> str | None:
        """The model whose Developer output the Test Engineer should be independent of."""
        if step_id is None or step_id not in self.state.steps:
            return None
        progress = self.state.steps[step_id]
        if progress.developer_models:
            counts = Counter(progress.developer_models).most_common()
            top = counts[0][1]
            tied = {model for model, count in counts if count == top}
            for model in reversed(progress.developer_models):
                if model in tied:
                    return model
        return progress.developer_plan_model

    def _next_pending_step(self) -> RoadmapStep | None:
        if self.roadmap is None:
            return None
        for step in self.roadmap.steps:
            progress = self.state.steps.get(step.step_id)
            if progress is None or progress.status != StepStatus.COMPLETED:
                return step
        return None

    # ------------------------------------------------------------------
    # Agent tasks
    # ------------------------------------------------------------------

    def _begin_task(self, task_id: str, kind: str, role: str) -> None:
        self.state.active_task = ActiveTask(
            task_id=task_id, kind=kind, role=role, step_id=self.state.current_step_id
        )
        self.state.current_agent = role
        self.state.retry_count = 0
        self._save()
        self._event("TaskStarted", task_id=task_id, kind=kind)

    def _on_attempt_start(self, attempt: ModelAttempt) -> None:
        self.state.active_model = attempt.model
        self.state.retry_count = attempt.overall_attempt - 1
        if self.state.active_task is not None:
            self.state.active_task.model = attempt.model
        self._save()
        self._event(
            "ModelAttemptStarted",
            task_id=attempt.task_id,
            model=attempt.model,
            attempt=attempt.overall_attempt,
        )

    def _save_prompt(self, attempt: ModelAttempt, prompt: str) -> None:
        self.store.save_task_input(
            attempt.task_id, attempt.overall_attempt, attempt.round, prompt
        )

    def _save_failed_output(self, attempt: ModelAttempt, text: str) -> None:
        self.store.save_failed_output(
            attempt.task_id, attempt.overall_attempt, attempt.round, text
        )

    def _finish_task(
        self,
        result: TaskResult,
        receiver: str,
        request: ContextRequest,
        value: Any,
        attempts: list[ModelAttempt],
    ) -> None:
        """Idempotently close a task: handoff, then checkpoint."""
        self._record_handoff(result, receiver, request, value, attempts)
        self.state.active_task = None
        self._record_operation(result.kind, result.task_id)
        self._save()

    def _record_handoff(
        self,
        result: TaskResult,
        receiver: str,
        request: ContextRequest,
        value: Any,
        attempts: list[ModelAttempt],
    ) -> None:
        location = self.store.logs_dir / "handoffs.jsonl"
        existing, _ = read_jsonl(location)
        if any(record.get("task_id") == result.task_id for record in existing):
            return

        decisions: list[str] = []
        rationale: list[str] = []
        for item in getattr(value, "decisions", []) or []:
            decisions.append(item.decision)
            rationale.append(f"{item.decision}: {item.rationale}")

        summary = getattr(value, "handoff_summary", "") or getattr(value, "summary", "")
        tried = list(dict.fromkeys(attempt.model for attempt in attempts)) or [result.model]
        record = HandoffRecord(
            handoff_id=uuid.uuid4().hex,
            task_id=result.task_id,
            sender=result.role,
            receiver=receiver,
            model=result.model,
            step_id=result.step_id,
            task_input=result.objective,
            relevant_context=self._sections_for_handoff(request, result),
            response_artifact=result.artifact,
            response_excerpt=result.response_excerpt,
            decisions=decisions,
            rationale=rationale,
            handoff_summary=summary,
            validation_status="validated" if result.artifact else "accepted_text",
            attempts=result.attempts,
            models_tried=tried,
            failed_over=len(tried) > 1,
        )
        line = record.model_dump_json()
        append_line(location, line)
        markdown = render.handoff_markdown(record)
        if result.step_id:
            directory = self.store.step_dir(result.step_id)
            append_line(directory / "handoffs.jsonl", line)
            append_line(directory / "transcript.md", f"\n{markdown}\n")
        else:
            append_line(self.store.logs_dir / "transcript.md", f"\n{markdown}\n")
        self._event(
            "HandoffRecorded",
            task_id=result.task_id,
            sender=record.sender,
            receiver=receiver,
            model=result.model,
        )
        if summary:
            self.reporter.agent(result.role, summary)

    def _sections_for_handoff(
        self, request: ContextRequest, result: TaskResult
    ) -> list[str]:
        if result.model == MANUAL_AUTHOR:
            return []
        return self.context.sections_used(request, result.model)

    def _model_task(
        self,
        *,
        task_id: str,
        kind: str,
        role: str,
        objective: str,
        request: ContextRequest,
        schema: type[BaseModel] | None,
        validator: Callable[[Any], None] | None = None,
        tools: list[Any] | None = None,
        postcondition: Callable[[str, list[dict]], None] | None = None,
        operation: str | None = None,
    ) -> ModelTask:
        chain = self.policy.chain_for(role, self._developer_model(request.step_id))
        self.store.task_dir(task_id).mkdir(parents=True, exist_ok=True)

        def build_prompt(model_id: str) -> str:
            return self.context.task_prompt(objective, request, model_id)

        def build_failover_prompt(model_id: str, failures: list[ModelAttempt]) -> str:
            return self.context.failover_prompt(objective, request, model_id, failures)

        return ModelTask(
            task_id=task_id,
            role=role,
            kind=kind,
            chain=chain,
            instruction=get_role_prompt(f"team/{role}"),
            build_prompt=build_prompt,
            build_failover_prompt=build_failover_prompt,
            schema=schema,
            validator=validator,
            tools=tools,
            postcondition=postcondition,
            io_logger=self.transcript,
            on_attempt_start=self._on_attempt_start,
            on_prompt=self._save_prompt,
            on_failed_output=self._save_failed_output,
            observer=ModelProgress(
                self.reporter,
                role,
                operation or self._operation_label(kind),
                self.policy.retry.attempts_per_model * len(chain),
                self.policy.timeouts.inference_seconds,
            ),
        )

    def _operation_label(self, kind: str) -> str:
        label = _OPERATION_LABELS.get(kind, kind.replace("_", " ").capitalize())
        if self.state.current_step_id is None:
            return label
        return f"{label}, Step {self._step().number}"

    def _block_for_exhaustion(
        self,
        exc: TaskExhausted,
        kind: str,
        role: str,
        artifact: str | None = None,
        schema: type[BaseModel] | None = None,
        expected_files: list[str] | None = None,
    ) -> WorkflowBlocked:
        label = role_label(role)
        deliverable: dict[str, Any] = {
            "task_kind": kind,
            "role": role,
            "artifact": artifact,
            "artifact_fields": schema.model_json_schema().get("required", []) if schema else [],
            "expected_files": expected_files,
        }
        if exc.failure_class in NON_RETRYABLE_FAILURES:
            return self._block(
                BlockKind.DETERMINISTIC_FAILURE,
                f"{label} could not continue because of a {exc.failure_class.value} problem, "
                f"which another model cannot fix: {exc}",
                failure_class=exc.failure_class,
                task_id=exc.task_id,
                **deliverable,
            )
        models = ", ".join(dict.fromkeys(attempt.model for attempt in exc.attempts))
        return self._block(
            BlockKind.MODEL_FAILURE,
            f"{label} could not finish the {kind.replace('_', ' ')}. Both models failed after "
            f"{len(exc.attempts)} attempts ({models}). "
            f"{_FAILURE_EXPLANATIONS.get(exc.failure_class, '')}".rstrip(),
            failure_class=exc.failure_class,
            task_id=exc.task_id,
            **deliverable,
        )

    def _attempts_of(self, task_id: str) -> list[ModelAttempt]:
        return DurableAttemptLog(self.store).load(task_id)

    async def _agent_task(
        self,
        *,
        task_id: str,
        kind: str,
        role: str,
        receiver: str,
        objective: str,
        schema: type[T],
        artifact: str,
        request: ContextRequest,
        validator: Callable[[Any], None] | None = None,
        block_on_failure: bool = True,
        operation: str | None = None,
        manual_source: Callable[[], T] | None = None,
        expected_files: list[str] | None = None,
    ) -> T:
        """Run one structured task, or reuse its result if it already finished."""
        existing = self.store.task_result(task_id)
        if existing is not None and existing.artifact:
            value = self.store.load_model(existing.artifact, schema)
            self._finish_task(existing, receiver, request, value, self._attempts_of(task_id))
            return value

        if self.state.manual_artifact_task == task_id:
            return self._adopt_manual_result(
                task_id=task_id,
                kind=kind,
                role=role,
                receiver=receiver,
                objective=objective,
                schema=schema,
                artifact=artifact,
                request=request,
                validator=validator,
                manual_source=manual_source,
                expected_files=expected_files,
            )

        self._begin_task(task_id, kind, role)
        task = self._model_task(
            task_id=task_id,
            kind=kind,
            role=role,
            objective=objective,
            request=request,
            schema=schema,
            validator=validator,
            operation=operation,
        )
        try:
            outcome = await self.router.execute(task)
        except TaskExhausted as exc:
            if not block_on_failure:
                self.state.active_task = None
                self._save()
                raise
            raise self._block_for_exhaustion(
                exc, kind, role, artifact, schema, expected_files
            ) from exc

        digest = self.store.save_model(artifact, outcome.value)
        result = TaskResult(
            task_id=task_id,
            kind=kind,
            role=role,
            step_id=self.state.current_step_id,
            model=outcome.model,
            round=self.store.current_round(task_id),
            attempts=len(outcome.attempts),
            artifact=artifact,
            artifact_sha256=digest,
            objective=objective[:2000],
        )
        self.store.save_task_result(result)
        self._finish_task(result, receiver, request, outcome.value, outcome.attempts)
        return outcome.value  # type: ignore[return-value]

    def _adopt_manual_result(
        self,
        *,
        task_id: str,
        kind: str,
        role: str,
        receiver: str,
        objective: str,
        schema: type[T],
        artifact: str,
        request: ContextRequest,
        validator: Callable[[Any], None] | None,
        manual_source: Callable[[], T] | None,
        expected_files: list[str] | None,
    ) -> T:
        """Take a result a person supplied, held to the task's own rules.

        No model is called and no attempt is used.  The same schema and validator that
        judge a model's answer judge this one.
        """
        self.state.manual_artifact_task = None
        try:
            if manual_source is not None:
                value = manual_source()
            else:
                value = schema.model_validate_json(self.store.read_text(artifact))
            if validator is not None:
                validator(value)
        except (ValueError, OSError) as exc:
            raise self._block(
                BlockKind.MODEL_FAILURE,
                f"Your file for the {kind.replace('_', ' ')} is not valid yet: "
                f"{failure_message(exc)}",
                task_id=task_id,
                task_kind=kind,
                role=role,
                artifact=artifact,
                artifact_fields=schema.model_json_schema().get("required", []),
                expected_files=expected_files,
            ) from exc

        digest = self.store.save_model(artifact, value)
        result = TaskResult(
            task_id=task_id,
            kind=kind,
            role=role,
            step_id=self.state.current_step_id,
            model=MANUAL_AUTHOR,
            round=self.store.current_round(task_id),
            attempts=0,
            artifact=artifact,
            artifact_sha256=digest,
            objective=objective[:2000],
        )
        self.store.save_task_result(result)
        self._finish_task(result, receiver, request, value, self._attempts_of(task_id))
        self._event("ManualResultAccepted", task_id=task_id, artifact=artifact)
        return value

    async def _tool_task(
        self,
        *,
        task_id: str,
        kind: str,
        objective: str,
        request: ContextRequest,
        workspace: JournaledWorkspace,
        receiver: str,
        postcondition: Callable[[str, list[dict]], None],
        operation: str | None = None,
    ) -> TaskResult:
        """Run one Developer task.  Success is what the journal shows, not what it says."""
        role = AgentRole.DEVELOPER.value
        existing = self.store.task_result(task_id)
        if existing is not None:
            self._finish_task(existing, receiver, request, None, self._attempts_of(task_id))
            return existing

        self._begin_task(task_id, kind, role)
        workspace.current_task = task_id
        task = self._model_task(
            task_id=task_id,
            kind=kind,
            role=role,
            objective=objective,
            request=request,
            schema=None,
            tools=build_developer_tools(workspace),
            postcondition=postcondition,
            operation=operation,
        )
        try:
            outcome = await self.router.execute(task)
        except TaskExhausted as exc:
            raise self._block_for_exhaustion(exc, kind, role) from exc

        result = TaskResult(
            task_id=task_id,
            kind=kind,
            role=role,
            step_id=self.state.current_step_id,
            model=outcome.model,
            round=self.store.current_round(task_id),
            attempts=len(outcome.attempts),
            objective=objective[:2000],
            response_excerpt=outcome.text[:600],
        )
        self.store.save_task_result(result)
        self._finish_task(result, receiver, request, None, outcome.attempts)
        return result

    def _record_decision_once(self, marker: str, text: str) -> None:
        tag = f"<!-- {marker} -->"
        if tag not in self.store.read_text("decisions.md"):
            self.store.append_decision(f"{tag}\n{text}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _at_boundary(self) -> None:
        if self.on_boundary is not None:
            self.on_boundary()

    def _pause_point(self) -> None:
        """Honour a pause between units of work inside a long handler."""
        self._at_boundary()
        if self.state.paused:
            raise WorkflowPaused

    async def advance(self) -> None:
        """Run automated phases until a human gate or a terminal state.

        Under an autonomous policy the step gates a human authorized are passed and
        recorded here.  Nothing else about the run differs from Manual Mode.
        """
        while True:
            self._at_boundary()
            if self.state.paused:
                return
            phase = self.state.phase
            if phase in AUTONOMOUS_GATES and self._pass_gate_by_policy():
                continue
            handler = self._automated.get(phase)
            if handler is None:
                return
            if phase == ProductPhase.DISCOVERY and not self.state.pending_input:
                return
            try:
                await handler()
            except (WorkflowBlocked, WorkflowPaused):
                return

    def resume(self, *, hold: bool = False) -> list[str]:
        """Inspect what an earlier session left behind and describe it.

        With *hold*, a pause stays in force until :meth:`release_pause`, so a person's
        pause cannot be lost if the session ends before they confirm.
        """
        notes: list[str] = []
        if not hold:
            self.state.paused = False

        task = self.state.active_task
        if task is not None:
            self._event("TaskInterrupted", task_id=task.task_id, kind=task.kind)
            if self.state.phase in HUMAN_GATES:
                self.state.active_task = None
            else:
                notes.append(
                    f"The {role_label(task.role)} was interrupted while working on "
                    f"{task.kind.replace('_', ' ')}. I will continue that same task from its "
                    "saved inputs."
                )

        if self.state.phase in (ProductPhase.IMPLEMENTING, ProductPhase.REPAIRING):
            report = self._reconcile(self._step().step_id)
            if report.recovered:
                notes.append(
                    f"{len(report.recovered)} file write(s) had landed just before the "
                    "interruption, and I confirmed them from their hashes."
                )
            if report.drifted:
                notes.append(
                    "These files changed outside the platform and I left them as they are: "
                    f"{', '.join(report.drifted)}."
                )

        self._save()
        self._event("ProductResumed", revision=self.state.revision)
        return notes

    def pause(self, decision: HumanDecision) -> None:
        self.state.paused = True
        self._save()
        self._event("ProductPaused", note=decision.text[:200])

    def release_pause(self, decision: HumanDecision) -> None:
        self.state.paused = False
        self._save()
        self._event("ProductUnpaused", note=decision.text[:200])

    def stop(self, decision: HumanDecision) -> None:
        """End the run and withdraw any authorization, so nothing continues by itself."""
        if self.autonomy_active:
            self._withdraw_autonomy(decision, "stopped by the user")
        self.pause(decision)

    # ------------------------------------------------------------------
    # Execution mode
    # ------------------------------------------------------------------

    @property
    def autonomy_active(self) -> bool:
        return self.state.autonomy.mode != ExecutionMode.MANUAL

    def execution_mode_label(self) -> str:
        return describe_mode(self.state.autonomy, self.roadmap)

    def change_autonomy(self, decision: HumanDecision) -> str:
        """Apply what the human typed after /autonomy, and say what it means.

        The change is saved at once and takes effect at the next gate: nothing running
        is interrupted.  Raises :class:`GateViolation` with a question or an explanation
        when the request cannot be honoured, leaving the mode as it was.
        """
        try:
            request = parse_autonomy_request(decision.text)
        except ValueError as exc:
            raise GateViolation(str(exc)) from exc
        if request.action == "status":
            return f"Execution mode: {self.execution_mode_label()}.\n{AUTONOMY_USAGE}"
        if request.action == "off":
            return self._autonomy_off(decision)

        self._require_authorizable_roadmap()
        if request.action == "on":
            self._authorize(AutonomyPolicy(mode=ExecutionMode.AUTONOMOUS_PRODUCT), decision)
        else:
            assert self.roadmap is not None
            resolution = resolve_checkpoint(request.target, self.roadmap, self.state)
            if resolution.step is None:
                raise GateViolation(clarification(request.target, resolution, self.roadmap))
            self._authorize(
                AutonomyPolicy(
                    mode=ExecutionMode.AUTONOMOUS_CHECKPOINT,
                    checkpoint=AutonomyCheckpoint(
                        step_id=resolution.step.step_id,
                        requested=request.target,
                        milestone=resolution.milestone,
                    ),
                ),
                decision,
            )
        return self._autonomy_confirmation()

    def _autonomy_confirmation(self) -> str:
        message = f"Execution mode: {self.execution_mode_label()}."
        if self.state.phase == ProductPhase.AWAITING_ROADMAP_APPROVAL:
            return (
                f"{message} This does not approve the roadmap. Autonomous execution starts "
                "once you approve it."
            )
        if (
            self.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
            and not self._progress().human_approval_required
        ):
            message += " That covers the proposal in front of you."
        return (
            f"{message} The workflow itself is unchanged: every agent still works, and tests "
            "and reviews still decide whether a step is finished. If anything blocks, I stop."
        )

    def _require_authorizable_roadmap(self) -> None:
        if self.roadmap is None:
            raise GateViolation(
                "There is no roadmap to authorize yet. You can choose an execution mode once "
                "you have seen it."
            )
        if self._next_pending_step() is None:
            raise GateViolation("Every roadmap step is already complete.")

    def _authorize(self, policy: AutonomyPolicy, decision: HumanDecision) -> None:
        """Record the human's authorization.  Every policy decision will cite this record."""
        policy.authorization_id = uuid.uuid4().hex
        policy.authorized_at = utcnow()
        self.state.autonomy = policy
        label = self.execution_mode_label()
        self.store.append_approval(
            ApprovalRecord(
                approval_id=policy.authorization_id,
                kind=ApprovalKind.AUTONOMY,
                decision="approved",
                subject=label,
                note=decision.text[:500],
            )
        )
        self.store.append_decision(f"Execution mode changed by the user: {label}.")
        self._save()
        self._event(
            "AutonomyChanged",
            mode=policy.mode.value,
            checkpoint=policy.checkpoint.model_dump() if policy.checkpoint else None,
            authorization_id=policy.authorization_id,
        )

    def _autonomy_off(self, decision: HumanDecision) -> str:
        if not self.autonomy_active:
            return "Execution mode is already Manual."
        self._withdraw_autonomy(decision, "returned to Manual Mode by the user")
        if self.state.phase in HUMAN_GATES or self.state.phase == ProductPhase.BLOCKED:
            return "Manual Mode. I will ask for your approval at every step."
        return (
            "Manual Mode. The work in progress finishes safely, and I will ask for your "
            "approval at the next gate."
        )

    def _withdraw_autonomy(self, decision: HumanDecision, reason: str) -> None:
        withdrawn = self.state.autonomy
        label = self.execution_mode_label()
        self.store.append_approval(
            ApprovalRecord(
                approval_id=uuid.uuid4().hex,
                kind=ApprovalKind.AUTONOMY,
                decision="withdrawn",
                subject=label,
                note=decision.text[:500],
                authorization_id=withdrawn.authorization_id,
            )
        )
        self._end_autonomy(withdrawn, reason)

    def _end_autonomy(self, ended: AutonomyPolicy, reason: str) -> None:
        self.state.autonomy = AutonomyPolicy()
        self.store.append_decision(f"Autonomous execution ended: {reason}.")
        self._save()
        self._event(
            "AutonomyChanged",
            mode=ExecutionMode.MANUAL.value,
            reason=reason,
            authorization_id=ended.authorization_id,
        )

    def _checkpoint_reached(self) -> bool:
        """Spend the authorization once its checkpoint is complete."""
        policy = self.state.autonomy
        checkpoint = policy.checkpoint
        if policy.mode != ExecutionMode.AUTONOMOUS_CHECKPOINT or checkpoint is None:
            return False
        progress = self.state.steps.get(checkpoint.step_id)
        if progress is None or progress.status != StepStatus.COMPLETED:
            return False
        label = self.execution_mode_label().removeprefix("Autonomous, until ")
        self._event("AutonomyCheckpointReached", checkpoint=checkpoint.model_dump())
        self._end_autonomy(policy, f"checkpoint reached, {label}")
        self.reporter.say(
            f"Checkpoint reached: {label} is complete, so autonomous execution has stopped "
            "here and nothing further has started. Execution mode is now Manual. You can "
            "inspect the work, then approve the next step or use /autonomy to authorize more."
        )
        return True

    def _hold_for_person(self, step_number: int) -> None:
        self.reporter.say(
            f"Step {step_number} changed at your request, so it needs your own decision "
            "rather than the autonomous policy. Autonomous execution continues after that."
        )

    def _pass_gate_by_policy(self) -> bool:
        """Apply the human's prior authorization at a step gate.  True if it moved on.

        This is the only difference between the modes.  It calls the same methods a
        human's answer calls, so sealing, approval records and every check stay as they
        are.  Anything those refuse comes back to the human.
        """
        policy = self.state.autonomy
        if policy.mode == ExecutionMode.MANUAL or policy.authorization_id is None:
            return False
        decision = PolicyDecision(policy.authorization_id, self.execution_mode_label())
        try:
            if self.state.phase == ProductPhase.AWAITING_STEP_START:
                return self._start_step_by_policy(decision)
            return self._approve_proposal_by_policy(decision)
        except GateViolation as exc:
            self.reporter.warn(f"Autonomous execution stopped at this gate: {exc}")
            return False

    def _start_step_by_policy(self, decision: PolicyDecision) -> bool:
        if self._checkpoint_reached():
            return False
        following = self._next_pending_step()
        if following is None:
            return False
        if self.state.steps[following.step_id].human_approval_required:
            self._hold_for_person(following.number)
            return False
        self.start_step(decision)
        self.reporter.status(
            f"Execution mode: {self.execution_mode_label()}. Starting Step {following.number}: "
            f"{following.name}."
        )
        return True

    def _approve_proposal_by_policy(self, decision: PolicyDecision) -> bool:
        step = self._step()
        if self._progress().human_approval_required:
            self._hold_for_person(step.number)
            return False
        self.approve_step_proposal(decision)
        self.reporter.status(
            f"Execution mode: {self.execution_mode_label()}. The proposal for Step "
            f"{step.number} is kept in steps/{step.step_id}/proposal.md and proceeds under "
            "your authorization."
        )
        return True

    def _revalidate_checkpoint(self) -> None:
        """Keep a checkpoint honest after the roadmap changes."""
        policy = self.state.autonomy
        checkpoint = policy.checkpoint
        if checkpoint is None or self.roadmap is None:
            return
        target = None
        if checkpoint.milestone:
            target = milestone_end(self.roadmap, checkpoint.milestone)
        elif any(step.step_id == checkpoint.step_id for step in self.roadmap.steps):
            return
        if target is not None:
            checkpoint.step_id = target.step_id
            return
        self._end_autonomy(policy, f"the checkpoint '{checkpoint.requested}' left the roadmap")
        self.reporter.warn(
            f"The checkpoint '{checkpoint.requested}' is no longer on the roadmap, so I am back "
            "in Manual Mode. Use /autonomy until <step> to choose a new one."
        )

    def _reconcile(self, step_id: str) -> FileReconciliation:
        report = self._workspace(step_id).reconcile()
        if report.recovered or report.drifted or report.not_applied:
            self._event("FilesReconciled", **report.model_dump())
        if report.drifted:
            self._record_decision_once(
                f"drift-{step_id}-{'-'.join(report.drifted)}",
                f"Files changed outside the platform during {step_id} were kept as found: "
                f"{', '.join(report.drifted)}.",
            )
        return report

    # ------------------------------------------------------------------
    # Discovery, brief, naming
    # ------------------------------------------------------------------

    async def discovery_message(self, text: str) -> None:
        """Take the user's next discovery message."""
        self._require_phase(ProductPhase.DISCOVERY)
        _require_text(text, "type a response")
        self.store.append_conversation("user", text)
        self.state.pending_input = text
        self.state.discovery_round += 1
        self._save()
        await self.advance()

    def finish_discovery(self, decision: HumanDecision) -> None:
        self._require_phase(ProductPhase.DISCOVERY)
        if self.state.discovery_round < 1:
            raise GateViolation("Tell me about the product first.")
        self._transition(ProductPhase.BRIEF_DRAFTING)

    def _check_category(self, category: str) -> None:
        if self.playbooks.get(category) is None:
            raise ValueError(
                f"Unknown category {category!r}. Use one of: {', '.join(self.playbooks.ids())}"
            )

    def _apply_category(self, category: str) -> None:
        playbook = self.playbooks.get(category)
        if playbook is None or self.meta.category == category:
            return
        self.meta.category = category
        self.meta.category_label = playbook.label
        self.store.save_metadata(self.meta)
        self._event("PlaybookLoaded", category=category)

    async def _handle_discovery(self) -> None:
        round_no = self.state.discovery_round
        must_conclude = round_no >= self.policy.workflow.discovery_max_rounds

        if self.meta.category is None and self.state.pending_input:
            suggestion = self.playbooks.suggest(self.state.pending_input)
            if suggestion:
                self._apply_category(suggestion)

        turn = await self._agent_task(
            task_id=f"discovery-{round_no}",
            kind="discovery",
            role=AgentRole.BUSINESS_ANALYST.value,
            receiver="human",
            objective=objectives.discovery(self.playbooks.catalog(), must_conclude=must_conclude),
            schema=DiscoveryTurn,
            artifact=f"artifacts/discovery/turn-{round_no}.json",
            request=ContextRequest(sections=("conversation", "playbook:considerations")),
            validator=lambda value: self._check_category(value.category),
        )
        self._apply_category(turn.category)
        self.meta.description = turn.understanding[:300]
        self.store.save_metadata(self.meta)

        conversation = self.store.read_conversation()
        if not conversation or conversation[-1].get("text") != turn.reply:
            self.store.append_conversation("business_analyst", turn.reply)
            self.reporter.say(turn.reply)

        # One checkpoint for both changes, so a crash cannot lose the turn's verdict.
        self.state.pending_input = None
        if turn.ready_for_brief or must_conclude:
            self._transition(ProductPhase.BRIEF_DRAFTING)
        else:
            self._save()

    async def _handle_brief(self) -> None:
        self.reporter.status("Preparing the product summary…")
        brief = await self._agent_task(
            task_id="brief-1",
            kind="product_brief",
            role=AgentRole.BUSINESS_ANALYST.value,
            receiver="human",
            objective=objectives.brief(self.playbooks.catalog()),
            schema=ProductBrief,
            artifact="artifacts/brief.v1.json",
            request=ContextRequest(sections=("conversation", "playbook:considerations")),
            validator=lambda value: self._check_category(value.category),
        )
        self._apply_category(brief.category)
        self.store.write_text("brief.md", render.brief_markdown(brief))
        if not self.meta.name_confirmed and self.meta.display_name != brief.product_name:
            self.manager.rename(self.store, brief.product_name)
            self.meta = self.store.load_metadata()
        self.meta.description = brief.one_line_summary
        self.store.save_metadata(self.meta)
        self._transition(ProductPhase.AWAITING_NAME)

    def confirm_name(self, decision: HumanDecision) -> None:
        self._require_phase(ProductPhase.AWAITING_NAME)
        self._finalize_name("accepted")

    def change_name(self, decision: HumanDecision) -> str:
        """Rename to the human's chosen name and return the resulting folder slug."""
        self._require_phase(ProductPhase.AWAITING_NAME)
        name = decision.text.strip()
        if not name:
            raise GateViolation("A product name cannot be empty.")
        self.manager.rename(self.store, name)
        self.meta = self.store.load_metadata()
        brief = self.store.load_model("artifacts/brief.v1.json", ProductBrief)
        brief.product_name = name
        self.store.save_model("artifacts/brief.json", brief)
        self.store.write_text("brief.md", render.brief_markdown(brief))
        self._finalize_name("changed")
        return self.store.slug

    def _finalize_name(self, how: str) -> None:
        self.meta.name_confirmed = True
        self.store.save_metadata(self.meta)
        self.store.append_approval(
            ApprovalRecord(
                approval_id=uuid.uuid4().hex,
                kind=ApprovalKind.NAME,
                decision="approved",
                subject=self.meta.display_name,
            )
        )
        self.store.append_decision(
            f"Product name {how}: {self.meta.display_name} (folder {self.store.slug})."
        )
        self._transition(ProductPhase.ROADMAP_DRAFTING)

    # ------------------------------------------------------------------
    # Roadmap
    # ------------------------------------------------------------------

    def _locked_steps(self) -> list[RoadmapStep]:
        if self.roadmap is None:
            return []
        locked = []
        for step in self.roadmap.steps:
            progress = self.state.steps.get(step.step_id)
            if progress is not None and progress.status == StepStatus.COMPLETED:
                locked.append(step)
        return locked

    async def _handle_roadmap(self) -> None:
        self.reporter.status("Preparing the roadmap…")
        version = self.state.roadmap_version + 1
        locked = self._locked_steps()
        first_number = len(locked) + 1
        change_request = self.state.roadmap_change_request

        def check(draft: RoadmapDraft) -> None:
            if draft.steps[0].number != first_number:
                raise ValueError(f"Number the steps starting at {first_number}.")

        extra: dict[str, str] = {}
        sections = ["brief", "playbook:development_phases"]
        if self.roadmap is not None:
            current = render.roadmap_for_people(self.roadmap, self.meta.display_name)
            extra["current_roadmap"] = current
            sections.append("current_roadmap")

        draft = await self._agent_task(
            task_id=f"roadmap-{version}",
            kind="roadmap",
            role=AgentRole.BUSINESS_ANALYST.value,
            receiver="human",
            objective=objectives.roadmap(first_number, change_request=change_request),
            schema=RoadmapDraft,
            artifact=f"artifacts/roadmap_draft.v{version}.json",
            request=ContextRequest(sections=tuple(sections), extra=extra),
            validator=check,
        )
        # Version and phase land in a single checkpoint.  If they were saved apart, a
        # crash between them would make the resumed handler draft yet another version.
        self._apply_roadmap(self._assemble_roadmap(draft, version, locked, change_request))
        self.state.roadmap_change_request = None
        self._transition(ProductPhase.AWAITING_ROADMAP_APPROVAL)
        self._event("RoadmapVersionSaved", version=version)

    def _assemble_roadmap(
        self,
        draft: RoadmapDraft,
        version: int,
        locked: list[RoadmapStep],
        change_request: str | None,
    ) -> Roadmap:
        """Give steps their identity.  Finished steps stay; ids are never reused."""
        previous = self.roadmap
        sequence = previous.next_step_seq if previous else 1
        reusable = {
            step.step_id: step
            for step in (previous.steps if previous else [])
            if step not in locked
        }
        by_name = {step.name.casefold(): step for step in reusable.values()}

        steps = [
            step.model_copy(update={"number": index})
            for index, step in enumerate(locked, start=1)
        ]
        for item in draft.steps:
            match = by_name.pop(item.name.casefold(), None)
            if match is not None:
                step_id = match.step_id
            else:
                step_id = f"step-{sequence:02d}"
                sequence += 1
            steps.append(RoadmapStep(step_id=step_id, **item.model_dump()))

        for orphan in by_name.values():
            progress = self.state.steps.get(orphan.step_id)
            if progress is not None:
                progress.status = StepStatus.REMOVED
        return Roadmap(
            version=version,
            summary=draft.summary,
            steps=steps,
            change_request=change_request,
            next_step_seq=sequence,
        )

    def _apply_roadmap(self, roadmap: Roadmap) -> None:
        """Adopt a new version.  Earlier versions stay on disk as audit history."""
        self.store.save_model(f"artifacts/roadmap.v{roadmap.version}.json", roadmap)
        for step in roadmap.steps:
            self.state.steps.setdefault(step.step_id, StepProgress(step_id=step.step_id))
            if self.state.steps[step.step_id].status == StepStatus.REMOVED:
                self.state.steps[step.step_id].status = StepStatus.PENDING
        self.roadmap = roadmap
        self.state.roadmap_version = roadmap.version
        self._write_roadmap_doc()
        self._revalidate_checkpoint()

    def _write_roadmap_doc(self) -> None:
        assert self.roadmap is not None
        self.store.write_text(
            "roadmap.md",
            render.roadmap_markdown(
                self.roadmap,
                self.meta.display_name,
                approved_version=self.state.approved_roadmap_version,
                progress=self.state.steps,
            ),
        )

    def approve_roadmap(self, decision: HumanDecision) -> None:
        self._require_phase(ProductPhase.AWAITING_ROADMAP_APPROVAL)
        assert self.roadmap is not None
        version = self.roadmap.version
        self.store.append_approval(
            ApprovalRecord(
                approval_id=uuid.uuid4().hex,
                kind=ApprovalKind.ROADMAP,
                decision="approved",
                subject=f"roadmap v{version}",
                artifact_sha256=self.store.sha256_of(f"artifacts/roadmap.v{version}.json"),
                note=decision.text[:300],
            )
        )
        self.store.append_decision(f"Roadmap version {version} approved as the baseline.")
        self.state.approved_roadmap_version = version
        self._write_roadmap_doc()
        self._transition(ProductPhase.AWAITING_STEP_START)

    def request_roadmap_change(self, decision: HumanDecision) -> None:
        self._require_phase(
            ProductPhase.AWAITING_ROADMAP_APPROVAL,
            ProductPhase.AWAITING_STEP_START,
            ProductPhase.PRODUCT_COMPLETE,
        )
        _require_text(decision.text, "tell me what should change")
        self.store.append_approval(
            ApprovalRecord(
                approval_id=uuid.uuid4().hex,
                kind=ApprovalKind.ROADMAP,
                decision="changes_requested",
                subject=f"roadmap v{self.state.roadmap_version}",
                note=decision.text[:500],
            )
        )
        self.store.append_decision(f"Roadmap change requested: {decision.text}")
        self.state.roadmap_change_request = decision.text
        self._transition(ProductPhase.ROADMAP_DRAFTING)

    def _edited_roadmap(self, steps: list[RoadmapStep], note: str) -> None:
        assert self.roadmap is not None
        renumbering = {step.number: index for index, step in enumerate(steps, start=1)}
        updated = []
        for step in steps:
            number = renumbering[step.number]
            dependencies = sorted(
                renumbering[dep] for dep in step.dependencies if dep in renumbering
            )
            if any(dep >= number for dep in dependencies):
                raise GateViolation(
                    f"'{step.name}' depends on a step that would come after it."
                )
            updated.append(
                step.model_copy(update={"number": number, "dependencies": dependencies})
            )
        roadmap = Roadmap(
            version=self.roadmap.version + 1,
            summary=self.roadmap.summary,
            steps=updated,
            change_request=note,
            next_step_seq=self.roadmap.next_step_seq,
        )
        self.store.append_decision(f"Roadmap edited by the user: {note}.")
        self._apply_roadmap(roadmap)
        self._save()
        self._event("RoadmapVersionSaved", version=roadmap.version)

    def remove_roadmap_step(self, number: int, decision: HumanDecision) -> None:
        """Deterministic edit: needs no model, and the result still needs approval."""
        self._require_phase(ProductPhase.AWAITING_ROADMAP_APPROVAL)
        assert self.roadmap is not None
        target = next((s for s in self.roadmap.steps if s.number == number), None)
        if target is None:
            raise GateViolation(f"There is no step {number}.")
        if target in self._locked_steps():
            raise GateViolation("A completed step cannot be removed.")
        if len(self.roadmap.steps) == 1:
            raise GateViolation("A roadmap needs at least one step.")
        remaining = [step for step in self.roadmap.steps if step.number != number]
        self.state.steps[target.step_id].status = StepStatus.REMOVED
        self._edited_roadmap(remaining, f"removed step {number}, {target.name}")

    def move_roadmap_step(self, number: int, position: int, decision: HumanDecision) -> None:
        self._require_phase(ProductPhase.AWAITING_ROADMAP_APPROVAL)
        assert self.roadmap is not None
        steps = list(self.roadmap.steps)
        locked = len(self._locked_steps())
        if not (locked < number <= len(steps) and locked < position <= len(steps)):
            raise GateViolation("Steps can only move within the unfinished part of the roadmap.")
        moved = steps.pop(number - 1)
        steps.insert(position - 1, moved)
        self._edited_roadmap(steps, f"moved step {number} to position {position}")

    # ------------------------------------------------------------------
    # Step planning (phases A to C)
    # ------------------------------------------------------------------

    def start_step(self, decision: HumanDecision | PolicyDecision) -> None:
        self._require_phase(ProductPhase.AWAITING_STEP_START)
        if self.state.approved_roadmap_version != self.state.roadmap_version:
            raise GateViolation("The current roadmap has not been approved.")
        step = self._next_pending_step()
        if step is None:
            raise GateViolation("Every roadmap step is already complete.")
        for dependency in step.dependencies:
            assert self.roadmap is not None
            needed = self.roadmap.steps[dependency - 1]
            if self.state.steps[needed.step_id].status != StepStatus.COMPLETED:
                raise GateViolation(f"Step {step.number} needs step {dependency} first.")

        progress = self.state.steps[step.step_id]
        was_rejected = progress.status == StepStatus.REJECTED
        progress.proposal_version += 1
        progress.status = StepStatus.PLANNING
        progress.rerun_agents = list(_ALL_SPECIALISTS)
        progress.approved_version = None
        progress.approved_proposal_sha256 = None
        progress.human_approval_required = was_rejected
        self.state.current_step_id = step.step_id
        self._record_approval(
            ApprovalKind.STEP_START, "approved", f"start {step.step_id}", decision
        )
        self._transition(ProductPhase.STEP_PREPARING)

    def _record_approval(
        self,
        kind: ApprovalKind,
        outcome: Any,
        subject: str,
        decision: HumanDecision | PolicyDecision,
        digest: str | None = None,
    ) -> None:
        self.store.append_approval(
            ApprovalRecord(
                approval_id=uuid.uuid4().hex,
                kind=kind,
                decision=outcome,
                subject=subject,
                artifact_sha256=digest,
                note=decision.text[:500],
                decided_by=decision.decided_by,  # type: ignore[arg-type]
                authorization_id=decision.authorization_id,
            )
        )

    async def _handle_step_preparing(self) -> None:
        step = self._step()
        progress = self._progress()
        self.reporter.status(f"Step {step.number}: preparing the work for the team…")
        brief = await self._agent_task(
            task_id=f"{step.step_id}-brief-v{progress.proposal_version}",
            kind="step_brief",
            role=AgentRole.BUSINESS_ANALYST.value,
            receiver=AgentRole.SOFTWARE_ARCHITECT.value,
            objective=objectives.step_brief(step),
            schema=StepBrief,
            artifact=self._artifact_path("step_brief"),
            request=self._request(
                "brief", "roadmap", "step", "earlier_steps", "workspace_tree", "modifications"
            ),
        )
        self.store.write_text(
            f"steps/{step.step_id}/brief.md", render.step_brief_markdown(brief)
        )
        self._transition(ProductPhase.STEP_ARCHITECT)

    async def _handle_step_architect(self) -> None:
        step = self._step()
        progress = self._progress()
        if AgentRole.SOFTWARE_ARCHITECT.value in progress.rerun_agents:
            self.reporter.status("Consulting the Software Architect…")
            design = await self._agent_task(
                task_id=f"{step.step_id}-architect-v{progress.proposal_version}",
                kind="architecture",
                role=AgentRole.SOFTWARE_ARCHITECT.value,
                receiver=AgentRole.DEVELOPER.value,
                objective=objectives.architect(step),
                schema=StepArchitecture,
                artifact=self._artifact_path("architect"),
                request=self._request(
                    "brief",
                    "roadmap",
                    "step",
                    "step_brief",
                    "modifications",
                    "earlier_steps",
                    "playbook:quality_requirements",
                    "workspace_tree",
                    "relevant_files",
                    "decisions",
                    query=f"{step.name} {step.objective}",
                ),
            )
            self.store.write_text(
                f"steps/{step.step_id}/architect.md", render.architecture_markdown(design)
            )
            self._record_decision_once(
                f"{step.step_id}-architect-v{progress.proposal_version}",
                f"Proposed design for Step {step.number} ({step.name}), revision "
                f"{progress.proposal_version}:\n"
                + "\n".join(
                    f"- {item.decision} Reason: {item.rationale}" for item in design.decisions
                ),
            )
        self._transition(ProductPhase.STEP_DEVELOPER_PLAN)

    async def _handle_step_developer_plan(self) -> None:
        step = self._step()
        progress = self._progress()
        if AgentRole.DEVELOPER.value in progress.rerun_agents:
            self.reporter.status("Asking the Developer for an implementation plan…")
            task_id = f"{step.step_id}-developer-plan-v{progress.proposal_version}"
            plan = await self._agent_task(
                task_id=task_id,
                kind="developer_plan",
                role=AgentRole.DEVELOPER.value,
                receiver=AgentRole.TEST_ENGINEER.value,
                objective=objectives.developer_plan(step),
                schema=DeveloperPlan,
                artifact=self._artifact_path("developer"),
                request=self._request(
                    "step",
                    "step_brief",
                    "architecture",
                    "modifications",
                    "workspace_tree",
                    "relevant_files",
                    query=f"{step.name} {step.objective}",
                ),
            )
            result = self.store.task_result(task_id)
            written_by_model = result is not None and result.model != MANUAL_AUTHOR
            progress.developer_plan_model = result.model if written_by_model else None
            self._save()
            self.store.write_text(
                f"steps/{step.step_id}/developer.md", render.developer_plan_markdown(plan)
            )
        self._transition(ProductPhase.STEP_TEST_PLAN)

    def _check_test_plan(self, plan: StepTestPlan) -> None:
        profile = self.profiles.get(plan.verification_profile)
        available = [
            item.profile_id for item in self.profiles.values() if profile_available(item)
        ]
        if profile is None or not profile_available(profile):
            raise ValueError(
                f"Verification profile {plan.verification_profile!r} cannot be used here. "
                f"Choose one of: {', '.join(available)}"
            )
        for item in plan.test_files:
            try:
                self.verification.check_path(item.path, profile)
            except InvalidTestFile as exc:
                raise ValueError(str(exc)) from exc

    async def _handle_step_test_plan(self) -> None:
        step = self._step()
        progress = self._progress()
        version = progress.proposal_version
        if AgentRole.TEST_ENGINEER.value not in progress.rerun_agents:
            self._transition(ProductPhase.STEP_PROPOSAL)
            return

        self.reporter.status("Asking the Test Engineer for an independent test plan…")
        plan = await self._agent_task(
            task_id=f"{step.step_id}-test-plan-v{version}",
            kind="test_plan",
            role=AgentRole.TEST_ENGINEER.value,
            receiver=AgentRole.BUSINESS_ANALYST.value,
            objective=objectives.test_plan(step),
            schema=StepTestPlan,
            artifact=self._artifact_path("test_plan"),
            # The Developer's plan is left out on purpose: tests come from the
            # requirements and the public contracts, not from the intended build.
            request=self._request(
                "brief",
                "step",
                "step_brief",
                "interfaces",
                "modifications",
                "verification_profiles",
                "playbook:edge_cases",
                "playbook:acceptance_checks",
            ),
            validator=self._check_test_plan,
        )
        self.store.write_text(
            f"steps/{step.step_id}/test_plan.md", render.test_plan_markdown(plan)
        )

        profile = self.profiles[plan.verification_profile]
        self.reporter.status("The Test Engineer is writing the automated tests…")
        for index, planned in enumerate(plan.test_files, start=1):
            self._pause_point()
            expected_path = planned.path

            def check_file(value: ExecutableTestFile, expected: str = expected_path) -> None:
                if value.path != expected:
                    raise ValueError(f"Set 'path' to exactly {expected!r}.")

            def read_manual_file(expected: str = expected_path) -> ExecutableTestFile:
                source = self.verification.step_dir(step.step_id) / expected
                return ExecutableTestFile(
                    path=expected, content=source.read_text(encoding="utf-8")
                )

            test_file = await self._agent_task(
                task_id=f"{step.step_id}-test-file-v{version}-{index}",
                kind="test_file",
                role=AgentRole.TEST_ENGINEER.value,
                receiver=AgentRole.BUSINESS_ANALYST.value,
                objective=objectives.test_file(planned, profile),
                schema=ExecutableTestFile,
                artifact=self._artifact_path(f"test_file_{index}"),
                request=self._request("step", "test_plan", "interfaces"),
                validator=check_file,
                operation=f"Writing test file {planned.path} ({index} of {len(plan.test_files)})",
                manual_source=read_manual_file,
                expected_files=[planned.path],
            )
            self.verification.stage(step.step_id, test_file.path, test_file.content, profile)
        self._transition(ProductPhase.STEP_PROPOSAL)

    def _carry_forward(self, name: str, previous: int) -> None:
        source = self.store.root / self._artifact_path(name, previous)
        if source.is_file():
            target = self.store.root / self._artifact_path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    async def _handle_step_proposal(self) -> None:
        step = self._step()
        progress = self._progress()
        version = progress.proposal_version
        self.reporter.status("Consolidating the team's work into a proposal…")
        proposal = await self._agent_task(
            task_id=f"{step.step_id}-proposal-v{version}",
            kind="step_proposal",
            role=AgentRole.BUSINESS_ANALYST.value,
            receiver="human",
            objective=objectives.proposal(step),
            schema=StepProposal,
            artifact=self._artifact_path("proposal"),
            request=self._request(
                "step", "step_brief", "architecture", "developer_plan", "test_plan",
                "modifications",
            ),
        )
        text = self._proposal_text(proposal)
        self.store.write_text(f"steps/{step.step_id}/proposal.md", text)
        self.store.write_text(f"steps/{step.step_id}/proposals/v{version}/proposal.md", text)
        progress.status = StepStatus.AWAITING_APPROVAL
        self._transition(ProductPhase.AWAITING_STEP_APPROVAL)

    def _proposal_text(self, proposal: StepProposal) -> str:
        step = self._step()
        return render.proposal_for_people(
            step.number,
            step.name,
            proposal,
            self.store.load_model(self._artifact_path("developer"), DeveloperPlan),
            self.store.load_model(self._artifact_path("test_plan"), StepTestPlan),
            self._progress().proposal_version,
        )

    # ------------------------------------------------------------------
    # The step proposal gate
    # ------------------------------------------------------------------

    def approve_step_proposal(self, decision: HumanDecision | PolicyDecision) -> None:
        self._require_phase(ProductPhase.AWAITING_STEP_APPROVAL)
        step = self._step()
        progress = self._progress()
        version = progress.proposal_version
        test_plan = self.store.load_model(self._artifact_path("test_plan"), StepTestPlan)
        if self.verification.is_sealed(step.step_id):
            # A carried-forward suite must prove itself before it is sealed again,
            # otherwise re-sealing would launder any tampering.
            try:
                self.verification.verify_seal(step.step_id)
            except VerificationTamperedError as exc:
                raise GateViolation(str(exc)) from exc
        self.verification.seal(step.step_id, test_plan.verification_profile, version)
        self._keep_sealed_copy(step.step_id)

        digest = self.store.sha256_of(self._artifact_path("proposal"))
        progress.approved_version = version
        progress.approved_proposal_sha256 = digest
        progress.approved_documents = self._approved_documents(step.step_id, version)
        progress.sealed_manifest_sha256 = self.verification.manifest_digest(step.step_id)
        progress.human_approval_required = False
        progress.manual_verification = False
        progress.status = StepStatus.IMPLEMENTING
        progress.implementation_tasks = []
        progress.verification_cycle = 1
        progress.repair_cycles_used = 0
        progress.repair_cycle_limit = self.policy.workflow.max_repair_cycles
        progress.pending_repair = None
        progress.verification_passed = False
        progress.evidence_file = None
        self._record_approval(
            ApprovalKind.STEP_PROPOSAL,
            "approved",
            f"{step.step_id} proposal v{version}",
            decision,
            digest,
        )
        approved_by = (
            f" under your autonomous authorization ({decision.text})"
            if isinstance(decision, PolicyDecision)
            else ""
        )
        self.store.append_decision(
            f"Proposal revision {version} for Step {step.number} ({step.name}) "
            f"approved{approved_by}."
        )
        self._transition(ProductPhase.IMPLEMENTING)

    def reject_step_proposal(self, decision: HumanDecision) -> None:
        self._require_phase(ProductPhase.AWAITING_STEP_APPROVAL)
        step = self._step()
        progress = self._progress()
        version = progress.proposal_version
        self._archive_tests(step.step_id, version)
        progress.status = StepStatus.REJECTED
        if decision.text:
            progress.modification_requests.append(f"Earlier proposal rejected: {decision.text}")
        self._record_approval(
            ApprovalKind.STEP_PROPOSAL,
            "rejected",
            f"{step.step_id} proposal v{version}",
            decision,
        )
        self.store.append_decision(
            f"Proposal revision {version} for Step {step.number} rejected. "
            f"Reason: {decision.text or 'none given'}."
        )
        progress.human_approval_required = True
        self.state.current_step_id = None
        self._transition(ProductPhase.AWAITING_STEP_START)

    def _archive_tests(self, step_id: str, version: int) -> None:
        self.verification.archive_and_clear(
            step_id, self.store.step_dir(step_id) / "proposals" / f"v{version}" / "tests"
        )

    async def request_step_modification(self, decision: HumanDecision) -> None:
        """Send a change back to the relevant agents, keeping earlier proposals."""
        self._require_phase(ProductPhase.AWAITING_STEP_APPROVAL, ProductPhase.BLOCKED)
        if self.state.phase == ProductPhase.BLOCKED and self.state.current_step_id is None:
            raise GateViolation("There is no step proposal to change.")
        _require_text(decision.text, "tell me what should change")
        step = self._step()
        progress = self._progress()
        previous = progress.proposal_version

        try:
            routing = await self._agent_task(
                task_id=f"{step.step_id}-route-change-{len(progress.modification_requests) + 1}",
                kind="route_modification",
                role=AgentRole.BUSINESS_ANALYST.value,
                receiver=AgentRole.BUSINESS_ANALYST.value,
                objective=objectives.route_modification(decision.text),
                schema=ModificationRouting,
                artifact=(
                    f"steps/{step.step_id}/changes/"
                    f"{len(progress.modification_requests) + 1}.json"
                ),
                request=self._request("step", "architecture", "developer_plan", "test_plan"),
                block_on_failure=False,
            )
            request_text = routing.restated_request
            affected = set(routing.affected_agents)
        except TaskExhausted:
            # Routing only saves effort.  Without it, everyone reviews the change.
            request_text = decision.text
            affected = set(_ALL_SPECIALISTS)

        # Downstream work depends on upstream work.  The Test Engineer never reads the
        # Developer's plan, so a Developer-only change need not disturb the tests.
        if AgentRole.SOFTWARE_ARCHITECT.value in affected:
            affected = set(_ALL_SPECIALISTS)
        # After an integrity failure the existing tests cannot be trusted, and the same
        # goes for tests a person edited and has not yet approved.
        blocked = self.state.blocked
        intervention = self.state.intervention
        if (blocked is not None and blocked.kind == BlockKind.INTEGRITY) or (
            intervention is not None and intervention.pending_revision is not None
        ):
            affected = set(_ALL_SPECIALISTS)

        self._record_approval(
            ApprovalKind.STEP_PROPOSAL,
            "changes_requested",
            f"{step.step_id} proposal v{previous}",
            decision,
        )
        self.store.append_decision(
            f"Change requested for Step {step.number}: {request_text}"
        )

        if AgentRole.TEST_ENGINEER.value in affected:
            self._archive_tests(step.step_id, previous)

        progress.modification_requests.append(request_text)
        progress.proposal_version = previous + 1
        progress.rerun_agents = [agent for agent in _ALL_SPECIALISTS if agent in affected]
        progress.status = StepStatus.PLANNING
        progress.approved_version = None
        progress.approved_proposal_sha256 = None
        progress.implementation_tasks = []
        progress.verification_passed = False
        progress.manual_verification = False
        # A person changed direction, so this proposal is theirs to decide, even in an
        # autonomous run.  It may also change what the step's acceptance criteria say.
        progress.human_approval_required = True

        self._carry_forward("step_brief", previous)
        for agent, name in (
            (AgentRole.SOFTWARE_ARCHITECT.value, "architect"),
            (AgentRole.DEVELOPER.value, "developer"),
            (AgentRole.TEST_ENGINEER.value, "test_plan"),
        ):
            if agent not in progress.rerun_agents:
                self._carry_forward(name, previous)

        self._close_intervention("new proposal requested")
        self.state.blocked = None
        self._transition(ProductPhase.STEP_ARCHITECT)

    @staticmethod
    def route_question(text: str) -> str:
        """Deterministically pick who answers a human question."""
        lowered = text.lower()
        routes = (
            (AgentRole.TEST_ENGINEER.value, ("test", "verify", "verification", "qa", "edge case")),
            (AgentRole.DEVELOPER.value, ("developer", "implement", "code", "file")),
            (
                AgentRole.SOFTWARE_ARCHITECT.value,
                ("architect", "design", "technology", "framework", "database", "stack", "why"),
            ),
        )
        for role, words in routes:
            if any(word in lowered for word in words):
                return role
        return AgentRole.BUSINESS_ANALYST.value

    async def ask_team(self, decision: HumanDecision) -> str:
        """Answer a question at a gate.  The workflow does not move."""
        self._require_phase(
            ProductPhase.AWAITING_STEP_APPROVAL, ProductPhase.AWAITING_ROADMAP_APPROVAL
        )
        _require_text(decision.text, "tell me your question")
        question = decision.text
        if self.state.phase == ProductPhase.AWAITING_ROADMAP_APPROVAL:
            role = AgentRole.BUSINESS_ANALYST.value
            count = sum(1 for _ in self.store.read_conversation())
            task_id = f"roadmap-question-{count}"
            artifact = f"artifacts/questions/{task_id}.json"
            assert self.roadmap is not None
            request = ContextRequest(
                sections=("brief", "current_roadmap"),
                extra={
                    "current_roadmap": render.roadmap_for_people(
                        self.roadmap, self.meta.display_name
                    )
                },
            )
        else:
            step = self._step()
            progress = self._progress()
            progress.question_count += 1
            self._save()
            role = self.route_question(question)
            task_id = f"{step.step_id}-question-{progress.question_count}"
            artifact = f"steps/{step.step_id}/questions/{progress.question_count}.json"
            request = self._request(
                "step", "step_brief", "architecture", "developer_plan", "test_plan"
            )

        try:
            answer = await self._agent_task(
                task_id=task_id,
                kind="team_question",
                role=role,
                receiver="human",
                objective=objectives.team_question(question),
                schema=TeamAnswer,
                artifact=artifact,
                request=request,
                block_on_failure=False,
            )
        except TaskExhausted:
            return (
                "I could not get an answer from the team just now, because the models are "
                "not responding. Nothing has changed, and the proposal is still waiting "
                "for your decision."
            )
        if role == AgentRole.BUSINESS_ANALYST.value:
            return answer.answer
        return f"I checked with the {role_label(role)}: {answer.answer}"

    # ------------------------------------------------------------------
    # Implementation (phase D)
    # ------------------------------------------------------------------

    def _assert_step_approved(self, step: RoadmapStep, progress: StepProgress) -> None:
        """No implementation without proof of approval of this exact proposal."""
        version = progress.approved_version
        problem: str | None = None
        if version is None or version != progress.proposal_version:
            problem = "the current proposal has not been approved"
        else:
            digest = self.store.sha256_of(self._artifact_path("proposal", version))
            recorded = [
                record
                for record in self.store.read_approvals()
                if record.kind == ApprovalKind.STEP_PROPOSAL
                and record.decision == "approved"
                and record.subject == f"{step.step_id} proposal v{version}"
            ]
            if not recorded:
                problem = "no approval record exists for this proposal"
            elif digest != progress.approved_proposal_sha256 or digest != recorded[-1].artifact_sha256:
                problem = "the proposal changed after it was approved"
            elif not self._authorization_holds(recorded[-1]):
                problem = "the approval cites an authorization the user never gave"
        if problem:
            raise self._block(
                BlockKind.INTEGRITY,
                f"Implementation of Step {step.number} was stopped: {problem}.",
                resume_phase=ProductPhase.IMPLEMENTING,
            )

    def _authorization_holds(self, record: ApprovalRecord) -> bool:
        """A policy's approval is only as good as the human authorization it cites."""
        if record.decided_by == "human":
            return True
        return any(
            item.approval_id == record.authorization_id
            and item.kind == ApprovalKind.AUTONOMY
            and item.decided_by == "human"
            and item.decision == "approved"
            for item in self.store.read_approvals()
        )

    async def _handle_implementing(self) -> None:
        step = self._step()
        progress = self._progress()
        self._assert_step_approved(step, progress)
        version = progress.approved_version
        assert version is not None
        plan = self.store.load_model(self._artifact_path("developer", version), DeveloperPlan)

        if not progress.implementation_tasks:
            progress.implementation_tasks = [
                ImplementationTask(
                    path=item.path,
                    action=item.action,
                    purpose=item.purpose,
                    implementation_notes=item.implementation_notes,
                )
                for item in plan.file_changes
            ]
            self._save()

        workspace = self._workspace(step.step_id)
        self._reconcile(step.step_id)
        total = len(progress.implementation_tasks)

        for index, item in enumerate(progress.implementation_tasks, start=1):
            if item.status == "done":
                continue
            self._pause_point()
            # The hash keeps paths like 'a/b.py' and 'a-b.py' from sharing a task id.
            task_id = (
                f"{step.step_id}-impl-v{version}-{slugify(item.path)[:40]}"
                f"-{sha256_text(item.path)[:8]}"
            )

            landed = workspace.committed_for_task(task_id).get(item.path)
            if landed is None:
                self.reporter.status(f"Implementing {item.path} ({index} of {total})…")
                item.status = "running"
                self._save()
                expected = item.path

                def require_write(text: str, events: list[dict], target: str = expected) -> None:
                    if target not in workspace.committed_for_task(task_id):
                        raise ToolExecutionFailure(
                            f"The file {target!r} was not written. The file only exists "
                            "after a successful write_project_file call."
                        )

                change = PlannedFileChange(
                    path=item.path,
                    action=item.action,
                    purpose=item.purpose,
                    implementation_notes=item.implementation_notes,
                )
                result = await self._tool_task(
                    task_id=task_id,
                    kind="implement_file",
                    objective=objectives.implement_file(change),
                    request=self._request(
                        "step",
                        "interfaces",
                        "developer_plan",
                        "architecture",
                        "relevant_files",
                        "workspace_tree",
                        query=f"{item.path} {item.purpose}",
                        paths=(item.path,),
                    ),
                    workspace=workspace,
                    receiver=AgentRole.TEST_ENGINEER.value,
                    postcondition=require_write,
                    operation=f"Writing {item.path} ({index} of {total})",
                )
                landed = workspace.committed_for_task(task_id).get(item.path)
                item.model = result.model
            else:
                recorded = self.store.task_result(task_id)
                item.model = recorded.model if recorded else self.state.active_model

            item.status = "done"
            item.sha256 = landed.sha256_after if landed else None
            if item.model:
                progress.developer_models.append(item.model)
            self.state.active_task = None
            self._save()

        self._write_execution_record(step, progress, workspace)
        self.reporter.status(f"All {total} planned file(s) are written and recorded.")
        progress.status = StepStatus.VERIFYING
        self._transition(ProductPhase.VERIFYING_TESTS)

    def _write_execution_record(
        self, step: RoadmapStep, progress: StepProgress, workspace: JournaledWorkspace
    ) -> None:
        committed = workspace.latest_committed()
        planned = [item.path for item in progress.implementation_tasks]
        result = ExecutionResult(
            step_id=step.step_id,
            proposal_version=progress.approved_version or progress.proposal_version,
            planned_files=planned,
            written_files={
                path: operation.sha256_after or "" for path, operation in committed.items()
            },
            unplanned_files=sorted(set(committed) - set(planned)),
            developer_models=list(dict.fromkeys(progress.developer_models)),
        )
        self.store.save_model(f"steps/{step.step_id}/execution.json", result)
        self.store.write_text(
            f"steps/{step.step_id}/implementation.md", render.implementation_markdown(result)
        )

    # ------------------------------------------------------------------
    # Verification and repair (phase E)
    # ------------------------------------------------------------------

    def _run_suite(
        self, step_id: str, scope: str, cycle: int
    ) -> VerificationRun:
        manifest = self.verification.verify_seal(step_id)
        profile = self.profiles[manifest.profile_id]
        assert self.roadmap is not None
        number = self.roadmap.step(step_id).number
        title = f"{role_label(AgentRole.TEST_ENGINEER.value)} — {scope.title()} tests, Step {number}"
        limit = profile.timeout_seconds
        with watching_test_process(self.reporter, title, limit):
            raw = run_profile(
                profile, self.verification.step_dir(step_id), self.store.workspace_dir
            )
        taken = format_duration(raw.duration_ms / 1000)
        if raw.timed_out:
            self.reporter.warn(
                f"Timed out: the test process exceeded its configured limit of "
                f"{format_duration(limit)} and was stopped."
            )
        else:
            self.reporter.progress_done(
                f"Test process finished — {taken}: {raw.passed_count} passed, "
                f"{raw.failed_count} failed.",
                title,
            )
        return VerificationRun(
            command=raw.command,
            exit_code=raw.exit_code,
            passed_count=raw.passed_count,
            failed_count=raw.failed_count,
            duration_ms=raw.duration_ms,
            stdout=raw.stdout,
            stderr=raw.stderr,
            success=raw.success,
            step_id=step_id,
            scope=scope,  # type: ignore[arg-type]
            profile_id=profile.profile_id,
            cycle=cycle,
            cwd=raw.cwd,
            timed_out=raw.timed_out,
            manifest_sha256=self.verification.manifest_digest(step_id),
            stdout_truncated=raw.truncated,
        )

    def _run_verification(self, step: RoadmapStep, cycle: int) -> VerificationEvidence:
        """Execute the sealed suite and every earlier suite.  No model is involved."""
        runs = [self._run_suite(step.step_id, "acceptance", cycle)]
        assert self.roadmap is not None
        for earlier in self.roadmap.steps:
            if earlier.number >= step.number:
                break
            progress = self.state.steps.get(earlier.step_id)
            if progress and progress.status == StepStatus.COMPLETED:
                if self.verification.is_sealed(earlier.step_id):
                    runs.append(self._run_suite(earlier.step_id, "regression", cycle))

        passed = all(run.success for run in runs)
        failures = [
            f"{run.scope} suite {run.step_id}: exit code {run.exit_code}, "
            f"{run.passed_count} passed, {run.failed_count} failed"
            + (" (timed out)" if run.timed_out else "")
            for run in runs
            if not run.success
        ]
        return VerificationEvidence(
            step_id=step.step_id,
            cycle=cycle,
            proposal_version=self._progress().approved_version or 0,
            tested_workspace_fingerprint=workspace_fingerprint(self.store.workspace_dir),
            manifest_sha256=runs[0].manifest_sha256 or "",
            runs=runs,
            passed=passed,
            failure_summary="; ".join(failures),
        )

    def _cycle_tag(self) -> str:
        """Names one verification cycle of one approved proposal.

        Cycles restart when a modified proposal is approved, so the version must be
        part of the name, or a new approval would reuse the old proposal's results.
        """
        progress = self._progress()
        return f"v{progress.approved_version}-c{progress.verification_cycle}"

    def _evidence_path(self, step_id: str) -> str:
        return f"steps/{step_id}/test_runs/{self._cycle_tag()}.json"

    def _failure_details(self, evidence: VerificationEvidence) -> str:
        blocks = [evidence.failure_summary]
        for run in evidence.runs:
            if not run.success:
                output = f"{run.stdout}\n{run.stderr}".strip()
                blocks.append(f"Command: {run.command}\nOutput:\n{output[-5000:]}")
        return "\n\n".join(blocks)

    async def _handle_verifying(self) -> None:
        step = self._step()
        progress = self._progress()
        cycle = progress.verification_cycle
        tag = self._cycle_tag()
        relative = self._evidence_path(step.step_id)

        if self.store.exists(relative):
            evidence = self.store.load_model(relative, VerificationEvidence)
        else:
            self.reporter.status("Running the authoritative tests…")
            try:
                evidence = self._run_verification(step, cycle)
            except VerificationTamperedError as exc:
                raise self._block(
                    BlockKind.INTEGRITY,
                    f"Verification stopped: {exc}",
                    resume_phase=ProductPhase.VERIFYING_TESTS,
                ) from exc
            except VerificationToolMissing as exc:
                raise self._block(
                    BlockKind.DETERMINISTIC_FAILURE,
                    str(exc),
                    failure_class=FailureClass.DEPENDENCY,
                    resume_phase=ProductPhase.VERIFYING_TESTS,
                ) from exc
            self.store.save_model(relative, evidence)
            self._event(
                "VerificationRecorded",
                cycle=cycle,
                passed=evidence.passed,
                summary=evidence.failure_summary[:300],
            )

        self.store.write_text(
            f"steps/{step.step_id}/test_results.md", render.test_results_markdown(evidence)
        )
        self._record_operation("verification", f"cycle {cycle}: passed={evidence.passed}")
        self._save()

        if not evidence.passed:
            self.reporter.warn(f"Tests did not pass: {evidence.failure_summary}")
            self._enter_repair_or_close(self._failure_details(evidence))
            return

        self.reporter.status("Tests passed. The Test Engineer is reviewing the results…")
        assessment = await self._agent_task(
            task_id=f"{step.step_id}-assess-{tag}",
            kind="test_assessment",
            role=AgentRole.TEST_ENGINEER.value,
            receiver=AgentRole.SOFTWARE_ARCHITECT.value,
            objective=objectives.assess(step),
            schema=TestAssessment,
            artifact=f"steps/{step.step_id}/assessments/{tag}.json",
            request=self._evidence_request(
                ContextRequest(
                    sections=("step", "test_plan", "test_results"),
                    step_id=step.step_id,
                    proposal_version=progress.approved_version,
                ),
                evidence,
            ),
        )
        results = self.store.read_text(f"steps/{step.step_id}/test_results.md")
        self.store.write_text(
            f"steps/{step.step_id}/test_results.md",
            f"{results}\n{render.assessment_markdown(assessment)}\n",
        )
        if assessment.verdict == "fail":
            concerns = "; ".join(assessment.concerns) or assessment.summary
            self.reporter.warn(f"The Test Engineer raised concerns: {concerns}")
            self._enter_repair_or_close(f"Test Engineer assessment failed: {concerns}")
            return

        progress.evidence_file = relative
        self._save()
        self._transition(ProductPhase.REVIEWING)

    @staticmethod
    def _evidence_request(
        request: ContextRequest, evidence: VerificationEvidence
    ) -> ContextRequest:
        lines = [
            f"{run.scope} suite {run.step_id}: {run.passed_count} passed, "
            f"{run.failed_count} failed, exit code {run.exit_code}\n"
            f"{(run.stdout or run.stderr)[-2500:]}"
            for run in evidence.runs
        ]
        return ContextRequest(
            sections=request.sections,
            step_id=request.step_id,
            proposal_version=request.proposal_version,
            paths=request.paths,
            extra={**request.extra, "test_results": "\n\n".join(lines)},
        )

    def _enter_repair_or_close(self, details: str) -> None:
        progress = self._progress()
        progress.pending_repair = details
        progress.verification_passed = False
        if progress.manual_verification:
            # A model repair could overwrite what the person wrote, so this takes their
            # explicit /retry.  It never starts by itself.
            raise self._block(
                BlockKind.VERIFICATION_FAILED,
                f"Your fix passed the tests, but the review found a problem: {details[:600]}",
                resume_phase=ProductPhase.REPAIRING,
            )
        if progress.repair_cycles_used < progress.repair_cycle_limit:
            progress.repair_cycles_used += 1
            self._transition(ProductPhase.REPAIRING)
        else:
            self._transition(ProductPhase.STEP_CLOSING)

    async def _handle_reviewing(self) -> None:
        step = self._step()
        progress = self._progress()
        tag = self._cycle_tag()
        self.reporter.status("The Architect is reviewing the actual implementation…")
        evidence = self.store.load_model(self._evidence_path(step.step_id), VerificationEvidence)
        review = await self._agent_task(
            task_id=f"{step.step_id}-review-{tag}",
            kind="implementation_review",
            role=AgentRole.SOFTWARE_ARCHITECT.value,
            receiver=AgentRole.BUSINESS_ANALYST.value,
            objective=objectives.review(step),
            schema=ImplementationReview,
            artifact=f"steps/{step.step_id}/reviews/{tag}.json",
            request=self._evidence_request(
                ContextRequest(
                    sections=("step", "architecture", "implementation_files", "test_results"),
                    step_id=step.step_id,
                    proposal_version=progress.approved_version,
                    paths=tuple(item.path for item in progress.implementation_tasks),
                ),
                evidence,
            ),
        )
        self.store.write_text(f"steps/{step.step_id}/review.md", render.review_markdown(review))
        if review.verdict == "approved":
            progress.verification_passed = True
            self._save()
            self._transition(ProductPhase.STEP_CLOSING)
            return
        findings = "\n".join(
            f"- [{item.severity}] {item.file_path}: {item.description} "
            f"Recommendation: {item.recommendation}"
            for item in review.findings
            if item.severity in ("blocker", "major")
        )
        self.reporter.warn("The Architect requested changes.")
        self._enter_repair_or_close(f"The Architect's review requested changes:\n{findings}")

    async def _handle_repairing(self) -> None:
        step = self._step()
        progress = self._progress()
        cycle = progress.repair_cycles_used
        task_id = f"{step.step_id}-repair-v{progress.approved_version}-{cycle}"
        workspace = self._workspace(step.step_id)
        self._reconcile(step.step_id)
        self.reporter.status(f"The Developer is fixing the findings (repair {cycle})…")

        def require_change(text: str, events: list[dict]) -> None:
            changed = [
                operation
                for operation in workspace.committed_for_task(task_id).values()
                if not operation.unchanged
            ]
            if not changed:
                raise ToolExecutionFailure(
                    "No file was changed. A repair must change the product code through "
                    "write_project_file."
                )

        result = await self._tool_task(
            task_id=task_id,
            kind="repair",
            objective=objectives.repair(progress.pending_repair or "See the test results."),
            request=ContextRequest(
                sections=("step", "interfaces", "relevant_files", "workspace_tree"),
                step_id=step.step_id,
                proposal_version=progress.approved_version,
                query=progress.pending_repair or "",
                paths=tuple(item.path for item in progress.implementation_tasks),
            ),
            workspace=workspace,
            receiver=AgentRole.TEST_ENGINEER.value,
            postcondition=require_change,
            operation=f"Repair {cycle} of {progress.repair_cycle_limit}",
        )
        progress.developer_models.append(result.model)
        # /verify adds cycles of its own.  Reusing a number would reuse its old evidence.
        progress.verification_cycle = max(progress.verification_cycle, cycle) + 1
        progress.pending_repair = None
        self._write_execution_record(step, progress, workspace)
        self._transition(ProductPhase.VERIFYING_TESTS)

    # ------------------------------------------------------------------
    # Closing (phase F)
    # ------------------------------------------------------------------

    def completion_problems(self, step_id: str) -> list[str]:
        """Check the evidence a step's completion rests on.  Empty means proven."""
        progress = self._progress(step_id)
        problems: list[str] = []
        if not progress.verification_passed:
            problems.append("verification and review have not both passed")
        if not progress.evidence_file or not self.store.exists(progress.evidence_file):
            problems.append("no test evidence is recorded")
            return problems

        evidence = self.store.load_model(progress.evidence_file, VerificationEvidence)
        acceptance = [run for run in evidence.runs if run.scope == "acceptance"]
        if not evidence.passed or not acceptance:
            problems.append("the recorded tests did not pass")
        if any(run.passed_count < 1 for run in acceptance):
            problems.append("the recorded run contains no passing tests")
        if evidence.proposal_version != progress.approved_version:
            problems.append("the evidence belongs to a different proposal")
        if evidence.tested_workspace_fingerprint != workspace_fingerprint(
            self.store.workspace_dir
        ):
            problems.append("the code changed after it was tested")
        try:
            self.verification.verify_seal(step_id)
            if self.verification.manifest_digest(step_id) != evidence.manifest_sha256:
                problems.append("the tests differ from those that ran")
        except VerificationTamperedError as exc:
            problems.append(str(exc))
        return problems

    async def _handle_closing(self) -> None:
        step = self._step()
        progress = self._progress()
        tag = self._cycle_tag()
        relative = self._evidence_path(step.step_id)
        evidence = (
            self.store.load_model(relative, VerificationEvidence)
            if self.store.exists(relative)
            else None
        )
        problems = self.completion_problems(step.step_id)

        def check(report: StepClosureReport) -> None:
            if len(report.conditions) < len(step.completion_conditions):
                raise ValueError(
                    f"Judge all {len(step.completion_conditions)} completion conditions."
                )

        facts = (
            "The system's verification verdict: "
            + ("PASSED." if not problems else f"NOT PASSED ({'; '.join(problems)}).")
        )
        sections = ["step", "step_brief", "architecture", "review", "test_results", "verdict"]
        request = ContextRequest(
            sections=tuple(sections),
            step_id=step.step_id,
            proposal_version=progress.approved_version,
            extra={
                "verdict": facts,
                "review": self.store.read_text(f"steps/{step.step_id}/review.md"),
            },
        )
        if evidence is not None:
            request = self._evidence_request(request, evidence)
        else:
            request.extra["test_results"] = "No tests were recorded."

        self.reporter.status("Preparing the step report…")
        report = await self._agent_task(
            task_id=f"{step.step_id}-closure-{tag}",
            kind="step_closure",
            role=AgentRole.BUSINESS_ANALYST.value,
            receiver="human",
            objective=objectives.closure(step),
            schema=StepClosureReport,
            artifact=f"steps/{step.step_id}/closures/{tag}.json",
            request=request,
            validator=check,
        )
        verified = not problems
        text = render.closure_for_people(
            step.number, step.name, report, evidence, verified=verified
        )
        self.store.write_text(f"steps/{step.step_id}/closure.md", text)

        if not verified:
            progress.status = StepStatus.FAILED_VERIFICATION
            detail = "; ".join(problems)
            if progress.pending_repair:
                detail = f"{detail}. Repairs used: {progress.repair_cycles_used}"
            raise self._block(
                BlockKind.VERIFICATION_FAILED,
                f"Step {step.number} did not pass verification: {detail}.",
                resume_phase=ProductPhase.REPAIRING,
                role=AgentRole.DEVELOPER.value,
            )

        unmet = [item.condition for item in report.conditions if not item.met]
        if unmet and not progress.accepted_by_human:
            raise self._block(
                BlockKind.CONDITIONS_UNMET,
                f"Tests passed, but these completion conditions were not met: "
                f"{'; '.join(unmet)}.",
                resume_phase=ProductPhase.REPAIRING,
                role=AgentRole.BUSINESS_ANALYST.value,
            )
        self._complete_step(step, progress)

    def _complete_step(self, step: RoadmapStep, progress: StepProgress) -> None:
        progress.status = StepStatus.COMPLETED
        progress.completed_at = utcnow()
        self.store.append_decision(f"Step {step.number} ({step.name}) completed and verified.")
        self._event("StepCompleted", step_id=step.step_id)
        self._write_roadmap_doc()
        self.state.current_step_id = None
        if self._next_pending_step() is None:
            if self.autonomy_active:
                self._end_autonomy(self.state.autonomy, "every roadmap step is complete")
            self._transition(ProductPhase.PRODUCT_COMPLETE)
        else:
            self._transition(ProductPhase.AWAITING_STEP_START)

    # ------------------------------------------------------------------
    # Blocked recovery
    # ------------------------------------------------------------------

    def retry_blocked(self, decision: HumanDecision) -> None:
        self._require_phase(ProductPhase.BLOCKED)
        blocked = self.state.blocked
        assert blocked is not None
        if blocked.kind == BlockKind.INTEGRITY:
            raise GateViolation(
                "This needs a new proposal rather than a retry. Tell me what should change."
            )
        intervention = self.state.intervention
        if intervention is not None and intervention.pending_revision is not None:
            raise GateViolation("First approve or reject the test changes you made.")
        if intervention is not None:
            # Keep what the person changed before a model may touch the same files.
            self._record_manual_changes(intervention)
        if blocked.kind in (BlockKind.VERIFICATION_FAILED, BlockKind.CONDITIONS_UNMET):
            progress = self._progress()
            progress.manual_verification = False
            progress.repair_cycle_limit = (
                progress.repair_cycles_used + max(self.policy.workflow.max_repair_cycles, 1)
            )
            progress.repair_cycles_used += 1
            progress.status = StepStatus.VERIFYING
            progress.pending_repair = progress.pending_repair or blocked.message
        elif blocked.task_id:
            self.store.start_new_round(blocked.task_id)
        self._event("BlockedRetried", kind=blocked.kind.value, task_id=blocked.task_id)
        self._close_intervention("another AI attempt authorized")
        self.state.blocked = None
        self._transition(blocked.resume_phase)

    def accept_with_limitations(self, decision: HumanDecision) -> None:
        """The human accepts a step whose tests passed but whose conditions were unmet."""
        self._require_phase(ProductPhase.BLOCKED)
        blocked = self.state.blocked
        assert blocked is not None
        if blocked.kind != BlockKind.CONDITIONS_UNMET:
            raise GateViolation("Only a step with passing tests can be accepted this way.")
        step = self._step()
        progress = self._progress()
        problems = self.completion_problems(step.step_id)
        if problems:
            raise GateViolation(f"The evidence no longer holds: {'; '.join(problems)}.")
        progress.accepted_by_human = True
        self._record_approval(
            ApprovalKind.STEP_ACCEPTANCE,
            "accepted_with_limitations",
            step.step_id,
            decision,
        )
        self.store.append_decision(
            f"Step {step.number} accepted by the user with limitations: {blocked.message}"
        )
        self._close_intervention("accepted with limitations")
        self.state.blocked = None
        self._transition(ProductPhase.STEP_CLOSING)
        self._complete_step(step, progress)

    # ------------------------------------------------------------------
    # Human intervention
    # ------------------------------------------------------------------

    def _report_path(self, intervention_id: str) -> str:
        return f"interventions/{intervention_id}/report.md"

    def _baseline_path(self, intervention_id: str) -> str:
        return f"interventions/{intervention_id}/baseline.json"

    def ensure_intervention(self) -> HumanIntervention | None:
        """Open the intervention for the current block, if it has none yet.

        A failure here must never mask the block itself, so it is logged and the state
        stays BLOCKED.  The next look at the product tries again.
        """
        if self.state.blocked is None:
            return None
        if self.state.intervention is not None:
            return self.state.intervention
        try:
            return self._open_intervention()
        except Exception as exc:
            logger.exception("Could not prepare the intervention report")
            self.reporter.warn(f"I could not write the intervention report: {exc}")
            return None

    def _open_intervention(self) -> HumanIntervention:
        blocked = self.state.blocked
        assert blocked is not None
        number = self.state.intervention_count + 1
        intervention_id = f"INT-{number:03d}"
        intervention = HumanIntervention(
            intervention_id=intervention_id,
            step_id=self.state.current_step_id,
            task_id=blocked.task_id,
            report_path=self._report_path(intervention_id),
        )
        self._keep_missing_sealed_copies()
        self.store.save_model(
            self._baseline_path(intervention_id),
            InterventionBaseline(
                files=self._snapshot_files(), approved=self._snapshot_approved()
            ),
        )
        self._write_report(intervention)
        # One checkpoint: the count and the intervention always land together.
        self.state.intervention_count = number
        self.state.intervention = intervention
        self._save()
        self._event(
            "HumanInterventionRequired",
            intervention_id=intervention_id,
            kind=blocked.kind.value,
            task_id=blocked.task_id,
            report=intervention.report_path,
            mode=self.state.autonomy.mode.value,
        )
        return intervention

    def _close_intervention(self, outcome: str) -> None:
        intervention = self.state.intervention
        if intervention is None:
            return
        self.state.intervention = None
        self.store.append_decision(
            f"Human intervention {intervention.intervention_id} closed: {outcome}."
        )
        self._event(
            "HumanInterventionClosed",
            intervention_id=intervention.intervention_id,
            outcome=outcome,
        )

    def _snapshot_files(self) -> dict[str, str | None]:
        """Hash everything a person may edit, keyed relative to the product folder."""
        files: dict[str, str | None] = {
            f"workspace/{path}": digest
            for path, digest in workspace_file_hashes(self.store.workspace_dir).items()
        }
        verification = self.verification
        if self.store.verification_dir.is_dir():
            for directory in sorted(self.store.verification_dir.iterdir()):
                if directory.is_dir():
                    for path, digest in verification.file_hashes(directory.name).items():
                        files[f"verification/{directory.name}/{path}"] = digest
        return files

    def _approved_documents(self, step_id: str, version: int) -> dict[str, str]:
        """Hash what an approval covered: the agents' documents behind the proposal."""
        directory = self.store.step_dir(step_id) / "proposals" / f"v{version}"
        return {
            path.relative_to(self.store.root).as_posix(): sha256_file(path) or ""
            for path in sorted(directory.glob("*.json"))
        }

    def _snapshot_approved(self) -> dict[str, str | None]:
        approved: dict[str, str | None] = {}
        step_id = self.state.current_step_id
        if step_id is not None and step_id in self.state.steps:
            version = self._progress(step_id).approved_version
            if version is not None:
                approved.update(self._approved_documents(step_id, version))
        roadmap_version = self.state.approved_roadmap_version
        if roadmap_version is not None:
            relative = f"artifacts/roadmap.v{roadmap_version}.json"
            approved[relative] = self.store.sha256_of(relative)
        return approved

    def _load_baseline(self, intervention: HumanIntervention) -> InterventionBaseline:
        return self.store.load_model(
            self._baseline_path(intervention.intervention_id), InterventionBaseline
        )

    def _record_manual_changes(self, intervention: HumanIntervention) -> list[ManualFileChange]:
        """Detect and record what a person changed since the last checkpoint."""
        baseline = self._load_baseline(intervention)
        current = self._snapshot_files()
        changes = [
            ManualFileChange(
                path=path,
                status=(
                    "added"
                    if path not in baseline.files
                    else "removed"
                    if path not in current
                    else "modified"
                ),
                old_sha256=baseline.files.get(path),
                new_sha256=current.get(path),
            )
            for path in sorted(set(baseline.files) | set(current))
            if baseline.files.get(path) != current.get(path)
        ]
        if not changes:
            return []

        intervention.change_sets += 1
        number = intervention.change_sets
        directory = self.store.intervention_dir(intervention.intervention_id)
        self.store.write_text(
            f"interventions/{intervention.intervention_id}/manual-changes-{number}.json",
            "[\n" + ",\n".join(item.model_dump_json() for item in changes) + "\n]\n",
        )
        for item in changes:
            source = self.store.root / item.path
            if source.is_file():
                atomic_write_bytes(
                    directory / f"manual_changes/{number}" / item.path, source.read_bytes()
                )

        if intervention.step_id is not None:
            workspace = self._workspace(intervention.step_id)
            for item in changes:
                if item.path.startswith("workspace/"):
                    workspace.record_external_change(
                        item.path.removeprefix("workspace/"),
                        item.old_sha256,
                        item.new_sha256,
                        intervention.intervention_id,
                    )

        baseline.files = current
        self.store.save_model(self._baseline_path(intervention.intervention_id), baseline)
        self.store.append_decision(
            f"Manual changes recorded for {intervention.intervention_id}: "
            + ", ".join(f"{item.path} ({item.status})" for item in changes)
            + "."
        )
        self._save()
        self._event(
            "ManualChangesDetected",
            intervention_id=intervention.intervention_id,
            changes=[item.model_dump() for item in changes],
        )
        return changes

    def _protected_problems(self, intervention: HumanIntervention) -> list[str]:
        """Approved material cannot be edited into agreement with the code."""
        problems: list[str] = []
        baseline = self._load_baseline(intervention)
        step_id = intervention.step_id
        recorded = dict(baseline.approved)
        if step_id is not None and step_id in self.state.steps:
            recorded.update(self.state.steps[step_id].approved_documents)
        for path, digest in sorted(recorded.items()):
            if self.store.sha256_of(path) != digest:
                problems.append(path)

        for sealed, progress in self.state.steps.items():
            path = f"verification/{sealed}/manifest.json"
            expected = progress.sealed_manifest_sha256
            if expected is not None and self.store.sha256_of(path) != expected:
                problems.append(path)
        return problems

    def _sealed_copy_dir(self, step_id: str) -> Path:
        revision = self.verification.manifest(step_id).revision
        return self.store.step_dir(step_id) / "sealed_tests" / f"r{revision}"

    def _keep_sealed_copy(self, step_id: str) -> None:
        """Keep the sealed bytes, since a manifest holds only their hashes."""
        copy = self._sealed_copy_dir(step_id)
        shutil.rmtree(copy, ignore_errors=True)
        self.verification.copy_suite(step_id, copy)

    def _keep_missing_sealed_copies(self) -> None:
        """Products sealed before copies were kept get one while the seal still holds."""
        verification = self.verification
        for step_id in verification.sealed_steps():
            try:
                if not self._sealed_copy_dir(step_id).is_dir() and not (
                    verification.seal_differences(step_id)
                ):
                    self._keep_sealed_copy(step_id)
            except VerificationTamperedError:
                continue  # An unreadable seal cannot vouch for a copy.

    def _write_report(self, intervention: HumanIntervention, note: str = "") -> InterventionReport:
        report = reports.build_report(self._report_inputs(intervention, note))
        folder = f"interventions/{intervention.intervention_id}"
        text = reports.report_markdown(report)
        # The newest report is always report.md.  Earlier versions stay beside it.
        self.store.write_text(f"{folder}/report.v{report.version}.md", text)
        self.store.write_text(intervention.report_path, text)
        self.store.save_model(f"{folder}/report.json", report)
        self._event(
            "InterventionReportWritten",
            intervention_id=intervention.intervention_id,
            version=report.version,
        )
        return report

    def _report_inputs(self, intervention: HumanIntervention, note: str) -> reports.ReportInputs:
        blocked = self.state.blocked
        assert blocked is not None
        step = progress = None
        if intervention.step_id is not None and self.roadmap is not None:
            step = self.roadmap.step(intervention.step_id)
            progress = self.state.steps[intervention.step_id]

        attempts: list[ModelAttempt] = []
        preserved: list[str] = []
        if blocked.task_id:
            this_round = self.store.current_round(blocked.task_id)
            attempts = [
                item for item in self._attempts_of(blocked.task_id) if item.round == this_round
            ]
            preserved = self.store.failed_outputs(blocked.task_id)
        chain: list[str] = []
        allowed = 0
        if blocked.role:
            chain = self.policy.chain_for(blocked.role, self._developer_model(intervention.step_id))
            allowed = self.policy.retry.attempts_per_model * len(chain)

        evidence = self._latest_evidence(progress, intervention.step_id)
        verification = self.verification
        suites = {}
        for run in evidence.runs if evidence else []:
            suites[run.step_id] = verification.read_files(run.step_id)
        version = progress.proposal_version if progress else None
        return reports.ReportInputs(
            intervention_id=intervention.intervention_id,
            version=intervention.report_version,
            product=self.meta.display_name,
            slug=self.store.slug,
            blocked=blocked,
            execution_mode=self.execution_mode_label(),
            step=step,
            progress=progress,
            chain=chain,
            attempts=attempts,
            attempts_allowed=allowed if attempts else 0,
            preserved_outputs=preserved,
            evidence=evidence,
            suites=suites,
            profiles=self.profiles,
            test_plan=self._step_artifact(intervention.step_id, version, "test_plan", StepTestPlan),
            step_brief=self._step_artifact(intervention.step_id, version, "step_brief", StepBrief),
            agent_notes=self._agent_notes(progress, intervention.step_id),
            planned_files=[item.path for item in progress.implementation_tasks] if progress else [],
            manual_changes=self._all_manual_changes(intervention),
            changed_sealed_files=self._changed_sealed_files(),
            note=note,
        )

    def _changed_sealed_files(self) -> list[str]:
        """Sealed files that no longer match.  A report must work when the seal does not."""
        changed: list[str] = []
        verification = self.verification
        for step_id in verification.sealed_steps():
            try:
                differences = verification.seal_differences(step_id)
            except VerificationTamperedError:
                changed.append(f"verification/{step_id}/manifest.json")
                continue
            changed.extend(f"verification/{step_id}/{path}" for path in differences)
        return changed

    def _step_artifact(
        self, step_id: str | None, version: int | None, name: str, schema: type[T]
    ) -> T | None:
        if step_id is None or not version:
            return None
        return self._model_or_none(proposal_artifact(step_id, version, name), schema)

    def _latest_evidence(
        self, progress: StepProgress | None, step_id: str | None
    ) -> VerificationEvidence | None:
        if progress is None or step_id is None or progress.approved_version is None:
            return None
        relative = self._evidence_path(step_id)
        if not self.store.exists(relative):
            return None
        return self.store.load_model(relative, VerificationEvidence)

    def _agent_notes(self, progress: StepProgress | None, step_id: str | None) -> list[str]:
        """What the agents said about the failure, quoted and attributed."""
        if progress is None or step_id is None or progress.approved_version is None:
            return []
        tag = self._cycle_tag()
        notes: list[str] = []
        assessment = self._model_or_none(f"steps/{step_id}/assessments/{tag}.json", TestAssessment)
        if assessment is not None:
            label = role_label(AgentRole.TEST_ENGINEER.value)
            notes.append(f"{label}: {assessment.summary}")
            notes.extend(f"{label}: {concern}" for concern in assessment.concerns)
        review = self._model_or_none(f"steps/{step_id}/reviews/{tag}.json", ImplementationReview)
        if review is not None:
            label = role_label(AgentRole.SOFTWARE_ARCHITECT.value)
            notes.append(f"{label}: {review.summary}")
            notes.extend(
                f"{label}: [{item.severity}] {item.description}" for item in review.findings
            )
        cycle = progress.repair_cycles_used
        repair = self.store.task_result(f"{step_id}-repair-v{progress.approved_version}-{cycle}")
        if repair is not None and repair.response_excerpt:
            label = role_label(AgentRole.DEVELOPER.value)
            notes.append(f"{label}, on its last repair: {repair.response_excerpt}")
        return notes

    def _model_or_none(self, relative: str, schema: type[T]) -> T | None:
        if not self.store.exists(relative):
            return None
        try:
            return self.store.load_model(relative, schema)
        except ValueError:
            return None

    def _all_manual_changes(self, intervention: HumanIntervention) -> list[ManualFileChange]:
        changes: list[ManualFileChange] = []
        folder = self.store.intervention_dir(intervention.intervention_id)
        for path in sorted(folder.glob("manual-changes-*.json")):
            changes.extend(
                ManualFileChange.model_validate(item)
                for item in json.loads(path.read_text(encoding="utf-8"))
            )
        return changes

    def intervention_summary(self) -> str:
        """The short version of the current report, for the terminal."""
        intervention = self.state.intervention
        if intervention is None:
            return "Nothing is waiting for your intervention."
        report = self.store.load_model(
            f"interventions/{intervention.intervention_id}/report.json", InterventionReport
        )
        return reports.terminal_summary(
            report, f"products_log/{self.store.slug}/{intervention.report_path}"
        )

    def revision_summary(self) -> str:
        intervention = self.state.intervention
        revision = intervention.pending_revision if intervention else None
        if revision is None:
            return "No revised tests are waiting for your approval."
        changes = [
            f"{item.path} ({item.status}): {(item.old_sha256 or 'none')[:12]} -> "
            f"{(item.new_sha256 or 'none')[:12]}"
            for item in revision.changes
        ]
        return reports.revision_summary(
            revision.diff_excerpt,
            f"products_log/{self.store.slug}/{revision.diff_path}",
            revision.warnings,
            revision.acceptance_criteria,
            changes,
        )

    def _refresh_report(self, intervention: HumanIntervention, note: str) -> None:
        """Replace the diagnosis.  The block stays exactly as it was."""
        intervention.report_version += 1
        self._write_report(intervention, note)
        self._save()

    def verify_manual_fix(self, decision: HumanDecision) -> VerifyOutcome:
        """/verify: check a person's fix without using any model.

        This runs only what Python owns, the same sealed tests the workflow runs, so no
        attempt is consumed and no budget is reset.  A failed check returns to the
        person: it never starts another model repair by itself.
        """
        self._require_phase(ProductPhase.BLOCKED)
        intervention = self.ensure_intervention()
        if intervention is None:
            return VerifyOutcome(VerifyStatus.REFUSED, "I could not prepare this check.")
        if intervention.pending_revision is not None:
            return VerifyOutcome(
                VerifyStatus.NEEDS_TEST_APPROVAL,
                "Your revised tests are waiting for your decision.",
            )
        intervention.verify_requests += 1
        changes = self._record_manual_changes(intervention)
        self._event(
            "ManualVerificationRequested",
            intervention_id=intervention.intervention_id,
            changed_files=[item.path for item in changes],
            note=decision.text[:200],
        )
        if self._has_sealed_step(intervention):
            return self._verify_with_authoritative_tests(intervention, changes)
        return self._verify_planning_fix(intervention)

    def _has_sealed_step(self, intervention: HumanIntervention) -> bool:
        step_id = intervention.step_id
        if step_id is None or step_id not in self.state.steps:
            return False
        return self.state.steps[step_id].approved_version is not None

    def _verify_with_authoritative_tests(
        self, intervention: HumanIntervention, changes: list[ManualFileChange]
    ) -> VerifyOutcome:
        protected = self._protected_problems(intervention)
        if protected:
            message = (
                "These approved files changed, so I cannot verify against them: "
                f"{', '.join(protected)}. They define what the step must do. Restore them, or "
                "use /change to ask for a new proposal, which needs your approval."
            )
            self._refresh_report(intervention, message)
            self._event("ManualVerificationRefused", files=protected)
            return VerifyOutcome(VerifyStatus.REFUSED, message)

        try:
            revision = self._detect_test_revision(intervention)
        except VerificationTamperedError as exc:
            self._refresh_report(intervention, str(exc))
            return VerifyOutcome(VerifyStatus.REFUSED, str(exc))
        if revision is not None:
            intervention.pending_revision = revision
            self.state.pending_approval = ApprovalKind.TEST_REVISION
            self._save()
            self._event(
                "TestRevisionProposed",
                intervention_id=intervention.intervention_id,
                step_id=revision.step_id,
                changes=[item.model_dump() for item in revision.changes],
                warnings=revision.warnings,
            )
            return VerifyOutcome(
                VerifyStatus.NEEDS_TEST_APPROVAL, "Your test changes need your approval."
            )
        return self._run_manual_verification(intervention, changes)

    def _detect_test_revision(self, intervention: HumanIntervention) -> TestRevision | None:
        """Hold the first sealed suite a person changed until a person approves it."""
        verification = self.verification
        for step_id in verification.sealed_steps():
            differences = verification.seal_differences(step_id)
            if differences:
                return self._prepare_revision(intervention, step_id, differences)
        return None

    def _prepare_revision(
        self,
        intervention: HumanIntervention,
        step_id: str,
        differences: dict[str, tuple[str | None, str | None]],
    ) -> TestRevision:
        verification = self.verification
        manifest = verification.manifest(step_id)
        copy = self._sealed_copy_dir(step_id)
        current = verification.read_files(step_id)

        def sealed_text(path: str) -> str | None:
            source = copy / path
            return source.read_text(encoding="utf-8", errors="replace") if source.is_file() else None

        old_files: dict[str, str | None] = {path: sealed_text(path) for path in manifest.files}
        patches = []
        changes = []
        for path, (old_hash, new_hash) in differences.items():
            status = "added" if old_hash is None else "removed" if new_hash is None else "modified"
            changes.append(
                TestFileChange(
                    path=f"verification/{step_id}/{path}",
                    status=status,  # type: ignore[arg-type]
                    old_sha256=old_hash,
                    new_sha256=new_hash,
                )
            )
            patches.append(reports.unified_diff(path, old_files.get(path), current.get(path)))
        diff = "\n".join(patches)

        profile = self.profiles[manifest.profile_id]
        diff_path = (
            f"interventions/{intervention.intervention_id}/"
            f"test-revision-{step_id}-r{manifest.revision + 1}.diff"
        )
        self.store.write_text(diff_path, diff)
        version = self.state.steps[step_id].approved_version
        plan = self._step_artifact(step_id, version, "test_plan", StepTestPlan)
        excerpt = diff.splitlines()[: reports.DIFF_EXCERPT_LINES]
        return TestRevision(
            revision_id=uuid.uuid4().hex,
            step_id=step_id,
            changes=changes,
            warnings=reports.compare_suites(
                old_files, current, profile.result_parser, profile.test_file_patterns
            ),
            acceptance_criteria=reports._acceptance_lines(plan),
            diff_path=diff_path,
            diff_excerpt="\n".join(excerpt),
            old_manifest_sha256=verification.manifest_digest(step_id),
        )

    def approve_test_revision(self, decision: HumanDecision) -> VerifyOutcome:
        """A person accepts a revised suite as authoritative, then it runs."""
        self._require_phase(ProductPhase.BLOCKED)
        intervention = self.state.intervention
        revision = intervention.pending_revision if intervention else None
        if intervention is None or revision is None:
            raise GateViolation("No revised tests are waiting for your approval.")
        reason = decision.text.strip()
        if revision.warnings and (not reason or is_affirmation(reason)):
            raise GateViolation(
                "This edit may weaken the tests, so a plain yes is not enough. Say why it is "
                "still right with /approve <reason>, or use /reject to restore the sealed tests."
            )

        step_id = revision.step_id
        verification = self.verification
        manifest = verification.reseal(step_id)
        digest = verification.manifest_digest(step_id)
        progress = self.state.steps.get(step_id)
        if progress is not None:
            progress.sealed_manifest_sha256 = digest
        self._keep_sealed_copy(step_id)
        self._record_approval(
            ApprovalKind.TEST_REVISION,
            "approved",
            f"{step_id} test suite revision {manifest.revision}",
            decision,
            digest,
        )
        self.store.append_decision(
            f"Revised test suite for {step_id} approved by the user as revision "
            f"{manifest.revision}. Files: "
            + "; ".join(
                f"{item.path} {(item.old_sha256 or 'none')[:12]} -> "
                f"{(item.new_sha256 or 'none')[:12]}"
                for item in revision.changes
            )
            + (f". Warnings accepted: {' '.join(revision.warnings)}" if revision.warnings else "")
            + (f". Reason: {reason}" if reason else ".")
        )
        self._event(
            "TestRevisionApproved",
            step_id=step_id,
            revision=manifest.revision,
            old_manifest_sha256=revision.old_manifest_sha256,
            new_manifest_sha256=digest,
            changes=[item.model_dump() for item in revision.changes],
        )
        intervention.pending_revision = None
        self.state.pending_approval = None
        self._save()
        return self.verify_manual_fix(decision)

    def reject_test_revision(self, decision: HumanDecision) -> str:
        """Put the sealed tests back.  The person's version is kept beside the report."""
        self._require_phase(ProductPhase.BLOCKED)
        intervention = self.state.intervention
        revision = intervention.pending_revision if intervention else None
        if intervention is None or revision is None:
            raise GateViolation("No revised tests are waiting for your decision.")
        step_id = revision.step_id
        copy = self._sealed_copy_dir(step_id)
        if not copy.is_dir():
            raise GateViolation(
                f"I have no copy of the sealed tests for {step_id}. Please restore them by hand."
            )
        verification = self.verification
        verification.copy_suite(
            step_id,
            self.store.intervention_dir(intervention.intervention_id)
            / f"rejected_tests/{revision.revision_id}",
        )
        verification.restore_suite(step_id, copy)
        verification.verify_seal(step_id)
        self._record_approval(
            ApprovalKind.TEST_REVISION,
            "rejected",
            f"{step_id} test suite revision",
            decision,
        )
        self._event("TestRevisionRejected", step_id=step_id, revision_id=revision.revision_id)
        intervention.pending_revision = None
        self.state.pending_approval = None
        self._record_manual_changes(intervention)
        self._save()
        return (
            "The sealed tests are back as they were, and your version is kept in "
            f"products_log/{self.store.slug}/interventions/{intervention.intervention_id}"
            "/rejected_tests."
        )

    def _run_manual_verification(
        self, intervention: HumanIntervention, changes: list[ManualFileChange]
    ) -> VerifyOutcome:
        """Run the same suites the workflow runs.  The tests decide, not the mode."""
        step = self._step()
        progress = self._progress()
        progress.verification_cycle += 1
        self._save()
        cycle = progress.verification_cycle
        unchanged = "" if changes else " I found no file changes since the failure."
        self.reporter.status(f"Running the authoritative tests…{unchanged}")
        try:
            evidence = self._run_verification(step, cycle)
        except (VerificationTamperedError, VerificationToolMissing) as exc:
            self._refresh_report(intervention, str(exc))
            return VerifyOutcome(VerifyStatus.FAILED, str(exc))

        relative = self._evidence_path(step.step_id)
        self.store.save_model(relative, evidence)
        self.store.write_text(
            f"steps/{step.step_id}/test_results.md", render.test_results_markdown(evidence)
        )
        self._event(
            "VerificationRecorded",
            cycle=cycle,
            passed=evidence.passed,
            summary=evidence.failure_summary[:300],
            manual=True,
        )
        if not evidence.passed:
            message = f"Tests still fail: {evidence.failure_summary}"
            self._refresh_report(intervention, f"After your fix. {message}")
            self._event(
                "ManualVerificationFailed",
                intervention_id=intervention.intervention_id,
                cycle=cycle,
                summary=evidence.failure_summary[:300],
            )
            return VerifyOutcome(
                VerifyStatus.FAILED,
                f"{message}. Your changes are kept, and no model was started. Fix it and use "
                "/verify again, or use /retry to authorize another AI attempt.",
            )

        self._adopt_manual_implementation(step, progress)
        progress.status = StepStatus.VERIFYING
        progress.pending_repair = None
        progress.manual_verification = True
        self._record_operation("manual_verification", f"cycle {cycle}: passed=True")
        self._event(
            "ManualVerificationPassed", intervention_id=intervention.intervention_id, cycle=cycle
        )
        self._close_intervention("verified by the authoritative tests")
        self.state.blocked = None
        self._transition(ProductPhase.VERIFYING_TESTS)
        return VerifyOutcome(
            VerifyStatus.PASSED,
            "The authoritative tests pass with your fix. The Test Engineer and the Architect "
            "will now review the result.",
        )

    def _adopt_manual_implementation(self, step: RoadmapStep, progress: StepProgress) -> None:
        """Planned files a person wrote count as written."""
        for item in progress.implementation_tasks:
            if item.status == "done":
                continue
            digest = sha256_file(self.store.workspace_dir / item.path)
            if digest is not None:
                item.status = "done"
                item.model = MANUAL_AUTHOR
                item.sha256 = digest
        self._write_execution_record(step, progress, self._workspace(step.step_id))

    def _verify_planning_fix(self, intervention: HumanIntervention) -> VerifyOutcome:
        """Before the tests are sealed, a fix means supplying what the task could not."""
        blocked = self.state.blocked
        assert blocked is not None
        if not blocked.task_id or not (blocked.artifact or blocked.expected_files):
            return VerifyOutcome(
                VerifyStatus.NOTHING_TO_VERIFY,
                "There is nothing on disk I can check for this problem. Once it is fixed, use "
                "/retry to authorize another attempt.",
            )
        problems = self._manual_deliverable_problems(blocked)
        if problems:
            message = " ".join(problems)
            self._refresh_report(intervention, message)
            return VerifyOutcome(VerifyStatus.FAILED, f"{message} Nothing has moved on.")

        self.state.manual_artifact_task = blocked.task_id
        self._close_intervention("the person supplied the missing result")
        self.state.blocked = None
        self._transition(blocked.resume_phase)
        return VerifyOutcome(
            VerifyStatus.PASSED, "I have your file, and I will check it against the task's rules."
        )

    def _manual_deliverable_problems(self, blocked: BlockedInfo) -> list[str]:
        if not blocked.expected_files:
            if blocked.artifact and not self.store.exists(blocked.artifact):
                return [f"I cannot find {blocked.artifact}."]
            return []
        step_id = self.state.current_step_id
        assert step_id is not None
        plan = self._step_artifact(
            step_id, self._progress(step_id).proposal_version, "test_plan", StepTestPlan
        )
        profile = self.profiles.get(plan.verification_profile) if plan else None
        problems = []
        for path in blocked.expected_files:
            target = self.store.verification_dir / step_id / path
            shown = f"verification/{step_id}/{path}"
            if not target.is_file():
                problems.append(f"I cannot find {shown}.")
                continue
            source = target.read_text(encoding="utf-8", errors="replace")
            if profile is not None and not reports.defined_tests(source, profile.result_parser):
                problems.append(f"{shown} defines no test cases.")
        return problems

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------

    def gate(self) -> Gate | None:
        """Describe the current human gate from saved files, repeating no agent work."""
        phase = self.state.phase
        if phase == ProductPhase.AWAITING_NAME:
            return Gate(
                key=f"name:{self.store.slug}",
                message=(
                    f"I suggest the product name {self.meta.display_name}. I have prepared "
                    f"the product workspace products_log/{self.store.slug}."
                ),
                question="Would you prefer this name or would you like a different one?",
            )
        if phase == ProductPhase.AWAITING_ROADMAP_APPROVAL:
            assert self.roadmap is not None
            return Gate(
                key=f"roadmap:{self.roadmap.version}",
                message=render.roadmap_for_people(self.roadmap, self.meta.display_name),
                question=(
                    "Do you approve this roadmap? You can also ask for changes, for example "
                    "'remove step 3', 'move step 4 to 2', or describe what to add or change."
                    + (
                        f" Once approved, the team runs as: {self.execution_mode_label()}."
                        if self.autonomy_active
                        else ""
                    )
                ),
            )
        if phase == ProductPhase.AWAITING_STEP_START:
            return self._step_start_gate()
        if phase == ProductPhase.AWAITING_STEP_APPROVAL:
            step = self._step()
            progress = self._progress()
            proposal = self.store.load_model(self._artifact_path("proposal"), StepProposal)
            return Gate(
                key=f"proposal:{step.step_id}:{progress.proposal_version}",
                message=self._proposal_text(proposal),
                question=(
                    "Do you approve this proposal? You can also request a change, ask the "
                    "team a question, or reject it."
                ),
            )
        return None

    def _step_start_gate(self) -> Gate:
        assert self.roadmap is not None
        following = self._next_pending_step()
        assert following is not None
        completed = [
            step
            for step in self.roadmap.steps
            if self.state.steps[step.step_id].status == StepStatus.COMPLETED
        ]
        if completed:
            last = completed[-1]
            closure = self.store.read_text(f"steps/{last.step_id}/closure.md")
            return Gate(
                key=f"next:{following.step_id}",
                message=closure,
                question=(
                    f"Step {last.number} is completed. Shall we proceed to Step "
                    f"{following.number}: {following.name}?"
                ),
            )
        return Gate(
            key=f"start:{following.step_id}:{self.state.roadmap_version}",
            message="",
            question=(
                f"Shall we start Step {following.number}: {following.name}? I will bring you "
                "the team's proposal before anything is built."
            ),
        )

    def completion_message(self) -> str:
        assert self.roadmap is not None
        last = self.roadmap.steps[-1]
        closure = self.store.read_text(f"steps/{last.step_id}/closure.md")
        return (
            f"{closure}\n\nEvery roadmap step for {self.meta.display_name} is complete and "
            f"verified. The product is in products_log/{self.store.slug}/workspace."
        )

    def status_report(self) -> str:
        lines = [
            f"{self.meta.display_name} — {self.meta.category_label or 'Category to be decided'}",
            f"Status: {status_label(self.state, self.roadmap)}",
            f"Progress: {progress_label(self.state, self.roadmap)}",
            f"Execution mode: {self.execution_mode_label()}",
            f"Next action: {self.state.next_expected_action}",
            f"Folder: products_log/{self.store.slug}",
        ]
        if self.state.blocked:
            lines.append(f"Blocked: {self.state.blocked.message}")
        if self.state.intervention:
            lines.append(
                "Intervention report: "
                f"products_log/{self.store.slug}/{self.state.intervention.report_path}"
            )
        return "\n".join(lines)

    def team_activity(self, limit: int = 8) -> str:
        """Recent agent handoffs, for the user who wants to see the team's decisions."""
        records, _ = read_jsonl(self.store.logs_dir / "handoffs.jsonl")
        if not records:
            return "The team has not exchanged any work yet."
        lines = []
        for raw in records[-limit:]:
            record = HandoffRecord.model_validate(raw)
            summary = record.handoff_summary or record.response_excerpt or record.task_id
            lines.append(
                f"- {role_label(record.sender)} to {role_label(record.receiver)}: {summary}"
            )
            lines.extend(f"    Decision: {item}" for item in record.decisions)
        return "\n".join(lines)

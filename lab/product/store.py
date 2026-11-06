"""Durable per-product storage.  The filesystem is the source of truth.

Layout of ``products_log/<slug>/``::

    product.json            metadata
    state.json              current checkpoint (state.prev.json is the one before)
    brief.md  roadmap.md  decisions.md
    workspace/              product source, the Developer's only writable area
    verification/           authoritative tests, written only by the orchestrator
    artifacts/              validated product-level artifacts, versioned
    steps/<step_id>/        one directory per roadmap step
    logs/                   events, approvals, conversation, per-task attempt logs
    interventions/          one directory per human intervention, with its reports
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from lab.schemas.product import (
    ApprovalRecord,
    ModelAttempt,
    ProductEvent,
    ProductMetadata,
    TaskResult,
    WorkflowState,
    utcnow,
)
from lab.tools.fsutil import (
    append_line,
    atomic_write_bytes,
    atomic_write_text,
    read_jsonl,
    sha256_file,
)

T = TypeVar("T", bound=BaseModel)


class StateCorruptedError(RuntimeError):
    """Neither the checkpoint nor its predecessor could be read."""


class ProductBusyError(RuntimeError):
    """Another live session already has this product open."""


def new_op_id() -> str:
    return uuid.uuid4().hex


class ProductStore:
    """All reads and writes for one product.  Holds no authoritative memory state."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    @property
    def slug(self) -> str:
        return self.root.name

    @property
    def workspace_dir(self) -> Path:
        return self.root / "workspace"

    @property
    def verification_dir(self) -> Path:
        return self.root / "verification"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def steps_dir(self) -> Path:
        return self.root / "steps"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def interventions_dir(self) -> Path:
        return self.root / "interventions"

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def previous_state_path(self) -> Path:
        return self.root / "state.prev.json"

    @property
    def metadata_path(self) -> Path:
        return self.root / "product.json"

    @property
    def lock_path(self) -> Path:
        return self.root / ".session.lock"

    def step_dir(self, step_id: str) -> Path:
        return self.steps_dir / step_id

    def task_dir(self, task_id: str) -> Path:
        return self.logs_dir / "tasks" / task_id

    def intervention_dir(self, intervention_id: str) -> Path:
        return self.interventions_dir / intervention_id

    def create_layout(self) -> None:
        for directory in (
            self.workspace_dir,
            self.verification_dir,
            self.artifacts_dir,
            self.steps_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Metadata and checkpoint
    # ------------------------------------------------------------------

    def load_metadata(self) -> ProductMetadata:
        meta = ProductMetadata.model_validate_json(self.metadata_path.read_text(encoding="utf-8"))
        if meta.slug != self.root.name:
            # A crash between renaming the folder and rewriting product.json lands
            # here.  The folder's name is what the user sees, so it wins.
            meta = meta.model_copy(update={"slug": self.root.name})
            self.save_metadata(meta)
        return meta

    def save_metadata(self, meta: ProductMetadata) -> None:
        meta.updated_at = utcnow()
        atomic_write_text(self.metadata_path, meta.model_dump_json(indent=2))

    def save_state(self, state: WorkflowState) -> None:
        """Write a new checkpoint, keeping the previous one as a fallback."""
        state.revision += 1
        state.updated_at = utcnow()
        if self.state_path.exists():
            atomic_write_bytes(self.previous_state_path, self.state_path.read_bytes())
        atomic_write_text(self.state_path, state.model_dump_json(indent=2))

    def load_state(self) -> WorkflowState:
        try:
            return WorkflowState.model_validate_json(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as primary_error:
            try:
                state = WorkflowState.model_validate_json(
                    self.previous_state_path.read_text(encoding="utf-8")
                )
            except (OSError, ValidationError, ValueError) as fallback_error:
                raise StateCorruptedError(
                    f"No readable checkpoint in {self.root}: {primary_error}; "
                    f"fallback also failed: {fallback_error}"
                ) from fallback_error
            self.append_event(
                "CheckpointRecovered",
                payload={"reason": str(primary_error)[:300], "revision": state.revision},
            )
            return state

    # ------------------------------------------------------------------
    # Append-only history
    # ------------------------------------------------------------------

    def append_event(
        self,
        event_type: str,
        *,
        op_id: str | None = None,
        phase: str | None = None,
        step_id: str | None = None,
        agent: str | None = None,
        model: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ProductEvent:
        event = ProductEvent(
            event_id=uuid.uuid4().hex,
            op_id=op_id,
            type=event_type,
            phase=phase,
            step_id=step_id,
            agent=agent,
            model=model,
            payload=payload or {},
        )
        line = event.model_dump_json()
        append_line(self.logs_dir / "events.jsonl", line)
        if step_id:
            append_line(self.step_dir(step_id) / "events.jsonl", line)
        return event

    def read_events(self, step_id: str | None = None) -> list[ProductEvent]:
        path = (self.step_dir(step_id) if step_id else self.logs_dir) / "events.jsonl"
        records, _ = read_jsonl(path)
        return [ProductEvent.model_validate(record) for record in records]

    def append_approval(self, record: ApprovalRecord) -> None:
        append_line(self.logs_dir / "approvals.jsonl", record.model_dump_json())

    def read_approvals(self) -> list[ApprovalRecord]:
        records, _ = read_jsonl(self.logs_dir / "approvals.jsonl")
        return [ApprovalRecord.model_validate(record) for record in records]

    def append_conversation(self, speaker: str, text: str) -> None:
        append_line(
            self.logs_dir / "conversation.jsonl",
            json.dumps({"at": utcnow().isoformat(), "speaker": speaker, "text": text}),
        )

    def read_conversation(self) -> list[dict[str, Any]]:
        records, _ = read_jsonl(self.logs_dir / "conversation.jsonl")
        return records

    def append_decision(self, text: str) -> None:
        """Add to decisions.md.  Earlier entries are never rewritten."""
        path = self.root / "decisions.md"
        if not path.exists():
            atomic_write_text(path, "# Decisions\n\nAppend-only record of product decisions.\n")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"\n## {utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n{text.strip()}\n")
            handle.flush()
            os.fsync(handle.fileno())

    # ------------------------------------------------------------------
    # Documents and artifacts
    # ------------------------------------------------------------------

    def write_text(self, relative_path: str, text: str) -> None:
        atomic_write_text(self.root / relative_path, text)

    def read_text(self, relative_path: str, default: str = "") -> str:
        path = self.root / relative_path
        return path.read_text(encoding="utf-8") if path.is_file() else default

    def exists(self, relative_path: str) -> bool:
        return (self.root / relative_path).exists()

    def save_model(self, relative_path: str, model: BaseModel) -> str:
        """Persist a validated artifact and return its SHA-256."""
        path = self.root / relative_path
        atomic_write_text(path, model.model_dump_json(indent=2))
        digest = sha256_file(path)
        assert digest is not None
        return digest

    def load_model(self, relative_path: str, schema: type[T]) -> T:
        return schema.model_validate_json((self.root / relative_path).read_text(encoding="utf-8"))

    def sha256_of(self, relative_path: str) -> str | None:
        return sha256_file(self.root / relative_path)

    # ------------------------------------------------------------------
    # Logical task ledger
    # ------------------------------------------------------------------

    def task_result(self, task_id: str) -> TaskResult | None:
        path = self.task_dir(task_id) / "result.json"
        if not path.is_file():
            return None
        return TaskResult.model_validate_json(path.read_text(encoding="utf-8"))

    def save_task_result(self, result: TaskResult) -> None:
        atomic_write_text(self.task_dir(result.task_id) / "result.json", result.model_dump_json(indent=2))

    def save_task_input(self, task_id: str, overall_attempt: int, round_no: int, prompt: str) -> None:
        name = f"input-r{round_no}-a{overall_attempt}.md"
        atomic_write_text(self.task_dir(task_id) / name, prompt)

    def save_failed_output(
        self, task_id: str, overall_attempt: int, round_no: int, text: str
    ) -> None:
        atomic_write_text(
            self.task_dir(task_id) / f"output-r{round_no}-a{overall_attempt}.md", text
        )

    def failed_outputs(self, task_id: str) -> list[str]:
        """Saved rejected answers, oldest first, relative to the product folder."""
        return [
            path.relative_to(self.root).as_posix()
            for path in sorted(self.task_dir(task_id).glob("output-r*-a*.md"))
        ]

    def start_new_round(self, task_id: str) -> int:
        """Grant a human-requested retry a fresh attempt budget, keeping history."""
        round_no = self.current_round(task_id) + 1
        atomic_write_text(self.task_dir(task_id) / "round.txt", str(round_no))
        return round_no

    def current_round(self, task_id: str) -> int:
        path = self.task_dir(task_id) / "round.txt"
        return int(path.read_text(encoding="utf-8")) if path.is_file() else 1


class DurableAttemptLog:
    """Stores every model attempt, so attempt budgets survive a process restart."""

    def __init__(self, store: ProductStore) -> None:
        self._store = store

    def _path(self, task_id: str) -> Path:
        return self._store.task_dir(task_id) / "attempts.jsonl"

    def record(self, attempt: ModelAttempt) -> None:
        append_line(self._path(attempt.task_id), attempt.model_dump_json())

    def current_round(self, task_id: str) -> int:
        return self._store.current_round(task_id)

    def load(self, task_id: str) -> list[ModelAttempt]:
        """Replay the log to the latest status of each attempt.

        An attempt with only a ``started`` line was cut off by a crash.  That was
        not the model's fault, so it is reported as ``interrupted`` and does not
        consume the task's budget.
        """
        records, _ = read_jsonl(self._path(task_id))
        latest: dict[str, ModelAttempt] = {}
        for record in records:
            attempt = ModelAttempt.model_validate(record)
            latest[attempt.attempt_id] = attempt
        result = list(latest.values())
        for attempt in result:
            if attempt.status == "started":
                attempt.status = "interrupted"
        return result


class ProductLock:
    """Advisory single-session lock.

    It stops two terminals mutating one product, and it is what lets startup call
    an unfinished task "interrupted" rather than "still running elsewhere".
    """

    def __init__(self, store: ProductStore) -> None:
        self._store = store
        self._held = False

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def holder(self) -> dict[str, Any] | None:
        """Return the live holder of the lock, or None if it is free or stale."""
        try:
            raw = self._store.lock_path.read_text(encoding="utf-8")
            info = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return None
        same_host = info.get("host") == socket.gethostname()
        if same_host and not self._pid_alive(int(info.get("pid", -1))):
            return None
        return info

    def acquire(self) -> None:
        payload = json.dumps({"pid": os.getpid(), "host": socket.gethostname()})
        for _ in range(2):
            try:
                fd = os.open(
                    str(self._store.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
                )
            except FileExistsError:
                holder = self.holder()
                if holder is not None and holder.get("pid") != os.getpid():
                    raise ProductBusyError(
                        f"Product '{self._store.slug}' is open in another session "
                        f"(pid {holder.get('pid')})."
                    ) from None
                self._store.lock_path.unlink(missing_ok=True)
                continue
            try:
                os.write(fd, payload.encode("utf-8"))
            finally:
                os.close(fd)
            self._held = True
            return
        raise ProductBusyError(f"Could not lock product '{self._store.slug}'.")

    def release(self) -> None:
        if self._held:
            self._store.lock_path.unlink(missing_ok=True)
            self._held = False

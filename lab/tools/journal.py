"""Write-ahead journal for Developer file writes.

Each write records an *intent* (with before/after hashes), performs the atomic
write, then records a *commit*.  Comparing that journal with the disk tells a
resume exactly what happened to a write that a crash interrupted, so the work can
be repeated safely instead of guessed at.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable, Sequence

from lab.schemas.artifacts import FileOperationRecord
from lab.schemas.product import FileReconciliation, JournaledFileOperation
from lab.tools.developer_tools import MAX_FILE_SIZE_BYTES, ProjectWorkspace, SecurityError
from lab.tools.fsutil import (
    append_line,
    atomic_write_bytes,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    sha256_text,
)

#: Excluded from the workspace fingerprint because running a product changes them
#: without changing what was built.
_FINGERPRINT_IGNORED_DIRS = frozenset(
    {"__pycache__", ".pytest_cache", "node_modules", ".venv", ".git"}
)
_FINGERPRINT_IGNORED_SUFFIXES = frozenset(
    {".pyc", ".db", ".sqlite", ".sqlite3", ".db-journal", ".db-wal", ".db-shm", ".log"}
)
_FINGERPRINT_IGNORED_NAMES = frozenset({".DS_Store"})


def workspace_file_hashes(root: Path | str) -> dict[str, str | None]:
    """Hash each source file in the workspace, in a stable order."""
    base = Path(root)
    entries: dict[str, str | None] = {}
    if base.is_dir():
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(base)
            if not path.is_file() or path.is_symlink():
                continue
            if _FINGERPRINT_IGNORED_DIRS.intersection(relative.parts):
                continue
            if path.suffix.lower() in _FINGERPRINT_IGNORED_SUFFIXES:
                continue
            if path.name in _FINGERPRINT_IGNORED_NAMES:
                continue
            entries[relative.as_posix()] = sha256_file(path)
    return entries


def workspace_fingerprint(root: Path | str) -> str:
    """Hash the workspace's source files.

    Recorded with test evidence, this proves that the code on disk at completion
    is the code that was actually tested.
    """
    return sha256_text(
        "\n".join(f"{path}:{digest}" for path, digest in workspace_file_hashes(root).items())
    )


class JournaledWorkspace(ProjectWorkspace):
    """A ProjectWorkspace whose writes are journaled and safely repeatable."""

    def __init__(
        self,
        root: Path | str,
        journal_path: Path | str,
        protected_paths: Sequence[Path | str] | None = None,
        on_write: Callable[[str, int, bool], None] | None = None,
    ) -> None:
        super().__init__(root, protected_paths=protected_paths)
        self.journal_path = Path(journal_path)
        self.current_task: str | None = None
        self._on_write = on_write

    def _journal(self, operation: JournaledFileOperation) -> None:
        append_line(self.journal_path, operation.model_dump_json())

    def write_file(self, relative_path: str, content: str) -> str:
        try:
            target = self._validate_safe_path(relative_path)
            data = content.encode("utf-8")
            if len(data) > MAX_FILE_SIZE_BYTES:
                raise SecurityError(
                    f"File content exceeds maximum permitted size of {MAX_FILE_SIZE_BYTES} bytes."
                )
        except SecurityError as exc:
            self.operations_log.append(
                FileOperationRecord(
                    file_path=relative_path,
                    action="write",
                    success=False,
                    message=f"Write refused: {exc}",
                )
            )
            raise

        normalized = target.relative_to(self.root).as_posix()
        before = sha256_file(target)
        after = sha256_bytes(data)
        op_id = uuid.uuid4().hex

        if before == after:
            self._journal(
                JournaledFileOperation(
                    op_id=op_id,
                    stage="commit",
                    path=normalized,
                    task_id=self.current_task,
                    sha256_before=before,
                    sha256_after=after,
                    bytes_written=0,
                    unchanged=True,
                )
            )
            message = f"{normalized} already has this exact content; nothing to do."
        else:
            intent = JournaledFileOperation(
                op_id=op_id,
                stage="intent",
                path=normalized,
                task_id=self.current_task,
                sha256_before=before,
                sha256_after=after,
                bytes_written=len(data),
            )
            self._journal(intent)
            atomic_write_bytes(target, data)
            self._journal(intent.model_copy(update={"stage": "commit"}))
            message = f"Successfully wrote {len(data)} bytes to {normalized}"

        self.operations_log.append(
            FileOperationRecord(
                file_path=normalized,
                action="write",
                success=True,
                bytes_transferred=len(data),
                message=message,
            )
        )
        if self._on_write is not None:
            self._on_write(normalized, len(data), before == after)
        return message

    def record_external_change(
        self, relative_path: str, before: str | None, after: str | None, task_id: str
    ) -> None:
        """Journal a change a person made, so reconciliation treats it as known."""
        self._journal(
            JournaledFileOperation(
                op_id=uuid.uuid4().hex,
                stage="commit",
                path=relative_path,
                task_id=task_id,
                sha256_before=before,
                sha256_after=after,
            )
        )

    # ------------------------------------------------------------------
    # Journal queries
    # ------------------------------------------------------------------

    def operations(self) -> list[JournaledFileOperation]:
        records, _ = read_jsonl(self.journal_path)
        return [JournaledFileOperation.model_validate(record) for record in records]

    def committed_for_task(self, task_id: str) -> dict[str, JournaledFileOperation]:
        """Latest committed write per path for one logical task."""
        result: dict[str, JournaledFileOperation] = {}
        for operation in self.operations():
            if operation.task_id == task_id and operation.stage in ("commit", "recovered"):
                result[operation.path] = operation
        return result

    def latest_committed(self) -> dict[str, JournaledFileOperation]:
        """Latest committed write per path across the whole step."""
        result: dict[str, JournaledFileOperation] = {}
        for operation in self.operations():
            if operation.stage in ("commit", "recovered"):
                result[operation.path] = operation
        return result

    def reconcile(self) -> FileReconciliation:
        """Compare the journal with the disk after an interruption.

        Each path's final journal entry is one of:

        * committed and on disk — confirmed.
        * intent only, disk holds the new content — the rename landed but the
          commit line did not; recorded as recovered.
        * intent only, disk holds the old content — never applied; the task reruns.
        * disk holds anything else — someone changed the file outside the
          platform.  It is reported and left alone rather than overwritten.
        """
        report = FileReconciliation()
        last_by_path: dict[str, JournaledFileOperation] = {}
        for operation in self.operations():
            last_by_path[operation.path] = operation

        for path, operation in sorted(last_by_path.items()):
            on_disk = sha256_file(self.root / path)
            if operation.stage in ("commit", "recovered"):
                bucket = report.confirmed if on_disk == operation.sha256_after else report.drifted
                bucket.append(path)
            elif on_disk == operation.sha256_after:
                self._journal(operation.model_copy(update={"stage": "recovered"}))
                report.recovered.append(path)
            elif on_disk == operation.sha256_before:
                report.not_applied.append(path)
            else:
                report.drifted.append(path)
        return report

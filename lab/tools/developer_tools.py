"""Controlled local development tools with strict boundary enforcement.

Safety constraints:
- File operations are strictly isolated to the designated project workspace.
- Path traversal (e.g. '..', absolute paths outside workspace) is blocked and raises SecurityError.
- Authoritative acceptance tests, git repos, virtual environments, and lab code cannot be modified.
- Bounded file size prevents memory or disk exhaustion.
- No shell execution, network access, or secret exposure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from lab.schemas.artifacts import FileOperationRecord

# Default workspace root is the minishop directory inside the repository (for backwards compatibility)
DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parents[2] / "minishop"
MAX_FILE_SIZE_BYTES = 1_048_576  # 1 MB


class SecurityError(PermissionError):
    """Raised when an agent attempts an unauthorized or out-of-bounds file operation."""


class ProjectWorkspace:
    """Encapsulates a sandboxed workspace directory for agent file operations."""

    def __init__(
        self,
        root: Path | str | None = None,
        protected_paths: Sequence[Path | str] | None = None,
    ) -> None:
        self.root = Path(root or DEFAULT_WORKSPACE_ROOT).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.protected_paths: list[Path] = [Path(p).resolve() for p in (protected_paths or [])]
        self.operations_log: list[FileOperationRecord] = []

    def _validate_safe_path(self, relative_path: str) -> Path:
        """Validate and resolve relative_path within the workspace root.

        Raises:
            SecurityError: if relative_path attempts to escape the workspace
                           or targets protected files/directories.
        """
        raw = relative_path.strip()
        if not raw:
            raise SecurityError("File path cannot be empty.")

        # Reject absolute paths immediately
        if raw.startswith(("/", "\\")) or Path(raw).is_absolute():
            raise SecurityError(f"Absolute paths are strictly prohibited: {relative_path!r}")

        # Reject obvious path traversal patterns
        clean_rel = raw
        parts = Path(clean_rel).parts
        if ".." in parts:
            raise SecurityError(f"Path traversal ('..') is strictly prohibited: {relative_path!r}")

        target = (self.root / clean_rel).resolve()

        # Strict containment check
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise SecurityError(
                f"Path {relative_path!r} resolves outside workspace root {self.root}: {target}"
            ) from exc

        # Protected names/extensions check
        lowered = target.as_posix().lower()
        if any(term in lowered for term in [".git", ".venv", "tests/acceptance", "acceptance_tests"]):
            raise SecurityError(
                f"Access to protected system or test path is forbidden: {relative_path!r}"
            )

        # Explicit protected paths check (e.g. custom acceptance test directories)
        for p in self.protected_paths:
            if target == p:
                raise SecurityError(
                    f"Access to protected path is forbidden: {relative_path!r}"
                )
            try:
                target.relative_to(p)
                raise SecurityError(
                    f"Access to protected path is forbidden: {relative_path!r}"
                )
            except ValueError:
                pass

        return target

    def write_file(self, relative_path: str, content: str) -> str:
        """Write content to a file inside the permitted workspace.

        Args:
            relative_path: Path relative to workspace root (e.g. 'main.py' or 'models.py')
            content: Text content to write.

        Returns:
            Success confirmation string.
        """
            content: Text content to write.

        Returns:
            Success confirmation string.
        """
        try:
            target = self._validate_safe_path(relative_path)
            content_bytes = content.encode("utf-8")
            if len(content_bytes) > MAX_FILE_SIZE_BYTES:
                raise SecurityError(
                    f"File content exceeds maximum permitted size of {MAX_FILE_SIZE_BYTES} bytes."
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

            record = FileOperationRecord(
                file_path=relative_path,
                action="write",
                success=True,
                bytes_transferred=len(content_bytes),
                message=f"Successfully wrote {len(content_bytes)} bytes to {relative_path}",
            )
            self.operations_log.append(record)
            return record.message
        except Exception as exc:
            record = FileOperationRecord(
                file_path=relative_path,
                action="write",
                success=False,
                bytes_transferred=0,
                message=f"Write failed: {exc}",
            )
            self.operations_log.append(record)
            raise

    def read_file(self, relative_path: str) -> str:
        """Read and return content from a file inside the permitted workspace.

        Args:
            relative_path: Path relative to workspace root

        Returns:
            The text content of the file.
        """
        try:
            target = self._validate_safe_path(relative_path)
            if not target.exists():
                raise FileNotFoundError(f"File not found: {relative_path}")
            if not target.is_file():
                raise IsADirectoryError(f"Target is a directory, not a file: {relative_path}")

            content = target.read_text(encoding="utf-8")
            record = FileOperationRecord(
                file_path=relative_path,
                action="read",
                success=True,
                bytes_transferred=len(content.encode("utf-8")),
                message=f"Successfully read {len(content)} characters from {relative_path}",
            )
            self.operations_log.append(record)
            return content
        except Exception as exc:
            record = FileOperationRecord(
                file_path=relative_path,
                action="read",
                success=False,
                bytes_transferred=0,
                message=f"Read failed: {exc}",
            )
            self.operations_log.append(record)
            raise

    def list_files(self) -> list[str]:
        """List all permitted files currently existing in the workspace.

        Returns:
            List of relative paths sorted alphabetically.
        """
        files: list[str] = []
        for path in self.root.rglob("*"):
            if path.is_file() and not path.name.startswith("."):
                rel = path.relative_to(self.root).as_posix()
                files.append(rel)
        files.sort()
        record = FileOperationRecord(
            file_path=".",
            action="list",
            success=True,
            bytes_transferred=len(files),
            message=f"Found {len(files)} files in workspace",
        )
        self.operations_log.append(record)
        return files


# Global singleton workspace instance for standard agent invocation
_ACTIVE_WORKSPACE = ProjectWorkspace()


def set_active_workspace(
    root: Path | str,
    protected_paths: Sequence[Path | str] | None = None,
) -> None:
    """Set the active workspace directory for module-level tool functions."""
    global _ACTIVE_WORKSPACE
    _ACTIVE_WORKSPACE = ProjectWorkspace(root, protected_paths=protected_paths)


def get_active_workspace() -> ProjectWorkspace:
    """Return the currently active ProjectWorkspace."""
    return _ACTIVE_WORKSPACE


def write_project_file(relative_path: str, content: str) -> str:
    """Write text content to a permitted project file inside the active workspace.

    Args:
        relative_path: Relative path such as 'main.py' or 'models.py'
        content: The text/code content to write.
    """
    return _ACTIVE_WORKSPACE.write_file(relative_path, content)


def read_project_file(relative_path: str) -> str:
    """Read text content from a permitted project file inside the active workspace.

    Args:
        relative_path: Relative path such as 'main.py' or 'database.py'
    """
    return _ACTIVE_WORKSPACE.read_file(relative_path)


def list_project_files() -> list[str]:
    """List all project files inside the active workspace."""
    return _ACTIVE_WORKSPACE.list_files()

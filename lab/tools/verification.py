"""Authoritative tests and their controlled execution.

Independence comes from three properties:

* Tests live in ``verification/``, outside the Developer's workspace root, and no
  agent holds a tool for that directory.  Python writes them after validation.
* A human-approved suite is sealed with a hash manifest, and every run first
  proves the suite is still exactly what was approved.
* Commands come from the verification-profile allow-list.  Models pick a profile
  id; they never supply a command, argument or working directory.

This is not an OS sandbox: tests are still code running as the current user.  The
runner limits the blast radius (no shell, scrubbed environment, timeout, process
group kill, bounded output) but cannot contain deliberately hostile code.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from lab.config import VerificationProfile
from lab.schemas.product import TestManifest
from lab.tools.developer_tools import ProjectWorkspace, SecurityError
from lab.tools.fsutil import atomic_write_bytes, atomic_write_text, sha256_file, sha256_text
from lab.tools.test_runner import _find_python_executable

MAX_CAPTURED_CHARS = 60_000
_MANIFEST_NAME = "manifest.json"
_IGNORED_DIRS = frozenset({"__pycache__", ".pytest_cache", "node_modules"})
# Finder drops this file into any folder a person opens, which is not a change to the tests.
_IGNORED_NAMES = frozenset({".DS_Store"})
_ENV_PASSTHROUGH = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER")


class VerificationTamperedError(RuntimeError):
    """A sealed test suite no longer matches what the human approved."""


class VerificationToolMissing(RuntimeError):
    """The profile's executable is not installed.  No model can fix this."""


class InvalidTestFile(ValueError):
    """A proposed test file path is not acceptable for the chosen profile."""


class VerificationStore:
    """The verification directory of one product.  Only the orchestrator uses it."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    def step_dir(self, step_id: str) -> Path:
        return self.root / step_id

    def _files(self, step_id: str) -> list[Path]:
        base = self.step_dir(step_id)
        if not base.is_dir():
            return []
        return sorted(
            path
            for path in base.rglob("*")
            if path.is_file()
            and path.name != _MANIFEST_NAME
            and path.name not in _IGNORED_NAMES
            and not _IGNORED_DIRS.intersection(path.relative_to(base).parts)
        )

    @staticmethod
    def check_path(relative_path: str, profile: VerificationProfile) -> None:
        name = Path(relative_path).name
        patterns = [*profile.test_file_patterns, *profile.helper_file_patterns]
        if not any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            raise InvalidTestFile(
                f"{relative_path!r} does not match the {profile.profile_id} profile. "
                f"Use a file name matching one of {patterns}."
            )

    def stage(
        self,
        step_id: str,
        relative_path: str,
        content: str,
        profile: VerificationProfile,
    ) -> None:
        """Write one validated test file into an unsealed suite."""
        self.check_path(relative_path, profile)
        if self.is_sealed(step_id):
            raise VerificationTamperedError(f"The test suite for {step_id} is sealed.")
        workspace = ProjectWorkspace(self.step_dir(step_id))
        try:
            workspace.write_file(relative_path, content)
            for support_path, support_content in profile.support_files.items():
                workspace.write_file(support_path, support_content)
        except SecurityError as exc:
            raise InvalidTestFile(str(exc)) from exc

    def is_sealed(self, step_id: str) -> bool:
        return (self.step_dir(step_id) / _MANIFEST_NAME).is_file()

    def archive_and_clear(self, step_id: str, archive_dir: Path | str) -> None:
        """Move the current suite into the audit trail ready for a new proposal."""
        base = self.step_dir(step_id)
        if not base.is_dir():
            return
        destination = Path(archive_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            target = destination / path.relative_to(base)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        shutil.rmtree(base)

    def seal(
        self,
        step_id: str,
        profile_id: str,
        proposal_version: int,
        *,
        revision: int = 0,
        previous_manifest_sha256: str | None = None,
    ) -> TestManifest:
        """Freeze the suite a human just approved."""
        files = self._files(step_id)
        base = self.step_dir(step_id)
        manifest = TestManifest(
            step_id=step_id,
            profile_id=profile_id,
            proposal_version=proposal_version,
            files={
                path.relative_to(base).as_posix(): sha256_file(path) or "" for path in files
            },
            revision=revision,
            previous_manifest_sha256=previous_manifest_sha256,
        )
        for path in files:
            os.chmod(path, 0o444)
        atomic_write_text(base / _MANIFEST_NAME, manifest.model_dump_json(indent=2), mode=0o444)
        return manifest

    def manifest_digest(self, step_id: str) -> str:
        digest = sha256_file(self.step_dir(step_id) / _MANIFEST_NAME)
        if digest is None:
            raise VerificationTamperedError(f"No sealed test suite exists for {step_id}.")
        return digest

    def manifest(self, step_id: str) -> TestManifest:
        path = self.step_dir(step_id) / _MANIFEST_NAME
        if not path.is_file():
            raise VerificationTamperedError(f"No sealed test suite exists for {step_id}.")
        try:
            return TestManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise VerificationTamperedError(
                f"The seal for {step_id} can no longer be read."
            ) from exc

    def file_hashes(self, step_id: str) -> dict[str, str]:
        base = self.step_dir(step_id)
        return {
            file.relative_to(base).as_posix(): sha256_file(file) or ""
            for file in self._files(step_id)
        }

    def seal_differences(self, step_id: str) -> dict[str, tuple[str | None, str | None]]:
        """Each file that differs from its seal, as (sealed hash, current hash)."""
        sealed = self.manifest(step_id).files
        actual = self.file_hashes(step_id)
        return {
            name: (sealed.get(name), actual.get(name))
            for name in sorted(set(sealed) | set(actual))
            if sealed.get(name) != actual.get(name)
        }

    def verify_seal(self, step_id: str) -> TestManifest:
        """Prove the suite is byte-identical to what was approved."""
        manifest = self.manifest(step_id)
        changed = list(self.seal_differences(step_id))
        if changed:
            raise VerificationTamperedError(
                f"The sealed tests for {step_id} were changed after approval: {changed}"
            )
        return manifest

    def sealed_steps(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(path.parent.name for path in self.root.glob(f"*/{_MANIFEST_NAME}"))

    def reseal(self, step_id: str) -> TestManifest:
        """Freeze a revised suite.  Only a person's recorded approval may lead here."""
        previous = self.manifest(step_id)
        return self.seal(
            step_id,
            previous.profile_id,
            previous.proposal_version,
            revision=previous.revision + 1,
            previous_manifest_sha256=self.manifest_digest(step_id),
        )

    def read_files(self, step_id: str) -> dict[str, str]:
        base = self.step_dir(step_id)
        return {
            file.relative_to(base).as_posix(): file.read_text(encoding="utf-8", errors="replace")
            for file in self._files(step_id)
        }

    def copy_suite(self, step_id: str, destination: Path | str) -> None:
        """Keep a read-only copy of the suite outside the verification directory."""
        base = self.step_dir(step_id)
        target = Path(destination)
        for file in self._files(step_id):
            atomic_write_bytes(
                target / file.relative_to(base), file.read_bytes(), mode=0o444
            )

    def restore_suite(self, step_id: str, source: Path | str) -> None:
        """Put a kept copy back, so the existing seal holds again."""
        base = self.step_dir(step_id)
        origin = Path(source)
        kept = [path for path in origin.rglob("*") if path.is_file()]
        wanted = {path.relative_to(origin).as_posix() for path in kept}
        for file in self._files(step_id):
            if file.relative_to(base).as_posix() not in wanted:
                file.unlink()
        for path in kept:
            atomic_write_bytes(base / path.relative_to(origin), path.read_bytes(), mode=0o444)


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def _parse_pytest(output: str) -> tuple[int, int]:
    summary = ""
    for line in output.splitlines():
        if re.search(r"\bin \d+(\.\d+)?s\b", line) and re.search(
            r"passed|failed|error|no tests ran|skipped|deselected", line
        ):
            summary = line
    passed = re.search(r"(\d+) passed", summary)
    failed = re.search(r"(\d+) failed", summary)
    errors = re.search(r"(\d+) errors?", summary)
    return (
        int(passed.group(1)) if passed else 0,
        (int(failed.group(1)) if failed else 0) + (int(errors.group(1)) if errors else 0),
    )


def _parse_unittest(output: str) -> tuple[int, int]:
    ran = re.search(r"^Ran (\d+) tests? in", output, re.MULTILINE)
    total = int(ran.group(1)) if ran else 0
    outcome = re.search(r"^FAILED \((.*)\)", output, re.MULTILINE)
    failed = 0
    if outcome:
        failed = sum(
            int(count) for count in re.findall(r"(?:failures|errors)=(\d+)", outcome.group(1))
        )
    return max(total - failed, 0), failed


def _parse_tap(output: str) -> tuple[int, int]:
    def count(label: str) -> int:
        found = re.findall(rf"^# {label} (\d+)\s*$", output, re.MULTILINE)
        return int(found[-1]) if found else 0

    return count("pass"), count("fail") + count("cancelled")


RESULT_PARSERS: dict[str, Callable[[str], tuple[int, int]]] = {
    "pytest": _parse_pytest,
    "unittest": _parse_unittest,
    "tap": _parse_tap,
}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawRun:
    command: str
    cwd: str
    exit_code: int
    passed_count: int
    failed_count: int
    duration_ms: float
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool

    @property
    def success(self) -> bool:
        return (
            self.exit_code == 0
            and self.failed_count == 0
            and self.passed_count > 0
            and not self.timed_out
        )


def profile_available(profile: VerificationProfile) -> bool:
    return profile.executable == "python" or shutil.which(profile.executable) is not None


def build_command(
    profile: VerificationProfile,
    tests_dir: Path | str,
    workspace_dir: Path | str,
) -> list[str]:
    """Expand a profile's fixed argv.  Only orchestrator-owned paths are substituted."""
    if profile.executable == "python":
        executable = _find_python_executable()
    else:
        located = shutil.which(profile.executable)
        if located is None:
            raise VerificationToolMissing(
                f"'{profile.executable}' is required by the {profile.profile_id} "
                "verification profile but is not installed."
            )
        executable = located
    values = {"tests_dir": str(tests_dir), "workspace": str(workspace_dir)}
    return [executable, *(arg.format(**values) for arg in profile.args)]


def _clip(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_CAPTURED_CHARS:
        return text, False
    return "[... earlier output omitted ...]\n" + text[-MAX_CAPTURED_CHARS:], True


def run_profile(
    profile: VerificationProfile,
    tests_dir: Path | str,
    workspace_dir: Path | str,
) -> RawRun:
    """Run a profile against a sealed suite and return what actually happened."""
    tests = Path(tests_dir).resolve()
    workspace = Path(workspace_dir).resolve()
    argv = build_command(profile, tests, workspace)

    environment = {key: os.environ[key] for key in _ENV_PASSTHROUGH if key in os.environ}
    environment.update(
        {
            "PYTHONPATH": str(workspace),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PRODUCT_WORKSPACE": str(workspace),
            "CI": "1",
        }
    )

    started = time.perf_counter()
    timed_out = False
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        process = subprocess.Popen(
            argv,
            cwd=workspace,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        try:
            exit_code = process.wait(timeout=profile.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = -1
        finally:
            # Also reaps anything the tests left behind, such as a test server.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            process.wait()
        out.seek(0)
        err.seek(0)
        stdout, out_clipped = _clip(out.read().decode("utf-8", "replace"))
        stderr, err_clipped = _clip(err.read().decode("utf-8", "replace"))

    if timed_out:
        stderr = f"{stderr}\nTest execution timed out after {profile.timeout_seconds} seconds."

    passed, failed = RESULT_PARSERS[profile.result_parser](f"{stdout}\n{stderr}")
    return RawRun(
        command=json.dumps(argv),
        cwd=str(workspace),
        exit_code=exit_code,
        passed_count=passed,
        failed_count=failed,
        duration_ms=(time.perf_counter() - started) * 1000,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        truncated=out_clipped or err_clipped,
    )


__all__ = [
    "InvalidTestFile",
    "RawRun",
    "VerificationStore",
    "VerificationTamperedError",
    "VerificationToolMissing",
    "build_command",
    "profile_available",
    "run_profile",
    "sha256_text",
]

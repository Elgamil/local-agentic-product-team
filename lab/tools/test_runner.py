"""Controlled test execution tool.

Safety constraints:
- Fixed test command and arguments (always pytest against specified test directory).
- Fixed working directory (workspace root).
- Enforced subprocess timeout (prevents hanging test runs).
- Captures stdout/stderr into structured TestRunResult.
- No shell injection possible (runs subprocess without shell=True).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from lab.schemas.artifacts import TestRunResult

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _find_python_executable() -> str:
    """Return the virtualenv python if available, else current sys.executable."""
    venv_python = WORKSPACE_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists() and os.access(venv_python, os.X_OK):
        return str(venv_python)
    return sys.executable


def run_controlled_tests(
    test_target: str = "tests/acceptance",
    timeout_seconds: float = 30.0,
    cwd: Path | str | None = None,
    extra_python_paths: Sequence[Path | str] | None = None,
) -> TestRunResult:
    """Execute pytest under strictly controlled parameters.

    Args:
        test_target: Target path for pytest, e.g. 'tests/acceptance'
        timeout_seconds: Maximum wall-clock time in seconds before terminating.
        cwd: Directory where pytest runs. Defaults to repository root.
        extra_python_paths: Additional directories to include in PYTHONPATH.

    Returns:
        Structured :class:`TestRunResult` with exit code, counts, and output.
    """
    python_bin = _find_python_executable()
    working_dir = Path(cwd or WORKSPACE_ROOT).resolve()

    # Fixed command list - never uses shell=True
    cmd = [python_bin, "-m", "pytest", test_target, "-v"]

    env = os.environ.copy()
    py_paths = [str(working_dir)]
    if extra_python_paths:
        for ep in extra_python_paths:
            resolved_ep = str(Path(ep).resolve())
            if resolved_ep not in py_paths:
                py_paths.append(resolved_ep)
    env["PYTHONPATH"] = os.pathsep.join(py_paths)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=working_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        duration_ms = (time.perf_counter() - t0) * 1000
        stdout = proc.stdout
        stderr = proc.stderr
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        duration_ms = (time.perf_counter() - t0) * 1000
        stdout = exc.stdout or "" if isinstance(exc.stdout, str) else ""
        stderr = f"Test execution timed out after {timeout_seconds} seconds."
        exit_code = -1
    except Exception as exc:
        duration_ms = (time.perf_counter() - t0) * 1000
        stdout = ""
        stderr = f"Failed to execute tests: {exc}"
        exit_code = -2

    # Parse passed and failed counts from pytest output
    # Typical line: "== 5 passed, 1 failed in 0.45s ==" or "== 5 passed in 0.12s =="
    passed_count = 0
    failed_count = 0

    combined_output = stdout + "\n" + stderr
    passed_match = re.search(r"(\d+)\s+passed", combined_output)
    if passed_match:
        passed_count = int(passed_match.group(1))

    failed_match = re.search(r"(\d+)\s+failed", combined_output)
    if failed_match:
        failed_count = int(failed_match.group(1))

    success = (exit_code == 0) and (failed_count == 0) and (passed_count > 0 or "no tests ran" not in combined_output.lower())

    return TestRunResult(
        command=" ".join(cmd),
        exit_code=exit_code,
        passed_count=passed_count,
        failed_count=failed_count,
        duration_ms=duration_ms,
        stdout=stdout,
        stderr=stderr,
        success=success,
    )

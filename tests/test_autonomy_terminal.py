"""The real terminal: a real process, a real pseudo-terminal, real keystrokes.

Everything else in these tests can fake the terminal.  This is the one place that
cannot, because the whole point is how a real tty, input() and the polling channel
behave together, including at exit, where a reader thread would abort Python.
"""

from __future__ import annotations

import os
import pty
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Builds a real session around scripted models, with a real ConsoleTerminal on the tty.
CHILD = """
import asyncio, sys
sys.path.insert(0, {root!r})
from pathlib import Path
from lab.cli import run_interruptibly
from lab.product.console import ConsoleTerminal
from lab.product.session import ProductSession
from tests.fakes import Harness, enable_autonomy, to_step_start_gate

harness = Harness(Path({tmp!r}))
harness.team.delay = {delay}

async def prepare():
    workflow = harness.new_workflow()
    await to_step_start_gate(workflow)
    enable_autonomy(workflow, {mode!r})
    return workflow.store.slug

slug = asyncio.run(prepare())
harness.router = harness.new_router()
harness.team.lifecycle = harness.lifecycle
session = ProductSession(
    harness.manager, harness.router, harness.policy, harness.playbooks, ConsoleTerminal()
)
print("READY", flush=True)
run_interruptibly(session.run_product(session.open_product(slug)), session.release_models)
print("LEFT CLEANLY", flush=True)
"""


class Tty:
    """Drives a child on a pseudo-terminal, as a person at the keyboard would."""

    def __init__(self, tmp_path: Path, mode: str = "on", delay: float = 0.3) -> None:
        self.master, replica = pty.openpty()
        script = CHILD.format(root=str(PROJECT_ROOT), tmp=str(tmp_path), delay=delay, mode=mode)
        environment = {**os.environ, "TERM": "xterm", "PYTHONDONTWRITEBYTECODE": "1"}
        self.process = subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=replica,
            stdout=replica,
            stderr=replica,
            env=environment,
            start_new_session=True,
        )
        os.close(replica)
        self.output = ""

    def wait_for(self, text: str, seconds: float = 60) -> None:
        deadline = time.monotonic() + seconds
        while text not in self.output:
            if time.monotonic() > deadline:
                raise AssertionError(f"Never saw {text!r} in:\n{self.output[-3000:]}")
            ready, _, _ = select.select([self.master], [], [], 0.1)
            if ready:
                try:
                    self.output += os.read(self.master, 65536).decode("utf-8", "replace")
                except OSError:
                    return

    def type(self, line: str) -> None:
        os.write(self.master, f"{line}\n".encode())

    def finish(self, seconds: float = 30) -> int:
        deadline = time.monotonic() + seconds
        while self.process.poll() is None and time.monotonic() < deadline:
            ready, _, _ = select.select([self.master], [], [], 0.1)
            if ready:
                try:
                    self.output += os.read(self.master, 65536).decode("utf-8", "replace")
                except OSError:
                    break
        if self.process.poll() is None:
            self.process.kill()
            raise AssertionError(f"The process did not exit:\n{self.output[-3000:]}")
        return self.process.wait()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        os.close(self.master)


@pytest.fixture
def tty(tmp_path):
    made: list[Tty] = []

    def start(**options) -> Tty:
        made.append(Tty(tmp_path, **options))
        return made[-1]

    yield start
    for item in made:
        item.close()


def assert_clean(output: str) -> None:
    assert "Fatal Python error" not in output
    assert "Traceback" not in output
    assert "LEFT CLEANLY" in output


def test_typing_during_a_run_is_heard_and_the_next_prompt_still_works(tty):
    session = tty()
    session.wait_for("READY")
    session.wait_for("Execution mode: Autonomous")
    session.wait_for("Preparing")

    session.type("/status")
    session.wait_for("Progress: Step 1 of 2")
    session.type("/autonomy off")
    session.wait_for("Manual Mode")

    # input() takes over at the gate, after polling has been reading the same terminal.
    session.wait_for("Shall we proceed to Step 2", seconds=120)
    session.type("/status")
    session.wait_for("Execution mode: Manual")
    session.type("/exit")

    assert session.finish() == 0
    assert_clean(session.output)


def test_leaving_during_a_run_exits_cleanly(tty):
    session = tty()
    session.wait_for("Execution mode: Autonomous")

    session.type("/exit")
    session.wait_for("Leaving at the next safe point")

    assert session.finish(seconds=60) == 0
    assert_clean(session.output)


def test_a_real_ctrl_c_during_a_run_stops_cleanly(tty):
    session = tty(delay=2.0)
    session.wait_for("Execution mode: Autonomous")
    time.sleep(0.5)

    os.killpg(session.process.pid, signal.SIGINT)

    assert session.finish() == 0
    assert "Fatal Python error" not in session.output
    assert "Traceback" not in session.output
    assert "Stopped." in session.output

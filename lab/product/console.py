"""How the conversation looks in a real terminal.

Every message has an explicit speaker label, so the transcript stays readable with
no color at all.  Color is an extra on top, using plain ANSI codes:

    Agent output     cyan
    Your input       green
    Warnings         yellow
    Errors           red
    System status    faint

Color is switched off when output is not a terminal, when ``NO_COLOR`` is set, and
when ``TERM`` says the terminal cannot show it.

Progress heartbeats arrive from a background thread, so every write takes a lock.
On a live terminal a heartbeat rewrites one line in place.  Anywhere else, such as a
log file, it is a complete timestamped line, at a slower pace.
"""

from __future__ import annotations

import os
import re
import select
import sys
import threading
import time
from collections import deque
from typing import Callable, Mapping, TextIO

from lab.product.presentation import role_label
from lab.schemas.product import AgentRole

RESET = "\033[0m"
AGENT_LABEL = "\033[1;36m"
AGENT_TEXT = "\033[36m"
USER_LABEL = "\033[1;32m"
USER_TEXT = "\033[32m"
WARNING_TEXT = "\033[33m"
ERROR_TEXT = "\033[31m"
# Faint rather than "bright black", which some color themes render invisibly.
STATUS_TEXT = "\033[2m"

HEARTBEAT_SECONDS = 10.0
LOG_HEARTBEAT_SECONDS = 30.0

_ESCAPE_SEQUENCES = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"  # Cursor movement, colors and other CSI sequences.
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # Window titles and other OSC sequences.
    r"|\x1b[@-_]?"  # Anything else that starts with ESC.
)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def strip_control(text: str) -> str:
    """Remove terminal escape sequences and control characters.

    Applied to model output, which could otherwise clear the screen or color itself
    to look like another speaker.  Applied to input too, because pressing an arrow
    key at a plain prompt types an escape sequence, not an empty line.
    """
    return _CONTROL_CHARACTERS.sub("", _ESCAPE_SEQUENCES.sub("", text))


def is_live_terminal(stream: TextIO, environ: Mapping[str, str] | None = None) -> bool:
    """True when *stream* is a terminal able to redraw a line."""
    env = os.environ if environ is None else environ
    if env.get("TERM", "") in ("", "dumb"):
        return False
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


def supports_color(stream: TextIO, environ: Mapping[str, str] | None = None) -> bool:
    """Decide whether color can be used on *stream*."""
    env = os.environ if environ is None else environ
    # https://no-color.org: any non-empty value disables color.
    return not env.get("NO_COLOR") and is_live_terminal(stream, env)


def _heartbeat_override(env: Mapping[str, str]) -> float | None:
    """Read LAB_HEARTBEAT_SECONDS, ignoring anything that is not a positive number."""
    try:
        seconds = float(env.get("LAB_HEARTBEAT_SECONDS", ""))
    except ValueError:
        return None
    return seconds if seconds > 0 else None


class ConsoleTerminal:
    """The real terminal."""

    def __init__(
        self,
        stream: TextIO | None = None,
        read_line: Callable[[], str] | None = None,
        environ: Mapping[str, str] | None = None,
        control_fd: int | None = None,
    ) -> None:
        env = os.environ if environ is None else environ
        self._out = stream if stream is not None else sys.stdout
        self._read_line = read_line if read_line is not None else input
        self._control_fd = control_fd
        self._polls_stdin = read_line is None and control_fd is None
        self._typed_lines: deque[str] = deque()
        self._partial_line = b""
        self._input_closed = False
        self._color = supports_color(self._out, env)
        self._live = is_live_terminal(self._out, env)
        self._speaker: str | None = None
        self._lock = threading.RLock()
        self._reading = False
        self._live_width = 0
        self.heartbeat_seconds = _heartbeat_override(env) or (
            HEARTBEAT_SECONDS if self._live else LOG_HEARTBEAT_SECONDS
        )

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _end_live_line(self) -> None:
        """Keep the last heartbeat on screen, so the next output starts on its own line."""
        if self._live_width:
            self._out.write("\n")
            self._live_width = 0

    def _write(self, text: str = "") -> None:
        with self._lock:
            self._end_live_line()
            self._out.write(f"{text}\n")
            self._out.flush()

    def _paint(self, text: str, style: str) -> str:
        """Style each line on its own, so color never bleeds past a line break."""
        if not self._color:
            return text
        return "\n".join(f"{style}{line}{RESET}" if line else "" for line in text.split("\n"))

    def _start_turn(self, speaker: str, style: str) -> None:
        """Label a new speaker.  One speaker's consecutive messages share a label."""
        with self._lock:
            self._write()
            if speaker != self._speaker:
                self._write(self._paint(f"{speaker}:", style))
            self._speaker = speaker

    def _log_line(self, text: str, context: str) -> None:
        """A heartbeat for a log: complete on its own, since nobody sees the lines before."""
        stamp = time.strftime("%H:%M:%S")
        self._write(f"[{stamp}] {context}: {text}" if context else f"[{stamp}] {text}")

    def _redraw(self, text: str) -> None:
        """Replace the live line.  Padding removes any longer text left from before."""
        line = f"  … {text}"
        padding = " " * max(self._live_width - len(line), 0)
        self._out.write(f"\r{self._paint(line, STATUS_TEXT)}{padding}")
        self._live_width = len(line)

    # ------------------------------------------------------------------
    # Terminal interface
    # ------------------------------------------------------------------

    def agent(self, role: str, text: str) -> None:
        with self._lock:
            self._start_turn(role_label(role), AGENT_LABEL)
            self._write(self._paint(strip_control(text), AGENT_TEXT))

    def say(self, text: str) -> None:
        self.agent(AgentRole.BUSINESS_ANALYST.value, text)

    def ask(self, prompt: str) -> str:
        """Ask a question, then read one line from the person.

        The prompt is printed before input() is called, rather than passed to it, so
        any color codes stay out of the line that input() is managing.
        """
        if prompt:
            self.say(prompt)
        if self._typed_lines:
            return self._typed_lines.popleft()
        with self._lock:
            self._start_turn("You", USER_LABEL)
            self._reading = True
            if self._color:
                self._out.write(USER_TEXT)
                self._out.flush()
        try:
            return self._read_line()
        finally:
            # Also runs on Ctrl-C and end of input, so green never leaks to the shell.
            with self._lock:
                self._reading = False
                if self._color:
                    self._out.write(RESET)
                    self._out.flush()

    def poll_line(self) -> str | None:
        """Return a line typed while the team works, or None.  It never blocks.

        This is a check on the input, not a reader thread.  A thread parked in a read of
        stdin cannot be stopped, and it can abort Python as the program exits.  A
        terminal only reports input once Enter is pressed, so a half-typed line stays
        with the terminal until it is finished.
        """
        if self._typed_lines:
            return self._typed_lines.popleft()
        fd = self._input_descriptor()
        if fd is None:
            return None
        try:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                return None
            data = os.read(fd, 4096)
        except (OSError, ValueError):
            return None
        if not data:
            self._input_closed = True
            return None
        *complete, self._partial_line = (self._partial_line + data).split(b"\n")
        self._typed_lines.extend(line.decode("utf-8", "replace") for line in complete)
        return self._typed_lines.popleft() if self._typed_lines else None

    def _input_descriptor(self) -> int | None:
        if self._input_closed:
            return None
        if not self._polls_stdin:
            return self._control_fd
        try:
            return sys.stdin.fileno()
        except (AttributeError, OSError, ValueError):
            return None

    def status(self, text: str) -> None:
        with self._lock:
            self._speaker = None
            self._write(self._paint(f"  … {strip_control(text)}", STATUS_TEXT))

    def warn(self, text: str) -> None:
        with self._lock:
            self._speaker = None
            self._write()
            self._write(self._paint(f"Warning: {strip_control(text)}", WARNING_TEXT))

    def error(self, text: str) -> None:
        with self._lock:
            self._speaker = None
            self._write()
            self._write(self._paint(f"Error: {strip_control(text)}", ERROR_TEXT))

    def progress(self, text: str, context: str = "") -> None:
        """Show a heartbeat.  Dropped while the person is typing, so it cannot garble them."""
        with self._lock:
            if self._reading:
                return
            self._speaker = None
            if self._live:
                self._redraw(strip_control(text))
                self._out.flush()
            else:
                self._log_line(strip_control(text), strip_control(context))

    def progress_done(self, text: str, context: str = "") -> None:
        """Show how a wait ended, in place of its last heartbeat when there was one."""
        with self._lock:
            self._speaker = None
            clean = strip_control(text)
            if not self._live:
                self._log_line(clean, strip_control(context))
            elif self._live_width:
                self._redraw(clean)
                self._end_live_line()
                self._out.flush()
            else:
                self.status(clean)

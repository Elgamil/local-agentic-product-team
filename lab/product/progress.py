"""What the person sees while the system waits.

Everything here only reports.  Nothing is saved, no model is contacted, and the
wording never claims more than is known: while a model has not answered, the only
true statement is that the system is still waiting, and for how long.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Protocol

from lab.product.presentation import role_label
from lab.runtime.heartbeat import Heartbeat
from lab.schemas.product import FailureClass, ModelAttempt

logger = logging.getLogger(__name__)

_FAILURE_WORDING = {
    FailureClass.MODEL_CONNECTION: "could not reach the model",
    FailureClass.INVALID_RESPONSE: "the model gave an empty or unusable answer",
    FailureClass.INVALID_JSON: "the answer was incomplete or badly formed",
    FailureClass.SCHEMA_VALIDATION: "the answer did not match the required structure",
    FailureClass.TOOL_EXECUTION: "the required file work was not completed",
    FailureClass.FILESYSTEM: "a problem with the disk",
    FailureClass.DEPENDENCY: "a required tool is not installed",
}


class ProgressDisplay(Protocol):
    heartbeat_seconds: float

    def status(self, text: str) -> None: ...

    def warn(self, text: str) -> None: ...

    def error(self, text: str) -> None: ...

    def progress(self, text: str, context: str = "") -> None: ...

    def progress_done(self, text: str, context: str = "") -> None: ...


def format_duration(seconds: float) -> str:
    """Whole seconds in plain words, such as "43 seconds" or "2 minutes 5 seconds"."""
    total = max(int(round(seconds)), 0)
    minutes, remainder = divmod(total, 60)
    parts = []
    if minutes:
        parts.append(f"{minutes} minute{'' if minutes == 1 else 's'}")
    if remainder or not minutes:
        parts.append(f"{remainder} second{'' if remainder == 1 else 's'}")
    return " ".join(parts)


def show(write: Callable[..., None], *arguments: str) -> None:
    """Display something.  A display problem must never affect the work."""
    try:
        write(*arguments)
    except Exception as exc:
        logger.warning("Could not display progress: %s", exc)


@contextmanager
def _ticking(
    display: ProgressDisplay,
    context: str,
    still: str,
    interrupted: str,
    clock: Callable[[], float],
) -> Iterator[Heartbeat]:
    def tick(elapsed: float) -> None:
        display.progress(f"{still} — {format_duration(elapsed)} elapsed.", context)

    beat = Heartbeat(tick, display.heartbeat_seconds, clock)
    try:
        with beat:
            yield beat
    except Exception:
        raise  # The caller knows what kind of failure this is, and reports it.
    except BaseException:
        show(display.warn, f"Interrupted {interrupted}, after {format_duration(beat.elapsed)}.")
        raise


class ModelProgress:
    """Follows one model task through its attempts."""

    def __init__(
        self,
        display: ProgressDisplay,
        role: str,
        operation: str,
        total_attempts: int,
        time_limit: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._display = display
        self._title = f"{role_label(role)} — {operation}"
        self._total_attempts = total_attempts
        self._time_limit = time_limit
        self._clock = clock

    def _context(self, attempt: ModelAttempt) -> str:
        return f"{self._title} ({attempt.model})"

    def _attempt_label(self, attempt: ModelAttempt) -> str:
        return f"attempt {attempt.overall_attempt} of {self._total_attempts}"

    @contextmanager
    def waiting(self, attempt: ModelAttempt) -> Iterator[None]:
        """Wraps the model call.  Runs only while nothing has been received."""
        title = self._title
        if attempt.overall_attempt > 1:
            title = f"{title} ({self._attempt_label(attempt)})"
        show(self._display.status, title)
        show(self._display.status, f"Model: {attempt.model}")
        show(
            self._display.status,
            "Waiting for model response... "
            f"(time limit: {format_duration(self._time_limit)})",
        )
        with _ticking(
            self._display,
            self._context(attempt),
            "Still waiting for model response",
            "while waiting for the model",
            self._clock,
        ) as beat:
            yield
            elapsed = beat.elapsed
        show(
            self._display.progress_done,
            f"Model response received — {format_duration(elapsed)}.",
            self._context(attempt),
        )

    @contextmanager
    def processing(self, attempt: ModelAttempt) -> Iterator[None]:
        """Wraps parsing and validation.  Silent unless it reaches a heartbeat."""
        with _ticking(
            self._display,
            self._context(attempt),
            "Still processing the response",
            "while processing the response",
            self._clock,
        ):
            yield

    def failed(self, attempt: ModelAttempt) -> None:
        """Report a failed attempt.  The router decides what happens next."""
        where = f"{self._title}, {self._attempt_label(attempt)}, on {attempt.model}"
        elapsed = format_duration(attempt.duration_ms / 1000)
        failure = attempt.failure_class
        if failure == FailureClass.INFERENCE_TIMEOUT:
            show(
                self._display.warn,
                f"Timed out: no model response within the configured limit of "
                f"{format_duration(self._time_limit)}. ({where})",
            )
        else:
            reason = _FAILURE_WORDING.get(failure, "the attempt failed") if failure else "failed"
            show(self._display.error, f"Failed after {elapsed}: {reason}. ({where})")


@contextmanager
def watching_test_process(
    display: ProgressDisplay,
    title: str,
    time_limit: float,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[Any]:
    """Wraps one real test run.  The caller reports how it ended."""
    show(display.status, title)
    show(display.status, f"Running test process... (time limit: {format_duration(time_limit)})")
    with _ticking(
        display, title, "Test process still running", "while running tests", clock
    ) as beat:
        yield beat

"""A background ticker for operations that block their caller.

It has to be a thread.  Test runs block the event loop in ``process.wait()``, so an
asyncio task would stop ticking at exactly the moment it was needed.

The heartbeat only ever calls the function it was given.  It holds no reference to
the operation, so it cannot start a model request or alter a result.
"""

from __future__ import annotations

import logging
import threading
import time
from types import TracebackType
from typing import Callable

logger = logging.getLogger(__name__)

THREAD_NAME = "progress-heartbeat"
# Only a blocked terminal could keep a tick running this long.  It must not hold up
# the work, and the thread is a daemon, so it cannot keep the process alive either.
_JOIN_SECONDS = 2.0


class Heartbeat:
    """Calls ``tick(elapsed_seconds)`` every *interval* seconds inside a ``with`` block.

    Leaving the block by any route, including an exception, cancellation or Ctrl-C,
    stops the thread and waits for it, so no tick can arrive afterwards.
    """

    def __init__(
        self,
        tick: Callable[[float], None],
        interval: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tick = tick
        self._interval = interval
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started_at = 0.0

    @property
    def elapsed(self) -> float:
        return self._clock() - self.started_at

    def __enter__(self) -> "Heartbeat":
        self.started_at = self._clock()
        self._thread = threading.Thread(target=self._run, name=THREAD_NAME, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join(timeout=_JOIN_SECONDS)

    def _run(self) -> None:
        next_tick = self.started_at + self._interval
        while not self._stop.wait(max(next_tick - self._clock(), 0.0)):
            try:
                self._tick(self.elapsed)
            except Exception as exc:  # A display problem must never reach the work.
                logger.warning("Heartbeat stopped: %s", exc)
                return
            next_tick += self._interval

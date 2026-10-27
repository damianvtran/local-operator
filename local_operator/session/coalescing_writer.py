"""One background thread that runs small evidence writes off the event loop.

WHY THIS EXISTS. Per-lane evidence (``subagent_ledger.note_lane_step``, the WIP
checkpoint's step receipt) is written from a child's ``on_tool_end`` relay,
which runs ON THE PARENT SESSION'S LOOP. Each write is a read-modify-write of a
JSON file: ``mkstemp`` + ``json.dump`` + ``chmod`` + ``os.replace``. Alone that is
a millisecond; on a manager with a dozen busy lanes, a contended disk and the
GIL shared with every lane's provider stream it measured 0.8 s (ledger) and
1.0 s (WIP) of loop wall per ~40 s busy window, with single writes up to 315 ms
(``scripts/bench_model_switch.py``, 12 lanes, 1000-row roster, 110 MB
transcript). Every control op — a model switch included — queues behind it.

WHAT IT GUARANTEES, and the strength of each:

* **Off-loop.** ``submit`` only records the op and wakes the thread; the file
  I/O runs on ``lop-evidence-writer``.
* **Coalesced, latest-wins per key.** A key (the target path) holds at most ONE
  pending op; a newer submission replaces it in place (keeping its queue
  position), optionally through ``merge`` so a caller can keep a property of the
  replaced op (the ledger keeps ``force``: a durable recovery ref must not be
  dropped by a later throttled step).
* **Ordered.** Ops run one at a time, FIFO by first submission, and every op on
  this writer — plus anything a caller runs inside :meth:`exclusive` — is
  serialised by one I/O lock. That is what keeps a deferred step write from
  RESURRECTING a receipt that a settle or ``clear()`` has just unlinked: the
  unlinker takes :meth:`exclusive`, which drops the key's pending op and waits
  out any op already running before the unlink happens.
* **Process-durable on a clean exit.** ``atexit`` flushes the queue (bounded),
  so a normal shutdown does not lose the last step. A HARD death can lose the
  ops still queued — at most one per key, and these are the "last completed
  step" breadcrumbs the writers already throttle to 5 s, so the loss window is
  the same order as the throttle that was always there. Launch receipts and stop
  receipts, whose whole value is surviving a hard death, are NOT routed here.

IMPORT GRAPH: stdlib only (no asyncio, no ``local_operator``). Both
``session.subagent_ledger`` (pinned import-light by
``tests/unit/test_import_graph.py``) and ``harness.wip_checkpoint`` (stdlib-only
by its own docstring) import it, so it must not widen either closure.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import threading
import time
from typing import Callable, Hashable, Iterator

logger = logging.getLogger(__name__)

Op = Callable[[], None]
Merge = Callable[[Op, Op], Op]


class CoalescingWriter:
    """A lazily started, fork-safe, single-thread latest-wins op queue."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._reset()
        # A forked child inherits neither the thread nor a usable lock state;
        # start it over rather than wait forever on a lock a dead thread held.
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._reset)

    def _reset(self) -> None:
        self._state = threading.Condition(threading.Lock())
        self._io = threading.RLock()
        self._pending: dict[Hashable, Op] = {}
        self._running = False
        self._thread: threading.Thread | None = None

    def submit(self, key: Hashable, op: Op, merge: Merge | None = None) -> None:
        """Queue ``op`` for ``key``, replacing (or merging into) a pending one."""
        with self._state:
            previous = self._pending.get(key)
            self._pending[key] = merge(previous, op) if previous and merge else op
            self._ensure_thread()
            self._state.notify_all()

    @contextlib.contextmanager
    def exclusive(self, key: Hashable) -> Iterator[None]:
        """Drop ``key``'s pending op and hold the I/O lock for the caller's write.

        For a synchronous writer of the same file (an unlink on settle, a
        snapshot receipt written from its own worker thread): once inside, no
        queued op for ``key`` can run afterwards with older state, and no op is
        mid-write.
        """
        with self._io:
            with self._state:
                self._pending.pop(key, None)
            yield

    @contextlib.contextmanager
    def serialized(self) -> Iterator[None]:
        """Hold the I/O lock WITHOUT dropping anything (a fresher sync write)."""
        with self._io:
            yield

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until nothing is pending or running; ``False`` on timeout."""
        deadline = time.monotonic() + timeout
        with self._state:
            while self._pending or self._running:
                if self._thread is None or not self._thread.is_alive():
                    # No thread to drain it (start failed): run inline.
                    self._state.release()
                    try:
                        self._drain_inline()
                    finally:
                        self._state.acquire()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._state.wait(remaining)
        return True

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        try:
            thread.start()
        except RuntimeError:
            # ``can't start new thread``: leave the ops queued; ``flush`` (and
            # the atexit hook) drain them inline. Never raise into a relay.
            logger.debug("evidence writer thread could not start", exc_info=True)
            return
        self._thread = thread

    def _take(self) -> Op | None:
        with self._state:
            if not self._pending:
                return None
            key = next(iter(self._pending))
            self._running = True
            return self._pending.pop(key)

    def _execute(self, op: Op) -> None:
        try:
            with self._io:
                op()
        except Exception:  # noqa: BLE001 — evidence must never kill the writer
            logger.warning("evidence write failed", exc_info=True)
        finally:
            with self._state:
                self._running = False
                self._state.notify_all()

    def _drain_inline(self) -> None:
        while (op := self._take()) is not None:
            self._execute(op)

    def _run(self) -> None:
        while True:
            with self._state:
                while not self._pending:
                    self._state.wait()
            op = self._take()
            if op is not None:
                self._execute(op)


#: The process-wide instance. One thread for every evidence file: these writes
#: are tiny, and a single serial lane is what makes the ordering argument above
#: hold across the ledger and the WIP receipt without a lock per file.
EVIDENCE_WRITER = CoalescingWriter("lop-evidence-writer")


@atexit.register
def _flush_at_exit() -> None:
    # Bounded: a wedged disk must not hold interpreter shutdown hostage.
    EVIDENCE_WRITER.flush(timeout=2.0)

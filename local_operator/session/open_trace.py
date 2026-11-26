"""Env-gated JSONL trace for the open-frame read path.

WHY AN ENV-GATED TRACE RATHER THAN A LOG LINE. Opening a peer conversation
spans THREE processes — the desktop daemon, this device's relay and the owner's
relay — and "where did the milliseconds go" is a question only rows from more
than one of them can answer, correlated by time. A log line forces a choice of
level the operator must enable globally; this trace is one file per process,
turned on with ``LOP_OPEN_TRACE=<path>`` in exactly the process under study
(a rig cell, or a live machine with the env set for one daemon). Disabled
tracing performs no JSON encoding or file I/O: the writer returns on a
module-level ``None`` check. Call sites still read span-start clocks, so the
disabled overhead is small, not literally zero.

ROWS ARE APPEND-ONLY JSONL under a lock: ``at`` (epoch ms), ``stage``,
``pid``, an optional ``ms`` span the caller just finished timing, and whatever fields the
site knows. The writer NEVER raises — a trace that can break the read it
observes is worse than no trace — and the module is inert when the variable is
unset.

THE MARKS SHIP with the change they instrumented (all guarded by
:func:`enabled`), so the numbers on the PR stay reproducible on any machine:
set the variable, run the same drive, read the JSONL rows back.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from enum import StrEnum
from typing import ParamSpec, TypeVar
from uuid import uuid4

#: Reads the variable ONCE at import: the mark sites guard on :func:`enabled`,
#: and a per-call ``getenv`` would be a string lookup on the read path the trace
#: exists to measure. A process that wants a trace sets the variable before it
#: starts — there is deliberately no setter, so no second opinion about whether
#: tracing is on can grow beside this one.
_PATH = os.environ.get("LOP_OPEN_TRACE") or None
_LOCK = threading.Lock()


def enabled() -> bool:
    """Whether this process traces. Gates every mark site."""
    return _PATH is not None


def mark(stage: str, ms: float | None = None, **fields: object) -> None:
    """Append one trace row. Never raises, never mutates ``fields``."""
    if _PATH is None:
        return
    # ``pid`` on EVERY row: session runtimes are children of ``lop serve`` and
    # inherit ``LOP_OPEN_TRACE``, so one file interleaves serve rows with
    # runtime rows. Without the pid a count such as ``index.refresh_start``
    # cannot say whether serve or a runtime scanned (the transcript-index ADR
    # could not settle its 14,483-row count for exactly that reason). Read per
    # row, not cached at import: a forked child must stamp its own pid. Only
    # reached when tracing is on, so the disabled path is unchanged.
    row: dict[str, object] = {"at": time.time() * 1000.0, "stage": stage, "pid": os.getpid()}
    if ms is not None:
        row["ms"] = round(ms, 3)
    row.update(fields)
    try:
        line = json.dumps(row, separators=(",", ":"), default=str)
        with _LOCK, open(_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:  # noqa: BLE001 — see the module docstring: observability
        pass  # must never fail the read it observes.


class DesktopEventsIngressTrace:
    """Join one authenticated desktop first-paint request across ASGI stages.

    The entry clock is sampled before middleware/auth awaits but written only
    after the validated route is reached. This is ASGI dispatch, not arrival on
    the socket. Epoch milliseconds join other processes' rows; monotonic
    elapsed milliseconds measure this request even if wall time jumps.

    On ``desktop.events.*`` rows, ``ms`` is always since ASGI entry, and
    ``open_id`` joins the request without its URL or session id. The acquire
    row's separate ``acquire_id`` joins to the existing ``a.acquire`` span
    (whose ``ms`` remains its own duration), and to ``cold.summary.span`` only
    when that acquire built the facade. A reused facade/shared cold work is
    not a cold construction cost attributable to each waiter.

    The optional client stream ID appears only on ``asgi_entry``. Later phases
    use the server's own ID so even repeated client IDs cannot conflate them.
    """

    def __init__(
        self, entry_mono: float, entry_at: float, *, client_stream: str | None = None
    ) -> None:
        self._entry_mono = entry_mono
        self._entry_at = entry_at
        self._client_stream = client_stream
        self.open_id = uuid4().hex
        self.started = False

    def start(self) -> None:
        route_ms = (time.monotonic() - self._entry_mono) * 1000
        route_at = time.time() * 1000
        self.started = True
        # Record both boundaries, not the writer's later emission times: the
        # first row's write must not count as time spent in middleware.
        client = {} if self._client_stream is None else {"client_stream": self._client_stream}
        mark("desktop.events.asgi_entry", 0.0, at=self._entry_at, open_id=self.open_id, **client)
        mark("desktop.events.route", route_ms, at=route_at, open_id=self.open_id)

    def mark(self, stage: str, **fields: object) -> None:
        mark(stage, (time.monotonic() - self._entry_mono) * 1000, open_id=self.open_id, **fields)


class ColdWorkerPhase(StrEnum):
    """The selected default-executor handoffs, never request-supplied labels."""

    MODEL_SYNTHESIS = "model_synthesis"
    ACCOUNT_CONTEXT = "account_context"
    DURABLE_VALIDATE = "durable_validate"


_P = ParamSpec("_P")
_T = TypeVar("_T")


class ColdOpenTrace:
    """A bounded summary per cold call, plus worker timing for desktop acquires.

    Phases include scheduling time in existing awaits. The final row also closes
    failed/cancelled calls, with no exception text or extra task/thread/timer.
    Construct only behind ``enabled()`` so disabled opens read no new clocks.
    Session + span joins the enclosing acquire when supplied; otherwise epoch
    interval + a fresh span disambiguates
    concurrent cold calls for the same session without a global registry.
    """

    def __init__(self, session_id: str, span: str | None = None) -> None:
        self.fields: dict[str, object] = {
            "session": session_id,
            "span": span or uuid4().hex,
            "started_at": time.time() * 1000.0,
        }
        self.started = time.monotonic()
        self._acquire_id = span or None
        # Non-acquire cold opens still have summaries but no existing acquire
        # ID to join to. Worker samples live only on this acquire's trace.
        self._workers: dict[ColdWorkerPhase, tuple[int, int, int, int, int]] | None = (
            {} if self._acquire_id is not None else None
        )

    async def worker(
        self, phase: ColdWorkerPhase, func: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        """Measure an existing ``to_thread`` call without moving or owning its work.

        Submit→start includes handoff, executor queuing and worker scheduling;
        end→resume includes future delivery and loop scheduling. Neither alone
        proves saturation or a blocked loop. The CPU delta belongs to the same
        worker thread; wall without CPU does not identify who held the GIL.
        """
        workers = self._workers
        if workers is None:
            return await asyncio.to_thread(func, *args, **kwargs)

        sample: tuple[int, int, int] | None = None

        def run() -> _T:
            nonlocal sample
            started = cpu_started = None
            # Publish only a complete numeric tuple, without I/O or locks on
            # the measured worker. The original to_thread Future delivers it.
            with suppress(Exception):
                started = time.monotonic_ns()
                cpu_started = time.thread_time_ns()
            try:
                return func(*args, **kwargs)
            finally:
                if started is not None and cpu_started is not None:
                    with suppress(Exception):
                        cpu_ns = time.thread_time_ns() - cpu_started
                        sample = (started, time.monotonic_ns(), cpu_ns)

        try:
            submitted = time.monotonic_ns()
        except Exception:  # noqa: BLE001 — failed diagnostics are not failed work
            submitted = None
        if submitted is None:
            return await asyncio.to_thread(func, *args, **kwargs)

        cancelled = False
        try:
            return await asyncio.to_thread(run)
        except asyncio.CancelledError:
            # Don't join the original worker or read a sample that it may
            # still be writing. Cancellation is not result delivery.
            cancelled = True
            raise
        finally:
            if not cancelled and sample is not None:
                with suppress(Exception):
                    resumed = time.monotonic_ns()
                    started, ended, cpu_ns = sample
                    if submitted <= started <= ended <= resumed and cpu_ns >= 0:
                        workers[phase] = (submitted, started, ended, resumed, cpu_ns)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.monotonic()
        try:
            yield
        finally:
            self.fields[name + "_ms"] = round((time.monotonic() - started) * 1000.0, 3)

    def finish(self, outcome: str) -> None:
        mark(
            "cold.summary",
            (time.monotonic() - self.started) * 1000.0,
            outcome=outcome,
            **self.fields,
        )
        # Bound output to one row per constant phase, after the existing
        # state install and summary. The acquire ID is the only correlation
        # value; no session, model, checkpoint or exception data enter here.
        if self._workers is not None:
            with suppress(Exception):
                for phase in ColdWorkerPhase:
                    sample = self._workers.get(phase)
                    if sample is not None:
                        submitted, started, ended, resumed, cpu_ns = sample
                        mark(
                            "cold.worker",
                            acquire_id=self._acquire_id,
                            phase=phase.value,
                            submit_ns=submitted,
                            worker_start_ns=started,
                            worker_end_ns=ended,
                            resume_ns=resumed,
                            worker_cpu_ns=cpu_ns,
                        )

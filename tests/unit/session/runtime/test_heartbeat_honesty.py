"""The beat publishes what it MEASURED, so ``wedged`` stops being one word.

WHY THIS FILE EXISTS, and it is the second half of the 2026-09-20 freeze. Five
runtimes read as ``wedged`` for 1.5-7.2 h, and that single word could not say
which of three different things was true of them: the runtime was burning its
own core (measured: ~0.9 core while the transcript took ZERO writes), the host
had stopped scheduling it, or it was simply gone. All three read ``wedged`` at
45 s, and the distinction is not academic — the first is a defect in the work,
the second is a defect in the machine, and the third needs no diagnosis at all.

So the heartbeat now publishes the gap it actually measured and how much CPU
this process spent over that same gap, both read in-process (``time.monotonic``
and ``time.process_time``) so a tick costs no ``ps`` fork per session — the
lesson ``control.py`` records at 201 forks per probe.

The vocabulary is deliberately UNTOUCHED: ``live``/``wedged``/``stale`` is a
wire value ~15 call sites and the desktop catalogue branch on, so these are two
ADDITIVE fields on the record and two additive keys on the published row, with
``None`` meaning "this build does not report" rather than zero.
"""

from __future__ import annotations

import ast
import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import StreamEndEvent
from local_operator.session.runtime import registry
from local_operator.session.runtime import server as server_module
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.types import SessionRecord
from tests.unit.session.test_session import make_session

REPO = Path(__file__).resolve().parents[4]

#: How long a test waits for a beat to land. The loop is driven at
#: ``BEAT_INTERVAL_S`` rather than its production 15 s so this is a test about
#: what a beat PUBLISHES and not a 15-second wait; the deadline is only there so
#: a loop that fell over fails instead of blocking the worker (there is no
#: pytest-timeout in this suite).
BEAT_INTERVAL_S = 0.05
#: How long a test waits for the FIRST beat to be published. Generous on
#: purpose: the reading is a fact about the record, not about this host, and
#: under the fleet load this box routinely carries (load 30-80) a boot plus
#: one beat was measured once outside a 20 s window — the deadline exists so a
#: loop that fell over fails instead of blocking the worker, never to assert
#: that a machine is fast (AGENTS.md, "Prefer a structural invariant").
BEAT_DEADLINE_S = 45.0


def _stream(request: Any, signal: Any) -> Any:
    """A stream that ends at once: nothing here needs a turn to run."""

    async def gen() -> Any:
        yield StreamEndEvent(stop_reason="stop")

    return gen()


async def _boot(tmp_path: Path) -> RuntimeServer:
    """A real session, real handle, thread-hosted runtime, published."""
    loop = asyncio.get_running_loop()
    session = make_session(tmp_path, _stream)
    handle = ServingSessionHandle(session, loop, cwd=str(tmp_path))
    runtime = RuntimeServer(handle, kind="daemon")
    runtime.start()
    assert await runtime.wait_until_published(), "the boot prologue never published"
    return runtime


def _published(pid: int) -> SessionRecord | None:
    """This process's own record, as a fresh reader would parse it."""
    for record, _state in registry.scan():
        if record.pid == pid:
            return record
    return None


@pytest.mark.asyncio
async def test_the_beat_publishes_the_gap_it_measured_and_the_cpu_it_burned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both readings, on a REAL runtime, parsed back from the record file.

    Read back through ``registry.scan`` rather than off the live object, because
    the claim is about what a reader finds in the file — a field set on the
    in-memory record but never serialized would satisfy the object and tell
    ``lop sessions`` nothing.

    The second half is the property that keeps the pair honest across a turn
    boundary: a transition republish rewrites the record WITHOUT re-measuring,
    so it must carry the last measurement forward rather than blanking it. A
    blanked field would read as "no reading taken", which is exactly the
    ambiguity the pair exists to remove.
    """
    monkeypatch.setattr(server_module, "HEARTBEAT_INTERVAL_S", BEAT_INTERVAL_S)
    runtime = await _boot(tmp_path)
    try:
        deadline = time.monotonic() + BEAT_DEADLINE_S
        record = None
        while time.monotonic() < deadline:
            record = _published(runtime._record.pid)
            if record is not None and record.beat_lag_s is not None:
                break
            await asyncio.sleep(BEAT_INTERVAL_S)
        assert record is not None, "the runtime published no record at all"
        assert record.beat_lag_s is not None, (
            "the heartbeat published no measured gap, so a reader still cannot tell a "
            "runtime burning its own CPU from one the host descheduled"
        )
        assert record.cpu_since_beat_s is not None
        assert isinstance(record.beat_lag_s, float) and record.beat_lag_s >= 0.0
        assert isinstance(record.cpu_since_beat_s, float) and record.cpu_since_beat_s >= 0.0
        measured = record.beat_lag_s

        # A TRANSITION REPUBLISH DOES NOT BLANK IT.
        runtime.set_busy(True)
        republished = _published(runtime._record.pid)
        assert republished is not None and republished.busy is True
        assert (
            republished.beat_lag_s is not None
        ), "a turn-boundary republish dropped the last measurement"
        assert republished.cpu_since_beat_s is not None
        assert republished.beat_lag_s >= measured

        # ...and the two readings are about the same runtime: the beat that took
        # them ran in this process, so the record's pid is this one.
        assert republished.kind == "daemon"
    finally:
        runtime.close()


def test_the_beat_reads_no_process_table() -> None:
    """No ``ps``/``lsof`` fork per tick, asserted against the loop's own body.

    A fork per heartbeat per session is the cost this design refuses (the
    control-plane probe's 201 forks at 200 records is the precedent), and it is
    the easy thing to add while "improving" the honesty of a beat. The readings
    must come from the process's own clocks: ``time.process_time`` is
    in-process CPU, and ``time.monotonic`` is a syscall with no child.
    """
    source = (REPO / "local_operator" / "session" / "runtime" / "server.py").read_text(
        encoding="utf-8"
    )
    loop = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_heartbeat_loop"
    )
    body = ast.unparse(loop)
    assert "process_time" in body, "the beat no longer measures this process's own CPU"
    for forbidden in ("subprocess", "Popen", "os.fork", "fork(", "spawn", "psutil"):
        assert forbidden not in body, f"the heartbeat reaches the process table through {forbidden}"


def test_a_record_without_the_fields_reads_back_as_unreported() -> None:
    """Mid-upgrade tolerance, asserted on the deserializer.

    ``lop sessions`` and ``/info`` are the surfaces opened ON a host that is
    part-way through an upgrade, so a record written by a runtime predating
    these fields must list cleanly — and must read as ``None`` (this build has
    not told us either way) rather than as a zero reading that would look like a
    runtime that measured nothing wrong.
    """
    old = {
        "pid": 1234,
        "kind": "daemon",
        "session_id": "abc",
        "conversation_name": "an older runtime",
        "cwd": "/tmp",
        "model_label": "mock",
        "control_port": 0,
        "control_key": "",
        "heartbeat_at": time.time(),
    }
    record = SessionRecord.from_json(old)
    assert record.beat_lag_s is None
    assert record.cpu_since_beat_s is None
    assert "beat_lag_s" in record.to_json()

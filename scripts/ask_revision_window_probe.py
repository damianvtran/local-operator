"""Ask revision-window probe (design §10 / #1936, the consumption bound).

Measures, over a REAL ``Session``, how the revision window closes in the
pre-fix and post-fix trees:

  mid-turn   a long turn held open by a gated provider stream; the answer
             parks on the steering queue; a revision is attempted {0.05, 0.5,
             2, 10}s after the answer; then the boundary append lands.
  idle       the delivery turn the answer spawns is held off with
             ``_turn_lock`` — answered, handed, nothing durable, which is the
             window's subject; a revision per delay; then the append.
  stopped    NO running loop at all (a cold answer): answer + revision are log
             writes; the next boot's reconcile delivers. The +60s is a clock
             shift on the queue's reader, not a sleep.
  width      NATURAL idle width, no holds: answer -> first durable row in ms,
             five reps (design §7.2: the fix does not widen this; report raw).
  control    a post-append revision must be REFUSED in BOTH trees, so an
             "always accepts" tree cannot pass this probe (hard assert).

WHY THE HOLDS IN THE MATRIX: the fix removes the ACK-close; it does not widen
the idle window — an idle session's spawned delivery turn consumes in
milliseconds (§7.2) — so the matrix holds the delivery off to stand in
"answered but not consumed", and the natural width is measured separately and
reported raw.

HOW TO RUN (isolated; never the operator's real home or sessions):

    ISO=$(mktemp -d "$LOCAL_OPERATOR_SCRATCHPAD/probe.XXXX")
    env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/config" PATH="$PATH" \
        TERM=xterm-256color [LO_PROBE_EXPECT=<tree-substring>] \
        <tree>/.venv/bin/python scripts/ask_revision_window_probe.py

The before-tree run: copy this script into that tree and run it from there
with ``PYTHONPATH=<that tree>`` (the script's own path bootstrap then resolves
``local_operator`` to the tree under test; ``LO_PROBE_EXPECT`` asserts it).
The probe prints and checks ``local_operator.__file__`` at start, refuses to
run without a redirected HOME/config, and fails (exit 1) if any control-arm
revision is accepted.
"""

from __future__ import annotations

import asyncio
import os
import pwd
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_operator  # noqa: E402
from local_operator.asks import store  # noqa: E402
from local_operator.harness.types import (  # noqa: E402
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.session.runtime.serving import ServingSessionHandle  # noqa: E402
from local_operator.session.session import Session  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)
DELAYS_S = (0.05, 0.5, 2.0, 10.0)
WIDTH_REPS = 5
#: Generous for a loaded host; every wait is on an event the code publishes.
WAIT_S = 30.0

_FAILED = False


def _fail(message: str) -> None:
    global _FAILED
    _FAILED = True
    print(f"PROBE-FAIL: {message}", flush=True)


# ---------------------------------------------------------------------------
# isolation and the tree under test
# ---------------------------------------------------------------------------


def _check_environment() -> None:
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    home = Path(os.environ.get("HOME", "")).resolve()
    config = os.environ.get("LOCAL_OPERATOR_CONFIG_DIR", "")
    print(f"local_operator.__file__ = {local_operator.__file__}", flush=True)
    expect = os.environ.get("LO_PROBE_EXPECT", "")
    if expect and expect not in str(local_operator.__file__):
        _fail(f"local_operator resolves to {local_operator.__file__!r}, not under {expect!r}")
    leaked = sorted(k for k in os.environ if k.startswith(("CMUX_", "LOP_")))
    if leaked:
        _fail(f"inherited session variables: {leaked}")
    if home == real_home:
        _fail("HOME is not redirected; run under a fresh HOME (AGENTS.md, isolated runs)")
    if not config or Path(config).resolve() == (real_home / ".local-operator").resolve():
        _fail("LOCAL_OPERATOR_CONFIG_DIR must be a fresh scratch root, not the real store")
    if not os.environ.get("LO_PROBE_SCRATCH"):
        _fail("LO_PROBE_SCRATCH is not set; scratch must live in the session scratchpad")
    print(
        f"isolation: HOME={home} CONFIG={config} "
        f"cmux_vars={sum(k.startswith('CMUX_') for k in os.environ)} "
        f"lop_vars={sum(k.startswith('LOP_') for k in os.environ)}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------


async def _wait_for(predicate: Any, timeout: float = WAIT_S) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("probe wait timed out")
        await asyncio.sleep(0.002)


async def _drain(session: Session) -> None:
    """Run every spawn the session made to completion, oldest first."""
    for _ in range(20):
        pending = [t for t in list(session._background_tasks) if not t.done()]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)
    raise AssertionError("probe drain never settled")


def _cell_dir(base: Path, name: str) -> Path:
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_session(cell_dir: Path, *, gate: asyncio.Event | None) -> tuple[Session, Any]:
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(cell_dir / "config")

    def stream(request: Any, signal: Any) -> Any:
        async def gen() -> Any:
            yield StreamTextDelta(delta="probe")
            if gate is not None:
                await gate.wait()
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    transcript = Transcript(cell_dir / "sess")
    session = Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=lambda: ["probe", "env"],
    )

    async def handler(questions: Any) -> Any:
        return None

    session.set_ask_handler(handler)
    return session, session.ask_queue()


def _enqueue(queue: Any) -> str:
    outcome = queue.enqueue(
        [
            {
                "id": "q0",
                "question": "Ship it?",
                "options": [],
                "multi": False,
                "secret": False,
                "persist": False,
                "recommended": None,
            },
            {
                "id": "q1",
                "question": "And when?",
                "options": [],
                "multi": False,
                "secret": False,
                "persist": False,
                "recommended": None,
            },
        ],
        None,
    )
    if not outcome.get("ok"):
        raise AssertionError(f"enqueue refused: {outcome}")
    return str(outcome["details"]["ask_id"])


def _durable(session: Session, row_id: str) -> bool:
    return session._transcript.has_entry(row_id)


def _handed(queue: Any, row_id: str) -> bool:
    return row_id in getattr(queue, "_handed", set())


def _landed_answers(session: Session, row_id: str) -> dict[str, Any] | None:
    for entry in session._transcript.entries():
        if entry.id == row_id and entry.payload.get("custom_type") == "ask_response":
            return dict(entry.payload.get("details", {}).get("answers") or {})
    return None


async def _attempt_revise(
    session: Session, queue: Any, handle: Any, ask_id: str, answers: dict[str, list[str]]
) -> tuple[str, str, str]:
    """``(outcome, reason, landed)`` for one revision attempt plus its landing.

    One attempt, then enough waiting for its effect to land (if accepted):
    release holds are the caller's job — this returns the raw facts only.
    """
    try:
        outcome = await handle.ask_revise(ask_id, answers, by="probe")
    except ValueError as exc:
        return "refused", str(exc), "n/a"
    if outcome not in ("revised", "answered"):
        return f"odd:{outcome}", "", "n/a"
    return "accepted", "", ""


def _result_line(
    arm: str,
    delay: float | None,
    outcome: str,
    reason: str,
    durable: bool,
    handed: bool,
    landed: str,
    control: str,
) -> None:
    delay_txt = "n/a  " if delay is None else f"{delay:5.2f}"
    reason_txt = f'  reason="{reason}"' if reason else ""
    print(
        f"arm={arm:8} delay={delay_txt}s  revise={outcome}  durable_at_attempt={durable}"
        f"  handed_at_attempt={handed}  landed={landed}  control={control}{reason_txt}",
        flush=True,
    )


async def _control_after_append(handle: Any, ask_id: str) -> str:
    """The control arm: a consumed row must refuse a revision, in EITHER tree."""
    try:
        await handle.ask_revise(ask_id, {"q0": ["post-append"], "q1": ["post-append"]}, by="probe")
    except ValueError as exc:
        if "already delivered" in str(exc):
            return "refused"
        return f"refused(other):{exc}"
    return "ACCEPTED"


async def run_mid_turn(base: Path) -> None:
    """A revision seconds into a running turn: the window must be the append."""
    print("\n-- mid-turn: long turn held open; answer steers; revise at +delay --", flush=True)
    for delay in DELAYS_S:
        gate = asyncio.Event()
        session, queue = _build_session(_cell_dir(base, f"mid-turn-{delay}"), gate=gate)
        loop = asyncio.get_running_loop()
        handle = ServingSessionHandle(session, loop, cwd=str(base))
        ask_id = _enqueue(queue)
        row_id = store.response_row_id(ask_id)
        turn = asyncio.create_task(session.prompt("probe: hold this turn open"))
        try:
            await _wait_for(lambda: session.is_streaming)
            # The answer mid-turn parks on the steering queue; wait for that
            # hand-off so the attempt below is against "answered, undelivered"
            # rather than against a scheduler race.
            answered = session.respond_ask(ask_id, {"q0": ["no"], "q1": ["maybe"]}, by="terminal")
            if not answered.get("ok"):
                _fail(f"mid-turn arm: the answer was refused: {answered}")
            await _wait_for(lambda: not session._steering_queue.empty())
            await asyncio.sleep(delay)  # the operator's matrix: a real interval
            durable = _durable(session, row_id)
            handed = _handed(queue, row_id)
            outcome, reason, _ = await _attempt_revise(
                session, queue, handle, ask_id, {"q0": ["yes"], "q1": ["maybe"]}
            )
            # Cross the boundary; the append is what lands.
            gate.set()
            await turn
            landed_answers = _landed_answers(session, row_id)
            landed = (
                "revised"
                if landed_answers == {"q0": ["yes"], "q1": ["maybe"]}
                else ("original" if landed_answers == {"q0": ["no"], "q1": ["maybe"]} else "other")
            )
            control = await _control_after_append(handle, ask_id)
            _result_line("mid-turn", delay, outcome, reason, durable, handed, landed, control)
            if control != "refused":
                _fail("mid-turn control: a consumed row accepted a revision")
        finally:
            if not turn.done():
                turn.cancel()
            await session.dispose()


async def run_idle(base: Path) -> None:
    """The idle session, its delivery held off: answered, handed, not durable."""
    print("\n-- idle: delivery turn held on _turn_lock; revise at +delay --", flush=True)
    for delay in DELAYS_S:
        session, queue = _build_session(_cell_dir(base, f"idle-{delay}"), gate=None)
        loop = asyncio.get_running_loop()
        handle = ServingSessionHandle(session, loop, cwd=str(base))
        ask_id = _enqueue(queue)
        row_id = store.response_row_id(ask_id)
        await session._turn_lock.acquire()
        try:
            session.respond_ask(ask_id, {"q0": ["no"], "q1": ["maybe"]}, by="terminal")
            await _wait_for(lambda: _handed(queue, row_id))
            await asyncio.sleep(delay)
            durable = _durable(session, row_id)
            handed = _handed(queue, row_id)
            outcome, reason, _ = await _attempt_revise(
                session, queue, handle, ask_id, {"q0": ["yes"], "q1": ["maybe"]}
            )
        finally:
            session._turn_lock.release()
        await _drain(session)
        landed_answers = _landed_answers(session, row_id)
        landed = (
            "revised"
            if landed_answers == {"q0": ["yes"], "q1": ["maybe"]}
            else ("original" if landed_answers == {"q0": ["no"], "q1": ["maybe"]} else "other")
        )
        control = await _control_after_append(handle, ask_id)
        _result_line("idle", delay, outcome, reason, durable, handed, landed, control)
        if control != "refused":
            _fail("idle control: a consumed row accepted a revision")
        await session.dispose()


def _settle_stray_schedules(session: Session) -> None:
    """Cancel tasks a no-loop phase left on a never-started policy loop.

    CPython's policy still hands plain ``ensure_future`` a fresh loop, so a
    cold answer's attempted schedules (wake arm/retire, reconcile kicks) sit
    unrun there — which IS "stopped". Settle them so the boot below owns the
    one loop that executes.
    """
    stray_loops = {t.get_loop() for t in session._background_tasks if not t.done()}
    for task in list(session._background_tasks):
        task.cancel()
    for stray in stray_loops:
        if not stray.is_running() and not stray.is_closed():
            stray.run_until_complete(asyncio.sleep(0))
            stray.close()


def run_stopped(base: Path) -> None:
    """The kept case: answered while no loop runs; the boot reconcile delivers."""
    print("\n-- stopped: no running loop at the answer; boot reconcile delivers --", flush=True)
    session, queue = _build_session(_cell_dir(base, "stopped"), gate=None)
    ask_id = _enqueue(queue)
    row_id = store.response_row_id(ask_id)
    answered = session.respond_ask(ask_id, {"q0": ["no"], "q1": ["maybe"]}, by="terminal")
    if not answered.get("ok"):
        _fail(f"stopped arm: the answer was refused: {answered}")
    # The +60s: a reader-clock shift, not a sleep — the property is that
    # elapsed time alone does not close the window (design §10).
    shifted = int(time.time() * 1000) + 60_000
    queue._now = lambda: shifted
    revised = session.revise_ask(ask_id, {"q0": ["yes"], "q1": ["maybe"]}, by="probe")
    outcome = "accepted" if revised.get("ok") else "refused"
    reason = "" if revised.get("ok") else str(revised.get("error"))
    _settle_stray_schedules(session)

    async def boot() -> None:
        await session.reconcile_asks(load_time=True)
        await _drain(session)

    asyncio.run(boot())
    landed_answers = _landed_answers(session, row_id)
    landed = (
        "revised"
        if landed_answers == {"q0": ["yes"], "q1": ["maybe"]}
        else ("original" if landed_answers == {"q0": ["no"], "q1": ["maybe"]} else "other")
    )
    _result_line("stopped", None, outcome, reason, False, False, landed, "n/a")
    if not revised.get("ok") or landed != "revised":
        _fail("stopped arm: the kept case broke (the revision must land)")


async def run_width(base: Path) -> None:
    """The NATURAL idle width: answer -> first durable row, reported raw."""
    print("\n-- width: answer -> first durable row (no holds, five reps) --", flush=True)
    session, queue = _build_session(_cell_dir(base, "width"), gate=None)
    loop = asyncio.get_running_loop()
    handle = ServingSessionHandle(session, loop, cwd=str(base))
    widths: list[float] = []
    for rep in range(WIDTH_REPS):
        ask_id = _enqueue(queue)
        row_id = store.response_row_id(ask_id)
        started = loop.time()
        session.respond_ask(ask_id, {"q0": ["yes"], "q1": ["maybe"]}, by="terminal")
        await _wait_for(lambda: _durable(session, row_id))
        width_ms = (loop.time() - started) * 1000.0
        widths.append(width_ms)
        control = await _control_after_append(handle, ask_id)
        _result_line("width", None, "n/a", "", False, False, "original", control)
        print(f"  width[rep{rep + 1}] = {width_ms:.3f} ms", flush=True)
        if control != "refused":
            _fail("width control: a consumed row accepted a revision")
        await _drain(session)
    ordered = sorted(widths)
    print(
        f"  width summary: min={ordered[0]:.3f} ms median={ordered[len(ordered) // 2]:.3f} ms "
        f"max={ordered[-1]:.3f} ms n={len(ordered)}",
        flush=True,
    )
    await session.dispose()


def main() -> int:
    print("== ask revision-window probe (design §10, consumption bound; #1936) ==", flush=True)
    _check_environment()
    scratch = Path(os.environ["LO_PROBE_SCRATCH"])
    base = Path(tempfile.mkdtemp(prefix="ask-window-probe-", dir=scratch))
    print(f"scratch base: {base}", flush=True)
    asyncio.run(run_mid_turn(base))
    asyncio.run(run_idle(base))
    run_stopped(base)
    asyncio.run(run_width(base))
    print(f"\nprobe complete; failed={_FAILED}", flush=True)
    if not _FAILED:
        shutil.rmtree(base, ignore_errors=True)
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

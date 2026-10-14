"""The queued move end to end: two devices, a live runtime, a viewer, one wake.

THE FOUR THINGS ONLY THIS CELL CAN SHOW (the parts are unit-covered elsewhere):

1. the PHASE RUN a person watches — ``queued → finishing → paused → copying →
   resumed`` — with each phase written by the party that owns it (the source
   relay's driver, the runtime's pause claim, the commit);
2. the MID-TURN WAIT: while a turn is in flight the queue sits at ``finishing``
   and does NOT pause (the design refuses to drain turns), and the pause
   happens at the boundary the turn's end creates;
3. the ATTACHED CLIENT'S exit: it hears ``move_pending`` naming the destination
   while the runtime is still alive, and its disconnect reads as the MOVE
   (``MOVED_REASON``) — the arm that must NOT re-engage a successor locally,
   because this conversation is leaving rather than refreshing;
4. the WAKE CARRY: the source's index is pruned at commit, the destination's is
   rebuilt from the copied transcript, the destination's supervisor scan picks
   the carried row up and fires it (the source, pruned, fires nothing), and the
   row's ``next_due_at``/``fired_count`` arrive exactly as stored (F5's
   no-consume property at the data level; the refusal path's own cell lives in
   ``test_carry.py``).

Two local config roots on one host — the mobility suite's own topology, which
the design names as the fallback for the two-machine drill: it loses SSH
transport coverage and keeps every state-carry-over and queue property here.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.mobile.attach_client import (
    MOVED_REASON,
    AttachClient,
    find_runtime_record,
)
from local_operator.network import carry, mobility, move_queue
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.types import VIEWED_MOVE_REFUSAL
from local_operator.wakes import store as wake_store
from tests.e2e.harness import build_session
from tests.unit.network.test_carry import MONITOR_ROW, WAKE_ROW, _custom_entry
from tests.unit.network.test_mobility import (  # noqa: F401 — fixtures and helpers
    SESSION,
    _owned_session,
    pair,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — the fixture `pair` reaches for
    _pair_settled,
    devices,
)

#: B's display name as ``identity.mint`` sets it — what ``move_pending.to`` and
#: the retiring frame's ``to`` must carry for the client to know where to follow.
IDENTITY_B_NAME = "device-b"

PHASES_IN_ORDER = ["queued", "finishing", "paused", "copying", "resumed"]


class _GatedStream:
    """One model turn that HOLDS THE SESSION BUSY until the test releases it.

    The queued move exists for a busy session, so the e2e must actually be busy:
    the first delta goes out (the turn is live), and the end event waits on the
    gate — which the test opens only after it has watched the queue sit at
    ``finishing``. The contract is the harness's own (a callable returning an
    async iterator), reduced to exactly what this cell needs.
    """

    def __init__(self) -> None:
        self.gate = asyncio.Event()

    def __call__(self, request: Any, signal: Any = None) -> Any:
        async def gen() -> Any:
            from local_operator.harness.types import StreamEndEvent, StreamTextDelta

            yield StreamTextDelta(delta="working")
            await self.gate.wait()
            yield StreamEndEvent(stop_reason="stop")

        return gen()


async def _await_phase(
    root: Path, session_id: str, phase: str, timeout: float = 20.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = move_queue.read_record(root, session_id) or {}
        if str(record.get("phase") or "") == phase:
            return record
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"never reached {phase!r}; record now: {move_queue.read_record(root, session_id)}"
    )


async def _wait(predicate: Any, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


@pytest.mark.asyncio
async def test_the_queued_move_pauses_at_the_boundary_carries_the_wake_and_lets_the_viewer_go(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    paired = request.getfixturevalue("pair")
    server_a, server_b = paired[0], paired[1]
    _pair_settled(paired, monkeypatch, role="admin")

    # ---- the source's session: a transcript carrying its wake and monitor ----
    session_dir = _owned_session(server_a, SESSION)
    with (session_dir / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(_custom_entry("wake_schedules", {"schedules": [dict(WAKE_ROW)]}) + "\n")
        handle.write(_custom_entry("monitor_schedules", {"monitors": [dict(MONITOR_ROW)]}) + "\n")
    # THE SESSION'S OWNERSHIP CLAIM, taken the way a production runtime's engage
    # takes it: ``resume.live_runtime_pid`` reads the ``.session.pid`` mirror, so
    # without it ``find_runtime_record`` can never resolve the in-process
    # runtime's discovery record (the driver's delivery needs exactly that
    # resolution, so the test must not fake it). It is RELEASED at the departure
    # step below, which is the half a real runtime supplies by exiting.
    from local_operator.session_lease import acquire_session_lease

    lease = acquire_session_lease(session_dir)
    # The supervisor installer shells out to launchctl/systemd; a test must never
    # install a unit. The promote still takes the §5.3 `ensure` step on this stub,
    # which answers the verify shape (`running`): the carried-wake receipt half is
    # exercised by ``test_move_carry_supervisor`` on a fake installer instead.
    monkeypatch.setattr(
        carry,
        "ensure_supervisor",
        lambda root: {"installed": True, "running": True, "detail": "stubbed"},
    )
    carry.rebuild_indexes(server_a.root, SESSION)
    from local_operator.monitors import state as monitor_state

    state_dir = monitor_state.state_dir(server_a.root, SESSION)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "m1.json").write_text("{}", encoding="utf-8")
    assert wake_store.read_entry(server_a.root, SESSION) is not None, "the source index is missing"

    # The attach window is what makes the client story observable in test time;
    # the key is the product's own (``network.move.attach_window_s``, §5.4).
    # READ-MERGE-WRITE rather than a blind overwrite: pairing already wrote this
    # file, and clobbering it would pull settings out from under components that
    # re-read it (the window reader is one; a relay restart is another).
    import yaml

    from local_operator.config import CONFIG_FILE_NAME

    config_path = server_a.root / CONFIG_FILE_NAME
    config_doc = {}
    if config_path.is_file():
        config_doc = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    # EVERY SETTING LIVES UNDER ``values:`` — a top-level ``network`` key is
    # invisible to ``ConfigManager`` (and the store says so on stderr, which
    # this cell must not be adding).
    values = config_doc.setdefault("values", {})
    values.setdefault("network", {}).setdefault("move", {})["attach_window_s"] = 0.4
    config_path.write_text(yaml.safe_dump(config_doc, sort_keys=False), encoding="utf-8")

    # ---- a live runtime for A on THIS loop, holding a turn ----
    stream = _GatedStream()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    session = build_session(session_dir, stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(session_dir))
    runtime = RuntimeServer(handle, kind="tui")
    await runtime.start_in_process()
    assert await runtime.wait_until_published()

    moves: list[dict[str, Any]] = []
    disconnects: list[str] = []
    retirings: list[dict[str, Any]] = []
    viewer = AttachClient(
        lambda _projection: None,
        lambda reason: disconnects.append(reason),
        on_retiring=lambda frame: retirings.append(dict(frame)),
        locality="local",
        surface="terminal",
        on_move_pending=lambda frame: moves.append(dict(frame)),
    )
    #: Bound before the cell's try so the teardown can be written for the case
    #    where setup fails before the turn exists (pyright's "possibly unbound"
    #    is the same reading).
    turn: asyncio.Task[Any] | None = None
    try:
        record, _pid = find_runtime_record(server_a.root, SESSION)
        assert record is not None, "the runtime published no record to attach to"
        await viewer.connect(record, SESSION)

        turn = asyncio.create_task(handle.prompt("hold the session busy", wait_complete=True))
        assert await _wait(lambda: handle.is_busy()), "the turn never opened"

        # ---- the queued move, issued from B the way its CLI would ----
        result = await asyncio.to_thread(
            mobility.request_move, SESSION, to="local", queue=True, root=server_b.root
        )
        payload = dict(result)
        assert payload.get("ok"), payload
        queue_block = payload.get("queue")
        assert isinstance(queue_block, dict), payload
        assert queue_block["to_device"] == server_b.identity.device_id
        await _await_phase(server_a.root, SESSION, "finishing")

        # THE MID-TURN WAIT: the queue must sit at ``finishing`` while the turn
        # runs. A pause here would be the drain the design refuses.
        await asyncio.sleep(1.2)
        held = move_queue.read_record(server_a.root, SESSION) or {}
        assert held.get("phase") == "finishing", held
        assert moves == [], "the move announced itself while the session was still busy"

        # ---- the boundary: release the turn; the move takes it ----
        stream.gate.set()
        await turn
        assert await _wait(lambda: not handle.is_busy())
        await _await_phase(server_a.root, SESSION, "paused", timeout=15.0)
        assert await _wait(
            lambda: moves and str(moves[0].get("to") or "") == IDENTITY_B_NAME
        ), f"the viewer never heard where the conversation was going: {moves}"

        # THE RETIRE ANNOUNCES ITSELF before the EOF (the window exists for
        # exactly this): wait for the frame, or the departure below races it and
        # the viewer reads a bare exit instead of the move it was told about.
        assert await _wait(
            lambda: retirings and str(retirings[0].get("reason")) == "moved"
        ), f"the viewer never heard the retirement: {retirings}"
        assert str(retirings[0].get("to") or "") == IDENTITY_B_NAME, retirings

        # ---- the runtime departs: the half a real process supplies by exiting ----
        # In production the process running the runtime exits here and the kernel
        # drops its claim. In-process the test performs both halves in order:
        # teardown (unpublish the record; the viewer already has the retiring
        # frame, so its socket EOF reads as the move), then release the lease.
        # Only then may the driver see "no writer" and start the copy — which is
        # the ordering contract this cell exists to hold the system to.
        await runtime.aclose()
        lease.release()

        # ---- the copy, the commit, the promote ----
        final = await _await_phase(server_a.root, SESSION, "resumed", timeout=90.0)
        history = [str(stamp.get("phase") or "") for stamp in final.get("phases") or []]
        for phase in PHASES_IN_ORDER:
            assert phase in history, f"{phase!r} missing from the phase history {history}"
        positions = [history.index(phase) for phase in PHASES_IN_ORDER]
        assert positions == sorted(positions), f"the phases ran out of order: {history}"

        # THE VIEWER: told where it went, and disconnected as a MOVE — the arm
        # that does not re-engage a successor locally.
        assert await _wait(lambda: disconnects, timeout=15.0), "the viewer was never disconnected"
        assert disconnects[0] == MOVED_REASON, disconnects
        assert viewer.moved_to == IDENTITY_B_NAME, viewer.moved_to

        # ---- the state carry-over, source side: pruned at COMMIT ----
        assert wake_store.read_entry(server_a.root, SESSION) is None, "the source index survived"
        assert not state_dir.exists(), "the source monitor state survived the prune"

        # ---- the state carry-over, destination side: rebuilt from the copy ----
        # THE PROMOTE IS B'S OWN STEP and it runs AFTER the source's commit — the
        # one `resumed` was awaited on above is stamped by the SOURCE at its
        # commit (``move_queue.note_committed``), the "[commit → promote-rebuild]
        # gap" ``network/carry.py`` names on purpose. B's integrate (the cold
        # index rebuild included) runs on B's pull thread a beat later, so
        # reading the index straight off `resumed` raced it: observed on CI
        # (`test (3.12, 2)`, "the destination index was not rebuilt") with a
        # passing rerun, and reproduced here by delaying only that rebuild.
        #
        # Wait on B's own `committed` stamp, not on the index: ``_destination_move``
        # notes it only AFTER ``_promote`` returns, and ``_promote`` runs
        # ``rebuild_indexes`` inline. So once it is set the rebuild has already
        # happened, and the read below is the product claim itself ("rebuilt by the
        # time B reports the move committed") with no timing in it. Waiting on the
        # index file instead would also pass a rebuild that landed late.
        def destination_phases() -> list[str]:
            return [
                str(stamp.get("phase")) for stamp in mobility.progress_for(server_b).phases(SESSION)
            ]

        # The message is built only when the assert fails, so it reads B's phases
        # AT the timeout: a stall shows how far B got (e.g. no phase at all vs
        # stuck short of `committed`) rather than just that it never finished.
        assert await _wait(
            lambda: "committed" in destination_phases(), timeout=30.0
        ), f"the destination never reported the move committed (its phases: {destination_phases()})"
        destination_entry = wake_store.read_entry(server_b.root, SESSION)
        assert destination_entry is not None, "the destination index was not rebuilt"
        carried = destination_entry["schedules"][0]
        # AS STORED (§5.3): the destination's index equals the COPIED TRANSCRIPT's
        # own row — the copy's source of truth — so whatever the source's store
        # last wrote is what B serves, with no recomputation on the way.
        stored = carry.latest_custom_details(server_b.root / "sessions" / SESSION, "wake_schedules")
        stored_row = dict((stored or {}).get("schedules", [{}])[0])
        assert carried["next_due_at"] == stored_row["next_due_at"], (carried, stored_row)
        assert carried["fired_count"] == stored_row["fired_count"], (carried, stored_row)
        # NO CONSUME, at the data level: the count is the source's arm-time value
        # or later — never reset to a fresh 0 by the carry.
        assert carried["fired_count"] >= int(WAKE_ROW["fired_count"]), carried

        # ---- the destination's supervisor fires it exactly once ----
        engagements: list[str] = []

        async def fake_engage(
            session_id: str, cwd: str, work: Any, *, config_dir: Path, **kwargs: Any
        ) -> None:
            engagements.append(session_id)

        monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)
        live = {"value": False}

        async def has_live(config_dir: Path, session_id: str) -> bool:
            return live["value"]

        monkeypatch.setattr("local_operator.wakes.supervisor._has_live_runtime", has_live)
        from local_operator.wakes.supervisor import fire_due_wakes

        # ITS DUE TIME, SYNTHESISED. The row is armed an hour out (so nothing on
        # the source fires it while the move runs), and B's scan is taken one
        # second past the stamp — "fires on the destination's first eligible
        # tick" without a real clock wait.
        now_ms = int(carried["next_due_at"]) + 1000
        # One tick: the scan sees the rebuilt index and fires the due row.
        assert await fire_due_wakes(server_b.root, now_ms=now_ms) == 1
        assert engagements == [SESSION], engagements
        # And ONCE means once: with the runtime that took it now live, the
        # supervisor stands down for this session...
        live["value"] = True
        assert await fire_due_wakes(server_b.root, now_ms=now_ms) == 0
        assert engagements == [SESSION], engagements
        # ...and the SOURCE, pruned, cannot fire it at all — the other half of
        # the no-double-fire ordering (§5.3).
        live["value"] = False
        assert await fire_due_wakes(server_a.root, now_ms=now_ms) == 0
        assert engagements == [SESSION], engagements
    finally:
        viewer.close()
        if turn is not None and not turn.done():
            stream.gate.set()
            turn.cancel()
        await session.dispose()
        # ``aclose`` rather than ``close``: awaiting the teardown is what proves
        # the record is unpublished before the test ends (``close`` is safe twice
        # and joins the same task, but only the await makes the ordering a fact).
        await runtime.aclose()
        lease.release()


@pytest.mark.asyncio
async def test_a_bare_move_with_a_viewer_reports_viewed_elsewhere(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA round 1 Q1: the attached-viewer refusal is ``viewed_elsewhere``.

    A BARE move keeps its behaviour — refuse while an observer is attached —
    but the refusal's CODE carries the §5.4 split: the blocker is the VIEWER
    (waiting cannot clear it; the queue can), not a turn, so a notice can offer
    Queue vs Wait instead of "Wait for the turn to finish". Driven at the wire
    the way the CLI drives it (the destination's own relay call, a real
    runtime, a real registered viewer), which is the QA repro's own shape.
    """
    paired = request.getfixturevalue("pair")
    server_a, server_b = paired[0], paired[1]
    _pair_settled(paired, monkeypatch, role="admin")
    session_dir = _owned_session(server_a, SESSION)
    # The ownership claim the in-process runtime's record resolution needs
    # (``.session.pid`` is what ``live_runtime_pid`` reads); released in the
    # teardown, the half a real runtime's process exit supplies.
    from local_operator.session_lease import acquire_session_lease

    lease = acquire_session_lease(session_dir)

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    session = build_session(session_dir, _GatedStream())
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(session_dir))
    runtime = RuntimeServer(handle, kind="tui")
    await runtime.start_in_process()
    assert await runtime.wait_until_published()

    viewer = AttachClient(
        lambda _projection: None, lambda _reason: None, locality="local", surface="terminal"
    )
    try:
        record, _pid = find_runtime_record(server_a.root, SESSION)
        assert record is not None, "the runtime published no record to attach to"
        await viewer.connect(record, SESSION)

        result = await asyncio.to_thread(
            mobility.request_move, SESSION, to="local", root=server_b.root
        )
        payload = dict(result)
        assert payload.get("ok") is False, payload
        assert payload.get("code") == "viewed_elsewhere", payload
        assert VIEWED_MOVE_REFUSAL in str(payload.get("message")), payload
    finally:
        viewer.close()
        await session.dispose()
        await runtime.aclose()
        lease.release()

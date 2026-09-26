"""A desktop READ never needs an answering owner (design D1/D2/D9).

THE REPORTED FAILURE. With a live-but-silent owner — the pid alive, the
discovery record naming the session, the socket accepting the dial and then
answering nothing — a desktop read answered ``503 Session owner is unavailable``
after ~15 s, on ``/snapshot`` and on ``/history``. The same durable rows were
readable in 0.02 s with no owner at all, in the same process, on the same
session directory: the only difference between the two cases was a live process
that did not answer.

These pin the properties that remove it, at the facade the routes call:

* a READ attempt is bounded (``READ_ATTACH_BUDGET_S``), never raises for a
  session that exists on disk, and says WHY it is cold (D1/D3);
* the presence re-assert on the dial path is best-effort, so a lost one cannot
  fail a bind (D2.1);
* an authenticated dial is RETAINED past the read's envelope, so a sync that
  lands afterwards installs state and publishes the rollover the renderer
  already handles (D2.2) — and is abandoned at a hard deadline, because the
  socket is a residency term of the runtime's own exit predicate;
* a record that exists but is not ``live`` — a wedged heartbeat — is not read as
  "no runtime", and is dialled rather than spawned (D9);
* no read path spawns anything (D4).

The ROUTE-level half (the 200 the app actually receives, with its latency, and
the ``cold_reason`` tokens on the wire) lives in
``tests/unit/server/test_desktop_read_without_owner.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import Message
from local_operator.session import attached as remote_module
from local_operator.session.attached import READ_ATTACH_BUDGET_S, AttachedSession
from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendSync,
    sync_wire_payload,
)
from local_operator.session.runtime import launch, registry
from local_operator.session.runtime.types import DESKTOP_WATCH_CAPABILITY, SessionRecord
from local_operator.session.transcript import Transcript

SESSION_ID = "abcdef123456"
#: The marker and the record must name a pid that is ALIVE, because that is the
#: liveness the registry's classification and ``live_runtime_pid`` read. This
#: process is alive for the whole test, and no socket is ever served by it —
#: the fake owner below is what answers, exactly as the architect's probe did.
OWNER_PID = os.getpid()
#: Upper bound on an awaited event, never a budget to sleep through.
DEADLOCK_GUARD_S = 10.0


async def _never_take_over() -> None:
    raise AssertionError("a desktop viewer must never take over a runtime")


class _FakeOwner:
    """A loopback owner that answers exactly what a test tells it to.

    The frames are the real wire shapes, because the facade's dial reads them:
    the welcome projection is the identity check, and SILENCE after it is the
    state that used to fail a read. The two halves of the dial are controlled
    independently on purpose — ``answer_watch`` is the presence re-assert,
    ``sync_on_connect`` is the canonical state — because the whole point of D2.1
    is that only the second one matters to a reader.

    ``mute`` is the OTHER silent-owner shape, and it is not the same leg: the
    owner here never writes a welcome at all, which is what a SIGSTOPped runtime
    looks like from the socket — the kernel accepts the connection and nothing
    ever comes back. A dial that answers cannot fail an unbounded wait, so a
    fake that always welcomes cannot catch a read whose budget does not cover
    the welcome (review round 1, MAJOR-1).

    ``answer_prompts``/``park_prompts``/``stand_down_on_prompt`` are the
    ADMISSION leg, added for the submit-acknowledgement tests: a ``prompt``/
    ``steer`` op is recorded on every arrival, held on ``park_prompts`` when one
    is given, acknowledged only when ``answer_prompts`` is set, and answered with
    a CLOSED SOCKET when ``stand_down_on_prompt`` is set (a runtime that leaves
    mid-admission). A parked admission is what lets a test
    assert that the host told its viewer something BEFORE the owner heard the
    turn — the ordering the acknowledgement exists for — without any timing
    bound standing in for the fact.
    """

    def __init__(
        self,
        session_id: str,
        cwd: Path,
        *,
        answer_watch: bool = False,
        sync_on_connect: bool = False,
        mute: bool = False,
        welcome_session_id: str | None = None,
        answer_prompts: bool = False,
        park_prompts: asyncio.Event | None = None,
        stand_down_on_prompt: bool = False,
    ) -> None:
        self.session_id = session_id
        self.welcome_session_id = welcome_session_id or session_id
        self.cwd = str(cwd)
        self.answer_watch = answer_watch
        self.sync_on_connect = sync_on_connect
        self.mute = mute
        self.answer_prompts = answer_prompts
        self.park_prompts = park_prompts
        self.stand_down_on_prompt = stand_down_on_prompt
        #: Every admission the owner's socket received, in order.
        self.prompts: list[dict[str, Any]] = []
        #: How many admissions the owner stood down on (see the flag's note).
        self.stand_downs = 0
        self.port = 0
        self.conns = 0
        self.watch_calls = 0
        #: How many ``desktop_withdraw`` frames this owner received (round 3).
        #: A dial replays the recorded withdrawal instead of a lease when the
        #: pane left; counting them apart from ``watch_calls`` is what lets a
        #: test say WHICH arm of the replay ran.
        self.withdraw_calls = 0
        self._server: asyncio.AbstractServer | None = None
        self._writers: list[asyncio.StreamWriter] = []
        self._connections: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.conns += 1
        self._writers.append(writer)
        try:
            await reader.readline()  # the attach auth frame
            if self.mute:
                # Read for ever and write nothing: the connection stays OPEN (so
                # the dial is waiting on a live socket, not a closed one) and no
                # welcome ever arrives.
                while await reader.readline():
                    pass
                return
            await self._write(
                writer,
                {
                    "op": "welcome",
                    "data": {
                        "session_id": self.welcome_session_id,
                        "cwd": self.cwd,
                        "pid": 0,
                    },
                },
            )
            if self.sync_on_connect:
                # What a real runtime does immediately after the welcome
                # (``runtime/server.py``'s connect-time push).
                await self._write(writer, self._sync_frame())
            # Then answer nothing unless this owner was told to answer it.
            # Reading keeps the connection OPEN, so a retained dial has a live
            # socket to be abandoned on rather than a closed one.
            while True:
                line = await reader.readline()
                if not line:
                    return
                try:
                    frame = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    continue
                if frame.get("op") == "desktop_watch":
                    self.watch_calls += 1
                    if self.answer_watch:
                        await self._write(
                            writer, {"op": "ack", "req": frame.get("req"), "data": {}}
                        )
                elif frame.get("op") == "desktop_withdraw":
                    # Answered under the same flag: both ops are presence hints
                    # with the same best-effort envelope on the dial.
                    self.withdraw_calls += 1
                    if self.answer_watch:
                        await self._write(
                            writer, {"op": "ack", "req": frame.get("req"), "data": {}}
                        )
                elif frame.get("op") in ("prompt", "steer"):
                    # RECORDED FIRST, so a test reading this list knows the
                    # admission reached the socket even when it is held below.
                    self.prompts.append(frame)
                    if self.stand_down_on_prompt:
                        # A runtime that LEAVES mid-admission: the turn reached a
                        # live owner, and then the socket went away before any
                        # answer. This is the refusal a caller sees as
                        # ``runtime_unreachable``, and the one that used to leave
                        # an acknowledgement with no outcome.
                        self.stand_downs += 1
                        writer.close()
                        try:
                            await writer.wait_closed()
                        except (ConnectionError, OSError):  # noqa: BLE001 — closing
                            pass
                        return
                    if self.park_prompts is not None:
                        await self.park_prompts.wait()
                    if self.answer_prompts:
                        await self._write(
                            writer,
                            {
                                "op": "ack",
                                "req": frame.get("req"),
                                "detail": (
                                    "steering queued"
                                    if frame.get("op") == "steer"
                                    else "prompt admitted"
                                ),
                                "duplicate": False,
                            },
                        )
        except (ConnectionError, BrokenPipeError, asyncio.IncompleteReadError):
            return

    def _sync_frame(self, *, epoch: str = "fake-owner", sequence: int = 1) -> dict[str, Any]:
        sync = _frontend_sync(self.session_id, Path(self.cwd), epoch=epoch, sequence=sequence)
        return {"op": "frontend_sync", "data": sync_wire_payload(sync)}

    async def send_sync(self, *, epoch: str = "fake-owner", sequence: int = 1) -> None:
        """Push the canonical sync a viewer is waiting for, on demand."""
        for writer in list(self._writers):
            await self._write(writer, self._sync_frame(epoch=epoch, sequence=sequence))

    async def _write(self, writer: asyncio.StreamWriter, frame: dict[str, Any]) -> None:
        writer.write(json.dumps(frame).encode() + b"\n")
        await writer.drain()

    async def stop(self) -> None:
        for writer in self._writers:
            writer.close()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        self._writers.clear()


async def _seed(root: Path, *, session_id: str = SESSION_ID, rows: int = 1) -> None:
    """The durable answer every read is entitled to.

    Parameterised by session rather than fixed to this module's id so the ROUTE
    tests (``tests/unit/server/test_desktop_read_without_owner.py``) can drive the
    same owner fixture against a session the pool minted for them.
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory)
    for index in range(rows):
        await transcript.append_message(Message.user(f"durable question {index}"))
        await transcript.append_message(Message.assistant(f"durable answer {index}"))


def _frontend_sync(
    session_id: str = SESSION_ID,
    cwd: Path | None = None,
    *,
    epoch: str = "fake-owner",
    sequence: int = 1,
) -> FrontendSync:
    """A canonical sync for this session, shaped as the owner sends it.

    Shared by the fake owner's wire frame and by the tests that drive the
    landing task directly, so the two cannot drift into disagreeing about what
    "a sync arrived" means.
    """
    return FrontendSync(
        epoch=epoch,
        sequence=sequence,
        snapshot=FrontendSessionState(
            session_id=session_id, epoch=epoch, cwd=str(cwd) if cwd is not None else ""
        ),
    )


def _record(session_id: str, port: int, *, cwd: Path) -> SessionRecord:
    return SessionRecord(
        pid=OWNER_PID,
        kind="tui",
        session_id=session_id,
        conversation_name="ownerless read",
        cwd=str(cwd),
        model_label="test/model",
        control_port=port,
        control_key="synthetic-key",
        protocol=5,
        capabilities=[DESKTOP_WATCH_CAPABILITY],
    )


def _publish_live(
    root: Path,
    owner: _FakeOwner,
    *,
    session_id: str = SESSION_ID,
    leaving: str = "",
) -> None:
    """A record `scan` calls live: pid alive, heartbeat now, dialable.

    ``leaving`` is the drain phrase the descriptor carries
    (``runtime.types.SessionRecord.leaving``), which is how a read tells a
    runtime that is finishing work in flight from one that is simply slow.
    """
    record = _record(session_id, owner.port, cwd=root)
    record.leaving = leaving
    registry.publish(record, root)
    (root / "sessions" / session_id / ".session.pid").write_text(str(OWNER_PID))


def _publish_wedged(root: Path, owner: _FakeOwner, *, session_id: str = SESSION_ID) -> None:
    """A record in the registry's THIRD state: pid alive, heartbeat stale.

    Written by hand rather than through :func:`registry.publish`, which stamps
    ``heartbeat_at`` with now — the staleness IS the state under test.
    """
    record = _record(session_id, owner.port, cwd=root)
    record.heartbeat_at = time.time() - (registry.HEARTBEAT_TIMEOUT_S + 5.0)
    registry.record_path(OWNER_PID, root).write_text(json.dumps(record.to_json()), encoding="utf-8")
    (root / "sessions" / session_id / ".session.pid").write_text(str(OWNER_PID))


async def _cold_viewer(root: Path, *, session_id: str = SESSION_ID) -> AttachedSession:
    return await AttachedSession.cold(
        session_id,
        config_dir=root,
        cwd=str(root),
        takeover_factory=_never_take_over,
        surface="desktop",
    )


async def _until(predicate: Any, *, why: str, timeout: float = DEADLOCK_GUARD_S) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, why
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_read_serves_cold_inside_its_budget_instead_of_raising(
    tmp_path: Path, monkeypatch
) -> None:
    """THE PROBE, as a test: a live owner that welcomes and never answers.

    Before the fix this call raised ``OwnerAckTimeout`` after 15.27 s out of
    ``_dial``'s presence re-assert, and the route ladder turned it into the 503
    the operator saw. What a read owes its caller is bounded patience and a cold
    answer, because the durable rows are already on disk.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path)
    await owner.start()
    _publish_live(tmp_path, owner)
    viewer = await _cold_viewer(tmp_path)

    started = time.monotonic()
    attached = await viewer.attach_existing(budget=READ_ATTACH_BUDGET_S)
    elapsed = time.monotonic() - started

    assert attached is False, "a read that did not sync is not an attached viewer"
    assert elapsed < READ_ATTACH_BUDGET_S + 1.0, f"a read waited {elapsed:.2f}s"
    assert owner.conns == 1, "the read did not dial the owner at all"
    assert viewer.is_cold is True
    assert viewer.cold_reason == "owner-silent", (
        "a pid holds the lease; reporting no-runtime is the false claim the "
        "renderer painted as a lost conversation"
    )
    assert viewer.attaching is True, "the retained dial must be announced"
    await viewer.dispose()
    await owner.stop()


@pytest.mark.asyncio
async def test_a_read_reports_no_runtime_only_when_there_is_none(
    tmp_path: Path, monkeypatch
) -> None:
    """The token comes from a fact, not from the absence of an answer."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    await _seed(tmp_path)
    viewer = await _cold_viewer(tmp_path)

    assert await viewer.attach_existing(budget=READ_ATTACH_BUDGET_S) is False
    assert viewer.cold_reason == "no-runtime"
    assert viewer.attaching is False
    await viewer.dispose()


@pytest.mark.asyncio
async def test_a_mute_owner_is_served_cold_inside_the_read_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MAJOR-1: the WELCOME leg is bounded too, not only the re-asserts.

    The silent owner above answers the dial and then stays quiet about the SYNC.
    This one never writes a welcome at all, which is what a wedged runtime looks
    like from the socket — the kernel accepts the connection and nothing ever
    comes back (the shape the e2e harness names: "A SIGSTOPped runtime cannot
    answer its socket"). Budgeting only what runs AFTER ``connect`` returns left
    this leg governed by ``ACK_TIMEOUT_S`` (15 s), so the read served cold at the
    right status and the wrong latency, the same order as the reported failure.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path, mute=True)
    await owner.start()
    _publish_live(tmp_path, owner)
    viewer = await _cold_viewer(tmp_path)

    started = time.monotonic()
    attached = await viewer.attach_existing(budget=READ_ATTACH_BUDGET_S)
    elapsed = time.monotonic() - started

    assert attached is False
    assert (
        elapsed < READ_ATTACH_BUDGET_S + 1.0
    ), f"a mute owner cost the read {elapsed:.2f}s; the welcome leg is unbounded"
    assert owner.conns == 1
    assert viewer.cold_reason == "owner-silent"
    # A dial that never authenticated is not RETAINED: there is no sync to wait
    # for on this socket, and holding it would be a slot for nothing.
    assert viewer.attaching is False
    await viewer.dispose()
    await owner.stop()


@pytest.mark.asyncio
async def test_a_connected_but_unsynced_facade_is_not_reported_as_no_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MINOR-1: the early return for a connected facade must CLASSIFY too.

    ``is_cold`` is true for a facade whose client is up while
    ``_ready_for_events`` is cleared for a display refresh — a state this file's
    own docstring names — and that facade takes ``attach_existing``'s
    ``owner_reachable`` early return. Without classifying there, the token fell
    back to ``no-runtime``: "no pid holds this session's transcript lease",
    reported while a socket to that very pid is connected and serving.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path, sync_on_connect=True)
    await owner.start()
    _publish_live(tmp_path, owner)
    viewer = await _cold_viewer(tmp_path)
    await viewer.attach_existing(budget=READ_ATTACH_BUDGET_S)
    assert viewer.is_cold is False, "the fixture did not reach a live facade"

    # The mid-resync window: connected, serving, and not ready for events.
    viewer._ready_for_events = False
    viewer._read_cold_reason = None

    assert await viewer.attach_existing(budget=READ_ATTACH_BUDGET_S) is False
    assert viewer.owner_reachable is True
    assert (
        viewer.cold_reason == "owner-silent"
    ), "a connected facade must not be reported as having no runtime"
    await viewer.dispose()
    await owner.stop()


@pytest.mark.asyncio
async def test_a_sync_that_lands_during_the_deadline_settle_is_still_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MINOR-4: an expired landing deadline is not evidence nothing landed.

    ``_await_frontend`` documents a reproduction where ``wait_for``'s deadline
    trips with the frame already buffered — a race against the SCHEDULER — and
    answers it with bounded settle turns. The retained dial needs the same turns
    for a sharper reason: without them a sync that HAS landed is judged absent,
    its socket closed, and canonical state the renderer was promised a rollover
    for is discarded. The deadline is forced to zero here so the expiry is
    certain, and the sync lands a few turns later.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    await _seed(tmp_path)
    monkeypatch.setattr(remote_module, "SYNC_LANDING_DEADLINE_S", 0.0)
    viewer = await _cold_viewer(tmp_path)
    rollovers: list[Any] = []
    assert viewer.subscribe_frontend(rollovers.append) is not None
    pending: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    viewer._frontend_future = pending
    viewer._socketed_unsynced = True

    async def _land_after_a_few_turns() -> None:
        for _ in range(3):
            await asyncio.sleep(0)
        pending.set_result(_frontend_sync(SESSION_ID, tmp_path))

    asyncio.get_running_loop().create_task(_land_after_a_few_turns())
    await viewer._await_late_sync(pending)

    # Asserted on what was ADOPTED rather than on ``is_cold``: this facade has no
    # client at all (the dial is the socket under test in the other files), so
    # ``is_cold`` is true for it either way and would prove nothing here. What
    # MINOR-4 is about is whether the landed state was installed and published or
    # thrown away with the socket.
    assert viewer.frontend_state.epoch == "fake-owner"
    assert viewer._ready_for_events is True
    assert viewer.attaching is False
    assert viewer._read_cold_reason is None
    assert (
        rollovers and rollovers[-1].epoch == "fake-owner"
    ), "the landed sync was installed silently; the renderer never learns"
    await viewer.dispose()


@pytest.mark.asyncio
async def test_a_leaving_record_is_reported_as_leaving(tmp_path: Path, monkeypatch) -> None:
    """``owner-leaving`` is the record's own ``leaving`` phrase, not a guess."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path)
    await owner.start()
    _publish_live(tmp_path, owner, leaving="leaving for the build on disk when its turn ends")
    viewer = await _cold_viewer(tmp_path)

    await viewer.attach_existing(budget=READ_ATTACH_BUDGET_S)

    assert viewer.cold_reason == "owner-leaving"
    await viewer.dispose()
    await owner.stop()


@pytest.mark.asyncio
async def test_a_lost_presence_re_assert_does_not_fail_the_bind(
    tmp_path: Path, monkeypatch
) -> None:
    """D2.1. The desktop-watch ack is a lease hint, not a gate on binding.

    Two members of the same dial, and only one of them is load-bearing: the
    canonical sync carries the state a viewer needs, while the presence
    re-assert is a hint whose own TTL (45 s) and the renderer's next ``/watch``
    beat (15 s) repair it. This owner sends the sync and REFUSES the watch —
    the shape that raised ``OwnerAckTimeout`` out of the dial before the sync was
    ever awaited.

    The CONTROL envelope deliberately: it is what every non-read caller takes,
    and it is the one whose previous behaviour was to fail here. The bound is
    asserted too, because "best-effort" must still be bounded — a wedged owner
    cannot hold a redial open for the full 15 s request timeout.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path, answer_watch=False, sync_on_connect=True)
    await owner.start()
    _publish_live(tmp_path, owner)
    viewer = await _cold_viewer(tmp_path)
    # A recorded, still-fresh presence pair — what a viewer that was watching
    # moments before the drop carries into the redial. (A viewer with NO
    # recorded state takes the withdrawal arm of the replay instead; see
    # ``test_a_recorded_withdrawal_replays_as_a_withdrawal_not_a_lease``.)
    await viewer.update_desktop_watch(visible=True, can_notify=True)

    started = time.monotonic()
    await viewer.attach_existing()
    elapsed = time.monotonic() - started

    assert viewer.is_cold is False, "a lost presence hint refused a healthy bind"
    assert owner.watch_calls == 1, "the re-assert was never attempted"
    assert elapsed < remote_module._DESKTOP_WATCH_ACK_BOUND_S + 2.0
    await viewer.dispose()
    await owner.stop()


@pytest.mark.asyncio
async def test_a_recorded_withdrawal_replays_as_a_withdrawal_not_a_lease(
    tmp_path: Path, monkeypatch
) -> None:
    """THE SUCCESSOR'S HALF OF THE WITHDRAWAL (round 3, C2's pin).

    A runtime engaged AFTER the pane left must start and STAY detached, and
    the dial is where that is decided: ``withdraw_desktop_watch`` records the
    withdrawal as the desired state, and the next dial replays the WITHDRAWAL
    OP rather than re-asserting a lease — a lease re-assert here would make a
    fresh successor claim an attached interface nobody holds for its own
    45 s. The mirror (a fresh recorded pair) is pinned by
    ``test_a_lost_presence_re_assert_does_not_fail_the_bind``, which asserts
    ``watch_calls == 1`` on exactly the same wire.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path, answer_watch=True, sync_on_connect=True)
    await owner.start()
    _publish_live(tmp_path, owner)
    viewer = await _cold_viewer(tmp_path)

    # Record the withdrawal while cold: the bridge's last-lease seam does
    # exactly this, and the RECORDING is the part that must survive to the
    # next dial.
    await viewer.withdraw_desktop_watch()

    await viewer.attach_existing()

    assert owner.withdraw_calls == 1, "the recorded withdrawal was not replayed"
    assert owner.watch_calls == 0, "the dial re-asserted a lease after a withdrawal"
    await viewer.dispose()
    await owner.stop()


@pytest.mark.asyncio
async def test_a_sync_that_lands_after_the_budget_is_still_adopted(
    tmp_path: Path, monkeypatch
) -> None:
    """D2.2. The retained dial is the difference between slow and refused.

    The read answers cold inside its budget; the owner then speaks. The state
    must install, the facade must stop reporting cold/attaching, and the rollover
    must be PUBLISHED — that frame is what the renderer consumes to move from the
    cold paint to the live one, so a silent install would leave it cold for the
    rest of its life.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path)
    await owner.start()
    _publish_live(tmp_path, owner)
    viewer = await _cold_viewer(tmp_path)
    rollovers: list[Any] = []

    await viewer.attach_existing(budget=0.05)
    assert viewer.attaching is True

    # Registered after the cold store exists, which is when a desktop bridge's
    # own subscription is made.
    assert viewer.subscribe_frontend(rollovers.append) is not None
    await owner.send_sync()

    await _until(lambda: not viewer.is_cold, why="the late sync was never adopted")
    assert viewer.attaching is False
    assert viewer.cold_reason is None
    assert rollovers, "the rollover was installed silently; the renderer never learns"
    assert rollovers[-1].epoch == "fake-owner"
    await viewer.dispose()
    await owner.stop()


@pytest.mark.asyncio
async def test_a_given_up_read_stops_holding_the_runtime_socket(
    tmp_path: Path, monkeypatch
) -> None:
    """The hard landing deadline, because the socket is a residency term.

    An attach socket is term 3 of the runtime's own exit predicate, so a viewer
    that has given up must give the slot back rather than pin an 82 MB process
    for as long as the tab stays open.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(remote_module, "SYNC_LANDING_DEADLINE_S", 0.05)
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path)
    await owner.start()
    _publish_live(tmp_path, owner)
    viewer = await _cold_viewer(tmp_path)

    await viewer.attach_existing(budget=0.05)
    assert viewer.attaching is True

    await _until(
        lambda: viewer.attaching is False and viewer._client is None,
        why="the retained dial was never abandoned at its landing deadline",
    )
    assert viewer.cold_reason == "owner-silent"
    await viewer.dispose()
    await owner.stop()


@pytest.mark.asyncio
async def test_a_socket_that_dies_before_the_late_sync_leaves_a_cold_facade(
    tmp_path: Path, monkeypatch
) -> None:
    """The retained dial's failure boundary is the ordinary discard.

    ``_discard_rejected_client`` exists because a connected client with no
    installed state makes ``is_cold`` lie, and the retained dial must take that
    same boundary: an owner that dies after the read has answered leaves a facade
    that is COLD and honest, with the runtime's attach slot given back.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path)
    await owner.start()
    _publish_live(tmp_path, owner)
    viewer = await _cold_viewer(tmp_path)

    await viewer.attach_existing(budget=0.05)
    assert viewer.attaching is True
    await owner.stop()

    await _until(
        lambda: viewer._client is None,
        why="the dial was left half-bound after the socket closed",
    )
    assert viewer.is_cold is True
    assert viewer.cold_reason == "owner-silent"
    await viewer.dispose()


@pytest.mark.asyncio
async def test_a_wedged_record_is_dialled_rather_than_called_absent(
    tmp_path: Path, monkeypatch
) -> None:
    """D9. Two of the registry's three states are not "no runtime".

    ``find_runtime_record`` selects on ``live`` alone, so a WEDGED record (pid
    alive, heartbeat older than the timeout) reaches a read as ``(None, pid)``.
    Reading that as "no runtime" is a claim the registry has not made — and the
    record IS dialable, which the welcome's identity check arbitrates.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path, sync_on_connect=True)
    await owner.start()
    _publish_wedged(tmp_path, owner)
    states = [state for _record, state in registry.scan(tmp_path)]
    assert states == ["wedged"], f"the fixture is not the state under test: {states}"
    viewer = await _cold_viewer(tmp_path)

    await viewer.attach_existing(budget=READ_ATTACH_BUDGET_S)

    assert owner.conns == 1, "the wedged owner's own record was never dialled"
    assert viewer.is_cold is False, "a dialable owner was answered as absent"
    await viewer.dispose()
    await owner.stop()


@pytest.mark.asyncio
async def test_a_wedged_and_silent_owner_is_not_reported_as_no_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    """The same case on the cold path, where the token is the whole answer."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path)
    await owner.start()
    _publish_wedged(tmp_path, owner)
    viewer = await _cold_viewer(tmp_path)

    await viewer.attach_existing(budget=READ_ATTACH_BUDGET_S)

    assert viewer.cold_reason == "owner-silent"
    assert owner.conns == 1
    await viewer.dispose()
    await owner.stop()


@pytest.mark.asyncio
async def test_a_read_never_spawns_a_runtime(tmp_path: Path, monkeypatch) -> None:
    """D4. A GET is documented as side-effect free (``docs/DESKTOP_API.md``).

    On a sidebar sweep, a read that kicked a warm would spawn for rows nobody
    clicked. ``/warm`` and the live visible watch lease stay the only creators.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path)
    await owner.start()
    _publish_live(tmp_path, owner)
    spawned: list[Any] = []

    async def _forbidden(*args: Any, **kwargs: Any) -> Any:
        spawned.append((args, kwargs))
        raise AssertionError("a read path spawned a runtime")

    # ``attached`` imports this INSIDE the bind, so patching the module it
    # imports from is what reaches the call site.
    monkeypatch.setattr(launch, "engage_runtime", _forbidden)
    viewer = await _cold_viewer(tmp_path)

    assert await viewer.attach_existing(budget=READ_ATTACH_BUDGET_S) is False
    assert spawned == [], "the read attempted to engage a runtime"
    await viewer.dispose()
    await owner.stop()


@pytest.mark.asyncio
async def test_a_connected_leaving_owner_is_reported_as_leaving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NIT-2: the connected arm can produce the third token.

    The arm exists for a facade whose client is connected while
    ``_ready_for_events`` is cleared for a display refresh, and it classified with
    ``record=None`` — a pair that can only ever answer ``owner-silent``. A record
    that is LEAVING is the more accurate token (the one state that is both alive
    and knowingly unavailable), and the record this facade dialled is in hand, so
    the arm now passes it.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    await _seed(tmp_path)
    owner = _FakeOwner(SESSION_ID, tmp_path, sync_on_connect=True)
    await owner.start()
    _publish_live(tmp_path, owner)
    viewer = await _cold_viewer(tmp_path)
    await viewer.attach_existing(budget=READ_ATTACH_BUDGET_S)
    assert viewer.is_cold is False, "the fixture did not reach a live facade"
    record = viewer._runtime_record
    assert record is not None, "the dial did not keep the record it read"

    # A retiring owner whose socket is still up, read during the mid-resync window.
    viewer._runtime_record = dataclasses.replace(record, leaving=True)
    viewer._ready_for_events = False
    viewer._read_cold_reason = None

    assert await viewer.attach_existing(budget=READ_ATTACH_BUDGET_S) is False
    assert viewer.cold_reason == "owner-leaving"
    await viewer.dispose()
    await owner.stop()

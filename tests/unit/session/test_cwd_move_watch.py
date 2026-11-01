"""The cwd-move consumer: consume at an idle boundary, never mid-work.

The GATE tests below drive ``_apply_queued_cwd_move`` against a fake server: the
risk they pin is the predicate, and the predicate is all they touch.

The WATCHER tests at the end are the real thing — a real ``RuntimeServer``, its
real ``_queued_move_watch`` coroutine, its real ``announce_retiring`` — because
the claim they guard is about the LOOP: this runtime starts the watcher at boot
for every session, so a watcher with nothing to do must exit rather than spin,
and one holding a cwd intent must consume it even though no mesh move exists.
The two behaviours live in the dispatch order inside ``_queued_move_run``, which
no fake-server test reaches.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.session import cwd_move
from local_operator.session.runtime.server import RuntimeServer, _ClientConn

OWN = "own0000000001"


class _Handle:
    """The production predicates, with the facts spelled out per case."""

    def __init__(
        self,
        *,
        turn: bool = False,
        child: bool = False,
        job: bool = False,
        gate: bool = False,
        wake: bool = False,
        latch: bool = True,
    ) -> None:
        self.turn, self.child, self.job, self.gate, self.wake = turn, child, job, gate, wake
        self.latch = latch
        self.calls: list[str] = []

    def may_refresh(self) -> str:
        if self.turn or self.child or self.job or self.gate:
            return "busy"
        return "wake due within the warm window" if self.wake else ""

    def begin_retire(self, cause: str, detail: str = "") -> bool:
        self.calls.append(f"begin_retire:{cause}")
        return self.latch

    def request_stop(self) -> None:
        self.calls.append("request_stop")


class _Server:
    """Only the attributes the consumer touches; the METHOD is the real one."""

    _apply_queued_cwd_move = RuntimeServer._apply_queued_cwd_move
    # The gate itself is pinned by the real-server cells below; here it is the
    # GATE's inputs this fake is about, so it answers "a successor is guaranteed"
    # unless a case says otherwise.
    _cwd_move_successor_guaranteed = RuntimeServer._cwd_move_successor_guaranteed
    # The hold recorder and the capability check ride along: the gate's outcome
    # IS the record it leaves, so a fake that gated without recording would test
    # half of the behaviour. The announce is spied (``held`` below) — the paint
    # and the journal it drives are pinned on the real server.
    _note_cwd_move_blocked = RuntimeServer._note_cwd_move_blocked
    _cwd_move_predicted_hold = RuntimeServer._cwd_move_predicted_hold
    _announce_cwd_move_expired = RuntimeServer._announce_cwd_move_expired
    # The handoff retirement is the same three steps the viewer path takes, so
    # the fake composes the REAL one instead of restating its order.
    _retire_for_queued_move = RuntimeServer._retire_for_queued_move

    def __init__(self, handle: _Handle, root: Path) -> None:
        self._handle = handle
        self._config_root = root
        self._record = SimpleNamespace(session_id=OWN)
        self._queued_move: dict[str, Any] | None = None
        self._queued_move_stage = ""
        self._retirement_committed = False
        self._cwd_move_watched_request = ""
        #: One attached viewer that declares cwd adoption — the successor the
        #: gate requires. The fake never sends a frame, so a stand-in is enough
        #: here; the real-server cells below register a real connection.
        self._clients: dict[int, Any] = {0: SimpleNamespace(kind="attach", cwd_adoption=True)}
        self.announced: list[tuple[str, dict[str, Any]]] = []
        #: The hold/expiry notices the spied ``_announce_cwd_move`` received.
        self.held: list[str] = []

    async def _handle_call_on_session_loop(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    async def announce_retiring(self, reason: str, **kwargs: Any) -> None:
        self.announced.append((reason, kwargs))

    async def _announce_cwd_move(self, text: str, raw: str) -> None:
        self.held.append(text)


def _queue(root: Path) -> cwd_move.MoveIntent:
    destination = root / "dest"
    destination.mkdir(exist_ok=True)
    return cwd_move.enqueue_self_move(
        root, OWN, requested_cwd=str(destination), effective_cwd=str(root), owner_pid=1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "facts",
    [
        {"turn": True},
        {"child": True},
        {"job": True},
        {"gate": True},
        {"wake": True},
    ],
    ids=["active-turn", "live-child", "live-job", "parked-gate", "wake-window"],
)
async def test_busy_keeps_the_intent_queued_and_touches_nothing(
    tmp_path: Path, facts: dict[str, bool]
) -> None:
    intent = _queue(tmp_path)
    handle = _Handle(**facts)
    server = _Server(handle, tmp_path)
    assert await server._apply_queued_cwd_move() == "waiting"
    assert handle.calls == [] and server.announced == []
    stored = cwd_move.read_move_intent(tmp_path, OWN)
    assert stored is not None and stored.stage == cwd_move.STAGE_QUEUED
    assert server._queued_move_stage == ""
    assert intent.request_id == stored.request_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup",
    [
        lambda s: setattr(s, "_queued_move_stage", "finishing"),
        lambda s: setattr(s, "_retirement_committed", True),
        lambda s: setattr(s, "_queued_move", {"to_device": "peer-1", "to_name": "peer-1"}),
    ],
    ids=["stage-held", "already-retiring", "mesh-move-latched"],
)
async def test_a_runtime_already_moving_does_not_start_a_second_move(tmp_path: Path, setup) -> None:
    _queue(tmp_path)
    handle = _Handle()
    server = _Server(handle, tmp_path)
    setup(server)
    assert await server._apply_queued_cwd_move() == ""
    assert handle.calls == [] and server.announced == []


@pytest.mark.asyncio
async def test_idle_cutover_latches_announces_and_stops_in_that_order(tmp_path: Path) -> None:
    intent = _queue(tmp_path)
    handle = _Handle()
    server = _Server(handle, tmp_path)
    assert await server._apply_queued_cwd_move() == "retiring"
    assert handle.calls == ["begin_retire:cwd-move", "request_stop"]
    assert server.announced == [("cwd-move", {"cwd": intent.destination_cwd})]
    assert server._queued_move_stage == "retiring"
    # no peer ever rides a cwd move
    assert all("to" not in kwargs for _reason, kwargs in server.announced)


@pytest.mark.asyncio
async def test_a_refused_latch_keeps_the_intent_queued(tmp_path: Path) -> None:
    _queue(tmp_path)
    handle = _Handle(latch=False)
    server = _Server(handle, tmp_path)
    assert await server._apply_queued_cwd_move() == "waiting"
    assert server.announced == []
    stored = cwd_move.read_move_intent(tmp_path, OWN)
    assert stored is not None and stored.stage == cwd_move.STAGE_QUEUED


def test_no_intent_means_nothing_to_do(tmp_path: Path) -> None:
    import asyncio

    handle = _Handle()
    server = _Server(handle, tmp_path)
    assert asyncio.run(server._apply_queued_cwd_move()) == ""
    assert handle.calls == []


@pytest.mark.asyncio
async def test_identical_repeat_is_the_same_request_and_a_new_destination_supersedes(
    tmp_path: Path,
) -> None:
    first = _queue(tmp_path)
    again = _queue(tmp_path)
    assert again.request_id == first.request_id, "a retried tool call is one intent"
    other = tmp_path / "other"
    other.mkdir()
    second = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(other), effective_cwd=str(tmp_path), owner_pid=1
    )
    assert second.request_id != first.request_id
    assert second.stage == cwd_move.STAGE_QUEUED
    stored = cwd_move.read_move_intent(tmp_path, OWN)
    assert stored is not None and stored.request_id == second.request_id


def test_the_retiring_frame_carries_cwd_additively() -> None:
    import inspect

    source = inspect.getsource(RuntimeServer._announce_retiring_on_loop)
    assert '"cwd": cwd,' in source
    assert 'cwd: str = "",' in inspect.getsource(RuntimeServer.announce_retiring)


def test_the_viewer_stages_the_frame_cwd_for_adoption(tmp_path: Path) -> None:
    """The RECEIVING half of the frame contract, on the real handler.

    A cwd-move announcement is not a drain: it carries no ``draining`` and no
    ``updating``, so the handler must record the destination and paint nothing.
    The destination is STAGED rather than assigned — ``_cwd`` is read by
    ``_bind_under_lock`` inside the bind lock, so writing it here would race that
    read and the successor could boot at the old path (the measured blocker the
    field exists for). The second half asserts the additive direction: a frame
    from a runtime that predates this key must not CLEAR a staged destination.
    """
    from local_operator.session.attached import AttachedSession

    facade = AttachedSession.__new__(AttachedSession)
    facade._pending_frame_cwd = ""  # type: ignore[attr-defined]
    facade._drain_callback = None  # type: ignore[attr-defined]
    destination = str(tmp_path / "dest")

    facade._on_retiring_frame({"op": "retiring", "reason": "cwd-move", "cwd": destination})
    assert facade._pending_frame_cwd == destination

    facade._on_retiring_frame({"op": "retiring", "reason": "build"})
    assert facade._pending_frame_cwd == destination, "an older frame must not un-stage it"


@pytest.mark.asyncio
async def test_a_move_queued_while_the_session_runs_is_consumed_at_the_next_boundary(
    tmp_path: Path,
) -> None:
    """THE PATH A TOOL CALL ACTUALLY TAKES, which no cell exercised (B1/U1).

    Every cell above queues BEFORE the watcher starts (or before a successor
    spawns), so "the asking runtime consumes its own intent" was never tested —
    and it did not work: the boot watcher exits with nothing queued, the tool only
    wrote the marker, and the intent waited for some later, unrelated boot.

    This cell reproduces the real order instead: the runtime is UP, the watcher has
    already exited, a turn is in flight, and only then does the tool write the
    marker and wake the consumer (``notify_move_queued``, the call
    ``_sessions_move`` makes). Two properties are pinned:

    * the woken watcher KEEPS POLLING while the runtime is busy — the first tick
      after a mid-turn enqueue is always "not at a boundary", so a watcher that
      gave up there would reproduce the original defect one layer down;
    * it retires as soon as the turn ends, with no second wake.
    """
    import asyncio

    from local_operator.session.runtime.server import notify_move_queued

    server, handle, announced = _real_server(tmp_path)
    # 1. The boot watcher, with nothing queued: it exits (the pinned behaviour).
    await asyncio.wait_for(server._queued_move_watch(), timeout=5)
    assert server._queued_move_task is None or server._queued_move_task.done()
    # 2. A turn is running, and the tool queues a move inside it.
    handle.busy = "active turn"
    destination = tmp_path / "dest"
    destination.mkdir()
    intent = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(destination), effective_cwd=str(tmp_path), owner_pid=1
    )
    assert notify_move_queued(OWN, tmp_path), "the tool-path wake must find this runtime"
    # 3. Busy at the boundary: the watcher stays on the intent, latching nothing.
    await asyncio.sleep(0.2)
    assert handle.calls == [], "nothing is cut over mid-turn"
    assert announced == []
    stored = cwd_move.read_move_intent(tmp_path, OWN)
    assert stored is not None and stored.stage == cwd_move.STAGE_QUEUED
    assert (
        server._queued_move_task is not None and not server._queued_move_task.done()
    ), "a queued intent keeps the consumer polling until it can act"
    # 4. The turn ends: the move happens with no further wake of any kind.
    watcher = server._queued_move_task
    assert watcher is not None and not watcher.done()
    handle.busy = ""
    # The latch is observable before announce_retiring's awaited work finishes.
    # Await THIS consumer, not its first side effect: otherwise the test tears
    # down its loop between begin_retire and request_stop. Preserve the old
    # 60 * 0.05s deadlock bound and propagate an actual watcher exception.
    await asyncio.wait_for(asyncio.shield(watcher), timeout=3)
    assert watcher.exception() is None
    assert handle.calls == ["begin_retire:cwd-move", "request_stop"]
    assert announced == [("cwd-move", {"cwd": intent.destination_cwd})]


@pytest.mark.asyncio
async def test_a_runtime_with_nobody_to_follow_records_the_handoff_and_retires(
    tmp_path: Path,
) -> None:
    """THE VIEWER-LESS CASE MOVES ITSELF, which it could never do before.

    The old contract here was the opposite: retiring toward a move nothing would
    adopt was a hold (``no_capable_viewer``) that lapsed after thirty minutes with
    the cwd unchanged, because a successor booting in the SOURCE would leave the
    intent queued and retire again on every open -- the B2 loop. The handoff
    removes the loop instead of the move: the destination is written as the
    session's recorded truth BEFORE the retire, so whichever process engages next
    boots there, and that boot commits it (``reconcile_move``).
    """
    destination = tmp_path / "dest"
    destination.mkdir()
    server, handle, announced = _real_server(tmp_path, viewer=False)
    intent = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(destination), effective_cwd=str(tmp_path), owner_pid=1
    )
    assert await server._apply_queued_cwd_move() == "retiring"
    assert handle.calls == ["begin_retire:cwd-move", "request_stop"], "it latches, then stops"
    assert announced == [("cwd-move", {"cwd": intent.destination_cwd})], "the destination is named"
    stored = cwd_move.read_move_intent(tmp_path, OWN)
    assert stored is not None and stored.stage == cwd_move.STAGE_HANDOFF
    assert stored.request_id == intent.request_id, "one request, one identity"
    assert stored.source_cwd == str(tmp_path) and stored.destination_cwd == str(destination)
    assert stored.blocked_reason == "", "a handoff is not a hold"

    # AND A VIEWER THAT CAN FOLLOW CHANGES NOTHING DURABLE: it retires exactly as
    # before, and the marker is left for the successor's boot to commit.
    capable_root = tmp_path / "second"
    (capable_root / "dest").mkdir(parents=True)
    capable, capable_handle, _ = _real_server(capable_root)
    cwd_move.enqueue_self_move(
        capable_root,
        OWN,
        requested_cwd=str(capable_root / "dest"),
        effective_cwd=str(capable_root),
        owner_pid=1,
    )
    assert await capable._apply_queued_cwd_move() == "retiring"
    assert capable_handle.calls == ["begin_retire:cwd-move", "request_stop"]
    still_queued = cwd_move.read_move_intent(capable_root, OWN)
    assert still_queued is not None and still_queued.stage == cwd_move.STAGE_QUEUED


def test_a_held_request_never_expires_and_nothing_ages_it(tmp_path: Path) -> None:
    """THE TTL IS GONE, and its removal is the fix rather than a regression.

    A bounded life existed because a request no successor ever satisfied would
    re-arm on every boot for the life of the session (B2). The handoff closes that
    loop by construction -- the successor boots at the destination, and a boot
    anywhere else supersedes the request once -- so there is nothing left for a
    clock to bound. What used to be ``expired`` is now either ``committed`` or an
    explicit ``superseded``, and a recorded reason is a notice, not a deadline:
    this cell pins that an ancient held request is still there to be acted on.
    """
    destination = tmp_path / "dest"
    destination.mkdir()
    intent = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(destination), effective_cwd=str(tmp_path), owner_pid=1
    )
    # A hold recorded a decade ago, by a build that would have expired it.
    cwd_move.mark_blocked(
        tmp_path, OWN, "no_capable_viewer", now=intent.created_at - 10 * 365 * 24 * 60 * 60
    )
    stored = cwd_move.read_move_intent(tmp_path, OWN)
    assert stored is not None and stored.stage == cwd_move.STAGE_QUEUED
    assert stored.blocked_reason == "no_capable_viewer", "the notice survives, readably"
    # NOTHING IN THE MODULE AGES ANYTHING ANY MORE: the API that did is retired,
    # and a reader that reintroduced it would be reintroducing the lapse.
    assert not hasattr(cwd_move, "expire_stale_intent")
    assert not hasattr(cwd_move, "hold_lapsed")
    assert not hasattr(cwd_move, "QUEUED_MOVE_TTL_S")


def test_a_symlinked_current_directory_is_not_a_move(tmp_path: Path) -> None:
    """Both sides of the comparison are realpath-ed (agent review m1, UX U7).

    A session whose cwd is reached through a symlink (every macOS ``/var`` path,
    any hand-made link) used to queue a move to the tree it was already in — a
    full restart that changed nothing.
    """
    real = tmp_path / "src"
    real.mkdir()
    link = tmp_path / "src-link"
    link.symlink_to(real)
    with pytest.raises(cwd_move.CwdMoveRefused) as refused:
        cwd_move.enqueue_self_move(
            tmp_path, OWN, requested_cwd=str(real), effective_cwd=str(link), owner_pid=1
        )
    assert refused.value.code == "sessions_move_destination_is_current"
    assert cwd_move.read_move_intent(tmp_path, OWN) is None, "and nothing was written"


# ---------------------------------------------------------------------------
# THE WATCHER ITSELF. Everything above drives one method against a fake server;
# these drive the coroutine the runtime really starts.
# ---------------------------------------------------------------------------


class _RealHandle:
    """The handle seam a real ``RuntimeServer`` reads, and nothing else.

    Deliberately not ``_Handle`` above: that double reimplements ``may_refresh``
    so the GATE can be spelled out per case. Here the server is real and the
    handle is an idle session — one answer, no parametrisation — so a change in
    the server's own dispatch is what the assertions see.
    """

    #: The runtime hops through this attribute when a handle publishes a loop;
    #: ``None`` is the in-process host the real dispatch runs inline for.
    session_loop = None

    def __init__(self, session_id: str, cwd: str) -> None:
        self.session_id = session_id
        self.cwd = cwd
        self.calls: list[str] = []
        #: What ``may_refresh`` answers — ``""`` is the idle boundary. A test
        #: sets it to model the turn a tool call is running inside.
        self.busy = ""

    @property
    def session_projection_seed(self) -> Any:
        from local_operator.mobile.types import SessionProjection

        return SessionProjection(
            session_id=self.session_id,
            pid=0,
            kind="tui",
            conversation_name="own-cwd watcher",
            cwd=self.cwd,
            model_label="test/model",
        )

    def may_refresh(self) -> str:
        return self.busy

    def begin_retire(self, cause: str, detail: str = "") -> bool:
        self.calls.append(f"begin_retire:{cause}")
        return True

    def request_stop(self) -> None:
        self.calls.append("request_stop")


class _FakeWriter:
    """A sink that satisfies ``_ClientConn.writer``: the frames go nowhere."""

    def __init__(self) -> None:
        self.written: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        return None


def _attach_fake_viewer(server: RuntimeServer) -> None:
    """Register a viewer that DECLARES cwd adoption, through the real registry.

    A real ``Connection`` on a started runtime is the whole point of the
    guarantee: the retiring announce is ordinary code and walks ``_clients``, so
    a stand-in that skipped the registry would test a runtime that never sends.
    """
    writer = _FakeWriter()
    conn = _ClientConn(writer=writer, kind="attach", cwd_adoption=True)  # type: ignore[arg-type]
    server._clients[id(writer)] = conn


def _real_server(
    tmp_path: Path,
    *,
    viewer: bool = True,
) -> tuple[RuntimeServer, _RealHandle, list[tuple[str, dict[str, Any]]]]:
    """A REAL ``RuntimeServer`` pointed at ``tmp_path`` as its config root.

    The two attributes set here are the two the boot path sets before the watcher
    starts (``_config_root`` from ``config_dir()``, the record's session id from
    the seed), so the watcher reads the same marker file it reads in production.
    The announcement is spied by CALLING THROUGH: the real frame construction and
    the real record rewrite still run, and only the arguments are recorded.
    """
    handle = _RealHandle(OWN, str(tmp_path))
    server = RuntimeServer(handle, kind="tui")  # type: ignore[arg-type]
    server._config_root = tmp_path
    server._record.session_id = OWN
    # REGISTERED THE WAY BOOT REGISTERS IT, so a tool-path wake finds this runtime
    # exactly as it does in production (agent review round 1, B1).
    # THE RUNTIME'S OWN LOOP, which a started runtime records before it serves:
    # the frame send refuses to run on a foreign loop, and the retiring announce
    # goes through it.
    server._loop = asyncio.get_running_loop()
    server._register_queued_move_waker()
    if viewer:
        # A viewer that can follow a move, declared the way one declares itself in
        # its auth frame. Without it this runtime is CORRECT to hold the intent —
        # see ``test_a_runtime_that_cannot_guarantee_a_successor_holds_it``. A
        # real ``_ClientConn`` with a sink writer, because the retiring announce
        # sends to every attached client through the ordinary path.
        _attach_fake_viewer(server)
    announced: list[tuple[str, dict[str, Any]]] = []
    real_announce = server.announce_retiring

    async def spy(reason: str, **kwargs: Any) -> None:
        announced.append((reason, kwargs))
        await real_announce(reason, **kwargs)

    server.announce_retiring = spy  # type: ignore[method-assign]
    return server, handle, announced


@pytest.mark.asyncio
async def test_the_real_watcher_exits_when_there_is_no_intent_at_all(tmp_path: Path) -> None:
    """Boot starts this watcher for EVERY session, so nothing-to-do must EXIT.

    A watcher that kept its half-second cadence here would re-read a missing
    marker forever in every idle runtime on the machine — the cost this asserts
    against is the loop, not the read. ``wait_for`` is the discriminator: a
    watcher that does not return trips it instead of hanging the suite.
    """
    import asyncio

    server, handle, announced = _real_server(tmp_path)
    await asyncio.wait_for(server._queued_move_watch(), timeout=5)
    assert handle.calls == [], "an idle session is never latched for a move it never got"
    assert announced == [], "and nothing is announced to viewers"
    assert server._queued_move_stage == ""
    assert cwd_move.read_move_intent(tmp_path, OWN) is None


@pytest.mark.asyncio
async def test_the_real_watcher_consumes_a_cwd_intent_with_no_mesh_move(tmp_path: Path) -> None:
    """The dispatch order: a cwd intent is consumed BEFORE the mesh gate.

    THE REGRESSION THIS PINS is the watcher's old first line, which returned the
    moment ``_queued_move`` (the MESH request) was unset. A cwd move never sets
    it: the intent is a document on disk. So the whole feature would have failed
    silently — the tool queues, and the runtime that is supposed to act on it
    returns in the same instant.

    ``_queued_move_stage`` becoming ``retiring`` and the latch/stop order is the
    same sequence the fake-server gate tests pin; what is new here is that the
    REAL coroutine reaches it, which is what makes the ordering claim true.
    """
    import asyncio

    destination = tmp_path / "dest"
    destination.mkdir()
    intent = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(destination), effective_cwd=str(tmp_path), owner_pid=1
    )
    server, handle, announced = _real_server(tmp_path)
    assert server._queued_move is None, "a cwd move is not a mesh move"

    await asyncio.wait_for(server._queued_move_watch(), timeout=5)

    assert handle.calls == ["begin_retire:cwd-move", "request_stop"]
    assert announced == [("cwd-move", {"cwd": intent.destination_cwd})]
    assert server._queued_move_stage == "retiring"
    # THE CONFIRMATION IS THE SUCCESSOR'S, still: this runtime only leaves.
    stored = cwd_move.read_move_intent(tmp_path, OWN)
    assert stored is not None and stored.stage == cwd_move.STAGE_QUEUED


# ---------------------------------------------------------------------------
# ROUND 2: the holds are ordered, announced, per request, and bounded by the
# hold's own clock; a stood-down mesh move no longer blocks a cwd move.
# ---------------------------------------------------------------------------


class _Announcing(_RealHandle):
    """A ``_RealHandle`` whose session records what the runtime told it.

    ``queue_notice`` is the user's half (painted on every viewer) and
    ``journal_incident`` the agent's (live context + transcript), the pair
    ``RuntimeServer._announce_cwd_move`` drives through the ordinary hop.
    """

    def __init__(self, session_id: str, cwd: str) -> None:
        super().__init__(session_id, cwd)
        painted: list[str] = []
        journalled: list[str] = []
        self.painted, self.journalled = painted, journalled

        class _Session:
            async def queue_notice(self, text: str, kind: str = "info") -> None:
                painted.append(text)

            async def journal_incident(self, raw: str, *, rendered: str = "", **_: Any) -> None:
                journalled.append(rendered)

        self._session = _Session()


def _announcing_server(
    tmp_path: Path, *, viewer: bool = True
) -> tuple[RuntimeServer, _Announcing, list[tuple[str, dict[str, Any]]]]:
    server, _plain, announced = _real_server(tmp_path, viewer=viewer)
    handle = _Announcing(OWN, str(tmp_path))
    server._handle = handle  # type: ignore[assignment]
    return server, handle, announced


async def _until(predicate: Any, *, ticks: int = 80) -> None:
    for _ in range(ticks):
        if predicate():
            return
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_a_stood_down_mesh_move_does_not_block_a_later_cwd_move(tmp_path: Path) -> None:
    """R2-M1: mesh cancel -> cwd move -> retire, in one runtime.

    A mesh move whose record is gone stands down with ``_queued_move_stage``
    left at ``cancelled`` (the relay's ``queue_move`` reply reads it). The cwd
    gate used to treat ANY non-empty stage as in flight, so every later cwd move
    returned at once and the watcher exited — while the tool said "next idle
    point". Driven through the real mesh stand-down, not a hand-set field.
    """
    server, handle, announced = _real_server(tmp_path)
    server._queued_move = {"to_device": "peer-1", "at": 1.0, "to_name": "peer-1"}
    await asyncio.wait_for(server._queued_move_watch(), timeout=5)
    assert server._queued_move is None and server._queued_move_stage == "cancelled"

    destination = tmp_path / "dest"
    destination.mkdir()
    intent = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(destination), effective_cwd=str(tmp_path), owner_pid=1
    )
    from local_operator.session.runtime.server import CWD_MOVE_READY, notify_move_queued

    assert notify_move_queued(OWN, tmp_path) == CWD_MOVE_READY
    await _until(lambda: handle.calls)
    assert handle.calls == ["begin_retire:cwd-move", "request_stop"]
    assert announced == [("cwd-move", {"cwd": intent.destination_cwd})]


@pytest.mark.asyncio
async def test_a_host_that_cannot_restart_holds_with_its_own_reason_and_says_so(
    tmp_path: Path,
) -> None:
    """U9's real defect: an incapable host is a HOLD with a true reason, said aloud.

    The production class is ``TuiSessionHandle`` — the in-process registrant an
    embedder (not plain ``lop``, whose TUI is always a viewer) builds over a
    ``Session`` it owns; it publishes no retire latch. Before this the gate
    recorded ``no_capable_viewer`` (the successor check ran first) and then
    returned silently, and the tool promised "the next idle point". Now the
    capability is judged FIRST — with no viewer attached too, so the order is
    what the reason proves — the tool is told the hold up front, and the user
    and the agent are both told why.
    """
    from local_operator.mobile.tui_handle import TuiSessionHandle
    from local_operator.session.runtime.server import notify_move_queued

    assert not hasattr(TuiSessionHandle, "begin_retire"), "the class this cell models"
    server, handle, announced = _announcing_server(tmp_path, viewer=False)
    # The incapable host: the latch and the probe are simply absent.
    type(handle).begin_retire = None  # type: ignore[assignment,misc]
    try:
        destination = tmp_path / "dest"
        destination.mkdir()
        cwd_move.enqueue_self_move(
            tmp_path, OWN, requested_cwd=str(destination), effective_cwd=str(tmp_path), owner_pid=1
        )
        assert notify_move_queued(OWN, tmp_path) == "host_cannot_restart"
        await _until(lambda: handle.painted)
        stored = cwd_move.read_move_intent(tmp_path, OWN)
        assert stored is not None and stored.stage == cwd_move.STAGE_QUEUED
        assert stored.blocked_reason == "host_cannot_restart", "not no_capable_viewer"
        assert announced == [] and handle.calls == []
        assert len(handle.painted) == 1 and "cannot restart it" in handle.painted[0]
        assert handle.journalled == [f"[session incident] {handle.painted[0]}"]
        # Once per reason: further ticks add nothing.
        await asyncio.sleep(0.6)
        assert len(handle.painted) == 1
    finally:
        del type(handle).begin_retire
        server._closed.set()


@pytest.mark.asyncio
async def test_a_move_queued_in_a_long_turn_is_acted_on_when_the_turn_ends(
    tmp_path: Path,
) -> None:
    """U10's sequence -- busy, then idle -- with the handoff as the outcome.

    The request was made 31 minutes ago (an ordinary turn here); a busy runtime
    acts on nothing and records no hold, and when the turn ends the move happens
    through the handoff because nobody is attached to follow it. Nothing lapses
    on the way: what used to be busy -> expired -> idle is now busy -> handed off.
    """
    server, handle, announced = _announcing_server(tmp_path, viewer=False)
    destination = tmp_path / "dest"
    destination.mkdir()
    intent = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(destination), effective_cwd=str(tmp_path), owner_pid=1
    )
    # Backdate the request, as a 31-minute turn would leave it.
    cwd_move._write(
        tmp_path,
        cwd_move._with_stage(intent, cwd_move.STAGE_QUEUED, created_at=intent.created_at - 31 * 60),
    )
    handle.busy = "active turn"
    assert await server._apply_queued_cwd_move() == "waiting"
    still = cwd_move.read_move_intent(tmp_path, OWN)
    assert still is not None and still.stage == cwd_move.STAGE_QUEUED, "busy is not a move"
    assert still.blocked_reason == "" and handle.painted == [], "busy is not a hold either"
    # The turn ends with nobody able to follow: the request is recorded and the
    # runtime retires, announcing the destination it will start in.
    handle.busy = ""
    assert await server._apply_queued_cwd_move() == "retiring"
    handed = cwd_move.read_move_intent(tmp_path, OWN)
    assert handed is not None and handed.stage == cwd_move.STAGE_HANDOFF
    assert handed.request_id == intent.request_id
    assert handle.painted == [], "a handoff is not a hold and paints nothing new"
    assert announced == [("cwd-move", {"cwd": intent.destination_cwd})]


@pytest.mark.asyncio
async def test_a_recorded_hold_is_cleared_by_the_tick_that_acts(tmp_path: Path) -> None:
    """A NOTICE IS NOT A GATE: the request still moves once its cause is gone.

    The destination is deleted after the request, so one tick records
    ``destination_unavailable`` and announces it. Recreating it and going idle
    lets the SAME request retire -- and the notice goes with the act, so nothing
    later reports a condition that no longer holds.
    """
    server, handle, announced = _announcing_server(tmp_path, viewer=False)
    destination = tmp_path / "dest"
    destination.mkdir()
    intent = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(destination), effective_cwd=str(tmp_path), owner_pid=1
    )
    destination.rmdir()
    assert await server._apply_queued_cwd_move() == "waiting"
    held = cwd_move.read_move_intent(tmp_path, OWN)
    assert held is not None and held.blocked_reason == "destination_unavailable"
    assert len(handle.painted) == 1 and "is gone" in handle.painted[0]
    # The condition clears; while the session is busy the notice stays (no write).
    destination.mkdir()
    handle.busy = "active turn"
    assert await server._apply_queued_cwd_move() == "waiting"
    kept = cwd_move.read_move_intent(tmp_path, OWN)
    assert kept is not None and kept.blocked_reason == "destination_unavailable"
    assert len(handle.painted) == 1, "a persisting condition is not re-announced"
    # The turn ends: it acts, and the notice goes with the act.
    handle.busy = ""
    assert await server._apply_queued_cwd_move() == "retiring"
    cleared = cwd_move.read_move_intent(tmp_path, OWN)
    assert cleared is not None and cleared.stage == cwd_move.STAGE_HANDOFF
    assert cleared.blocked_reason == "" and cleared.blocked_at == 0.0
    assert announced == [("cwd-move", {"cwd": intent.destination_cwd})]


@pytest.mark.asyncio
async def test_a_destination_deleted_after_the_request_is_held_not_retired_toward(
    tmp_path: Path,
) -> None:
    """Q2 / m-d: re-check the destination at the idle point, before the latch."""
    server, handle, announced = _announcing_server(tmp_path)
    destination = tmp_path / "vanishing"
    destination.mkdir()
    cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(destination), effective_cwd=str(tmp_path), owner_pid=1
    )
    destination.rmdir()  # the same turn's `rmdir`
    assert await server._apply_queued_cwd_move() == "waiting"
    stored = cwd_move.read_move_intent(tmp_path, OWN)
    assert stored is not None and stored.blocked_reason == "destination_unavailable"
    assert handle.calls == [] and announced == [], "no successor booted in a missing directory"
    assert "is gone" in handle.painted[0]
    # Recreated: the hold clears and the move proceeds.
    destination.mkdir()
    assert await server._apply_queued_cwd_move() == "retiring"
    assert handle.calls == ["begin_retire:cwd-move", "request_stop"]


@pytest.mark.asyncio
async def test_a_new_request_under_the_same_hold_is_recorded_and_announced_again(
    tmp_path: Path,
) -> None:
    """m-a: A held, A cancelled, B queued under the same hold -> B shows the hold."""
    server, handle, _announced = _announcing_server(tmp_path, viewer=False)
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    # Each destination is accepted at the CALL and deleted before the idle point,
    # so both requests hit the same transient reason: the memo has to be per
    # REQUEST, never per runtime.
    cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(first), effective_cwd=str(tmp_path), owner_pid=1
    )
    first.rmdir()
    assert await server._apply_queued_cwd_move() == "waiting"
    cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(tmp_path), effective_cwd=str(tmp_path), owner_pid=1
    )
    assert await server._apply_queued_cwd_move() == ""
    b = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(second), effective_cwd=str(tmp_path), owner_pid=1
    )
    second.rmdir()
    assert await server._apply_queued_cwd_move() == "waiting"
    stored = cwd_move.read_move_intent(tmp_path, OWN)
    assert stored is not None and stored.request_id == b.request_id
    assert stored.blocked_reason == "destination_unavailable"
    assert len(handle.painted) == 2 and second.name in handle.painted[1]

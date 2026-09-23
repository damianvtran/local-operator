"""A desktop submit is ANNOUNCED before the door it has to wait at.

THE DEFECT THESE TESTS PIN. On a session with no runtime, the acknowledgement a
user waits for used to be produced inside the submit's own engage — measured at
p50 2,622 ms on a quiet box (``scripts/bench_ttft.py``, scenario
``desktop-cold``), and at ~1.0 s even when the engage itself was already warm,
because ``DesktopSessionBridge.acquire`` waits on the bind lock behind whatever
speculative warm the visible ``/watch`` lease armed
(``session/attached.py``'s ``_BACKGROUND_YIELD_BUDGET_S``). So the frame is now
published BEFORE the door, by ``DesktopSessions.announce_admission``, on the bridge that is
already resident (``routes/desktop_sessions.py::ADMISSION_ACCEPTED_FRAME``).

WHY THE ASSERTIONS ARE STRUCTURAL AND NOT TIMING-BASED (AGENTS.md, "Wait on the
event, never on the clock"). The property is an ORDER, so every cell here waits
for a publication from the code under test and then asserts a fact about it:

* the acknowledgement is on the viewer's stream while the owner's socket has not
  yet been handed the turn — a parked admission, not a measurement;
* it takes no bridge reference and leaves the facade cold, which is what makes
  ``announce`` unable to spawn anything;
* a repeated request id announces once, and still replays its receipt;
* a latched daemon announces nothing, so it cannot say "taken" and then refuse.

The owner is a REAL loopback socket speaking the real wire (``_FakeOwner``): the
route's own dial, welcome, canonical sync and admission ack are all exercised,
which is the half a bridge double cannot show.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import (
    desktop_catalogues,
    desktop_lifecycle,
    desktop_sessions,
)
from local_operator.server.utils.desktop_sessions import (
    ADMISSION_ACCEPTED_FRAME,
    ADMISSION_FAILED_FRAME,
    DesktopSessions,
)
from tests.unit.session.test_ownerless_read import _FakeOwner, _publish_live, _seed

TOKEN = "synthetic-admission-ack-token"

#: A bound on an awaited publication, never a budget to sleep through.
DEADLOCK_GUARD_S = 20.0


class _Harness:
    """One pool, one real router app, one synthetic session and its cwd.

    Deliberately the same shape as the sibling route harnesses' (a real app over
    a real ``DesktopSessions``), because the subject here IS the pool: the
    announcement resolves a resident bridge out of it, and only the real pool has
    the resident set to resolve.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.app = FastAPI()
        self.app.include_router(desktop_sessions.router)
        self.app.include_router(desktop_lifecycle.router)
        self.app.include_router(desktop_catalogues.router)
        self.pool = DesktopSessions(root)
        self.app.state.desktop_sessions = self.pool

        self.app.state.config_manager = ConfigManager(config_dir=root)
        self.inputs = root / "workspace"
        self.inputs.mkdir(parents=True, exist_ok=True)
        self.session_id = ""
        self.client: AsyncClient | None = None

    async def __aenter__(self) -> "_Harness":
        self.session_id = await self.pool.create(str(self.inputs))
        await _seed(self.root, session_id=self.session_id, rows=2)
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self.client is not None:
            await self.client.aclose()


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)


class _Attached:
    """A mounted viewer: one subscription holding the session's bridge.

    THE BRIDGE IS ONLY RESIDENT WHILE SOMETHING HOLDS IT, which is exactly the
    condition ``DesktopSessions.announce_admission`` documents and the reason a mounted
    viewer is the one it can reach. This holds it through the pool and reads the
    bridge's own event generator — the shape the sibling route tests use, and the
    same frames the ``/events`` route serves, without putting a never-ending ASGI
    response in front of the assertion.
    """

    def __init__(self, harness: _Harness) -> None:
        self.harness = harness
        self.bridge: Any = None
        self.frames: list[dict[str, Any]] = []
        self._task: asyncio.Task[None] | None = None
        self._arrived = asyncio.Event()

    async def __aenter__(self) -> "_Attached":
        self._held = self.harness.pool.session(self.harness.session_id, read=True)
        self.bridge = await self._held.__aenter__()
        self._subscription = self.bridge.subscribe()
        self._stream = self.bridge.events(self._subscription, epoch=None, after_seq=0)
        self._task = asyncio.create_task(self._pump())
        await self.wait_for(lambda: any(f.get("type") == "snapshot" for f in self.frames))
        return self

    async def _pump(self) -> None:
        async for frame in self._stream:
            self.frames.append(frame)
            self._arrived.set()

    async def wait_for(self, predicate: Any) -> None:
        """Wait until ``predicate`` holds, re-testing on each arriving frame."""
        async with asyncio.timeout(DEADLOCK_GUARD_S):
            while not predicate():
                self._arrived.clear()
                await self._arrived.wait()

    def acked(self) -> list[dict[str, Any]]:
        return [f for f in self.frames if f.get("type") == ADMISSION_ACCEPTED_FRAME]

    def failed(self) -> list[dict[str, Any]]:
        return [f for f in self.frames if f.get("type") == ADMISSION_FAILED_FRAME]

    async def __aexit__(self, *_: Any) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(BaseException):
                await self._task
        await self._held.__aexit__(None, None, None)


async def _post(harness: _Harness, text: str = "hello", *, request_id: str) -> Any:
    assert harness.client is not None
    return await harness.client.post(
        f"/v1/desktop/sessions/{harness.session_id}/messages",
        json={"request_id": request_id, "text": text},
    )


@pytest.mark.asyncio
async def test_the_acknowledgement_leaves_before_the_owner_hears_the_turn(
    tmp_path: Path,
) -> None:
    """The ack is on the stream while the admission is still in flight.

    THE WHOLE POINT, and the reason it is asserted against a PARKED admission:
    the owner's socket has received nothing yet, so anything the viewer can
    already read was produced by the host on its own behalf rather than after the
    engage answered. A version that published after ``admit_prompt`` cannot pass
    this cell — the frame would not exist until the park is released.
    """
    park = asyncio.Event()
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(
            harness.session_id,
            tmp_path,
            sync_on_connect=True,
            answer_prompts=True,
            park_prompts=park,
        )
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        try:
            async with _Attached(harness) as viewer:
                request_id = str(uuid.uuid4())
                submitted = asyncio.create_task(_post(harness, request_id=request_id))

                await viewer.wait_for(lambda: bool(viewer.acked()))

                assert owner.prompts == [], "the owner saw the turn before the ack was published"
                assert not submitted.done(), "the ack cannot be the submit's own answer"
                ack = viewer.acked()[0]
                assert ack["payload"] == {"request_id": request_id, "mode": "prompt"}
                assert ack["seq"] <= max(f["seq"] for f in viewer.frames)

                park.set()
                response = await asyncio.wait_for(submitted, DEADLOCK_GUARD_S)
                assert response.status_code == 200, response.text
                assert response.json()["result"]["status"] == "admitted"
                assert len(owner.prompts) == 1, "the turn reached the owner exactly once"
        finally:
            await owner.stop()


@pytest.mark.asyncio
async def test_the_acknowledgement_does_not_wait_at_the_door(tmp_path: Path) -> None:
    """The frame leaves while the submit is still QUEUED FOR A BRIDGE.

    THE DISCRIMINATING CELL, and the reason the acknowledgement is announced
    rather than published inside the receipt: a cold submit's first stop is
    ``DesktopSessionBridge.acquire``, and that lock is exactly what a racing
    acquire — the speculative warm a visible ``/watch`` lease arms — holds for
    ``_BACKGROUND_YIELD_BUDGET_S``. Occupying it here stands in for that race
    without needing a warm, a spawn or a clock: while it is held the submit
    cannot have reached the owner, yet the frame must already be on the viewer's
    stream. A version that published after the door cannot pass this cell.
    """
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(harness.session_id, tmp_path, sync_on_connect=True, answer_prompts=True)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        try:
            async with _Attached(harness) as viewer:
                bridge = viewer.bridge
                request_id = str(uuid.uuid4())
                async with bridge.lock:
                    submitted = asyncio.create_task(_post(harness, request_id=request_id))
                    await viewer.wait_for(lambda: bool(viewer.acked()))
                    assert not submitted.done(), "the door was supposed to be occupied"
                    assert owner.prompts == []
                    assert viewer.acked()[0]["payload"]["request_id"] == request_id
                response = await asyncio.wait_for(submitted, DEADLOCK_GUARD_S)
                assert response.status_code == 200, response.text
        finally:
            await owner.stop()


async def _replay_until_snapshot(bridge: Any, *, after_seq: int) -> list[dict[str, Any]]:
    """Every frame a reconnecting viewer is replayed, up to its snapshot.

    ``bridge.events`` is ordered replay-then-snapshot and then live, so the
    snapshot is the boundary that makes "what did this cursor receive" a finite
    question — the same boundary the sibling replay cells read.

    The EPOCH is passed, because that is what a reconnect is: ``/events`` takes
    ``epoch`` and ``after_seq`` from the client, and a viewer that supplies
    neither is a FRESH attach — it gets the snapshot and no replay at all, which
    is the transport's documented behaviour rather than a gap.
    """
    subscription = bridge.subscribe()
    stream = bridge.events(subscription, epoch=bridge.epoch, after_seq=after_seq)
    seen: list[dict[str, Any]] = []
    try:
        async with asyncio.timeout(DEADLOCK_GUARD_S):
            while True:
                frame = await stream.__anext__()
                seen.append(frame)
                if frame.get("type") == "snapshot":
                    return seen
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_a_reconnect_receives_the_acknowledgement_exactly_once(tmp_path: Path) -> None:
    """Idempotence on reconnect, in both directions, keyed by the client's cursor.

    The frame is published through the bridge, so it takes the bridge's own
    monotone ``seq`` and enters its replay buffer; that is what makes "a client
    that lost its socket during the engage is told once" a property rather than
    a hope. Both halves are asserted because only the pair is idempotent: a
    cursor BEFORE the frame must receive it exactly once, and a cursor AT it — a
    client that already painted it — must receive it not at all.
    """
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(harness.session_id, tmp_path, sync_on_connect=True, answer_prompts=True)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        try:
            async with _Attached(harness) as viewer:
                request_id = str(uuid.uuid4())
                response = await _post(harness, request_id=request_id)
                assert response.status_code == 200, response.text
                await viewer.wait_for(lambda: bool(viewer.acked()))
                ack = viewer.acked()[0]

                replayed = await _replay_until_snapshot(viewer.bridge, after_seq=0)
                assert [f.get("type") for f in replayed].count(ADMISSION_ACCEPTED_FRAME) == 1

                already_seen = await _replay_until_snapshot(viewer.bridge, after_seq=ack["seq"])
                assert [f.get("type") for f in already_seen].count(ADMISSION_ACCEPTED_FRAME) == 0
        finally:
            await owner.stop()


@pytest.mark.asyncio
async def test_a_refused_admission_resolves_the_acknowledgement(tmp_path: Path) -> None:
    """An owner that leaves mid-admission leaves NO acknowledgement hanging.

    THE CELL WHOSE ABSENCE MADE CI GREEN ON A DEFECT (review round 1, R1 / QA Q1).
    The acknowledgement is published before the door, so the refusal that follows
    it — here a runtime whose socket closes under the admission, which is the
    ``runtime_unreachable`` 503 a caller sees — must be published on the same
    stream, carrying the same ``request_id``, or every viewer of that stream
    (including a second one reading the replay) holds a promise that never
    resolves. Asserted on the wire in both directions: the caller's 503 and the
    viewer's outcome frame, and the fact that they name the SAME request.
    """
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(
            harness.session_id,
            tmp_path,
            sync_on_connect=True,
            stand_down_on_prompt=True,
        )
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        try:
            async with _Attached(harness) as viewer:
                request_id = str(uuid.uuid4())
                response = await _post(harness, request_id=request_id)

                assert response.status_code == 503, response.text
                assert response.json()["detail"]["code"] == "runtime_unreachable"
                assert owner.stand_downs == 1, "the owner must have refused, not been absent"

                await viewer.wait_for(lambda: bool(viewer.failed()))
                acked = viewer.acked()
                failed = viewer.failed()
                assert len(acked) == 1, "the acknowledgement precedes the refusal"
                assert len(failed) == 1, "and the refusal resolves it exactly once"
                assert failed[0]["payload"]["request_id"] == request_id
                assert failed[0]["payload"]["request_id"] == acked[0]["payload"]["request_id"]
                assert failed[0]["payload"]["status"] == "failed"
                assert failed[0]["payload"]["detail"] == (
                    "failed; the session owner could not be reached"
                ), "the vetted sentence, never the transport's own text"
                assert acked[0]["seq"] < failed[0]["seq"], "and it arrives after the ack"

                replayed = await _replay_until_snapshot(viewer.bridge, after_seq=0)
                kinds = [f.get("type") for f in replayed]
                assert kinds.count(ADMISSION_ACCEPTED_FRAME) == 1
                assert (
                    kinds.count(ADMISSION_FAILED_FRAME) == 1
                ), "a viewer that reconnects mid-refusal reads BOTH frames"
        finally:
            await owner.stop()


@pytest.mark.asyncio
async def test_a_submit_nobody_was_told_about_reports_no_outcome(tmp_path: Path) -> None:
    """No acknowledgement, no outcome frame — the pair is all-or-nothing.

    The refusal path publishes only what it ANNOUNCED (``announced``), because
    inventing an outcome for a submit no viewer was told about would show a
    failure for a request this host never took — an unknown session here, a
    latched daemon in production. Asserted as an absence on the stream, which is
    the only place it could appear.
    """
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(
            harness.session_id,
            tmp_path,
            sync_on_connect=True,
            stand_down_on_prompt=True,
        )
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        try:
            # NO VIEWER IS ATTACHED, so there is no bridge to publish on and
            # nothing to acknowledge: the submit is refused (the owner stands
            # down) with no frame anywhere, which is the honest state — nobody was
            # told the request was taken.
            assert harness.pool.bridges == {}
            response = await _post(harness, request_id=str(uuid.uuid4()))
            assert response.status_code == 503, response.text

            # And the viewer that arrives afterwards reads a FRESH bridge: the
            # stream the failing submit could have written to never existed, so
            # neither frame can be read back from it.
            async with _Attached(harness) as viewer:
                assert viewer.acked() == []
                assert viewer.failed() == [], "nothing was acknowledged, so nothing resolves"
        finally:
            await owner.stop()


@pytest.mark.asyncio
async def test_a_duplicate_submit_is_announced_once_and_replays_its_receipt(
    tmp_path: Path,
) -> None:
    """One request id, one acknowledgement — and the receipt still replays.

    The announcement is published BEFORE the receipt journal claims the request,
    so the journal's own at-most-once guard cannot be what makes this true: the
    bridge's bounded memory of what it has announced is. Both halves are asserted
    because either alone passes on a defect — a host that announced every submit
    could still replay the receipt, and a host that never announced would have
    nothing to duplicate.
    """
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(harness.session_id, tmp_path, sync_on_connect=True, answer_prompts=True)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        try:
            async with _Attached(harness) as viewer:
                request_id = str(uuid.uuid4())
                first = await _post(harness, request_id=request_id)
                assert first.status_code == 200, first.text
                assert first.json()["result"]["replayed"] is False

                second = await _post(harness, request_id=request_id)
                assert second.status_code == 200, second.text
                assert second.json()["result"]["replayed"] is True

                await viewer.wait_for(lambda: len(viewer.acked()) >= 1)
                assert len(viewer.acked()) == 1, "one submit is announced once"
                assert len(owner.prompts) == 1, "and the owner still saw one turn"
        finally:
            await owner.stop()


@pytest.mark.asyncio
async def test_announcing_takes_no_reference_and_starts_nothing(tmp_path: Path) -> None:
    """``announce`` publishes on a RESIDENT bridge without acquiring one.

    That is the mechanism, so it is asserted directly rather than inferred from
    a latency: no bridge user is taken (a taken reference is what lets a pool
    dispose a facade under a live request), the cold facade is untouched, and a
    session with NO resident bridge answers ``False`` without building one —
    which is what makes this safe to call before the door. The repeated
    correlation id answers ``False`` too, which is the bounded memory in one
    line.
    """
    async with _Harness(tmp_path) as harness:
        announced = harness.pool.announce_admission(
            harness.session_id, request_id="no-viewer", mode="prompt"
        )
        assert await announced is False
        assert harness.pool.bridges == {}, "announcing built a bridge nobody asked for"

        owner = _FakeOwner(harness.session_id, tmp_path, sync_on_connect=True)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        try:
            async with _Attached(harness) as viewer:
                bridge = viewer.bridge
                before_remote = bridge.remote
                users_before = bridge.users
                assert (
                    await harness.pool.announce_admission(
                        harness.session_id, request_id="no-viewer", mode="prompt"
                    )
                    is True
                )
                assert bridge.users == users_before, "announcing took a reference"
                assert bridge.remote is before_remote, "announcing engaged a facade"
                assert (
                    await harness.pool.announce_admission(
                        harness.session_id, request_id="no-viewer", mode="prompt"
                    )
                    is False
                ), "a repeated correlation id is announced once"
                await viewer.wait_for(lambda: len(viewer.acked()) == 1)
        finally:
            await owner.stop()


@pytest.mark.asyncio
async def test_a_latched_daemon_announces_nothing(tmp_path: Path) -> None:
    """A leaving daemon must not say "taken" and then refuse.

    The refusal itself is the pool's door and has its own tests; what this cell
    pins is that the acknowledgement cannot outrun it, because a frame telling a
    viewer its submit was accepted followed by a 503 telling it the daemon is
    leaving is the one contradiction an acknowledgement must never produce.
    """
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(harness.session_id, tmp_path, sync_on_connect=True)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        try:
            async with _Attached(harness) as viewer:
                harness.pool.retiring_probe = lambda: True
                assert (
                    await harness.pool.announce_admission(
                        harness.session_id, request_id="latched", mode="prompt"
                    )
                    is False
                )
                assert viewer.acked() == []
        finally:
            await owner.stop()

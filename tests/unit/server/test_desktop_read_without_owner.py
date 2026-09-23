"""A desktop ROUTE answers a read without an answering owner (design D1-D4).

The facade's own contract is pinned in
``tests/unit/session/test_ownerless_read.py``; this file is the half the operator
actually hit — the HTTP routes, the 200 instead of the 503, the latency against
the budget, the ``cold_reason``/``attaching`` tokens on the wire, and the fact
that no read path spawns.

THE MEASUREMENT IT REPRODUCES. Isolated config root, synthetic session id, a
live-but-silent owner (a socket that accepts the dial, sends the welcome and then
answers nothing). Before this change ``GET /v1/desktop/sessions/{id}`` and
``/history`` answered ``503 Session owner is unavailable. Reconnect and reconcile
before retrying.`` after 15.1-15.2 s; the same durable rows were readable in
0.02 s with no owner at all. After it: 200, cold, ``owner-silent``, inside the
budget, with the same rows on ``/history``.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.server.routes import (
    desktop_catalogues,
    desktop_lifecycle,
    desktop_sessions,
)
from local_operator.server.utils.desktop_sessions import (
    DesktopSessionBridge,
    DesktopSessions,
)

# The presence hint's own bound, private to the facade: the beat's documented
# envelope is ``READ_ATTACH_BUDGET_S`` + this, and a test that hard-coded 5 s
# would keep passing after the constant moved (review round 2, MINOR-1).
from local_operator.session.attached import (
    _DESKTOP_WATCH_ACK_BOUND_S as DESKTOP_WATCH_ACK_BOUND_S,
)
from local_operator.session.attached import (
    DESKTOP_CONTROL_ATTACH_S,
    READ_ATTACH_BUDGET_S,
    AttachedSession,
)
from local_operator.session.runtime import launch, registry
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import ScriptedStream, build_session
from tests.unit.session.test_ownerless_read import _FakeOwner, _publish_live, _seed

TOKEN = "synthetic-desktop-token"
#: Upper bound on an awaited event, never a budget to sleep through.
DEADLOCK_GUARD_S = 20.0


class _Harness:
    """One pool, one real router app, one synthetic session and its cwd."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.app = FastAPI()
        self.app.include_router(desktop_sessions.router)
        # The durable session-scoped GETs live in these two routers (MINOR-5), and
        # a read-envelope row is only worth asserting against the real route.
        self.app.include_router(desktop_lifecycle.router)
        self.app.include_router(desktop_catalogues.router)
        self.pool = DesktopSessions(root)
        self.app.state.desktop_sessions = self.pool
        self.inputs = root / "workspace"
        self.inputs.mkdir(parents=True, exist_ok=True)
        # The receipts journal behind the control routes resolves its store
        # through app state; ``host()`` still prefers the pool above.
        from local_operator.config import ConfigManager

        # The catalogue routes resolve slash-command AUTH through app state; the
        # isolated manager above keeps them off the operator's own credentials.
        self.app.state.config_manager = ConfigManager(config_dir=root)
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
def _desktop_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every row runs against a synthetic HOME, not the operator's.

    ``/mcp`` resolves its rows through ``load_all_mcp_configs``, which reads the
    user scope out of ``Path.home()`` (``~/.codex/config.toml``, ``~/.claude.json``
    and friends) as well as the project scope. Without this, the row quietly reads
    the machine it is running on — read-only, but a test that asserts isolation
    and then opens the operator's own config is worse than one that does not
    claim it (review round 2, NIT-3).
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)


async def _classified(bridge: DesktopSessionBridge) -> None:
    """Wait for the bridge's read attempt to CLASSIFY, on the event, not the clock.

    The first frame no longer waits for the attempt (``READ_FIRST_FRAME_GRACE_S``),
    so a test that asserts the classified token must wait for the thing that
    produces it: the attempt's own task settling, or its dial being retained.
    """
    deadline = time.monotonic() + READ_ATTACH_BUDGET_S + DEADLOCK_GUARD_S
    while time.monotonic() < deadline:
        task = bridge.read_attach_task
        remote = bridge.remote
        if task is None or task.done() or (remote is not None and remote.attaching):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the read attempt never settled")


async def _get(client: AsyncClient, url: str) -> tuple[int, float, dict[str, Any]]:
    started = time.monotonic()
    response = await client.get(url)
    elapsed = time.monotonic() - started
    try:
        body = response.json()
    except ValueError:
        body = {}
    return response.status_code, elapsed, body


@pytest.mark.asyncio
async def test_a_bridge_read_serves_a_silent_owner_cold_with_its_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE PROBE at the call the route makes, with no HTTP in the way.

    ``DesktopSessionBridge.acquire`` is where the reported failure happened:
    ``attach_existing`` -> ``_bind_to`` raised ``OwnerAckTimeout`` out of a dial
    into a live-but-silent owner, and the route ladder turned that into the 503.
    The read mode must return the cold facade, carry the reason it is cold, and
    leave the durable rows readable — in the same process, from the same session
    directory that had them all along.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(harness.session_id, tmp_path)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        bridge = DesktopSessionBridge(tmp_path, harness.session_id, str(harness.inputs))

        started = time.monotonic()
        await bridge.acquire(read=True)
        elapsed = time.monotonic() - started

        assert elapsed < READ_ATTACH_BUDGET_S + 1.0, f"a read waited {elapsed:.2f}s"
        await _classified(bridge)
        snapshot = await bridge.snapshot()
        assert snapshot["payload"]["cold"] is True
        assert snapshot["payload"]["cold_reason"] == "owner-silent"
        assert snapshot["payload"]["attaching"] is True
        history = await bridge.history(limit=10)
        assert len(history["entries"]) == 4
        await bridge.release()
        await owner.stop()


@pytest.mark.asyncio
async def test_the_read_routes_answer_200_for_a_silent_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's route, end to end: 200, cold, and the reason named."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        owner = _FakeOwner(harness.session_id, tmp_path)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        base = f"/v1/desktop/sessions/{harness.session_id}"

        status, elapsed, body = await _get(harness.client, base)
        assert status == 200, body
        assert elapsed < READ_ATTACH_BUDGET_S + 1.0, f"the snapshot waited {elapsed:.2f}s"
        payload = body["result"]["payload"]
        assert payload["cold"] is True
        assert payload["attaching"] is True
        # The first frame no longer waits for the attempt, so under load it can
        # predate the CLASSIFICATION and carry the documented unclassified
        # default; it may never claim a live owner or a runtime that is leaving.
        assert payload["cold_reason"] in {"owner-silent", "no-runtime"}
        # The attempt classifies within its own budget, and every later read
        # (the retained dial is ``attaching``) names the live-but-silent owner.
        deadline = time.monotonic() + READ_ATTACH_BUDGET_S + DEADLOCK_GUARD_S
        while payload["cold_reason"] != "owner-silent" and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
            _status, _elapsed, body = await _get(harness.client, base)
            payload = body["result"]["payload"]
        assert payload["cold_reason"] == "owner-silent"
        assert payload["attaching"] is True

        status, elapsed, body = await _get(harness.client, f"{base}/history")
        assert status == 200, body
        assert elapsed < READ_ATTACH_BUDGET_S + 1.0, f"the history read waited {elapsed:.2f}s"
        assert len(body["result"]["entries"]) == 4
        await owner.stop()


@pytest.mark.asyncio
async def test_a_session_with_no_owner_at_all_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fast cold read stays fast, and now says WHY it was cold."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        assert registry.scan(tmp_path) == []
        base = f"/v1/desktop/sessions/{harness.session_id}"

        status, elapsed, body = await _get(harness.client, base)

        assert status == 200, body
        assert elapsed < 1.0, f"a cold read with no owner waited {elapsed:.2f}s"
        payload = body["result"]["payload"]
        assert payload["cold"] is True
        assert payload["cold_reason"] == "no-runtime"
        assert payload["attaching"] is False


@pytest.mark.asyncio
async def test_a_healthy_owner_is_still_a_live_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bounded attempt must not turn a working owner into a cold paint.

    This is the half of the trade the design rejected a 0 s budget for: a healthy
    owner's canonical state — a turn in flight, a pending gate — reaches the
    panel's first frame exactly as it did before.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        directory = tmp_path / "sessions" / harness.session_id
        owner_session = build_session(directory, ScriptedStream([]), cwd=harness.inputs)
        handle = ServingSessionHandle(
            owner_session, asyncio.get_running_loop(), cwd=str(harness.inputs)
        )
        server = RuntimeServer(handle, kind="daemon")
        await server.start_in_process()
        await asyncio.sleep(0.2)
        (directory / ".session.pid").write_text(str(registry.scan(tmp_path)[0][0].pid))
        try:
            status, _elapsed, body = await _get(
                harness.client, f"/v1/desktop/sessions/{harness.session_id}"
            )
        finally:
            await server.aclose()

        assert status == 200, body
        payload = body["result"]["payload"]
        assert payload["cold"] is False, "a healthy owner was served cold"
        assert payload["cold_reason"] is None
        assert payload["attaching"] is False


@pytest.mark.asyncio
async def test_the_events_route_opens_for_a_silent_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/events`` is acquired BEFORE the headers are written.

    So the old value failed the STREAM the same way, and the renderer then spent
    its own 23.5 s of retry delays before painting a lost connection. The stream
    must open; the generator's body is stubbed because ``ASGITransport`` buffers
    until the app returns and the real one is an SSE loop.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        owner = _FakeOwner(harness.session_id, tmp_path)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        acquired: list[bool] = []
        real_acquire = DesktopSessionBridge.acquire

        async def spy(self: DesktopSessionBridge, *, read: bool = False) -> Any:
            acquired.append(read)
            return await real_acquire(self, read=read)

        async def no_frames(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            if False:  # pragma: no cover — keeps this an async generator
                yield {}

        monkeypatch.setattr(DesktopSessionBridge, "acquire", spy)
        monkeypatch.setattr(DesktopSessionBridge, "events", no_frames)
        url = f"/v1/desktop/sessions/{harness.session_id}/events"

        status, _elapsed, body = await _get(harness.client, url)

        assert status == 200, body
        assert acquired == [True], "the stream did not take the read envelope"
        await owner.stop()


@pytest.mark.asyncio
async def test_a_rollover_reaches_the_stream_after_a_late_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D2.2 on the wire: the frame the renderer consumes, on the stream it holds.

    A cold paint that never becomes live is only half a fix. The read opens the
    stream with ``cold``/``attaching``; the owner answers a moment later, and the
    subscription must receive a ``frontend.update`` carrying the new epoch — the
    rollover the renderer already handles for a canonical epoch change.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(harness.session_id, tmp_path)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        frames: list[dict[str, Any]] = []

        async with harness.pool.session(harness.session_id, read=True) as bridge:
            subscription = bridge.subscribe()
            stream = bridge.events(subscription, epoch=None, after_seq=0)
            try:
                snapshot = await asyncio.wait_for(stream.__anext__(), timeout=DEADLOCK_GUARD_S)
                while snapshot["type"] != "snapshot":
                    snapshot = await asyncio.wait_for(stream.__anext__(), timeout=DEADLOCK_GUARD_S)
                frames.append(snapshot)
                # Classified, or the documented unclassified default (see above).
                assert snapshot["payload"]["cold_reason"] in {"owner-silent", "no-runtime"}

                await owner.send_sync()

                deadline = time.monotonic() + DEADLOCK_GUARD_S
                while time.monotonic() < deadline:
                    frame = await asyncio.wait_for(stream.__anext__(), timeout=DEADLOCK_GUARD_S)
                    frames.append(frame)
                    if frame["type"] == "frontend.update" and frame["payload"]["epoch"] != (
                        snapshot["payload"]["frontend"]["epoch"]
                    ):
                        break
                else:  # pragma: no cover — the loop only exits by break
                    raise AssertionError("no rollover frame arrived")
            finally:
                await stream.aclose()

        rollover = frames[-1]
        assert rollover["type"] == "frontend.update"
        assert rollover["payload"]["epoch"] == "fake-owner"
        assert rollover["payload"]["cold"] is False
        assert bridge.remote is None or bridge.remote.cold_reason is None
        await owner.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("mute", [False, True], ids=["welcomes-then-silent", "never-welcomes"])
async def test_a_control_route_answers_a_silent_owner_busy_and_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mute: bool
) -> None:
    """D-F4: a live-but-silent owner is ``runtime_busy`` inside the control envelope.

    Before: the welcome waited ``ACK_TIMEOUT_S`` and the sync
    ``FRONTEND_SYNC_FOREGROUND_S``, so ``POST /messages`` answered a generic
    ``503 runtime_unreachable`` at 15.0-15.7 s -- most of the renderer's 20 s
    deadline, for a runtime that was alive the whole time. Both silent shapes
    are driven: one that welcomes and never syncs, and one that never welcomes
    (the SIGSTOPped shape, which only a deadline over the DIAL can bound).

    The copy is unchanged (D8): ``message`` keeps the exact sentence the shipped
    app matches by prefix; ``code``/``retryable``/``retry_after_ms`` and the
    ``Retry-After`` header are additive.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        owner = _FakeOwner(harness.session_id, tmp_path, mute=mute)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        url = f"/v1/desktop/sessions/{harness.session_id}/messages"

        started = time.monotonic()
        response = await harness.client.post(
            url, json={"request_id": str(uuid.uuid4()), "text": "hi"}
        )
        elapsed = time.monotonic() - started

        assert response.status_code == 503, response.text
        assert elapsed < DESKTOP_CONTROL_ATTACH_S + 1.5, f"the refusal took {elapsed:.2f}s"
        detail = response.json()["detail"]
        assert detail["code"] == "runtime_busy"
        assert detail["retryable"] is True
        assert detail["retry_after_ms"] == 2000
        assert response.headers["retry-after"] == "2"
        assert detail["message"].startswith("Session owner is unavailable.")
        await owner.stop()


@pytest.mark.asyncio
async def test_a_control_route_still_names_an_unreachable_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OTHER refusal keeps its code: a record whose socket refuses the dial.

    ``runtime_busy`` is reserved for an owner that is alive and reachable; an
    owner nobody can connect to is not busy, and telling the renderer to retry it
    would be the overstatement the code split exists to avoid.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        owner = _FakeOwner(harness.session_id, tmp_path)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        # The record stays published and the pid alive, but the port is closed.
        await owner.stop()
        url = f"/v1/desktop/sessions/{harness.session_id}/messages"

        response = await harness.client.post(
            url, json={"request_id": str(uuid.uuid4()), "text": "hi"}
        )

        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "runtime_unreachable"
        assert "retryable" not in detail
        assert detail["message"].startswith("Session owner is unavailable.")


@pytest.mark.asyncio
async def test_a_read_route_never_engages_a_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D4 over the real route: a GET is side-effect free.

    Every route in this path is reachable from a sidebar sweep, so a read that
    kicked a warm would spawn for rows nobody clicked. The patch is on the module
    ``attached`` imports from, because that import happens inside the bind.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        spawned: list[Any] = []

        async def _forbidden(*args: Any, **kwargs: Any) -> Any:
            spawned.append((args, kwargs))
            raise AssertionError("a read route spawned a runtime")

        monkeypatch.setattr(launch, "engage_runtime", _forbidden)
        base = f"/v1/desktop/sessions/{harness.session_id}"

        for url in (base, f"{base}/history"):
            status, _elapsed, body = await _get(harness.client, url)
            assert status == 200, body

        assert spawned == [], "a read route attempted to engage a runtime"


@pytest.mark.asyncio
async def test_a_read_route_declares_the_read_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which routes are reads is a decision, so it is asserted route by route.

    The write paths keep the control envelope deliberately: a request that was
    not admitted must be able to say so, and only a read has a durable answer to
    fall back on.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        budgets: list[tuple[str, float | None]] = []
        real = AttachedSession.attach_existing

        async def spy(
            self: AttachedSession,
            *,
            budget: float | None = None,
            control_budget: float | None = None,
        ) -> bool:
            budgets.append(("call", budget))
            return await real(self, budget=budget, control_budget=control_budget)

        monkeypatch.setattr(AttachedSession, "attach_existing", spy)
        base = f"/v1/desktop/sessions/{harness.session_id}"

        await _get(harness.client, base)
        await _get(harness.client, f"{base}/history")
        # ``/interrupt`` is the control half, and it is safe to drive with no
        # owner: the route answers ``idle`` without dialling (its own review
        # round 1 MAJOR-1) — but it still ACQUIRES, which is what is under test.
        await harness.client.post(f"{base}/interrupt", json={"request_id": str(uuid.uuid4())})

        assert budgets == [
            ("call", READ_ATTACH_BUDGET_S),
            ("call", READ_ATTACH_BUDGET_S),
            ("call", None),
        ], f"the route envelopes are wrong: {budgets}"


@pytest.mark.asyncio
async def test_the_snapshot_payload_still_validates_for_an_older_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3's additive rule, asserted rather than promised.

    A host that does not track the distinction (an in-process one, a test's
    stand-in) builds the same payload without the two new keys, and it must still
    validate — which is what makes the field safe for a renderer that predates it.
    """
    from local_operator.server.models.desktop_sessions import SnapshotPayload

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        response = await harness.client.get(f"/v1/desktop/sessions/{harness.session_id}")
        payload = response.json()["result"]["payload"]

        trimmed = {
            key: value for key, value in payload.items() if key not in ("cold_reason", "attaching")
        }
        validated = SnapshotPayload.model_validate(trimmed)
        assert validated.cold is True
        # And the documented fallback for a reader that never saw the fields.
        assert validated.cold_reason is None
        assert validated.attaching is False


@pytest.mark.asyncio
async def test_a_mute_owner_is_served_cold_by_the_route_inside_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MAJOR-1 over HTTP: a wedged owner (accepted socket, no welcome) is bounded.

    The silent owner in the tests above answers the dial and then withholds the
    sync; this one never writes anything, which is the SIGSTOPped shape. The
    budget has to cover the WELCOME leg too, or the read answers cold at the
    right status and the wrong latency (`ACK_TIMEOUT_S`, 15 s).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        owner = _FakeOwner(harness.session_id, tmp_path, mute=True)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        url = f"/v1/desktop/sessions/{harness.session_id}"

        status, elapsed, body = await _get(harness.client, url)

        assert status == 200, body
        assert elapsed < READ_ATTACH_BUDGET_S + 1.0, f"a mute owner cost the read {elapsed:.2f}s"
        # First frame: classified, or the documented unclassified default.
        assert body["result"]["payload"]["cold_reason"] in {"owner-silent", "no-runtime"}
        # Held across the attempt (a mounted stream does exactly this), the
        # token is the honest one once the attempt has spent its budget.
        async with harness.pool.session(harness.session_id, read=True) as bridge:
            if bridge.read_attach_task is not None:
                await asyncio.wait_for(asyncio.shield(bridge.read_attach_task), DEADLOCK_GUARD_S)
            snapshot = await bridge.snapshot()
        assert snapshot["payload"]["cold_reason"] == "owner-silent"
        await owner.stop()


@pytest.mark.asyncio
async def test_the_watch_beat_answers_200_for_a_silent_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MINOR-2: the presence beat is a read-envelope route, and it is asserted.

    `/watch` is the route the renderer calls every 15 s, so a 503 here is the
    panel reporting a lost connection for a session that is running. The beat
    needs a live subscription to address — an unknown id is a 404 by design — so
    the row holds one open exactly as a mounted stream does.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        owner = _FakeOwner(harness.session_id, tmp_path)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        async with harness.pool.session(harness.session_id, read=True) as held:
            subscription = held.subscribe()
            started = time.monotonic()
            response = await harness.client.post(
                f"/v1/desktop/sessions/{harness.session_id}/watch",
                json={
                    "subscription_id": subscription.id,
                    "visible": False,
                    "can_notify": False,
                },
            )
            elapsed = time.monotonic() - started
        assert response.status_code == 200, response.text
        assert response.json()["result"]["lease_seconds"] == 45
        # The beat's envelope is its ATTACH budget plus the presence hint's own
        # bound, and the doc says so for this one route: the hint is a lease
        # renewal (``_DESKTOP_WATCH_ACK_BOUND_S``, 5 s), not part of serving the
        # read, so it is not clamped by ``READ_ATTACH_BUDGET_S``. Asserted rather
        # than described, because the round-2 review found the doc claiming the
        # read budget here while the route measured 5.04 s for a silent owner
        # (review round 2, MINOR-1).
        assert (
            elapsed < READ_ATTACH_BUDGET_S + DESKTOP_WATCH_ACK_BOUND_S + 1.0
        ), f"the beat took {elapsed:.2f}s, past its documented envelope"
        await owner.stop()


#: The session-scoped GETs whose answer exists without an owner, with the marker
#: that proves the answer came from the COLD source rather than from a runtime.
#: ``/skills`` and ``/command-entities`` carry no cold/live distinction in their
#: payload — their rows are discovered from disk either way — so their marker is
#: the shape of the answer itself.
_DURABLE_SESSION_READS = (
    ("/v1/desktop/sessions/{session}/mcp", {"data": {"cold": True}}),
    # A SILENT owner is not an absent one: the cold payload says the namespace is
    # UNREAD (retryable), not observed-and-empty (review round 2, MINOR-2). The
    # absent case keeps the observed/empty reading and has its own test below.
    ("/v1/desktop/sessions/{session}/variables", {"data": {"state": "busy"}}),
    ("/v1/desktop/skills?session_id={session}", {}),
    # Marker-less on purpose, like ``/skills``: every field in this payload is a
    # literal in the route or a config default, so no value here discriminates a
    # cold answer from a live one and asserting one would be a test that cannot
    # fail (review round 2, NIT-1). The row's content is the 200 inside the
    # budget.
    ("/v1/desktop/sessions/{session}/failovers", {}),
    (
        "/v1/desktop/sessions/{session}/command-entities?command=approvals",
        {"command": "approvals", "entities": [{"value": "auto"}, {"value": "ask"}]},
    ),
)


def _assert_subset(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    """``expected`` is nested inside ``actual``, key by key."""
    for key, value in expected.items():
        assert key in actual, f"{key!r} missing from {actual}"
        if isinstance(value, dict):
            assert isinstance(actual[key], dict), f"{key!r} is not a mapping: {actual[key]!r}"
            _assert_subset(value, actual[key])
        else:
            assert actual[key] == value, f"{key!r} was {actual[key]!r}, expected {value!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize(("template", "marker"), _DURABLE_SESSION_READS)
async def test_a_durable_session_read_answers_200_for_a_silent_owner(
    template: str, marker: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MINOR-5: the answer exists without an owner, so the owner must not refuse it.

    Each of these GETs already served a COMPLETE answer cold — the MCP row and the
    variables panel branch on ``is_cold`` explicitly — while a live-but-silent
    owner turned the same request into a 503 after ~15 s. That asymmetry is the
    operator's report surviving on the MCP row, the model/effort pickers and the
    failover chips.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        owner = _FakeOwner(harness.session_id, tmp_path)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        url = template.format(session=harness.session_id)

        status, elapsed, body = await _get(harness.client, url)

        assert status == 200, body
        assert elapsed < READ_ATTACH_BUDGET_S + 1.0, f"{url} waited {elapsed:.2f}s"
        _assert_subset({"result": marker} if marker else {}, body)
        await owner.stop()


@pytest.mark.asyncio
async def test_a_durable_session_read_is_unchanged_for_a_healthy_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LIVE branch of the same reads, against a real in-process owner.

    Read mode changes only the acquire envelope, so a bound owner must still be
    answered from its own runtime rather than from the checkpoint. ``/mcp`` is
    deliberately absent: its live branch routes a ``desktop_mcp`` slash over the
    socket, which this in-process harness does not implement, and the diff does
    not touch the branch itself (it changes only how the bridge is acquired).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        directory = tmp_path / "sessions" / harness.session_id
        owner_session = build_session(directory, ScriptedStream([]), cwd=harness.inputs)
        handle = ServingSessionHandle(
            owner_session, asyncio.get_running_loop(), cwd=str(harness.inputs)
        )
        server = RuntimeServer(handle, kind="daemon")
        await server.start_in_process()
        await asyncio.sleep(0.2)
        (directory / ".session.pid").write_text(str(registry.scan(tmp_path)[0][0].pid))
        try:
            for template in (
                "/v1/desktop/sessions/{session}/variables",
                "/v1/desktop/skills?session_id={session}",
                "/v1/desktop/sessions/{session}/failovers",
                "/v1/desktop/sessions/{session}/command-entities?command=approvals",
            ):
                status, _elapsed, body = await _get(
                    harness.client, template.format(session=harness.session_id)
                )
                assert status == 200, (template, body)
        finally:
            await server.aclose()


@pytest.mark.asyncio
async def test_variables_keeps_the_observed_empty_reading_when_no_pid_holds_the_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MINOR-2's other half: the absent case is genuinely absent.

    ``variables: []`` means observed and empty, never unknown, so the payload only
    keeps that reading where it is true. With no owner at all there is no
    namespace to read and the panel's "no code memory yet" is correct; with a
    runtime holding the lease and not answering, the same payload would render
    that sentence over a namespace nobody read — which is the `data.state: "busy"`
    row in the silent-owner case above.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        status, elapsed, body = await _get(
            harness.client, f"/v1/desktop/sessions/{harness.session_id}/variables"
        )

        assert status == 200, body
        assert elapsed < READ_ATTACH_BUDGET_S + 1.0
        assert body["result"]["data"] == {
            "state": "observed",
            "runtime": "absent",
            "kernel": "absent",
            "variables": [],
            "truncated": False,
        }


@pytest.mark.asyncio
async def test_the_mcp_row_answers_from_the_projects_own_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NIT-3: the row's isolation is asserted, not claimed.

    ``load_all_mcp_configs`` reads the user scope out of ``Path.home()``, so
    without a redirected HOME this row answers from the operator's own
    ``~/.codex/config.toml``. With it, the synthetic project file is the only
    source and its path is the assertion — which also makes the row discriminate
    something the cold flag alone does not (review round 2, NIT-3).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        project = harness.inputs / ".mcp.json"
        project.write_text(
            json.dumps({"mcpServers": {"synthetic-one": {"command": "true"}}}), encoding="utf-8"
        )
        owner = _FakeOwner(harness.session_id, tmp_path)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)

        status, _elapsed, body = await _get(
            harness.client, f"/v1/desktop/sessions/{harness.session_id}/mcp"
        )

        assert status == 200, body
        data = body["result"]["data"]
        assert data["cold"] is True, data
        assert [server["name"] for server in data["servers"]] == ["synthetic-one"], data["servers"]
        assert data["servers"][0]["source"] == str(project), data["servers"][0]
        await owner.stop()


@pytest.mark.asyncio
async def test_a_read_never_queues_behind_a_control_attach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-F1: reads answer while a control call is still attaching to a mute owner.

    The reported 17-20 s: a first-keystroke ``/warm`` (or a send) dials an owner
    whose loop does not answer, and every read of that conversation issued in the
    meantime queued on the bridge lock behind it, then paid its own read budget.
    The mute owner is the SIGSTOPped shape. The bound asserted is the operator's
    300 ms, with the control call demonstrably still in flight when the reads
    answer.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        owner = _FakeOwner(harness.session_id, tmp_path, mute=True)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        base = f"/v1/desktop/sessions/{harness.session_id}"
        control = asyncio.create_task(
            harness.client.post(
                f"{base}/messages", json={"request_id": str(uuid.uuid4()), "text": "hi"}
            )
        )
        # Wait on the EVENT, not the clock: the control call is inside its dial.
        deadline = time.monotonic() + DEADLOCK_GUARD_S
        while owner.conns == 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert owner.conns >= 1, "the control call never dialled"

        for url in (base, f"{base}/history"):
            status, elapsed, body = await _get(harness.client, url)
            assert status == 200, body
            assert elapsed < 0.3, f"{url} queued behind the control attach: {elapsed:.2f}s"
        assert not control.done(), "the reads were not measured during the control attach"

        response = await asyncio.wait_for(control, timeout=DEADLOCK_GUARD_S)
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "runtime_busy"
        await owner.stop()


@pytest.mark.asyncio
async def test_a_busy_owner_is_painted_cold_at_once_and_goes_live_behind_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-F2: the first frame does not spend ``READ_ATTACH_BUDGET_S`` on a silent owner.

    The read answers inside the first-frame grace with the cold facade and
    ``attaching`` set; when the owner answers, the stream receives the
    ``frontend.replace`` whose ``cold`` flag the shipped renderer reads.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(harness.session_id, tmp_path)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)

        started = time.monotonic()
        async with harness.pool.session(harness.session_id, read=True) as bridge:
            elapsed = time.monotonic() - started
            subscription = bridge.subscribe(frontend_replace=True)
            stream = bridge.events(subscription, epoch=None, after_seq=0)
            try:
                frame = await asyncio.wait_for(stream.__anext__(), timeout=DEADLOCK_GUARD_S)
                while frame["type"] != "snapshot":
                    frame = await asyncio.wait_for(stream.__anext__(), timeout=DEADLOCK_GUARD_S)
                assert elapsed < 0.3, f"the first frame waited {elapsed:.2f}s"
                assert frame["payload"]["cold"] is True
                assert frame["payload"]["cold_reason"] in {"owner-silent", "no-runtime"}
                assert frame["payload"]["attaching"] is True
                # The cold page is FILLED, so first paint needs no /history.
                assert len(frame["payload"]["history"]["entries"]) == 4

                await owner.send_sync()
                deadline = time.monotonic() + DEADLOCK_GUARD_S
                while time.monotonic() < deadline:
                    frame = await asyncio.wait_for(stream.__anext__(), timeout=DEADLOCK_GUARD_S)
                    if frame["type"] == "frontend.replace" and frame["payload"]["cold"] is False:
                        break
                else:  # pragma: no cover — the loop only exits by break
                    raise AssertionError("no live frontend.replace arrived")
            finally:
                await stream.aclose()
        await owner.stop()


@pytest.mark.asyncio
async def test_a_healthy_owner_lands_inside_the_first_frame_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of D-F2's trade: a live owner still paints LIVE first.

    The grace exists so a healthy owner's attach (tens of ms) is in the first
    frame; a read that always answered cold would flash "attaching" over every
    running conversation.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    async with _Harness(tmp_path) as harness:
        owner = _FakeOwner(harness.session_id, tmp_path, answer_watch=True, sync_on_connect=True)
        await owner.start()
        _publish_live(tmp_path, owner, session_id=harness.session_id)
        async with harness.pool.session(harness.session_id, read=True) as bridge:
            snapshot = await bridge.snapshot()
        assert snapshot["payload"]["cold"] is False, "a healthy owner was painted cold"
        await owner.stop()


@pytest.mark.asyncio
async def test_a_cancelled_control_acquire_gives_its_reference_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F4 (review round 1): acquire's failure path must not leak its reference.

    The interleaving that pinned the bridge: a control acquire is cancelled
    mid-attach, its failure handler awaits ``release``, the bridge lock is held
    by another route on the same session, and a SECOND cancellation lands while
    the release waits for that lock. Unshielded, the cancellation went into the
    release and ``users`` stayed incremented forever, so the bridge could never
    be evicted. Shielded, the release completes once the lock frees.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    attaching = asyncio.Event()

    async def parked_attach(**_: Any) -> bool:
        attaching.set()
        await asyncio.Event().wait()
        return False  # pragma: no cover — only ever cancelled

    async with pool.session(sid, read=True) as bridge:
        assert bridge.remote is not None
        monkeypatch.setattr(bridge.remote, "attach_existing", parked_attach)
        acquiring = asyncio.create_task(bridge.acquire())
        await asyncio.wait_for(attaching.wait(), timeout=DEADLOCK_GUARD_S)
        assert bridge.users == 2

        # Another route holds the bridge lock, so the failure path's release
        # has to wait for it; the second cancellation lands inside that wait.
        await bridge.lock.acquire()
        try:
            acquiring.cancel()
            for _ in range(5):
                await asyncio.sleep(0)
            acquiring.cancel()
            with pytest.raises(asyncio.CancelledError):
                await acquiring
        finally:
            bridge.lock.release()
        for _ in range(5):
            await asyncio.sleep(0)
        assert bridge.users == 1, "the cancelled acquire kept its reference"
    assert bridge.users == 0
    await pool.close()


@pytest.mark.asyncio
async def test_a_settle_behind_a_newer_attempt_still_corrects_its_cold_paint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F3 (review round 1): the outran fact belongs to the attempt, not the bridge.

    Reader A answers cold ahead of attempt T1. T1 settles, and before its
    done-callback runs, reader B finds T1 done and the facade still cold and
    starts T2. With one bridge-level flag, starting T2 cleared it and T1's
    callback published nothing, so A's pane stayed cold with no transition.
    B's start is modelled at the exact point the reviewer named, the gap
    between T1 finishing and its callback, by wrapping the callback rather than
    racing the loop.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    gates: list[asyncio.Event] = []

    async def gated_attach(**_: Any) -> bool:
        gate = asyncio.Event()
        gates.append(gate)
        await gate.wait()
        return False

    async with pool.session(sid, read=True) as bridge:
        remote = bridge.remote
        assert remote is not None and remote.is_cold
        monkeypatch.setattr(remote, "attach_existing", gated_attach)
        published: list[int] = []
        monkeypatch.setattr(bridge, "publish_frontend_replace", lambda: published.append(1))
        original = bridge._read_attach_settled
        newer: list[asyncio.Task[bool] | None] = []

        def settled(owner: AttachedSession, task: asyncio.Task[bool]) -> None:
            if not newer:
                # Reader B, between T1 finishing and T1's callback.
                newer.append(bridge._start_read_attach(owner))
            original(owner, task)

        monkeypatch.setattr(bridge, "_read_attach_settled", settled)

        await bridge.acquire(read=True)  # reader A, outruns T1
        first = bridge.read_attach_task
        assert first is not None and not first.done()
        gates[0].set()
        await asyncio.wait_for(asyncio.shield(first), timeout=DEADLOCK_GUARD_S)
        for _ in range(5):
            await asyncio.sleep(0)
        second = newer[0]
        assert second is not None and second is not first, "B did not start a new attempt"
        assert published == [1], "T1's settle lost the frame that corrects A's cold paint"

        # T2 was outrun by nobody, so its own settle owes no frame.
        gates[1].set()
        await asyncio.wait_for(asyncio.shield(second), timeout=DEADLOCK_GUARD_S)
        for _ in range(5):
            await asyncio.sleep(0)
        assert published == [1]
        assert not bridge.read_attach_outran
        await bridge.release()
    await pool.close()

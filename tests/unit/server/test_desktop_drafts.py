"""The new-chat draft pre-engage: mint, the scoped door, and create acceptance.

PR-2a's own suite (design-pr2.md §1.7). The properties pinned here are the
lifecycle's EDGES rather than its happy path, because the edges are what a later
edit can break silently:

* **single-use** — a draft is consumed by the create that materialises it, so a
  second create naming it can never answer with a DIFFERENT conversation;
* **expiry / eviction resolve as unknown** — a stale pane's send still works
  (fresh id), while an id that already became a conversation refuses (422)
  instead of duplicating;
* **the containment** — only the five opted-in doors (snapshot, history, watch,
  warm, events) can see a registered draft, and nothing a refused route does can
  materialise one;
* **the warm's residue** — the engage boots the runtime in the draft's own
  directory, leaving exactly ``.execution-lease`` + ``.session.pid`` (measured),
  so ``create`` must tolerate that directory; materialisation is read off
  ``desktop.json``, which only ``create`` writes.

Route tests use a minimal app over the test's own root (``draft_app``, the same
shape as ``test_desktop_sessions.py``'s ``draft_api``/``move_api``); pool tests
drive ``DesktopSessions`` directly. NO TEST MAY SPAWN A RUNTIME: every path that
would engage is either not reached or has ``AttachedSession._ensure_bound``
replaced with a recorder (the pattern ``test_desktop_sessions.py`` established
for the warm tests).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.harness.types import ModelSpec
from local_operator.server.retire import RETIRING_STATE_ATTR
from local_operator.server.routes import capabilities, desktop_sessions
from local_operator.server.utils import desktop_sessions as module
from local_operator.server.utils.desktop_sessions import (
    DRAFT_COUNT_MAX,
    DRAFT_TTL_S,
    DesktopSessions,
    DraftAlreadyMaterialised,
)
from local_operator.session.attached import AttachedSession
from local_operator.session.runtime import registry as runtime_registry
from local_operator.session.runtime.types import SessionRecord

#: The session-id shape a minted draft must have (``SESSION_ID`` in the pool).
DRAFT_ID_SHAPE = re.compile(r"[a-f0-9]{12}\Z")


def _mint_body(cwd: Path, **extra: Any) -> dict[str, Any]:
    """A valid mint body over ``cwd``; ``request_id`` fresh unless overridden."""
    return {"request_id": str(uuid.uuid4()), "cwd": str(cwd), **extra}


async def _until(predicate: Callable[[], bool], why: str, timeout: float = 10.0) -> None:
    """Poll a structural fact with a real-clock bound (never a turn count).

    ``time`` here is the REAL clock: these tests patch the pool module's ``time``
    where they need a frozen one, not this module's.
    """
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, why
        await asyncio.sleep(0.01)


@pytest_asyncio.fixture
async def draft_app(tmp_path: Path, monkeypatch):
    """A minimal app over THIS test's config root, with the pool ATTACHED.

    The same shape as ``test_desktop_sessions.py``'s ``draft_api``/``move_api``
    (own ``tmp_path``, every ``CMUX_*`` stripped, the pool closed after the
    client), with two deliberate differences: the pool is constructed up front
    rather than lazily, because several tests read the registry and the bridge
    cache back through ``app.state.desktop_sessions`` AFTER a route call; and
    its retirement probe reads ``app.state`` exactly as the real ``host()``
    wiring does, so a test can latch the daemon.
    """
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "draft-warm-test")
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.desktop_sessions = DesktopSessions(
        tmp_path,
        retiring=lambda: bool(getattr(app.state, RETIRING_STATE_ATTR, False)),
    )
    app.include_router(desktop_sessions.router)
    app.include_router(capabilities.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer draft-warm-test"},
    ) as client:
        yield client, app, tmp_path.resolve()
    if hasattr(app.state, "desktop_sessions"):
        await app.state.desktop_sessions.close()


async def _mint(client: AsyncClient, root: Path, **extra: Any) -> str:
    response = await client.post("/v1/desktop/sessions/draft", json=_mint_body(root, **extra))
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert DRAFT_ID_SHAPE.fullmatch(result["draft_id"]), result
    return result["draft_id"]


def _write_warm_residue(root: Path, draft_id: str) -> Path:
    """The directory the deferred engage leaves, as the probe measured it.

    Simulated rather than spawned: a unit test must not start a runtime, and the
    exact contents are the fact under test (``.execution-lease`` +
    ``.session.pid``, no transcript, no marker).
    """
    residue = root / "sessions" / draft_id
    residue.mkdir(parents=True)
    (residue / ".execution-lease").write_text('{"pid": 1}', encoding="utf-8")
    (residue / ".session.pid").write_text("1", encoding="utf-8")
    return residue


# ---------------------------------------------------------------------------
# The mint route
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_mint_registers_a_draft_and_writes_nothing(draft_app) -> None:
    """§1.1: allocate + validate + register, nothing else — no dir, no runtime.

    The pane's id must be usable against the five doors immediately, and the
    mint itself must cost nothing durable: the one row a retry needs is the
    receipt's, and even that only claims on the not-yet-recorded path.
    """
    client, app, root = draft_app
    body = _mint_body(root)
    response = await client.post("/v1/desktop/sessions/draft", json=body)
    assert response.status_code == 200, response.text
    draft_id = response.json()["result"]["draft_id"]
    assert DRAFT_ID_SHAPE.fullmatch(draft_id), draft_id
    assert draft_id in app.state.desktop_sessions.drafts
    assert not (root / "sessions").exists(), "a mint must not create a session"
    from local_operator.session.runtime import registry

    assert registry.scan(root) == [], "a mint must not leave a runtime record"
    # The retry-safety row exists (the ONLY durable trace a mint leaves)...
    from local_operator.server.utils.desktop_receipts import DesktopReceipts

    assert DesktopReceipts(root).recorded("draft:" + body["request_id"]) is True
    # ...and the capability key is what a renderer gates the whole path on.
    capabilities_response = await client.get("/v1/capabilities")
    assert (
        capabilities_response.json()["result"]["features"]["session_draft_warm"] == 1
    ), "the draft-warm op is gated on this key, so it must be published"


@pytest.mark.asyncio
async def test_a_mint_retry_replays_the_same_id(draft_app) -> None:
    """§1.1: one pane, one id — a StrictMode double-fire must not warm twice.

    Two ids for one pane would warm two runtimes and leave a registry entry
    nobody can consume; the receipt is what makes the retry at-most-once. A
    DIFFERENT request id is a different pane and must mint a fresh draft.
    """
    client, app, root = draft_app
    body = _mint_body(root)
    first = await client.post("/v1/desktop/sessions/draft", json=body)
    second = await client.post("/v1/desktop/sessions/draft", json=body)
    assert first.status_code == second.status_code == 200
    assert first.json()["result"]["draft_id"] == second.json()["result"]["draft_id"]
    assert second.json()["result"]["replayed"] is True
    third = await client.post("/v1/desktop/sessions/draft", json=_mint_body(root))
    assert third.json()["result"]["draft_id"] != first.json()["result"]["draft_id"]
    assert len(app.state.desktop_sessions.drafts) == 2


@pytest.mark.asyncio
async def test_a_latched_daemon_mints_nothing(draft_app) -> None:
    """§1.6: the latch covers the mint exactly as it covers create/warm.

    And the refusal lands BEFORE the receipt is claimed — a claimed-then-refused
    mint would answer the client's retry with "outcome indeterminate".
    """
    client, app, root = draft_app
    app.state.serve_retiring = True
    response = await client.post("/v1/desktop/sessions/draft", json=_mint_body(root))
    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == "daemon-retiring"
    assert app.state.desktop_sessions.drafts == {}


@pytest.mark.asyncio
async def test_a_mint_admits_the_same_refusals_as_create(draft_app) -> None:
    """One body gets one answer from the mint and from create (cwd first)."""
    client, app, root = draft_app
    missing = root / "no-such-directory"
    for route in ("/v1/desktop/sessions/draft", "/v1/desktop/sessions"):
        body = _mint_body(missing)
        response = await client.post(route, json=body)
        assert response.status_code == 409, (route, response.status_code, response.text)
    assert app.state.desktop_sessions.drafts == {}


# ---------------------------------------------------------------------------
# The scoped door (§1.3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_opted_in_doors_serve_a_registered_draft(draft_app, monkeypatch) -> None:
    """snapshot / history / watch / events answer a draft as their cold shapes.

    The bridge is built from the draft's own spec — there is no directory to
    read until create — and the stream body is stubbed because ``ASGITransport``
    buffers a response until the app returns while a real SSE generator only ends
    when the client leaves (the pattern ``test_desktop_sessions.py`` uses).
    """
    client, app, root = draft_app
    pool = app.state.desktop_sessions
    draft_id = await _mint(client, root)

    snapshot = await client.get(f"/v1/desktop/sessions/{draft_id}")
    assert snapshot.status_code == 200, snapshot.text
    assert snapshot.json()["result"]["session_id"] == draft_id
    history = await client.get(f"/v1/desktop/sessions/{draft_id}/history")
    assert history.status_code == 200, history.text
    assert history.json()["result"]["entries"] == []
    # The pane's /events stand-in: a watch beat must carry a LIVE subscription
    # id (the renderer's first beat fires on mount, after its stream has
    # subscribed), so one is created on the draft bridge exactly as the stream
    # route would create it.
    bridge = pool.bridges[draft_id]
    sub = bridge.subscribe()
    watch = await client.post(
        f"/v1/desktop/sessions/{draft_id}/watch",
        json={"subscription_id": sub.id, "visible": False, "can_notify": False},
    )
    assert watch.status_code == 200, watch.text

    async def no_frames(*args: Any, **kwargs: Any):
        if False:  # pragma: no cover — keeps this an async generator
            yield {}

    monkeypatch.setattr(module.DesktopSessionBridge, "events", no_frames)
    events = await client.get(f"/v1/desktop/sessions/{draft_id}/events")
    assert events.status_code == 200, events.text

    assert draft_id in pool.bridges, "no door reached the draft"


@pytest.mark.asyncio
async def test_the_doors_404_a_draft_that_expired(draft_app, monkeypatch) -> None:
    """Expiry resolves as UNKNOWN: read doors 404, exactly like a random id."""
    client, app, root = draft_app
    clock = {"now": 500.0}
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
    draft_id = await _mint(client, root)
    clock["now"] = 500.0 + DRAFT_TTL_S + 1.0
    snapshot = await client.get(f"/v1/desktop/sessions/{draft_id}")
    assert snapshot.status_code == 404, snapshot.text
    watch = await client.post(
        f"/v1/desktop/sessions/{draft_id}/watch",
        json={"subscription_id": uuid.uuid4().hex, "visible": False, "can_notify": False},
    )
    assert watch.status_code == 404, watch.text
    assert draft_id not in app.state.desktop_sessions.drafts


@pytest.mark.asyncio
async def test_every_other_door_refuses_a_registered_draft(draft_app) -> None:
    """THE CONTAINMENT: a draft id is not an authorization surface (§1.3).

    Every route that did not opt in refuses it — including a route that would
    otherwise find the pane's RESIDENT bridge in the cache, which is why the
    registry is asked before the cache — and nothing the refusals touch may
    materialise the draft (no directory, still registered).
    """
    client, app, root = draft_app
    draft_id = await _mint(client, root)
    # Make the bridge resident first (the opted-in door), so the refusals below
    # are tested against the cache-hit path as well as the cold one.
    resident = await client.get(f"/v1/desktop/sessions/{draft_id}")
    assert resident.status_code == 200
    assert draft_id in app.state.desktop_sessions.bridges

    refusals: list[tuple[str, str, dict[str, Any] | None]] = [
        (
            "POST",
            f"/v1/desktop/sessions/{draft_id}/messages",
            {"request_id": str(uuid.uuid4()), "text": "hello"},
        ),
        (
            "POST",
            f"/v1/desktop/sessions/{draft_id}/commands",
            # A VALID command (`/session` bare), so the request survives the
            # route's pre-door command validation and the refusal measured here
            # is the DOOR's, not the command registry's.
            {"request_id": str(uuid.uuid4()), "command": "session", "args": ""},
        ),
        (
            "POST",
            f"/v1/desktop/sessions/{draft_id}/interrupt",
            {"request_id": str(uuid.uuid4())},
        ),
        (
            "POST",
            f"/v1/desktop/sessions/{draft_id}/asides",
            {"request_id": str(uuid.uuid4()), "text": "hello"},
        ),
        (
            "POST",
            f"/v1/desktop/sessions/{draft_id}/working-directory",
            {"request_id": str(uuid.uuid4()), "cwd": str(root)},
        ),
        (
            "GET",
            f"/v1/desktop/sessions/{draft_id}/children/0123456789ab/transcript",
            None,
        ),
    ]
    for method, url, payload in refusals:
        response = await client.request(method, url, json=payload)
        assert response.status_code == 404, f"{url}: {response.status_code} {response.text[:200]}"

    assert not (root / "sessions").exists(), "a refused route materialised the draft"
    assert draft_id in app.state.desktop_sessions.drafts, "a refused route consumed the draft"


@pytest.mark.asyncio
async def test_a_visible_watch_beat_arms_the_drafts_warm(draft_app, monkeypatch) -> None:
    """§1.3: the lease is what keeps the intent across the engage — for drafts too.

    The first visible beat arms the lease-warm loop, which engages through the
    ordinary background bind. ``_ensure_bound`` is recorded rather than run: a
    unit test must not spawn a runtime, and what is under test is that the draft
    bridge reaches the SAME warm seam a session's does.
    """
    client, app, root = draft_app
    pool = app.state.desktop_sessions
    draft_id = await _mint(client, root)
    engaged: list[tuple[Any, ...]] = []

    async def record_engage(self: Any, *, foreground: bool = True) -> None:
        # ``self`` because this is a CLASS patch: a plain function set on the
        # class binds as a method, and a recorder that did not take the facade
        # would raise a TypeError that ``warm_runtime``'s own ``except``
        # swallows — a dead instrument returning "task finished" (caught by
        # driving the same recorder through an instance patch first).
        engaged.append(("engage", foreground))

    monkeypatch.setattr(AttachedSession, "_ensure_bound", record_engage)
    # HOLD THE BRIDGE, as the pane's /events subscription does for the stream's
    # life: a warm whose only user is its own request is cancelled on release
    # (``warm()``'s docstring states it), which is exactly what the bench rig's
    # bridge-holding rule exists to prevent.
    async with pool.session(draft_id, read=True, allow_draft=True) as bridge:
        sub = bridge.subscribe()
        response = await client.post(
            f"/v1/desktop/sessions/{draft_id}/watch",
            json={"subscription_id": sub.id, "visible": True, "can_notify": False},
        )
        assert response.status_code == 200, response.text
        assert pool.bridges[draft_id] is bridge
        await _until(
            lambda: bool(engaged),
            why="a visible lease did not arm the draft bridge's warm",
        )
        assert engaged[0] == ("engage", False), "the lease warm must be a BACKGROUND bind"
        assert bridge.warm_task is not None


@pytest.mark.asyncio
async def test_the_warm_route_engages_a_draft(draft_app, monkeypatch) -> None:
    """The pane's keystroke warms through the one existing op, unmodified."""
    client, app, root = draft_app
    pool = app.state.desktop_sessions
    draft_id = await _mint(client, root)
    engaged: list[tuple[Any, ...]] = []

    async def record_engage(self: Any, *, foreground: bool = True) -> None:
        # ``self`` because this is a CLASS patch: a plain function set on the
        # class binds as a method, and a recorder that did not take the facade
        # would raise a TypeError that ``warm_runtime``'s own ``except``
        # swallows — a dead instrument returning "task finished" (caught by
        # driving the same recorder through an instance patch first).
        engaged.append(("engage", foreground))

    monkeypatch.setattr(AttachedSession, "_ensure_bound", record_engage)
    # HOLD THE BRIDGE across the warm request — the same second user the pane's
    # /events subscription is. Without it the warm is cancelled the instant the
    # request releases the last reference, and this test would measure the
    # cancellation rather than the engage.
    async with pool.session(draft_id, read=True, allow_draft=True) as holder:
        assert holder.remote is not None
        response = await client.post(f"/v1/desktop/sessions/{draft_id}/warm", json={})
        assert response.status_code == 200, response.text
        assert response.json()["result"]["state"] == "warming"
        await _until(lambda: bool(engaged), why="the warm never reached the engage seam")
        assert engaged[0] == ("engage", False)
        bridge = pool.bridges[draft_id]
        await bridge.warm_task


# ---------------------------------------------------------------------------
# create acceptance (§1.4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_materialises_the_draft_id_and_consumes_it(draft_app) -> None:
    """The warm pays off: create uses the id, writes the marker, spends the draft.

    A SECOND create naming the same draft — a client that lost the first answer
    and retried with a fresh request id — must refuse rather than mint a
    DIFFERENT conversation under the user's nose.
    """
    client, app, root = draft_app
    draft_id = await _mint(client, root)
    created = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "cwd": str(root), "draft_id": draft_id},
    )
    assert created.status_code == 200, created.text
    assert created.json()["result"]["session_id"] == draft_id
    assert draft_id not in app.state.desktop_sessions.drafts, "the draft was not consumed"
    marker_path = root / "sessions" / draft_id / "desktop.json"
    assert json.loads(marker_path.read_text(encoding="utf-8"))["cwd"] == str(root)

    second = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "cwd": str(root), "draft_id": draft_id},
    )
    assert second.status_code == 422, second.text
    assert second.json()["detail"]["code"] == "draft_already_materialised"
    # And the refusal left the ONE materialised session alone.
    assert sorted(p.name for p in (root / "sessions").iterdir()) == [draft_id]


@pytest.mark.asyncio
async def test_create_tolerates_the_warm_residue_directory(draft_app) -> None:
    """The engage booted a runtime in the draft's own directory; create still works.

    The measured residue (``.execution-lease`` + ``.session.pid``) must not
    refuse the materialisation — a create that insisted on an empty directory
    would refuse every WARMED draft its own create, the one path this feature
    exists to serve — and the marker, not the directory, is what decides that a
    draft was materialised.
    """
    client, app, root = draft_app
    draft_id = await _mint(client, root)
    residue = _write_warm_residue(root, draft_id)
    created = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "cwd": str(root), "draft_id": draft_id},
    )
    assert created.status_code == 200, created.text
    assert created.json()["result"]["session_id"] == draft_id
    assert (residue / "desktop.json").is_file()
    # The runtime's own bookkeeping survives untouched: those files are its.
    assert (residue / ".execution-lease").exists()


@pytest.mark.asyncio
async def test_create_with_an_unknown_draft_id_mints_fresh(draft_app) -> None:
    """A stale pane's send STILL WORKS: unknown → fresh id, no error (§1.4)."""
    client, app, root = draft_app
    stale = "0123456789ab"
    created = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "cwd": str(root), "draft_id": stale},
    )
    assert created.status_code == 200, created.text
    session_id = created.json()["result"]["session_id"]
    assert session_id != stale
    assert (root / "sessions" / session_id / "desktop.json").is_file()


@pytest.mark.asyncio
async def test_a_malformed_draft_id_is_a_422_and_never_a_path(draft_app) -> None:
    """The declaration's ``pattern`` is the gate: a bad value is a 422, not an id."""
    client, app, root = draft_app
    for bad in ("../../etc", "no", "0123456789AB", "0123456789abcd"):
        response = await client.post(
            "/v1/desktop/sessions",
            json={"request_id": str(uuid.uuid4()), "cwd": str(root), "draft_id": bad},
        )
        assert response.status_code == 422, (bad, response.status_code, response.text)
    assert not (root / "sessions").exists()


# ---------------------------------------------------------------------------
# The pool's own lifecycle edges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_use_is_enforced_by_the_pool_itself(tmp_path) -> None:
    """Callers that reach the pool directly get the same single-use discipline."""
    pool = DesktopSessions(tmp_path)
    draft_id = await pool.mint_draft(str(tmp_path))
    assert await pool.create(str(tmp_path), draft_id=draft_id) == draft_id
    with pytest.raises(DraftAlreadyMaterialised) as raised:
        await pool.create(str(tmp_path), draft_id=draft_id)
    assert raised.value.code == "draft_already_materialised"
    assert [p.name for p in (tmp_path / "sessions").iterdir()] == [draft_id]
    await pool.close()


@pytest.mark.asyncio
async def test_a_lost_registry_falls_back_to_a_fresh_id(tmp_path) -> None:
    """Daemon restart: registry gone, warm residue on disk — the send still works.

    The residue is NOT a materialisation (no marker), so the create mints fresh
    rather than refusing: the warm was wasted, and the user's message still
    lands in a session of its own.
    """
    first = DesktopSessions(tmp_path)
    draft_id = await first.mint_draft(str(tmp_path))
    _write_warm_residue(tmp_path, draft_id)
    await first.close()
    # The restart: same root, a new pool — the registry is empty.
    second = DesktopSessions(tmp_path)
    session_id = await second.create(str(tmp_path), draft_id=draft_id)
    assert session_id != draft_id
    assert (tmp_path / "sessions" / session_id / "desktop.json").is_file()
    await second.close()


@pytest.mark.asyncio
async def test_an_expired_draft_resolves_as_unknown_and_frees_its_id(tmp_path, monkeypatch) -> None:
    """TTL: past ``DRAFT_TTL_S`` the id is unknown (fresh create, 404 door)."""
    clock = {"now": 100.0}
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
    pool = DesktopSessions(tmp_path)
    draft_id = await pool.mint_draft(str(tmp_path))
    clock["now"] = 100.0 + DRAFT_TTL_S + 1.0
    with pytest.raises(KeyError):
        async with pool.session(draft_id, allow_draft=True):
            pass  # pragma: no cover — the door refuses on entry
    assert draft_id not in pool.drafts, "an expired draft was not pruned where it was asked"
    session_id = await pool.create(str(tmp_path), draft_id=draft_id)
    assert session_id != draft_id
    await pool.close()


@pytest.mark.asyncio
async def test_the_cap_evicts_the_oldest_draft(tmp_path, monkeypatch) -> None:
    """DRAFT_COUNT_MAX: the newest wins, the oldest id resolves as unknown."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
    pool = DesktopSessions(tmp_path)
    ids: list[str] = []
    for index in range(DRAFT_COUNT_MAX + 1):
        ids.append(await pool.mint_draft(str(tmp_path)))
        clock["now"] += 1.0
    assert len(pool.drafts) == DRAFT_COUNT_MAX
    assert ids[0] not in pool.drafts, "the oldest draft survived the cap"
    assert ids[-1] in pool.drafts
    session_id = await pool.create(str(tmp_path), draft_id=ids[0])
    assert session_id != ids[0], "an evicted draft must resolve as unknown"
    await pool.close()


@pytest.mark.asyncio
async def test_a_draft_bridge_is_built_from_the_draft_spec(tmp_path) -> None:
    """§1.3: the bridge's birth selection is the MINTED spec, not a marker read.

    The marker does not exist until ``create``; reading ``session_id`` at
    facade time would birth the first turn on the configured default instead of
    the pane's pick. The override flag rides along exactly as the marker path's
    does, so the child PINS the choice.
    """
    pool = DesktopSessions(tmp_path)
    spec = ModelSpec(provider="anthropic", model_id="claude-opus-5", reasoning_effort="high")
    draft_id = await pool.mint_draft(str(tmp_path), model=spec)
    async with pool.session(draft_id, read=True, allow_draft=True) as bridge:
        assert bridge.remote is not None
        assert bridge.remote.is_cold
        assert bridge.remote._birth_model == spec
        assert bridge.remote._model_selection_override is True
        assert bridge.cwd == str(tmp_path)
    # The containment at the pool level, with the bridge now RESIDENT: the
    # ordinary door must refuse the registered draft even though the cache
    # would serve it, and the opted-in door must reuse the same bridge.
    with pytest.raises(KeyError):
        async with pool.session(draft_id):
            pass  # pragma: no cover — the door refuses on entry
    async with pool.session(draft_id, allow_draft=True) as again:
        assert again is bridge
    await pool.close()


@pytest.mark.asyncio
async def test_a_draft_materialised_elsewhere_refuses_before_the_registry(tmp_path) -> None:
    """The read-only refusal probe: registered passes, materialised raises.

    Exercised directly because it is the ONE predicate ``create``'s route
    pre-flight and ``create`` itself share; if it ever answered differently
    from the resolution below it, the route's "refusal precedes the claim"
    contract would quietly stop holding.
    """
    pool = DesktopSessions(tmp_path)
    draft_id = await pool.mint_draft(str(tmp_path))
    # A registered draft is fine (it is unmaterialised by definition) and the
    # probe must not SPEND it.
    pool.assert_draft_unmaterialised(draft_id)
    assert draft_id in pool.drafts
    # An unregistered id with no directory is fine (mint fresh later).
    pool.assert_draft_unmaterialised("0123456789ab")
    # An unregistered id whose marker is on disk refuses.
    materialised = tmp_path / "sessions" / "abcdefabcdef"
    materialised.mkdir(parents=True)
    (materialised / "desktop.json").write_text('{"version": 1, "cwd": "/"}', encoding="utf-8")
    with pytest.raises(DraftAlreadyMaterialised):
        pool.assert_draft_unmaterialised("abcdefabcdef")
    # The engine's residue directory (no marker) does NOT refuse.
    _write_warm_residue(tmp_path, "0123456789cd")
    pool.assert_draft_unmaterialised("0123456789cd")
    await pool.close()


# ---------------------------------------------------------------------------
# M1 (round-1 review): a draft whose ENTRY is gone must be unknown everywhere
# ---------------------------------------------------------------------------


async def _door_matrix(client: AsyncClient, draft_id: str, root: Path) -> dict[str, int]:
    """Every door this file drives for one id, as ``{label: status}``.

    The five opted-in doors first, then the mutating/foreign ones the
    containment exists for — the same set round-1 review's
    ``repro_expired_draft_bridge.py`` drove, so the rig and this file cannot
    disagree about what "every door" means.
    """
    calls: dict[str, tuple[str, str, dict[str, Any] | None]] = {
        "snapshot": ("GET", f"/v1/desktop/sessions/{draft_id}", None),
        "history": ("GET", f"/v1/desktop/sessions/{draft_id}/history", None),
        "events": ("GET", f"/v1/desktop/sessions/{draft_id}/events", None),
        "watch": (
            "POST",
            f"/v1/desktop/sessions/{draft_id}/watch",
            {"subscription_id": uuid.uuid4().hex, "visible": False, "can_notify": False},
        ),
        "warm": ("POST", f"/v1/desktop/sessions/{draft_id}/warm", {}),
        "messages": (
            "POST",
            f"/v1/desktop/sessions/{draft_id}/messages",
            {"request_id": str(uuid.uuid4()), "text": "hello"},
        ),
        "commands": (
            "POST",
            f"/v1/desktop/sessions/{draft_id}/commands",
            {"request_id": str(uuid.uuid4()), "command": "session", "args": ""},
        ),
        "interrupt": (
            "POST",
            f"/v1/desktop/sessions/{draft_id}/interrupt",
            {"request_id": str(uuid.uuid4())},
        ),
        "working-directory": (
            "POST",
            f"/v1/desktop/sessions/{draft_id}/working-directory",
            {"request_id": str(uuid.uuid4()), "cwd": str(root)},
        ),
    }
    statuses: dict[str, int] = {}
    for label, (method, url, body) in calls.items():
        response = await client.request(method, url, json=body)
        statuses[label] = response.status_code
    return statuses


@pytest.mark.asyncio
async def test_an_expired_drafts_resident_bridge_is_dropped_and_refused_everywhere(
    draft_app, monkeypatch
) -> None:
    """M1 (round-1 review): expiry must be indistinguishable from unknown.

    The pane holds /events, so the draft's bridge is RESIDENT when the registry
    entry expires. Before the fix the containment could not fire (it asks the
    live registry) and the cached bridge served every door — snapshot/history
    200, commands/interrupt/working-directory 200, and ``POST messages``
    reached ``_ensure_bound(foreground=True)``, a mutating route engaging a
    materialising draft. Round 1's own rig is the source of this cell
    (``scratchpad/review1607/repro_expired_draft_bridge.py``).
    """
    client, app, root = draft_app
    pool = app.state.desktop_sessions
    engaged: list[str] = []

    async def record_engage(self: Any, *, foreground: bool = True) -> None:
        engaged.append(f"foreground={foreground}")

    monkeypatch.setattr(AttachedSession, "_ensure_bound", record_engage)
    draft_id = await _mint(client, root)
    resident = await client.get(f"/v1/desktop/sessions/{draft_id}")
    assert resident.status_code == 200
    assert draft_id in pool.bridges, "the pane's stream did not make a resident bridge"
    # While REGISTERED the containment still refuses a non-opted door.
    refused = await client.post(
        f"/v1/desktop/sessions/{draft_id}/messages",
        json={"request_id": str(uuid.uuid4()), "text": "hello"},
    )
    assert refused.status_code == 404

    clock = {"now": time.monotonic() + DRAFT_TTL_S + 1.0}
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock["now"]))

    statuses = await _door_matrix(client, draft_id, root)
    assert statuses == {label: 404 for label in statuses}, statuses
    assert draft_id not in pool.bridges, "the dead draft's bridge stayed resident"
    assert engaged == [], "a refused door reached the engage seam"
    with pytest.raises(KeyError):
        async with pool.session(draft_id):
            pass  # pragma: no cover — the door refuses on entry
    with pytest.raises(KeyError):
        async with pool.session(draft_id, allow_draft=True):
            pass  # pragma: no cover — the door refuses on entry


@pytest.mark.asyncio
async def test_a_warmed_expired_draft_leaves_nothing_servable(draft_app, monkeypatch) -> None:
    """M1's SECOND LEG: the residue directory must not re-open as a phantom.

    A warmed draft leaves ``sessions/<id>`` behind (the engage's residue), and
    once the bridge is dropped the LOCATE path is what a later ask walks. It
    used to answer that directory with the checkpoint fallback — a bridge on
    ``root.parent`` — so the id kept serving 200s after the drop. The residue
    is simulated exactly as measured (``.execution-lease`` + ``.session.pid``);
    a unit test must not spawn the runtime that writes it.
    """
    client, app, root = draft_app
    pool = app.state.desktop_sessions
    draft_id = await _mint(client, root)
    _write_warm_residue(root, draft_id)
    resident = await client.get(f"/v1/desktop/sessions/{draft_id}")
    assert resident.status_code == 200
    assert draft_id in pool.bridges

    clock = {"now": time.monotonic() + DRAFT_TTL_S + 1.0}
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock["now"]))

    statuses = await _door_matrix(client, draft_id, root)
    assert statuses == {label: 404 for label in statuses}, statuses
    assert draft_id not in pool.bridges, "the dead draft's bridge stayed resident"
    # And create still mints fresh — the warm was wasted, the send can go on.
    created = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "cwd": str(root), "draft_id": draft_id},
    )
    assert created.status_code == 200, created.text
    assert created.json()["result"]["session_id"] != draft_id


@pytest.mark.asyncio
async def test_a_restart_refuses_a_drafts_residue_and_still_creates_fresh(draft_app) -> None:
    """Post-restart (round-1 review's second requested cell).

    A restarted daemon has NO bridge cache and NO registry — only what the
    engage wrote on disk — and BEFORE this fix the locate fallback turned
    exactly that directory into a live-looking cold session with
    ``root.parent`` as its cwd. The replacement pool is what keeps the
    fixture's teardown honest: it closes ``app.state.desktop_sessions``.
    """
    client, app, root = draft_app
    old_pool = app.state.desktop_sessions
    draft_id = await old_pool.mint_draft(str(root))
    _write_warm_residue(root, draft_id)
    await old_pool.close()
    app.state.desktop_sessions = DesktopSessions(
        root, retiring=lambda: bool(getattr(app.state, RETIRING_STATE_ATTR, False))
    )

    statuses = await _door_matrix(client, draft_id, root)
    assert statuses == {label: 404 for label in statuses}, statuses
    created = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "cwd": str(root), "draft_id": draft_id},
    )
    assert created.status_code == 200, created.text
    assert created.json()["result"]["session_id"] != draft_id


@pytest.mark.asyncio
async def test_a_materialised_drafts_bridge_becomes_an_ordinary_session_bridge(draft_app) -> None:
    """The other side of M1's detection: create turns the id into a session.

    ``bridge.draft`` is cleared once the marker exists — the marker is the
    birth source from then on — and the resident bridge keeps serving, which is
    what the pane needs the moment its own create returns.
    """
    client, app, root = draft_app
    pool = app.state.desktop_sessions
    draft_id = await _mint(client, root)
    resident = await client.get(f"/v1/desktop/sessions/{draft_id}")
    assert resident.status_code == 200
    assert pool.bridges[draft_id].draft is not None
    created = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "cwd": str(root), "draft_id": draft_id},
    )
    assert created.status_code == 200, created.text
    again = await client.get(f"/v1/desktop/sessions/{draft_id}")
    assert again.status_code == 200, again.text
    assert pool.bridges[draft_id].draft is None, "the dead draft reference was not cleared"


# ---------------------------------------------------------------------------
# R2 (round-2 review): create-in-flight keeps serving; a bound draft stays
# unlisted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_create_in_flight_keeps_the_panes_bridge_serving(draft_app, monkeypatch) -> None:
    """R2-1: consumption precedes the marker; the gap is not death.

    ``create`` deletes the registry entry a beat before ``persist`` publishes
    the marker, and a door ask inside that window used to hit the M1 dead-draft
    branch (entry gone, no marker -> pop + close + 404) — a rare, transient
    regression the M1 fix itself introduced. The window is opened
    DETERMINISTICALLY here: the marker writer blocks until the test releases
    it, so the middle state is observable instead of raced for.
    """
    client, app, root = draft_app
    pool = app.state.desktop_sessions
    engaged: list[str] = []

    async def record_engage(self: Any, *, foreground: bool = True) -> None:
        engaged.append(f"foreground={foreground}")

    monkeypatch.setattr(AttachedSession, "_ensure_bound", record_engage)
    draft_id = await _mint(client, root)
    resident = await client.get(f"/v1/desktop/sessions/{draft_id}")
    assert resident.status_code == 200

    import threading

    blocked = threading.Event()
    release = threading.Event()
    real_writer = module.write_desktop_marker

    def slow_writer(path, *args, **kwargs):
        blocked.set()
        assert release.wait(timeout=15.0), "the test never released the marker write"
        return real_writer(path, *args, **kwargs)

    monkeypatch.setattr(module, "write_desktop_marker", slow_writer)
    create_call = asyncio.create_task(pool.create(str(root), draft_id=draft_id))
    try:
        await _until(blocked.is_set, why="the create never reached the marker write")
        # The window is OPEN: consumed, marker not yet published.
        assert draft_id not in pool.drafts
        assert pool.bridges[draft_id].draft is not None, "the birth spec was cleared early"
        during = await client.get(f"/v1/desktop/sessions/{draft_id}")
        assert during.status_code == 200, during.text
        # The other realistic caller is the 15 s beat, and a beat needs a
        # REGISTERED subscription — register one the way the pane's /events
        # stream does, then drive the REAL watch route with it. (The pool call
        # that gets the bridge is itself in-window: same door, same answer.)
        async with pool.session(draft_id, read=True, allow_draft=True) as bridge:
            sub = bridge.subscribe()
        beat = await client.post(
            f"/v1/desktop/sessions/{draft_id}/watch",
            json={"subscription_id": sub.id, "visible": True, "can_notify": False},
        )
        assert beat.status_code == 200, beat.text
    finally:
        release.set()
    created_id = await create_call
    assert created_id == draft_id
    after = await client.get(f"/v1/desktop/sessions/{draft_id}")
    assert after.status_code == 200, after.text
    assert pool.bridges[draft_id].draft is None, "the marker did not clear the draft reference"


def _publish_record(root: Path, session_id: str, name: str) -> Path:
    """A REAL live record for ``session_id``, keyed by this test process's pid.

    The draft-Q-2 phantom row comes from the catalogue's live branch, which
    reads exactly these records; the real ones are published by a runtime, and
    a unit test must not spawn one — so the record is constructed and published
    through the same writer, with a live pid (this process) and a fresh
    heartbeat, which is what ``registry.classify`` reads as ``live``.
    """
    record = SessionRecord(
        pid=os.getpid(),
        kind="daemon",
        session_id=session_id,
        conversation_name=name,
        cwd=str(root),
        model_label="",
        control_port=0,
        control_key="",
    )
    return runtime_registry.publish(record, root)


@pytest.mark.asyncio
async def test_a_bound_drafts_live_record_is_not_listed(draft_app) -> None:
    """Q-2 (QA round 2): the catalogue's live branch must not carry a draft.

    A bound warm publishes a runtime record, and ``decorate_rows`` appends a
    row for any record whose id has a session directory — so the draft listed
    as "Untitled conversation" while the doors called the id unknown. The
    record is published for real (a live pid and a fresh heartbeat), and the
    control is the same machinery over an ordinary session's id, which must
    still list: the fix filters DRAFTS, not live rows.
    """
    client, app, root = draft_app
    pool = app.state.desktop_sessions
    draft_id = await _mint(client, root)
    _write_warm_residue(root, draft_id)
    # The pane's bridge, so both of the filter's draft states are live at once.
    resident = await client.get(f"/v1/desktop/sessions/{draft_id}")
    assert resident.status_code == 200
    control_id = await pool.create(str(root))

    # The record really is one the live branch would read: the scan classifies
    # it as anything-but-stale, which is the source the phantom row came from.
    _publish_record(root, draft_id, "Untitled conversation")
    try:
        scanned = [rec.session_id for rec, state in runtime_registry.scan(root) if state != "stale"]
        assert draft_id in scanned, "the probe's own record was not live; the cell would be blind"
        page = await pool.list(limit=20)
        assert draft_id not in {row["id"] for row in page.rows}, "a bound draft was listed"
    finally:
        runtime_registry.unpublish(os.getpid(), root)

    # The same machinery over an ordinary session's id still appends its row:
    # the filter must not be "no live rows at all".
    _publish_record(root, control_id, "Untitled conversation")
    try:
        page = await pool.list(limit=20)
        assert control_id in {row["id"] for row in page.rows}, "ordinary live rows went missing"
    finally:
        runtime_registry.unpublish(os.getpid(), root)

    # Runtime exit: the record is gone, the residue directory stays — still
    # nothing, which is the state QA measured clean before and after.
    page = await pool.list(limit=20)
    assert draft_id not in {row["id"] for row in page.rows}

    # Materialised: the marker is the fact that makes the id ordinary, and the
    # row appears by the normal rules even while the stale bridge reference
    # still sits there (the filter must stand down on the MARKER, not on the
    # bridge).
    created = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": str(uuid.uuid4()), "cwd": str(root), "draft_id": draft_id},
    )
    assert created.status_code == 200, created.text
    assert created.json()["result"]["session_id"] == draft_id
    _publish_record(root, draft_id, "Untitled conversation")
    try:
        page = await pool.list(limit=20)
        assert draft_id in {row["id"] for row in page.rows}, "the materialised id stayed hidden"
        assert control_id in {row["id"] for row in page.rows}
    finally:
        runtime_registry.unpublish(os.getpid(), root)


@pytest.mark.asyncio
async def test_a_page_refills_past_a_bound_draft(draft_app) -> None:
    """C1 (round-2 review): the excluded draft must not eat a page slot.

    Measured on the first cut (the pool filtered the ASSEMBLED page): a store
    with 4 sessions + 1 bound draft answered ``limit=2`` with one row and
    ``limit=1`` with an empty page. The exclusion now runs inside the
    catalogue's ranking→window step, so every page fills from the rows behind
    the excluded one (and the same assertions fail on the pre-C1/cut tree).
    """
    client, app, root = draft_app
    pool = app.state.desktop_sessions
    draft_id = await _mint(client, root)
    _write_warm_residue(root, draft_id)
    resident = await client.get(f"/v1/desktop/sessions/{draft_id}")
    assert resident.status_code == 200
    for _ in range(4):
        await pool.create(str(root))
    _publish_record(root, draft_id, "Untitled conversation")
    try:
        for limit in (1, 2, 3):
            page = await pool.list(limit=limit)
            listed = [row["id"] for row in page.rows]
            assert draft_id not in listed
            assert len(listed) == limit, (limit, listed)
        page = await pool.list(limit=2)
        assert page.truncated is True, "more rows still follow the refilled page"
    finally:
        runtime_registry.unpublish(os.getpid(), root)

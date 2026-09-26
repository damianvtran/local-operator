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
            lambda: engaged, why="a visible lease did not arm the draft bridge's warm"
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
        response = await client.post(
            f"/v1/desktop/sessions/{draft_id}/warm", json={}
        )
        assert response.status_code == 200, response.text
        assert response.json()["result"]["state"] == "warming"
        await _until(lambda: engaged, why="the warm never reached the engage seam")
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
    draft_id = await pool.mint_draft(str(tmp_path), request_id="r1")
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
    draft_id = await first.mint_draft(str(tmp_path), request_id="r1")
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
    draft_id = await pool.mint_draft(str(tmp_path), request_id="r1")
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
        ids.append(await pool.mint_draft(str(tmp_path), request_id=f"r{index}"))
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
    draft_id = await pool.mint_draft(str(tmp_path), model=spec, request_id="r1")
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
    draft_id = await pool.mint_draft(str(tmp_path), request_id="r1")
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

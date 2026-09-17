"""The desktop pin route and the row's ``pinned`` flag.

Three things here are load-bearing rather than incidental, and each is a bug the
feature has actually been pictured shipping:

* **``pinned`` on EVERY row, both values.** The renderer's row merge is
  ``{...current, ...incoming}`` under the rule "an absent key is not a claim", so
  a list read that omitted ``pinned`` on an unpinned row would leave a stale
  optimistic ``true`` in place forever — the pin glyph and the section membership
  outliving a successful unpin made on another surface. The ``false`` rows are
  the ones that carry the assertion.

* **A delegated run is pinnable AND unpinnable.** The route deliberately does
  NOT validate with ``is_user_session``, which is exactly what its neighbour
  ``/seen`` does. The sidebar pins delegated runs, so such a check would answer
  404 for a pin the TUI can make and — the worst shape of bug in this feature —
  refuse the very action the user would take to remove it.

* **The route is a desired-state write, so a retry is a no-op that does not
  reorder.** Pinned at the store level too (``test_sidebar_pins.py``); repeated
  here through HTTP because the retry the design worries about is a retry of the
  REQUEST.

Everything runs against a real store on a real temporary filesystem. The one
thing stubbed is the app object's plumbing — the routes still go through
``errors()``, the real response models and the real ``DesktopSessions`` adapter.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.resume import ORIGIN_SUBAGENT, is_user_session, mark_session_origin
from local_operator.server.routes import capabilities, desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.tui.sidebar_pins import PINS_FILE, read_pins, toggle_pin


def _session(root: Path, session_id: str) -> Path:
    """An ordinary conversation the catalogue lists, seeded the way the desktop
    create route leaves one: a marker plus the transcript's parent directory."""
    path = root / "sessions" / session_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "created_at.json").write_text("1700000000")
    (path / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(root)}))
    return path


def _delegated_run(root: Path, session_id: str) -> Path:
    """A session a subagent started: same store, a subagent origin marker.

    Marked through the real writer rather than by hand, because the marker's
    shape is what ``is_user_session`` reads and a hand-written copy could agree
    with a wrong implementation.
    """
    path = _session(root, session_id)
    mark_session_origin(path, ORIGIN_SUBAGENT)
    return path


@pytest_asyncio.fixture
async def pins_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A minimal app over THIS test's config root, for the pin route.

    Same shape as ``move_api``/``draft_api`` in ``test_desktop_sessions.py``: its
    own ``tmp_path``, every ``CMUX_*`` stripped, ``app.state.desktop_sessions``
    reachable so a test can assert against the adapter as well as the JSON.
    """
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "pins-test-token")
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    app.include_router(desktop_sessions.router)
    app.include_router(capabilities.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer pins-test-token"},
    ) as client:
        yield client, tmp_path.resolve()
    if hasattr(app.state, "desktop_sessions"):
        await app.state.desktop_sessions.close()


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload["result"]["sessions"]


@pytest.mark.asyncio
async def test_the_route_sets_and_clears_a_pin(pins_api) -> None:
    client, root = pins_api
    session_id = "aaaaaaaaaaa1"
    _session(root, session_id)

    pinned = await client.post(f"/v1/desktop/sessions/{session_id}/pin", json={"pinned": True})
    assert pinned.status_code == 200, pinned.text
    assert pinned.json()["result"] == {"session_id": session_id, "pinned": True}
    assert read_pins(root) == [session_id]

    unpinned = await client.post(f"/v1/desktop/sessions/{session_id}/pin", json={"pinned": False})
    assert unpinned.status_code == 200, unpinned.text
    assert unpinned.json()["result"] == {"session_id": session_id, "pinned": False}
    # The store's resting state for "no pins" is an EMPTY ARRAY, not an absent
    # file — `sidebar_pins` states it, and the route must not invent a second
    # resting state by deleting the file.
    assert json.loads((root / PINS_FILE).read_text()) == []


@pytest.mark.asyncio
async def test_a_retried_pin_does_not_reorder(pins_api) -> None:
    """The retry the desired-state body exists for.

    Pin A, pin B (so B leads), then repeat A's request — the file must still be
    ``[B, A]``. A toggle-shaped route answers the retry by UNPINNING A instead,
    which is the user-visible "the pin keeps un-pinning itself".
    """
    client, root = pins_api
    for session_id in ("aaaaaaaaaaa1", "aaaaaaaaaaa2"):
        _session(root, session_id)

    for session_id in ("aaaaaaaaaaa1", "aaaaaaaaaaa2", "aaaaaaaaaaa1"):
        response = await client.post(
            f"/v1/desktop/sessions/{session_id}/pin", json={"pinned": True}
        )
        assert response.status_code == 200, response.text

    assert read_pins(root) == ["aaaaaaaaaaa2", "aaaaaaaaaaa1"]


@pytest.mark.asyncio
async def test_a_repeated_request_writes_nothing(pins_api) -> None:
    """Mtime-identical, not merely "the same bytes": a same-content rewrite is
    still a write, and it is what would wake the feed's catalogue probe — and
    every subscribed desktop sidebar with it — for a request that changed
    nothing."""
    client, root = pins_api
    session_id = "aaaaaaaaaaa1"
    _session(root, session_id)
    await client.post(f"/v1/desktop/sessions/{session_id}/pin", json={"pinned": True})
    path = root / PINS_FILE
    before = path.stat().st_mtime_ns

    await client.post(f"/v1/desktop/sessions/{session_id}/pin", json={"pinned": True})

    assert path.stat().st_mtime_ns == before


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", ["deadbeefcafe", "zz", "../../etc", "not-hex-0000"])
async def test_an_unknown_or_malformed_id_is_a_404_and_touches_nothing(
    pins_api, session_id: str
) -> None:
    """One generic 404 for both, which is what ``errors()`` already funnels: the
    reader cannot act on the difference between "no such session" and "bad id".

    The traversal shape is the one worth naming — ``self.root / "sessions" /
    session_id`` joins the id straight onto a path, so an id that is not a bare
    directory name is refused BEFORE any filesystem question is asked.
    """
    client, root = pins_api
    _session(root, "aaaaaaaaaaa1")
    toggle_pin(root, "aaaaaaaaaaa1")

    response = await client.post(f"/v1/desktop/sessions/{session_id}/pin", json={"pinned": True})

    assert response.status_code == 404, response.text
    assert read_pins(root) == ["aaaaaaaaaaa1"], "a refusal must not touch the store"
    assert not (root / "etc").exists()


@pytest.mark.asyncio
async def test_a_delegated_run_is_pinnable_and_unpinnable(pins_api) -> None:
    """THE ``is_user_session`` TRAP.

    The route's closest neighbour (``/seen``) requires ``is_user_session``; a pin
    route that copied it would answer 404 for a session the TUI happily pins, and
    would then refuse to UNPIN it — leaving the user a pin they can see in the
    desktop list and cannot remove. The premise is asserted, not assumed: the
    fixture must be a session ``is_user_session`` rejects, and it must be listed
    the way the TUI's own catalog lists a pinned hidden run.
    """
    client, root = pins_api
    run_id = "bbbbbbbbbbb1"
    path = _delegated_run(root, run_id)
    assert not is_user_session(path), "the fixture is not the state under test"

    pinned = await client.post(f"/v1/desktop/sessions/{run_id}/pin", json={"pinned": True})
    assert pinned.status_code == 200, pinned.text
    assert pinned.json()["result"]["pinned"] is True
    assert read_pins(root) == [run_id]

    unpinned = await client.post(f"/v1/desktop/sessions/{run_id}/pin", json={"pinned": False})
    assert unpinned.status_code == 200, unpinned.text
    assert unpinned.json()["result"]["pinned"] is False
    assert read_pins(root) == []


@pytest.mark.asyncio
async def test_the_row_carries_pinned_on_every_row_both_values(pins_api) -> None:
    """The omission the client's merge cannot recover from.

    Three rows, one pinned: every row carries the key, and the two unpinned rows
    carry ``False`` rather than nothing. Asserted per row over the RAW JSON, not
    through the model, because the model is what would hide the omission — a
    required field makes the whole response fail rather than drop a key, and this
    test is the one that says the wire shape is what the renderer needs.
    """
    client, root = pins_api
    for session_id in ("aaaaaaaaaaa1", "aaaaaaaaaaa2", "aaaaaaaaaaa3"):
        _session(root, session_id)
    pinned_id = "aaaaaaaaaaa2"
    toggle_pin(root, pinned_id)

    response = await client.get("/v1/desktop/sessions")

    assert response.status_code == 200, response.text
    rows = _rows(response.json())
    assert len(rows) == 3
    assert all("pinned" in row for row in rows), rows
    assert {row["id"] for row in rows if row["pinned"]} == {pinned_id}
    assert sum(1 for row in rows if row["pinned"] is False) == 2


@pytest.mark.asyncio
async def test_an_empty_store_answers_with_an_empty_envelope(pins_api) -> None:
    """The other half of the row assertion: an empty store carries no row to
    prove anything, so the ENVELOPE is what is checked — a 200 with the page
    fields, rather than the "no pins" answer being mistaken for a store error."""
    client, _root = pins_api

    response = await client.get("/v1/desktop/sessions")

    assert response.status_code == 200, response.text
    payload = response.json()["result"]
    assert payload["sessions"] == []
    assert payload["degraded"] == []


@pytest.mark.asyncio
async def test_a_pinned_id_whose_directory_is_gone_produces_no_row(pins_api) -> None:
    """Cleanup removed the session under a live pin.

    The store prunes such an id at READ, so the route never sees it and the
    listing must simply not contain the row — no 500, and no row resurrected from
    the pin. This is the same prune the TUI depends on, seen from the desktop
    side, and it is why neither the route nor the projection re-implements it.
    """
    client, root = pins_api
    live_id, gone_id = "aaaaaaaaaaa1", "aaaaaaaaaaa2"
    _session(root, live_id)
    gone = _session(root, gone_id)
    toggle_pin(root, gone_id)
    toggle_pin(root, live_id)
    assert set(read_pins(root)) == {live_id, gone_id}

    _remove(gone)

    response = await client.get("/v1/desktop/sessions")

    assert response.status_code == 200, response.text
    rows = _rows(response.json())
    assert {row["id"] for row in rows} == {live_id}
    assert read_pins(root) == [live_id], "the read-time prune drops the dead id"


def _remove(path: Path) -> None:
    """Remove a seeded session directory and its contents."""
    for child in path.iterdir():
        child.unlink()
    path.rmdir()


@pytest.mark.asyncio
async def test_a_pin_written_by_the_other_surface_is_what_the_route_serves(pins_api) -> None:
    """The shared-file claim, desktop-read half.

    The pin is written by the STORE (which is what the TUI's f10 calls) and read
    back through the route, with no pin state on either side of it: one file, two
    readers. The TUI→desktop direction is proved end to end in the pilot; this is
    the same claim at the seam, where a second pin table on the desktop would
    show up immediately.
    """
    client, root = pins_api
    _session(root, "aaaaaaaaaaa1")
    _session(root, "aaaaaaaaaaa2")

    assert toggle_pin(root, "aaaaaaaaaaa1") is True

    rows = _rows((await client.get("/v1/desktop/sessions")).json())
    assert {row["id"] for row in rows if row["pinned"]} == {"aaaaaaaaaaa1"}


@pytest.mark.asyncio
async def test_the_body_is_closed(pins_api) -> None:
    """The field is the whole request: an omitted ``pinned`` is a 422 rather
    than a silent false (which would unpin something), and an invented key is a
    named 422 rather than an ignored field."""
    client, root = pins_api
    session_id = "aaaaaaaaaaa1"
    _session(root, session_id)

    missing = await client.post(f"/v1/desktop/sessions/{session_id}/pin", json={})
    extra = await client.post(
        f"/v1/desktop/sessions/{session_id}/pin", json={"pinned": True, "toggle": True}
    )

    assert missing.status_code == 422, missing.text
    assert extra.status_code == 422, extra.text
    assert read_pins(root) == []


@pytest.mark.asyncio
async def test_the_route_is_advertised_by_session_pins_only(pins_api) -> None:
    """Its own key, and no neighbour moves.

    A renderer that does not see ``session_pins`` mounts NO affordance, so the
    key has to be there for the surface to exist at all; and bumping
    ``session_catalogue`` to advertise it would hide a working chats list behind
    an update it does not need — that list renders perfectly without pins.
    """
    client, _root = pins_api

    response = await client.get("/v1/capabilities")
    features = response.json()["result"]["features"]

    assert response.status_code == 200, response.text
    assert features["session_pins"] == 1
    # Unmoved neighbours, including the one this key could most plausibly have
    # been folded into.
    assert features["session_catalogue"] == 3
    assert features["desktop_feed"] == 1


@pytest.mark.asyncio
async def test_the_route_is_reachable_without_the_feed_or_a_running_session(pins_api) -> None:
    """Cold, like its neighbour ``/notified``: no bridge is acquired and no
    runtime is started, so pinning a finished conversation works on a daemon
    where nothing is running. Asserted over the adapter's own bridge table rather
    than inferred, because "no runtime started" is exactly the claim a future
    refactor would break silently."""
    client, root = pins_api
    session_id = "aaaaaaaaaaa1"
    _session(root, session_id)
    pool: DesktopSessions = client._transport.app.state.desktop_sessions

    response = await client.post(f"/v1/desktop/sessions/{session_id}/pin", json={"pinned": True})

    assert response.status_code == 200, response.text
    assert pool.bridges == {}, "a pin must not acquire a bridge"

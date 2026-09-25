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


async def _speak(root: Path, session_id: str, text: str = "retention sweep notes") -> None:
    """Give a seeded session a REAL transcript, written through the real writer.

    Required by the SEARCH tests and not by the list ones: ``load_catalog``
    lists a directory carrying only the session markers, while the search scans
    ``recent_session_rows``, which needs something actually said. A fixture that
    skimped on this would assert that a pinned conversation is findable by
    asking the search about a session it cannot see.
    """
    from local_operator.harness.types import Message, TextContent
    from local_operator.session.transcript import Transcript

    await Transcript(root / "sessions" / session_id).append_message(
        Message(role="user", content=[TextContent(text=text)])
    )


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
    await _speak(root, live_id)
    await _speak(root, gone_id)
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
@pytest.mark.parametrize("body", ['{"pinned": "yes"}', '{"pinned": 1}', '{"pinned": "true"}'])
async def test_only_a_real_boolean_is_accepted(pins_api, body: str) -> None:
    """``StrictBool``, matching every other boolean on this plane.

    A bare ``bool`` coerces these: ``"yes"``, ``1`` and ``"true"`` would all
    answer 200 and pin the session, so a client whose serialiser is emitting the
    wrong type gets no signal at all and the fault surfaces later as a pin that
    came from nowhere. Every neighbour on this plane (``PresenceWindow``,
    ``PresenceBeat``, ``Watch``, ``Answer``) refuses that shape, and a pin is
    durable state rather than a display hint.
    """
    client, root = pins_api
    session_id = "aaaaaaaaaaa1"
    _session(root, session_id)

    response = await client.post(
        f"/v1/desktop/sessions/{session_id}/pin",
        content=body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422, response.text
    assert read_pins(root) == [], "a coerced truthy value must not pin anything"


@pytest.mark.asyncio
async def test_a_pinned_session_beyond_the_clients_page_carries_pinned_in_search(pins_api) -> None:
    """THE SEARCH ROW'S ``pinned``, and why it is not optional.

    A client that synthesises a row from a search hit — which the app does, for a
    conversation beyond the 500 rows its own page holds — renders the row in an
    ordinary section when the answer omits the flag, with no Pinned section and a
    pin control whose press is an idempotent no-op that cannot repair it (the pin
    is already true server-side). This is that conversation: pinned, off the
    client's page, and answered by the search.
    """
    client, root = pins_api
    ids = [f"aaaaaaaaaa{index:02x}" for index in range(6)]
    for session_id in ids:
        _session(root, session_id)
        await _speak(root, session_id)

    full = _rows((await client.get("/v1/desktop/sessions")).json())
    pinned_id = full[-1]["id"]  # the last row, so a one-row page cannot hold it
    assert toggle_pin(root, pinned_id) is True

    # The PAGE, not the whole answer: the answer also carries the pinned row as
    # an extra now, which is the fix, so the premise has to name the page.
    page = _rows((await client.get("/v1/desktop/sessions", params={"limit": 1})).json())
    assert pinned_id not in {
        row["id"] for row in page[:1]
    }, "the fixture is not the state under test"

    answer = await client.get("/v1/desktop/sessions/search", params={"q": pinned_id})

    assert answer.status_code == 200, answer.text
    rows = answer.json()["result"]["sessions"]
    assert [row["id"] for row in rows] == [pinned_id]
    assert rows[0]["pinned"] is True


@pytest.mark.asyncio
async def test_every_search_row_carries_pinned_both_values(pins_api) -> None:
    """The other half: the ``false`` rows carry the key too.

    Answered over the raw JSON for the reason the list's version is: the model
    would make the whole response fail rather than drop a key, and what the
    client needs is the key being PRESENT — an absent one is read as "no claim",
    which is what leaves a stale optimistic pin in place.
    """
    client, root = pins_api
    ids = [f"aaaaaaaaaa{index:02x}" for index in range(3)]
    for session_id in ids:
        _session(root, session_id)
        await _speak(root, session_id)
    toggle_pin(root, ids[1])

    answer = await client.get("/v1/desktop/sessions/search", params={"q": "aaaaaaaaaa"})

    assert answer.status_code == 200, answer.text
    rows = answer.json()["result"]["sessions"]
    assert {row["id"] for row in rows} == set(ids)
    assert all("pinned" in row for row in rows), rows
    assert {row["id"] for row in rows if row["pinned"]} == {ids[1]}
    assert sum(1 for row in rows if row["pinned"] is False) == 2


@pytest.mark.asyncio
async def test_a_pinned_id_whose_directory_is_gone_is_not_searchable(pins_api) -> None:
    """The store's read-time prune, seen from the search: no row, no 500.

    The pin file still names the id and the search INDEX may still describe it,
    so this is the assertion that the answer is driven by what the store can
    still resolve rather than by what the pin file remembers."""
    client, root = pins_api
    gone_id, live_id = "aaaaaaaaaa01", "aaaaaaaaaa02"
    _session(root, live_id)
    gone = _session(root, gone_id)
    toggle_pin(root, gone_id)
    toggle_pin(root, live_id)

    _remove(gone)

    answer = await client.get("/v1/desktop/sessions/search", params={"q": gone_id})

    assert answer.status_code == 200, answer.text
    assert answer.json()["result"]["sessions"] == []
    assert read_pins(root) == [live_id]


@pytest.mark.skipif(os.getuid() == 0, reason="root writes to unwritable directories anyway")
@pytest.mark.asyncio
async def test_a_read_only_root_still_answers_200_and_writes_nothing(pins_api) -> None:
    """The ONE case where the response is not a durability claim, pinned.

    ``_write_pins`` swallows its ``OSError`` by the never-raise contract the
    store inherits from ``toggle_pin``, so on a config root this process cannot
    write the route answers success over a file that did not change. The client
    reconciles its row on that answer, so the pin stays on screen until the next
    catalogue read contradicts it.

    ASSERTED RATHER THAN LEFT AS PROSE, because both of the alternatives were
    considered and rejected on the record: escaping the failure breaks the
    contract ``toggle_pin``'s own tests pin, and a read-back would reintroduce
    the race between two writers that the store documents as accepted. This is
    what the code does, so a future change to it has to fail here and argue with
    the reasoning rather than discover the behaviour in production.
    """
    client, root = pins_api
    session_id = "aaaaaaaaaaa1"
    _session(root, session_id)
    root.chmod(0o500)
    try:
        response = await client.post(
            f"/v1/desktop/sessions/{session_id}/pin", json={"pinned": True}
        )
        assert response.status_code == 200, response.text
        assert response.json()["result"]["pinned"] is True
        assert not (root / PINS_FILE).exists(), "nothing was written, and nothing may be claimed"
        # And the list does NOT agree, which is the half the client relies on to
        # settle the row it optimistically pinned.
        rows = _rows((await client.get("/v1/desktop/sessions")).json())
        assert {row["id"]: row["pinned"] for row in rows} == {session_id: False}
    finally:
        root.chmod(0o700)


@pytest.mark.asyncio
async def test_both_projections_read_the_pin_file_once_per_request(pins_api, monkeypatch) -> None:
    """ONE ``read_pins`` per request, not one per row.

    Asserted by COUNTING, because the property is a cost the answer cannot
    betray: a per-row read returns exactly the same JSON for a fixture this size,
    and it is the store walk it is attached to that made the field worth sending
    at all. Both routes are checked, since both grew the same read.
    """
    from local_operator.server.utils import desktop_sessions as module

    client, root = pins_api
    ids = [f"aaaaaaaaaa{index:02x}" for index in range(5)]
    for session_id in ids:
        _session(root, session_id)
        await _speak(root, session_id)
    toggle_pin(root, ids[0])

    calls: list[Path] = []
    original = module.read_pins

    def counted(config_dir):
        calls.append(config_dir)
        return original(config_dir)

    monkeypatch.setattr(module, "read_pins", counted)

    assert (await client.get("/v1/desktop/sessions")).status_code == 200
    assert len(calls) == 1, calls

    calls.clear()
    listed = await client.get("/v1/desktop/sessions/search", params={"q": "aaaaaaaaaa"})
    assert listed.status_code == 200, listed.text
    assert len(listed.json()["result"]["sessions"]) == 5, "the fixture must answer several rows"
    assert len(calls) == 1, calls


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


# -- the pinned rows the PAGE does not carry -------------------------------------
#
# THE OPERATOR'S CASE, and it is the ordinary one rather than an edge: the store
# holds 5,267 sessions while the client asks for a 500-row page, so a pin made in
# the TUI on anything older than the newest 500 — most of the store — had no row
# in the app at all. UX round 5 measured the app drawing no row, no count and no
# trace for 11.2 s and beyond while the store held the pin.
#
# These tests drive the route, not the catalogue: what has to be true is that the
# ANSWER carries the row, because the answer is what the client replaces its rows
# with. `test_catalog_scan_cost.py` pins the catalogue half.


async def _json(response: Any) -> dict[str, Any]:
    """Await one response and assert it answered 200."""
    answered = await response
    assert answered.status_code == 200, answered.text
    return answered.json()


async def _all_ids(client) -> list[str]:
    """Every id the store's own ranking yields, extras included."""
    return [row["id"] for row in _rows(await _json(client.get("/v1/desktop/sessions")))]


async def _page_ids(client, limit: int) -> list[str]:
    """The ids the PAGE carries, with any extras excluded.

    Taken as the first `limit` rows of the answer, which is what the route's
    ordering guarantees: the page comes first and the extras are appended after
    it. Every test below relies on that rather than re-deriving it, which is why
    it is said here once.
    """
    rows = _rows(await _json(client.get("/v1/desktop/sessions", params={"limit": limit})))
    return [row["id"] for row in rows[:limit]]


@pytest.mark.asyncio
async def test_a_scoped_answer_carries_no_off_page_pin(pins_api) -> None:
    """A scoped page speaks for its GROUP; only the head speaks for the pins.

    ``pinned_off_page`` exists so a client holding one page can still draw a pin
    made on an older conversation, and that argument is about the listing as a
    whole -- which is the one thing a scoped request does not ask about. The
    client gates its pin facts on the head answer for the same reason
    (``pinFacts`` settled by a scope answer would erase a pin made outside that
    scope), so an extra here would be a row belonging to another team arriving
    under this team's name.
    """
    from local_operator.resume import write_session_attachment

    client, root = pins_api
    for index in range(4):
        session_id = f"lop{index:07d}"
        _session(root, session_id)
        write_session_attachment(root / "sessions" / session_id, team="lopdev", agent="", goal="")
    # Ranked LAST, so a page bound of 2 cannot carry it: the fixture stamps every
    # session with the same `created_at`, so the id break-tie decides, and
    # ``oldpin000001`` sorts after every ``lop...`` id.
    older = "oldpin000001"
    _session(root, older)
    write_session_attachment(root / "sessions" / older, team="lopdev", agent="", goal="")
    assert toggle_pin(root, older) is True

    head = (await _json(client.get("/v1/desktop/sessions", params={"limit": 2})))["result"]
    assert [row["id"] for row in head["sessions"]] == [
        "lop0000000",
        "lop0000001",
        older,
    ], "the head page still appends a pinned row it does not carry"

    scoped = (
        await _json(
            client.get(
                "/v1/desktop/sessions",
                params={"limit": 2, "scope_kind": "team", "scope_name": "lopdev"},
            )
        )
    )["result"]

    assert [row["id"] for row in scoped["sessions"]] == ["lop0000000", "lop0000001"]
    assert all(row["binding"]["team"] == "lopdev" for row in scoped["sessions"])
    assert scoped["scope"] == {"kind": "team", "name": "lopdev"}
    # ``truncated`` still means "this SCOPE held more than the page", and the
    # cursor is the position that more can be read from -- the two are one fact
    # (see ``SessionList.next_cursor``), so they agree on every answer.
    assert scoped["truncated"] is True
    assert scoped["next_cursor"] is not None


@pytest.mark.asyncio
async def test_the_extras_ride_one_page_of_a_walk(pins_api) -> None:
    """A pinned row is an EXTRA once, on the first page -- not on every page.

    ``pinned_off_page`` is a promise about the page the client paints first: the
    whole pinned set rides the answer the Pinned section is drawn from. Built from
    ``ranked[limit:]``, that list was appended on EVERY page of a walk that still
    had the pin below it -- so one pinned row came back as a surplus row page after
    page (QA measured the same id twice over a seven-page walk), and the further
    down the listing the pin ranked, the more duplicates a walk accumulated. The
    design sanctions the row's OWN later position (the row union is id-keyed and a
    pinned row "may additionally be in the head"); what it does not sanction is a
    second EXTRA.
    """
    client, root = pins_api
    for index in range(6):
        _session(root, f"lop{index:07d}")
    # Ranked LAST: the fixture stamps one ``created_at`` on every session, so the
    # id breaks the tie and ``oldpin...`` sorts after every ``lop...`` id -- which
    # is what puts it below a page bound of 2 for the whole walk.
    older = "oldpin000001"
    _session(root, older)
    assert toggle_pin(root, older) is True

    cursor: str | None = None
    surpluses: list[list[str]] = []
    seen: list[str] = []
    while True:
        params: dict[str, Any] = {"limit": 2}
        if cursor is not None:
            params["cursor"] = cursor
        result = (await _json(client.get("/v1/desktop/sessions", params=params)))["result"]
        rows = [row["id"] for row in result["sessions"]]
        # The client splits on ``limit``: the page, then whatever was appended.
        surpluses.append(rows[2:])
        seen += rows
        assert result["truncated"] == (result["next_cursor"] is not None)
        cursor = result["next_cursor"]
        if cursor is None:
            break

    assert len(surpluses) == 4, seen
    assert surpluses == [[older], [], [], []], surpluses
    # Once as the extra, once at its own rank -- and no third time.
    assert seen.count(older) == 2, seen


@pytest.mark.asyncio
async def test_a_foreign_cursor_on_the_head_still_carries_the_extras(pins_api) -> None:
    """A token that DECODES but is not usable here is still this scope's first page.

    ``cursor_missing`` is the disjunction the extras gate reads, and it has two
    halves that have to be treated alike: an unreadable token, and a decodable
    FOREIGN one (a group's cursor sent on the head request -- the client bug the
    flag exists to absorb). Gating on ``position is None`` covered only the first:
    a foreign token decodes, so the resume filter was skipped and the answer WAS
    the head's first page, but the pinned extras were omitted from it. That is the
    one shape where the Pinned section could lose the rows this list exists to
    keep, since the same answer is what settles ``pinFacts``.
    """
    from local_operator.resume import write_session_attachment

    client, root = pins_api
    for index in range(4):
        session_id = f"lop{index:07d}"
        _session(root, session_id)
        write_session_attachment(root / "sessions" / session_id, team="lopdev", agent="", goal="")
    # Ranked LAST (one shared `created_at`, so the id breaks the tie), and pinned,
    # so a page bound of 2 cannot carry it and only the extras half can.
    older = "oldpin000001"
    _session(root, older)
    write_session_attachment(root / "sessions" / older, team="lopdev", agent="", goal="")
    assert toggle_pin(root, older) is True

    scoped = (
        await _json(
            client.get(
                "/v1/desktop/sessions",
                params={"limit": 2, "scope_kind": "team", "scope_name": "lopdev"},
            )
        )
    )["result"]
    foreign = scoped["next_cursor"]
    assert foreign is not None, "the scoped page must be truncatable for this shape"

    cursorless = (await _json(client.get("/v1/desktop/sessions", params={"limit": 2})))["result"]
    resumed = (
        await _json(client.get("/v1/desktop/sessions", params={"limit": 2, "cursor": foreign}))
    )["result"]

    # The foreign token is refused the way an unreadable one is -- and the answer
    # is the first page, so it carries the extras exactly as the cursorless one.
    assert resumed["cursor_missing"] is True
    assert [row["id"] for row in cursorless["sessions"]] == [
        "lop0000000",
        "lop0000001",
        older,
    ]
    assert resumed["sessions"] == cursorless["sessions"]
    assert resumed["next_cursor"] == cursorless["next_cursor"]


@pytest.mark.asyncio
async def test_a_pinned_conversation_beyond_the_page_is_carried_in_the_answer(pins_api) -> None:
    """THE FIX. A pin outside the page is a ROW in the answer, with its name and
    `pinned: true`, so the client's pinned section has something to draw."""
    client, root = pins_api
    ids = [f"aaaaaaaaaa{index:02x}" for index in range(5)]
    for session_id in ids:
        _session(root, session_id)
        # A real transcript, because that is what the naming path reads: a
        # marker-only directory is LISTED but never enters the scan whose
        # candidate list the hydration call resolves names from, so a fixture
        # without one cannot tell a hydrated row from a placeholder.
        await _speak(root, session_id, f"conversation {session_id}")
    full = await _all_ids(client)
    page = await _page_ids(client, 2)
    off_page = next(session_id for session_id in full if session_id not in page)
    assert toggle_pin(root, off_page) is True

    payload = (await _json(client.get("/v1/desktop/sessions", params={"limit": 2})))["result"]
    rows = payload["sessions"]

    assert [row["id"] for row in rows[:2]] == page, "the page must not change"
    assert [row["id"] for row in rows[2:]] == [off_page]
    extra = rows[2]
    assert extra["pinned"] is True
    assert (
        extra["name"] == f"conversation {off_page}"
    ), "the extra was not hydrated through the naming path"
    # The extras carry the fields a page row carries, because the client renders
    # both in one list.
    assert set(extra) >= {"name", "mtime", "preview", "active", "status", "binding", "degraded"}
    # And `truncated`/`limit` still describe the PAGE.
    assert payload["limit"] == 2
    assert payload["truncated"] is True


@pytest.mark.asyncio
async def test_the_same_conversation_is_absent_when_it_is_not_pinned(pins_api) -> None:
    """The control, so the assertion above is about the pin rather than about the
    ranking happening to include the row."""
    client, root = pins_api
    for index in range(5):
        _session(root, f"aaaaaaaaaa{index:02x}")

    page = await _page_ids(client, 2)
    rows = _rows(await _json(client.get("/v1/desktop/sessions", params={"limit": 2})))

    assert [row["id"] for row in rows] == page
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_the_extras_come_after_the_page_in_the_store_s_own_order(pins_api) -> None:
    """No pin-recency order and no re-sort: the extras are appended in the
    ranking's order, which is what makes the app's Pinned section and the TUI's
    present one list in one order."""
    client, root = pins_api
    for index in range(6):
        _session(root, f"aaaaaaaaaa{index:02x}")
    full = await _all_ids(client)
    page = await _page_ids(client, 2)
    off_page = [session_id for session_id in full if session_id not in page]
    assert len(off_page) >= 2
    # PINNED IN THE ORDER THAT MAKES THE TWO ORDERINGS DISAGREE. `read_pins` is
    # newest-first, so setting `off_page[-2]` first and `off_page[-1]` second
    # leaves the store holding [off_page[-1], off_page[-2]] while the ranking
    # holds [off_page[-2], off_page[-1]]. An earlier revision pinned them the
    # other way round, which made pin order and rank order identical and left
    # this test passing for either implementation — asserted below rather than
    # left to a comment, so a future fixture change cannot quietly make it
    # vacuous again.
    toggle_pin(root, off_page[-2])
    toggle_pin(root, off_page[-1])
    assert read_pins(root) == [off_page[-1], off_page[-2]], "the fixture must discriminate"

    rows = _rows(await _json(client.get("/v1/desktop/sessions", params={"limit": 2})))

    assert [row["id"] for row in rows[2:]] == [off_page[-2], off_page[-1]]
    assert [row["id"] for row in rows[2:]] != read_pins(root), "a pin-recency sort must be visible"


@pytest.mark.asyncio
async def test_a_pinned_conversation_inside_the_page_is_not_duplicated(pins_api) -> None:
    """A pin the page already carries stays in the page, once."""
    client, root = pins_api
    for index in range(4):
        _session(root, f"aaaaaaaaaa{index:02x}")
    page = await _page_ids(client, 2)
    toggle_pin(root, page[0])

    rows = _rows(await _json(client.get("/v1/desktop/sessions", params={"limit": 2})))
    ids = [row["id"] for row in rows]

    assert ids == page
    assert len(ids) == len(set(ids))
    assert next(row for row in rows if row["id"] == page[0])["pinned"] is True


@pytest.mark.asyncio
async def test_a_deleted_pinned_conversation_is_absent_rather_than_a_500(pins_api) -> None:
    """The store prunes at read, so the extra simply does not resolve."""
    client, root = pins_api
    ids = [f"aaaaaaaaaa{index:02x}" for index in range(5)]
    for session_id in ids:
        _session(root, session_id)
    full = await _all_ids(client)
    page = await _page_ids(client, 2)
    gone = next(session_id for session_id in full if session_id not in page)
    toggle_pin(root, gone)

    _remove(root / "sessions" / gone)

    rows = _rows(await _json(client.get("/v1/desktop/sessions", params={"limit": 2})))
    assert gone not in {row["id"] for row in rows}
    assert read_pins(root) == [], "the read-time prune drops the dead id"


@pytest.mark.asyncio
async def test_the_pin_store_s_cap_bounds_the_extras(pins_api) -> None:
    """The 50-cap, seen where it matters: the answer never claims more pins than
    the store holds, and the extras cannot grow past it however large the store
    is. Seeded past the cap so the trim is exercised rather than assumed."""
    from local_operator.tui.sidebar_pins import PINS_LIMIT

    client, root = pins_api
    for index in range(PINS_LIMIT + 5):
        _session(root, f"{index:012x}")
    for index in range(PINS_LIMIT + 5):
        toggle_pin(root, f"{index:012x}")
    assert len(read_pins(root)) == PINS_LIMIT, "the fixture must exceed the cap"

    rows = _rows(await _json(client.get("/v1/desktop/sessions", params={"limit": 2})))
    ids = [row["id"] for row in rows]
    pinned = [row for row in rows if row["pinned"]]

    assert len(pinned) <= PINS_LIMIT
    assert len(pinned) == PINS_LIMIT, "every kept pin is either on the page or an extra"
    assert len(ids) == len(set(ids)), "a dropped pin must not bring a duplicate row"


@pytest.mark.asyncio
async def test_truncated_still_speaks_about_the_page(pins_api) -> None:
    """A full page plus extras is not a promise that nothing was withheld: with
    a store bigger than the page, `truncated` says so, and it says nothing about
    the extras."""
    client, root = pins_api
    ids = [f"aaaaaaaaaa{index:02x}" for index in range(6)]
    for session_id in ids:
        _session(root, session_id)

    small = (await _json(client.get("/v1/desktop/sessions", params={"limit": 2})))["result"]
    large = (await _json(client.get("/v1/desktop/sessions", params={"limit": 50})))["result"]

    assert small["truncated"] is True and small["limit"] == 2
    assert large["truncated"] is False and large["limit"] == 50
    assert len(large["sessions"]) == 6, "a page that fits the store carries no extras"

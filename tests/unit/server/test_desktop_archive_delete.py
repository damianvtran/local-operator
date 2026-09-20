"""The archive and delete surface: rows, routes, capability keys, native payloads.

Four things here are load-bearing, and each is a bug this feature was pictured
shipping:

* **``archived`` on EVERY row and EVERY hit, both values.** The renderer's merge
  is ``{...current, ...incoming}`` under the rule "an absent key is not a
  claim", so a listing that omitted it on an unarchived row would leave a stale
  ``true`` in place forever. The ``false`` rows are the ones that carry the
  assertion, and they are asserted on BOTH answers — the default list and the
  one that asked for archived rows.
* **A separate capability key per verb, never a ``session_catalogue`` bump.**
  Absent means no affordance, no slot, no handler: a renderer that does not see
  ``session_delete`` must not offer a control the backend will 404.
* **Delete is refused with a SENTENCE, not a status.** A live session, an armed
  wake and unread spooled mail have three different remedies, and the client
  renders the sentence the machine composed rather than inventing one from a
  code.
* **The native payloads are CONTRACTS.** The desktop app builds its confirm
  control and its request from them, so their exact shape is pinned here — a
  changed key is a changed contract, and the failure it would cause is a button
  that silently does nothing.

Everything runs against a real store on a real temporary filesystem. The only
stub is the app object's plumbing; the routes go through the real ``errors()``
ladder, the real response models and the real ``DesktopSessions`` adapter.
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
from local_operator.resume import ORIGIN_SUBAGENT, mark_session_origin
from local_operator.server.routes import capabilities, desktop_sessions
from local_operator.server.utils.desktop_commands import native_action
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session.archived import read_archived
from local_operator.session.cleanup import mark_store
from local_operator.session.retention import LIVE_MARKER_NAME
from local_operator.slash_commands import SLASH_COMMANDS

MINE = "a" * 12
OTHER = "b" * 12
CHILD = "c" * 12


def _session(root: Path, session_id: str) -> Path:
    path = root / "sessions" / session_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "created_at.json").write_text("1700000000")
    (path / "desktop.json").write_text(json.dumps({"version": 1, "cwd": str(root)}))
    return path


async def _speak(root: Path, session_id: str, text: str = "retention sweep notes") -> None:
    """Give a seeded session a REAL transcript, through the real writer.

    The search scans transcripts, so a fixture that only laid down the session
    markers would assert that an archived conversation is missing from a result
    set it could never have appeared in.
    """
    from local_operator.harness.types import Message, TextContent
    from local_operator.session.transcript import Transcript

    await Transcript(root / "sessions" / session_id).append_message(
        Message(role="user", content=[TextContent(text=text)])
    )


@pytest_asyncio.fixture
async def archive_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "archive-test-token")
    mark_store(tmp_path / "sessions")
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    app.include_router(desktop_sessions.router)
    app.include_router(capabilities.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer archive-test-token"},
    ) as client:
        yield client, tmp_path.resolve()
    if hasattr(app.state, "desktop_sessions"):
        await app.state.desktop_sessions.close()


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload["result"]["sessions"]


async def _archive(client, session_id: str, archived: bool = True):
    return await client.post(
        f"/v1/desktop/sessions/{session_id}/archive", json={"archived": archived}
    )


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_verb_has_its_own_capability_key(archive_api) -> None:
    client, _root = archive_api
    response = await client.get("/v1/capabilities")
    assert response.status_code == 200
    features = response.json()["result"]["features"]
    assert features["session_archive"] == 1
    assert features["session_delete"] == 1
    # NOT a catalogue bump: an additive row field is not a shape change, and
    # bumping would hide a working catalogue from a client that predates this.
    assert features["session_catalogue"] == 3
    assert features["session_search"] == 1


# ---------------------------------------------------------------------------
# The listing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_listing_hides_archived_sessions_and_states_the_flag_either_way(
    archive_api,
) -> None:
    client, root = archive_api
    _session(root, MINE)
    _session(root, OTHER)

    listed = _rows((await client.get("/v1/desktop/sessions")).json())
    assert {row["id"]: row["archived"] for row in listed} == {MINE: False, OTHER: False}

    assert (await _archive(client, OTHER)).status_code == 200

    listed = _rows((await client.get("/v1/desktop/sessions")).json())
    assert {row["id"] for row in listed} == {MINE}
    assert all(row["archived"] is False for row in listed), "the false rows are the assertion"

    both = _rows((await client.get("/v1/desktop/sessions?include_archived=true")).json())
    assert {row["id"]: row["archived"] for row in both} == {MINE: False, OTHER: True}


@pytest.mark.asyncio
async def test_a_pinned_archived_session_is_not_a_phantom_row(archive_api) -> None:
    """The pinned-and-archived case, at the surface where the client merges.

    The row is absent from the default listing — so a renderer cannot put it in
    a pinned section — and the pins response is untouched. Asking for archived
    rows brings it back WITH ``pinned`` and ``archived`` both true, which is the
    only way the client can render it in the section it belongs to.
    """
    client, root = archive_api
    _session(root, MINE)
    _session(root, OTHER)
    assert (
        await client.post(f"/v1/desktop/sessions/{OTHER}/pin", json={"pinned": True})
    ).status_code == 200
    assert (
        await client.post(f"/v1/desktop/sessions/{OTHER}/archive", json={"archived": True})
    ).status_code == 200

    listed = _rows((await client.get("/v1/desktop/sessions")).json())
    assert OTHER not in {row["id"] for row in listed}
    assert all(not row["pinned"] for row in listed)

    revealed = _rows((await client.get("/v1/desktop/sessions?include_archived=true")).json())
    row = next(item for item in revealed if item["id"] == OTHER)
    assert row["pinned"] is True and row["archived"] is True


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_search_hides_archived_hits_and_carries_the_flag_on_every_hit(
    archive_api,
) -> None:
    client, root = archive_api
    _session(root, MINE)
    _session(root, OTHER)
    await _speak(root, MINE, "retention sweep notes")
    await _speak(root, OTHER, "retention sweep notes")

    default = (await client.get("/v1/desktop/sessions/search?q=retention")).json()
    hits = default["result"]["sessions"]
    assert {hit["id"]: hit["archived"] for hit in hits} == {MINE: False, OTHER: False}

    assert (await _archive(client, OTHER)).status_code == 200

    default = (await client.get("/v1/desktop/sessions/search?q=retention")).json()
    hits = default["result"]["sessions"]
    assert {hit["id"] for hit in hits} == {MINE}
    assert all(hit["archived"] is False for hit in hits)

    revealed = (
        await client.get("/v1/desktop/sessions/search?q=retention&include_archived=true")
    ).json()
    assert {hit["id"]: hit["archived"] for hit in revealed["result"]["sessions"]} == {
        MINE: False,
        OTHER: True,
    }


@pytest.mark.asyncio
async def test_an_archived_session_still_resolves_by_id(archive_api) -> None:
    """A listing narrows what is OFFERED, never what exists.

    The snapshot route resolves the session from its own directory, so archiving
    a conversation cannot make it unreachable — which is the whole difference
    between an archive and a delete.
    """
    client, root = archive_api
    _session(root, MINE)
    await _speak(root, MINE, "keep me")
    assert (
        await client.post(f"/v1/desktop/sessions/{MINE}/archive", json={"archived": True})
    ).status_code == 200

    response = await client.get(f"/v1/desktop/sessions/{MINE}")
    assert response.status_code == 200
    assert response.json()["result"]["session_id"] == MINE


# ---------------------------------------------------------------------------
# The archive route
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_archive_is_a_desired_state_write_and_a_retry_is_a_no_op(archive_api) -> None:
    client, root = archive_api
    _session(root, MINE)

    first = await client.post(f"/v1/desktop/sessions/{MINE}/archive", json={"archived": True})
    assert first.status_code == 200
    assert first.json()["result"] == {"session_id": MINE, "archived": True}
    assert read_archived(root) == [MINE]

    again = await client.post(f"/v1/desktop/sessions/{MINE}/archive", json={"archived": True})
    assert again.json()["result"] == {"session_id": MINE, "archived": True}
    assert read_archived(root) == [MINE], "a retry must not duplicate or reorder"

    assert (
        await client.post(f"/v1/desktop/sessions/{MINE}/archive", json={"archived": False})
    ).status_code == 200
    assert read_archived(root) == []


@pytest.mark.asyncio
async def test_archive_admits_a_delegated_run_because_it_is_reversible(archive_api) -> None:
    """The asymmetry with delete, asserted on both routes.

    Archive is id-shape-plus-is-dir like the pin route: the state is reversible,
    and the sidebar shows delegated runs, so a route that refused one would
    leave a state the user can see and cannot change. Delete is stricter for the
    opposite reason — see the route's own test below.
    """
    client, root = archive_api
    child = _session(root, CHILD)
    mark_session_origin(child, ORIGIN_SUBAGENT)

    response = await client.post(f"/v1/desktop/sessions/{CHILD}/archive", json={"archived": True})
    assert response.status_code == 200
    assert response.json()["result"]["archived"] is True


@pytest.mark.asyncio
async def test_archive_answers_404_for_an_unknown_or_malformed_id(archive_api) -> None:
    client, root = archive_api
    _session(root, MINE)
    for session_id in ("nope", "a/b"):
        response = await client.post(
            f"/v1/desktop/sessions/{session_id}/archive", json={"archived": True}
        )
        assert response.status_code == 404, session_id


@pytest.mark.asyncio
async def test_archive_requires_the_state_and_refuses_a_loose_type(archive_api) -> None:
    """A missing or non-boolean state is a 422, not a silent ``false``."""
    client, root = archive_api
    _session(root, MINE)
    assert (await client.post(f"/v1/desktop/sessions/{MINE}/archive", json={})).status_code == 422
    assert (
        await client.post(f"/v1/desktop/sessions/{MINE}/archive", json={"archived": "yes"})
    ).status_code == 422


# ---------------------------------------------------------------------------
# The delete route
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_removes_the_session_and_answers_with_the_receipt(archive_api) -> None:
    client, root = archive_api
    _session(root, MINE)
    _session(root, OTHER)

    response = await client.request(
        "DELETE", f"/v1/desktop/sessions/{MINE}", json={"confirmed": True}
    )
    assert response.status_code == 200
    assert response.json()["result"] == {"session_id": MINE, "deleted": True}
    assert not (root / "sessions" / MINE).exists()
    assert (root / "sessions" / OTHER).is_dir()


@pytest.mark.asyncio
async def test_delete_is_refused_for_a_live_session_with_the_guard_sentence(archive_api) -> None:
    client, root = archive_api
    session = _session(root, MINE)
    (session / LIVE_MARKER_NAME).write_text(str(os.getpid()), encoding="utf-8")

    response = await client.request(
        "DELETE", f"/v1/desktop/sessions/{MINE}", json={"confirmed": True}
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "session_delete_refused"
    assert "running session" in detail["message"]
    assert "Stop it" in detail["message"], "the remedy is what a client renders"
    assert session.is_dir()


@pytest.mark.asyncio
async def test_delete_answers_404_for_an_unknown_malformed_or_delegated_id(archive_api) -> None:
    """One generic refusal for all three, so an authenticated renderer cannot enumerate."""
    client, root = archive_api
    _session(root, MINE)
    child = _session(root, CHILD)
    mark_session_origin(child, ORIGIN_SUBAGENT)

    for session_id in ("nope", "a/b", CHILD):
        response = await client.request(
            "DELETE", f"/v1/desktop/sessions/{session_id}", json={"confirmed": True}
        )
        assert response.status_code == 404, session_id
    assert child.is_dir(), "a delegated run is not a conversation anyone opened"


@pytest.mark.asyncio
async def test_delete_refuses_a_body_that_does_not_confirm(archive_api) -> None:
    """The confirmation is the request. Without it there is no delete.

    ``{}`` and ``{"confirmed": false}`` are both 422 — the first because the
    field is required, the second because a body that explicitly withholds
    consent is a client bug, not a decision the server should re-interpret.
    """
    client, root = archive_api
    _session(root, MINE)
    for body in ({}, {"confirmed": False}, {"confirmed": "yes"}):
        response = await client.request("DELETE", f"/v1/desktop/sessions/{MINE}", json=body)
        assert response.status_code == 422, body
    assert (root / "sessions" / MINE).is_dir()


# ---------------------------------------------------------------------------
# The native payloads — the UI implementer's contract
# ---------------------------------------------------------------------------


def _spec(name: str):
    return next(entry for entry in SLASH_COMMANDS if entry.name == name)


def test_the_native_payloads_are_exactly_this() -> None:
    """Pinned verbatim: the desktop builds a confirm control and a request from them.

    The failure a drift would cause is a control that silently does nothing —
    the worst shape for an irreversible act, because the user believes it
    happened.
    """
    endpoint = "/v1/desktop/sessions/sess123"
    archive = native_action(_spec("archive"), "sess123", "")
    assert archive == {
        "kind": "native_action",
        "destination": "sessions.archive",
        "session_id": "sess123",
        "fields": [],
        "args": "",
        "data": {
            "submit": {"method": "POST", "path": endpoint + "/archive"},
            "archived": True,
            "source": "/v1/desktop/sessions",
        },
    }
    unarchive = native_action(_spec("unarchive"), "sess123", "")
    assert unarchive["data"] == {
        "submit": {"method": "POST", "path": endpoint + "/archive"},
        "archived": False,
        "source": "/v1/desktop/sessions",
    }

    delete = native_action(_spec("delete"), "sess123", "yes")
    assert delete == {
        "kind": "native_action",
        "destination": "sessions.delete",
        "session_id": "sess123",
        "fields": [{"name": "confirmed", "kind": "boolean", "required": True}],
        "args": "yes",
        "data": {
            "submit": {
                "method": "DELETE",
                "path": endpoint,
                "body": {"confirmed": True},
            },
            "source": "/v1/desktop/sessions",
            "confirm_word": "yes",
        },
    }


@pytest.mark.asyncio
async def test_the_command_route_answers_the_delete_action_rather_than_running_it(
    archive_api,
) -> None:
    """The commands endpoint hands the renderer a NATIVE action; it does not delete.

    Asserted because a native command that executed on the server would make the
    confirmation decorative: the renderer's control is what stands in for the
    typed word, and it has to be the thing that decides.
    """
    client, root = archive_api
    _session(root, MINE)

    response = await client.post(
        f"/v1/desktop/sessions/{MINE}/commands",
        json={
            "request_id": "11111111-2222-3333-4444-555555555555",
            "command": "delete",
            "args": "yes",
        },
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["result"]["kind"] == "native_action"
    assert result["result"]["data"]["submit"]["method"] == "DELETE"
    assert (root / "sessions" / MINE).is_dir(), "the server must not have deleted anything"


# ---------------------------------------------------------------------------
# The LIVE row, over the wire (QA round 1, Q1's second surface)
# ---------------------------------------------------------------------------


def _publish_live(root: Path, session_id: str) -> None:
    """A running owner's discovery record, through the real writer.

    ``decorate_rows`` reads the registry's own ``scan``/``classify``, so a
    hand-written record file would agree with a wrong implementation; the live
    pid and the heartbeat ``publish`` stamps are what make it classify ``live``.
    """
    from local_operator.session.runtime.registry import publish
    from local_operator.session.runtime.types import SessionRecord

    publish(
        SessionRecord(
            pid=os.getpid(),
            kind="tui",
            session_id=session_id,
            conversation_name="synthetic owner record",
            cwd=str(root),
            model_label="test/model",
            control_port=0,
            control_key="synthetic",
        ),
        root,
    )


@pytest.mark.asyncio
async def test_a_live_archived_session_is_hidden_and_stamped_over_the_wire(archive_api) -> None:
    """The catalogue route must agree with the archive store about a LIVE session.

    THE DEFECT QA FOUND (Q1): the live row is appended by
    ``decorate_rows`` from the registry, which knows nothing about archives, so
    this endpoint answered a row for an archived conversation carrying
    ``archived: false`` — while the same request family returned
    ``archived: true`` for that id under ``include_archived``. Two answers about
    one row, and the row was the conversation a user archives from inside it.
    """
    client, root = archive_api
    directory = root / "sessions" / MINE
    directory.mkdir(parents=True)
    (directory / "created_at.json").write_text("1700000000")
    _publish_live(root, MINE)

    listed = _rows((await client.get("/v1/desktop/sessions")).json())
    assert {row["id"]: row["archived"] for row in listed} == {MINE: False}

    assert (await _archive(client, MINE)).status_code == 200

    listed = _rows((await client.get("/v1/desktop/sessions")).json())
    assert MINE not in {row["id"] for row in listed}, "no live row for an archived session"

    revealed = _rows((await client.get("/v1/desktop/sessions?include_archived=true")).json())
    row = next(item for item in revealed if item["id"] == MINE)
    assert row["archived"] is True, "the row the reveal returns must state the truth"


# ---------------------------------------------------------------------------
# A delete must stop the daemon that performed it (desktop QA round 2, PR #390)
# ---------------------------------------------------------------------------

# Every read a client probes for a session it thinks it still has.
_READS = (
    "/v1/desktop/sessions/{id}",
    "/v1/desktop/sessions/{id}/history",
    "/v1/desktop/sessions/{id}/mcp",
)
# The subset a RESIDENT bridge answers 200 for without a live runtime: the MCP
# read needs an owner to hand the status back from, so a seeded conversation
# answers 404 there for its own reason and cannot be the failing cell here. The
# 200 the QA report measured on `/mcp` came from a running app; the mechanism
# under test — residency — is the same bridge for all three.
_SERVED_READS = _READS[:2]


@pytest.mark.asyncio
async def test_a_deleted_session_is_not_served_by_the_daemon_that_deleted_it(
    archive_api,
) -> None:
    """The removal must be observable WITHOUT a restart of the process.

    Desktop QA (PR #390) reloaded the app and landed on a fresh draft over the
    removed id: `sessions.get`, `/history` and `/mcp` all answered 200 from the
    daemon that had just deleted the conversation, so the client's 404 tombstone
    never fired. A fresh daemon answers 404 for the same store, and the same
    daemon answers 404 for an id it has never seen — so the 200 was residency,
    not the store.

    The cause is by design one layer down: ``DesktopSessions.session`` hands out a
    RESIDENT bridge without re-checking the directory, because that lookup is the
    expensive half of every read (see ``docs/evidence/session-load-central-cache``)
    — and this test is what makes the delete the one event that tells it so.
    """
    client, root = archive_api
    _session(root, MINE)
    await _speak(root, MINE, "delete me")
    _session(root, OTHER)

    # The daemon serves the conversation FIRST, so what follows is about a
    # session it has already opened rather than one it never could answer for.
    for path in _SERVED_READS:
        assert (await client.get(path.format(id=MINE))).status_code == 200, path

    response = await client.request(
        "DELETE", f"/v1/desktop/sessions/{MINE}", json={"confirmed": True}
    )
    assert response.status_code == 200, response.text

    for path in _READS:
        observed = (await client.get(path.format(id=MINE))).status_code
        assert observed == 404, f"{path} still answers {observed} on the deleting daemon"

    # CONTROLS: a blanket refusal cannot pass this, and the two daemons must
    # agree about the same store — the session the delete did not address still
    # answers here, and an id neither daemon has ever seen answers 404 there.
    assert (await client.get(_READS[0].format(id=OTHER))).status_code == 200

    # A SECOND daemon over the same store, and the comparison that names the
    # defect: before the fix the deleting one answered 200 where this one
    # answered 404, for the same directory tree.
    fresh = DesktopSessions(root)
    fresh_app = FastAPI()
    fresh_app.state.config_manager = ConfigManager(root)
    fresh_app.state.desktop_sessions = fresh
    fresh_app.include_router(desktop_sessions.router)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=fresh_app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {os.environ['LOCAL_OPERATOR_DESKTOP_TOKEN']}"},
        ) as fresh_client:
            for path in _READS:
                ours = (await client.get(path.format(id=MINE))).status_code
                theirs = (await fresh_client.get(path.format(id=MINE))).status_code
                assert ours == theirs == 404, f"{path}: deleting {ours}, fresh {theirs}"
    finally:
        await fresh.close()

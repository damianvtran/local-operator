"""Opening and reading a conversation ANOTHER DEVICE holds, from the desktop.

WHAT THIS FILE PINS, and each item is a requirement of mesh slice DB2 rather
than an implementation detail:

* a peer's row OPENS: the pool resolves the id to a bridge whose owner is the
  remote device (``remote_open.open_remote_viewer`` — the same seam the TUI's
  pick and ``/resume`` use), so the snapshot, ``/history`` and every control act
  ride the mesh instead of answering 404 about the user's own conversation;
* the transcript is read OFF THE WIRE, never out of ``<root>/sessions/<id>`` —
  the local read ``mesh-session-mobility.md`` §3.4 forbids by name, which for a
  peer's id is at best absent and at worst a different conversation wearing the
  same id. Since D5-core that includes the COLD page, which is served from the
  OWNER's stored journal by the OWNER's own relay
  (``docs/design/mesh-cold-read-stored-history.md``) rather than from this disk;
* an UNSERVABLE cold page says so (``cursor_missing``) instead of answering the
  empty triple a conversation with no rows produces — the wire envelope no longer
  conflates "no rows" with "nobody could tell us";
* an UNREACHABLE peer still refuses, with the SAME sentence the TUI refuses the
  same state with (``remote_open.unreachable_peer_sentence`` — the shared
  composer exists so two surfaces cannot describe one situation two ways);
* an id that did not RESOLVE while a device stayed SILENT is refused with its
  own code (``409 session_unresolved``) rather than the shared 404 — the miss is
  not evidence of absence, and the 404 is what the renderer paints as a deleted
  conversation (mesh-wire-honesty.md §S2);
* an id nobody holds is still the shared 404, and a LOCAL id is untouched —
  including its cost: the peer projection is never consulted for a directory
  this device already holds;
* this device's own stores are not written or believed about somebody else's
  conversation: a peer's completion receipts come from the peer's sync, not from
  a local ``attention.db`` that is empty by construction.

The REAL relay pair is the network suite's job
(``tests/unit/network/test_remote_viewer.py`` runs the desktop bridge over two
real relays and a real runtime on the peer). What is exercised here is the
mapping either side of that transport, against a real store on a real
filesystem and through the real ``errors()`` ladder.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.harness.types import Message
from local_operator.resume import ORIGIN_NAME, SessionRow
from local_operator.server.routes import desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session import peer_rows as peer_rows_mod
from local_operator.session.attached import AttachedSession
from local_operator.session.attachments import ATTACHMENTS_DIRNAME, AttachmentStore
from local_operator.session.owner import SessionSeed
from local_operator.session.peer_rows import UnansweredPeer
from local_operator.session.placement import SessionPlacement
from local_operator.session.remote_open import (
    unreachable_peer_sentence,
    unresolved_peer_sentence,
)
from local_operator.session.retention import DESKTOP_MARKER_NAME
from tests.unit.session.test_peer_rows import _Catalog, _Facts

MINE = "c" * 12
OTHER = "e" * 12
PEER = "d_" + "7" * 32
NET_ONE = "n_" + "3" * 24
DESKTOP_TOKEN = "synthetic-desktop-token"


class StubRemoteOwner:
    """The owner seam, answered without a peer.

    ``AttachedSession`` asks its owner five things (``placement``, ``seed``,
    ``locate``, ``engage``, ``make_client``). This answers the two a READ needs
    and FAILS LOUDLY on the other two, because a read that engaged a runtime or
    dialled a socket is the defect this file exists to keep out: the desktop's
    read envelope must never start work on somebody else's machine.
    """

    def __init__(self) -> None:
        self.placement = SessionPlacement(mode="peer", network_id=NET_ONE, home_device=PEER)

    def seed(self) -> SessionSeed:
        return SessionSeed(name="build box chat", model_label="", cwd="", device_name="build-box")

    def locate(self) -> tuple[Any, Any]:
        """No runtime on the peer, as far as a read can tell: the cold answer."""
        return None, None

    async def engage(self, **_kwargs: Any) -> None:
        raise AssertionError("a desktop READ must not engage a runtime on the peer")

    def make_client(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a desktop READ must not dial the peer")


@pytest_asyncio.fixture
async def remote_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The desktop router, a real store, and a peer row the projection will answer."""
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", DESKTOP_TOKEN)
    from local_operator.session.cleanup import mark_store

    mark_store(tmp_path / "sessions")
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.state.desktop_sessions = DesktopSessions(tmp_path)
    app.include_router(desktop_sessions.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {DESKTOP_TOKEN}"},
    ) as client:
        yield client, tmp_path.resolve()


def _peer_row(**overrides: Any) -> SessionRow:
    base: dict[str, Any] = {
        "id": OTHER,
        "mtime": 1789400000.0,
        "name": "build box chat",
        "live_state": "idle",
        "locality": "remote",
        "owner_device": PEER,
        "owner_device_name": "build-box",
        "reachable": True,
        "unreachable_reason": "",
    }
    base.update(overrides)
    return SessionRow(**base)


def _answer_rows(monkeypatch: pytest.MonkeyPatch, row: SessionRow | None) -> list[str]:
    """Point the pool's ONE remote question at ``row``, recording every ask.

    ON ``remote_row_and_silence``, THE SEAM THE POOL ACTUALLY CALLS (agent review
    round 1, R-1): the row and the silence come back from one call, so a stub on
    ``remote_row_for`` — now the row-only projection of that function — would
    leave the pool's own call unanswered. The empty silence half is the honest
    default for these cells: they are about the row path, and a peer that did not
    answer is what ``row.reachable`` carries.
    """
    from local_operator.session import remote_open

    asked: list[str] = []

    def fake_resolution(
        session_id: str, root: Any = None
    ) -> tuple[SessionRow | None, tuple[Any, ...]]:
        asked.append(session_id)
        resolved = row if row is not None and row.id == session_id else None
        return resolved, ()

    monkeypatch.setattr(remote_open, "remote_row_and_silence", fake_resolution)
    return asked


def _remote_facade(monkeypatch: pytest.MonkeyPatch, root: Path) -> list[dict[str, Any]]:
    """Answer the viewer seam with a REAL cold facade whose owner is a stub peer."""
    from local_operator.session import remote_open

    built: list[dict[str, Any]] = []

    async def fake_open(session_id: str, **kwargs: Any) -> Any:
        built.append({"session_id": session_id, **kwargs})

        async def refuse_takeover() -> None:
            raise AssertionError("a remote viewer never takes over")

        return await AttachedSession.cold(
            session_id,
            config_dir=kwargs.get("config_dir", root),
            cwd="",
            takeover_factory=refuse_takeover,
            surface=kwargs.get("surface", "desktop"),
            owner=StubRemoteOwner(),
            seed=StubRemoteOwner().seed(),
        )

    monkeypatch.setattr(remote_open, "open_remote_viewer", fake_open)
    return built


def _seed_local(root: Path, session_id: str) -> None:
    path = root / "sessions" / session_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "created_at.json").write_text("1700000000")
    (path / "conversation.json").write_text(json.dumps({"name": "local chat"}))
    # THE MARKER, because a directory THIS device holds is one that was created here:
    # main's M1 guard refuses a directory that has no marker document, no transcript and
    # no mail spool — the shape a draft's engage leaves behind (measured as
    # ``probe_residue_locate``) — and that guard is the newer decision, so a seed
    # without a marker describes residue rather than a session. Every other desktop
    # seed on this tree writes one; this helper was the shortcut, and the fold is where
    # it showed: without it these cells read 404.
    (path / DESKTOP_MARKER_NAME).write_text(json.dumps({"version": 1, "cwd": str(root)}))


# ---------------------------------------------------------------------------
# The refusal that REMAINS, and the 404 that must not move
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_peer_listing() -> Iterator[None]:
    """``peer_rows`` caches by root at MODULE level: no test may inherit a read."""
    from local_operator.session import peer_rows

    peer_rows.clear_cache()
    yield
    peer_rows.clear_cache()


class _JumpingClock:
    """A monotonic clock that jumps PAST the listing TTL on every reading (R-1).

    The injected clock the round-1 reviewer reproduced ``calls=2`` with. The
    "no second dial" property was an argument about durations — the consult's
    freshness test compares against the moment the FIRST read started, and a
    listing that spends its documented budget sits on the edge — so a real clock
    cannot discriminate: the injected catalogue answers instantly. Jumping 100 s
    per reading makes a two-call shape re-dial deterministically.

    Patched onto the ``peer_rows`` module (its only use of ``time``), never the
    shared ``time`` module.
    """

    def __init__(self) -> None:
        self.readings = 0

    def monotonic(self) -> float:
        self.readings += 1
        return 1_000.0 + 100.0 * self.readings


def _relay(
    monkeypatch: pytest.MonkeyPatch,
    *,
    silent: Sequence[tuple[str, str]] = (),
    live: Sequence[tuple[str, str]] = (),
    rows: Sequence[Any] = (),
) -> _Catalog:
    """Put a relay record behind this root and answer its reads from a counting read.

    THE COUNT IS THE POINT (mesh-wire-honesty.md §S2's evidence plan): the new
    refusal must consult the silence carried by the SAME listing read that
    missed, so ``calls`` — one per ``peers()``, i.e. one per listing read — is
    the observable that distinguishes "rode the read that already happened"
    from "paid a second fan-out". The catalogue is the injection seam the
    producer's own tests use (``tests/unit/session/test_peer_rows``), and every
    layer above it — the TTL, ``_read_all``, the miss path's live read, the
    consult — is production code.
    """
    from local_operator.network import projection, store

    facts = [
        _Facts(device_id, name, reachable=False, reason="connect_failed:ConnectionRefusedError")
        for device_id, name in silent
    ]
    facts += [_Facts(device_id, name, reachable=True) for device_id, name in live]
    catalog = _Catalog(facts, list(rows))
    monkeypatch.setattr(store, "find_own_relay", lambda root=None: object())
    monkeypatch.setattr(projection, "RelayPeerCatalog", lambda root: catalog)
    return catalog


@pytest.mark.asyncio
async def test_an_unreachable_peer_refuses_in_the_tuis_own_words(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """409 ``session_is_remote``, carrying the ONE composer both surfaces use.

    The row is visible, so 404 would be a lie about the user's own conversation;
    a viewer built for a device that cannot answer could never bind, so the
    honest answer is the sentence that names the device, the reason in words and
    the command that diagnoses the link — the SAME words the TUI's pick refuses
    with, asserted here against the composer itself rather than against a copy.
    """
    client, _root = remote_api
    row = _peer_row(reachable=False, unreachable_reason="connect_failed:ConnectionRefusedError")
    _answer_rows(monkeypatch, row)

    response = await client.get(f"/v1/desktop/sessions/{OTHER}")
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "session_is_remote"
    assert detail["message"] == unreachable_peer_sentence(OTHER, row)
    assert "build-box" in detail["message"]
    assert "/network doctor" in detail["message"]
    assert (
        "connect_failed:ConnectionRefusedError" not in detail["message"]
    ), "the raw transport token reached the user"


@pytest.mark.asyncio
async def test_an_unknown_id_is_still_the_shared_404(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing about the pool's remote arm may change this answer."""
    client, _root = remote_api
    asked = _answer_rows(monkeypatch, None)

    response = await client.get(f"/v1/desktop/sessions/{'f' * 12}")
    assert response.status_code == 404, response.text
    assert asked == ["f" * 12], "the id was not even asked about"


@pytest.mark.asyncio
async def test_a_silent_device_makes_an_unknown_id_unresolved_not_missing(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """409 ``session_unresolved``: the read missed AND a device did not answer.

    THE ANSWER THIS REPLACES WAS A CLAIM ABOUT THE USER'S WORK. One 404 answered
    both "every device answered and none holds this id" and "a device did not
    reply", and the renderer turns a 404 into ``missing`` — "This conversation
    is no longer on this machine", composer refused. The silence is not evidence
    of absence, so it gets its own code, its own sentence, and a remedy that
    names no holder to distrust.
    """
    client, _root = remote_api
    catalog = _relay(
        monkeypatch, silent=[("d_silent", "build-box")], live=[("d_live", "radiant-m4")]
    )
    unknown = "f" * 12

    response = await client.get(f"/v1/desktop/sessions/{unknown}")

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "session_unresolved"
    assert detail["message"] == unresolved_peer_sentence(
        unknown,
        (UnansweredPeer("d_silent", "build-box", "connect_failed:ConnectionRefusedError"),),
    )
    assert "build-box" in detail["message"], "the silent device went unnamed"
    assert "did not answer" in detail["message"], "silence was not named as silence"
    assert "connect_failed:ConnectionRefusedError" not in detail["message"]
    assert " is on " not in detail["message"], "silence was turned into an ownership claim"
    assert "may be on that device" in detail["message"]
    assert "no longer" not in detail["message"], "an unprovable absence was stated as a deletion"
    assert "could not be resolved" not in detail["message"], "the miss was the headline"
    assert catalog.calls == 1, "the consult paid a SECOND listing read"


@pytest.mark.asyncio
async def test_the_plural_route_answer_says_one_of_them(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-6/Q1: the plural copy, on the wire, asserted where it is produced.

    Neutering the branch to ``subject = "that device"`` for every count left the
    whole suite green before this cell, and both branches are reachable: the relay
    reports every device it could not reach.
    """
    client, _root = remote_api
    _relay(
        monkeypatch,
        silent=[("d_1", "build-box"), ("d_2", "radiant-m4")],
        live=[("d_live", "pixel-8")],
    )

    response = await client.get(f"/v1/desktop/sessions/{'f' * 12}")

    assert response.status_code == 409, response.text
    message = response.json()["detail"]["message"]
    assert "build-box and radiant-m4 did not answer" in message
    assert "one of them" in message
    assert "that device" not in message
    assert " is on " not in message


@pytest.mark.asyncio
async def test_an_unnamed_silent_device_still_appears(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-6: ``peer_rows`` builds ``name=str(facts.name or "")``, so this is a real input."""
    client, _root = remote_api
    _relay(monkeypatch, silent=[("d_1", "build-box"), ("d_2", "")])

    response = await client.get(f"/v1/desktop/sessions/{'f' * 12}")

    assert response.status_code == 409, response.text
    message = response.json()["detail"]["message"]
    assert "build-box and unnamed device did not answer" in message
    assert "  " not in message


@pytest.mark.asyncio
async def test_a_miss_whose_own_read_outlasts_the_ttl_still_issues_no_second_dial(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-1, at the route: ONE READ, BOTH HALVES — by construction, not by duration.

    Reproduces the reviewer's ``the consult issued a SECOND fan-out: calls=2`` with
    an injected clock, and asserts ``1`` on the one-read shape. The rows and the
    silence come back from the same call, so the re-dial (and the row loss it
    caused: an id the listing HELD answering "not a peer's") cannot happen at all.
    """
    client, _root = remote_api
    monkeypatch.setattr(peer_rows_mod, "time", _JumpingClock())
    catalog = _relay(
        monkeypatch, silent=[("d_silent", "build-box")], live=[("d_live", "radiant-m4")]
    )

    response = await client.get(f"/v1/desktop/sessions/{'f' * 12}")

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "session_unresolved"
    assert catalog.calls == 1, "the consult issued a SECOND fan-out"


@pytest.mark.asyncio
async def test_a_door_closed_residue_directory_answers_the_shared_404(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-3: no read means no silence, so the residue corner is the 404 the note claims.

    A local ``sessions/<id>/`` the door will not open (``origin: subagent``) makes
    the miss perform NO listing read, and a device's silence may only be reported
    from a read this device actually made. The two-call shape answered 409 here
    from whatever the cache happened to hold — cold AND warm, measured by the
    reviewer — which is a silence about a fan-out that never asked about this id.
    """
    client, root = remote_api
    residue = root / "sessions" / "a1b2c3d4e5f6"
    residue.mkdir(parents=True, exist_ok=True)
    (residue / ORIGIN_NAME).write_text(json.dumps({"origin": "subagent"}))
    monkeypatch.setattr(peer_rows_mod, "time", _JumpingClock())
    catalog = _relay(monkeypatch, silent=[("d_silent", "build-box")])

    for attempt in ("cold", "warm"):
        response = await client.get("/v1/desktop/sessions/a1b2c3d4e5f6")
        assert response.status_code == 404, f"{attempt}: {response.text}"
    assert catalog.calls == 0, "a directory this device holds is not a peer question"


@pytest.mark.asyncio
async def test_the_unresolved_refusal_rides_the_same_door_as_the_snapshot(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/history`` inherits it: one code for one situation, whichever read was made."""
    client, _root = remote_api
    _relay(monkeypatch, silent=[("d_silent", "build-box")])

    response = await client.get(f"/v1/desktop/sessions/{'f' * 12}/history")

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "session_unresolved"


@pytest.mark.asyncio
async def test_every_device_answering_keeps_the_shared_404(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real unknown must STAY unknown: no silence, no new state, same cost."""
    client, _root = remote_api
    catalog = _relay(monkeypatch, live=[("d_live", "radiant-m4")])

    response = await client.get(f"/v1/desktop/sessions/{'f' * 12}")

    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "Requested session, profile, team or subscription not found"
    assert catalog.calls == 1, "the miss itself still costs exactly ONE listing read"


@pytest.mark.asyncio
async def test_an_unreachable_row_is_never_replaced_by_the_silence_refusal(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resolved-but-unreachable row is a DIFFERENT fact from an unresolved id.

    The branch order is the assertion. A row that resolved names the device that
    holds the conversation and why it cannot be reached, which is strictly MORE
    than "a device did not answer"; reaching the silence consult for it would
    throw that away. The consult is installed as a tripwire rather than counted,
    so a future reorder fails loudly instead of quietly changing the sentence.
    """
    client, _root = remote_api
    row = _peer_row(reachable=False, unreachable_reason="connect_failed:ConnectionRefusedError")
    _answer_rows(monkeypatch, row)
    from local_operator.session import peer_rows

    def _must_not_be_consulted(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the silence was consulted for a RESOLVED row")

    monkeypatch.setattr(peer_rows, "unanswered_peers", _must_not_be_consulted)

    response = await client.get(f"/v1/desktop/sessions/{OTHER}")

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "session_is_remote"


@pytest.mark.asyncio
async def test_a_local_session_never_asks_the_peer_projection(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The zero-cost half: a directory this device holds answers from the disk.

    A conversation that moved HOME still has a local directory while the peer's
    cached listing can name it for one TTL, so the order (directory FIRST) is
    also what keeps "where does this live" answered from the durable fact.
    """
    client, root = remote_api
    _seed_local(root, MINE)
    asked = _answer_rows(monkeypatch, _peer_row(id=MINE))

    response = await client.get(f"/v1/desktop/sessions/{MINE}")
    assert response.status_code == 200, response.text
    assert asked == [], "a local read paid a peer lookup"


# ---------------------------------------------------------------------------
# The open path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_peers_row_opens_through_the_one_viewer_seam(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route answers 200 and the bridge is built by the SHARED seam."""
    client, _root = remote_api
    _answer_rows(monkeypatch, _peer_row())
    built = _remote_facade(monkeypatch, _root)

    response = await client.get(f"/v1/desktop/sessions/{OTHER}")
    assert response.status_code == 200, response.text
    assert built, "the desktop built its own facade instead of using remote_open"
    assert built[0]["session_id"] == OTHER
    # THE DESKTOP SURFACE IS DECLARED, and it has to be: the owner's runtime
    # advertises the desktop watch capability against a surface that says
    # "desktop", and the renderer's presence lease rides that same negotiation.
    assert built[0]["surface"] == "desktop"
    # THE ROW THE POOL RESOLVED is handed to the seam, so the placement and the
    # seed come from ONE read rather than from a second lookup that could
    # disagree with it.
    assert built[0]["row"].id == OTHER
    # NOTHING IS WRITTEN ON THIS DEVICE for a conversation it does not hold.
    assert not (_root / "sessions" / OTHER).exists()


@pytest.mark.asyncio
async def test_the_transcript_is_read_off_the_wire_not_off_this_disk(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The peer's rows reach ``/history`` in the contract's entry shape.

    The wire carries MESSAGES, so the projection into an entry is asserted here:
    the id from the row, the ``kind`` the durable encoder writes, and the content
    itself — which is what makes a remote page indistinguishable to the renderer
    from a local one.
    """
    client, _root = remote_api
    _answer_rows(monkeypatch, _peer_row())
    _remote_facade(monkeypatch, _root)

    rows = [Message.user("hello from the peer"), Message.assistant("hello back")]

    def fake_history(self: AttachedSession) -> list[Any]:
        return list(rows)

    monkeypatch.setattr(AttachedSession, "history", fake_history)

    response = await client.get(f"/v1/desktop/sessions/{OTHER}/history")
    assert response.status_code == 200, response.text
    page = response.json()["result"]
    assert page["cursor_missing"] is False
    assert [entry["id"] for entry in page["entries"]] == [row.id for row in rows]
    assert [entry["payload"]["kind"] for entry in page["entries"]] == ["message", "message"]
    assert "hello from the peer" in json.dumps(page["entries"])
    # ONE STAMP PER PAGE, and it is THIS device's clock: this request did NOT
    # negotiate ``entry_ts``, so the wire carries no entry time for these rows and
    # a page is dated when it is served rather than pretending to know when the
    # user sent it. The vocabulary says so on every row (see the method and
    # ``docs/DESKTOP_API.md``) instead of leaving the reader to infer it.
    assert len({entry["ts"] for entry in page["entries"]}) == 1
    assert page["entries"][0]["ts"] > 0
    assert {entry["ts_source"] for entry in page["entries"]} == {"served"}


@pytest.mark.asyncio
async def test_a_cold_peer_history_is_never_published_as_an_empty_conversation(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """D5-core's failure state: an unservable page must not read as "no messages".

    The defect this pins was precisely that conflation: a cold peer read answered
    ``{entries: [], has_more: false, cursor_missing: false}`` — the same envelope a
    conversation with no rows produces — so the renderer could not tell "there is
    nothing here" from "we could not fetch it". The stored-page fallback closes it
    from both sides: when the owner's journal CAN be reached the page carries its
    rows (``tests/unit/network/test_remote_viewer.py``, over two real relays), and
    when it cannot, the empty answer is marked untrustworthy with the contract's
    existing word for it rather than being published as an empty conversation.

    Here there is no relay on this device at all — the facade is cold, its window
    raised, and the relay dial answers nothing — which is the ordinary shape of a
    viewer whose own ``lop`` relay is down. Nothing is written on this disk either
    way: the fallback reads the OWNER's journal over the wire, never
    ``<root>/sessions/<id>`` (§3.4).
    """
    client, root = remote_api
    _answer_rows(monkeypatch, _peer_row())
    _remote_facade(monkeypatch, root)

    response = await client.get(f"/v1/desktop/sessions/{OTHER}/history")
    assert response.status_code == 200, response.text
    page = response.json()["result"]
    assert page == {
        "entries": [],
        "has_more": False,
        "cursor_missing": True,
        "has_newer": None,
    }
    assert not (root / "sessions" / OTHER).exists()


@pytest.mark.asyncio
async def test_a_page_that_runs_dry_asks_the_peer_for_older_rows(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``has_more`` must not mean "the window ended" while the peer holds more."""
    client, _root = remote_api
    _answer_rows(monkeypatch, _peer_row())
    _remote_facade(monkeypatch, _root)

    newest = Message.user("newest")
    older = Message.user("older")
    loaded = [newest]

    def fake_history(self: AttachedSession) -> list[Any]:
        # WHAT THE FACADE HOLDS, which ``load_older_display_page`` grows in place:
        # the same contract the real method has (it inserts at the front of the
        # loaded window and returns the page it added).
        return list(loaded)

    asked: list[str] = []

    async def fake_older(self: AttachedSession) -> list[Any]:
        asked.append("older")
        loaded[:0] = [older]
        return [older]

    monkeypatch.setattr(AttachedSession, "history", fake_history)
    monkeypatch.setattr(AttachedSession, "load_older_display_page", fake_older)
    # The facade reports a token while it still has rows to fetch; only then is
    # asking the peer the right move.
    monkeypatch.setattr(
        AttachedSession, "history_before_token", property(lambda self: "tok"), raising=True
    )

    response = await client.get(f"/v1/desktop/sessions/{OTHER}/history")
    assert response.status_code == 200, response.text
    page = response.json()["result"]
    assert asked == ["older"], "the peer was never asked for the rows above the window"
    assert [entry["id"] for entry in page["entries"]] == [older.id, newest.id]
    assert page["has_more"] is True, "a page with rows still behind it claimed to be the end"


@pytest.mark.asyncio
async def test_a_peer_completion_is_read_from_the_peer_not_from_a_local_store(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """This device's ``attention.db`` is empty-by-construction for an id it lacks.

    Publishing that empty read as the conversation's receipts claims "nothing
    unseen here" about a completion that arrived on the peer -- the exact claim a
    notification is built on -- so the owner's own state is what the snapshot
    reports, and the local row for the same id is not even consulted.
    """
    client, root = remote_api
    _answer_rows(monkeypatch, _peer_row())
    _remote_facade(monkeypatch, root)

    import uuid as _uuid

    from local_operator.session.attention import AttentionStore

    local = AttentionStore(root / "attention.db")
    # A REAL token shape: the store validates it as the CANONICAL form of a
    # UUID, because the token is the runtime's own completion identity rather
    # than any opaque string.
    local_token = str(_uuid.uuid4())
    local.publish(f"session/{OTHER}", local_token, "entry-1", "complete")
    local_state = local.state(f"session/{OTHER}")
    assert local_state["unseen"] is True, f"the fixture's local row reads {local_state}"

    from local_operator.session.frontend_state import FrontendSessionState

    # THE WIRE'S ANSWER, deliberately different from the local row above: the
    # owner says the completion has been seen, and this is what must be published.
    wire_attention = {
        "conversation_id": f"session/{OTHER}",
        "completion_token": local_token,
        "anchor_id": "entry-1",
        "kind": "complete",
        "reason": "done",
        "cause": "",
        "unseen": False,
        "revision": [1, 1],
    }
    monkeypatch.setattr(
        AttachedSession,
        "frontend_state",
        property(
            lambda self: FrontendSessionState(
                session_id=OTHER, epoch="e1", attention=dict(wire_attention)
            )
        ),
        raising=True,
    )

    response = await client.get(f"/v1/desktop/sessions/{OTHER}")
    assert response.status_code == 200, response.text
    snapshot = response.json()["result"]
    published = snapshot["payload"]["frontend"]["snapshot"]["attention"]
    assert published["unseen"] is False, (
        "the desktop published this device's own receipt row about a peer's "
        f"conversation instead of the owner's answer: {published}"
    )
    assert published["conversation_id"] == f"session/{OTHER}"


# ---------------------------------------------------------------------------
# Attachments: a sentence, never a 500
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_peers_attachment_names_the_device_instead_of_500ing(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bytes are on the peer's disk, and the answer says so."""
    client, root = remote_api
    _answer_rows(monkeypatch, _peer_row())
    digest = "a" * 32

    response = await client.get(f"/v1/desktop/sessions/{OTHER}/attachments/{digest}")
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "attachment_on_peer"
    assert "build-box" in detail["message"], detail

    # AND THE ORDINARY CASES DO NOT MOVE: an unknown id is still the 404, and a
    # LOCAL conversation whose digest resolves nowhere is still the 404 the store's
    # own contract gives.
    _seed_local(root, MINE)
    missing = await client.get(f"/v1/desktop/sessions/{'f' * 12}/attachments/{digest}")
    assert missing.status_code == 404, missing.text
    local_miss = await client.get(f"/v1/desktop/sessions/{MINE}/attachments/{digest}")
    assert local_miss.status_code == 404, local_miss.text


@pytest.mark.asyncio
async def test_a_peer_attachment_this_device_holds_is_still_served(
    remote_api: tuple[AsyncClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conversation that moved here keeps its images: the digest IS the content.

    The store is content-addressed, so "this device has a copy" is answered by
    the store itself rather than by where the conversation currently lives.
    """
    client, root = remote_api
    _answer_rows(monkeypatch, _peer_row())
    import base64

    payload = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 64).decode()
    ref = AttachmentStore(root / ATTACHMENTS_DIRNAME).put(payload, "image/png")
    assert ref is not None

    response = await client.get(f"/v1/desktop/sessions/{OTHER}/attachments/{ref.digest}")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("image/png")


# ---------------------------------------------------------------------------
# The bytes a peer-bound prompt sends must stay resolvable on THIS device
# ---------------------------------------------------------------------------


def _real_png(width: int, height: int) -> bytes:
    """A real, decodable PNG with compressible but structured content.

    REAL BYTES because the transform under test IS an image ingest: a
    hand-built ``b"\x89PNG..." + zeros`` payload is not decodable, so
    ``image_blocks`` drops it and a cell built on one would measure the drop
    rather than the bound. Compressible because the shapes that matter here are
    a few hundred pixels wide and must stay well under the store's floor and the
    route's own body bound.
    """
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return len(data).to_bytes(4, "big") + tag + data + zlib.crc32(tag + data).to_bytes(4, "big")

    rows = []
    for y in range(height):
        line = bytearray(b"\x00")
        for x in range(width):
            tile = ((x // 32) * 6 + (y // 32) * 3) % 256
            line += bytes((tile, (tile + 40) % 256, (tile + 90) % 256))
        rows.append(bytes(line))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00",
        )
        + chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
        + chunk(b"IEND", b"")
    )


def _wire(raw: bytes, mime: str = "image/png") -> dict[str, str]:
    return {"data_b64": base64.b64encode(raw).decode("ascii"), "mime_type": mime}


@pytest.mark.asyncio
async def test_a_peer_bound_prompt_sends_and_stages_what_the_owner_will_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The transform's contract, at the object the route calls — and its four gates.

    WHY IT MUST EXIST. The owner's journal row references
    ``{"attachment": <digest>}``, the owner's runtime is another process with its
    own config dir, and this device's only read resolves ``<root>/attachments``.
    Without a copy here the picture the user had just sent answered ``409
    attachment_on_peer`` while the prompt reported ``admitted`` — the drop the
    sweep over two real relays in ``tests/unit/network/test_remote_viewer.py``
    reproduces row by row.

    WHY IT RETURNS WHAT IT SENT. The owner journals the digest of what it
    RECEIVES, so a mirror of the raw wire bytes only resolves while the owner
    keeps those bytes verbatim — and it does not, above ``IMAGE_INGEST_MAX_EDGE``
    (1024 px), over ``IMAGE_MAX_BYTES``, or with an EXIF ``Orientation`` to bake
    in. Round 1 measured that end to end; this cell pins the shape that fixes
    it: the returned payload is what the owner's own ingest makes of the input,
    and the digest staged is the digest OF THAT.

    THE GATES, each because it is a way to pay for nothing: a LOCAL conversation
    comes back untouched and stages nothing (this device's runtime runs that very
    ingest and writes that very store); an image the owner would DISCARD is
    dropped rather than sent (the alternative is the same loss one layer down);
    a payload under the transcript's externalise floor stages nothing, because
    such a row keeps its bytes inline; and ``AttachmentStore.put``'s silence is
    respected — a failed write must never refuse a prompt the owner would admit.
    """
    from local_operator.imaging import sniff_image
    from local_operator.server.utils.desktop_sessions import (
        DesktopSessionBridge,
        DesktopSessions,
    )

    root = tmp_path.resolve()
    row = _peer_row(id=OTHER)
    remote = DesktopSessionBridge(root, OTHER, "", remote_row=row)
    local = DesktopSessionBridge(root, MINE, "")
    store = root / ATTACHMENTS_DIRNAME

    oversize = _wire(_real_png(1025, 640))
    original = oversize["data_b64"]
    assert sniff_image(base64.b64decode(original)).width == 1025  # type: ignore[union-attr]

    # A LOCAL CONVERSATION IS UNTOUCHED, byte for byte, and pays no write.
    assert await local.prepare_peer_images([oversize]) == [oversize]
    assert not store.exists(), "a LOCAL conversation staged an image it already owns"

    # AN IMAGE-LESS PROMPT WRITES NOTHING.
    assert await remote.prepare_peer_images([]) == []
    assert not store.exists(), "an image-less prompt wrote a store"

    # THE BOUND: what comes back is the OWNER'S ingest output, not the wire bytes.
    prepared = await remote.prepare_peer_images([oversize])
    assert len(prepared) == 1
    settled_bytes = base64.b64decode(prepared[0]["data_b64"])
    assert settled_bytes != base64.b64decode(original), (
        "an image over IMAGE_INGEST_MAX_EDGE came back verbatim, so the owner's ingest "
        "will rewrite it and the digest staged here cannot be the one it journals"
    )
    assert sniff_image(settled_bytes).width <= 1024  # type: ignore[union-attr]
    digest = hashlib.sha256(settled_bytes).hexdigest()[:32]

    # AND IT IS WHAT IS STAGED: exactly one blob, under the digest of the returned
    # bytes. The old shape of the defect is a blob under a name no row references.
    assert sorted(p.name for p in store.glob("*.bin")) == [f"{digest}.bin"]

    # THE READ THIS EXISTS FOR, through the pool's own door (the one the route
    # opens): the bytes served are the bytes the owner's row will name, and the
    # peer is never consulted for them.
    _answer_rows(monkeypatch, row)
    pool = DesktopSessions(root)
    served, served_mime = await pool.attachment(OTHER, digest)
    assert served == settled_bytes
    assert served_mime == prepared[0]["mime_type"]

    # AN IMAGE THE OWNER WOULD DISCARD IS NOT SENT, and stages nothing.
    junk = {
        "data_b64": base64.b64encode(b"not an image at all").decode("ascii"),
        "mime_type": "image/png",
    }
    before = sorted(p.name for p in store.glob("*.bin"))
    assert await remote.prepare_peer_images([junk]) == []
    assert sorted(p.name for p in store.glob("*.bin")) == before

    # AND THE FLOOR: a payload the owner would leave INLINE is returned (it must
    # still be sent) but not staged, because nothing would reference the blob.
    tiny = _wire(_real_png(16, 16))
    tiny_prepared = await remote.prepare_peer_images([tiny])
    assert tiny_prepared, "a sub-floor image was dropped instead of sent"
    tiny_digest = hashlib.sha256(base64.b64decode(tiny_prepared[0]["data_b64"])).hexdigest()[:32]
    assert not (store / f"{tiny_digest}.bin").exists()
    assert sorted(p.name for p in store.glob("*.bin")) == [f"{digest}.bin"]


# ---------------------------------------------------------------------------
# The entry-time vocabulary at the ROUTE layer (agent review round 1, R-2).
#
# `entry_ts=1` is the per-request signal the whole feature hangs on: it is what
# tells the daemon a renderer can read `ts_source`, and therefore what decides
# whether a wire row the owner cannot stamp comes back `null` + "unstated"
# instead of a fabricated serve-stamp. The unit cells for it all called
# `bridge.history(entry_times=…)` DIRECTLY or drove the raw socket, so a renamed
# or unthreaded parameter on any of the three route doors shipped silently —
# measured by the reviewer: wrapping `snapshot` to force `entry_times=False` left
# `tests/unit/server` fully green.
# ---------------------------------------------------------------------------

#: The owner's own clock, far enough from `time.time()` that a serve-stamp cannot
#: be mistaken for it.
OWNER_ENTRY_TS = 1_700_000_000.0


def _wire_facade(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    rows: list[Any],
    entry_times: dict[str, float],
) -> None:
    """A REAL cold facade whose wire window is ``rows`` and whose join is ``entry_times``.

    ``is_cold`` is the ONE predicate this lies about, and it is the predicate that
    chooses between the wire branch and the owner's stored journal — which is the
    branch the ``ts_source`` table lives on. Everything else is the production
    facade (its frontend state store, its tokens, its placement), so the pool, the
    identity check and the snapshot's own field reads stay real.
    """
    from local_operator.session import remote_open

    async def fake_open(session_id: str, **kwargs: Any) -> Any:
        async def refuse_takeover() -> None:
            raise AssertionError("a remote viewer never takes over")

        owner = StubRemoteOwner()
        session = await AttachedSession.cold(
            session_id,
            config_dir=kwargs.get("config_dir", root),
            cwd="",
            takeover_factory=refuse_takeover,
            surface=kwargs.get("surface", "desktop"),
            owner=owner,
            seed=owner.seed(),
        )
        session.history = lambda: list(rows)  # type: ignore[method-assign]
        session.history_entry_times = lambda: dict(entry_times)  # type: ignore[method-assign]
        return session

    monkeypatch.setattr(remote_open, "open_remote_viewer", fake_open)
    monkeypatch.setattr(AttachedSession, "is_cold", property(lambda self: False))


def _entry_time_rig(monkeypatch: pytest.MonkeyPatch, root: Path) -> tuple[str, str]:
    """Install the peer row + facade, and answer ``(shippable id, unshippable id)``.

    A mid-turn pair as the wire would carry it: a user row the owner CAN stamp
    (it is in the join) beside one it cannot (a live suffix, absent from the join
    by construction). One page, two vocabularies — the real shape.
    """
    _answer_rows(monkeypatch, _peer_row())
    shippable = Message.user("the durable row")
    unshippable = Message.user("a live suffix")
    _wire_facade(
        monkeypatch,
        root,
        rows=[shippable, unshippable],
        entry_times={shippable.id: OWNER_ENTRY_TS},
    )
    return shippable.id, unshippable.id


def _by_id(page: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in page["entries"]}


@pytest.mark.asyncio
async def test_the_history_route_honours_entry_ts(remote_api, monkeypatch) -> None:
    """R-2, door one: ``/history``, both settings of the flag, one page."""
    client, root = remote_api
    shippable, unshippable = _entry_time_rig(monkeypatch, root)

    asked = (await client.get(f"/v1/desktop/sessions/{OTHER}/history?entry_ts=1")).json()["result"]
    rows = _by_id(asked)
    assert rows[shippable]["ts"] == OWNER_ENTRY_TS, "the owner's instant was not used"
    assert rows[shippable]["ts_source"] == "entry"
    assert rows[unshippable]["ts"] is None, "an unprovable instant was fabricated"
    assert rows[unshippable]["ts_source"] == "unstated"

    # The same request WITHOUT the flag: today's bytes, one stamp per page, and
    # the vocabulary says which clock it is.
    legacy = (await client.get(f"/v1/desktop/sessions/{OTHER}/history")).json()["result"]
    legacy_rows = _by_id(legacy)
    # A row the owner CAN stamp keeps "entry" without the flag too: the join is a
    # FACT about the row, not something the renderer turns on. The flag decides
    # only what happens to a row the owner CANNOT stamp.
    assert legacy_rows[shippable]["ts_source"] == "entry"
    assert legacy_rows[shippable]["ts"] == OWNER_ENTRY_TS
    assert legacy_rows[unshippable]["ts_source"] == "served"
    assert (
        abs(legacy_rows[unshippable]["ts"] - OWNER_ENTRY_TS) > 1_000_000
    ), "the serve-stamp is this device's clock, not the owner's"


@pytest.mark.asyncio
async def test_the_snapshot_route_threads_entry_ts_into_its_embedded_page(
    remote_api, monkeypatch
) -> None:
    """R-2, door two — and the one an opening renderer actually paints from.

    The snapshot's ``payload.history`` is served by the same reader, so it must
    answer the flag identically; a snapshot that ignored it would hand the
    renderer one vocabulary at open and another on its first scroll.
    """
    client, root = remote_api
    shippable, unshippable = _entry_time_rig(monkeypatch, root)

    asked = (await client.get(f"/v1/desktop/sessions/{OTHER}?entry_ts=1")).json()["result"]
    rows = _by_id(asked["payload"]["history"])
    assert rows[shippable]["ts_source"] == "entry"
    assert rows[shippable]["ts"] == OWNER_ENTRY_TS
    assert rows[unshippable]["ts_source"] == "unstated" and rows[unshippable]["ts"] is None

    legacy = (await client.get(f"/v1/desktop/sessions/{OTHER}")).json()["result"]
    legacy_rows = _by_id(legacy["payload"]["history"])
    assert legacy_rows[shippable]["ts_source"] == "entry"
    assert legacy_rows[shippable]["ts"] == OWNER_ENTRY_TS
    assert legacy_rows[unshippable]["ts_source"] == "served"
    assert abs(legacy_rows[unshippable]["ts"] - OWNER_ENTRY_TS) > 1_000_000


@pytest.mark.asyncio
async def test_the_events_route_forwards_entry_ts_into_the_open_frame(remote_api, monkeypatch):
    """R-2, door three, split the way the transport forces it to be.

    The ``/events`` response is an SSE stream and QA round 1 could not get a frame
    out of it under httpx's ASGITransport in a 25 s bound, so this pins the two
    halves separately rather than faking one: the ROUTE forwards the flag into
    ``bridge.events`` (a recording double — the half a missing forward fails), and
    the GENERATOR's open frame is asserted directly on the real bridge below.
    """
    client, root = remote_api
    _answer_rows(monkeypatch, _peer_row())
    seen: list[dict[str, Any]] = []

    class _RecordingBridge(SimpleNamespace):
        def subscribe(self, **kwargs: Any) -> Any:
            return SimpleNamespace(id="sub", frontend_replace=kwargs.get("frontend_replace"))

        def note_stream_ended(self, sub: Any) -> None:
            return None

        def events(self, sub: Any, **kwargs: Any) -> Any:
            seen.append(kwargs)

            async def _empty() -> Any:
                if False:  # pragma: no cover — keeps this an async generator
                    yield {}

            return _empty()

    bridge = _RecordingBridge()

    class _Pool:
        @contextlib.asynccontextmanager
        async def session(self, session_id: str, **kwargs: Any):
            yield bridge

    client._transport.app.state.desktop_sessions = _Pool()
    await client.get(f"/v1/desktop/sessions/{OTHER}/events?entry_ts=1")
    assert seen and seen[0].get("entry_times") is True, seen

    seen.clear()
    await client.get(f"/v1/desktop/sessions/{OTHER}/events")
    assert seen and seen[0].get("entry_times") is False, seen


@pytest.mark.asyncio
async def test_the_events_open_frame_obeys_entry_ts(remote_api, monkeypatch) -> None:
    """The generator half of the cell above, on the real bridge.

    QA's row 10 in-tree: the frame the stream opens with is the snapshot, so its
    embedded page must carry the same vocabulary the standalone ``/history`` does
    for the same rows.
    """
    _client, root = remote_api
    shippable, unshippable = _entry_time_rig(monkeypatch, root)

    pool = DesktopSessions(root)
    async with pool.session(OTHER) as bridge:
        sub = bridge.subscribe()
        stream = bridge.events(sub, epoch=bridge.epoch, after_seq=bridge.sequence, entry_times=True)
        try:
            frame = None
            async for candidate in stream:
                if candidate.get("type") == "snapshot":
                    frame = candidate
                    break
        finally:
            await stream.aclose()
    assert frame is not None, "the stream opened without ever yielding its snapshot"
    page = frame["payload"]["history"]
    rows = _by_id(page)
    assert rows[shippable]["ts_source"] == "entry"
    assert rows[unshippable]["ts_source"] == "unstated"

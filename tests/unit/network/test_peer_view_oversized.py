"""A peer viewer reaches a session whose NEWEST entry is larger than one page.

THE DEFECT: a session on another device could neither ``send`` to nor ``peek`` a
conversation whose newest transcript row was too large for one display page
(``history_window.DISPLAY_HISTORY_BYTES``) or one mesh record
(``wire.MAX_RECORD_BYTES``). The owner answered the live view with
``full_required`` — "replay the transcript locally", which a viewer on another
device cannot do — so the remote facade refused the BIND ("this session's
newest history is too large to send here"); a mesh ``send``/``peek`` rides that
same bound viewer, so the message was never delivered. The relay's stored read
refused the whole page the same way ("a single entry too large to serve").

THE FIX keeps every row and cuts only the oversized BODY, marked with
``history_window.TRUNCATED_ENTRY_MARKER``. These tests pin it on the real
helpers and over two real relays (the ``test_remote_viewer`` rig: a real
``Session`` behind a real ``RuntimeServer`` on the owning device).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import CustomMessage, Message
from local_operator.network import relay
from local_operator.session.history_window import display_window
from local_operator.session.peer_rows import clear_cache
from local_operator.session.transcript import Transcript, TranscriptEntry, _entry_to_message
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — fixtures
from tests.unit.network.test_remote_viewer import _open, _user_texts, _wait
from tests.unit.network.test_session_plane import (
    Devices,
    _create_named_session_on_a_real_peer,
    _dial_to,
    _listen,
    _pair,
    _seed_journal,
)

MARKER_WORDS = "open the conversation on the device that holds it for the full entry"


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Devices:
    pair: Devices = request.getfixturevalue("devices")
    return pair


@pytest.fixture(autouse=True)
def _no_cached_rows() -> Any:
    clear_cache()
    yield
    clear_cache()


def _window(transcript: Transcript, **kwargs: Any) -> Any:
    entries = transcript.entries()
    return display_window(
        transcript,
        conversation_id="c",
        owner_epoch="e",
        through_id=entries[-1].id if entries else None,
        **kwargs,
    )


# --- the relay's stored read -------------------------------------------------


def test_an_oversized_newest_row_is_served_bounded_and_marked_not_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = 64 * 1024
    monkeypatch.setattr(relay, "SESSION_HISTORY_REPLY_BUDGET_BYTES", budget)
    older = {"id": "old", "ts": 1.0, "type": "message", "payload": {"role": "user", "content": []}}
    newest = {
        "id": "big",
        "ts": 2.0,
        "type": "message",
        "payload": {
            "role": "assistant",
            "id": "big",
            "content": [{"type": "text", "text": "y" * (budget * 2)}],
            "tool_calls": [{"id": "c1", "name": "bash", "arguments": {"cmd": "z" * budget}}],
        },
    }
    page = {"entries": [older, newest], "has_more": False, "cursor_missing": False}

    fitted = relay.fit_session_history_reply(1, page)

    assert relay.session_history_reply_bytes(1, fitted) <= budget
    # BOTH rows fit once the newest is cut, so nothing was left behind and the
    # page must not claim more (CI run 43367540518: a one-row journal came back
    # ``has_more: true``, which sends a reader paging back into an empty page).
    assert [entry["id"] for entry in fitted["entries"]] == ["old", "big"]
    assert fitted["has_more"] is False, "no row was dropped, so there is no more"
    row = fitted["entries"][-1]
    # IDENTITY KEPT: id/ts/type and the role and tool-call identity the
    # renderer pairs on; only the body was cut, and it says so.
    assert (row["id"], row["ts"], row["type"]) == ("big", 2.0, "message")
    assert row["payload"]["role"] == "assistant"
    assert row["payload"]["tool_calls"] == [{"id": "c1", "name": "bash", "arguments": {}}]
    text = row["payload"]["content"][0]["text"]
    assert MARKER_WORDS in text and text.startswith("y")
    # The caller's page is not mutated: the canonical row stays whole.
    assert len(page["entries"][-1]["payload"]["content"][0]["text"]) == budget * 2

    # An OLDER row that no longer fits beside the bounded one IS left behind,
    # and then ``has_more`` must say so: the reader reaches it by cursor.
    heavy = {**older, "payload": {"role": "user", "content": [{"text": "o" * budget}]}}
    dropped = relay.fit_session_history_reply(1, {**page, "entries": [heavy, newest]})
    assert [entry["id"] for entry in dropped["entries"]] == ["big"]
    assert dropped["has_more"] is True, "rows left behind must be reachable by cursor"
    assert relay.session_history_reply_bytes(1, dropped) <= budget


def test_metadata_that_cannot_fit_is_still_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(relay, "SESSION_HISTORY_REPLY_BUDGET_BYTES", 4096)
    page = {
        "entries": [{"id": "a", "ts": 1.0, "type": "message", "payload": {"role": "user"}}],
        "has_more": False,
        "journal_ordinals": {"x": "x" * 8192},
    }
    with pytest.raises(relay.MeshRefusal, match="metadata"):
        relay.fit_session_history_reply(1, page)


def test_a_custom_row_with_bulk_outside_details_is_bounded_not_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1-3: a ``CustomMessage`` is ``extra="allow"``, so its bulk can live in a
    list ``content`` or an extra key. Cutting only ``details`` left it oversized
    and the page was refused as "metadata too large"; identity is all it keeps."""
    budget = 64 * 1024
    monkeypatch.setattr(relay, "SESSION_HISTORY_REPLY_BUDGET_BYTES", budget)
    newest = {
        "id": "cm",
        "ts": 3.0,
        "type": "custom",
        "payload": {
            "custom_type": "handoff",
            "attribution": "agent",
            "id": "cm",
            "details": {"note": "d" * budget},
            "content": [{"type": "text", "text": "c" * budget}],
            "extra_blob": "e" * (budget * 2),
        },
    }
    page = {"entries": [newest], "has_more": False, "cursor_missing": False}

    fitted = relay.fit_session_history_reply(1, page)

    assert relay.session_history_reply_bytes(1, fitted) <= budget
    assert fitted["has_more"] is False, "the only row was served, so there is no more"
    row = fitted["entries"][-1]
    assert (row["id"], row["ts"], row["type"]) == ("cm", 3.0, "custom")
    assert set(row["payload"]) == {"custom_type", "attribution", "id", "details"}
    assert row["payload"]["custom_type"] == "handoff"
    assert MARKER_WORDS in row["payload"]["details"]["truncated"]
    # A string ``content`` is kept as a key, carrying the marker.
    stringy = {**newest, "payload": {**newest["payload"], "content": "s" * (budget * 2)}}
    cut = relay.bounded_history_entry(stringy, budget // 4)
    assert MARKER_WORDS in cut["payload"]["content"]
    assert "extra_blob" not in cut["payload"]


@pytest.mark.asyncio
async def test_a_bounded_stored_custom_message_still_decodes_as_custom(tmp_path: Path) -> None:
    """R2-1: a REAL stored ``CustomMessage`` row is ``type: "message"`` with
    ``payload.kind: "custom"`` and its id on the envelope only. Dropping
    ``kind`` made the decoder read it as a plain message, fail, and drop it —
    the oversized row vanished instead of showing the marker."""
    transcript = Transcript(tmp_path)
    await transcript.append_message(
        CustomMessage(custom_type="hub_message", details={"body": "x" * 600_000})
    )
    stored = transcript.entries()[-1]
    raw = {"id": stored.id, "ts": stored.ts, "type": stored.type, "payload": stored.payload}
    assert raw["type"] == "message" and raw["payload"].get("kind") == "custom"

    cut = relay.bounded_history_entry(raw, 16 * 1024)

    assert len(json.dumps(cut)) < 16 * 1024
    assert cut["payload"]["kind"] == "custom"
    decoded = _entry_to_message(TranscriptEntry(**cut))
    assert isinstance(decoded, CustomMessage), decoded
    assert decoded.id == stored.id and decoded.custom_type == "hub_message"
    assert MARKER_WORDS in decoded.details["truncated"]


# --- the owner's display window ----------------------------------------------


@pytest.mark.asyncio
async def test_the_bounded_page_keeps_the_row_and_the_exact_page_still_escalates(
    tmp_path: Path,
) -> None:
    transcript = Transcript(tmp_path)
    older = Message.user("before the big one")
    big = Message.assistant("oversized prose " * 100_000)
    await transcript.append_messages([older, big])

    # A LOCAL viewer's answer is unchanged: escalate to its exact local replay.
    assert _window(transcript).status == "full_required"

    page = _window(transcript, bound_oversized=True)
    assert page.status == "ok"
    assert [m.id for m in page.messages][-1] == big.id
    assert MARKER_WORDS in page.messages[-1].text
    assert len(page.model_dump_json().encode()) < 512 * 1024
    # Gap-free: the older row is either on this page or behind its cursor.
    assert older.id in [m.id for m in page.messages] or page.before_token
    # The canonical journal still holds every byte.
    replayed = transcript.build_llm_history()[-1]
    assert isinstance(replayed, Message) and replayed.text == big.text


# --- end to end, over two real relays -----------------------------------------


@pytest.mark.asyncio
async def test_a_peer_send_reaches_a_session_whose_newest_entry_is_oversized(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's report: the send failed and the message was NOT delivered."""
    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer, peer_pair, monkeypatch, name="big", prompt=""
    )
    try:
        big = Message.user("x" * (relay.wire.MAX_RECORD_BYTES + 1024))
        created.owner.on_session_loop(created.owner.session.transcript.append_message(big))
        viewer = await _open(created, monkeypatch)
        try:
            await viewer.bind_runtime()
            assert not viewer.is_cold, viewer.cold_reason
            # PEEK'S SOURCE: the bound viewer's synced rows carry the big row,
            # bounded and marked, under its own id.
            painted = [
                m for m in viewer.display_history_window() if getattr(m, "id", "") == big.id
            ]
            assert painted, "the oversized row was dropped from the peer's view"
            assert MARKER_WORDS in json.dumps(painted[0].model_dump(mode="json"))
            # And the real peek verb over that viewer answers instead of refusing.
            from local_operator.network.cli import _pilot_peek

            payload, lines = await _pilot_peek(viewer, created.session_id, "owner", 12)
            assert MARKER_WORDS in json.dumps(payload) + "\n".join(lines), (payload, lines)

            await viewer.prompt("hello past the big row")
            assert await _wait(
                lambda: any("hello past the big row" in t for t in _user_texts(created))
            ), "the message was not delivered to the owner"
        finally:
            await viewer.dispose()
    finally:
        await asyncio.to_thread(created.stop)


@pytest.mark.asyncio
async def test_a_cold_peer_read_serves_an_oversized_newest_row_marked(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relay half end to end: the stored read no longer refuses the page."""
    from local_operator.server.utils.desktop_sessions import DesktopSessions

    session = "a7c41e0b9d22"
    server_a, server_b, _host_a, _port_a = peer_pair
    # Bounded below the real 8 MiB record so the rig stays fast; the function
    # reads the module constant at call time, on both relays of this process.
    monkeypatch.setattr(relay, "SESSION_HISTORY_REPLY_BUDGET_BYTES", 256 * 1024)
    ids = _seed_journal(server_b.root, session, ["first", "z" * (512 * 1024)])
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    link = _dial_to(server_a, record, host_b, port_b)
    pool = DesktopSessions(server_a.root)
    try:
        async with pool.session(session, read=True) as bridge:
            page = await bridge.history(limit=50)
        # EVERY ROW, IN ORDER, NOTHING DROPPED: the oversized one bounded and
        # marked under its own id, the older one exact beside it.
        assert [entry["id"] for entry in page["entries"]] == ids, page
        newest = page["entries"][-1]["payload"]["content"][0]["text"]
        assert MARKER_WORDS in newest and newest.startswith("z"), newest[:200]
        assert page["entries"][0]["payload"]["content"][0]["text"] == "first"
    finally:
        await pool.close()
        link.close("test")


# --- the locality gate ----------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("locality", ["local", "remote"])
async def test_only_a_non_local_viewer_gets_the_bounded_page(
    tmp_path: Path, locality: str
) -> None:
    """R1-2: the two server routes that pick exact vs bounded by locality.

    A LOCAL viewer can replay the transcript itself, so its sync frame and its
    ``history_page`` answer stay ``full_required`` byte-for-byte; a REMOTE one
    cannot, so both routes hand it the bounded, marked page instead.
    """
    from types import SimpleNamespace

    from local_operator.session.runtime.server import (
        RuntimeServer,
        _serve_peer_bounded_history,
    )
    from tests.unit.session.runtime.test_server import FakeHandle

    transcript = Transcript(tmp_path)
    big = Message.assistant("oversized prose " * 100_000)
    await transcript.append_messages([Message.user("before"), big])
    exact = _window(transcript)
    bounded = _window(transcript, bound_oversized=True)
    assert (exact.status, bounded.status) == ("full_required", "ok")
    expected = "full_required" if locality == "local" else "ok"

    # Route 1: the sync frame (pushed frame and ``frontend_sync`` RPC).
    exact_json = exact.model_dump(mode="json")
    payload: dict[str, Any] = {"display_history": exact_json}
    sync = SimpleNamespace(_peer_display_history=bounded)
    _serve_peer_bounded_history(payload, sync, locality)
    assert payload["display_history"]["status"] == expected
    if locality == "local":
        assert payload["display_history"] == exact_json

    # Route 2: the ``history_page`` op, through the real dispatcher.
    class PagingHandle(FakeHandle):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[bool] = []

        def history_page(self, before: str, anchor: str, bound_oversized: bool = False):
            self.calls.append(bound_oversized)
            return _window(transcript, bound_oversized=bound_oversized).model_dump(mode="json")

    handle = PagingHandle()
    server = RuntimeServer(handle, kind="tui")
    page = await server._dispatch_payload(
        "history_page", {"op": "history_page", "before": "t", "anchor": ""}, locality=locality
    )
    assert handle.calls == [locality != "local"]
    assert page["status"] == expected
    if locality == "remote":
        assert MARKER_WORDS in json.dumps(page)

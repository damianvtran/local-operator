"""The outbound ledger and the per-ask ``ref`` (reply_guard ``outbound_posts``).

The ledger answers "which of my sessions posted this message?" for the routing
and the authorship labels. These tests pin its SHAPE (additive, no body column,
pruned, fail open) and its two call sites (``dm``, ``channel_post``), driven
through the real ``execute_bridge`` with the recording clients the neighbouring
files use.
"""

from __future__ import annotations

import itertools
import re
import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge import reply_guard
from local_operator.team_bridge.client import BridgeUnreachable, TeamBridgeClient
from local_operator.team_bridge.store import OwnerReceipt, Store
from local_operator.team_bridge.tool import BridgeRuntime, execute_bridge
from tests.unit.team_bridge.ref_text import without_ref
from tests.unit.team_bridge.test_bridge_honest_refusals import AWS_KEY_ID, AWS_SECRET
from tests.unit.team_bridge.test_channel_tool import RecordingClient
from tests.unit.team_bridge.test_dm_addressing_and_refusals import (
    DM_ID,
    PEER_ENDPOINT,
    DmClient,
    _member,
    _owner,
)

ENDPOINT = "bep_1"
CONV = "bcv_aaaaaaaa"
REF_LINE = re.compile(r"\n— ref ([0-9A-HJKMNP-TV-Z]{5})\Z")


@pytest.fixture(autouse=True)
def _gates_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "outbound_tool_fence", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)


def _rows(config_dir: Path) -> list[dict[str, Any]]:
    path = reply_guard.guard_path(config_dir)
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM outbound_posts ORDER BY id")]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _runtime(client: Any, config_dir: Path, session_id: str = "sess_1") -> BridgeRuntime:
    return BridgeRuntime(cast(TeamBridgeClient, client), config_dir, "ten_1", _owner(), session_id)


def _record(config_dir: Path, **over: Any) -> reply_guard.OutboundRecord | None:
    args: dict[str, Any] = {
        "endpoint_id": ENDPOINT,
        "conversation_id": CONV,
        "session_id": "sess_1",
        "client_ref": "c1",
        "kind": "dm",
        "ask": True,
    }
    args.update(over)
    return reply_guard.record_outbound(config_dir, **args)


# -- the table itself ---------------------------------------------------------


def test_the_ledger_has_no_body_column_and_never_stores_message_text(tmp_path: Path) -> None:
    assert _record(tmp_path) is not None
    conn = sqlite3.connect(reply_guard.guard_path(tmp_path))
    columns = [row[1] for row in conn.execute("PRAGMA table_info(outbound_posts)")]
    conn.close()
    assert columns == [
        "id",
        "endpoint_id",
        "conversation_id",
        "session_id",
        "client_ref",
        "message_id",
        "seq",
        "thread_root_id",
        "ref",
        "kind",
        "created_at",
    ]


@pytest.mark.asyncio
async def test_a_sent_message_text_is_nowhere_in_the_guard_file(tmp_path: Path) -> None:
    client = DmClient([_member()])
    text = "zebra-quartz-7731 is the marker nobody else writes"
    out = await execute_bridge(
        _runtime(client, tmp_path), {"op": "dm", "endpoint_id": PEER_ENDPOINT, "body": text}
    )
    assert out["ok"] is True, out
    assert _rows(tmp_path), "the send must have left a ledger row"
    assert b"zebra-quartz-7731" not in reply_guard.guard_path(tmp_path).read_bytes()


def test_an_old_shape_guard_file_opens_and_gains_the_table(tmp_path: Path) -> None:
    """An older build's file (no ``outbound_posts``) is extended, not refused."""

    path = reply_guard.guard_path(tmp_path)
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE posts (id INTEGER PRIMARY KEY AUTOINCREMENT, endpoint_id TEXT NOT NULL, "
        "conversation_id TEXT NOT NULL, at REAL NOT NULL, agent_to_agent INTEGER NOT NULL)"
    )
    conn.execute(
        "INSERT INTO posts (endpoint_id, conversation_id, at, agent_to_agent) VALUES ('e','c',1,0)"
    )
    conn.commit()
    conn.close()

    assert _record(tmp_path) is not None
    assert len(_rows(tmp_path)) == 1
    kept = sqlite3.connect(path).execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    assert kept == 1


def test_rows_older_than_14_days_are_pruned_on_write(tmp_path: Path) -> None:
    _record(tmp_path, client_ref="old", now=1_000.0)
    fresh = 1_000.0 + reply_guard.OUTBOUND_RETAIN_S + 10
    _record(tmp_path, client_ref="new", now=fresh)
    assert [r["client_ref"] for r in _rows(tmp_path)] == ["new"]


def test_the_ledger_keeps_at_most_2000_rows_per_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reply_guard, "OUTBOUND_MAX_ROWS", 5)
    for n in range(8):
        _record(tmp_path, client_ref=f"c{n}", now=100.0 + n)
    _record(tmp_path, endpoint_id="bep_other", client_ref="other", now=200.0)
    rows = _rows(tmp_path)
    mine = [r["client_ref"] for r in rows if r["endpoint_id"] == ENDPOINT]
    assert mine == ["c3", "c4", "c5", "c6", "c7"]
    assert [r["client_ref"] for r in rows if r["endpoint_id"] == "bep_other"] == ["other"]
    assert reply_guard.OUTBOUND_MAX_ROWS == 5  # the patched value; real one is below
    monkeypatch.undo()
    assert reply_guard.OUTBOUND_MAX_ROWS == 2000


# -- the ref ------------------------------------------------------------------


def test_a_ref_is_five_crockford_characters_and_unique_per_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [_record(tmp_path, client_ref=f"c{n}") for n in range(60)]
    refs = [r.ref for r in records if r is not None]
    assert len(refs) == 60
    assert all(r and re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{5}", r) for r in refs)
    assert len(set(refs)) == 60

    # A forced collision is retried, not reused.
    draws = iter(list(refs[0] or "") + list("23456"))
    monkeypatch.setattr(reply_guard.secrets, "choice", lambda alphabet: next(draws))
    again = _record(tmp_path, client_ref="again")
    assert again is not None and again.ref == "23456"


def test_a_ref_that_cannot_be_made_unique_is_dropped_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _record(tmp_path)
    assert first is not None and first.ref
    cycle = itertools.cycle(first.ref)
    monkeypatch.setattr(reply_guard.secrets, "choice", lambda alphabet: next(cycle))
    # Every draw spells the first ref again: the attempts run out and the row is
    # written WITHOUT a ref (a ref is never worth a failed send).
    stuck = _record(tmp_path, client_ref="stuck")
    assert stuck is not None and stuck.ref is None and stuck.suffix == ""


@pytest.mark.asyncio
async def test_a_dm_ask_gets_a_ref_line_the_ledger_and_the_receipt(tmp_path: Path) -> None:
    client = DmClient([_member()])
    out = await execute_bridge(
        _runtime(client, tmp_path), {"op": "dm", "endpoint_id": PEER_ENDPOINT, "body": "ping"}
    )
    assert out["ok"] is True, out
    sent = [c["text"] for c in client.calls if c["method"] == "conversation_send"][0]
    match = REF_LINE.search(sent)
    assert match and without_ref(sent) == "ping"
    assert out["ref"] == match.group(1) and f"ref {match.group(1)}" in out["content"]
    [row] = _rows(tmp_path)
    assert row["ref"] == match.group(1)
    assert row["kind"] == "dm" and row["conversation_id"] == DM_ID and row["session_id"] == "sess_1"
    assert row["message_id"] == "bcmsg_1"  # filled after the send, from the hub's answer


@pytest.mark.asyncio
async def test_a_channel_post_ask_gets_a_ref_line_and_a_reply_does_not(tmp_path: Path) -> None:
    owner = _owner()
    (tmp_path / "config.yml").write_text("values:\n  team_bridge:\n    enabled: true\n")
    store = Store(tmp_path)
    store.commit_enrollment(
        owner, OwnerReceipt("bridge_consent_v1", owner.endpoint_id), cos_session_id="sess_1"
    )
    store.record_body_reservation(
        owner,
        conversation_id=CONV,
        message_id="bcmsg_aaaaaaaa",
        conversation_seq=1,
        mention=False,
        body_sha256="a" * 64,
        screening_fingerprint="b" * 64,
        policy_epoch=1,
        consent_version="bridge_consent_v1",
        designated_session="sess_1",
        now=1,
    )
    client = RecordingClient()
    runtime = _runtime(client, tmp_path)

    ask = await execute_bridge(runtime, {"op": "channel_post", "channel_id": CONV, "body": "q?"})
    answer = await execute_bridge(
        runtime,
        {
            "op": "channel_post",
            "channel_id": CONV,
            "body": "an answer",
            "in_reply_to_message_id": "bcmsg_aaaaaaaa",
        },
    )
    assert ask["ok"] and answer["ok"], (ask, answer)
    texts = [c["text"] for c in client.calls]
    assert REF_LINE.search(texts[0]) and without_ref(texts[0]) == "q?"
    assert texts[1] == "an answer" and "ref" not in answer
    asked, answered = _rows(tmp_path)
    assert asked["ref"] and answered["ref"] is None
    assert asked["kind"] == answered["kind"] == "channel_post"


@pytest.mark.asyncio
async def test_a_secret_is_refused_before_any_ref_or_ledger_row(tmp_path: Path) -> None:
    client = DmClient([_member()])
    out = await execute_bridge(
        _runtime(client, tmp_path),
        {
            "op": "dm",
            "endpoint_id": PEER_ENDPOINT,
            "body": f"aws_access_key_id = {AWS_KEY_ID}\naws_secret_access_key = {AWS_SECRET}",
        },
    )
    assert out["ok"] is False
    assert _rows(tmp_path) == []
    assert not [c for c in client.calls if c["method"] == "conversation_send"]


def test_empty_text_gets_no_ref(tmp_path: Path) -> None:
    from local_operator.team_bridge import tool as tool_mod

    runtime = _runtime(DmClient([]), tmp_path)
    record, outgoing = tool_mod._begin_outbound(
        runtime,
        ENDPOINT,
        CONV,
        kind="dm",
        client_ref="c",
        text="  ",
        thread_root_id="",
        in_reply_to="",
    )
    assert record is not None and record.ref is None and outgoing == "  "


@pytest.mark.asyncio
async def test_a_ref_does_not_change_the_duplicate_hash(tmp_path: Path) -> None:
    """The same body twice within the window is still skipped: the guard hashes
    the caller's text, and the ref (different each time) is appended after."""

    client = DmClient([_member()])
    runtime = _runtime(client, tmp_path)
    call = {"op": "dm", "endpoint_id": PEER_ENDPOINT, "body": "same words"}
    first = await execute_bridge(runtime, call)
    second = await execute_bridge(runtime, call)
    assert first["ok"] and second.get("skipped") is True
    assert len(_rows(tmp_path)) == 1


# -- failure handling -----------------------------------------------------------


@pytest.mark.asyncio
async def test_the_row_is_released_when_the_send_fails(tmp_path: Path) -> None:
    class Refusing(DmClient):
        async def conversation_send(self, *args: Any, **kwargs: Any) -> Any:
            raise BridgeUnreachable("hub down")

    out = await execute_bridge(
        _runtime(Refusing([_member()]), tmp_path),
        {"op": "dm", "endpoint_id": PEER_ENDPOINT, "body": "never lands"},
    )
    assert out["ok"] is False and out["code"] == "post_failed"
    assert _rows(tmp_path) == []


@pytest.mark.asyncio
async def test_a_down_ledger_fails_open_the_message_goes_out_unchanged(tmp_path: Path) -> None:
    """A guard path that is a DIRECTORY cannot be opened as a database."""

    reply_guard.guard_path(tmp_path).mkdir(parents=True)
    client = DmClient([_member()])
    out = await execute_bridge(
        _runtime(client, tmp_path), {"op": "dm", "endpoint_id": PEER_ENDPOINT, "body": "still sent"}
    )
    assert out["ok"] is True, out
    sent = [c["text"] for c in client.calls if c["method"] == "conversation_send"]
    assert sent == ["still sent"]  # no ref line, no failure
    assert "ref" not in out

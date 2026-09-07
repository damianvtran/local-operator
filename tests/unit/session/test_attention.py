"""Receipt membership, cross-process durability and passive-read invariants."""

from __future__ import annotations

import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from local_operator.session.attention import AttentionStore, conversation_identity


def test_delayed_duplicate_and_foreign_acknowledgements(tmp_path: Path) -> None:
    path = tmp_path / "attention.db"
    store = AttentionStore(path)
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    store.publish("session/a", a, "message-a", "complete")
    store.publish("session/a", b, "message-b", "error")
    assert store.acknowledge("session/a", a)["unseen"] is True
    assert store.acknowledge("session/a", b)["unseen"] is False
    revision = store.state("session/a")["revision"]
    assert AttentionStore(path).acknowledge("session/a", a)["revision"] == revision
    with pytest.raises(ValueError):
        store.acknowledge("session/b", b)
    with pytest.raises(ValueError):
        store.publish("session/b", b, "message-b", "error")
    assert store.state("session/a")["revision"] == revision


def test_concurrent_clients_converge_without_lost_receipts(tmp_path: Path) -> None:
    path = tmp_path / "attention.db"
    store = AttentionStore(path)
    tokens = [str(uuid.uuid4()) for _ in range(20)]
    for token in tokens:
        store.publish("session/a", token, token, "complete")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda token: AttentionStore(path).acknowledge("session/a", token), reversed(tokens)
            )
        )
    assert not AttentionStore(path).state("session/a")["unseen"]
    # Replaying the journal after owner restart cannot mint a second completion.
    store.publish("session/a", tokens[-1], tokens[-1], "complete")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM completions").fetchone()[0] == 20


def test_reads_do_not_create_or_mutate_storage(tmp_path: Path) -> None:
    path = tmp_path / "attention.db"
    store = AttentionStore(path)
    assert not store.state("session/a")["unseen"]
    assert not path.exists()
    token = str(uuid.uuid4())
    store.publish("session/a", token, "result", "interrupted")
    before = path.stat().st_mtime_ns
    state = store.state("session/a")
    assert state["unseen"]
    assert path.stat().st_mtime_ns == before
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_real_turn_publishes_after_durability_and_survives_resume(tmp_path: Path) -> None:
    from local_operator.harness.types import StreamEndEvent, StreamTextDelta
    from tests.unit.session.test_session import ScriptedStream, make_session

    session = make_session(
        tmp_path,
        ScriptedStream(
            [[StreamTextDelta(delta="Finished result"), StreamEndEvent(stop_reason="stop")]]
        ),
    )
    try:
        await session.prompt("Work")
        state = await session.refresh_attention()
        assert state["unseen"] is True
        assert session._transcript.has_entry(state["anchor_id"])
        await session.acknowledge_attention(state["completion_token"])
        token = state["completion_token"]
    finally:
        await session.dispose()
    resumed = make_session(tmp_path, ScriptedStream([]))
    try:
        state = await resumed.refresh_attention()
        assert state["completion_token"] == token
        assert state["unseen"] is False
    finally:
        await resumed.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["error", "interrupted"])
async def test_empty_failed_turn_has_durable_viewable_outcome(tmp_path: Path, kind: str) -> None:
    from local_operator.harness.types import AgentEndEvent
    from tests.unit.session.test_session import ScriptedStream, make_session

    session = make_session(tmp_path, ScriptedStream([]))
    try:
        session._attention_outcome = AgentEndEvent(
            messages=[],
            error="Fixture failure" if kind == "error" else None,
            aborted=kind == "interrupted",
        )
        await session._publish_attention_outcome()
        state = await session.refresh_attention()
        assert state["kind"] == kind
        assert state["anchor_id"].startswith("completion-")
        assert state["unseen"]
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("seen", [None, "before", "after"])
async def test_legacy_bootstrap_preserves_determinable_seen_state(
    tmp_path: Path, seen: str | None
) -> None:
    import json

    from local_operator.harness.types import Message, TextContent
    from local_operator.session.attention import bootstrap_transcript
    from local_operator.session.transcript import Transcript

    directory = tmp_path / "sessions" / "legacy"
    transcript = Transcript(directory)
    result = Message(role="assistant", content=[TextContent(text="Historical result")])
    await transcript.append_message(result)
    timestamp = transcript.entries()[-1].ts
    if seen:
        (tmp_path / "mobile-seen.json").write_text(
            json.dumps({"sessions": {"legacy": timestamp + (1 if seen == "after" else -1)}})
        )
    store = AttentionStore(tmp_path / "attention.db")
    bootstrap_transcript(transcript, store)
    state = store.state("session/legacy")
    assert state["completion_token"]
    assert state["unseen"] is (seen == "before")
    store.acknowledge("session/legacy", state["completion_token"])
    bootstrap_transcript(transcript, AttentionStore(store.path))
    assert not store.state("session/legacy")["unseen"]
    newer = str(uuid.uuid4())
    store.publish("session/legacy", newer, "new-result", "complete")
    bootstrap_transcript(transcript, store)
    assert store.state("session/legacy")["completion_token"] == newer
    assert store.state("session/legacy")["unseen"]


@pytest.mark.asyncio
async def test_crash_and_fork_journal_identity(tmp_path: Path) -> None:
    from local_operator.harness.types import Message, TextContent
    from local_operator.session.attention import (
        ATTENTION_CUSTOM_TYPE,
        bootstrap_transcript,
    )
    from local_operator.session.transcript import Transcript

    store = AttentionStore(tmp_path / "attention.db")
    owner = Transcript(tmp_path / "sessions" / "owner")
    token = str(uuid.uuid4())
    await owner.append_custom(
        "attention_started", {"conversation_id": "session/owner", "token": token}
    )
    await owner.append_message(
        Message(role="assistant", content=[TextContent(text="durable result")])
    )
    bootstrap_transcript(owner, store)
    assert store.state("session/owner")["kind"] == "interrupted"
    assert store.state("session/owner")["unseen"]
    store.acknowledge("session/owner", token)
    bootstrap_transcript(owner, store)
    assert not store.state("session/owner")["unseen"]
    fork = Transcript(tmp_path / "sessions" / "fork")
    message = Message(role="assistant", content=[TextContent(text="inherited result")])
    await fork.append_message(message)
    await fork.append_custom(
        ATTENTION_CUSTOM_TYPE,
        {"conversation_id": "session/owner", "token": token, "anchor": "old", "kind": "complete"},
    )
    bootstrap_transcript(fork, store)
    assert store.state("session/fork")["completion_token"] != token
    assert not store.state("session/fork")["unseen"]


@pytest.mark.parametrize(
    "damage", ["corrupt_bytes", "partial_schema", "missing_receipts", "dropped_schema"]
)
def test_existing_database_damage_is_not_an_empty_read_state(tmp_path: Path, damage: str) -> None:
    path = tmp_path / "attention.db"
    store = AttentionStore(path)
    if damage == "corrupt_bytes":
        path.write_bytes(b"this is not a SQLite database")
    elif damage == "missing_receipts":
        store.publish("session/a", str(uuid.uuid4()), "result", "complete")
        with sqlite3.connect(path) as conn:
            conn.execute("DROP TABLE receipts")
    else:
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE completions(sequence INTEGER)")
            if damage == "dropped_schema":
                conn.execute("DROP TABLE completions")
    before = path.read_bytes()
    with pytest.raises(sqlite3.DatabaseError):
        store.state_many(["session/a"])
    with pytest.raises(sqlite3.DatabaseError):
        store.revision()
    with pytest.raises(sqlite3.DatabaseError):
        store.publish("session/a", str(uuid.uuid4()), "result", "complete")
    assert path.read_bytes() == before


def test_agent_profiles_cannot_alias_session_conversations(tmp_path: Path) -> None:
    assert conversation_identity(tmp_path / "agents" / "same") == "agent/same"
    assert conversation_identity(tmp_path / "sessions" / "same") == "session/same"


def test_delivery_claim_is_exactly_once_across_racing_processes(tmp_path: Path) -> None:
    """Eleven frontends see one completion; exactly one of them may announce it.

    The whole reason `deliveries` exists. Every running TUI polls attention for
    every session, so without arbitration the operator's eleven sessions would
    each fire a banner for one background completion.
    """
    path = tmp_path / "attention.db"
    store = AttentionStore(path)
    token = str(uuid.uuid4())
    store.publish("session/a", token, "result", "complete")
    # A separate AttentionStore per worker, which is what a separate PROCESS
    # gets: no shared connection, no shared cache, only the file.
    with ThreadPoolExecutor(max_workers=11) as pool:
        claims = list(
            pool.map(
                lambda _: AttentionStore(path).claim_delivery("session/a", token, "test"),
                range(11),
            )
        )
    assert claims.count(True) == 1
    # An observer starting later inherits the watermark rather than re-firing.
    assert AttentionStore(path).claim_delivery("session/a", token, "test") is False


def test_delivering_never_acknowledges(tmp_path: Path) -> None:
    """Routing a notification must not mark a session read (SESSION_SIDEBAR.md)."""
    path = tmp_path / "attention.db"
    store = AttentionStore(path)
    token = str(uuid.uuid4())
    store.publish("session/a", token, "result", "complete")
    before = store.state("session/a")
    assert store.claim_delivery("session/a", token, "test") is True
    after = store.state("session/a")
    assert after["unseen"] is True
    assert after["revision"] == before["revision"]
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM receipts").fetchone()[0] == 0


def test_a_newer_completion_is_claimable_after_an_older_one_was_delivered(
    tmp_path: Path,
) -> None:
    """The watermark is per-sequence, not per-conversation: new work still notifies."""
    path = tmp_path / "attention.db"
    store = AttentionStore(path)
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    store.publish("session/a", first, "one", "complete")
    assert store.claim_delivery("session/a", first, "test") is True
    assert store.claim_delivery("session/a", first, "test") is False
    store.publish("session/a", second, "two", "complete")
    assert store.claim_delivery("session/a", second, "test") is True


def test_an_unknown_token_is_never_invented_into_the_watermark(tmp_path: Path) -> None:
    path = tmp_path / "attention.db"
    store = AttentionStore(path)
    store.publish("session/a", str(uuid.uuid4()), "result", "complete")
    assert store.claim_delivery("session/a", str(uuid.uuid4()), "test") is False
    assert store.claim_delivery("session/missing", str(uuid.uuid4()), "test") is False
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM deliveries").fetchone()[0] == 0


def test_claiming_does_not_create_the_store(tmp_path: Path) -> None:
    """Arbitration is a read of published work; it cannot be what publishes it."""
    path = tmp_path / "attention.db"
    assert AttentionStore(path).claim_delivery("session/a", str(uuid.uuid4()), "test") is False
    assert not path.exists()


def test_a_released_claim_reopens_exactly_that_event(tmp_path: Path) -> None:
    """A backend that delivered nothing hands the claim back, not a silent hole."""
    path = tmp_path / "attention.db"
    store = AttentionStore(path)
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    store.publish("session/a", first, "one", "complete")
    store.publish("session/a", second, "two", "complete")
    assert store.claim_delivery("session/a", second, "test") is True
    assert store.release_delivery("session/a", second) is True
    # Re-claimable, because the rollback returned the watermark to the state
    # before this claim...
    assert store.claim_delivery("session/a", second, "test") is True
    # ...and the OLDER completion stays delivered rather than being re-armed.
    assert store.claim_delivery("session/a", first, "test") is False
    # A stale release cannot clobber a newer claim by another observer.
    assert store.release_delivery("session/a", first) is False


def test_upgrading_an_established_store_baselines_instead_of_flooding(tmp_path: Path) -> None:
    """The no-flood rule, at the exact seam an upgrade crosses.

    A database written before `deliveries` existed carries a backlog of unseen
    completions. The first observer to open it must inherit them as already
    delivered, or every one of them fires a banner at once.
    """
    path = tmp_path / "attention.db"
    store = AttentionStore(path)
    tokens = [str(uuid.uuid4()) for _ in range(8)]
    for index, token in enumerate(tokens):
        store.publish(f"session/{index}", token, f"result-{index}", "complete")
    # Reproduce a pre-feature database exactly: the table simply is not there.
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE deliveries")
    assert all(AttentionStore(path).state(f"session/{i}")["unseen"] for i in range(8))
    upgraded = AttentionStore(path)
    # Zero claims on first contact — the point of the whole test.
    assert not any(
        upgraded.claim_delivery(f"session/{index}", token, "test")
        for index, token in enumerate(tokens)
    )
    # The backlog is still UNREAD; only its notifications are considered spent.
    assert all(upgraded.state(f"session/{i}")["unseen"] for i in range(8))
    # And work published after the upgrade still notifies.
    fresh = str(uuid.uuid4())
    upgraded.publish("session/0", fresh, "new", "complete")
    assert upgraded.claim_delivery("session/0", fresh, "test") is True


def test_a_missing_deliveries_table_is_not_read_as_corruption(tmp_path: Path) -> None:
    """The migration hazard, pinned: an old database must stay fully usable.

    `_uninitialized` treats missing tables in an established database as
    corruption. Adding `deliveries` to that probe would make every database
    written by an earlier release read as corrupt, so the probe deliberately
    does not mention it — this test fails the moment someone "tidies" it.
    """
    path = tmp_path / "attention.db"
    store = AttentionStore(path)
    token = str(uuid.uuid4())
    store.publish("session/a", token, "result", "complete")
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE deliveries")
    reopened = AttentionStore(path)
    assert reopened.state("session/a")["unseen"] is True
    assert reopened.revision()[0] > 0
    assert reopened.acknowledge("session/a", token)["unseen"] is False
    # The two ORIGINAL tables are still corruption when absent.
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE receipts")
    with pytest.raises(sqlite3.DatabaseError):
        AttentionStore(path).publish("session/a", str(uuid.uuid4()), "result", "complete")

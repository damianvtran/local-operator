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


def test_agent_profiles_cannot_alias_session_conversations(tmp_path: Path) -> None:
    assert conversation_identity(tmp_path / "agents" / "same") == "agent/same"
    assert conversation_identity(tmp_path / "sessions" / "same") == "session/same"

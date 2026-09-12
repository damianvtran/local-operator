"""Receipt membership, cross-process durability and passive-read invariants."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.attention import AttentionStore, conversation_identity
from local_operator.session.transcript import Transcript


def _provisional_anchor(token: str) -> str:
    """The provisional shape, spelled out rather than imported from the product.

    DELIBERATE DUPLICATION. Importing `provisional_anchor` here made these tests
    fail against pre-fix code with `ImportError` — which proves only that the
    symbol is new, not that the behaviour changed, and would fail identically
    against a tree where supersession was implemented WRONGLY (QA round 1, Q2).
    Writing the literal makes every assertion below discriminate on behaviour.
    `test_the_provisional_shape_is_what_the_product_actually_writes` pins this
    against the product's own writer so the duplication cannot drift silently.
    """
    return f"completion-{token}"


def test_the_provisional_shape_is_what_the_product_actually_writes() -> None:
    """Anchor the test-local literal to the product, so it cannot rot.

    This is the ONE test allowed to import the helper, and it is a shape
    assertion rather than a behaviour one: if the product ever changes the
    provisional anchor format, this fails loudly instead of letting the
    duplicated literal above quietly stop matching anything.
    """
    from local_operator.session.attention import provisional_anchor

    token = str(uuid.uuid4())
    assert provisional_anchor(token) == _provisional_anchor(token)


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
    # An in-flight run whose owner is GONE and whose record left no evidence is
    # an ERROR naming the uncertainty, not an interruption: the taxonomy's
    # default flipped because a cut-off we cannot explain is not a stop. This
    # assertion read ``interrupted`` before the cut-off work, and the cause
    # assertion read ``runtime-killed`` before a later round made the
    # no-evidence branch stop naming a mechanism it cannot observe.
    crashed = store.state("session/owner")
    assert crashed["kind"] == "error"
    assert crashed["cause"] == ""
    assert "the cause could not be determined" in crashed["reason"]
    assert crashed["unseen"]
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


@pytest.mark.asyncio
async def test_a_completed_turn_supersedes_its_own_interrupted_marker(tmp_path: Path) -> None:
    """The permanent-brick regression: a session that could never be opened again.

    The operator lost a 65 MB, 14,905-entry conversation to this. A turn writes
    `attention_started` with token T; anything that bootstraps while that turn
    is still in flight publishes the provisional `(completion-T, interrupted)`
    marker. The turn then finishes and journals the SAME T with its real anchor
    and kind `complete`, so the next open published a second outcome for T,
    `publish` called it a conflict, and the raise propagated out of
    `Session.__init__` — killing the runtime process on EVERY spawn attempt.
    Not a timeout and not corruption: the normal turn lifecycle, unopenable
    forever. A long-running session is the one that gets hit, because it has
    the most turns and the widest window to be observed mid-turn.
    """
    from local_operator.harness.types import Message, TextContent
    from local_operator.session.attention import (
        ATTENTION_CUSTOM_TYPE,
        bootstrap_transcript,
    )
    from local_operator.session.transcript import Transcript

    directory = tmp_path / "sessions" / "long-running"
    transcript = Transcript(directory)
    identity = conversation_identity(directory)
    token = str(uuid.uuid4())
    await transcript.append_custom(
        "attention_started", {"conversation_id": identity, "token": token}
    )
    await transcript.append_message(
        Message(role="assistant", content=[TextContent(text="the finished answer")])
    )
    anchor = transcript.entries()[-1].id
    await transcript.append_custom(
        ATTENTION_CUSTOM_TYPE,
        {"conversation_id": identity, "token": token, "anchor": anchor, "kind": "complete"},
    )

    store = AttentionStore(tmp_path / "attention.db")
    # What a bootstrap observing the turn in flight leaves behind.
    store.publish(identity, token, _provisional_anchor(token), "interrupted")
    assert store.state(identity)["kind"] == "interrupted"

    # SELF-HEALING: an already-poisoned store reconciles on next load, with no
    # manual SQL. Users bricked by an older release are repaired by opening.
    bootstrap_transcript(transcript, AttentionStore(store.path))
    state = store.state(identity)
    assert state["kind"] == "complete"
    assert state["anchor_id"] == anchor
    assert state["completion_token"] == token
    # Corrected in place, never minted: a second row would re-fire a
    # notification for a result the human may already have read.
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT count(*) FROM completions").fetchone()[0] == 1
    # Still idempotent once healed.
    bootstrap_transcript(transcript, AttentionStore(store.path))
    assert store.state(identity)["anchor_id"] == anchor


def test_supersession_never_relaxes_the_cross_conversation_refusal(tmp_path: Path) -> None:
    """The integrity property the conflict check exists for, unchanged.

    A forked transcript carries its parent's journal, so a token presented
    under a DIFFERENT conversation must stay an error however provisional the
    stored record looks. Nor may an authoritative record be dragged back to a
    synthetic anchor by a late bootstrap racing a finished turn.
    """
    store = AttentionStore(tmp_path / "attention.db")
    token = str(uuid.uuid4())
    store.publish("session/owner", token, _provisional_anchor(token), "interrupted")
    # Same provisional shape, different conversation: still a hard error.
    with pytest.raises(ValueError):
        store.publish("session/fork", token, "message-1", "complete")
    assert store.state("session/fork")["completion_token"] is None
    assert store.state("session/owner")["kind"] == "interrupted"

    store.publish("session/owner", token, "message-1", "complete")
    # No downgrade: a real outcome is not replaced by a provisional one.
    with pytest.raises(ValueError):
        store.publish("session/owner", token, _provisional_anchor(token), "interrupted")
    assert store.state("session/owner")["anchor_id"] == "message-1"
    # Nor by a different real outcome under the same token.
    with pytest.raises(ValueError):
        store.publish("session/owner", token, "message-2", "complete")


def test_superseding_does_not_resurrect_an_acknowledged_turn_as_unread(
    tmp_path: Path,
) -> None:
    """Correcting a record is not a new completion.

    `sequence` is the receipt watermark, so healing must update in place. A new
    row would sort above the acknowledgement and re-announce a result the human
    has already read — the exact flood `_BASELINE_DELIVERIES` exists to prevent.
    """
    store = AttentionStore(tmp_path / "attention.db")
    token = str(uuid.uuid4())
    store.publish("session/a", token, _provisional_anchor(token), "interrupted")
    before = store.state("session/a")["revision"][0]
    store.acknowledge("session/a", token)
    assert store.state("session/a")["unseen"] is False
    # Somebody already announced this run, per the no-flood rule.
    assert store.claim_delivery("session/a", token, "test") is True

    store.publish("session/a", token, "message-1", "complete")
    healed = store.state("session/a")
    assert healed["kind"] == "complete"
    assert healed["revision"][0] == before
    assert healed["unseen"] is False
    # And the delivery watermark is likewise not rewound into a second toast.
    assert store.claim_delivery("session/a", token, "test") is False


@pytest.mark.asyncio
async def test_a_failing_attention_bootstrap_cannot_stop_a_session_from_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attention bookkeeping is an observability nicety; boot outranks it.

    The severity of the brick came from `Session.__init__` calling the
    bootstrap unguarded: a raise there kills the runtime process before it can
    serve, on every attach, so no retry ever helps. Whatever else goes wrong in
    attention, the conversation must still open.
    """
    import local_operator.session.attention as attention_module
    from tests.unit.session.test_session import ScriptedStream, make_session

    calls: list[str] = []

    def exploding(transcript, store=None):  # type: ignore[no-untyped-def]
        calls.append(transcript.directory.name)
        raise RuntimeError("attention store is unreachable")

    monkeypatch.setattr(attention_module, "bootstrap_transcript", exploding)
    session = make_session(tmp_path, ScriptedStream([]))
    try:
        assert calls, "the guard must not skip the import, only survive it"
        # The session is fully constructed and usable.
        assert session._transcript.directory.name == "sess"
        assert session.session_id
    finally:
        await session.dispose()


def test_bootstrap_swallows_and_logs_a_broken_conversation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One poisoned session may not take the mobile daemon's other 99 with it.

    The daemon sweeps up to 100 session directories in one loop, so the guard
    lives in `bootstrap_transcript` itself rather than only at its callers. The
    failure is logged with the conversation and the exception: swallowed is not
    the same as invisible.
    """
    import logging

    from local_operator.session.attention import bootstrap_transcript

    class Unreadable:
        directory = tmp_path / "sessions" / "broken"

        def latest_custom(self, _type: str) -> dict[str, object]:
            raise OSError("transcript is unreadable")

    with caplog.at_level(logging.WARNING, logger="local_operator.session.attention"):
        bootstrap_transcript(Unreadable(), AttentionStore(tmp_path / "attention.db"))
    assert "broken" in caplog.text
    assert "transcript is unreadable" in caplog.text


@pytest.mark.asyncio
async def test_a_broken_attention_import_cannot_stop_a_session_from_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The path the OUTER guard uniquely exists for: the import itself failing.

    `test_a_failing_attention_bootstrap_cannot_stop_a_session_from_loading`
    monkeypatches `bootstrap_transcript`, so it exercises a failure the INNER
    guard in `attention.py` would have absorbed anyway — it proves the pair
    works, not that the outer half is load-bearing (review round 1, minor-4).
    The outer guard's own coverage is the `from ... import bootstrap_transcript`
    STATEMENT raising: a circular import, a half-installed package, a broken
    .pyc. There is no inner guard to fall back on there, because the module
    holding it never loads, and an unguarded ImportError would brick boot
    exactly as the original defect did.

    Simulated by making the import machinery itself raise for that one module,
    which is what a genuinely broken install looks like from inside `__init__`.
    """
    import builtins

    from tests.unit.session.test_session import ScriptedStream, make_session

    real_import = builtins.__import__
    attempted: list[str] = []

    def broken_import(  # type: ignore[no-untyped-def]
        name, globals=None, locals=None, fromlist=(), level=0
    ):
        if name == "local_operator.session.attention" and "bootstrap_transcript" in (
            fromlist or ()
        ):
            attempted.append(name)
            raise ImportError("cannot import name 'bootstrap_transcript'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    session = make_session(tmp_path, ScriptedStream([]))
    monkeypatch.setattr(builtins, "__import__", real_import)
    try:
        assert attempted, "the import must actually have been attempted and failed"
        # Boot survived an ImportError with no inner guard available.
        assert session._transcript.directory.name == "sess"
        assert session.session_id
    finally:
        await session.dispose()


def test_a_heal_moves_the_change_detector_without_moving_the_watermark(
    tmp_path: Path,
) -> None:
    """A supersede must be DETECTABLE, or the phone keeps serving the stale row.

    `revision()` is the store's cheap change detector and two consumers gate on
    it: the desktop bridge skips `refresh_attention` when it is unchanged, and
    the TUI's background notifier skips its catalog scan. An in-place UPDATE
    moves neither `MAX(sequence)` nor `SUM(acknowledged)`, so a healed record
    reached neither of them — a phone could show "Interrupted" for a turn that
    completed successfully until some unrelated session happened to publish
    (review round 1, major-1).

    The two properties are in tension only if they share a counter, so they do
    not: `sequence` stays pinned (no resurrected unread, no re-fired toast) and
    a third term moves. This test pins BOTH halves — the fix is wrong if either
    the revision fails to move or the watermark does move.
    """
    store = AttentionStore(tmp_path / "attention.db")
    token = str(uuid.uuid4())
    store.publish("session/a", token, _provisional_anchor(token), "interrupted")
    store.acknowledge("session/a", token)
    # Announced already, so a re-claim after the heal would be a SECOND toast
    # for one turn — the flood the pinned sequence exists to prevent.
    assert store.claim_delivery("session/a", token, "test") is True
    before = store.revision()
    sequence_before = store.state("session/a")["revision"][0]

    store.publish("session/a", token, "message-real", "complete")

    after = store.revision()
    assert after != before, "a heal must be visible to the change detector"
    # The watermark terms specifically must NOT be what moved.
    assert after[0] == before[0]
    assert after[1] == before[1]
    assert store.state("session/a")["revision"][0] == sequence_before
    assert store.state("session/a")["unseen"] is False
    assert store.claim_delivery("session/a", token, "test") is False
    # Read cross-process, since the consumers gating on this are other processes.
    assert AttentionStore(store.path).revision() == after


def test_a_mismatched_provisional_anchor_cannot_supersede(tmp_path: Path) -> None:
    """An incoming anchor belonging to ANOTHER token is not a heal.

    `_supersedes_provisional` checked the STORED anchor's shape but nothing
    checked the incoming one, so a publish for T carrying `completion-<U>`
    superseded successfully and left a row wearing an anchor matching no token:
    unviewable, and permanently unhealable, because no later real outcome could
    supersede a record whose stored anchor is no longer `completion-<T>` — a
    one-way trip into a dead state (review round 1, minor-2).
    """
    store = AttentionStore(tmp_path / "attention.db")
    token, other = str(uuid.uuid4()), str(uuid.uuid4())
    store.publish("session/a", token, _provisional_anchor(token), "interrupted")

    with pytest.raises(ValueError):
        store.publish("session/a", token, _provisional_anchor(other), "complete")

    # Refused, and — the point of the finding — still healable afterwards.
    with sqlite3.connect(store.path) as conn:
        stored = conn.execute("SELECT anchor FROM completions WHERE token=?", (token,)).fetchone()
    assert stored[0] == _provisional_anchor(token)
    store.publish("session/a", token, "message-real", "complete")
    assert store.state("session/a")["kind"] == "complete"
    assert store.state("session/a")["anchor_id"] == "message-real"


def test_the_desktop_is_just_another_claimant_with_no_special_path(tmp_path: Path) -> None:
    """A desktop claim and a TUI claim for one completion produce ONE winner.

    The desktop app reaches this primitive through `DesktopSessions.
    claim_notification`, which passes `backend="desktop"`. The `backend` column
    is diagnostics only — no decision may read it, because a claim that
    consulted anything beyond the monotonic sequence would stop being
    clock-free and two observers with disagreeing clocks would both deliver —
    so the important property is that naming a new backend buys no privilege
    whatsoever.

    That matters because the two surfaces are genuinely concurrent in the real
    configuration this feature ships into: with the desktop window unfocused, a
    session's `live_state` is `idle`, which makes BOTH a TUI observer and the
    desktop eligible for the same completion. Exactly one banner is the
    contract; which one wins is deliberately unspecified.
    """
    path = tmp_path / "attention.db"
    store = AttentionStore(path)
    token = str(uuid.uuid4())
    store.publish("session/a", token, "result", "complete")

    # Separate stores, as two separate PROCESSES would hold: no shared
    # connection, no shared cache, only the file.
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(
            pool.map(
                lambda backend: AttentionStore(path).claim_delivery("session/a", token, backend),
                ("desktop", "cmux"),
            )
        )
    assert claims.count(True) == 1, "two surfaces both announced one completion"

    # Whoever lost stays locked out, in either order, and a late third surface
    # inherits the watermark rather than re-firing.
    assert AttentionStore(path).claim_delivery("session/a", token, "desktop") is False
    assert AttentionStore(path).claim_delivery("session/a", token, "detached") is False

    # And the claim did not touch the READ watermark on any of those paths:
    # the sidebar's mark belongs to a human opening the conversation.
    assert store.state("session/a")["unseen"] is True

    # A NEWER completion is a fresh arbitration, so a delivered conversation is
    # not permanently silenced on either surface.
    newer = str(uuid.uuid4())
    store.publish("session/a", newer, "result-2", "complete")
    assert AttentionStore(path).claim_delivery("session/a", newer, "desktop") is True
    assert AttentionStore(path).claim_delivery("session/a", newer, "cmux") is False


# -- the cut-off taxonomy's restore path (design §3.4/§8.1 tests 2-6) ---------


def _record_json(session_id: str, pid: int) -> dict[str, Any]:
    """One discovery record naming ``session_id``, in the shape the runtime writes."""
    return {
        "pid": pid,
        "kind": "daemon",
        "session_id": session_id,
        "conversation_name": session_id,
        "cwd": "/tmp",
        "model_label": "test/mock",
        "control_port": 1,
        "control_key": "k",
        "version": "1.2.3",
        "source_ref": "abcdef0",
    }


def _seed_started(root: Path, session_id: str) -> str:
    """An ``attention_started`` with no outcome, as a killed runtime leaves it."""
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    token = str(uuid.uuid4())
    transcript = Transcript(directory)
    asyncio.run(
        transcript.append_custom(
            "attention_started", {"conversation_id": f"session/{session_id}", "token": token}
        )
    )
    return token


def _write_record(root: Path, session_id: str, pid: int) -> None:
    run = root / "run" / "mobile"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"{pid}.json").write_text(json.dumps(_record_json(session_id, pid)), encoding="utf-8")


def test_an_orphaned_run_without_evidence_is_an_error_with_a_named_uncertainty(
    tmp_path: Path,
) -> None:
    """Design test 1: no live owner, no record, no stop marker.

    The kind is still ``error`` — a cut-off nobody can explain is not a stop —
    but the CAUSE is left empty and the sentence is the honest one. It used to
    report ``runtime-killed`` with ``(the cause could not be determined)``
    appended, which every surface that trims a parenthetical (the sidebar's
    ``_error_label``) turned into a positive claim about a mechanism nobody
    observed (design/UX review round 1, D3/U3).
    """
    from local_operator.incidents import CUT_OFF_UNKNOWN
    from local_operator.session.attention import bootstrap_transcript
    from local_operator.session.transcript import Transcript

    session_id = "orphan1"
    token = _seed_started(tmp_path, session_id)
    store = AttentionStore(tmp_path / "attention.db")
    result = bootstrap_transcript(Transcript(tmp_path / "sessions" / session_id), store)
    assert result is not None
    kind, cause, reason, returned = result
    assert (kind, cause, returned) == ("error", "", token)
    assert reason == CUT_OFF_UNKNOWN
    state = store.state(f"session/{session_id}")
    assert state["kind"] == "error"
    assert state["cause"] == ""
    assert state["reason"] == CUT_OFF_UNKNOWN
    assert state["unseen"] is True
    # The sentence and the token must agree with each other, since the inverse
    # (`cause_from_reason`) is what the live surfaces use to name a cause.
    from local_operator.incidents import cause_from_reason

    assert cause_from_reason(reason) == ""


def test_a_live_owner_stops_the_import_from_publishing_anything(tmp_path: Path) -> None:
    """Design test 2 — the false-alarm guard, and the load-bearing part of this PR.

    A daemon sweep (or a resume) that runs mid-turn must publish NOTHING: a
    provisional ``error`` written for a healthy run would be believed by every
    surface and hidden by the busy-suppression, which is the worst combination.
    A FOREIGN live pid is the case; our own must NOT count, or a successor
    runtime could never classify its predecessor's orphaned run.
    """
    import subprocess
    import sys

    from local_operator.session.attention import bootstrap_transcript
    from local_operator.session.transcript import Transcript

    session_id = "live1"
    _seed_started(tmp_path, session_id)
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _write_record(tmp_path, session_id, foreign.pid)
        store = AttentionStore(tmp_path / "attention.db")
        result = bootstrap_transcript(Transcript(tmp_path / "sessions" / session_id), store)
        assert result is None
        assert store.state(f"session/{session_id}")["kind"] is None
    finally:
        foreign.kill()
        foreign.wait(timeout=10)


def test_our_own_record_does_not_block_the_classification(tmp_path: Path) -> None:
    """A successor publishes its own record BEFORE it builds its Session."""
    from local_operator.session.attention import bootstrap_transcript
    from local_operator.session.transcript import Transcript

    session_id = "successor1"
    _seed_started(tmp_path, session_id)
    _write_record(tmp_path, session_id, os.getpid())
    store = AttentionStore(tmp_path / "attention.db")
    result = bootstrap_transcript(Transcript(tmp_path / "sessions" / session_id), store)
    assert result is not None, "a runtime must be able to classify its predecessor's run"


def test_a_dead_record_names_the_runtime_that_died(tmp_path: Path) -> None:
    """Design test 3: the record's build, pid and start time ride the detail."""
    from local_operator.session.attention import bootstrap_transcript
    from local_operator.session.transcript import Transcript

    session_id = "dead1"
    _seed_started(tmp_path, session_id)
    dead_pid = 2**22 + 7
    _write_record(tmp_path, session_id, dead_pid)
    store = AttentionStore(tmp_path / "attention.db")
    result = bootstrap_transcript(Transcript(tmp_path / "sessions" / session_id), store)
    assert result is not None
    kind, cause, reason, _token = result
    assert (kind, cause) == ("error", "runtime-killed")
    assert f"pid {dead_pid}" in reason
    assert "1.2.3@abcdef0" in reason


def test_a_recorded_stop_marker_keeps_an_interruption(tmp_path: Path) -> None:
    """Design test 4: positive evidence of a deliberate act, honoured after death."""
    from local_operator.session.attention import bootstrap_transcript
    from local_operator.session.transcript import Transcript
    from local_operator.wakes.store import write_entry

    session_id = "stopped1"
    _seed_started(tmp_path, session_id)
    schedule = {"id": "w1", "message": "later", "next_due_at": 1, "created_at": 1}
    write_entry(
        tmp_path,
        session_id,
        cwd=str(tmp_path),
        schedules=[schedule],
        preserve={"stopped_at": 12345},
    )
    store = AttentionStore(tmp_path / "attention.db")
    result = bootstrap_transcript(Transcript(tmp_path / "sessions" / session_id), store)
    assert result is not None
    kind, cause, reason, _token = result
    assert (kind, cause) == ("interrupted", "user-stop")
    assert "stopped by the user" in reason
    assert store.state(f"session/{session_id}")["kind"] == "interrupted"


def test_a_provisional_error_is_still_superseded_by_the_real_completion(
    tmp_path: Path,
) -> None:
    """Design test 5: the anchor, not the kind, is the replaceable signal.

    Requiring ``stored_kind == "interrupted"`` here would brick exactly the
    session the store's own docstring describes: the provisional record is now
    published as an ``error``, and the same token's real ``complete`` must still
    heal it.
    """
    store = AttentionStore(tmp_path / "attention.db")
    token = str(uuid.uuid4())
    store.publish("session/a", token, _provisional_anchor(token), "error", reason="boom")
    assert store.state("session/a")["kind"] == "error"
    store.publish("session/a", token, "the-real-entry", "complete")
    state = store.state("session/a")
    assert state["kind"] == "complete"
    assert state["anchor_id"] == "the-real-entry"


def test_reason_and_cause_round_trip_and_an_old_database_migrates(
    tmp_path: Path,
) -> None:
    """Design test 6: the additive columns, plus the pre-change schema."""
    path = tmp_path / "attention.db"
    store = AttentionStore(path)
    token = str(uuid.uuid4())
    store.publish(
        "session/a",
        token,
        "anchor",
        "error",
        reason="the runtime went away",
        cause="runtime-killed",
    )
    state = store.state("session/a")
    assert state["reason"] == "the runtime went away"
    assert state["cause"] == "runtime-killed"
    assert store.state_many(["session/a"])["session/a"]["reason"] == "the runtime went away"

    # An OLD database: the pre-taxonomy schema, with a row already in it.
    old = tmp_path / "old.db"
    with sqlite3.connect(old) as conn:
        conn.execute(
            "CREATE TABLE completions (sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
            "conversation TEXT NOT NULL, token TEXT NOT NULL UNIQUE, anchor TEXT NOT NULL, "
            "kind TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE receipts (conversation TEXT PRIMARY KEY, acknowledged INTEGER NOT NULL)"
        )
        old_token = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO completions(conversation,token,anchor,kind) VALUES(?,?,?,?)",
            ("session/old", old_token, "anchor", "interrupted"),
        )
    migrated = AttentionStore(old)
    old_state = migrated.state("session/old")
    assert old_state["kind"] == "interrupted"
    assert old_state["reason"] == ""
    assert old_state["cause"] == ""
    # A READ did not migrate it: the read-only path must never alter a schema.
    with sqlite3.connect(old) as conn:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(completions)")}
    assert "reason" not in columns
    # A WRITE migrates it, additively, and the new fields then round-trip.
    new_token = str(uuid.uuid4())
    migrated.publish("session/old", new_token, "anchor2", "error", reason="r", cause="c")
    assert migrated.state("session/old")["cause"] == "c"
    with sqlite3.connect(old) as conn:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(completions)")}
    assert {"reason", "cause"} <= columns

"""The deferred cold facade: metadata-only cold state, and an unread message list.

WHAT THIS PINS, and why each half is a separate risk.

A cold desktop facade dominates the cost of opening a conversation: the replay
walks the journal, materialises every ``Message`` and resolves attachments before
the first frame, and the frame itself costs 3-8 ms. The four values that frame
actually needs from that read — the durable checkpoint, the spend receipt, their
order and the usage seed — are written by ``build_cold_replay`` into a durable
record (:mod:`local_operator.session.cold_metadata`), so a later process can
restore the SAME first frontend state without reading the journal at all. The
message list is the one output that record deliberately does not carry, and it
is DEFERRED rather than faked: ``_DeferredHistory`` makes every accessor that
needs messages raise until the explicit hydration runs.

The splits this file protects:

* FIRST OPEN IS STILL EAGER AND EXACT. A journal with no record — every journal
  the first time this build sees it — takes today's read, and that read is what
  writes the record. Nothing here changes what a first open costs.
* THE HIT IS THE SAME FIRST FRAME, not merely the same title: the whole
  frontend state is compared field for field against the eager build's.
* A MISS IS A FALLBACK, NOT A FAILURE: a changed, truncated, replaced, corrupt,
  unknown-schema, other-request or other-journal record takes the eager path and
  produces the identical payload.
* THE DEFERRAL IS EXPLICIT AND OPT-IN. ``cold()`` without the flag stays eager
  even when a perfectly good record is on disk — the TUI, peer viewers, mobile,
  resume and new sessions construct their facades that way and every one of them
  paints or folds ``history()``.
* NOTHING OBSERVES A HALF-HYDRATED SESSION: concurrent hydrations share one read,
  and a seed or a close during deferral leaves no message list to be installed
  behind it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator.harness.types import Message, MessageRole, TextContent, Usage
from local_operator.session import cold_metadata, compression, replay_cache
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import (
    FRONTEND_CHECKPOINT_CUSTOM_TYPE,
    FrontendModelSpec,
    FrontendSessionState,
    JobState,
    TodoItemState,
    TodoPhaseState,
)
from local_operator.session.replay_cache import replay_key
from local_operator.session.spend import SESSION_SPEND_CUSTOM_TYPE
from local_operator.session.transcript import Transcript

SESSION_ID = "abcdef012345"


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A store, a HOME and an empty replay cache for every test."""
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "config"
    config.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    replay_cache.replay_cache().clear()
    yield config
    replay_cache.replay_cache().clear()


async def _never_take_over() -> None:
    raise AssertionError("a cold facade never takes over")


def _session_dir(config: Path, session_id: str = SESSION_ID) -> Path:
    directory = config / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _message(id_: str, role: MessageRole, text: str) -> Message:
    return Message(id=id_, role=role, content=[TextContent(text=text)])


async def seed_journal(
    config: Path,
    *,
    turns: int = 4,
    checkpoint: bool = True,
    spend: bool = True,
    session_id: str = SESSION_ID,
) -> Path:
    """A journal shaped like a real resumed conversation: turns, receipts, a row.

    The checkpoint carries the roster, todos, title and the money, because those
    are the four families the metadata record exists to restore; the spend rows
    give ``_seed_cold_usage`` something to order against the checkpoint.
    """
    directory = _session_dir(config, session_id)
    transcript = Transcript(directory)
    if spend:
        for turn in range(turns):
            await transcript.append_custom(
                SESSION_SPEND_CUSTOM_TYPE,
                {
                    "usd": 0.01 * (turn + 1),
                    "calls": turn + 1,
                    "input_tokens": 10,
                    "output_tokens": 5,
                },
            )
    for turn in range(turns):
        await transcript.append_message(_message(f"u{turn}", "user", f"question {turn}"))
        # A receipt on the last turn, so the usage seed is a REAL reading rather
        # than ``None``: the seed is one of the four values the record carries and
        # a fixture without one would never exercise it.
        answer = _message(f"a{turn}", "assistant", f"answer {turn}")
        if turn == turns - 1:
            answer = answer.model_copy(
                update={"usage": Usage(input_tokens=11, output_tokens=22, context_tokens=333)}
            )
        await transcript.append_message(answer)
    if checkpoint:
        await transcript.append_custom(
            FRONTEND_CHECKPOINT_CUSTOM_TYPE,
            {
                "checkpoint_id": "c1",
                "state": FrontendSessionState(
                    session_id=session_id,
                    epoch="previous-owner",
                    conversation_title="Article search rollout",
                    goal="ship the rollout",
                    goal_status="active",
                    jobs=[JobState(id="job1", type="task", label="auditor", status="succeeded")],
                    todos=[
                        TodoPhaseState(
                            name="Verification",
                            items=[TodoItemState(text="run the gate", status="pending")],
                        )
                    ],
                    cumulative_parent_cost=12.5,
                    context_tokens=322_546,
                    context_window=1_000_000,
                    selected_model=FrontendModelSpec(provider="", model_id=""),
                ).model_dump(mode="json"),
            },
        )
    transcript.flush()
    return directory


async def cold_open(
    config: Path, *, deferred: bool, session_id: str = SESSION_ID
) -> AttachedSession:
    return await AttachedSession.cold(
        session_id,
        config_dir=config,
        cwd=str(config),
        takeover_factory=_never_take_over,
        surface="desktop",
        deferred_history=deferred,
    )


def frontend_payload(facade: AttachedSession) -> dict[str, Any]:
    """The WHOLE first frontend state, as the wire carries it.

    Compared whole rather than field by field: the claim is that a hit restores
    the same first frame, and a spot check of the title would pass for a payload
    missing the roster, the todos or the money.
    """
    return facade.frontend_state.model_dump(mode="json")


def _record_path(config: Path, session_id: str = SESSION_ID) -> Path:
    return cold_metadata.cold_metadata_path(config, session_id)


def counting_replay(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Journal bytes read, per ``read_replay_suffix`` call, for this test."""
    calls: list[int] = []
    original = replay_cache.read_replay_suffix

    def counted(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        calls.append(result.bytes_read)
        return result

    monkeypatch.setattr(replay_cache, "read_replay_suffix", counted)
    return calls


# --------------------------------------------------------------------------
# The first open stays eager, and it is the write
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_first_open_of_a_journal_is_eager_and_writes_the_record(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NEW record is a write-behind: the first open pays today's read.

    This is the honest boundary of the change. Nothing makes a journal's
    first-ever open cheap; the record that makes later opens cheap is a
    by-product of that first read.
    """
    await seed_journal(isolated_config)
    assert not _record_path(isolated_config).exists()
    calls = counting_replay(monkeypatch)

    facade = await cold_open(isolated_config, deferred=True)
    try:
        assert facade._deferred_history is None, "no record yet, so nothing to defer"
        assert facade._history_hydrated is True
        assert len(facade.history()) == 8
        assert sum(calls) > 0, "the first open reads the journal"
        assert _record_path(isolated_config).exists()
    finally:
        await facade.dispose()


@pytest.mark.asyncio
async def test_the_record_carries_no_message_list(
    isolated_config: Path,
) -> None:
    """The one thing the record must never grow into is a transcript cache.

    Messages are the expensive half AND the half a stale record could serve
    wrongly, so a record that started carrying them would silently turn a
    metadata reuse into a history cache with no validation for it.
    """
    await seed_journal(isolated_config)
    facade = await cold_open(isolated_config, deferred=False)
    await facade.dispose()

    payload = json.loads(_record_path(isolated_config).read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema",
        "witness",
        "checkpoint",
        "checkpoints",
        "order",
        "seed_usage",
        "bytes_read",
        "selection",
    }
    assert "question 0" not in json.dumps(payload), "no message text may be stored"
    assert payload["witness"]["through_id"] is None


# --------------------------------------------------------------------------
# The hit: the same first frame, without the journal
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_second_process_restores_the_whole_frontend_state_from_the_record(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hit is judged on the WHOLE payload and on the work it did not do."""
    await seed_journal(isolated_config)

    eager = await cold_open(isolated_config, deferred=False)
    try:
        expected = frontend_payload(eager)
        eager_rows = [m.id for m in eager.history()]
    finally:
        await eager.dispose()
    assert _record_path(isolated_config).exists()

    # A NEW PROCESS: the in-memory replay cache is gone, so anything the second
    # open skips it skips because of the durable record alone.
    replay_cache.replay_cache().clear()
    calls = counting_replay(monkeypatch)

    deferred = await cold_open(isolated_config, deferred=True)
    try:
        assert deferred._deferred_history is not None, "a valid record must defer"
        assert frontend_payload(deferred) == expected, "the first frame is the same frame"
        assert deferred._history_ids == set(), "no message list was read"
        assert sum(calls) == 0, "a hit reads no journal bytes"
        # Not an empty conversation: the accessor refuses instead of answering.
        with pytest.raises(RuntimeError, match="deferred"):
            deferred.history()
        await deferred.hydrate_deferred_history()
        assert [m.id for m in deferred.history()] == eager_rows
        assert sum(calls) > 0, "hydration pays exactly the read the open deferred"
    finally:
        await deferred.dispose()


@pytest.mark.asyncio
async def test_a_checkpointless_journal_is_still_a_hit(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record of four EMPTY values is still a valid record.

    A pre-ledger journal has no checkpoint and no receipts, so every value in
    the record is ``None`` — and that is the answer too. Nothing about the open
    is right or wrong by being empty, so the hit is taken and the parse skipped
    exactly as it is for a full record.
    """
    await seed_journal(isolated_config, checkpoint=False, spend=False)
    first = await cold_open(isolated_config, deferred=False)
    try:
        expected = frontend_payload(first)
    finally:
        await first.dispose()

    replay_cache.replay_cache().clear()
    calls = counting_replay(monkeypatch)
    second = await cold_open(isolated_config, deferred=True)
    try:
        assert second._deferred_history is not None
        assert frontend_payload(second) == expected
        assert sum(calls) == 0
    finally:
        await second.dispose()


# --------------------------------------------------------------------------
# Opt-in, and nothing else
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_default_cold_open_stays_eager_with_a_valid_record_present(
    isolated_config: Path,
) -> None:
    """THE INVARIANT EVERY OTHER SURFACE DEPENDS ON.

    The TUI, peer viewers, mobile, ``/resume`` and a new session all build their
    cold facade without the flag, and each of them paints or folds ``history()``
    synchronously. A good record on disk must not change what they get: the
    default is eager, hydrated, and identical.
    """
    await seed_journal(isolated_config)
    warm = await cold_open(isolated_config, deferred=False)
    await warm.dispose()
    assert _record_path(isolated_config).exists()

    facade = await cold_open(isolated_config, deferred=False)
    try:
        assert facade._deferred_history is None
        assert facade._history_hydrated is True
        assert len(facade.history()) == 8
    finally:
        await facade.dispose()


# --------------------------------------------------------------------------
# Misses: every one takes the eager path and produces the same payload
# --------------------------------------------------------------------------


async def _eager_reference(config: Path) -> tuple[dict[str, Any], list[str]]:
    """The eager answer for the journal AS IT IS NOW, with the stale record kept.

    The record is what makes the eager read rewrite itself, so measuring the
    eager answer would destroy the very stale record the deferred open is
    supposed to refuse. It is stashed, the reference is taken without it, and the
    stale bytes are put back — so the two opens being compared read the same
    journal and only the record differs.
    """
    record = _record_path(config)
    stale = record.read_bytes() if record.exists() else None
    record.unlink(missing_ok=True)
    replay_cache.replay_cache().clear()
    facade = await cold_open(config, deferred=False)
    try:
        payload = frontend_payload(facade)
        rows = [m.id for m in facade.history()]
    finally:
        await facade.dispose()
    record.unlink(missing_ok=True)
    if stale is not None:
        record.write_bytes(stale)
    replay_cache.replay_cache().clear()
    return payload, rows


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["append", "truncate", "replace"])
async def test_a_stale_record_for_a_moved_journal_is_a_miss(
    isolated_config: Path, damage: str
) -> None:
    """The journal half of the witness: append, truncation and replacement.

    Each is a different writer shape and each must miss. ``append`` moves size,
    ``truncate`` moves size and destroys the checkpoint row, and ``replace``
    (how a compaction writes) keeps the bytes and the size and changes the
    INODE — the half of the witness a size+mtime check cannot see, which is why
    the record carries the inode too.
    """
    directory = await seed_journal(isolated_config)
    first = await cold_open(isolated_config, deferred=False)
    await first.dispose()
    assert _record_path(isolated_config).exists()
    journal = directory / "transcript.jsonl"

    if damage == "append":
        transcript = Transcript(directory)
        await transcript.append_message(_message("late", "user", "written after the record"))
        transcript.flush()
    elif damage == "truncate":
        body = journal.read_bytes()
        journal.write_bytes(body[: len(body) // 2])
    else:
        staged = journal.with_suffix(".staged")
        staged.write_bytes(journal.read_bytes())
        staged.replace(journal)

    expected_payload, expected_rows = await _eager_reference(isolated_config)
    facade = await cold_open(isolated_config, deferred=True)
    try:
        assert facade._deferred_history is None, f"{damage} must not be served from a record"
        assert frontend_payload(facade) == expected_payload
        assert [m.id for m in facade.history()] == expected_rows
    finally:
        await facade.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage",
    ["unparseable", "unknown_schema", "missing_usage_field", "other_cut", "other_types"],
)
async def test_an_unusable_record_is_a_miss(isolated_config: Path, damage: str) -> None:
    """The record half of the witness: shapes and requests this build cannot use.

    ``missing_usage_field`` is a record from a build whose shape had no usage
    seed. A present-but-null seed is a valid answer (a pre-ledger journal has
    none); a MISSING key is a record this code cannot read, and filling it in
    from the default would silently claim a reading nobody stored.
    """
    await seed_journal(isolated_config)
    first = await cold_open(isolated_config, deferred=False)
    await first.dispose()

    record = _record_path(isolated_config)
    assert record.exists()
    if damage == "unparseable":
        payload = json.loads(record.read_text(encoding="utf-8"))
        record.write_text(json.dumps(payload)[: len(json.dumps(payload)) // 2], encoding="utf-8")
    else:
        payload = json.loads(record.read_text(encoding="utf-8"))
        if damage == "unknown_schema":
            payload["schema"] = 999
        elif damage == "missing_usage_field":
            payload.pop("seed_usage")
        elif damage == "other_cut":
            payload["witness"]["through_id"] = "u1"
        else:
            payload["witness"]["checkpoint_types"] = []
        record.write_text(json.dumps(payload), encoding="utf-8")

    expected_payload, expected_rows = await _eager_reference(isolated_config)
    facade = await cold_open(isolated_config, deferred=True)
    try:
        assert facade._deferred_history is None, f"{damage} must not defer"
        assert frontend_payload(facade) == expected_payload
        assert [m.id for m in facade.history()] == expected_rows
    finally:
        await facade.dispose()


@pytest.mark.asyncio
async def test_a_journal_that_moves_while_the_record_is_read_is_a_miss(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The after-read recheck, on the function that performs it.

    A record is read in its own window. An append landing inside that window
    leaves the record describing a version the caller is no longer opening, so
    the re-stat AFTER the read refuses it. The seam patched here is
    ``_witness_of`` — the step between parsing the record and re-checking the
    journal — which is the only deterministic place to land the append between
    those two reads.
    """
    directory = await seed_journal(isolated_config)
    first = await cold_open(isolated_config, deferred=False)
    await first.dispose()
    assert _record_path(isolated_config).exists()
    expected_payload, expected_rows = await _eager_reference(isolated_config)
    journal = directory / "transcript.jsonl"

    original = cold_metadata._witness_of

    def witness_then_append(raw: Any) -> Any:
        witness = original(raw)
        with journal.open("ab") as handle:
            handle.write(
                b'{"id":"mid","ts":1.0,"type":"message","payload":{"kind":"message",'
                b'"role":"user","content":[{"text":"appended mid-read"}]}}\n'
            )
        return witness

    monkeypatch.setattr(cold_metadata, "_witness_of", witness_then_append)
    replay_cache.replay_cache().clear()
    facade = await cold_open(isolated_config, deferred=True)
    try:
        assert facade._deferred_history is None, "a moved journal must not be served"
        # And the answer is the eager one, which now includes the late row.
        assert [m.id for m in facade.history()][: len(expected_rows)] == expected_rows
        assert facade.history()[-1].id == "mid"
        assert frontend_payload(facade) == expected_payload
    finally:
        await facade.dispose()


@pytest.mark.asyncio
async def test_a_record_whose_usage_no_longer_validates_is_a_miss(
    isolated_config: Path,
) -> None:
    """A stale reading is a miss, not a crash and not a confident zero."""
    await seed_journal(isolated_config)
    first = await cold_open(isolated_config, deferred=False)
    await first.dispose()

    record = _record_path(isolated_config)
    payload = json.loads(record.read_text(encoding="utf-8"))
    # A shape pydantic REFUSES (an int field holding a list), because most of
    # ``Usage``'s fields are optional and a merely-odd payload would validate.
    payload["seed_usage"] = {"input_tokens": [1, 2]}
    record.write_text(json.dumps(payload), encoding="utf-8")

    replay_cache.replay_cache().clear()
    facade = await cold_open(isolated_config, deferred=True)
    try:
        assert facade._deferred_history is None
    finally:
        await facade.dispose()


@pytest.mark.asyncio
async def test_a_record_is_never_written_for_a_reconnect_cut(
    isolated_config: Path,
) -> None:
    """Only the cold-open request shape is worth a slot.

    A record made for a negotiated cut could never serve a cold open, and it
    would displace the one that can — one file per session, atomically replaced.
    """
    await seed_journal(isolated_config)
    facade = await cold_open(isolated_config, deferred=False)
    try:
        before = _record_path(isolated_config).read_bytes()
        # A reconnect-shaped read: a cut, and no wanted checkpoint rows.
        await facade._read_transcript(through_id="u2", strict_cut=True)
        await facade._read_transcript(want_checkpoint=True, through_id="u2")
        assert _record_path(isolated_config).read_bytes() == before
    finally:
        await facade.dispose()


# --------------------------------------------------------------------------
# The deferral itself
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_synchronous_history_accessor_raises_while_deferred(
    isolated_config: Path,
) -> None:
    """An unread list must never answer as an empty one."""
    await seed_journal(isolated_config)
    first = await cold_open(isolated_config, deferred=False)
    await first.dispose()

    facade = await cold_open(isolated_config, deferred=True)
    try:
        assert facade._deferred_history is not None
        accessors: list[Callable[[], Any]] = [
            facade.history,
            facade.display_history_window,
            lambda: facade.history_message_count,
            lambda: facade.history_theme_turn_count,
            lambda: facade.history_opener_text,
            facade.history_last_message,
        ]
        for accessor in accessors:
            with pytest.raises(RuntimeError, match="deferred"):
                accessor()
    finally:
        await facade.dispose()


@pytest.mark.asyncio
async def test_hydration_is_idempotent_and_matches_the_eager_messages(
    isolated_config: Path,
) -> None:
    await seed_journal(isolated_config)
    first = await cold_open(isolated_config, deferred=False)
    try:
        expected = [(m.id, m.role, m.text) for m in first.history()]
        expected_count = first.history_message_count
    finally:
        await first.dispose()

    facade = await cold_open(isolated_config, deferred=True)
    try:
        await facade.hydrate_deferred_history()
        assert facade._deferred_history is None
        assert [(m.id, m.role, m.text) for m in facade.history()] == expected
        assert facade.history_message_count == expected_count
        # Idempotent: a second call must not re-read or duplicate.
        await facade.hydrate_deferred_history()
        assert [(m.id, m.role, m.text) for m in facade.history()] == expected
    finally:
        await facade.dispose()


@pytest.mark.asyncio
async def test_concurrent_hydrations_share_one_journal_read(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two callers, one read: the facade owns the task, not the caller."""
    await seed_journal(isolated_config)
    first = await cold_open(isolated_config, deferred=False)
    await first.dispose()
    replay_cache.replay_cache().clear()
    calls = counting_replay(monkeypatch)

    facade = await cold_open(isolated_config, deferred=True)
    try:
        await asyncio.gather(
            facade.hydrate_deferred_history(),
            facade.hydrate_deferred_history(),
            facade.hydrate_deferred_history(),
        )
        assert len(calls) == 1, f"expected one journal pass, got {len(calls)}"
        assert len(facade.history()) == 8
    finally:
        await facade.dispose()


@pytest.mark.asyncio
async def test_two_deferred_opens_of_one_journal_hydrate_once(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two OPEN facades of one journal version share the replay, not the state.

    The desktop builds a facade per request, so the concurrent case is two
    facades rather than two calls on one — the replay cache's singleflight is
    what makes them one read, and each facade still installs its own list.
    """
    await seed_journal(isolated_config)
    first = await cold_open(isolated_config, deferred=False)
    await first.dispose()
    replay_cache.replay_cache().clear()
    calls = counting_replay(monkeypatch)

    one = await cold_open(isolated_config, deferred=True)
    two = await cold_open(isolated_config, deferred=True)
    try:
        assert one._deferred_history is not None and two._deferred_history is not None
        await asyncio.gather(one.hydrate_deferred_history(), two.hydrate_deferred_history())
        assert len(calls) == 1, f"expected one shared journal pass, got {len(calls)}"
        assert [m.id for m in one.history()] == [m.id for m in two.history()]
    finally:
        await one.dispose()
        await two.dispose()


@pytest.mark.asyncio
async def test_seed_history_invalidates_a_pending_hydration(
    isolated_config: Path,
) -> None:
    """A seed is a statement about what the history IS; it must win.

    The in-flight read is left to finish and DISCARD itself rather than being
    cancelled — a snapshot may legitimately be awaiting the same read — so the
    assertions here are that the seed's rows are what the facade carries, before
    and after the late read lands.
    """
    await seed_journal(isolated_config)
    first = await cold_open(isolated_config, deferred=False)
    await first.dispose()

    facade = await cold_open(isolated_config, deferred=True)
    try:
        pending = asyncio.create_task(facade.hydrate_deferred_history())
        await asyncio.sleep(0)
        seeded = [_message("seed-1", "user", "supplied by the caller")]
        await facade.seed_history(seeded)
        await pending
        assert [m.id for m in facade.history()] == ["seed-1"]
        assert facade._deferred_history is None
    finally:
        await facade.dispose()


@pytest.mark.asyncio
async def test_dispose_during_deferral_installs_nothing(
    isolated_config: Path,
) -> None:
    """A close during a hydration must not resurrect a message list."""
    await seed_journal(isolated_config)
    first = await cold_open(isolated_config, deferred=False)
    await first.dispose()

    facade = await cold_open(isolated_config, deferred=True)
    task = asyncio.create_task(facade.hydrate_deferred_history())
    await asyncio.sleep(0)
    await facade.dispose()
    with pytest.raises((asyncio.CancelledError, RuntimeError)):
        await task
    assert facade._deferred_history is None
    assert facade._history_ids == set()


@pytest.mark.asyncio
async def test_a_cache_write_failure_never_fails_the_open(
    isolated_config: Path,
) -> None:
    """The record is an optimisation; losing it costs the next open, nothing else."""
    await seed_journal(isolated_config)
    # A DIRECTORY where the record file belongs: every write and replace fails.
    blocked = _record_path(isolated_config)
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.mkdir()

    facade = await cold_open(isolated_config, deferred=True)
    try:
        assert facade._deferred_history is None, "nothing was stored, so nothing defers"
        assert len(facade.history()) == 8
    finally:
        await facade.dispose()
    # And the second open behaves the same way rather than raising.
    replay_cache.replay_cache().clear()
    again = await cold_open(isolated_config, deferred=True)
    try:
        assert again._deferred_history is None
        assert len(again.history()) == 8
    finally:
        await again.dispose()


@pytest.mark.asyncio
async def test_a_warm_record_is_a_miss_once_the_journal_is_archived(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record written for the PLAIN journal never describes an archive.

    TWO LAYERS ENFORCE THAT, and this test separates them because only one of them
    was obvious. ``replay_key`` is archive-aware, so a session compressed BEFORE
    the open already presents a different witness and misses on the comparison
    alone. The ``codec is not None`` branch in ``_journal_still_at`` covers the
    narrower window the comparison cannot: the key was taken while the journal was
    plain and the archive landed before the record read's re-stat. That window has
    no other guard, so it is asserted here directly by compressing inside the read.

    Without that branch a stale record would be paired with an archived journal —
    the one hit this record must never serve — and the eager path's answer looks
    identical either way, so nothing else in the suite would notice.
    """
    directory = await seed_journal(isolated_config, turns=3)
    record = _record_path(isolated_config)
    eager = await cold_open(isolated_config, deferred=False)
    try:
        expected_payload = frontend_payload(eager)
        expected_rows = [m.id for m in eager.history()]
    finally:
        await eager.dispose()
    assert record.exists(), "the eager open is the write"
    # The witness the record holds: taken while the journal is still plain.
    plain_key = replay_key(
        directory,
        through_id=None,
        strict_cut=False,
        checkpoint_types=(FRONTEND_CHECKPOINT_CUSTOM_TYPE,),
        opportunistic_types=(SESSION_SPEND_CUSTOM_TYPE,),
    )
    assert plain_key is not None, "the seeded journal exists, so it has a key"
    plain_witness = cold_metadata.witness_for(plain_key)

    # PRECONDITION: on the plain journal this record is a hit.
    replay_cache.replay_cache().clear()
    warm = await cold_open(isolated_config, deferred=True)
    try:
        assert warm._deferred_history is not None, "precondition: a plain journal hits"
    finally:
        await warm.dispose()

    # THE RACE THE BRANCH EXISTS FOR: the archive lands between the key snapshot
    # and the re-stat. ``_witness_of`` is that seam — the step after the record is
    # parsed and before the journal is re-checked.
    original = cold_metadata._witness_of

    def witness_then_archive(raw: Any) -> Any:
        witness = original(raw)
        outcome = compression.compress_session(directory)
        assert outcome.status == "compressed", outcome
        return witness

    monkeypatch.setattr(cold_metadata, "_witness_of", witness_then_archive)
    try:
        assert (
            cold_metadata.read_cold_metadata(record, plain_witness) is None
        ), "a plain-journal record must not be served once the journal is archived"
    finally:
        monkeypatch.setattr(cold_metadata, "_witness_of", original)
    assert not (directory / "transcript.jsonl").exists(), "the session is archived now"

    # END TO END on that archived session: no deferral, the eager read, the same
    # rows — the behaviour a stale hit would have broken.
    replay_cache.replay_cache().clear()
    calls = counting_replay(monkeypatch)
    archived = await cold_open(isolated_config, deferred=True)
    try:
        assert (
            archived._deferred_history is None
        ), "an archived journal must not be served from a plain-journal record"
        assert frontend_payload(archived) == expected_payload
        assert [m.id for m in archived.history()] == expected_rows
        assert sum(calls) > 0, "the miss takes the eager path and replays"
    finally:
        await archived.dispose()

    # RESTORED: a new file, so the OLD record is not revived — a fresh eager
    # replay writes the record for the version that now exists, and the next
    # deferred open hits it again.
    assert compression.restore_for_write(directory) is True
    assert (directory / "transcript.jsonl").exists()
    replay_cache.replay_cache().clear()
    rebuilt = await cold_open(isolated_config, deferred=False)
    try:
        assert [m.id for m in rebuilt.history()] == expected_rows
    finally:
        await rebuilt.dispose()
    replay_cache.replay_cache().clear()
    again = await cold_open(isolated_config, deferred=True)
    try:
        assert again._deferred_history is not None, "the restored version hits its own record"
        await again.hydrate_deferred_history()
        assert [m.id for m in again.history()] == expected_rows
    finally:
        await again.dispose()


# --------------------------------------------------------------------------
# The record's model selection: a HIT skips the selection read, never its answer
# --------------------------------------------------------------------------

#: A direct-provider pair the SHIPPED registry carries metadata for, so the
#: resolution is answered from shipped data and never from a listing fetch.
SELECTED = "deepseek/deepseek-flash"


async def _append_selection(
    config: Path, row: dict[str, Any], kind: str = "selected_model"
) -> None:
    await Transcript(_session_dir(config)).append_custom(kind, row)


def counting_selection_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every journal selection read the COLD MODEL RESOLUTION makes in this test.

    Patched where ``cold_model`` looks it up, so the record writer's own read
    (``replay_cache``) is not counted: the claim is about the open's resolution.
    """
    from local_operator.session import cold_model

    calls: list[Path] = []
    original = cold_model.read_model_selection

    def counted(directory: Path) -> Any:
        calls.append(directory)
        return original(directory)

    monkeypatch.setattr(cold_model, "read_model_selection", counted)
    return calls


def _model(facade: AttachedSession) -> FrontendModelSpec:
    model = facade.frontend_state.selected_model
    assert model is not None, "a cold frame always carries a model"
    return model


async def _deferred_matches_eager(config: Path) -> AttachedSession:
    """Open deferred, assert the first frame equals a fresh eager open's; return it."""
    expected_payload, _rows = await _eager_reference(config)
    replay_cache.replay_cache().clear()
    facade = await cold_open(config, deferred=True)
    assert frontend_payload(facade) == expected_payload
    return facade


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape", ["no_row", "v2_row", "legacy_row", "checkpoint_row", "unusable_v2_row"]
)
async def test_a_hit_resolves_the_model_from_the_record_exactly_as_the_eager_path(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """Every selection shape the fold distinguishes, served from the record.

    The frame must equal the eager build's (model, provider, effort, fast dial,
    and everything else) AND the open must not have read the journal's selection
    at all — that skipped read is the whole point of carrying it.
    """
    await seed_journal(isolated_config)
    if shape == "v2_row":
        await _append_selection(
            isolated_config, {"version": 2, "selector": SELECTED, "effort": "high", "fast": True}
        )
    elif shape == "legacy_row":
        await _append_selection(isolated_config, {"selector": SELECTED, "effort": "low"})
    elif shape == "checkpoint_row":
        state = {"selected_model": {"provider": "deepseek", "model_id": "deepseek-flash"}}
        await _append_selection(isolated_config, {"state": state}, FRONTEND_CHECKPOINT_CUSTOM_TYPE)
    elif shape == "unusable_v2_row":
        await _append_selection(isolated_config, {"version": 2, "selector": "nope"})
    first = await cold_open(isolated_config, deferred=False)
    await first.dispose()
    stored = json.loads(_record_path(isolated_config).read_text(encoding="utf-8"))
    assert stored["selection"] is not None, "the eager open records the selection"
    if shape == "no_row":
        assert stored["selection"]["value"] is None, "no row is PROVED absence, not unrecorded"

    expected_payload, _rows = await _eager_reference(isolated_config)
    if shape in ("v2_row", "legacy_row", "checkpoint_row"):
        assert expected_payload["selected_model"]["model_id"] == "deepseek-flash", "precondition"
    reads = counting_selection_reads(monkeypatch)
    replay_cache.replay_cache().clear()
    facade = await cold_open(isolated_config, deferred=True)
    try:
        assert facade._deferred_history is not None, "precondition: a record HIT"
        assert frontend_payload(facade) == expected_payload
        assert reads == [], "a HIT must not re-read the journal's selection"
        # ``_seed_cold_usage``'s fallback reads the selection through the sink.
        assert facade._deferred_history.selection is not None
    finally:
        await facade.dispose()


@pytest.mark.asyncio
async def test_a_selection_appended_after_the_record_is_never_hidden_by_it(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HIT, then a ``selected_model`` append: the next open MISSES and shows it.

    The record's witness is the journal's stat tuple, and an append moves the
    size, so the stored "no selection" answer cannot outlive the row that
    contradicts it.
    """
    await seed_journal(isolated_config)
    first = await cold_open(isolated_config, deferred=False)
    await first.dispose()
    replay_cache.replay_cache().clear()
    hit = await cold_open(isolated_config, deferred=True)
    try:
        assert hit._deferred_history is not None, "precondition: a record HIT"
        assert _model(hit).model_id != "deepseek-flash"
    finally:
        await hit.dispose()

    await _append_selection(isolated_config, {"version": 2, "selector": SELECTED, "effort": "max"})
    reads = counting_selection_reads(monkeypatch)
    replay_cache.replay_cache().clear()
    after = await cold_open(isolated_config, deferred=True)
    try:
        assert after._deferred_history is None, "the append must turn the record into a MISS"
        model = _model(after)
        assert (model.provider, model.model_id) == ("deepseek", "deepseek-flash")
        assert reads, "the miss resolves the selection from the journal"
    finally:
        await after.dispose()
    # And the rewritten record now carries the NEW answer for the new version.
    stored = json.loads(_record_path(isolated_config).read_text(encoding="utf-8"))
    assert stored["selection"]["value"]["model_id"] == "deepseek-flash"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage",
    [
        "schema_one",
        "missing_selection",
        "wrong_source",
        "bad_value",
        "extra_key",
        "missing_ctime",
        "corrupt_ctime",
        "wrong_ctime",
    ],
)
async def test_an_unusable_selection_falls_back_exactly(isolated_config: Path, damage: str) -> None:
    """A record whose selection or witness this build did not write is a MISS.

    ``schema_one`` is the record an older build wrote: no ``selection`` key and
    schema 1, refused by the schema check before the field is ever looked at.
    ``missing_ctime`` is the unreleased schema-2 shape without the ctime stamp:
    it must MISS rather than hit on the three fields it does carry.
    """
    await seed_journal(isolated_config)
    await _append_selection(isolated_config, {"version": 2, "selector": SELECTED, "effort": "low"})
    first = await cold_open(isolated_config, deferred=False)
    await first.dispose()
    record = _record_path(isolated_config)
    payload = json.loads(record.read_text(encoding="utf-8"))
    if damage == "schema_one":
        payload.pop("selection")
        payload["schema"] = 1
    elif damage == "missing_selection":
        payload.pop("selection")
    elif damage == "wrong_source":
        payload["selection"]["source"][1] += 1
    elif damage == "bad_value":
        payload["selection"]["value"]["authoritative"] = "yes"
    elif damage == "extra_key":
        payload["selection"]["value"]["extra"] = 1
    elif damage == "missing_ctime":
        payload["witness"].pop("ctime_ns")
    elif damage == "corrupt_ctime":
        payload["witness"]["ctime_ns"] = "now"
    else:
        payload["witness"]["ctime_ns"] += 1
    record.write_text(json.dumps(payload), encoding="utf-8")

    expected_payload, _rows = await _eager_reference(isolated_config)
    replay_cache.replay_cache().clear()
    facade = await cold_open(isolated_config, deferred=True)
    try:
        assert facade._deferred_history is None, f"{damage} must not defer"
        assert frontend_payload(facade) == expected_payload
        assert _model(facade).model_id == "deepseek-flash"
    finally:
        await facade.dispose()


@pytest.mark.asyncio
async def test_an_unrecorded_selection_still_hits_and_reads_the_journal(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``selection: null`` is NOT RECORDED, never "no selection": read the journal."""
    await seed_journal(isolated_config)
    await _append_selection(isolated_config, {"version": 2, "selector": SELECTED, "effort": "low"})
    first = await cold_open(isolated_config, deferred=False)
    await first.dispose()
    record = _record_path(isolated_config)
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["selection"] = None
    record.write_text(json.dumps(payload), encoding="utf-8")

    reads = counting_selection_reads(monkeypatch)
    facade = await _deferred_matches_eager(isolated_config)
    try:
        assert facade._deferred_history is not None, "the rest of the record still hits"
        assert _model(facade).model_id == "deepseek-flash"
        assert len(reads) >= 1 and reads[-1].name == SESSION_ID
    finally:
        await facade.dispose()


@pytest.mark.asyncio
async def test_a_reading_under_other_selection_rules_is_re_read(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider registry that moved since the write invalidates the stored answer.

    Simulated by a record whose ``rules`` fingerprint is not this process's: the
    open still hits for the other four values, but the selection comes from the
    journal under THIS build's rules.
    """
    await seed_journal(isolated_config)
    await _append_selection(isolated_config, {"version": 2, "selector": SELECTED, "effort": "low"})
    first = await cold_open(isolated_config, deferred=False)
    await first.dispose()
    record = _record_path(isolated_config)
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["selection"]["rules"] = "0" * 16
    payload["selection"]["value"] = None  # a stale "absent" this build must not trust
    record.write_text(json.dumps(payload), encoding="utf-8")

    reads = counting_selection_reads(monkeypatch)
    facade = await _deferred_matches_eager(isolated_config)
    try:
        assert facade._deferred_history is not None
        assert _model(facade).model_id == "deepseek-flash"
        assert reads, "a foreign-rules reading is re-read from the journal"
    finally:
        await facade.dispose()


def test_a_reading_from_another_journal_version_is_not_recorded(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer files a reading only when its source IS the record's key.

    An append between the replay's key and the writer's selection read leaves
    the reading describing a newer version; filing it would let a HIT on the
    older version serve a selection that version does not have.
    """
    from local_operator.session.model_selection import SelectionReading

    directory = _session_dir(isolated_config)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    key = replay_key(
        directory,
        through_id=None,
        strict_cut=False,
        checkpoint_types=(FRONTEND_CHECKPOINT_CUSTOM_TYPE,),
        opportunistic_types=(SESSION_SPEND_CUSTOM_TYPE,),
    )
    assert key is not None
    monkeypatch.setattr(
        replay_cache,
        "read_selection_reading",
        lambda _d: SelectionReading(
            None, "x", (key.inode, key.size + 1, key.mtime_ns, key.ctime_ns)
        ),
    )
    path = _record_path(isolated_config)
    replay_cache._record_cold_metadata(path, key, {})
    assert json.loads(path.read_text(encoding="utf-8"))["selection"] is None

    monkeypatch.setattr(
        replay_cache,
        "read_selection_reading",
        lambda _d: SelectionReading(None, "x", (key.inode, key.size, key.mtime_ns, key.ctime_ns)),
    )
    replay_cache._record_cold_metadata(path, key, {})
    assert json.loads(path.read_text(encoding="utf-8"))["selection"]["value"] is None


def _rewrite_in_place_keeping_mtime(journal: Path, old: bytes, new: bytes) -> None:
    """Rewrite ``old`` to ``new`` (same length) inside ``journal``, mtime restored.

    The shape a stat-tuple witness without ctime cannot see: same inode, same
    size, same mtime — only the bytes and the inode change time moved. It is
    what ``Transcript._write_entries``' ``preserve_mtime`` makes possible for its
    own bookkeeping, done here to a selection row to prove the stale-model case.
    """
    import os
    import time

    assert len(old) == len(new), "the rewrite must keep the size"
    before = journal.stat()
    data = journal.read_bytes()
    assert data.count(old) == 1, "the fixture must name the row exactly once"
    # A ctime that ticks within the clock's resolution would make this test
    # depend on luck; let it advance first.
    time.sleep(0.01)
    with journal.open("r+b") as handle:
        handle.seek(data.index(old))
        handle.write(new)
    os.utime(journal, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = journal.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ), "precondition: the three pre-ctime witness fields are unchanged"
    assert after.st_ctime_ns != before.st_ctime_ns, "precondition: ctime moved"


@pytest.mark.asyncio
async def test_a_same_size_rewrite_with_mtime_restored_is_a_miss(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review's repro: HIT, interior rewrite of the selection, mtime restored.

    Without ctime in the witness the next open HIT and served the OLD model from
    the record. With it, the open MISSES and resolves the rewritten selection —
    exactly what an eager open of the same bytes resolves.
    """
    directory = await seed_journal(isolated_config)
    # Two effort spellings of equal length, so the rewrite keeps the size.
    await _append_selection(isolated_config, {"version": 2, "selector": SELECTED, "effort": "low"})
    first = await cold_open(isolated_config, deferred=False)
    await first.dispose()
    replay_cache.replay_cache().clear()
    hit = await cold_open(isolated_config, deferred=True)
    try:
        assert hit._deferred_history is not None, "precondition: a record HIT"
        assert _model(hit).reasoning_effort == "low"
    finally:
        await hit.dispose()

    _rewrite_in_place_keeping_mtime(
        directory / "transcript.jsonl", b'"effort":"low"', b'"effort":"max"'
    )
    replay_cache.replay_cache().clear()
    reads = counting_selection_reads(monkeypatch)
    after = await cold_open(isolated_config, deferred=True)
    try:
        assert after._deferred_history is None, "a same-size rewrite must not HIT"
        assert _model(after).reasoning_effort == "max", "the rewritten selection is served"
        assert reads, "the miss resolves the selection from the journal"
    finally:
        await after.dispose()
    expected_payload, _rows = await _eager_reference(isolated_config)
    replay_cache.replay_cache().clear()
    again = await cold_open(isolated_config, deferred=True)
    try:
        assert again._deferred_history is not None, "the rewrite re-recorded the new version"
        assert frontend_payload(again) == expected_payload
    finally:
        await again.dispose()

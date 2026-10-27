"""The desktop's FIRST FRAME, eager versus deferred, over the real bridge.

This is the wire contract the renderer depends on. The deferred cold facade
changes where a few durable facts are read from — a durable metadata record
instead of a journal parse — and it must not change the frame the client
receives for the same fixture.

THE ORACLE IS CANONICAL, NOT RAW. Two independent opens legitimately differ in
fields nothing derives from the journal: the bridge mints a fresh ``epoch``, the
frame carries a serve timestamp, and per-page cursors are opaque tokens minted
per read. ``_canonical`` drops exactly those, by name, from one list — anything
NOT on that list differing is a real regression and fails.

Shapes covered, because they are the inputs condensation and grouping read: a
settled conversation with a checkpoint, a checkpointless (pre-ledger) journal, a
journal whose last tool call has no result yet, and a compacted journal whose
head was cut.

The journals are built through the real writer (``Transcript``), so every row is
a row this build can parse; the compaction marker is appended as a raw line
because its id-referencing payload is exactly what the index reads for
``head_cut``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.harness.types import Message, TextContent, ToolCall, Usage
from local_operator.server.routes import desktop_sessions
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.session import cold_metadata, replay_cache
from local_operator.session import transcript_index as ti
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
from local_operator.session.spend import SESSION_SPEND_CUSTOM_TYPE
from local_operator.session.transcript import TRANSCRIPT_FILENAME, Transcript

SESSION_ID = "defeed012345"


async def _never_take_over() -> None:
    raise AssertionError("a cold facade never takes over")


#: Frame keys two independent opens cannot make equal, and why. ``epoch`` is
#: minted per facade (it names one attachment's lifetime), the serve clocks are
#: wall time, and cursors/tokens are opaque positions a reader mints per call.
#: Nothing on this list is derived from history.
_VOLATILE_KEYS = frozenset(
    {
        "epoch",
        "served_at",
        "now",
        "built_at",
        "generated_at",
        "ts",
        "ts_source",
        "cursor",
        "before_token",
        "snapshot_token",
        "subscription_id",
        "took_ms",
        "elapsed_ms",
    }
)


def _canonical(value: Any) -> Any:
    """``value`` with every documented-volatile field removed, recursively."""
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items() if key not in _VOLATILE_KEYS}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


async def _seed(
    root: Path,
    *,
    turns: int = 3,
    checkpoint: bool = True,
    spend: bool = True,
    unsettled_tail: bool = False,
    head_cut: bool = False,
) -> Path:
    """One conversation in the four shapes the first frame has to handle."""
    directory = root / "sessions" / SESSION_ID
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory)
    kept_from = ""
    for turn in range(turns):
        question = Message(role="user", content=[TextContent(text=f"question {turn}")])
        if turn == turns - 1:
            kept_from = question.id
        await transcript.append_message(question)
        if spend:
            await transcript.append_custom(
                SESSION_SPEND_CUSTOM_TYPE,
                {"usd": 0.5, "calls": turn + 1},
            )
        await transcript.append_message(
            Message(
                role="assistant",
                content=[TextContent(text=f"answer {turn}")],
                tool_calls=[ToolCall(id=f"call{turn}", name="bash", arguments={"command": "true"})],
                usage=Usage(input_tokens=11 * (turn + 1), output_tokens=7, context_tokens=500),
            )
        )
        if unsettled_tail and turn == turns - 1:
            # The last call has no result: the shape the renderer paints as
            # still running, which a settled-only row list cannot show.
            continue
        await transcript.append_message(
            Message(
                role="tool",
                tool_call_id=f"call{turn}",
                tool_name="bash",
                content=[TextContent(text=f"output {turn}")],
            )
        )
    if checkpoint:
        await transcript.append_custom(
            FRONTEND_CHECKPOINT_CUSTOM_TYPE,
            {
                "checkpoint_id": "c1",
                "state": {
                    "session_id": SESSION_ID,
                    "epoch": "previous-owner",
                    "sequence": 3,
                    "conversation_title": "Article search rollout",
                    "cumulative_parent_cost": 4.5,
                    "context_tokens": 900,
                    "context_window": 1_000_000,
                },
            },
        )
    transcript.flush()
    # CLOSE, do not just drop: the lock must be free when this returns, and a
    # dropped writer frees it only when its LAST reference dies. The executor
    # worker behind ``asyncio.to_thread`` keeps one (its work item holds the
    # write closure, which holds this object) until it runs ``del work_item``
    # after waking the loop, so a descheduled worker leaves the lock held while
    # the caller already runs ``compress_session`` -> "session in use". Seen
    # as a CI flake on unrelated PRs (runs 40423185038, 40430478001).
    transcript.close()
    if head_cut:
        # A compaction marker whose kept window starts ABOVE row zero is what the
        # index reports as ``head_cut``; the rows above it stay on disk, which is
        # the state the renderer's "older rows exist" affordance is built on.
        with (directory / TRANSCRIPT_FILENAME).open("a", encoding="utf-8") as handle:
            handle.write(
                '{"id":"compacted","ts":9999.0,"type":"compaction","payload":'
                '{"first_kept_entry_id":"' + kept_from + '","summary":"earlier work"}}\n'
            )
    return directory


#: Journal parses counted for the current test, and the hook that resets it.
#:
#: Counted rather than timed: a parse's duration moves by seconds under fleet
#: load, and the claim is a COUNT — a deferred open performs no journal parse.
_replay_calls = 0


def _reset_replay_calls() -> None:
    global _replay_calls
    _replay_calls = 0


@pytest.fixture(autouse=True)
def count_replay_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_replay_calls()
    original = replay_cache.read_replay_suffix

    def counted(*args: Any, **kwargs: Any) -> Any:
        global _replay_calls
        _replay_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(replay_cache, "read_replay_suffix", counted)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    replay_cache.replay_cache().clear()
    yield
    replay_cache.replay_cache().clear()
    ti._reset_for_tests()


async def _open(root: Path, sid: str) -> dict[str, Any]:
    """One cold desktop open of ``sid``, and the exposure it created.

    A FRESH POOL PER CALL, which is what makes the second open a second process
    as far as anything in memory can tell: no resident bridge, no replay-cache
    entry, no facade. Everything returned is read INSIDE the handover, because
    leaving the block detaches and disposes the facade it describes, and the
    parse count is sampled BEFORE the hydration so the two costs stay separable.
    """
    pool = DesktopSessions(root)
    facts: dict[str, Any] = {"session_id": sid, "parses_at_entry": _replay_calls}
    async with pool.session(sid) as bridge:
        sub = bridge.subscribe()
        stream = bridge.events(sub, epoch=bridge.epoch, after_seq=bridge.sequence)
        opened = await anext(stream)
        snapshot = await anext(stream)
        await stream.aclose()
        assert opened["type"] == "open" and snapshot["type"] == "snapshot"
        facade = bridge.remote
        assert facade is not None, "an open builds the facade this test is about"
        facts["frame"] = snapshot
        facts["deferred"] = facade._deferred_history is not None
        facts["parses_before_hydration"] = _replay_calls
        facts["after_hydration"] = None
        if facts["deferred"]:
            await facade.hydrate_deferred_history()
            assert facade._deferred_history is None, "hydration settles the deferral"
            facts["after_hydration"] = await bridge.snapshot()
        facts["rows"] = len(facade.history())
    return facts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape",
    [
        {"turns": 3},
        {"turns": 3, "checkpoint": False, "spend": False},
        {"turns": 3, "unsettled_tail": True},
        {"turns": 4, "head_cut": True},
    ],
    ids=["settled", "checkpointless", "unsettled-tail", "head-cut"],
)
async def test_the_first_events_frame_is_the_same_frame_either_way(
    tmp_path: Path, shape: dict[str, Any]
) -> None:
    """Wire equality, on the frame rather than on the rows.

    The deferred run must have ACTUALLY deferred and must have skipped the journal
    parse entirely, and the eager run must have paid it — both checked beside the
    equality, because two eager opens would otherwise satisfy it.
    """
    root = tmp_path / "config"
    root.mkdir(parents=True, exist_ok=True)
    ConfigManager(config_dir=root)
    await _seed(root, **shape)

    # Hold only the new optimization off: this is the pre-tail exact path.
    with patch("local_operator.session.attached.build_cold_tail_metadata", return_value=None):
        eager = await _open(root, SESSION_ID)
    assert eager["deferred"] is False
    assert cold_metadata.cold_metadata_path(
        root, SESSION_ID
    ).exists(), "the eager open is the write"
    assert (
        eager["parses_before_hydration"] > eager["parses_at_entry"]
    ), "the eager open parses the journal"

    # A SECOND PROCESS: nothing in memory survives, so only the durable record
    # can explain what the next open does.
    replay_cache.replay_cache().clear()
    _reset_replay_calls()
    deferred = await _open(root, SESSION_ID)
    assert deferred["deferred"] is True, "a matching record defers the message list"
    assert deferred["parses_before_hydration"] == 0, "a deferred open parses no journal"
    assert _replay_calls > 0, "and the hydration it defers does parse, once"

    assert _canonical(deferred["frame"]) == _canonical(eager["frame"])
    # The read it deferred produces the rows the eager open already had.
    assert deferred["rows"] == eager["rows"] > 0
    # And the frame does not shift once that read has landed.
    assert _canonical(deferred["after_hydration"]) == _canonical(eager["frame"])


@pytest.mark.asyncio
async def test_an_open_after_the_journal_changed_defers_and_paints_the_new_row(
    tmp_path: Path,
) -> None:
    """The record is for the journal version it describes, and no other.

    The row appended here must be painted, which is the visible consequence of
    the eager path being taken rather than served from a record that predates it.
    """
    root = tmp_path / "config"
    root.mkdir(parents=True, exist_ok=True)
    ConfigManager(config_dir=root)
    directory = await _seed(root, turns=2)

    first = await _open(root, SESSION_ID)
    assert first["deferred"] is True

    transcript = Transcript(directory)
    await transcript.append_message(
        Message(role="user", content=[TextContent(text="written later")])
    )
    transcript.flush()

    replay_cache.replay_cache().clear()
    second = await _open(root, SESSION_ID)
    assert second["deferred"] is True, "a moved journal rebuilds its record from fresh tail facts"
    assert second["rows"] == first["rows"] + 1


# --------------------------------------------------------------------------
# The ORDER the renderer actually uses: /history first, then /events
# --------------------------------------------------------------------------

TOKEN = "synthetic-deferred-token"


def _app(root: Path) -> tuple[FastAPI, Any]:
    app = FastAPI()
    app.include_router(desktop_sessions.router)
    pool = DesktopSessions(root)
    app.state.desktop_sessions = pool
    app.state.config_manager = ConfigManager(config_dir=root)
    return app, pool


async def _settle_index(root: Path) -> ti.TranscriptIndex | None:
    """Run this session's transcript-index refresh to completion; await any in flight.

    WHY: a frame's ``runs``/``runs_state`` come from the background transcript
    index, not from the facade under test. An open that finds no resident index
    starts a scan and, inside its 0.12 s snapshot budget, answers ``building``;
    the scan then lands whenever the thread pool gets to it. Two arms opened in
    sequence against the same root therefore compared two snapshots of a moving
    background state (CI run 40353003484: one arm ``building, runs: []``, the
    other ``ready`` with a run) -- a race unrelated to the eager-vs-tail claim.

    Settling through the module's own single-flight entry point (which also
    makes the result resident, the state ``fresh_resident`` answers from) gives
    every arm the same settled index, and draining what an arm left in flight
    keeps the next arm from observing a transition the previous one caused. No
    sleep and no retry: the task is awaited, not timed.
    """
    loop = asyncio.get_running_loop()
    for owner, task in list(ti._IN_FLIGHT.values()):
        if owner is loop:
            await task
    task, _started = ti.start_refresh(root, SESSION_ID)
    return await task


async def _page_then_frame(root: Path) -> dict[str, Any]:
    """``/history`` over the REAL route, then the ``/events`` handshake.

    That is the order the renderer asks in: the journal page paints first and the
    event stream follows it. The facts are sampled inside both handovers, because
    the facade a page builds is disposed when its bridge is released.

    The transcript index is settled before the arm and drained after it (see
    :func:`_settle_index`), so two calls compare the facades, not the index's
    progress. ``facts["index_ready"]`` records whether a settled index existed.
    """
    facts: dict[str, Any] = {"index_ready": await _settle_index(root) is not None}
    app, pool = _app(root)
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        response = await client.get(
            f"/v1/desktop/sessions/{SESSION_ID}/history", params={"open_frame": 1}
        )
        assert response.status_code == 200, response.text
        facts["page"] = _canonical(response.json()["result"])
    pool = DesktopSessions(root)
    async with pool.session(SESSION_ID) as bridge:
        facade = bridge.remote
        assert facade is not None
        facts["deferred_after_page"] = facade._deferred_history is not None
        facts["history_hydrated_after_page"] = facade._history_hydrated
        sub = bridge.subscribe()
        stream = bridge.events(sub, epoch=bridge.epoch, after_seq=bridge.sequence)
        facts["open"] = _canonical(await anext(stream))
        facts["frame"] = _canonical(await anext(stream))
        await stream.aclose()
    await _settle_index(root)
    return facts


@pytest.mark.asyncio
async def test_the_history_page_before_events_needs_no_hydrated_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The journal page is answered from the JOURNAL, not from the message list.

    This pins the ordering the renderer depends on: ``/history`` is requested
    before ``/events``, so the deferred facade has to serve a page while
    ``_history`` is still unread. The page must be the same page the eager facade
    serves, and asking for it must not quietly hydrate (which would turn the
    deferral into a per-page parse and defeat the point of the record).

    A page is read from ``load_transcript_page`` — the journal index — so the
    claim is that nothing in that path touches ``_history``, which the
    ``deferred_after_page`` assertion states directly rather than inferring from
    "it did not raise".
    """
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    root = tmp_path / "config"
    root.mkdir(parents=True, exist_ok=True)
    ConfigManager(config_dir=root)
    await _seed(root)

    with patch("local_operator.session.attached.build_cold_tail_metadata", return_value=None):
        eager = await _page_then_frame(root)
    assert cold_metadata.cold_metadata_path(root, SESSION_ID).exists()

    replay_cache.replay_cache().clear()
    _reset_replay_calls()
    deferred = await _page_then_frame(root)
    assert eager["index_ready"] and deferred["index_ready"], "both arms read one settled index"

    assert (
        deferred["deferred_after_page"] is True
    ), "serving /history must not hydrate the deferred message list"
    assert deferred["history_hydrated_after_page"] is False
    assert deferred["page"] == eager["page"], "the deferred page is the same page"
    assert deferred["frame"] == eager["frame"], "and so is the frame that follows it"


# --------------------------------------------------------------------------
# WHAT the deferred read installs: the journal as of the READ, and no other
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hydration_reads_the_journal_as_of_the_hydration_not_the_open(
    tmp_path: Path,
) -> None:
    """A row written while the facade sat deferred is INSTALLED when it reads.

    THE SEMANTICS, pinned rather than assumed. A deferred cold open captures a
    ``through_id`` of ``None`` — the same cut the eager read uses — so deferral
    moves the MOMENT of the read, never its cut. A row appended in between is
    therefore part of the hydrated list, exactly as it would have been had the
    eager read happened at hydration time.

    That is safe on every surface that reads it, and this test states the two
    reasons instead of relying on timing: the hydrated list equals what an EAGER
    facade reads at the same instant (so no row is lost, and none is duplicated),
    and ``_cold_painted_ids`` is derived from that same list (so the cold-to-live
    gap replay measures against what was actually installed).

    The alternative — fencing the read at the open's journal version — would need
    a byte cut the reader does not support, and would install a list OLDER than
    the page the renderer has already painted from the same journal.
    """
    root = tmp_path / "config"
    root.mkdir(parents=True, exist_ok=True)
    ConfigManager(config_dir=root)
    directory = await _seed(root, turns=2)
    # The first open is the record's writer, and it is eager by construction:
    # the deferral under test is a LATER process's open of the same journal.
    first = await AttachedSession.cold(
        SESSION_ID,
        config_dir=root,
        cwd=str(root),
        takeover_factory=_never_take_over,
        surface="desktop",
        deferred_history=True,
    )
    try:
        assert first._deferred_history is not None, "tail facts defer even a record MISS"
    finally:
        await first.dispose()
    replay_cache.replay_cache().clear()

    facade = await AttachedSession.cold(
        SESSION_ID,
        config_dir=root,
        cwd=str(root),
        takeover_factory=_never_take_over,
        surface="desktop",
        deferred_history=True,
    )
    try:
        assert facade._deferred_history is not None, "precondition: a hit deferred the read"
        transcript = Transcript(directory)
        late = Message(role="user", content=[TextContent(text="written while deferred")])
        await transcript.append_message(late)
        transcript.flush()

        await facade.hydrate_deferred_history()
        rows = [m.id for m in facade.history()]
        assert late.id in rows, "the read reflects the journal as of the read"
        assert rows.count(late.id) == 1, "and installs it exactly once"
        assert facade._cold_painted_ids == set(rows), (
            "the paint set is derived from what was installed, so the cold-to-live "
            "gap replay measures against this list and not an earlier one"
        )
    finally:
        await facade.dispose()

    # The same read, taken eagerly at the same moment, is the same list.
    replay_cache.replay_cache().clear()
    eager = await AttachedSession.cold(
        SESSION_ID,
        config_dir=root,
        cwd=str(root),
        takeover_factory=_never_take_over,
        surface="desktop",
    )
    try:
        assert [m.id for m in eager.history()] == rows
    finally:
        await eager.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape",
    [
        "unchanged",
        "append",
        "checkpoint",
        "compaction",
        "archive",
        "truncated",
        "corrupt",
        "legacy",
    ],
)
async def test_tail_miss_wire_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """Compare the real history route and event handshake to the exact eager path."""
    from local_operator.session import compression

    root = tmp_path / "config"
    root.mkdir()
    ConfigManager(config_dir=root)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    directory = await _seed(root, turns=5, checkpoint=shape != "legacy")
    journal = directory / TRANSCRIPT_FILENAME
    transcript = Transcript(directory)
    if shape == "append":
        await transcript.append_message(
            Message(id="tail-new", role="user", content=[TextContent(text="new append")])
        )
    elif shape == "checkpoint":
        await transcript.append_custom(
            FRONTEND_CHECKPOINT_CUSTOM_TYPE,
            {
                "checkpoint_id": "new",
                "state": {
                    "session_id": SESSION_ID,
                    "epoch": "new",
                    "conversation_title": "newest title",
                },
            },
        )
    elif shape == "compaction":
        kept = next(
            entry.id
            for entry in reversed(transcript.entries())
            if entry.type == "message" and entry.payload.get("role") == "user"
        )
        await transcript.append_compaction("new summary", kept, 2000)
    transcript.flush()
    transcript.close()
    if shape == "archive":
        outcome = compression.compress_session(directory)
        assert outcome.status == "compressed", outcome
    elif shape == "truncated":
        with journal.open("ab") as out:
            out.write(b'{"type":"custom"')
    elif shape == "corrupt":
        with journal.open("ab") as out:
            out.write(b"{not json}\n")

    with patch("local_operator.session.attached.read_cold_metadata", return_value=None):
        with patch("local_operator.session.attached.build_cold_tail_metadata", return_value=None):
            eager = await _page_then_frame(root)
        replay_cache.replay_cache().clear()
        tail = await _page_then_frame(root)
    # Both arms opened over the same settled index, so the page's run facts
    # compare facades rather than how far a background scan had got. An
    # archived session has no plain journal for the index to read, so it
    # settles to "no index" on both arms instead.
    assert eager["index_ready"] is tail["index_ready"] is (shape != "archive")
    assert tail["page"] == eager["page"]
    assert tail["open"] == eager["open"]
    assert tail["frame"] == eager["frame"]
    assert tail["deferred_after_page"] == (shape in {"unchanged", "append", "checkpoint"})
    if shape == "checkpoint":
        assert "newest title" in str(tail["frame"])

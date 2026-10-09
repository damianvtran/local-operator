"""Scroll-back REACH on the relay's history route.

WHY THESE EXIST
---------------
The phone pages history out of the durable fold's render, and that render begins
at the newest compaction's ``first_kept_entry_id`` — so everything a compaction
dropped was unreachable from the phone while the desktop's own history route
(``read_transcript_page`` over the journal) had always served it. Measured on
the S6 first-paint fixture before the fix, on the real relay: the seed's oldest
row sat at journal line 2706 of 3,348, one ``/history?before=`` page answered
that row's single neighbour, ``has_more`` went false, and the walk stopped —
639 of 3,348 journal lines reachable, and the conversation's opening turns
simply absent from the phone.

The property these tests pin is REACH, not speed: every row a fold of the whole
journal would paint has to be reachable by paging, exactly once, in order. The
fixtures are small on purpose — the defect is structural, and a small transcript
makes the expected row list readable.

Deliberately NOT a fixture-replay test: the shapes that broke it are ordinary
(messages before a compaction, a compaction chain, a message whose rows outnumber
a page), so they are built here with the core's own writer.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import AgentMessage, Message, TextContent, ToolCall
from local_operator.mobile.daemon import _durable_projection, _history_page

# Only PRE-existing names are imported at module scope, from the fold and the
# transcript: the tests that exercise the fix reach it through the ROUTE, so on
# a tree without it they fail on their own assertions rather than at import time.
from local_operator.mobile.durable import (
    DurableFoldCache,
    _apply_prune_to,
    _attachments,
    _fold,
)
from local_operator.session.transcript import (
    ENTRY_COMPACTION,
    ENTRY_MESSAGE,
    Transcript,
    TranscriptEntry,
    _compaction_marker,
    _entry_to_message,
)

SESSION_ID = "reach-session"
#: Small enough that a tool-heavy message's rows straddle a page cut (the
#: stranded-sibling case), large enough to hold several turns.
PAGE = 4


def _build(
    config: Path,
    *,
    early_turns: int = 6,
    kept_turns: int = 3,
    earlier_kept_turns: int = 1,
    mid_heavy_calls: int = 5,
    tool_calls: int = 6,
) -> dict[str, Any]:
    """A journal with TWO compactions, a tool-heavy message, and a prune.

    Journal order, oldest first: [early turns] [c1] [earlier-kept turns] [c2]
    [tool-heavy message] [kept turns] [prune of an early answer]. So the newest
    compaction's window holds only the last two groups, and everything before it
    is what the phone has to be able to reach.
    """
    directory = config / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    ids: dict[str, str] = {}

    async def run() -> None:
        for i in range(early_turns):
            user = Message.user(f"early user {i}", id=f"early-u{i}")
            answer = Message.assistant(f"early answer {i}", id=f"early-a{i}")
            await transcript.append_message(user)
            await transcript.append_message(answer)
            ids[f"early_u{i}"] = user.id
            ids[f"early_a{i}"] = answer.id

        await transcript.append_compaction("summary of the early turns", "mid-u0", 1000)

        # A MULTI-ROW GROUP BEHIND THE BOUNDARY, which is what a page cut can
        # land inside (see the strand test): the rows of this one message are
        # the only ones in the archive that outnumber a one-row group.
        await transcript.append_message(
            Message(
                role="assistant",
                content=[TextContent(text="running the mid checks")],
                tool_calls=[
                    ToolCall(id=f"mid-call-{i}", name="bash", arguments={"command": "pytest -q"})
                    for i in range(mid_heavy_calls)
                ],
                stop_reason="toolUse",
                id="mid-heavy",
            )
        )
        for i in range(mid_heavy_calls):
            await transcript.append_message(
                Message(
                    role="tool",
                    content=[TextContent(text=f"mid output {i}")],
                    tool_call_id=f"mid-call-{i}",
                    tool_name="bash",
                    id=f"mid-res-{i}",
                )
            )

        for i in range(earlier_kept_turns):
            user = Message.user(f"mid user {i}", id=f"mid-u{i}")
            answer = Message.assistant(f"mid answer {i}", id=f"mid-a{i}")
            await transcript.append_message(user)
            await transcript.append_message(answer)
            ids[f"mid_u{i}"] = user.id
            ids[f"mid_a{i}"] = answer.id

        await transcript.append_compaction("summary of the middle turns", "tool-heavy", 2000)

        # ONE message, MANY rows: the fold paints the assistant row plus one row
        # per tool call, so a page cut inside this group is a real cut.
        await transcript.append_message(
            Message(
                role="assistant",
                content=[TextContent(text="running the checks")],
                tool_calls=[
                    ToolCall(id=f"call-{i}", name="bash", arguments={"command": "pytest -q"})
                    for i in range(tool_calls)
                ],
                stop_reason="toolUse",
                id="tool-heavy",
            )
        )
        for i in range(tool_calls):
            await transcript.append_message(
                Message(
                    role="tool",
                    content=[TextContent(text=f"output {i}")],
                    tool_call_id=f"call-{i}",
                    tool_name="bash",
                    id=f"res-{i}",
                )
            )

        for i in range(kept_turns):
            user = Message.user(f"kept user {i}", id=f"kept-u{i}")
            answer = Message.assistant(f"kept answer {i}", id=f"kept-a{i}")
            await transcript.append_message(user)
            await transcript.append_message(answer)
            ids[f"kept_u{i}"] = user.id
            ids[f"kept_a{i}"] = answer.id

        await transcript.append_prune("early-a0", "[pruned: superseded]")

    asyncio.run(run())
    return ids


def _journal_rows(directory: Path) -> list[Any]:
    """Rows a fold of the WHOLE journal paints, in journal order.

    The ground truth for reach: the messages, plus the in-band compaction marker
    each compaction entry renders as (the same helper the live prefix uses).
    """
    messages: list[AgentMessage] = []
    prunes: dict[str, str] = {}
    entries = [
        entry
        for entry in (
            TranscriptEntry.from_json(line)
            for line in (directory / "transcript.jsonl").read_text().splitlines()
            if line.strip()
        )
        if entry is not None
    ]
    for entry in entries:
        if entry.type == "prune" and entry.payload.get("target"):
            prunes[str(entry.payload["target"])] = str(entry.payload.get("notice", ""))
    for entry in entries:
        if entry.type == ENTRY_COMPACTION:
            messages.append(_compaction_marker(entry))
            continue
        if entry.type != ENTRY_MESSAGE:
            continue
        message = _entry_to_message(entry, _attachments())
        if message is None:
            continue
        if entry.id in prunes and isinstance(message, Message):
            _apply_prune_to(message, prunes[entry.id])
        messages.append(message)
    return _fold(messages)


def _walk(config: Path, monkeypatch: pytest.MonkeyPatch, *, page: int = PAGE) -> dict[str, Any]:
    """Drive the route the way the web client does, from the seed's oldest row."""
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)
    projection = _durable_projection(SESSION_ID)
    assert projection is not None
    seed_ids = [row.id for row in projection.transcript]

    pages: list[list[str]] = []
    cursor = seed_ids[0]
    for _ in range(60):
        entries, has_more = _history_page(SESSION_ID, cursor, page)
        if not entries:
            break
        pages.append([row.id for row in entries])
        cursor = entries[0].id
        if not has_more:
            break
    return {"seed_ids": seed_ids, "pages": pages, "directory": config / "sessions" / SESSION_ID}


def _observed(walk: dict[str, Any]) -> list[str]:
    """The reader's merged list, oldest first, de-duplicated by id.

    The web client PREPENDS each page above the next-newer one and keeps the
    older row on an id collision (``transcript.tsx``'s merge), so a page that
    overlaps the render's head costs a duplicate, never a hole.
    """
    ordered: list[str] = list(walk["seed_ids"])
    for page in walk["pages"]:
        ordered = page + [row for row in ordered if row not in set(page)]
    return ordered


def test_pages_reach_the_rows_behind_the_newest_compaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect: a phone reader stopped at the last compaction.

    Before the fix the first page answered one row and ``has_more`` went false,
    so nothing below the newest ``first_kept_entry_id`` was reachable.
    """
    config = tmp_path / "config"
    ids = _build(config)
    walk = _walk(config, monkeypatch)
    observed = set(_observed(walk))
    for name in ("early_u0", "early_a0", "mid_u0", "mid_a0"):
        assert ids[name] in observed, f"{name} is unreachable from the phone"
    assert "mid-heavy" in observed, "the message behind the boundary is unreachable"
    assert "mid-heavy:mid-call-4" in observed, "its last tool row is unreachable"
    assert len(walk["pages"]) > 1, "one page cannot hold this journal's rows"


def test_paging_recovers_every_folded_row_exactly_once_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reach as a property, not a spot check: nothing missing, nothing twice.

    The reader's merged list (seed + prepended pages, de-duplicated) has to BE
    the whole-journal fold — same rows, same order — because a row a fold would
    paint and the route cannot reach is a conversation the phone cannot show.
    """
    config = tmp_path / "config"
    _build(config)
    walk = _walk(config, monkeypatch)
    rows = _journal_rows(walk["directory"])
    expected = [row.id for row in rows]
    observed = _observed(walk)

    # The render's compaction marker stands in for the newest compaction's row,
    # which the fold of the whole journal also paints — so the lists line up
    # including markers, and a MISSING row is a hole the reader could not pass.
    assert [row for row in expected if row not in set(observed)] == []
    assert observed[: len(expected)] == expected
    assert len(observed) == len(set(observed)), "a page re-served a row the reader held"


def test_a_page_cut_does_not_strand_a_tool_rows_older_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cut inside a row GROUP must not lose the group's older rows.

    The web client pages with the id of the oldest row it holds, and the journal
    reader's cursor is the MESSAGE that row belongs to. So a page that starts
    mid-group — at, say, a tool row — makes the next page start strictly below
    the message, and the rows above that tool row are never asked for again
    (measured on the S6 fixture: 2 of 1,813 rows were unreachable exactly this
    way, before the cut was moved onto group boundaries).
    """
    config = tmp_path / "config"
    _build(config, tool_calls=6)
    walk = _walk(config, monkeypatch)
    observed = _observed(walk)
    heavy = [row for row in observed if row.startswith("tool-heavy")]
    assert heavy, "the tool-heavy message's group is not reachable at all"
    # The assistant row leads its group; every call row follows it.
    assert heavy[0] == "tool-heavy"
    assert sorted(heavy) == sorted(["tool-heavy"] + [f"tool-heavy:call-{i}" for i in range(6)])


def test_a_session_without_a_compaction_still_terminates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No compaction means no archive: the walk ends at the journal's first row.

    A second read path that never returns false on ``has_more`` would leave the
    client paging for ever; this pins the honest end of history.
    """
    config = tmp_path / "config"
    directory = config / "sessions" / "plain-session"
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    for i in range(3):
        asyncio.run(transcript.append_message(Message.user(f"user {i}", id=f"u{i}")))
        asyncio.run(transcript.append_message(Message.assistant(f"answer {i}", id=f"a{i}")))
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)

    pages = 0
    cursor = "a2"
    while pages < 20:
        entries, has_more = _history_page("plain-session", cursor, 2)
        pages += 1
        if not entries:
            assert not has_more
            break
        cursor = entries[0].id
        if not has_more:
            break
    assert pages < 20, "the walk never reported the end of history"
    assert cursor == "u0", "the walk did not reach the journal's own first row"


def test_an_unknown_cursor_ends_the_walk_without_serving_the_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pruned anchor is end-of-history, not a licence to re-serve the tail.

    The journal reader reports an unlocatable ``before_id`` as the current tail
    with ``reconciled=True``; passing that through would duplicate the reader's
    live window and leave the client looping on rows it already holds.
    """
    config = tmp_path / "config"
    _build(config)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)
    assert _history_page(SESSION_ID, "no-such-row-anywhere", 10) == ([], False)
    assert _history_page(SESSION_ID, "early-u0:call-not-a-real-call", 10) == ([], False)


def test_the_render_starts_at_the_newest_compactions_kept_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The boundary the archive pages from is the newest window's first row.

    It is the fold's own answer (``_replay`` knows the index) rather than a
    second scan of the file, and the render's head has to agree with it.
    """
    config = tmp_path / "config"
    _build(config)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)

    state = DurableFoldCache().load(config / "sessions" / SESSION_ID)
    assert state.keep_start_id == "tool-heavy"


def test_the_compaction_marker_carries_the_journal_id_other_surfaces_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One row, one id — and the id the phone pages with.

    The marker was built locally and left id-less, so pydantic minted a uuid:
    a NEW one on every fold. The web client pages with its oldest row's id, so
    for any session whose render is short enough to start at the marker, the
    cursor it sent named nothing — and the walk stopped there. The transcript
    module's own ``_compaction_marker`` stamps ``id=entry.id``; the phone now
    uses it.
    """
    config = tmp_path / "config"
    _build(config)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)

    directory = config / "sessions" / SESSION_ID
    state = DurableFoldCache().load(directory)
    assert state.render[0].kind == "notice"
    compaction_ids = {
        entry.id
        for entry in (
            TranscriptEntry.from_json(line)
            for line in (directory / "transcript.jsonl").read_text().splitlines()
            if line.strip()
        )
        if entry is not None and entry.type == ENTRY_COMPACTION
    }
    assert state.render[0].id in compaction_ids
    # And a fresh fold of the same history agrees: the id is a property of the
    # journal row, not of the fold that happened to build it.
    rebuilt = _fold(state.history)
    assert rebuilt[0].id == state.render[0].id


def test_the_paged_rows_serialize_through_the_routes_own_serializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The payload shape is unchanged: the archive's rows are wire rows.

    ``_transcript_entry_json`` is what the history route writes for every row,
    so a row that cannot pass through it is a row the phone would never
    receive. Paged through the ROUTE and serialized with the route's own
    function — the two ends of the same wire.
    """
    config = tmp_path / "config"
    _build(config)
    from local_operator.mobile.daemon import _transcript_entry_json

    directory = config / "sessions" / SESSION_ID
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)
    state = DurableFoldCache().load(directory)
    page, has_more = _history_page(SESSION_ID, state.keep_start_id, 12)
    assert has_more is True
    # The page is the row above the boundary plus rows from behind it, and the
    # cut lands on a row-group boundary (see the group test), so the length is
    # bounded by the limit rather than equal to it.
    assert 9 <= len(page) <= 12
    assert any(row.id.startswith("early-") for row in page)
    for row in page:
        payload = _transcript_entry_json(row)
        assert payload["id"] == row.id
        assert payload["kind"] == row.kind
        json.dumps(payload)

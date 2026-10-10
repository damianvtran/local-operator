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

# The projection's own cap (``_cap_tail``): the seed window is only capped once the
# render passes it, and the client's cursor rule turns on being AT it.
from local_operator.mobile.types import PROJECTION_TRANSCRIPT_LIMIT
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
    launch_row: bool = False,
    secret_prune: bool = False,
    textless_calls: int = 0,
    colon_call: bool = False,
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
        if secret_prune:
            # EARLY, because the leak this pins is an OLD row whose prune marker
            # sits ABOVE it: the page that serves the row comes from the journal,
            # and a fold that read only a suffix has no prune for it.
            call = ToolCall(id="secret-call", name="bash", arguments={"command": "cat credentials"})
            await transcript.append_message(
                Message(
                    role="assistant",
                    content=[TextContent(text="")],
                    tool_calls=[call],
                    stop_reason="toolUse",
                    id="secret-call-row",
                )
            )
            await transcript.append_message(
                Message(
                    role="tool",
                    content=[TextContent(text="SECRET-OUTPUT")],
                    tool_call_id=call.id,
                    tool_name="bash",
                    id="secret-result",
                )
            )
            await transcript.append_prune("secret-result", "[pruned]")
            ids["secret_row"] = "secret-call-row"
            ids["secret_result"] = "secret-result"

        for i in range(early_turns):
            user = Message.user(f"early user {i}", id=f"early-u{i}")
            answer = Message.assistant(f"early answer {i}", id=f"early-a{i}")
            await transcript.append_message(user)
            await transcript.append_message(answer)
            ids[f"early_u{i}"] = user.id
            ids[f"early_a{i}"] = answer.id
            if launch_row and i == 0:
                # ``harness/subagent.py`` mints colon-bearing ids, and 13 of the
                # 60 largest journals on this host name one as their newest
                # ``first_kept_entry_id``. It sits BELOW the newest compaction's
                # cut, so a cursor naming it needs the JOURNAL (and the id
                # narrowing) rather than the render — the half of the defect the
                # render-side path hides.
                launch = Message.user("launch the reviewer", id="subagent-launch:job-7f3")
                await transcript.append_message(launch)
                await transcript.append_message(
                    Message.assistant("launched", id="subagent-launch:job-7f3:answer")
                )
                ids["launch"] = launch.id

        if textless_calls:
            # A TEXT-LESS multi-call message paints NO row of its own — only
            # ``<id>:<call id>`` per call — so its own id is never among the row
            # ids, and a page cut that grouped rows by row ids alone split the
            # siblings (review round 2, R2-1). Each of the 12 largest real
            # journals here carries 25-572 of these.
            await transcript.append_message(
                Message(
                    role="assistant",
                    content=[],
                    tool_calls=[
                        ToolCall(id=f"toolonly-call-{i}", name="bash", arguments={})
                        for i in range(textless_calls)
                    ],
                    stop_reason="toolUse",
                    id="toolonly",
                )
            )
            for i in range(textless_calls):
                await transcript.append_message(
                    Message(
                        role="tool",
                        content=[TextContent(text=f"toolonly output {i}")],
                        tool_call_id=f"toolonly-call-{i}",
                        tool_name="bash",
                        id=f"toolonly-res-{i}",
                    )
                )
            ids["toolonly"] = "toolonly"

        if colon_call:
            # Real providers mint call ids with a colon of their own
            # (``call_00_x7:06c08d4847`` in journal 439818272d84), so the ROW id
            # carries two — and one narrowing step cannot name the message
            # (review round 2, R2-2).
            call = ToolCall(id="call_00_x7:06c08d4847", name="bash", arguments={"i": "true"})
            await transcript.append_message(
                Message(
                    role="assistant",
                    content=[TextContent(text="colon call")],
                    tool_calls=[call],
                    stop_reason="toolUse",
                    id="coloncall",
                )
            )
            await transcript.append_message(
                Message(
                    role="tool",
                    content=[TextContent(text="ok")],
                    tool_call_id=call.id,
                    tool_name="bash",
                    id="coloncall-res",
                )
            )
            ids["colon_call_row"] = "coloncall:call_00_x7:06c08d4847"

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


def _journal_entries(directory: Path) -> list[TranscriptEntry]:
    """Every entry in a session's journal, as the reader would parse it."""
    return [
        entry
        for entry in (
            TranscriptEntry.from_json(line)
            for line in (directory / "transcript.jsonl").read_text().splitlines()
            if line.strip()
        )
        if entry is not None
    ]


def _walk(config: Path, monkeypatch: pytest.MonkeyPatch, *, page: int = PAGE) -> dict[str, Any]:
    """Drive the route the way the web client does, from the seed's oldest row."""
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)
    projection = _durable_projection(SESSION_ID)
    assert projection is not None
    seed_ids = [row.id for row in projection.transcript]

    pages: list[list[str]] = []
    page_rows: list[Any] = []
    # THE CLIENT'S OWN FIRST CURSOR. A projection at its cap pins the
    # conversation's opening user row ahead of a disjoint tail, and that pinned
    # row is not the tail's chronological neighbour — the web client pages from
    # the row after it (``transcript.tsx``), and every fixture whose projection is
    # UNDER the cap behaves identically either way.
    cursor = (
        seed_ids[1]
        if len(seed_ids) >= PROJECTION_TRANSCRIPT_LIMIT
        and seed_ids[1:]
        and projection.transcript[0].kind == "user"
        else seed_ids[0]
    )
    seen_more = True
    for _ in range(60):
        entries, has_more = _history_page(SESSION_ID, cursor, page)
        seen_more = has_more
        if not entries:
            break
        pages.append([row.id for row in entries])
        page_rows.extend(entries)
        cursor = entries[0].id
        if not has_more:
            break
    return {
        "seed_ids": seed_ids,
        "pages": pages,
        "page_rows": page_rows,
        "exhausted": not seen_more,
        "directory": config / "sessions" / SESSION_ID,
    }


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
    """The boundary the archive pages from names the journal's own kept row.

    Behavioural rather than an attribute assertion (review round 1, F7): a
    boundary is only useful if it names a row the JOURNAL has — a stale or
    synthesised id makes the reader reconcile, and the walk stops there — and if
    the rows below it are reachable, which is the whole reason it exists.
    """
    config = tmp_path / "config"
    ids = _build(config)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)
    directory = config / "sessions" / SESSION_ID

    state = DurableFoldCache().load(directory)
    journal = _journal_entries(directory)
    assert state.keep_start_id in {entry.id for entry in journal}
    newest = [entry for entry in journal if entry.type == ENTRY_COMPACTION][-1]
    assert state.keep_start_id == newest.payload["first_kept_entry_id"]

    # The render opens with the compaction row; paging from it crosses the
    # boundary into the rows that compaction dropped.
    page, has_more = _history_page(SESSION_ID, state.render[0].id, 10)
    served = {row.id for row in page}
    assert ids["mid_u0"] in served, "the compaction row is not a usable cursor"
    assert has_more is True


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
        entry.id for entry in _journal_entries(directory) if entry.type == ENTRY_COMPACTION
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


# ---------------------------------------------------------------------------
# The seam the fixtures could not see
# ---------------------------------------------------------------------------


def _build_preserved(config: Path, *, turns: int = 5) -> dict[str, Any]:
    """A session compacted by the CORE'S OWN helpers, preserved turns included.

    Built with ``extract_preserved_user_turns`` + ``cap_preserved_user_turns``
    over the block the cut summarises — the two helpers the runtime's compaction
    pass calls — so the payload the fold reads is the payload a real journal
    carries. A hand-written payload would pin this file's idea of the shape
    instead of the runtime's, which is how the first version of this lane's
    parity walk passed while real sessions cycled: every fixture compacted with
    ``{"summary", "first_kept_entry_id", "tokens_before"}`` and none carried
    preserved turns, which 60 of the 60 largest journals on this host do.
    """
    from local_operator.compaction.cutpoint import (
        cap_preserved_user_turns,
        extract_preserved_user_turns,
    )

    directory = config / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    summarised: list[Any] = []

    async def run() -> None:
        for i in range(turns):
            user = Message.user(f"summarised user {i}", id=f"kept-out-u{i}")
            answer = Message.assistant(f"summarised answer {i}", id=f"kept-out-a{i}")
            await transcript.append_message(user)
            await transcript.append_message(answer)
            summarised.extend([user, answer])
        cut = Message.user("the kept window opens here", id="kept-u0")
        await transcript.append_message(cut)
        genuine = {str(m.id) for m in summarised if m.role == "user"}
        preserved = [dict(t) for t in extract_preserved_user_turns(list(summarised), genuine)]
        capped = cap_preserved_user_turns(preserved, cap=40_000)
        await transcript.append_compaction(
            "summary of the summarised turns",
            cut.id,
            180_000,
            # ``cap_preserved_user_turns`` answers Mappings; the writer's payload
            # wants plain ``{id, text}`` pairs, which is the shape the pass
            # itself hands over (``Session._run_compaction``).
            preserved_user_turns=[
                {"id": str(turn["id"]), "text": str(turn["text"])} for turn in capped.turns
            ],
            preserved_turns_cap=40_000,
        )
        for i in range(3):
            await transcript.append_message(Message.user(f"kept user {i}", id=f"kept-u{i + 1}"))
            await transcript.append_message(Message.assistant(f"kept answer {i}", id=f"kept-a{i}"))

    asyncio.run(run())
    return {"preserved_user_ids": [f"kept-out-u{i}" for i in range(turns)], "directory": directory}


def test_preserved_turns_are_served_once_at_their_journal_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shape that can serve one row twice — and did, on every real journal.

    The render re-injects the newest compaction's preserved turns at its head
    UNDER THEIR ORIGINAL ROW IDS while the journal still holds those rows below
    the cut. A cursor that lands on one of them therefore answered a page around
    the render's head plus a refill from the compaction boundary: ``has_more``
    stayed true, the cursors cycled, and the phone held 117,020 mounted rows for
    408 distinct ids (synthetic: 4,920 rows for 351).
    """
    config = tmp_path / "config"
    built = _build_preserved(config)
    walk = _walk(config, monkeypatch)
    observed = _observed(walk)

    assert len(observed) == len(set(observed)), "a row was served twice"
    expected = [row.id for row in _journal_rows(walk["directory"])]
    assert [row for row in expected if row not in set(observed)] == [], "a row is unreachable"
    for row_id in built["preserved_user_ids"]:
        assert observed.count(row_id) == 1, f"{row_id} was served {observed.count(row_id)} times"
        assert observed.index(row_id) == expected.index(
            row_id
        ), f"{row_id} was served out of journal order"


def test_a_walk_over_preserved_turns_terminates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Termination as a bound on REQUESTS, not just a ``has_more`` at the end.

    The cycle this pins was not slow: it was unbounded requests, each one
    re-serving pages the reader already held, which no row-count assertion can
    see (the client de-dupes) and which a real phone showed as a page that never
    stopped fetching.
    """
    config = tmp_path / "config"
    _build_preserved(config, turns=8)
    walk = _walk(config, monkeypatch)
    rows = len(_journal_rows(walk["directory"]))
    assert walk["exhausted"], "the walk never reported the end of history"
    assert (
        len(walk["pages"]) <= rows // PAGE + 2
    ), f"{len(walk['pages'])} requests for {rows} rows: the cursors are cycling"


def test_a_colon_bearing_entry_id_is_a_usable_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``subagent-launch:<job>`` is a journal id, and 13 of 60 real journals name one.

    Narrowing the cursor on the FIRST colon turned it into ``subagent-launch``,
    which resolves nowhere: the reader reconciled, the walk ended, and every row
    below became unreachable — the defect this lane exists to fix, reintroduced
    by an assumption about id shape (reproduced on a real 14 MB journal: 1
    request, ``has_more: false``, lines 0-4,648 unreachable).
    """
    config = tmp_path / "config"
    _build(config, launch_row=True)
    walk = _walk(config, monkeypatch)
    observed = set(_observed(walk))
    assert "subagent-launch:job-7f3" in observed, "the launch row is unreachable"
    assert "early-u0" in observed, "the walk stopped at the colon id"
    assert "mid-heavy" in observed

    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)
    # The discriminating call: this cursor names a row BEHIND the cut, so only
    # the journal can answer it — and the base answers `([], False)`, because
    # narrowing on the first colon asked for ``subagent-launch``.
    page, _ = _history_page(SESSION_ID, "subagent-launch:job-7f3", 10)
    served = [row.id for row in page]
    assert served, "a cursor naming a colon-bearing entry served nothing"
    assert "early-u0" in served and "early-a0" in served


def test_a_call_split_across_a_read_boundary_still_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paged tool row must paint what the whole-journal fold paints.

    Each read is folded on its own, and ``fold_messages_to_entries`` pairs a call
    with its result by looking the call up in what it already walked — so a call
    whose result sits in the NEXT (newer) read is left ``interrupted``, a lie
    about a call that returned. Measured on the fixtures: 2 tool rows per walk
    came out ``done -> interrupted``. The read bound is squeezed here so the
    split is deterministic rather than a function of the fixture's size.
    """
    from local_operator.mobile import durable as durable_module

    config = tmp_path / "config"
    _build(config, tool_calls=6)
    monkeypatch.setattr(durable_module, "_ARCHIVE_READ_LIMIT", 3)
    walk = _walk(config, monkeypatch)

    whole = {
        row.id: row.tool_state for row in _journal_rows(walk["directory"]) if row.kind == "tool"
    }
    paged = {row.id: row.tool_state for row in walk["page_rows"] if row.kind == "tool"}
    assert paged, "no tool rows were paged at all"
    mismatches = {
        row_id: (whole[row_id], state)
        for row_id, state in paged.items()
        if whole.get(row_id) != state
    }
    assert mismatches == {}, f"paged rows disagree with the whole-journal fold: {mismatches}"


def test_a_prune_above_the_fold_window_still_blanks_the_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A suffix fold's prune map covers its window; the archive must not leak.

    ``DurableFoldState.prunes`` is complete only when the fold read the journal
    from its first row. Lane T3's bounded read does not, and a prune marker sits
    ABOVE the row it blanks — so a page served from the fold's map alone hands
    the phone the tool output the live fold had hidden (measured on the merged
    tree: a result pruned more than 1 MiB above the window came back verbatim).
    """
    from local_operator.mobile.durable import journal_rows_older_than

    config = tmp_path / "config"
    _build(config, secret_prune=True)
    directory = config / "sessions" / SESSION_ID
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)

    # The CONTROL: the whole-journal fold's own map redacts it.
    state = DurableFoldCache().load(directory)
    with_map = journal_rows_older_than(directory, state.prunes, before_id="tool-heavy", limit=200)
    assert with_map is not None
    assert "SECRET-OUTPUT" not in json.dumps(_jsonable(with_map[0]))

    # THE CASE: a fold that read a suffix has no map for these rows, so the
    # reader rebuilds it from the journal before serving anything.
    rebuilt = journal_rows_older_than(
        directory, {}, before_id="tool-heavy", limit=200, prunes_complete=False
    )
    assert rebuilt is not None
    assert "SECRET-OUTPUT" not in json.dumps(
        _jsonable(rebuilt[0])
    ), "a pruned tool output paged back unredacted"
    assert "[pruned]" in json.dumps(_jsonable(rebuilt[0]))
    # The row the prune targets really is in this page: without it the assertions
    # above would pass for the wrong reason. The row's id is the CALL's row
    # (``<message id>:<call id>``), which is the row a reader scrolls past.
    leaked = [row for row in rebuilt[0] if str(row.id).startswith("secret-call-row")]
    assert leaked, "the pruned row is not in this page at all"
    assert any(row.tool_call_id == "secret-call" for row in leaked)


def _jsonable(rows: list[Any]) -> list[dict[str, Any]]:
    """Rows as the route serializes them (``to_json``), for content assertions."""
    return [row.to_json() for row in rows]


def test_a_suffix_fold_keeps_its_boundary_at_the_kept_window(
    tmp_path: Path,
) -> None:
    """``at_bof`` decides whether index 0 means "the journal starts here".

    A compaction's entry is written AFTER the rows it keeps, so a bounded suffix
    read that begins exactly at ``first_kept`` still holds that compaction — and
    the kept row is then the suffix's FIRST entry. Reading that index as "the
    journal's first row" answers ``keep_start_id=None`` and switches the archive
    off for a conversation whose older rows are all still on disk (reproduced on
    the merged tree: the span from ``first_kept`` to EOF at 1 MiB - 4 and - 1
    bytes gave ``None``, at +1 and +3 bytes the kept row).
    """
    from local_operator.mobile.durable import _replay

    config = tmp_path / "config"
    _build_preserved(config)
    directory = config / "sessions" / SESSION_ID
    entries = _journal_entries(directory)
    kept_at = next(i for i, entry in enumerate(entries) if entry.id == "kept-u0")
    suffix = entries[kept_at:]
    assert suffix[0].id == "kept-u0"
    assert any(
        entry.type == ENTRY_COMPACTION for entry in suffix
    ), "the compaction must be inside the suffix for this to be the seam"

    _, boundary_at_bof = _replay(suffix, at_bof=True)
    assert boundary_at_bof is None, "an index-0 boundary is the journal's own start"

    _, boundary_suffix = _replay(suffix, at_bof=False)
    assert (
        boundary_suffix == "kept-u0"
    ), "a suffix read starting at the kept window must still open the archive"
    _, boundary_whole = _replay(entries, at_bof=True)
    assert boundary_whole == "kept-u0"


def test_a_compaction_without_a_kept_row_serves_its_marker_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``first_kept_entry_id=None``: the marker is a journal row, served once.

    Its boundary becomes the row after the compaction, so an archive that also
    emits the compaction row delivers it twice — the same seam as the preserved
    turns, with the id supplied by the fold instead of by a payload.
    """
    config = tmp_path / "config"
    directory = config / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    transcript = Transcript(directory)

    async def run() -> None:
        for i in range(4):
            await transcript.append_message(Message.user(f"user {i}", id=f"u{i}"))
            await transcript.append_message(Message.assistant(f"answer {i}", id=f"a{i}"))
        # NO ``first_kept_entry_id`` KEY AT ALL, which is the payload a
        # pre-field revision leaves behind: the writer's own signature requires
        # the id, so the row is journalled through the primitive that writes
        # every entry. The fold reads the key with ``.get`` and treats a missing
        # one as "the window opens after the compaction".
        await transcript._append(
            ENTRY_COMPACTION, {"summary": "summary with no kept row", "tokens_before": 1000}
        )
        for i in range(4, 8):
            await transcript.append_message(Message.user(f"user {i}", id=f"u{i}"))
            await transcript.append_message(Message.assistant(f"answer {i}", id=f"a{i}"))

    asyncio.run(run())
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)
    compaction_ids = {
        entry.id for entry in _journal_entries(directory) if entry.type == ENTRY_COMPACTION
    }
    assert len(compaction_ids) == 1

    walk = _walk(config, monkeypatch)
    observed = _observed(walk)
    marker_id = next(iter(compaction_ids))
    assert (
        observed.count(marker_id) == 1
    ), f"the compaction marker was served {observed.count(marker_id)} times"


def test_a_textless_multi_call_message_is_not_split_by_a_page_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cut must not land BETWEEN an assistant message's call rows.

    That message paints no row of its own, so its id is only ever a JOURNAL entry
    id. Resolving groups against the page's row ids alone therefore made every
    sibling its own group, the cut landed inside the message, and the cursor it
    handed over named the message — which serves only rows OLDER than it, leaving
    the first siblings unreachable for good. Reproduced by review round 2 at page
    sizes 2, 3 and 4 (two rows lost) and on a real 3,548-row journal at a 20-row
    page (three rows lost); 25-572 such messages sit in each of the 12 largest
    journals on this host.
    """
    config = tmp_path / "config"
    _build(config, textless_calls=3)
    for page in (2, 3, 4):
        walk = _walk(config, monkeypatch, page=page)
        observed = _observed(walk)
        expected = [row.id for row in _journal_rows(walk["directory"])]
        missing = [row_id for row_id in expected if row_id not in set(observed)]
        assert missing == [], f"page {page} stranded {missing}"
        assert len(observed) == len(set(observed)), f"page {page} served a row twice"


def test_a_cursor_on_a_colon_bearing_call_id_still_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call id can carry a colon of its own, so the row id carries two.

    ``call_00_x7:06c08d4847`` is the shape real providers mint (journal
    439818272d84). The cursor misses as given, and one narrowing step — by the
    last colon — names nothing, so the route answered ``([], False)`` and the
    client ended history with every older row still unserved. The walk below is
    the client's: it must reach the journal's own first row.
    """
    config = tmp_path / "config"
    ids = _build(config, colon_call=True)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)

    page, _ = _history_page(SESSION_ID, ids["colon_call_row"], 10)
    served = [row.id for row in page]
    assert served, "a cursor on a colon-bearing call id served nothing"
    # The message's OWN row comes back with it. The caller's id is one of that
    # message's call rows — which is exactly the capped projection's shape (the
    # cap drops a group's row and keeps its calls) — so the rows above the
    # cursor row inside the group are this page's to serve; nothing else will
    # ever ask for them (review round 3, R3-1).
    assert "coloncall" in served
    # …and the page crossed the cursor rather than answering end-of-history.
    assert any(row_id.startswith("early-") for row_id in served)

    walk = _walk(config, monkeypatch, page=5)
    observed = set(_observed(walk))
    expected = {row.id for row in _journal_rows(walk["directory"])}
    assert expected <= observed, f"unreachable rows: {sorted(expected - observed)[:4]}"


#: Row counts that put the cap's dropped row EXACTLY on a message's own row.
#: ``_cap_tail`` drops ``render[-PROJECTION_TRANSCRIPT_LIMIT]``, so the shape is
#: arithmetic rather than luck: 2 x 10 plain rows, then a text + 3-call message
#: (4 rows), then 2 x 38 plain — the dropped index is 100 - 80 = 20.
_CAPPED_TURNS_BEFORE = 10
_CAPPED_TURNS_AFTER = 38


def _build_capped(config: Path) -> dict[str, Any]:
    """A render whose cap DROPS a message's own row and keeps its call rows.

    The mobile projection makes room for its pinned opener by dropping the oldest
    row of the tail (``projection.py:_cap_tail``), so a message with tool calls
    can be split across the cap: the phone keeps ``m:call_0…`` and loses ``m``.
    Its oldest row is then a call row, that call row is its first cursor, and the
    reader has to serve the rows ABOVE it inside that message's group or they are
    in no page at all (review round 3, R3-1).
    """
    directory = config / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    ids: dict[str, Any] = {}

    async def run() -> None:
        for i in range(_CAPPED_TURNS_BEFORE):
            await transcript.append_message(Message.user(f"before {i}", id=f"capped-b{i}"))
            await transcript.append_message(Message.assistant(f"answer {i}", id=f"capped-ba{i}"))
        calls = [
            ToolCall(id=f"capped-call-{i}", name="bash", arguments={"i": "run the checks"})
            for i in range(3)
        ]
        await transcript.append_message(
            Message(
                role="assistant",
                content=[TextContent(text="the row the cap drops")],
                tool_calls=calls,
                stop_reason="toolUse",
                id="capped-own",
            )
        )
        for call in calls:
            await transcript.append_message(
                Message(
                    role="tool",
                    content=[TextContent(text="ok")],
                    tool_call_id=call.id,
                    tool_name="bash",
                    id=f"{call.id}-result",
                )
            )
        for i in range(_CAPPED_TURNS_AFTER):
            await transcript.append_message(Message.user(f"after {i}", id=f"capped-a{i}"))
            await transcript.append_message(Message.assistant(f"reply {i}", id=f"capped-aa{i}"))
        ids["own_row"] = "capped-own"
        ids["first_call_row"] = "capped-own:capped-call-0"

    asyncio.run(run())
    return ids


def test_a_capped_projection_does_not_strand_the_row_the_cap_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row the projection's cap drops has to come back on the first page.

    ``_cap_tail`` makes room for the pinned opener by dropping the OLDEST row of
    the tail. When that row is a message's own row and the kept tail begins at one
    of its call rows, the client's first cursor IS a call row: the reader resolved
    it to the message and served only rows strictly older than the message, so the
    dropped row sat above everything the walk could reach. Round 3 measured one
    stranded row on a real journal at page 20 AND at page 120, and 5 of the 14
    largest journals here carry the shape — while no fixture could show it, since
    every fixture's projection sits under the cap.
    """
    config = tmp_path / "config"
    ids = _build_capped(config)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)

    projection = _durable_projection(SESSION_ID)
    assert projection is not None
    seed = [row.id for row in projection.transcript]
    assert len(seed) == PROJECTION_TRANSCRIPT_LIMIT, "the fixture must sit AT the cap"
    assert ids["own_row"] not in seed, "the cap was supposed to drop the own row"
    assert ids["first_call_row"] in seed
    assert seed[1] == ids["first_call_row"], "the client's first cursor is the call row"

    for page in (20, 120):
        walk = _walk(config, monkeypatch, page=page)
        observed = set(_observed(walk))
        expected = {row.id for row in _journal_rows(walk["directory"])}
        assert ids["own_row"] in observed, f"page {page} stranded the dropped row"
        assert expected <= observed, f"page {page} stranded {sorted(expected - observed)[:3]}"


#: The R4-1 shape: the compaction entry is appended AFTER the start of its own kept
#: window, and that window paints more rows than the render's cap keeps.
_COMPACT_TURNS = 70
_COMPACT_KEEP_TURN = 30


def _build_capped_compaction(config: Path) -> dict[str, Any]:
    """A capped render whose NEWEST COMPACTION entry is newer than the cursor.

    The fold hoists the newest compaction's marker into its prefix, so the render
    paints it as its FIRST row — and ``ProjectionFold._cap_tail`` drops that row on
    any render past its cap, because it pins a USER row for a marker that renders
    as a ``notice``. The compaction ENTRY stays where it was written: an
    auto-compaction is appended after the start of its own kept window, so the
    rows between ``first_kept_entry_id`` and the entry are NEWER than the cursor
    the client sends, and no page's ``older`` half can ever contain it (review
    round 4, R4-1 — measured on two of eleven large real journals, one row each).
    Seventy turns keep the window at 82 painted rows so the cap has to drop it.
    """
    directory = config / "sessions" / SESSION_ID
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    ids: dict[str, Any] = {}

    async def run() -> None:
        for i in range(_COMPACT_TURNS):
            user = Message.user(f"turn {i}", id=f"cmp-u{i}")
            await transcript.append_message(user)
            await transcript.append_message(Message.assistant(f"reply {i}", id=f"cmp-a{i}"))
            if i == _COMPACT_KEEP_TURN:
                ids["kept"] = user.id
        entry = await transcript.append_compaction(
            summary="the boundary summary",
            first_kept_entry_id=ids["kept"],
            tokens_before=180_000,
        )
        ids["marker"] = entry.id

    asyncio.run(run())
    return ids


def test_a_capped_render_still_serves_the_newest_compactions_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The boundary marker has to outlive the cap that dropped it.

    The marker is the one row that tells a reader history continues behind the
    boundary, and on a freshly compacted session it is the only row the walk could
    not reach: the render's cap drops it, and its journal position — after the
    start of its own kept window — sits above the oldest row the client holds, so
    every page (which serves only entries strictly older than its anchor) skipped
    past it. The walk must serve it exactly once, on the page the client fetches
    first.
    """
    config = tmp_path / "config"
    ids = _build_capped_compaction(config)
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config)

    projection = _durable_projection(SESSION_ID)
    assert projection is not None
    seed = [row.id for row in projection.transcript]
    assert len(seed) == PROJECTION_TRANSCRIPT_LIMIT, "the render must sit AT its cap"
    assert ids["marker"] not in seed, "the cap was supposed to drop the marker"

    for page in (20, 120):
        walk = _walk(config, monkeypatch, page=page)
        observed = _observed(walk)
        expected = {row.id for row in _journal_rows(walk["directory"])}
        assert ids["marker"] in observed, f"page {page} stranded the boundary marker"
        assert expected <= set(
            observed
        ), f"page {page} stranded {sorted(expected - set(observed))[:3]}"
        served = [row for rows in walk["pages"] for row in rows]
        assert (
            served.count(ids["marker"]) == 1
        ), f"page {page} served the marker {served.count(ids['marker'])} times"

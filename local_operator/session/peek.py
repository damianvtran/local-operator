"""Bounded transcript peek: the pure readers behind ``sessions op='peek'``.

WHY THIS MODULE EXISTS. The readers (tail/head/cursor walks, the bounded
needle search, the digest fold) used to live inside ``tools/builtin.py``, which
meant only the sessions TOOL on the device holding a journal could run them. A
coordinator peeking a session on a mesh peer therefore got a degraded tail of
the synced display window and was refused ``head``/``query``/``regex``/
``digest`` outright — it could not see a remote session's original request or
search its history. The owner device now answers ``net_session_peek``
(``network/peek.py``) with THESE functions, so a remote peek and a local one
run the same code under the same budgets, and the rows cannot drift apart.

CONSTRAINTS. Everything here is synchronous file I/O meant to run in a worker
thread (the tool's ``asyncio.to_thread``; the relay's slow lane). Nothing here
imports the tool layer: the relay imports this module, and ``builtin`` is far
too heavy (and too circular) for a relay to pay for. Every read is bounded by
``PEEK_MAX_STEPS`` (``harness.comms``), ``_PEEK_MAX_PAGES`` page reads, and the
``_PEEK_SCAN_BYTES`` search depth; :func:`peek_validation_error` is the one
validator both the local tool and the owner-side handler apply, so an
over-budget request refuses with the same sentence on either path.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, AbstractSet, Any, NamedTuple, Protocol

if TYPE_CHECKING:
    from local_operator.session.transcript import TranscriptEntry


class PeekRequest(Protocol):
    """The fields a peek reads — ``SessionsParams`` satisfies it, and so does
    :class:`WirePeekRequest`, which is what the owner rebuilds from a frame."""

    # Read-only properties, so a frozen dataclass and a pydantic model both
    # satisfy it (a plain attribute in a Protocol demands writability).
    @property
    def steps(self) -> int | None: ...
    @property
    def head(self) -> int | None: ...
    @property
    def before_id(self) -> str | None: ...
    @property
    def around_id(self) -> str | None: ...
    @property
    def query(self) -> str | None: ...
    @property
    def regex(self) -> bool | None: ...
    @property
    def digest(self) -> bool | None: ...


#: The window selectors, mutually exclusive (see :func:`peek_validation_error`).
PEEK_WINDOW_FIELDS = ("steps", "head", "before_id", "around_id")


def peek_validation_error(params: PeekRequest, given: AbstractSet[str]) -> str | None:
    """``peek``'s window rules (design §3.2): one window, and windows that fit.

    The window selectors are mutually exclusive because each names a DIFFERENT
    place to read from; two at once describe no window this tool could honour,
    and honouring one silently is the substitution the transcript reader's own
    ``validate_page_request`` exists to refuse one level down. ``digest`` is a
    different READ (a fold of the newest rows), so a window beside it would be
    dropped on the floor, and a search beside it would fight over what the
    window means — both refused for the reason the op table refuses stray
    fields at all: a misspelled intent must not be silently ignored.
    """
    from local_operator.harness.comms import PEEK_MAX_STEPS

    windows = [field for field in PEEK_WINDOW_FIELDS if field in given]
    if len(windows) > 1:
        spelled = ", ".join(f"`{field}`" for field in windows)
        return (
            "peek reads one window at a time: pass at most one of `steps`, `head`, "
            f"`before_id`, `around_id` (got {spelled})."
        )
    for field in ("steps", "head"):
        value = getattr(params, field)
        if field in given and (value is None or value < 1):
            return f"`{field}` needs a count of at least 1."
        if value is not None and value > PEEK_MAX_STEPS:
            return (
                f"`{field}={value}` is too many for one peek (max {PEEK_MAX_STEPS}); "
                "page through with `before_id` instead."
            )
    for field in ("before_id", "around_id"):
        if field in given and not str(getattr(params, field) or "").strip():
            return f"`{field}` needs the entry id an earlier peek returned."

    query = str(params.query or "").strip()
    # Hoisted ABOVE the digest branch (round-1 review, R-1): `digest=true
    # regex=true` was the one regex-without-query shape that validated clean
    # and then dropped the flag on the floor. `regex` selects how `query`
    # matches, so it needs one wherever it appears.
    if params.regex and not query:
        return "`regex` needs `query`: it selects how the query matches."
    if params.digest:
        if windows:
            spelled = ", ".join(f"`{field}`" for field in windows)
            return f"`digest` folds the newest rows on its own; drop {spelled}."
        if query:
            return (
                "`digest` and `query` are different reads — digest folds the tail, "
                "`query` locates one step. Pass one."
            )
        return None
    # Derived from the shared tuple, so a fifth window field lands in this
    # refusal by construction; `steps` is the one that SURVIVES beside a query
    # (it sizes the match window).
    non_steps = [field for field in PEEK_WINDOW_FIELDS if field in given and field != "steps"]
    if query and non_steps:
        return (
            "`query` locates one step, so the window is `steps=N` around the match "
            "(default 12): drop `head`/`before_id`/`around_id` or drop `query`."
        )
    return None


#: Steps a peek shows when the caller names no count (design §8.2). The hard
#: ceiling is ``comms.PEEK_MAX_STEPS`` (50, imported where it is used) — the
#: same number ``hub``'s peek refuses with, so both peeks turn an absurd count
#: into the same legible refusal instead of two private ideas of "too many".
_PEEK_DEFAULT_STEPS = 12


#: How many ROWS one page read asks for while hunting ``need`` steps. A step is
#: one message row; a page also carries the bookkeeping rows (checkpoints, todo
#: snapshots, spend records) that render as nothing, so the ask is a multiple
#: of the step count and the walk reads another page only when density is worse
#: than that. Both terms are deliberate: too small and a dense tail pays a
#: second page read; too large and the first read stops being "the chunk that
#: carries the window" that ``read_transcript_page`` prices at.
#: ``_PEEK_MAX_PAGES`` bounds the walk itself: the ordinary cost is ONE read,
#: and a journal whose tail is a wall of bookkeeping rows still terminates —
#: the footer's cursor is then the honest continuation point.
_PEEK_MAX_PAGES = 4


#: Byte depth the backward needle search may scan before it returns the honest
#: miss (design §8.1). It is the SAME 16 MiB window the reader gives its cursor
#: locator (``transcript._PAGE_LOCATE_WINDOW_BYTES``), so "deep" means one
#: thing in the reader and in the search that rides it; a test pins the two
#: numbers equal so they cannot drift apart silently.
_PEEK_SCAN_BYTES = 16 << 20


#: Rows a digest reads from the tail (design §8.3's "~40 entries").
_PEEK_DIGEST_ROWS = 40


def _peek_page_rows(need: int) -> int:
    """Entries per page read for a ``need``-step walk — see the constants."""
    return min(400, max(48, need * 4))


def _peek_short_id(entry_id: str) -> str:
    """First 8 chars of an entry id, for HEADERS only.

    Never for the footer's cursors: those must be pasted back verbatim, and a
    truncated cursor is the one suggestion this surface must not make.
    """
    return entry_id[:8] + "…" if len(entry_id) > 8 else entry_id


class _PeekWindow(NamedTuple):
    """One rendered step window — or the honest miss that answered a search.

    ``steps`` is oldest→newest and numbered 1..N WITHIN THE WINDOW: operator
    journals are not small (262 MB measured), so there is no counted total to
    number against — cursor ids and booleans, not absolute indices (design
    §8.2/§14.3). ``miss`` is the search's sentence when the needle was not
    found inside the scan budget; a window with steps and a miss never occurs.
    The cursor fields carry ENTRY ids (not step ids): stable across compaction,
    which byte offsets are not, and exclusive/inclusive exactly as
    ``read_transcript_page`` documents them.
    """

    steps: list[Any]
    has_older: bool = False
    has_newer: bool | None = None
    before_cursor: str | None = None
    after_cursor: str | None = None
    scanned_bytes: int | None = None
    match_id: str | None = None
    match_step: int | None = None
    miss: str = ""
    note: str = ""


class _PeekDigest(NamedTuple):
    """The fold ``peek digest=true`` returns, already rendered."""

    text: str
    steps_seen: int
    tool_calls: int
    has_older: bool


def _peek_message_entries(entries: Sequence[TranscriptEntry]) -> list[TranscriptEntry]:
    """The entries that RENDER as steps — the renderer's own filter.

    ``_render_transcript_steps`` numbers only the rows it renders, so any step
    arithmetic must count with the SAME predicate; keeping it in one place is
    what stops the walk's window from disagreeing with the renderer's count.
    Since round 1 of #3098 that predicate also refuses the account-change
    notice (``harness/comms.py`` ``is_peek_step_entry``: its sentence names the
    account, and peek lands in model context), so this function must drop it
    too — otherwise the window would reserve a slot for a step that is never
    rendered and every cursor computed from these rows would sit one step off.
    """

    from local_operator.harness.comms import is_peek_step_entry

    return [entry for entry in entries if is_peek_step_entry(entry)]


def _peek_window(
    entries: list[TranscriptEntry],
    *,
    need: int,
    has_older: bool,
    has_newer: bool | None,
    match_id: str | None = None,
    scanned_bytes: int | None = None,
    note: str = "",
) -> _PeekWindow:
    """Assemble the SHOWN window from chronological entries — the last ``need``
    steps, which is the newest end for every walk this tool runs (tail reads,
    ``before_id`` pages, and search windows all page backward from a newer
    edge; the head read passes its own prefix where the same final slice lands
    on the same rows because it IS the prefix).

    ``has_older`` is true when the reader's stop page says rows follow below
    the page OR when anything was read below the oldest SHOWN step — a
    bookkeeping row under the window is still a row the reader has not seen
    (design §8.2: a ``False`` must be a fact, never just "the window ran out").
    """

    from local_operator.harness.comms import PeekStep, _render_transcript_steps

    message_entries = _peek_message_entries(entries)
    shown_messages = message_entries[-need:]
    oldest_shown: TranscriptEntry | None = shown_messages[0] if shown_messages else None
    older_row_below = False
    if oldest_shown is not None:
        for entry in entries:
            if entry is oldest_shown:
                break
            older_row_below = True
    steps = _render_transcript_steps(entries)
    shown = steps[-need:] if len(steps) > need else steps
    # Renumber 1..N within the window: the renderer numbers every row it was
    # given, so a window that dropped older steps would otherwise start at 5.
    # The same slice re-bases ``match_step`` (computed against the SHOWN
    # messages below), keeping the marker pointed at the step it names.
    if shown and shown[0].index != 1:
        offset = shown[0].index - 1
        shown = [
            PeekStep(step.index - offset, step.kind, step.heading, step.body) for step in shown
        ]
    match_step = None
    if match_id is not None:
        for position, entry in enumerate(shown_messages, start=1):
            if entry.id == match_id:
                match_step = position
                break
    # The cursors page from the SHOWN steps, never from rows the walk read past
    # while hunting for them: a `before_id` cursor into a row the caller never
    # saw makes the next page SKIP everything between the two (a dense 48-row
    # page would skip 36 message rows). The reader's cursor is exclusive, so
    # "oldest shown entry" continues exactly where the window stopped — no
    # gap, no repeat. Only an empty window falls back to the read rows, and
    # then only for the has_older case, where there is nothing to continue from.
    if shown_messages:
        before_cursor: str | None = shown_messages[0].id
        after_cursor: str | None = shown_messages[-1].id
    else:
        before_cursor = entries[0].id if entries else None
        after_cursor = entries[-1].id if entries else None
    return _PeekWindow(
        steps=shown,
        has_older=has_older or older_row_below,
        has_newer=has_newer,
        before_cursor=before_cursor,
        after_cursor=after_cursor,
        scanned_bytes=scanned_bytes,
        match_id=match_id,
        match_step=match_step,
        note=note,
    )


def _walk_steps(directory: Path, *, cursor: str | None, need: int) -> _PeekWindow | str:
    """Collect up to ``need`` steps walking backward from ``cursor`` (or EOF).

    Reads with ``read_transcript_page`` — the page reader the desktop uses and
    the one whose cost contract ("the tail page costs the chunk that carries
    it") this walk keeps — one page at a time until ``need`` message rows are
    collected, the journal's start is reached, or ``_PEEK_MAX_PAGES`` reads are
    spent. ``before_id`` is the reader's exclusive cursor, so consecutive pages
    can neither repeat nor skip a row.

    Returns the rendered window, or the sentence for a caller-named cursor that
    names nothing: the reader answers that case by reconciling to the current
    TAIL, and this tool refuses instead — a tail under a caller's cursor is the
    silent substitution the reader's own ``validate_page_request`` refuses one
    level down, and the caller may then re-ask with an id a real read returned.
    """

    from local_operator.session.transcript import read_transcript_page

    collected: list[TranscriptEntry] = []
    read_cursor = cursor
    stop_page = None
    pages = 0
    while True:
        page = read_transcript_page(directory, before_id=read_cursor, limit=_peek_page_rows(need))
        if read_cursor is not None and page.reconciled:
            return (
                f"no transcript entry with id {read_cursor!r} — entry ids are stable "
                "across compaction, but a replaced journal drops old ones; use an id "
                "an earlier peek returned, or read the tail."
            )
        collected = list(page.entries) + collected
        stop_page = page
        pages += 1
        if len(_peek_message_entries(collected)) >= need or not page.has_more:
            break
        if pages >= _PEEK_MAX_PAGES or not page.entries:
            break
        read_cursor = page.entries[0].id
    assert stop_page is not None  # the loop body always runs at least once
    return _peek_window(
        collected,
        need=need,
        # The stop page's ``has_more`` is the reader's own answer to "older rows
        # exist"; ``_peek_window`` adds the bookkeeping rows read below the
        # window. ``has_newer``: the tail read starts at the file's end, so
        # nothing is newer (False is a fact here, not a bound); a ``before_id``
        # window has every row above it newer BY CONSTRUCTION — the cursor row
        # itself (the reader's own docstring makes the same two claims).
        has_older=bool(stop_page.has_more),
        has_newer=False if cursor is None else True,
        note=(
            "the bounded walk stopped before it collected every requested step; "
            "continue with `before_id` below"
            if (
                pages >= _PEEK_MAX_PAGES
                and stop_page.has_more
                and len(_peek_message_entries(collected)) < need
            )
            else ""
        ),
    )


def _peek_head(directory: Path, *, need: int) -> _PeekWindow:
    """First ``need`` steps, read forward from byte zero and STOPPING.

    The transcript module's forward walker is a generator exactly so the
    caller's stop is the bound: this consumes rows until it holds ``need``
    message rows (or the journal ends, or ``_PEEK_SCAN_BYTES`` of rows have
    been read — the search's own honest cap, said out loud in the footer).
    A head read must not parse a whole journal to answer "how did this start"
    (design §8.2/§11.8); one lookahead row beyond the window answers
    ``has_newer`` the way the anchored page's lookahead answers it.
    """

    from local_operator.session import compression
    from local_operator.session.transcript import (
        TranscriptEntry,
        _iter_complete_lines_forward,
    )

    # A seekable, whole transcript whichever form it is on disk in: the plain
    # journal, or the archive decompressed into compression's bounded cache. A
    # peek is a READ, so it must never be answered from the plain name alone.
    path = compression.transcript_for_read(directory)
    if path is None:
        raise FileNotFoundError(f"{directory} holds no transcript")
    entries: list[TranscriptEntry] = []
    scanned = 0
    capped = False
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        end_of_file = handle.tell()
        walker = _iter_complete_lines_forward(handle, 0, end_of_file=end_of_file)
        for raw in walker:
            scanned += len(raw) + 1
            entry = TranscriptEntry.from_json(raw.decode("utf-8", errors="replace"))
            if entry is not None:
                entries.append(entry)
            if len(_peek_message_entries(entries)) >= need:
                break
            if scanned >= _PEEK_SCAN_BYTES:
                capped = True
                break
        # One lookahead beyond the window answers ``has_newer``; blank lines are
        # not rows, so skip them rather than counting a torn journal's noise.
        has_newer = any(raw.strip() for raw in walker)
    return _peek_window(
        entries,
        need=need,
        has_older=False,  # the walk began at byte zero; nothing precedes it
        has_newer=has_newer,
        scanned_bytes=scanned,
        note=(
            f"the read stopped at the {_PEEK_SCAN_BYTES >> 20} MiB scan bound before "
            "collecting every requested step"
            if capped
            else ""
        ),
    )


def _peek_around(
    directory: Path, *, anchor: str, need: int, mark: str | None = None
) -> _PeekWindow | str:
    """The anchored window around one entry id — the shape a cursor call and a
    search hit share (design §8.1: a match renders through the anchored page).

    One page read: ``before`` older rows, the anchor, ``after`` newer rows, with
    the reader's own one-row-lookahead booleans for both edges — so "older"/
    "newer" are facts, not window lengths. ``mark`` is the entry a search hit on
    (the anchor itself, in that call): it picks the ``← match`` marker and, when
    the matched row is not a message row, deliberately marks NO step — the
    match is then named by id in the footer only.
    """

    from local_operator.session.transcript import read_transcript_page

    before = need // 2
    after = max(0, need - before - 1)
    page = read_transcript_page(directory, around_id=anchor, before=before, after=after, limit=need)
    if page.reconciled and not page.entries:
        return (
            f"no transcript entry with id {anchor!r} — entry ids are stable across "
            "compaction, but a replaced journal drops old ones; use an id an earlier "
            "peek returned."
        )
    entries = list(page.entries)
    return _peek_window(
        entries,
        need=need,
        has_older=bool(page.has_more),
        has_newer=page.has_newer,
        match_id=mark,
    )


def _peek_search(
    directory: Path, *, needle: str, pattern: re.Pattern[str] | None, need: int
) -> _PeekWindow | str:
    """Locate a needle walking backward from EOF, then window around the hit.

    The walk decodes rows newest-first and tests each row's text (a literal
    substring when ``pattern`` is ``None``, else the pre-compiled
    ``pattern.search`` — the design's own two modes); it stops at the first
    hit or once the row walk has crossed `_PEEK_SCAN_BYTES` (row-granular —
    see the counter's comment for the one-row caveat). The budget is the
    point: the reader's docstring measured what a deep walk costs on a 262 MB
    journal, so this never scans one silently — a miss is reported honestly,
    with the pointer the caller needs to widen from a known position. The
    window itself is the same anchored read ``around_id`` runs, so a hit and a
    cursor call render one shape.

    ``str`` return: an anchor that vanished between the walk and the window
    read (a compaction between the two reads, reported as the miss it is).
    """

    from local_operator.harness.comms import is_peek_step_entry
    from local_operator.session import compression
    from local_operator.session.transcript import (
        TranscriptEntry,
        _iter_complete_lines_backward,
    )

    # Same resolution as ``_peek_head``: the seekable source, archive included.
    path = compression.transcript_for_read(directory)
    if path is None:
        raise FileNotFoundError(f"{directory} holds no transcript")
    scanned = 0
    hit_id = ""
    budget_stop = False
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        end_of_file = handle.tell()
        for _chunk_start, lines in _iter_complete_lines_backward(handle, end_of_file):
            for raw in lines:
                if not raw.strip():
                    continue
                # Depth is counted ROW by ROW — the bytes walked past, newline
                # included — not per chunk read, because a chunk-granular
                # count breaks on any journal smaller than one chunk. Caveat,
                # deliberate: the budget is checked BEFORE the next row is
                # added, so the row that crosses it is still walked and
                # counted — `scanned_bytes` can overshoot `_PEEK_SCAN_BYTES`
                # by up to ONE row's length (transcript.py measures rows of
                # tens of MB; the miss test tolerates +2,000 for its ~950-byte
                # rows). Under that, a hit reports its true depth and a miss
                # reports the budget.
                if scanned >= _PEEK_SCAN_BYTES:
                    budget_stop = True
                    break
                scanned += len(raw) + 1
                text = raw.decode("utf-8", errors="replace")
                # A REFUSED ROW CANNOT MATCH (QA round 1, Q1). The step filter is
                # what keeps the account label out of a peek — and testing the raw
                # JSON first made the FILTER itself an oracle: a query for a
                # substring of a label this surface will never print answered
                # yes/no, and 513 of those rebuilt an email. So the row is parsed
                # and gated BEFORE the needle is tested, and a refused row can
                # neither match nor become an anchor. The cost is one parse per
                # row already inside the byte budget, which is the same decode
                # this loop was doing anyway.
                entry = TranscriptEntry.from_json(text)
                if entry is None or not is_peek_step_entry(entry):
                    continue
                if pattern is None:
                    found = needle in text
                else:
                    found = pattern.search(text) is not None
                if found:
                    hit_id = entry.id
                    break
            if hit_id or budget_stop:
                break
    if not hit_id:
        if end_of_file == 0:
            return _PeekWindow(
                steps=[],
                scanned_bytes=0,
                miss=f"no match for {needle!r}: this transcript is empty.",
            )
        whole = not budget_stop
        where = "this transcript" if whole else f"the last {scanned:,} bytes of this transcript"
        return _PeekWindow(
            steps=[],
            scanned_bytes=scanned,
            miss=(
                f"no match for {needle!r} in {where} — the search is bounded on "
                "purpose and never scans an operator-size journal silently; widen "
                "it by starting from a known entry id with `peek around_id=<id>`."
            ),
        )
    window = _peek_around(directory, anchor=hit_id, need=need, mark=hit_id)
    if isinstance(window, str):
        return window  # the anchor vanished between the walk and the window read
    return window._replace(scanned_bytes=scanned)


def _peek_live_state(record: Any) -> str:
    """``stored``, or ``live`` plus the record's own busy/pending words.

    Reads the SAME fields the listings paint (``SessionRecord.busy``,
    ``.pending``), so the digest's state line cannot drift from the sidebar's.
    """

    if record is None:
        return "stored"
    bits: list[str] = []
    if getattr(record, "busy", False):
        bits.append("busy")
    pending = getattr(record, "pending", None)
    if pending:
        bits.append(f"pending {pending}")
    return "live" + (f", {', '.join(bits)}" if bits else "")


def _fold_line(text: str, limit: int) -> str:
    """One digest line: the first non-empty line, collapsed, clipped to ``limit``.

    "The newest assistant line" is taken literally — a multi-line message
    contributes its opening line, not a paragraph folded into a run-on where
    the assistant's prose and its tool-call list run together. Head-only with
    an ellipsis: the full step is one `peek steps=` away, and a fold that kept
    head AND tail per line would print more of the transcript than the digest
    exists to avoid.
    """

    flat = ""
    for raw in text.splitlines():
        flat = " ".join(raw.split())
        if flat:
            break
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _peek_digest_fold(steps: list[Any], *, has_older: bool, live_state: str) -> _PeekDigest:
    """Fold a tail window of steps to ≤10 lines (design §8.3).

    The LINE COUNT is an invariant, not a hope: the header, the newest ask and
    the newest assistant line each take one line, and the tool tail keeps the
    last calls that fit the remaining budget (never more than eight). Counts
    are per kind as §8.3 pins them; other kinds (hub traffic, system notes)
    are counted in the total and folded into one "+N more" only when present.
    """

    counts = {"user": 0, "assistant": 0, "tool": 0}
    others = 0
    for step in steps:
        if step.kind in counts:
            counts[step.kind] += 1
        else:
            others += 1
    parts = [f"user {counts['user']}", f"assistant {counts['assistant']}", f"tool {counts['tool']}"]
    if others:
        parts.append(f"+{others} more")
    older = "more above" if has_older else "whole transcript"
    lines = [f"digest: {len(steps)} steps seen ({', '.join(parts)}) — {live_state}; {older}"]

    ask = next((step for step in reversed(steps) if step.kind == "user"), None)
    if ask is not None:
        lines.append(f"ask: {_fold_line(ask.body, 80) or ask.heading}")
    reply = next((step for step in reversed(steps) if step.kind == "assistant"), None)
    if reply is not None:
        lines.append(f"assistant: {_fold_line(reply.body, 120) or reply.heading}")

    calls = [step for step in steps if step.kind == "tool"][-8:]
    room = max(0, 10 - len(lines))
    for step in calls[-min(8, room) :] if room else []:
        base, _, rest = step.heading.partition(" result")
        label = base + (" (error)" if "error" in rest else "")
        summary = _fold_line(step.body, 60)
        lines.append(f"tool: {label} · {summary}" if summary else f"tool: {label}")
    return _PeekDigest(
        text="\n".join(lines),
        steps_seen=len(steps),
        tool_calls=counts["tool"],
        has_older=has_older,
    )


def _peek_digest(directory: Path, *, live_state: str) -> _PeekDigest:
    """Read the tail window and fold it — the same bounded reader as `steps`."""

    from local_operator.harness.comms import _render_transcript_steps
    from local_operator.session.transcript import read_transcript_page

    page = read_transcript_page(directory, limit=_PEEK_DIGEST_ROWS)
    return _peek_digest_fold(
        _render_transcript_steps(list(page.entries)),
        has_older=bool(page.has_more),
        live_state=live_state,
    )


def _peek_body(
    name: str,
    session_id: str,
    mode: str,
    window: _PeekWindow,
    *,
    anchor: str = "",
    query: str = "",
) -> str:
    """One window's text: header, steps, and the footer's continuation hints.

    The shape mirrors ``hub``'s peek (``_hub_peek``: header, numbered steps with
    indented bodies, one parenthesized footer) with the two deviations §8.2
    names: no counted total — step numbers are window-relative — and hints that
    are cursor ids. Footer cursors are printed in FULL: they are meant to be
    pasted back, and a truncated id names nothing.
    """

    phrase = {
        "tail": "from the tail",
        "head": "from the start",
        "before": f"immediately before {_peek_short_id(anchor)}",
        "around": f"centred on {_peek_short_id(anchor)}",
        "search": f"around the match for {_fold_line(query, 40)!r}",
    }.get(mode, mode)
    if not window.steps:
        if mode == "tail":
            return f'"{name}" ({session_id}) — no steps yet: the transcript has no ' "message rows."
        return f'"{name}" ({session_id}) — no steps {phrase}.'
    lines = [f'"{name}" ({session_id}) — {len(window.steps)} step(s) {phrase}:']
    for step in window.steps:
        marker = "  ← match" if step.index == window.match_step else ""
        lines.append(f"{step.index:>4}  {step.heading}{marker}")
        if step.body:
            for body_line in step.body.splitlines():
                lines.append(f"      {body_line}")
    hints: list[str] = []
    if window.match_id:
        hints.append(f"match {window.match_id}")
    if window.scanned_bytes is not None:
        hints.append(f"scanned {window.scanned_bytes:,} bytes")
    if window.has_older and window.before_cursor:
        hints.append(f"before_id={window.before_cursor} for earlier")
    if window.has_newer and window.after_cursor and mode in ("head", "around", "search"):
        hints.append(f"around_id={window.after_cursor} for later")
    if window.note:
        hints.append(window.note)
    if hints:
        lines.append("(" + "; ".join(hints) + ")")
    return "\n".join(lines)


def _peek_read(
    directory: Path,
    params: PeekRequest,
    *,
    need: int,
    pattern: re.Pattern[str] | None,
) -> tuple[_PeekWindow | str, str]:
    """Run the one read the request describes; return ``(window|refusal, mode)``.

    The order is the design's two-level location: a ``query`` locates a step
    (the store-level search stays `list`'s job — this addresses ONE session),
    and the window selectors are the rest. Validation has already refused
    combinations; this only picks the single remaining one. ``pattern`` is the
    request's compiled regex (``None`` = literal mode), compiled by the caller
    so a malformed one was classified as a fault before the thread started.
    """

    query = str(params.query or "").strip()
    if query:
        return _peek_search(directory, needle=query, pattern=pattern, need=need), "search"
    if params.head is not None:
        return _peek_head(directory, need=need), "head"
    if params.before_id is not None:
        return _walk_steps(directory, cursor=params.before_id, need=need), "before"
    if params.around_id is not None:
        return _peek_around(directory, anchor=params.around_id, need=need), "around"
    return _walk_steps(directory, cursor=None, need=need), "tail"


# ---------------------------------------------------------------------------
# The owner-side half of a REMOTE peek (``net_session_peek``)
# ---------------------------------------------------------------------------

#: Longest ``query`` a PEER may hand this device as a regex. A peer-supplied
#: pattern runs on THIS machine's CPU against THIS machine's journal, and
#: Python's ``re`` has no timeout, so the remote path bounds the pattern itself
#: on top of the walk's ``_PEEK_SCAN_BYTES``. 256 characters is far longer than
#: any needle a coordinator types ("original request", a ticket id, a SHA) and
#: short enough that the nested-quantifier check below reads the whole thing.
#: The LOCAL tool does not apply it: there the caller already owns the CPU.
PEER_REGEX_MAX_CHARS = 256

#: ``(…+)+``-shaped patterns — a quantified group whose body is itself
#: quantified — are the classic catastrophic-backtracking shape. Refusing them
#: on the peer path costs a coordinator nothing it needs (a literal ``query``
#: still works) and keeps one hostile or careless pattern from pinning an owner
#: thread no deadline can cancel.
_NESTED_QUANTIFIER = re.compile(r"\((?:[^()\\]|\\.)*[+*}](?:[^()\\]|\\.)*\)\s*(?:[+*]|\{\d)")


@dataclass(frozen=True)
class WirePeekRequest:
    """A peek request rebuilt from a ``net_session_peek`` frame.

    ``given`` mirrors the tool's ``model_fields_set``: validation refuses on
    what the CALLER named, not on values, so the owner must know which keys the
    frame actually carried.
    """

    steps: int | None = None
    head: int | None = None
    before_id: str | None = None
    around_id: str | None = None
    query: str | None = None
    regex: bool = False
    digest: bool = False
    given: frozenset[str] = frozenset()


#: The request keys a ``net_session_peek`` frame may carry (beside
#: ``session_id``/``req``/``op``). Anything else is ignored rather than refused
#: so a newer viewer's optional field never breaks an older owner.
WIRE_PEEK_FIELDS = ("steps", "head", "before_id", "around_id", "query", "regex", "digest")


def wire_request(params: PeekRequest, given: AbstractSet[str]) -> dict[str, Any]:
    """The frame fields for one peek — only the keys the caller named."""

    out: dict[str, Any] = {}
    for name in WIRE_PEEK_FIELDS:
        if name in given:
            out[name] = getattr(params, name)
    return out


def request_from_wire(frame: Mapping[str, Any]) -> WirePeekRequest | str:
    """Rebuild and TYPE-CHECK a frame's request; a refusal sentence on bad types.

    A granted peer is still untrusted input: a string where a count belongs must
    be a refusal, never a ``TypeError`` on the owner's link thread.
    """

    given = frozenset(name for name in WIRE_PEEK_FIELDS if name in frame)
    values: dict[str, Any] = {}
    for name in ("steps", "head"):
        if name in given:
            raw = frame.get(name)
            if raw is not None and (isinstance(raw, bool) or not isinstance(raw, int)):
                return f"`{name}` must be a whole number."
            values[name] = raw
    for name in ("before_id", "around_id", "query"):
        if name in given:
            raw = frame.get(name)
            if raw is not None and not isinstance(raw, str):
                return f"`{name}` must be text."
            values[name] = raw
    for name in ("regex", "digest"):
        if name in given:
            values[name] = bool(frame.get(name))
    return WirePeekRequest(**values, given=given)


def compile_peek_pattern(params: PeekRequest, *, remote: bool) -> re.Pattern[str] | None | str:
    """The request's regex, ``None`` for literal mode, or a refusal sentence.

    The STRIPPED query is compiled, the same spelling every other consumer
    strips to (round-2 review of the local tool, R-6). ``remote=True`` adds the
    peer-path bounds documented on :data:`PEER_REGEX_MAX_CHARS`.
    """

    if not params.regex:
        return None
    query = str(params.query or "").strip()
    if remote:
        if len(query) > PEER_REGEX_MAX_CHARS:
            return (
                f"a `regex` query from another device is capped at "
                f"{PEER_REGEX_MAX_CHARS} characters (got {len(query)}); shorten it, or "
                "search literally without `regex`."
            )
        if _NESTED_QUANTIFIER.search(query):
            return (
                "a `regex` query from another device may not nest quantifiers "
                "(`(a+)+`-shaped patterns can run without bound); simplify it, or "
                "search literally without `regex`."
            )
    try:
        return re.compile(query)
    except re.error as exc:
        return f"invalid regex '{query}': {exc}"


def peek_need(params: PeekRequest) -> int:
    """Steps the window holds: ``steps``, else ``head``, else the default."""

    if params.steps is not None:
        return params.steps
    if params.head is not None:
        return params.head
    return _PEEK_DEFAULT_STEPS


def window_to_wire(window: _PeekWindow) -> dict[str, Any]:
    """A window as JSON — every field, so the viewer re-renders it byte-for-byte."""

    out = window._asdict()
    out["steps"] = [
        {"index": step.index, "kind": step.kind, "heading": step.heading, "body": step.body}
        for step in window.steps
    ]
    return out


def window_from_wire(raw: Mapping[str, Any]) -> _PeekWindow:
    """The inverse of :func:`window_to_wire` (tolerant of absent optional keys)."""

    from local_operator.harness.comms import PeekStep

    steps = [
        PeekStep(
            index=int(step.get("index") or 0),
            kind=str(step.get("kind") or ""),
            heading=str(step.get("heading") or ""),
            body=str(step.get("body") or ""),
        )
        for step in raw.get("steps") or []
        if isinstance(step, Mapping)
    ]
    fields: dict[str, Any] = {name: raw.get(name) for name in _PeekWindow._fields if name in raw}
    fields["steps"] = steps
    fields["miss"] = str(fields.get("miss") or "")
    fields["note"] = str(fields.get("note") or "")
    return _PeekWindow(**fields)


def serve_peek(
    directory: Path, params: PeekRequest, given: AbstractSet[str], *, live_state: str
) -> dict[str, Any]:
    """Answer one REMOTE peek from this device's own journal (owner side).

    Returns ``{"mode": ..., "window": {...}}``, ``{"mode": "digest", "digest":
    {...}}``, or ``{"refusal": "<sentence>"}`` — the SAME sentences the local
    tool refuses with, because validation and the readers are the same
    functions. ``FileNotFoundError`` propagates: the caller owns the wording for
    a session with no transcript yet, exactly like the local tool does.
    """

    refusal = peek_validation_error(params, given)
    if refusal is not None:
        return {"refusal": refusal}
    pattern = compile_peek_pattern(params, remote=True)
    if isinstance(pattern, str):
        return {"refusal": pattern}
    if params.digest:
        digest = _peek_digest(directory, live_state=live_state)
        return {"mode": "digest", "digest": digest._asdict()}
    window, mode = _peek_read(directory, params, need=peek_need(params), pattern=pattern)
    if isinstance(window, str):
        return {"refusal": window, "mode": mode}
    return {"mode": mode, "window": window_to_wire(window)}

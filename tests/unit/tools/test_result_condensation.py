"""Tool results that re-bill what the model already has, condensed at the source.

Each result here is appended to the transcript verbatim and re-sent on every
later turn, so its size is paid many times over. The fleet measurements behind
each bound (2026-10-08, 1,143 transcripts) are cited at the constant that sets
it; these tests pin the SHAPES, and that nothing actionable was hidden:

- ``hub op=list`` folds old completed children into a count plus a handle, and
  states the transcript how-to once instead of under every row;
- ``todo`` mutation receipts carry counts and clipped names, while ``view`` and
  the partial-match error keep the full texts the model must echo exactly;
- ``secret op=list`` clips descriptions, keeps every name, spills the full list;
- ``jobs op=peek`` returns only the latest frame of a redrawing watcher;
- ``read spill://…?q=`` no longer returns whole multi-hundred-KB lines.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.types import AbortSignal, AgentTool, ToolContext, ToolResult
from local_operator.tools import builtin
from local_operator.tools.builtin import (
    HUB_LIST_COMPLETED_SHOWN,
    READ_OUTPUT_LIMIT_CHARS,
    _collapse_refreshing_frames,
    _header_candidates,
    _hub_list,
)
from local_operator.tools.registry import create_tools
from local_operator.tools.spill import get_store


def _text(result: ToolResult) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


async def _call(
    tools: dict[str, AgentTool], name: str, args: dict[str, Any], context: ToolContext
) -> ToolResult:
    return await tools[name].execute("call-1", args, None, None, context)  # type: ignore[operator]


@pytest.fixture
def isolated_spill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The spill store resolves under the config dir per call; keep it here."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    return tmp_path


# --------------------------------------------------------------------------- hub


class _Row:
    """The slice of ``ChildInfo`` the roster renderer reads."""

    def __init__(self, job_id: str, status: str, *, session_id: str | None = None) -> None:
        self.job_id = job_id
        self.label = f"label-{job_id}"
        self.status = status
        self.resumable = status != "running"
        self.age_s = 100.0
        self.detail = None
        self.cut_off_cause = ""
        self.session_id = session_id
        self.last_progress_at = None
        self.ended_by = ""
        self.cancel_reason = ""


class _Comms:
    def __init__(self, rows: list[_Row]) -> None:
        self._rows = rows

    def roster(self) -> list[_Row]:
        return self._rows


def test_hub_list_folds_old_completed_children_and_keeps_every_live_one(isolated_spill):
    rows = [_Row(f"done{i:03d}", "completed", session_id=f"s{i:03d}") for i in range(40)]
    rows.insert(5, _Row("failedjob", "failed", session_id="sfail"))
    rows.append(_Row("livejob", "running"))
    result = _hub_list("c", _Comms(rows), None, ToolContext(cwd=str(isolated_spill)))
    text = _text(result)

    # Running and failed rows are always shown, however old.
    assert "(livejob): running" in text and "(failedjob): failed" in text
    # Only the newest N completed rows are listed in full.
    shown_completed = [f"done{i:03d}" for i in range(40 - HUB_LIST_COMPLETED_SHOWN, 40)]
    assert all(f"({job_id})" in text for job_id in shown_completed)
    assert "(done000)" not in text
    assert f"+ {40 - HUB_LIST_COMPLETED_SHOWN} older completed subagent(s) not shown" in text
    # The folded rows are one read away, rendered the same way.
    details = result.details or {}
    handle = details["spill"]["handle"]
    assert handle in text
    stored = get_store().read_lines(handle, 1, None)
    assert stored is not None and any("(done000)" in line for line in stored[0])
    # The machine payload still carries every child.
    assert details["count"] == 42 and len(details["children"]) == 42
    # The transcript how-to appears ONCE, not under every row.
    assert text.count("lop --resume") == 1
    assert "transcript s039" in text


def test_hub_list_with_few_children_is_not_folded(isolated_spill):
    rows = [_Row(f"j{i}", "completed", session_id=f"s{i}") for i in range(3)]
    result = _hub_list("c", _Comms(rows), None, ToolContext(cwd=str(isolated_spill)))
    assert "not shown" not in _text(result)
    assert "spill" not in (result.details or {})


# -------------------------------------------------------------------------- todo


@pytest.fixture
def todo_env(tmp_path: Path):
    context = ToolContext(cwd=str(tmp_path), session_id="condense-todo")
    tools = {tool.name: tool for tool in create_tools(context)}
    yield tools, context
    builtin.TODO_STORE.pop("condense-todo", None)


LONG = "Implement the very long item text that describes a whole slice of work in detail"


@pytest.mark.asyncio
async def test_todo_mutation_receipts_are_compact(todo_env):
    tools, context = todo_env
    items = [f"{LONG} #{i}" for i in range(12)]
    await _call(tools, "todo", {"op": "init", "items": items}, context)

    done = await _call(tools, "todo", {"op": "done", "items": items}, context)
    text = _text(done)
    assert text.startswith("Marked done 12 item(s): ")
    assert "+4 more" in text and text.endswith("(12/12 resolved).")
    # Names are clipped: the model sent the full texts and still holds them.
    assert LONG not in text
    assert len(text) < 600

    added = await _call(tools, "todo", {"op": "add", "items": [LONG]}, context)
    assert "Added 1 item(s): " in _text(added) and LONG not in _text(added)
    assert _text(added).endswith("(12/13 resolved).")

    dropped = await _call(tools, "todo", {"op": "drop", "items": [LONG]}, context)
    assert _text(dropped).startswith("Dropped 1 item(s): ")


@pytest.mark.asyncio
async def test_todo_miss_error_and_view_keep_full_open_texts(todo_env):
    """The texts the model must echo exactly are never clipped."""
    tools, context = todo_env
    await _call(tools, "todo", {"op": "init", "items": [LONG, "short"]}, context)
    miss = await _call(tools, "todo", {"op": "done", "items": ["short", "ghost"]}, context)
    assert miss.is_error and f"- [ ] {LONG}" in _text(miss)
    view = await _call(tools, "todo", {"op": "view"}, context)
    assert f"- [ ] {LONG}" in _text(view)


@pytest.mark.asyncio
async def test_todo_view_folds_fully_resolved_phases(todo_env):
    tools, context = todo_env
    await _call(
        tools,
        "todo",
        {
            "op": "init",
            "phases": [
                {"phase": "Done phase", "items": ["a", "b"]},
                {"phase": "Open phase", "items": ["c", "d"]},
            ],
        },
        context,
    )
    await _call(tools, "todo", {"op": "done", "phase": "Done phase"}, context)
    await _call(tools, "todo", {"op": "done", "items": ["c"]}, context)

    view = await _call(tools, "todo", {"op": "view"}, context)
    assert _text(view).splitlines() == [
        "Done phase · 2/2 — all resolved",
        "Open phase · 1/2",
        "- [x] c",
        "- [ ] d",
    ]


# ------------------------------------------------------------------------ secret


@pytest.fixture
def secret_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """HOME and the config dir both inside tmp_path, as ``tests/unit/secrets``
    does — the tool opens the default store, so the redirect is asserted."""
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.secrets.keys import secrets_dir

    root = tmp_path / "config"
    root.mkdir()
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv(CONFIG_DIR_ENV, str(root))
    assert secrets_dir().is_relative_to(root), "store escaped the sandbox"
    return root


async def _secret(op: str, **kwargs: object) -> ToolResult:
    from local_operator.tools.secret_tool import execute_secret

    return await execute_secret("c", {"op": op, **kwargs}, AbortSignal(), None, ToolContext())


@pytest.mark.asyncio
async def test_secret_list_clips_descriptions_and_spills_the_full_list(secret_store):
    long_description = "Production API token for the billing integration " * 4
    await _secret("store", name="BILLING_TOKEN", value="v1", description=long_description)
    await _secret("store", name="SHORT", value="v2", description="short one")

    listed = await _secret("list")
    text = _text(listed)
    assert "BILLING_TOKEN" in text and "SHORT" in text
    assert long_description.strip() not in text and "\u2026" in text
    handle = (listed.details or {})["spill"]["handle"]
    assert handle in text and "descriptions clipped to 80 chars" in text
    stored = get_store().read_lines(handle, 1, None)
    assert stored is not None and long_description.strip() in "\n".join(stored[0])


@pytest.mark.asyncio
async def test_a_long_secret_list_keeps_every_name_inline(secret_store):
    names = [f"SERVICE_{i:03d}_ACCESS_TOKEN" for i in range(60)]
    for name in names:
        await _secret("store", name=name, value="v", description="x" * 70)
    text = _text(await _secret("list"))
    assert all(name in text for name in names), "a name hidden behind a handle can be missed"
    assert "descriptions omitted" in text
    assert len(text) < 4_000


@pytest.mark.asyncio
async def test_a_short_secret_list_is_unchanged(secret_store):
    await _secret("store", name="A", value="v", description="first")
    listed = await _secret("list")
    assert _text(listed).splitlines() == [
        "1 stored secret(s) — names only, never values:",
        "  A  first",
    ]
    assert "spill" not in (listed.details or {})


# -------------------------------------------------------------------------- jobs

#: A TRUE re-render: the same lines written again, which is what the collapse is
#: allowed to drop. A watcher whose frames EVOLVE (``gh run watch`` marking jobs
#: done, a timestamp ticking) is deliberately NOT this shape — see
#: ``test_evolving_frames_are_left_alone``.
REDRAW_FRAME = "== run 123 status ==\nline a\nline b\nline c\n"
REDRAW_MARKER = "\x1b[H\x1b[J"

#: The operator's real git-diff-shaped shell output, reproduced in shape: a
#: repeated filename line becomes a "frame header", and two chunks that share
#: enough context lines look "similar" while each carries content the other
#: lacks. The old heuristic elided the first chunk here (105 of 8,380 chars kept,
#: 114 distinct lines gone — agent review round 1, MAJOR). Containment refuses:
#: ``+unique added line 1`` is in no later chunk.
GIT_DIFF_SHAPED = (
    "iceberg.webp\n"
    "diff --git a/iceberg.webp b/iceberg.webp\n"
    "index 1111111..2222222 100644\n"
    "--- a/iceberg.webp\n"
    "+++ b/iceberg.webp\n"
    "@@ -1,3 +1,4 @@\n"
    " context line\n"
    "+unique added line 1\n"
    "iceberg.webp\n"
    "diff --git a/iceberg.webp b/iceberg.webp\n"
    "index 1111111..2222222 100644\n"
    "--- a/iceberg.webp\n"
    "+++ b/iceberg.webp\n"
    "@@ -1,3 +1,4 @@\n"
    " context line\n"
    "-removed line\n"
    "+unique added line 2\n"
)

#: The QA round-1 false positive: an ordinary sequential loop over six hosts. The
#: old heuristic picked a recurring status line (``ping ok``) as the frame header,
#: which made 3-line "frames" whose similarity landed exactly on the threshold, so
#: the loop folded to its last host and hid ``disk FULL (98%)``.
SIX_HOST_LOOP = "".join(
    f"== checking host web-{n} ==\nping ok\n"
    f"{'disk FULL (98%)' if n == 2 else 'disk ok'}\nmem ok\n"
    for n in (1, 2, 3, 4, 5, 6)
)


def test_a_verified_redraw_collapses_to_its_latest_frame_and_keeps_the_final_line():
    text = REDRAW_MARKER + REDRAW_FRAME + REDRAW_MARKER + REDRAW_FRAME
    text += REDRAW_MARKER + REDRAW_FRAME + "FINAL: run 999 completed with conclusion success\n"

    latest, elided = _collapse_refreshing_frames(text)

    assert elided == 2
    assert latest == REDRAW_FRAME + "FINAL: run 999 completed with conclusion success"
    # The `FINAL:` line states the outcome and is written once, after the last
    # frame: it must never ride an elision (QA round 1, row 4a).
    assert "FINAL: run 999 completed with conclusion success" in latest


def test_a_repeating_frame_header_collapses_without_escape_sequences():
    """A watcher on a pipe writes frames with no clear-screen markers."""
    text = (REDRAW_FRAME + "\n") * 4
    latest, elided = _collapse_refreshing_frames(text)
    assert elided == 3
    assert latest.startswith("== run 123 status ==")


def test_a_git_diff_shaped_payload_is_left_alone():
    """The MAJOR: a diff is not a redraw, whatever repeats inside it."""
    assert _collapse_refreshing_frames(GIT_DIFF_SHAPED) == (GIT_DIFF_SHAPED, 0)


def test_a_six_host_sequential_loop_is_left_alone():
    """Q1: folding this hid the one host worth reading about."""
    assert _collapse_refreshing_frames(SIX_HOST_LOOP) == (SIX_HOST_LOOP, 0)


def test_evolving_frames_are_left_alone():
    """The trade the containment rule makes, pinned so it cannot drift back.

    ``gh run watch`` marks jobs done between frames and ticks a timestamp, so an
    earlier frame carries lines the latest one does not. Eliding those bytes
    while calling them "repeated frames" is the unverified claim the review
    rejected, so the whole delta is returned instead.
    """
    frames = "".join(
        f"* v1.0 Build · 123\nTriggered via release about {n} minute ago\nJOBS\n"
        f"{'* Publish to npm' if n < 3 else '✓ Publish to npm'}\n  ✓ Set up job\n"
        for n in (1, 2, 3)
    )
    assert _collapse_refreshing_frames(frames) == (frames, 0)


@pytest.mark.parametrize(
    "text",
    [
        # An ordinary log with a repeated line is not a redraw.
        "\n".join("PASS" if i % 5 == 0 else f"test {i} ok" for i in range(40)),
        "a\nb\na\nc\nd\ne",
        "single line",
        # A "frame" of two lines is below the minimum: too small to be a frame.
        "hdr\nx\nhdr\ny",
    ],
)
def test_ordinary_output_is_left_alone(text):
    assert _collapse_refreshing_frames(text) == (text, 0)


@pytest.mark.asyncio
async def test_peek_returns_the_latest_frame_and_keeps_the_rest_reachable(isolated_spill):
    manager = AsyncJobManager()
    context = ToolContext(cwd=str(isolated_spill), session_id="s", jobs=manager)
    tools = {tool.name: tool for tool in create_tools(context)}
    started = asyncio.Event()

    async def runner(job_id: str, signal: Any, report_progress) -> str:
        # Signals that the manager really entered the coroutine: cancelling
        # before its first step drops it un-awaited and the suite reports the
        # RuntimeWarning (agent review round 1, NIT 5).
        started.set()
        await asyncio.sleep(30)
        return "never"

    job_id = manager.register("bash", "gh run watch", runner)
    await asyncio.wait_for(started.wait(), 5)
    frames = (REDRAW_MARKER + REDRAW_FRAME) * 5
    manager.append_output(job_id, frames)
    peek = await _call(tools, "jobs", {"op": "peek", "job_id": job_id}, context)
    text = _text(peek)
    details = peek.details or {}
    assert (
        "[4 earlier frame(s) elided — every line of text they carried is in the frame below" in text
    )
    assert "line a" in text and text.count("line a") == 1
    assert details["frames_elided"] == 4
    assert details["new_chars"] == len(frames)
    # The receipt reports the delta AND what the collapse removed from it, and
    # exposes the frames' handle under its own key, not only in prose.
    assert details["shown_chars"] < details["new_chars"]
    assert details["elided_chars"] > 0
    assert details["frames_spill"]["handle"] in text
    stored = get_store().read_lines(details["frames_spill"]["handle"], 1, None)
    assert stored is not None and "line a" in stored[0]
    await manager.cancel(job_id)
    await manager.dispose()


@pytest.mark.asyncio
async def test_peek_does_not_collapse_when_the_spill_store_refuses(isolated_spill, monkeypatch):
    """No handle means no elision: the dropped frames would be unrecoverable."""
    monkeypatch.setattr(builtin, "_spill", lambda *a, **k: None)
    manager = AsyncJobManager()
    context = ToolContext(cwd=str(isolated_spill), session_id="s", jobs=manager)
    tools = {tool.name: tool for tool in create_tools(context)}
    started = asyncio.Event()

    async def runner(job_id: str, signal: Any, report_progress) -> str:
        started.set()
        await asyncio.sleep(30)
        return "never"

    job_id = manager.register("bash", "gh run watch", runner)
    await asyncio.wait_for(started.wait(), 5)
    frames = (REDRAW_MARKER + REDRAW_FRAME) * 3
    manager.append_output(job_id, frames)
    peek = await _call(tools, "jobs", {"op": "peek", "job_id": job_id}, context)
    text = _text(peek)
    details = peek.details or {}

    assert text.count("line a") == 3
    assert "elided" not in text
    assert "frames_elided" not in details
    await manager.cancel(job_id)
    await manager.dispose()


# -------------------------------------------------------------------------- read


@pytest.mark.asyncio
async def test_spill_search_is_bounded_even_when_matched_lines_are_huge(isolated_spill):
    """Every ``read`` over 20k chars on the fleet since 09-29 came from here."""
    context = ToolContext(cwd=str(isolated_spill), session_id="s")
    tools = {tool.name: tool for tool in create_tools(context)}
    huge = "\n".join(f"export {{ Icon{i} }}; " + "x" * 50_000 + " needle" for i in range(8))
    meta = get_store().write(huge, tool_name="bash", session_id="s")
    assert meta is not None

    result = await _call(tools, "read", {"path": f"{meta.handle}?q=needle"}, context)
    text = _text(result)
    assert len(text) <= READ_OUTPUT_LIMIT_CHARS + 1_000
    assert "needle" in text and "chars\u2026]" not in text.split("\n", 1)[0]
    # Each hit is shown in a window that says how much it dropped.
    assert "[\u2026" in text
    # A clipped match list names its next page in match coordinates.
    assert "of 8 match(es)" in text


# --------------------------------------------------- read `?q=` paging (Q2, NIT 4)


async def _spill_with_matches(tmp_path: Path, count: int = 12):
    context = ToolContext(cwd=str(tmp_path), session_id="s")
    tools = {tool.name: tool for tool in create_tools(context)}
    text = "\n".join(f"hit {i}" if i % 2 else f"filler {i}" for i in range(count * 2))
    meta = get_store().write(text, tool_name="bash", session_id="s")
    assert meta is not None
    return tools, context, meta


@pytest.mark.asyncio
async def test_every_match_page_advertises_the_next_one(isolated_spill, tmp_path):
    """A page that consumes its slice exactly must still name the next page.

    It used to advertise nothing, so ``range="19-36"`` (page 2 of 90) was a dead
    end: the model had seen 18 of 90 and was told nothing further (QA round 1, Q2).
    """
    tools, context, meta = await _spill_with_matches(tmp_path, count=12)
    query = f"{meta.handle}?q=hit"

    page = await _call(tools, "read", {"path": query, "range": "1-5"}, context)
    text = _text(page)
    assert "5 of 12 match(es)" in text
    assert 'range="6-10"' in text and "7 more match(es)" in text

    second = await _call(tools, "read", {"path": query, "range": "6-10"}, context)
    assert 'range="11-12"' in _text(second), "the pointer is clamped to the last match"
    assert "2 more match(es)" in _text(second)

    last = await _call(tools, "read", {"path": query, "range": "11-12"}, context)
    assert "that is every match" in _text(last)
    assert "next page" not in _text(last)


@pytest.mark.asyncio
async def test_a_page_past_the_last_match_is_not_reported_as_no_matches(isolated_spill, tmp_path):
    """Matches exist; the PAGE does not. The two need different next calls."""
    tools, context, meta = await _spill_with_matches(tmp_path, count=12)

    past = await _call(tools, "read", {"path": f"{meta.handle}?q=hit", "range": "13-20"}, context)
    text = _text(past)

    assert "No lines match" not in text
    assert "That page is past the last match: 12 match(es)" in text
    assert (past.details or {})["total_matches"] == 12


@pytest.mark.asyncio
async def test_a_query_that_really_matches_nothing_still_says_so(isolated_spill, tmp_path):
    tools, context, meta = await _spill_with_matches(tmp_path, count=12)
    none = await _call(tools, "read", {"path": f"{meta.handle}?q=absent"}, context)
    assert "No lines match 'absent'" in _text(none)
    assert none.useless is True


def test_hub_list_does_not_fold_when_the_spill_store_refuses(isolated_spill, monkeypatch):
    """No handle means no fold: the folded rows would vanish with no note.

    Their ``transcript <id>`` is what a ``--resume`` needs, so losing them
    silently is worse than the bytes the fold saves (agent review round 1,
    MINOR 2).
    """
    monkeypatch.setattr(builtin, "_spill", lambda *a, **k: None)
    rows = [_Row(f"done{i:03d}", "completed", session_id=f"s{i:03d}") for i in range(18)]
    rows.append(_Row("livejob", "running"))

    result = _hub_list("c", _Comms(rows), None, ToolContext(cwd=str(isolated_spill)))
    text = _text(result)
    details = result.details or {}

    assert all(f"(done{i:03d})" in text for i in range(18)), "every completed row is listed"
    assert "transcript s000" in text
    assert "not shown" not in text
    assert details["count"] == 19
    assert "folded_completed" not in details and "spill" not in details


# ------------------------------------------- round-2 findings (F1, F2, F3, Q5)


@pytest.mark.asyncio
async def test_paging_reaches_matches_past_the_stores_per_call_limit(isolated_spill, tmp_path):
    """F1/Q3: the pointer must not run past what one search can materialize.

    A page is addressed by match NUMBER, so page 2 (matches 101-200) needs the
    search to retain at least 200 matches. With the fixed default limit the store
    returned the first 100 and every range from 101 up answered "past the last
    match" — a false statement the model then followed in a loop.
    """
    context = ToolContext(cwd=str(tmp_path), session_id="s")
    tools = {tool.name: tool for tool in create_tools(context)}
    text = "\n".join(f"hit-{i}" if i % 2 else f"filler {i}" for i in range(434))
    meta = get_store().write(text, tool_name="bash", session_id="s")
    assert meta is not None
    assert text.count("hit-") == 217
    query = f"{meta.handle}?q=hit-"

    first = await _call(tools, "read", {"path": query}, context)
    assert "100 of 217 match(es)" in _text(first)
    assert 'range="101-200"' in _text(first)

    second = await _call(tools, "read", {"path": query, "range": "101-200"}, context)
    second_text = _text(second)
    assert "No lines match" not in second_text and "past the last match" not in second_text
    assert "100 of 217 match(es)" in second_text
    # match k is `hit-(2k-1)`, on line 2k: page 2 starts at match 101 == hit-201
    # on line 202, and ends at match 200 == hit-399 on line 400.
    assert "202| hit-201" in second_text, "page 2 really is matches 101-200"
    assert "400| hit-399" in second_text and "2| hit-1" not in second_text
    assert 'range="201-217"' in second_text

    third = await _call(tools, "read", {"path": query, "range": "201-217"}, context)
    assert "17 of 217 match(es)" in _text(third)
    assert "that is every match" in _text(third)

    # An open range past the first page works too ("101-").
    opened = await _call(tools, "read", {"path": query, "range": "101-"}, context)
    assert "202| hit-201" in _text(opened)


@pytest.mark.asyncio
async def test_a_page_beyond_a_single_calls_ceiling_is_named_as_such(isolated_spill, tmp_path):
    """Past the ceiling the answer names the limit rather than lying about the content."""
    tools, context, meta = await _spill_with_matches(tmp_path, count=3)
    past = await _call(
        tools, "read", {"path": f"{meta.handle}?q=hit", "range": "9999-10010"}, context
    )
    assert "at most 10000 matches" in _text(past)
    assert "past the last match" not in _text(past)


def test_an_indentation_difference_blocks_the_elision():
    """F2: lines are compared verbatim, so `    x` is not `x`.

    The earlier rule stripped each line before comparing, so an earlier frame's
    `    indented: deep detail` was dropped because the kept frame held the
    unindented spelling — while the note said the line was preserved.
    """
    indented = "HDR\n    indented: deep detail\nplain line\n"
    flush = "HDR\nindented: deep detail\nplain line\n"

    assert _collapse_refreshing_frames(indented + flush) == (indented + flush, 0)
    assert _collapse_refreshing_frames(flush + indented) == (flush + indented, 0)
    # The same text on both sides still collapses: the rule blocks DIFFERENCES,
    # it does not refuse indented frames.
    assert _collapse_refreshing_frames(indented * 3)[1] == 2


def test_the_frame_start_preference_actually_applies():
    """F3: the shaped-candidate preference could never fire, so it was dead weight.

    ``before.rstrip("\n").endswith("\n\n")`` stripped the very bytes it tested
    for, so only a line at the very top of the text ever qualified. Here ``H``
    sits under a blank line at every occurrence (4) and ``D`` is the text's first
    line and otherwise trails content (1); the working check orders ``H`` first,
    the broken one ordered ``D`` first. Asserted on the ORDER because the
    preference is a search order, not a correctness lever: every candidate is
    still validated by containment, and a brute-force search found no payload
    where two candidates both validate (see ``_header_candidates``).
    """
    text = "D\n\nH\nx\ny\n" * 4
    lines = text.splitlines()
    occurrences: dict[str, list[int]] = {}
    for index, line in enumerate(lines):
        if line.strip():
            occurrences.setdefault(line, []).append(index)

    ordered = _header_candidates(lines, occurrences)

    assert ordered[0][0] == "H", "the frame-start-shaped header is tried first"
    assert "D" in [line for line, _offsets in ordered]
    # ...and the collapse still runs through the same validated path.
    assert _collapse_refreshing_frames(text)[1] >= 1


def test_a_clear_style_redraw_collapses():
    """Q5: `clear`/`tput clear` emit ESC[H ESC[2J, and the cursor move broke it.

    Matching only the erase code left a dangling ESC[H line at the end of the
    previous chunk — a line the kept frame does not have, which failed
    containment and made every `clear`-cleared screen uncollapsible.
    """
    text = "".join("\x1b[H\x1b[2J" + REDRAW_FRAME for _ in range(4))
    latest, elided = _collapse_refreshing_frames(text)
    assert elided == 3
    assert latest == REDRAW_FRAME.rstrip("\n")
    # The reverse order (erase, then home) is what some terminals emit.
    reversed_text = "".join("\x1b[2J\x1b[H" + REDRAW_FRAME for _ in range(3))
    assert _collapse_refreshing_frames(reversed_text)[1] == 2

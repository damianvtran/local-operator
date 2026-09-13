"""The `/resume` picker: names, filtering, cursor, and the id it hands back.

The picker replaced a block of `<hex id>  3h ago` rows printed into the
transcript. What that surface could not do — be navigated, be searched, and
answer with a choice — is what these tests pin.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.color import Color
from rich.style import Style
from textual import events
from textual.app import App, ComposeResult

from local_operator.resume import (
    NAME_MAX_CHARS,
    NAME_SCAN_CHARS,
    ORIGIN_SUBAGENT,
    SessionRow,
    mark_session_origin,
    recent_session_rows,
    session_name,
)
from local_operator.session.preview import GAP_TEXT, PREVIEW_TAIL_BYTES
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.session_picker import (
    _EXEC_LEGEND,
    _MARKER_LEGEND,
    BODY_MATCH_MARKER,
    CARD_PADDING_ROWS,
    EXEC_MARKER,
    GUTTER_CELLS,
    NAME_MIN_CELLS,
    PICKER_MIN_WIDTH,
    SessionPickerScreen,
    _footer_hints,
    _meta_legends,
    filter_rows,
    matched_in_body,
    plan_columns,
    plan_layout,
    rank_rows,
    render_rows,
)

NOW = 1_000_000.0


def _row(session_id: str, name: str, age_s: float = 60.0) -> SessionRow:
    return SessionRow(id=session_id, mtime=NOW - age_s, name=name)


def _write_transcript(root: Path, session_id: str, entries: list[dict[str, object]]) -> Path:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "transcript.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    return directory


def _message(role: str, text: str, **payload: object) -> dict[str, object]:
    return {
        "id": "e1",
        "ts": 0,
        "type": "message",
        "payload": {"kind": "message", "role": role, "content": [{"text": text}], **payload},
    }


# --- naming -----------------------------------------------------------------


def test_a_session_is_named_by_its_opening_user_message(tmp_path: Path) -> None:
    """The only per-session title on disk is what the user typed first."""
    directory = _write_transcript(
        tmp_path,
        "abc123",
        [_message("user", "Make an asteroids game"), _message("assistant", "sure")],
    )
    assert session_name(directory) == "Make an asteroids game"


def test_a_tool_result_never_names_the_conversation(tmp_path: Path) -> None:
    """``tool`` is also a four-character role and its content is command
    output; matching the role loosely would name sessions after directory
    listings."""
    directory = _write_transcript(
        tmp_path,
        "abc123",
        [
            _message("tool", "total 48\ndrwxr-xr-x  12 damian", tool_call_id="c1", tool_name="ls"),
            _message("user", "why is the build failing"),
        ],
    )
    assert session_name(directory) == "why is the build failing"


def test_a_multi_line_prompt_becomes_one_scannable_line(tmp_path: Path) -> None:
    """A prompt is usually several lines; the picker has one row per session."""
    directory = _write_transcript(
        tmp_path, "abc123", [_message("user", "fix   the parser\n\nit crashes on   empty input")]
    )
    assert session_name(directory) == "fix the parser it crashes on empty input"


def test_a_long_prompt_is_ellipsised_within_the_budget(tmp_path: Path) -> None:
    directory = _write_transcript(tmp_path, "abc123", [_message("user", "word " * 100)])
    name = session_name(directory)
    assert len(name) <= NAME_MAX_CHARS
    assert name.endswith("…")


def test_an_unreadable_or_empty_transcript_yields_no_name(tmp_path: Path) -> None:
    """A nameless row beats taking the picker down: this runs over every
    session directory, including ones a live session is still writing."""
    empty = _write_transcript(tmp_path, "empty", [])
    assert session_name(empty) == ""
    # A half-written final line is normal for a running session.
    partial = tmp_path / "sessions" / "partial"
    partial.mkdir(parents=True)
    (partial / "transcript.jsonl").write_text('{"id": "e1", "type": "mess', encoding="utf-8")
    assert session_name(partial) == ""
    assert session_name(tmp_path / "sessions" / "missing") == ""


def _message_with_image(text: str, data_chars: int) -> dict[str, object]:
    """A user message carrying a pasted image, text block first — the order
    ``Message.user(text, images)`` produces and the writer preserves."""
    return {
        "id": "e1",
        "ts": 0,
        "type": "message",
        "payload": {
            "kind": "message",
            "role": "user",
            "content": [
                {"text": text},
                {"data": "A" * data_chars, "mime_type": "image/png"},
            ],
        },
    }


def test_a_session_opening_with_a_screenshot_is_still_named(tmp_path: Path) -> None:
    """One pasted image puts the first line past the scan window, and the
    fragment used to be dropped — which left every session that begins with a
    screenshot reading `(unnamed session)` in the picker for the rest of its
    life. Measured on two real sessions whose first lines were 115,289 and
    733,034 characters.

    The window still bounds the read; what changed is that a fragment is mined
    for the opener instead of discarded, which is safe because the text block
    precedes the image data on the line.
    """
    directory = _write_transcript(
        tmp_path,
        "shot01",
        [_message_with_image("why does the resume picker forget my sessions", NAME_SCAN_CHARS * 2)],
    )
    with (directory / "transcript.jsonl").open(encoding="utf-8") as handle:
        assert len(handle.readline()) > NAME_SCAN_CHARS, "this case needs an oversized line"
    assert session_name(directory) == "why does the resume picker forget my sessions"


def test_a_fragment_is_never_named_after_the_image_it_carries(tmp_path: Path) -> None:
    """A name taken from base64 would be worse than no name. When the text block
    does NOT come first, the scan declines rather than reaching past the data."""
    directory = tmp_path / "sessions" / "shot02"
    directory.mkdir(parents=True)
    payload = {
        "id": "e1",
        "ts": 0,
        "type": "message",
        "payload": {
            "kind": "message",
            "role": "user",
            "content": [
                {"data": "A" * (NAME_SCAN_CHARS * 2), "mime_type": "image/png"},
                {"text": "this text is past the image"},
            ],
        },
    }
    (directory / "transcript.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert session_name(directory) == ""


def test_a_fragment_whose_text_is_cut_off_yields_no_name(tmp_path: Path) -> None:
    """A title cut mid-word reads like a bug. The value must close inside the
    window to be used at all, so an opener longer than the window is declined
    rather than truncated to whatever the read happened to reach."""
    directory = tmp_path / "sessions" / "shot03"
    directory.mkdir(parents=True)
    line = '{"id":"e1","ts":0,"type":"message","payload":{"kind":"message","role":"user"'
    line += ',"content":[{"text":"' + "word " * (NAME_SCAN_CHARS // 2)
    (directory / "transcript.jsonl").write_text(line, encoding="utf-8")
    assert session_name(directory) == ""


def test_a_fragment_from_a_non_user_opener_is_declined(tmp_path: Path) -> None:
    """The strict path matches ``role`` exactly so a tool result cannot name a
    session; the fragment path has to hold the same line."""
    directory = tmp_path / "sessions" / "shot04"
    directory.mkdir(parents=True)
    payload = {
        "id": "e1",
        "ts": 0,
        "type": "message",
        "payload": {
            "kind": "message",
            "role": "tool",
            "tool_call_id": "c1",
            "content": [{"text": "total 48"}, {"data": "A" * (NAME_SCAN_CHARS * 2)}],
        },
    }
    (directory / "transcript.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert session_name(directory) == ""


def test_rows_are_newest_first_and_carry_their_name(tmp_path: Path) -> None:
    older = _write_transcript(tmp_path, "older1", [_message("user", "the older one")])
    newer = _write_transcript(tmp_path, "newer1", [_message("user", "the newer one")])
    import os

    os.utime(older / "transcript.jsonl", (1_000, 1_000))
    os.utime(newer / "transcript.jsonl", (2_000, 2_000))
    rows = recent_session_rows(tmp_path)
    assert [row.id for row in rows] == ["newer1", "older1"]
    assert [row.name for row in rows] == ["the newer one", "the older one"]


def test_the_picker_offers_only_sessions_the_user_started(tmp_path: Path) -> None:
    """A subagent's child session is not a conversation the user can recognise.

    Children land in the same ``sessions/`` tree with the same shape, so the
    picker named them by their opening message — which for a delegated run is
    the role preamble the parent wrote. On one machine 40 of 50 offered rows
    were ``[role: reviewer] You are an INDEPENDENT reviewer…`` and the user's
    own sessions were paged off the bottom.
    """
    _write_transcript(tmp_path, "mine", [_message("user", "fix the resume picker")])
    child = _write_transcript(
        tmp_path, "child", [_message("user", "[role: reviewer] You are an INDEPENDENT reviewer")]
    )
    mark_session_origin(child, ORIGIN_SUBAGENT, label="reviewer")

    rows = recent_session_rows(tmp_path)
    assert [row.id for row in rows] == ["mine"]
    # Hidden from the listing, still on disk: the transcript is what makes a
    # stopped child resumable by id and readable after the fact.
    assert (child / "transcript.jsonl").is_file()


def test_a_session_with_no_marker_is_still_the_user_s(tmp_path: Path) -> None:
    """Every conversation that predates the marker must keep appearing.

    The filter reads absence as "the user's" precisely so an upgrade does not
    empty the picker of real work.
    """
    _write_transcript(tmp_path, "before", [_message("user", "an older conversation")])
    assert [row.id for row in recent_session_rows(tmp_path)] == ["before"]


# --- filtering --------------------------------------------------------------


def test_filtering_matches_name_or_id_and_preserves_order() -> None:
    """Order must not change under a filter: a row that moved under the cursor
    while the query grew would resume the wrong conversation."""
    rows = [_row("aaa1", "asteroids game"), _row("bbb2", "parser crash"), _row("ccc3", "asteroid")]
    assert [r.id for r in filter_rows(rows, "aster")] == ["aaa1", "ccc3"]
    assert [r.id for r in filter_rows(rows, "BBB")] == ["bbb2"]
    assert [r.id for r in filter_rows(rows, "")] == ["aaa1", "bbb2", "ccc3"]
    assert filter_rows(rows, "nothing here") == []


# --- rendering --------------------------------------------------------------


def test_a_row_shows_the_name_the_age_and_the_id() -> None:
    lines = [line.plain for line in render_rows([_row("abc123def456", "ship it")], 0, 74, NOW)]
    assert "ship it" in lines[0]
    assert "abc123def456" in lines[0]
    assert "1m ago" in lines[0]


def test_the_cursor_marks_exactly_one_row() -> None:
    rows = [_row("a1", "one"), _row("b2", "two"), _row("c3", "three")]
    lines = [line.plain for line in render_rows(rows, 1, 74, NOW)]
    assert [line.startswith("❯") for line in lines] == [False, True, False]


def test_a_live_exec_run_is_named_as_one_rather_than_passing_as_a_conversation() -> None:
    """The discoverability gap this tag closes.

    Since #804 every ``lop exec`` publishes an ordinary attachable record, and
    ``decorate_rows(include_live=True)`` has been folding those into this list
    ever since — rendered identically to a conversation the user started. An
    idle one-shot and a session they were sitting in both read as a bare ``●``.
    """
    rows = [
        _row("aaaaaaaaaaaa", "my own conversation")._replace(live_state="idle"),
        _row("bbbbbbbbbbbb", "nightly audit")._replace(live_state="idle", kind="exec"),
    ]
    mine, execrun = (line.plain for line in render_rows(rows, 0, 74, NOW))
    assert EXEC_MARKER.strip() in execrun
    assert EXEC_MARKER.strip() not in mine


def test_the_exec_column_is_reserved_for_every_row_so_names_stay_flush() -> None:
    """The same fixed-chrome rule ``FORK_MARKER`` follows, and for its reason.

    Painting the tag only on tagged rows moves the start of the name between
    rows, ragging the left edge of the one field the user reads down the list.
    Asserted as a column position rather than as a substring because that is
    the property the eye actually reads.
    """
    rows = [
        _row("aaaaaaaaaaaa", "alpha")._replace(live_state="idle"),
        _row("bbbbbbbbbbbb", "beta")._replace(live_state="idle", kind="exec"),
    ]
    plain = [line.plain for line in render_rows(rows, 0, 74, NOW)]
    assert plain[0].index("alpha") == plain[1].index("beta")


def test_a_list_with_no_exec_run_reserves_no_exec_column() -> None:
    """A column nobody needs costs the name its cells on every row."""
    rows = [_row("aaaaaaaaaaaa", "alpha")._replace(live_state="idle")]
    with_exec = plan_columns(rows, 74, ["1m ago"], False, False, True, True)
    without = plan_columns(rows, 74, ["1m ago"], False, False, True, False)
    assert without[0] == with_exec[0] + cell_len(EXEC_MARKER)


def test_the_repaint_signature_covers_every_field_the_rows_render() -> None:
    """The guard that failed to catch `kind` (agent review round 1, MAJOR-1/Q1).

    The previous assertion only checked that every signature NAME is a real
    ``SessionRow`` field, which catches a rename and is blind to the error that
    actually shipped: a field ADDED to the row, rendered by the picker, and
    never added to the signature. ``_tick`` then compares two different rows
    equal and leaves stale pixels on screen.

    Asserted as the partition rather than as a membership test, so a future
    field must be classified one way or the other and cannot simply be
    forgotten.
    """
    covered = set(SessionPickerScreen._SIGNATURE_FIELDS)
    excluded = set(SessionPickerScreen._SIGNATURE_EXCLUDED)
    assert covered.isdisjoint(excluded), "a field cannot be both compared and excluded"
    assert covered | excluded == set(SessionRow._fields)
    # The specific escape, pinned by name: `kind` drives EXEC_MARKER and the
    # reserved column, so it must be compared.
    assert "kind" in covered


def test_a_kind_only_change_is_visible_to_the_repaint_decision() -> None:
    """Two rows identical but for ``kind`` must not compare equal.

    The rendered proof that the signature fix matters: these two rows paint
    differently, so a repaint decision that called them equal would show the
    first row's `[exec]` for a session that is no longer one.
    """
    was_exec = _row("bbbbbbbbbbbb", "nightly audit")._replace(live_state="idle", kind="exec")
    now_tui = was_exec._replace(kind="tui")

    def signature(row: SessionRow) -> tuple[object, ...]:
        return tuple(getattr(row, f, None) for f in SessionPickerScreen._SIGNATURE_FIELDS)

    assert signature(was_exec) != signature(now_tui)
    # ... and they really do render differently, so the comparison is load-bearing.
    painted_exec = render_rows([was_exec], 0, 74, NOW)[0].plain
    painted_tui = render_rows([now_tui], 0, 74, NOW)[0].plain
    assert EXEC_MARKER.strip() in painted_exec
    assert EXEC_MARKER.strip() not in painted_tui


def test_the_exec_column_does_not_collapse_when_the_one_shot_is_reaped() -> None:
    """Design round 1, D1: the reserved column may widen, never narrow.

    A reap is not a user action, and it happens at the NORMAL END of every
    one-shot. Letting the column collapse then moved every name 7 cells left
    with no keystroke — the list lurching at exactly the moment the tag exists
    to explain. The reaped row loses its own tag (that change is real and must
    stay visible); nothing else moves.
    """
    live = [
        _row("aaaaaaaaaaaa", "alpha")._replace(live_state="idle"),
        _row("bbbbbbbbbbbb", "nightly audit")._replace(live_state="idle", kind="exec"),
    ]
    # The same rows after the one-shot ends: the record is gone, so is the kind.
    reaped = [live[0], live[1]._replace(live_state="", kind="")]

    screen = SessionPickerScreen(live, NOW)
    assert screen._exec_column_latched(live) is True
    before = plan_columns(live, 74, ["1m ago"] * 2, False, False, True, True)

    # The reap: the result set itself no longer carries a tagged row.
    assert not any(row.kind == "exec" for row in reaped)
    assert screen._exec_column_latched(reaped) is True, "the column must stay reserved"
    after = plan_columns(reaped, 74, ["1m ago"] * 2, False, False, True, True)
    assert before == after, "no column may move because a one-shot ended"

    # Without the latch the same reap collapses the column — the defect, pinned.
    unlatched = plan_columns(reaped, 74, ["1m ago"] * 2, False, False, True, False)
    assert unlatched[0] == after[0] + cell_len(EXEC_MARKER)

    # A picker that has never seen an exec row still pays nothing.
    fresh = SessionPickerScreen([live[0]], NOW)
    assert fresh._exec_column_latched([live[0]]) is False


def test_the_footer_explains_the_exec_tag_where_the_picker_can_show_it() -> None:
    """Design round 1, D2: `[exec]` states provenance; the legend states lifetime.

    The words that carry ephemerality ("Running headless (exec)") render only in
    the sidebar's hover tooltip, a surface the picker never shows. The legend is
    the picker's own established mechanism for explaining a mark, so the fact
    lands there — and only when an exec row is actually present, on the same
    "teach the mark where it is used" rule the body-match legend follows.
    """
    with_exec = _meta_legends(74, has_marked=False, has_exec=True)
    assert _EXEC_LEGEND in with_exec
    assert _EXEC_LEGEND not in _meta_legends(74, has_marked=False, has_exec=False)
    # It says how long the row lasts, not merely what it is.
    assert "one-shot" in _EXEC_LEGEND[1]

    # AT THE WIDTHS THE CARD CAN ACTUALLY PASS, and in the shape the user is
    # overwhelmingly in. Design round 2 (D2) found this legend structurally
    # unreachable because the tests asserted it at `_footer_hints(100, ...)` —
    # a width `_card_width()` caps at PICKER_MAX_WIDTH = 74 and can never
    # produce — and because `scrolls` defaulted to False while a real store
    # (a list of hundreds against one screen) always scrolls. A check
    # that cannot observe the case it exists for certifies nothing.
    for counter_cells in (0, len("showing 1–10 of 501")):
        legends = _meta_legends(74, has_marked=False, has_exec=True, counter_cells=counter_cells)
        assert _EXEC_LEGEND in legends, f"unreachable at counter_cells={counter_cells}"

    # Legends teach; keys operate. On a card too narrow for both, the legend
    # goes and never survives as a bare unlabelled glyph.
    narrow = _meta_legends(30, has_marked=False, has_exec=True, counter_cells=19)
    assert _EXEC_LEGEND not in narrow
    assert (_EXEC_LEGEND[0], "") not in narrow
    keys = [key for key, _ in _footer_hints(30)]
    assert "enter" in keys and "esc" in keys


def test_both_legends_coexist_and_the_cryptic_one_survives_longer() -> None:
    """Two marks, one footer, and a deliberate order.

    Displayed, the body mark leads because that is the order the marks appear
    in a row. Shed, `[exec]` goes first: it is a readable word that still means
    something without its gloss, while a lone right-quote with nothing
    explaining it is the rendering artifact its legend exists to prevent.
    """
    wide = [key for key, _ in _meta_legends(74, has_marked=True, has_exec=True)]
    assert wide.index(_MARKER_LEGEND[0]) < wide.index(_EXEC_LEGEND[0])

    # Under pressure the readable word gives up its gloss first. Squeezed by a
    # REAL constraint — the counter sharing the row on a narrow card — rather
    # than by a width the card cannot pass.
    squeezed = [
        key for key, _ in _meta_legends(58, has_marked=True, has_exec=True, counter_cells=19)
    ]
    assert _EXEC_LEGEND[0] not in squeezed
    assert _MARKER_LEGEND[0] in squeezed


def test_the_exec_legend_paints_on_a_scrolling_list_which_is_the_ordinary_case() -> None:
    """Design round 2, D2: the legend was unreachable in the shape users are in.

    Driven through the REAL card rather than the shed helper, because that is
    exactly the gap the finding exploited: `_footer_hints` was correct about its
    own inputs while the card could never supply the width those tests used.
    The legend is read off the FILTER ROW specifically. A grep over the whole
    frame would match the `[exec]` tag in the LIST and pass without the legend
    existing at all — a false pass this PR's QA round hit with that instrument.
    """
    rows = [
        SessionRow(
            id=f"{index:012d}",
            name=f"session number {index}",
            mtime=NOW - index * 60,
            created_at=NOW - index * 60,
            forked=False,
            live_state="",
            pending=None,
            wakes=0,
            wakes_dormant=False,
            kind="exec" if index == 1 else "",
        )
        for index in range(30)
    ]
    screen = SessionPickerScreen(rows, NOW)
    # 120 rather than 100: the counter, the legend and the keys together need
    # 97 cells and a 100-column terminal leaves 96 after the screen's padding,
    # so at 100 the legend is correctly shed and the frame proves nothing about
    # whether it can paint at all. The finding this test pins is that the
    # legend was unreachable at EVERY width, which 120 exercises.
    screen._layout = lambda: plan_layout(120, 30)  # type: ignore[method-assign]

    assert len(rows) > screen._page_rows(), "this list must scroll or it tests the wrong shape"
    meta = screen.render_footer_for_test()

    assert EXEC_MARKER.strip() in meta, f"legend absent from the meta row: {meta!r}"
    assert "one-shot" in meta
    # The keys kept their place: the legend no longer buys it by evicting a
    # hint, which is what made it unreachable when it did.
    assert "esc" in meta
    # CANARY: the list really is drawing the tag, so the legend has something to
    # explain and this is not a frame that would pass with no exec row at all.
    # And the tag itself is still in the LIST, which is the thing the legend
    # explains — the two must both be present, never one without the other.
    assert any(EXEC_MARKER.strip() in line for line in screen.render_lines_for_test())


def test_a_cold_row_carries_no_kind_so_a_reaped_exec_run_stops_claiming_to_be_one() -> None:
    """``kind`` comes off the LIVE record, so it must vanish with the record.

    An exec record is deliberately ephemeral. If the tag outlived it the picker
    would keep labelling a plain cold transcript as a running headless job.
    """
    cold = _row("bbbbbbbbbbbb", "nightly audit")
    assert cold.kind == ""
    assert EXEC_MARKER.strip() not in render_rows([cold], 0, 74, NOW)[0].plain


def test_an_unnamed_session_says_so_rather_than_rendering_a_blank() -> None:
    """An empty cell reads as a rendering fault; the row is still pickable."""
    line = render_rows([_row("abc123", "")], 0, 74, NOW)[0].plain
    assert "(unnamed session)" in line
    assert "abc123" in line


# --- the screen -------------------------------------------------------------


class _PickerHost(App[None]):
    """A host whose only job is to own the modal under test."""

    def __init__(self, rows: list[SessionRow], digests: dict[str, str] | None = None) -> None:
        super().__init__()
        self._rows = rows
        self._digests = digests
        self.chosen: list[str | None] = []

    def compose(self) -> ComposeResult:
        return iter(())

    async def open_picker(self) -> SessionPickerScreen:
        screen = SessionPickerScreen(self._rows, NOW, self._digests)
        self.push_screen(screen, self.chosen.append)
        return screen


async def _picker(rows: list[SessionRow], size: tuple[int, int] = (100, 30)):
    app = _PickerHost(rows)
    return app, size


@pytest.mark.asyncio
async def test_enter_answers_with_the_highlighted_session_id() -> None:
    """The whole point of the two-way surface: it hands a choice back."""
    rows = [_row("first1", "one"), _row("second", "two"), _row("third3", "three")]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        await app.open_picker()
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
    assert app.chosen == ["second"]


@pytest.mark.asyncio
async def test_escape_answers_nothing_and_resumes_no_session() -> None:
    app = _PickerHost([_row("first1", "one")])
    async with app.run_test(size=(100, 30)) as pilot:
        await app.open_picker()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert app.chosen == [None]


@pytest.mark.asyncio
async def test_typing_filters_the_list_and_enter_takes_the_match() -> None:
    """The ids are unmemorable and the names are not — with a hundred
    sessions, typing is how the right one is found."""
    rows = [_row("aaa111", "asteroids game"), _row("bbb222", "parser crash")]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        for char in "parser":
            await pilot.press(char)
        await pilot.pause()
        assert screen.filter_query == "parser"
        assert [row.id for row in screen.visible_rows] == ["bbb222"]
        await pilot.press("enter")
        await pilot.pause()
    assert app.chosen == ["bbb222"]


@pytest.mark.asyncio
async def test_backspace_widens_the_filter_again() -> None:
    rows = [_row("aaa111", "asteroids"), _row("bbb222", "parser")]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        for char in "past":
            await pilot.press(char)
        await pilot.pause()
        assert screen.visible_rows == []
        for _ in range(2):
            await pilot.press("backspace")
        await pilot.pause()
        assert screen.filter_query == "pa"
        assert [row.id for row in screen.visible_rows] == ["bbb222"]


@pytest.mark.asyncio
async def test_the_cursor_clamps_instead_of_wrapping() -> None:
    """A Down at the bottom that returned to the top reads as the list having
    reset itself."""
    rows = [_row("a1", "one"), _row("b2", "two")]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        for _ in range(5):
            await pilot.press("down")
        await pilot.pause()
        assert screen.selected_index == 1
        for _ in range(5):
            await pilot.press("up")
        await pilot.pause()
        assert screen.selected_index == 0


@pytest.mark.asyncio
async def test_a_filter_that_empties_the_list_says_so_and_answers_nothing() -> None:
    """Enter on no match must not resume an arbitrary session."""
    app = _PickerHost([_row("aaa111", "asteroids")])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        for char in "zzz":
            await pilot.press(char)
        await pilot.pause()
        assert "no session matches that filter" in "\n".join(screen.render_lines_for_test())
        await pilot.press("enter")
        await pilot.pause()
    assert app.chosen == [None]


@pytest.mark.asyncio
async def test_a_long_list_pages_and_reports_its_position() -> None:
    rows = [_row(f"id{index:04d}", f"session {index}") for index in range(30)]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 40)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("end")
        await pilot.pause()
        assert screen.selected_index == len(rows) - 1
        text = "\n".join(screen.render_lines_for_test())
        # The last row is on screen, and the position is stated — in the filter
        # row, which is where the counter lives now.
        assert f"session {len(rows) - 1}" in text
        assert f"of {len(rows)}" in screen.render_footer_for_test()


@pytest.mark.asyncio
async def test_the_picker_names_every_session_it_offers() -> None:
    """The regression the whole change exists for: the list used to be ids."""
    rows = [_row("abc123def456", "review the usage endpoint")]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        text = "\n".join(screen.render_lines_for_test())
    assert "review the usage endpoint" in text
    # The keys are advertised on the FILTER ROW now, not inside the card.
    assert "type to filter" in screen.render_footer_for_test()


def _wheel(widget, *, down: bool):
    """A real ``MouseScrollDown``/``Up`` aimed at ``widget``.

    Posted rather than calling the handler directly: the wiring under test is
    Textual's ``on_mouse_scroll_*`` dispatch, and a direct call would pass
    even if the method were named something Textual never looks for.
    """
    kind = events.MouseScrollDown if down else events.MouseScrollUp
    return kind(
        widget=widget,
        x=1,
        y=1,
        delta_x=0,
        delta_y=1 if down else -1,
        button=0,
        shift=False,
        meta=False,
        ctrl=False,
    )


@pytest.mark.asyncio
async def test_the_mouse_wheel_moves_the_cursor_and_clamps() -> None:
    """A wheel notch moves a row. Clamped, never wrapping: a scroll gesture
    that teleported to the other end would read as the list resetting."""
    rows = [_row(f"id{index:02d}", f"session {index}") for index in range(5)]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        for _ in range(2):
            screen.post_message(_wheel(screen, down=True))
        await pilot.pause()
        assert screen.selected_index == 2
        for _ in range(9):
            screen.post_message(_wheel(screen, down=True))
        await pilot.pause()
        assert screen.selected_index == len(rows) - 1  # clamped at the bottom
        for _ in range(20):
            screen.post_message(_wheel(screen, down=False))
        await pilot.pause()
        assert screen.selected_index == 0  # clamped at the top, not wrapped


class _FakeScroll:
    """The one thing the handlers use from a scroll event."""

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def test_the_wheel_handlers_stop_the_event_so_the_transcript_stays_put() -> None:
    """The card floats over the conversation; an un-stopped wheel scrolls
    both surfaces for one gesture."""
    screen = SessionPickerScreen([_row("a1", "one")], NOW)
    down, up = _FakeScroll(), _FakeScroll()
    screen.on_mouse_scroll_down(down)
    screen.on_mouse_scroll_up(up)
    assert down.stopped and up.stopped


# --- measured geometry (the card reads the terminal) ------------------------


def test_a_narrow_card_drops_the_id_rather_than_cutting_it() -> None:
    """A cut hex id still LOOKS like a valid id, and it is the one field a
    user copies into `/resume <id>`. Columns are dropped, never truncated —
    the id first, then the age, and the name last (a prefix of a sentence is
    still recognisable)."""
    rows = [_row("abc123def456", "port the CLI to typer")]
    ages = ["3h ago"]
    wide_name, wide_age, wide_id = plan_columns(rows, 74, ages)
    assert wide_id == 12 and wide_age == 6

    # Too narrow for the id: it goes, the age stays, nothing is cut.
    _, age_col, id_col = plan_columns(rows, 34, ages)
    assert id_col == 0 and age_col == 6

    # Narrower still: the age goes too, and the name keeps its floor.
    name_col, age_col, id_col = plan_columns(rows, 20, ages)
    assert (age_col, id_col) == (0, 0)
    assert name_col >= NAME_MIN_CELLS


@pytest.mark.asyncio
async def test_the_card_never_renders_wider_than_the_terminal() -> None:
    """The first cut was a fixed 78 cells that a 70-column terminal simply
    clipped, amputating the id column mid-token."""
    rows = [_row(f"id{index:010d}", f"session number {index}") for index in range(20)]
    for width in (60, 70, 80, 100, 120, 190):
        app = _PickerHost(rows)
        async with app.run_test(size=(width, 30)) as pilot:
            screen = await app.open_picker()
            await pilot.pause()
            for line in screen.render_lines_for_test():
                assert cell_len(line) <= width, (width, line)


@pytest.mark.asyncio
async def test_a_short_terminal_loses_list_rows_not_the_way_out() -> None:
    """Chrome is reserved first. The footer is the only place the card says
    `esc cancel`, so a clip that ate it left no stated way out."""
    rows = [_row(f"id{index:04d}", f"session {index}") for index in range(40)]
    for height in (16, 18, 22, 30, 48):
        app = _PickerHost(rows)
        async with app.run_test(size=(100, height)) as pilot:
            screen = await app.open_picker()
            await pilot.pause()
            lines = screen.render_lines_for_test()
            # The way out is stated in the FILTER ROW now, which is reserved
            # unconditionally and so cannot be clipped by a short terminal.
            assert "esc" in screen.render_footer_for_test(), height
            # And the list fits the rows the layout gave it: the picker is
            # full-screen, so the budget is the pane's, not 80% of the screen.
            budget = plan_layout(100, height).list_rows
            assert len(lines) <= budget, (height, len(lines), budget)


@pytest.mark.asyncio
async def test_the_cursor_is_always_on_a_row_the_card_actually_draws() -> None:
    """A fixed page size let the cursor sit on a row that was never rendered,
    and Enter then resumed a session the user could not see."""
    rows = [_row(f"id{index:04d}", f"session {index}") for index in range(40)]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 16)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        for _ in range(20):
            await pilot.press("down")
        await pilot.pause()
        drawn = "\n".join(screen.render_lines_for_test())
        chosen = screen.visible_rows[screen.selected_index]
        assert chosen.name in drawn, (screen.selected_index, drawn)


# --- selection legibility ---------------------------------------------------


@pytest.mark.parametrize("theme_name", ["dark", "light"])
def test_the_caret_is_muted_like_both_sibling_pickers(theme_name: str) -> None:
    """The caret is `muted`, never the violet meta ink.

    It was `label` — the ramp's violet for tips and skill labels — held in a
    local named `accent`, which put the one cool mark on a warm card and said
    "meta" where the frame meant "position". Asserted per ramp and against the
    TOKEN rather than a hex, because the defect was a token choice: violet is
    `#b48cd6` on dark and `#7c5a9e` on paper, and pinning one hex would let the
    other ramp keep the bug.
    """
    original = theme_mod.current_theme()
    theme_mod.set_theme(theme_name)
    try:
        rows = [_row("aaa111", "a named session"), _row("bbb222", "")]
        for selected in (0, 1):  # named and unnamed: one caret, one ink
            span = render_rows(rows, selected, 74, NOW)[selected].spans[0]
            # Rich types ``Span.style`` as ``str | Style``; a span this file
            # built carries the object, and parsing narrows it either way.
            style = span.style if isinstance(span.style, Style) else Style.parse(span.style)
            colour = style.color
            assert colour is not None and colour.triplet is not None
            assert colour.triplet.hex == theme_mod.semantic_color("muted")
            assert colour.triplet.hex != theme_mod.semantic_color("label")
            assert colour.triplet.hex != theme_mod.semantic_color("accent")
    finally:
        theme_mod.set_theme(original)


def test_selecting_an_unnamed_row_brightens_it_like_any_other() -> None:
    """Pinning the placeholder to the dim floor made a SELECTED unnamed row
    darker than every unselected named row, inverting the highlight."""
    rows = [_row("aaa1", ""), _row("bbb2", "a named session")]

    def name_colour(lines, index: int) -> str:
        colour = lines[index].spans[1].style.color
        assert colour is not None and colour.triplet is not None
        return colour.triplet.hex

    at_rest = render_rows(rows, 1, 74, NOW)  # the NAMED row is selected
    selected = render_rows(rows, 0, 74, NOW)  # the UNNAMED row is selected
    assert name_colour(selected, 0) != name_colour(at_rest, 0)
    # And selecting it does not make it dimmer than an unselected named row.
    assert name_colour(selected, 0) == theme_mod.semantic_color("muted")


def test_every_row_style_clears_the_dim_step() -> None:
    """`faint` is 1.49:1 against this card's raised ground — the ramp is
    calibrated against the app background, and an overlay lifts the ground
    without lifting the text."""
    faint = theme_mod.semantic_color("faint")
    lines = render_rows([_row("abc123def456", "a session")], 0, 74, NOW)
    for span in lines[0].spans:
        style = span.style
        if isinstance(style, str):  # rich allows a named style; ours are objects
            continue
        colour = style.color
        if colour is not None and colour.triplet is not None:
            assert colour.triplet.hex != faint, span


# --- mouse ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clicking_a_row_resumes_it() -> None:
    """The card invited the mouse in with the wheel; a list you can scroll
    with the mouse and cannot click is a half-built affordance."""
    rows = [_row("first1", "one"), _row("second", "two"), _row("third3", "three")]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        body = screen.query_one("#session-picker-results")
        # The results pane holds rows and nothing else — the title and the
        # tally moved to the filter row — so row 1 is the pane's second line.
        await pilot.click(body, offset=(4, 1))
        await pilot.pause()
    assert app.chosen == ["second"]


@pytest.mark.asyncio
async def test_narrowing_the_filter_selects_the_first_match() -> None:
    """Clamping the old index landed the cursor on the LAST match, so Enter
    took the least related row still standing."""
    rows = [
        _row("aaa111", "alpha session"),
        _row("bbb222", "beta session"),
        _row("ccc333", "gamma session"),
    ]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("end")  # cursor on the LAST row
        await pilot.pause()
        assert screen.selected_index == 2
        for char in "session":  # matches all three
            await pilot.press(char)
        await pilot.pause()
        assert screen.selected_index == 0
        await pilot.press("enter")
        await pilot.pause()
    assert app.chosen == ["aaa111"]


def test_a_single_entry_with_no_trailing_newline_still_names_the_session(
    tmp_path: Path,
) -> None:
    """The window's last line is dropped only when the read was TRUNCATED. A
    complete final line simply has no newline after it, and dropping that lost
    the name of any session whose transcript is one entry."""
    directory = tmp_path / "sessions" / "onlyone"
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text(
        json.dumps(_message("user", "the only line, no newline")), encoding="utf-8"
    )
    assert session_name(directory) == "the only line, no newline"


def test_a_truncated_first_line_is_not_parsed_as_a_name(tmp_path: Path) -> None:
    """The other half of the same rule: a line the cap cut in half is a
    fragment, not a message, and must not be mined for a name."""
    directory = tmp_path / "sessions" / "huge"
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text(
        json.dumps(_message("user", "x" * (NAME_SCAN_CHARS * 2))) + "\n",
        encoding="utf-8",
    )
    assert session_name(directory) == ""


# --- painted-frame checks (the real app, so the stylesheet actually applies) --
#
# The round-1 D2 test asserted `render_lines_for_test()` — the widget's OWN
# arithmetic — inside `_PickerHost`, which declares no CSS_PATH. It therefore
# agreed with the bug: the card computed rows the container then clipped. These
# mount the real `OperatorApp` and measure what was drawn.


async def _real_picker(rows, size):
    """Push the picker onto the REAL app so `local_operator.tcss` applies."""
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    app = OperatorApp(lambda: _factory(FakeSession()))
    return app, size


@pytest.mark.asyncio
async def test_the_footer_survives_at_every_height_on_the_real_stylesheet() -> None:
    """`max-height: 80%` resolves against the screen's CONTENT box, which
    `Screen { padding: 1 }` insets by two rows. Measuring the terminal instead
    over-counted the room, so Textual clipped the overflow off the bottom —
    silently, taking the footer (the only statement of `esc`) with it."""
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    rows = [_row(f"id{index:04d}", f"session {index}") for index in range(40)]
    for height in (14, 16, 18, 20, 23, 30, 48):
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, height)) as pilot:
            await pilot.pause()
            screen = SessionPickerScreen(rows, NOW)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.pause()
            card = screen.query_one(".session-picker")
            drawn = card.region.height
            composed = len(screen.render_lines_for_test()) + CARD_PADDING_ROWS
            assert composed <= drawn, (height, composed, drawn)


@pytest.mark.asyncio
async def test_clicking_the_chrome_or_the_backdrop_resumes_nothing() -> None:
    """A false positive here disposes the live session and reboots onto another
    one. The first cut resolved a footer click to session #12, the blank spacer
    to #10, and the dimmed backdrop beside the card to row 0."""
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    rows = [_row(f"id{index:04d}", f"session {index}") for index in range(40)]
    chosen: list[str | None] = []
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = SessionPickerScreen(rows, NOW)
        app.push_screen(screen, chosen.append)
        await pilot.pause()
        await pilot.pause()
        body = screen.query_one("#session-picker-results")
        region = body.region

        class _At:
            def __init__(self, x: int, y: int) -> None:
                self.screen_x = x
                self.screen_y = y

        # The filter row and the preview are separate WIDGETS now, so chrome is
        # no longer inside this pane's region — which is exactly why the
        # hit-test can measure against it. What must still hold is that
        # anything outside the drawn rows resolves to nothing: a false positive
        # here DISPOSES THE LIVE SESSION and reboots onto another one.
        drawn = len(screen.render_lines_for_test())
        # Below the last drawn row, still inside the pane.
        assert screen._index_at(_At(region.x + 4, region.y + drawn + 1)) is None
        # The backdrop to the left of the pane, on a row that IS a list row.
        assert screen._index_at(_At(max(0, region.x - 12), region.y + 3)) is None
        # And below the pane entirely.
        assert screen._index_at(_At(region.x + 4, region.y + region.height + 2)) is None
        assert chosen == []

        # A ROW THAT DRAWS A CONTEXT LINE OCCUPIES TWO LINES. Without walking
        # the per-row costs, every click below the first context line resolves
        # to the wrong session — off by one more row for each context line
        # above it.
        screen._digests = {row.id: f"body text for {row.name}" for row in rows}
        screen.set_query("body text")
        await pilot.pause()
        costs = screen._row_costs()
        assert any(cost == 2 for cost in costs), "no context line drawn; the case is untested"
        # Walk the rows the pane actually DREW, counting lines rather than
        # rows: the two differ exactly by the number of context lines.
        drawn_rows = 0
        used = 0
        for cost in costs[screen._offset :]:
            if used + cost > len(screen.render_lines_for_test()):
                break
            used += cost
            drawn_rows += 1
        line = 0
        for index, cost in enumerate(costs[:drawn_rows]):
            for step in range(cost):
                hit = screen._index_at(_At(region.x + 4, region.y + line))
                assert hit == index, (line, hit, index, step)
                line += 1


@pytest.mark.asyncio
async def test_the_card_never_outgrows_a_narrow_terminal() -> None:
    """The minimum width is a PREFERENCE; the terminal is not. Applying the
    floor unconditionally returned a 30-cell content box inside 4 cells of
    padding on a 30-column screen — a 38-wide card, rule and header cut."""
    rows = [_row("abc123def456", "a session")]
    for width in (24, 30, 34, 40, 50):
        app = _PickerHost(rows)
        async with app.run_test(size=(width, 30)) as pilot:
            screen = await app.open_picker()
            await pilot.pause()
            for line in screen.render_lines_for_test():
                assert cell_len(line) <= width, (width, cell_len(line), line)


@pytest.mark.asyncio
async def test_typing_a_long_filter_does_not_grow_the_card() -> None:
    """An unbounded filter echo made the card grow — and shift, since it was
    centred — with every character typed, moving the list under the eye of the
    person searching it.

    The echo now lives in the FILTER ROW, so that is the pane this asserts
    about; the assertion itself is unchanged. The row is fixed-width with
    ellipsis overflow, so 60 characters of query cannot widen it.
    """
    rows = [_row("abc123def456", "a session")]
    app = _PickerHost(rows)
    async with app.run_test(size=(80, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        row_width = screen.query_one("#session-picker-filter").size.width
        for char in "a" * 60:
            await pilot.press(char)
        await pilot.pause()
        after = cell_len(screen.render_footer_for_test())
        assert screen.filter_query == "a" * 60
    # The row is bounded by the terminal, not by the query: 60 characters of
    # filter cannot widen it, which is what stops the list shifting under the
    # eye of the person searching it.
    assert after <= row_width, (row_width, after)


@pytest.mark.asyncio
async def test_a_narrow_painted_header_keeps_the_active_filter() -> None:
    """At 50 columns the header spent the row on its static title and clipped
    the query — the only receipt that typing reached the modal.

    The shed ladder moved to the FILTER ROW, so the query is now echoed as
    ``/ asteroid`` rather than ``filter asteroid``; the rule it pins is
    unchanged, and this still reads the text off the REAL COMPOSITOR rather
    than off an accessor, which is what makes it evidence about what is drawn.
    """
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    rows = [_row(f"id{index:04d}", f"asteroid session {index}") for index in range(40)]
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(50, 20)) as pilot:
        await pilot.pause()
        screen = SessionPickerScreen(rows, NOW)
        app.push_screen(screen)
        await pilot.pause()
        for char in "asteroid":
            await pilot.press(char)
        await pilot.pause()
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())

    assert screen.filter_query == "asteroid"
    assert "asteroid" in painted, painted


def test_a_right_click_never_resumes_a_session() -> None:
    """The action behind a click disposes the live session and reboots; a
    context-menu click must not reach it."""
    screen = SessionPickerScreen([_row("a1", "one")], NOW)
    resumed: list[str] = []
    screen.dismiss = lambda value=None: resumed.append(value)  # type: ignore[assignment]

    class _RightClick:
        button = 3
        screen_x = 0
        screen_y = 0

    screen.on_click(_RightClick())
    assert resumed == []


def test_the_empty_card_says_whose_sessions_are_missing() -> None:
    """ "no previous sessions to resume" was false when only children exist.

    Delegated runs share the sessions directory and are deliberately unlisted,
    and retention evicts an older parent before its newer children — so a
    machine can reach a state with resumable directories on disk and nothing
    the picker will offer. The old sentence stated a fact about the disk that
    was untrue, and named no way forward.
    """
    from local_operator.tui.widgets.session_picker import RESUME_EMPTY_NOTICE

    screen = SessionPickerScreen([], NOW)
    card = "\n".join(screen.render_lines_for_test())
    assert RESUME_EMPTY_NOTICE in card
    assert "subagent runs are not listed" in card


def test_one_session_is_not_announced_as_1_sessions() -> None:
    """Filtering makes a one-row list the common case rather than the rare
    one, so the header's plural now shows up routinely."""
    single = SessionPickerScreen([_row("aabbcc", "the only one")], NOW).render_footer_for_test()
    assert "1 session" in single
    assert "1 sessions" not in single

    plural = "".join(
        SessionPickerScreen(
            [_row("aabbcc", "one"), _row("ddeeff", "two")], NOW
        ).render_footer_for_test()
    )
    assert "2 sessions" in plural


@pytest.mark.asyncio
async def test_the_empty_card_never_renders_wider_than_the_terminal() -> None:
    """The empty body is the one line with no truncation behind it.

    Every other row is bounded by the width the card MEASURES; the notice was
    a constant, so it satisfied the 74-cell ceiling and still overflowed the
    real card on any narrow terminal — at 60 columns it was cut to
    "…subagent runs are", losing the clause that explains why the list is
    empty, which is the whole reason the wording changed.

    Asserted against the TERMINAL width like the populated-rows guard above,
    never against ``PICKER_MAX_WIDTH``: the ceiling is 74 while an 80-column
    screen gives the card 70, so a constant-based assertion passes on a string
    that overflows.
    """
    for width in (60, 70, 80, 100, 120):
        app = _PickerHost([])
        async with app.run_test(size=(width, 30)) as pilot:
            screen = await app.open_picker()
            await pilot.pause()
            lines = screen.render_lines_for_test()
            for line in lines:
                assert cell_len(line) <= width, (width, line)
            # The explanation survives the narrow case rather than being the
            # first thing dropped: it is what makes the empty state honest.
            body = " ".join(line.strip() for line in lines)
            assert "subagent runs are not listed" in body, (width, body)


# --- searching the conversation body ----------------------------------------
# The reported failure: a session was findable only by the words in its NAME,
# so a conversation whose title did not happen to contain what the user
# remembered was unreachable. These pin the widened filter and the marker that
# keeps its results explicable.


def test_a_row_matches_on_its_conversation_body(tmp_path: Path) -> None:
    rows = [_row("aaa1", "vague title"), _row("bbb2", "another")]
    assert [r.id for r in filter_rows(rows, "retention", {"aaa1"})] == ["aaa1"]


def test_a_body_match_never_reorders_the_list() -> None:
    """Same invariant as every other filter here: a row that moved under the
    cursor while the query grew would resume the wrong conversation."""
    rows = [_row("aaa1", "one"), _row("bbb2", "two"), _row("ccc3", "three")]
    assert [r.id for r in filter_rows(rows, "topic", {"ccc3", "aaa1"})] == ["aaa1", "ccc3"]


def test_a_caller_without_an_index_keeps_the_old_behaviour() -> None:
    """Hosts with no index — tests, embedders — must not lose the filter."""
    rows = [_row("aaa1", "asteroids game"), _row("bbb2", "parser crash")]
    assert [r.id for r in filter_rows(rows, "aster")] == ["aaa1"]
    assert filter_rows(rows, "retention") == []


def test_only_a_body_match_is_marked() -> None:
    """A row whose visible name contains the query needs no explanation; one
    that does not would otherwise read as an arbitrary result."""
    assert matched_in_body(_row("aaa1", "vague"), "retention", {"aaa1"}) is True
    assert matched_in_body(_row("bbb2", "retention sweep"), "retention", {"bbb2"}) is False
    assert matched_in_body(_row("ccc3", "vague"), "retention", set()) is False
    assert matched_in_body(_row("aaa1", "vague"), "", {"aaa1"}) is False


def test_a_marked_row_still_occupies_exactly_one_row_width() -> None:
    """The marker is reserved as a column, so it cannot push the row past the
    card and silently eat the age and id columns."""
    row = _row("abc123def456", "a name long enough to need the whole budget here")
    plain = render_rows([row], 0, 74, NOW)[0].plain
    marked = render_rows([row], 0, 74, NOW, None, {"abc123def456"})[0].plain
    assert cell_len(marked) == cell_len(plain)
    assert BODY_MATCH_MARKER.strip() in marked
    assert BODY_MATCH_MARKER.strip() not in plain
    assert "abc123def456" in marked


def test_the_marker_never_pushes_the_name_below_its_floor() -> None:
    """The marker must not jump the queue in which the id and the age give up
    their cells before the name gives up any.

    Subtracting it from the name AFTER the budget was already spent down to
    the floor rendered marked names at 14 cells at several reachable widths.
    """
    rows = [_row("abc123def456", "a long conversation title"), _row("bbb222ccc333", "another")]
    for width in range(PICKER_MIN_WIDTH, 140):
        name_col, _, _ = plan_columns(rows, width, ["1m ago", "1h ago"], True)
        assert name_col >= NAME_MIN_CELLS, width


def test_every_row_starts_its_name_at_the_same_column() -> None:
    """Reserving the marker only on matched rows ragged the left edge of the
    one field the user reads down the list."""
    rows = [_row("abc123def456", "matched inside"), _row("bbb222ccc333", "not matched")]
    lines = [line.plain for line in render_rows(rows, 0, 74, NOW, None, {"abc123def456"})]
    # The marker column is reserved on BOTH rows, so each name begins at the
    # same offset: the marked row spends it on the mark, the other on blanks.
    assert lines[0].index("matched inside") == lines[1].index("not matched")


def test_a_marked_list_fills_the_card_at_every_reachable_width() -> None:
    rows = [_row("abc123def456", "a long conversation title"), _row("bbb222ccc333", "another")]
    for width in range(PICKER_MIN_WIDTH, 140):
        for line in render_rows(rows, 0, width, NOW, None, {"abc123def456"}):
            assert cell_len(line.plain) == width, width


@pytest.mark.asyncio
async def test_typing_finds_a_session_by_its_conversation_not_its_name() -> None:
    """End to end through the real screen: the query appears nowhere in the
    row's name, and the picker still hands that session back."""
    rows = [_row("aaa111", "a forgettable opening line"), _row("bbb222", "something else")]
    app = _PickerHost(rows, {"aaa111": "we fixed the retention sweep eviction"})
    async with app.run_test(size=(100, 30)) as pilot:
        await app.open_picker()
        await pilot.pause()
        for char in "retention":
            await pilot.press(char)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
    assert app.chosen == ["aaa111"]


def test_the_marker_column_does_not_depend_on_what_is_scrolled_into_view() -> None:
    """A page is not the result set.

    Deciding the reservation from the rows currently ON SCREEN made it vanish
    when the one marked row scrolled off, so every name jumped two cells
    sideways on a single arrow press and truncation changed for rows that had
    not changed. Needs more rows than a page holds -- the earlier fixtures
    could not scroll, which is why this was missed.
    """
    rows = [_row(f"id{index:09d}", f"session number {index}") for index in range(6)]
    marked = {"id000000000"}  # only the first row matched inside its body
    with_marked = render_rows(rows[0:3], 0, 74, NOW, None, marked)
    without_marked = render_rows(rows[3:6], 0, 74, NOW, None, marked)
    assert with_marked[0].plain.index("session number 0") == (
        without_marked[0].plain.index("session number 3")
    )


def test_an_unfiltered_list_reserves_no_marker_column() -> None:
    """The reservation costs nothing when nothing matched, so an ordinary
    open of the picker renders exactly as it did before the marker existed.

    Asserted against the marked rendering rather than against another call
    with the same arguments: comparing a no-match render to a no-match render
    passes however the reservation behaves, which is a test that cannot fail.
    """
    rows = [_row("abc123def456", "one"), _row("bbb222ccc333", "two")]
    unmarked = render_rows(rows, 0, 74, NOW, None, set())[0].plain
    marked = render_rows(rows, 0, 74, NOW, None, {"abc123def456"})[0].plain
    # The name starts flush against the cursor gutter when nothing matched,
    # and exactly the marker's width later when something did.
    assert unmarked.index("one") == GUTTER_CELLS
    assert marked.index("one") == GUTTER_CELLS + cell_len(BODY_MATCH_MARKER)


# --- relevance ranking ------------------------------------------------------
# Ordering is a property of the QUERY, applied by rank_rows only when a query is
# active, so the "no reorder under the cursor" invariant holds: the only event
# that reorders (a query change) is the same one that re-homes the cursor.


def test_ranking_orders_name_above_body_above_soft() -> None:
    """A query hit in the visible name outranks one in the body, which outranks
    a soft-only match — the tiered rule the design specifies."""
    name_hit = _row("aaa1", "classifier tuning", age_s=300)  # matches by name
    body_hit = _row("bbb2", "unrelated title", age_s=200)  # exact body match
    soft_hit = _row("ccc3", "another title", age_s=100)  # soft-only match
    rows = [name_hit, body_hit, soft_hit]
    # body_matches carries only the exact-body id; the soft id is in `rows`
    # (already filtered) but not in body_matches, so it takes the soft tier.
    ranked = rank_rows(rows, "classifier", {"bbb2"})
    assert [r.id for r in ranked] == ["aaa1", "bbb2", "ccc3"]


def test_ranking_breaks_ties_by_recency_within_a_tier() -> None:
    """Two rows in the same tier keep newest-first order (stable tie-break)."""
    older = _row("aaa1", "classifier one", age_s=500)
    newer = _row("bbb2", "classifier two", age_s=100)
    # Passed newest-first, as the picker builds them; both match by name.
    ranked = rank_rows([newer, older], "classifier", set())
    assert [r.id for r in ranked] == ["bbb2", "aaa1"]


def test_an_empty_query_keeps_recency_order_unchanged() -> None:
    """No query means no ordering: the list stays exactly as it arrived."""
    rows = [_row("aaa1", "one"), _row("bbb2", "two"), _row("ccc3", "three")]
    assert rank_rows(rows, "") == rows
    assert rank_rows(rows, "   ") == rows


def test_ranking_is_stable_for_a_fixed_query() -> None:
    """A fixed query must produce byte-for-byte the same order every call, so a
    repaint or resize never moves a row under the cursor."""
    rows = [_row("aaa1", "classifier"), _row("bbb2", "unrelated"), _row("ccc3", "classify")]
    first = rank_rows(rows, "class", {"bbb2"})
    second = rank_rows(rows, "class", {"bbb2"})
    assert [r.id for r in first] == [r.id for r in second]


@pytest.mark.asyncio
async def test_the_cursor_re_homes_to_the_top_match_on_a_query_change() -> None:
    """set_query pins the cursor to rank 0, so it tracks the best match rather
    than sitting on a row ranking then slides away from."""
    rows = [_row("aaa1", "one"), _row("bbb2", "two"), _row("ccc3", "three")]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        screen._move_to(2)  # move the cursor off the top
        assert screen.selected_index == 2
        screen.set_query("three")
        await pilot.pause()
        # Cursor is re-homed to index 0 (the top match), not clamped to the
        # last surviving row.
        assert screen.selected_index == 0
        assert screen.selected_id() == "ccc3"


@pytest.mark.asyncio
async def test_visible_rows_is_stable_across_repaints_for_a_fixed_query() -> None:
    """The invariant end to end: repeated repaints under a FIXED query never
    reorder the visible rows."""
    rows = [_row("aaa1", "classifier"), _row("bbb2", "unrelated"), _row("ccc3", "classify")]
    digests = {"bbb2": "a body mentioning classifier deep inside"}
    app = _PickerHost(rows, digests)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        screen.set_query("classifier")
        await pilot.pause()
        first = [r.id for r in screen.visible_rows]
        screen._repaint()
        screen._repaint()
        assert [r.id for r in screen.visible_rows] == first


@pytest.mark.asyncio
async def test_a_row_matched_only_by_a_soft_query_is_shown_and_marked() -> None:
    """A soft match (typo) surfaces the row via the body path and carries the
    body-match marker, since it did not match the visible name."""
    rows = [_row("aaa1", "vague title"), _row("bbb2", "another")]
    digests = {"aaa1": "improve adm classifier throughput"}
    app = _PickerHost(rows, digests)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        screen.set_query("classifer")  # typo, soft match only
        await pilot.pause()
        assert [r.id for r in screen.visible_rows] == ["aaa1"]
        assert screen.body_matched_ids == {"aaa1"}


@pytest.mark.asyncio
async def test_a_row_matched_only_by_a_past_name_is_shown_and_marked() -> None:
    """A session found by a name it was renamed AWAY from surfaces via the body
    path (the digest folds past names in) and carries the marker."""
    rows = [_row("aaa1", "Current Title"), _row("bbb2", "another")]
    # The digest carries a PAST name the visible row name no longer shows.
    digests = {"aaa1": "Old Abandoned Name the session body text"}
    app = _PickerHost(rows, digests)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        screen.set_query("abandoned")
        await pilot.pause()
        assert [r.id for r in screen.visible_rows] == ["aaa1"]
        assert screen.body_matched_ids == {"aaa1"}


def test_footer_legend_appears_only_when_a_row_is_marked() -> None:
    """D2: the ``"`` marker is meaningless without a legend, but advertising it
    when nothing is marked would explain a glyph the user cannot see. So the
    legend is present exactly when the row has room AND a marked row exists,
    and absent otherwise."""
    # A card at its widest, in BOTH list shapes: the counter shares this row on
    # a scrolling list, and a legend that only fits when it is absent is the
    # unreachable-in-the-ordinary-case defect of round 2 (D2).
    for counter_cells in (0, len("showing 1–10 of 501")):
        with_legend = _meta_legends(
            74, has_marked=True, has_exec=False, counter_cells=counter_cells
        )
        assert _MARKER_LEGEND in with_legend, f"unreachable at counter_cells={counter_cells}"
        # Same width, nothing marked: no legend.
        without = _meta_legends(74, has_marked=False, has_exec=False, counter_cells=counter_cells)
        assert _MARKER_LEGEND not in without
    # The key row is untouched either way — legends no longer buy space from it.
    # (``scrolls=True``: a list that fits one page does not advertise paging at
    # all any more, so the existence of the hint is asserted where it applies.)
    assert ("pgup/pgdn", "page") in _footer_hints(74, scrolls=True)


def test_footer_legend_drops_before_the_movement_and_action_keys() -> None:
    """The legend teaches; the keys OPERATE the card, so they never compete.

    Since round 2 (D2) they do not even share a row: a legend sheds against the
    counter's row and cannot evict a key at all. What must still hold is that a
    card too narrow for a legend keeps every essential key, and that a legend
    never survives as a bare unlabelled glyph (the artifact-looking mark D2
    flagged in round 1).
    """
    # Wide: legend shown, and the full key row survives beside it.
    wide = _meta_legends(74, has_marked=True, has_exec=False)
    assert _MARKER_LEGEND in wide
    assert ("pgup/pgdn", "page") in _footer_hints(74, scrolls=True)
    # Narrow: the legend is gone entirely — not reduced to a lone glyph — and
    # the essential keys survive.
    narrow = _meta_legends(30, has_marked=True, has_exec=False, counter_cells=19)
    assert _MARKER_LEGEND not in narrow
    assert (_MARKER_LEGEND[0], "") not in narrow
    keys = _footer_hints(40)
    assert ("enter", "resume") in keys
    assert ("esc", "cancel") in keys


@pytest.mark.asyncio
async def test_the_soft_tier_is_skipped_once_the_exact_tiers_fill_a_page() -> None:
    """The soft tier's first call tokenises every digest and builds a vocabulary
    over them — 324 ms and 95 MB resident over the 2681-digest store the picker
    now reaches uncapped, against ~5 ms for the exact tier.

    Since the picker went uncapped that build sat on the first character typed.
    It is deferred behind the cheap tiers: a query the exact tiers can answer at
    all has nothing for soft matching to rescue.
    """
    rows = [_row(f"s{i:03d}", f"classifier run {i}") for i in range(15)]
    digests = {row.id: "body text" for row in rows}
    app = _PickerHost(rows, digests)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        calls: list[str] = []
        real = screen._soft_index.search

        def counting(digests_arg, query):
            calls.append(query)
            return real(digests_arg, query)

        screen._soft_index.search = counting  # type: ignore[method-assign]
        screen.set_query("classifier")
        await pilot.pause()

        assert len(screen.visible_rows) == len(rows)
        assert calls == [], "the soft tier ran for a query the exact tiers already answered"


@pytest.mark.asyncio
async def test_the_soft_tier_still_runs_when_the_exact_tiers_come_up_short() -> None:
    """The other half of the gate: a typo the exact tiers cannot answer must
    still reach the soft tier, or deferring it would silently cost recall."""
    rows = [_row("aaa1", "vague title"), _row("bbb2", "another")]
    digests = {"aaa1": "improve adm classifier throughput"}
    app = _PickerHost(rows, digests)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        screen.set_query("classifer")  # typo: no exact hit anywhere
        await pilot.pause()
        assert [r.id for r in screen.visible_rows] == ["aaa1"]
        assert screen.body_matched_ids == {"aaa1"}


@pytest.mark.asyncio
async def test_the_header_tally_reports_the_stores_true_total() -> None:
    """Once the cap is gone ``_all`` IS the store's user-session total, so the
    counter stops reporting a number that is not the total and never said so.

    The filtered ``showing a-b of N`` counter keeps reporting the FILTERED
    count — the two answer different questions and must not converge.
    """
    rows = [_row(f"s{i:04d}", f"session {i}") for i in range(2700)]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        # Grouped: the separator is part of the fix for reading a 4-digit
        # total at a glance (design round 1, D4).
        # The tally and the position counter both live in the filter row now.
        # Unfiltered, a list this long scrolls, so the row states the store's
        # true total as the counter's denominator — grouped, because a 4-digit
        # total is read at a glance and parsed as a digit string otherwise
        # (design round 1, D4).
        assert f"of {len(rows):,}" in screen.render_footer_for_test()

        screen.set_query("session 1")
        await pilot.pause()
        footer = screen.render_footer_for_test()
        shown = len(screen.visible_rows)
        # The position counter reports the FILTERED count: the two answer
        # different questions and must not converge.
        assert f"of {shown:,}" in footer
        assert shown != len(rows)


@pytest.mark.asyncio
async def test_a_large_row_set_does_not_move_a_row_under_the_cursor() -> None:
    """R1 at the scale the uncapped picker now reaches.

    A fixed query must order rows identically across repaints AND a resize, and
    the row under the cursor must keep its identity — a row moving out from
    under the cursor is how a user resumes the wrong session.
    """
    rows = [_row(f"s{i:04d}", f"session {i} work") for i in range(2700)]
    digests = {row.id: f"body {row.id}" for row in rows}
    app = _PickerHost(rows, digests)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        screen.set_query("session 12")
        await pilot.pause()
        first = [r.id for r in screen.visible_rows]
        screen.action_move(1)
        screen.action_move(1)
        selected = screen.selected_id()

        screen._repaint()
        screen._repaint()
        assert [r.id for r in screen.visible_rows] == first
        assert screen.selected_id() == selected

        # A resize repaints at a different width; ordering is a pure function of
        # the query, so it must survive that too.
        await pilot.resize_terminal(60, 30)
        await pilot.pause()
        assert [r.id for r in screen.visible_rows] == first
        assert screen.selected_id() == selected


@pytest.mark.asyncio
async def test_the_result_list_never_grows_as_the_user_types() -> None:
    """The filter must narrow monotonically within a typing run.

    The first soft-tier gate fired when the exact tiers returned fewer than a
    page, which is a threshold typing CROSSES: on the operator's real store,
    `watc` and `watch` showed 41 rows and `watchl` showed 1519 — the list grew
    twentyfold on a longer query, every row acquired a body-match marker, and
    the card changed height under the cursor.

    The gate that replaced it (soft when the exact tiers return NOTHING) failed
    the same way for a different reason: on the operator's real store `watch`
    has 10 genuine matches and `watchl` has none, so the exact set empties, the
    tier fires, and 46 typo-neighbours replace all 10 real matches — the top row
    changes identity on the keystroke before the user presses enter.

    The fixture mirrors that real shape and is built to DISCRIMINATE: the
    previous version made every digest contain `watch`, so the counts were
    constant and the assertion passed on any gate, including the two broken
    ones. Verified to fail on both before being trusted.
    """
    rows = [_row(f"s{i:03d}", f"session {i}") for i in range(50)]
    digests = {}
    for i, row in enumerate(rows):
        if i < 10:
            # Genuine matches for `watch`, and NOT for `watchl` — so the exact
            # set is non-empty at `watch` and empty one keystroke later. This is
            # what makes a zero-hit gate fire mid-word.
            digests[row.id] = "watch the retention sweep"
        elif i < 12:
            # A second, smaller exact population so a count threshold has a
            # boundary to cross: `wa` admits 12, `watch` admits 10.
            digests[row.id] = "wander through the logs"
        else:
            # Edit-distance neighbours of `watchl` (`wotchel` is within two edits
            # away) that contain neither `watch` nor `watchl` as a SUBSTRING, so
            # the exact tier never admits them at any keystroke and only the
            # soft tier can reach them. A neighbour that merely contains the
            # query would be admitted legitimately and prove nothing.
            digests[row.id] = "wotchel batch rollup"
    app = _PickerHost(rows, digests)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()

        counts: list[int] = []
        seen: set[str] | None = None
        removed: list[tuple[str, int]] = []
        # Real keystrokes, not ``set_query`` per whole word: the gate this
        # guards is decided per typing RUN, and jumping straight to each final
        # query cannot observe a run at all. That shape is how a tier that
        # never ran was validated as working for two rounds.
        typed = ""
        for key in "watchl":
            await pilot.press(key)
            typed += key
            await pilot.pause()
            query = typed
            ids = {row.id for row in screen.visible_rows}
            counts.append(len(ids))
            if seen is not None:
                gone = seen - ids
                # A row leaving because it genuinely stopped matching the longer
                # query is narrowing, not eviction. Eviction is a row leaving on
                # a keystroke that ALSO admitted new ones — the list swapping
                # its contents rather than narrowing them.
                if gone and (ids - seen):
                    removed.append((query, len(gone)))
            seen = ids

        # The property is NOT "the count never rises". Once the soft tier
        # latches on (at the first keystroke whose exact tiers find nothing) it
        # legitimately admits rows, and it must — a gate that kept the count
        # monotonic by never running the tier made typo search unreachable from
        # a keyboard, which is a worse bug than a count that goes up.
        #
        # What must hold is that rows are never admitted while the user still
        # has results to read: growth is only allowed out of an EMPTY list,
        # where there is nothing on screen to displace.
        # Rows may be ADDED when the soft tier latches on — that is the tier
        # doing its job, and it is what makes a typo findable. What must never
        # happen is a row the user was already reading being REMOVED to make
        # room, which is the eviction that swapped the cursor in round 2.
        assert not removed, f"rows the user was already shown were withdrawn: {removed}"


@pytest.mark.asyncio
async def test_the_same_query_renders_identically_by_either_route() -> None:
    """The property this surface actually guarantees: order is a function of the
    QUERY, not of the route taken to it.

    A user cannot know which route they took, so a picker that answers the same
    visible query two ways is one they cannot reason about at all. Reaching
    `watchl` by typing it, and by typing `watchlq` then backspacing, must give
    byte-identical rows in byte-identical order.

    This replaces a test asserting that the cursor never lands on a row absent
    one keystroke earlier. That property was real but could only be held by
    ranking on run history, which is exactly what made the same query render
    two ways (D11). The two are not both satisfiable by ordering; see
    ``_soft_tier_wanted``. What remains of the displacement concern is recorded
    honestly in that docstring rather than asserted here falsely.
    """
    rows = [_row(f"s{i:03d}", f"session {i}", age_s=60.0 * (100 - i)) for i in range(50)]
    digests = {
        row.id: ("watch the retention sweep" if i >= 40 else "wotchel batch rollup")
        for i, row in enumerate(rows)
    }

    async def route(extra: str | None) -> list[str]:
        app = _PickerHost(rows, digests)
        async with app.run_test(size=(100, 30)) as pilot:
            screen = await app.open_picker()
            await pilot.pause()
            for key in "watchl" if extra is None else "watchl" + extra:
                await pilot.press(key)
                await pilot.pause()
            if extra is not None:
                for _ in extra:
                    await pilot.press("backspace")
                    await pilot.pause()
            return [row.id for row in screen.visible_rows]

    forward = await route(None)
    backspaced = await route("q")
    assert (
        forward == backspaced
    ), f"the same query rendered differently by route: {forward[:5]} vs {backspaced[:5]}"


@pytest.mark.asyncio
async def test_the_soft_tier_runs_only_when_the_exact_tiers_are_empty() -> None:
    """The one-line rule the gate implements, observed rather than reconstructed.

    The soft tier runs ONLY for a query the cheap tiers cannot answer at all.
    That is what bounds the disruption: with exact hits the user keeps exactly
    those, and with none there is nothing on screen for an addition to displace.

    Deliberately NOT asserting "the list never grows" or "no row appears that
    was absent one keystroke earlier". Both were asserted in earlier rounds and
    both are false on real data (`sesion` typed goes 83 to 129 rows with the
    tier active on both sides). A guard stating a false invariant passes only
    until someone measures it.
    """
    rows = [_row(f"s{i:03d}", f"session {i}", age_s=60.0 * (100 - i)) for i in range(50)]
    digests = {
        row.id: ("watch the retention sweep" if i >= 40 else "wotchel batch rollup")
        for i, row in enumerate(rows)
    }
    app = _PickerHost(rows, digests)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()

        ran: list[str] = []
        real = screen._soft_index.search

        def counting(digests_arg, query):
            ran.append(query)
            return real(digests_arg, query)

        screen._soft_index.search = counting  # type: ignore[method-assign]

        typed = ""
        for key in "watchl":
            await pilot.press(key)
            await pilot.pause()
            typed += key
            # The gate is NAME/ID hits, not any exact hit. Gating on any hit
            # (which includes body substrings) is what destroyed typo recall:
            # one unrelated conversation mentioning the query silenced the tier
            # for the whole store. Computed the way the widget computes it so
            # the two cannot drift.
            precise = [row for row in rows if typed in row.name.lower() or typed in row.id.lower()]
            if precise:
                assert typed not in ran, f"soft tier ran at {typed!r} despite a name/id hit"
            else:
                assert typed in ran, f"soft tier did not run at {typed!r} with no name/id hit"


def test_the_paging_hint_outranks_the_filter_hint_once_the_list_scrolls() -> None:
    """The paging hint was offered where paging does nothing and withdrawn where
    it is the fastest way through the list (design round 1, D3).

    Round 2 moved the legends off this row, so the competition `pgup/pgdn` now
    survives is against `type to filter` — the genuinely disposable hint, since
    a user who is filtering already knows they can type and the query is echoed
    in the header regardless.
    """
    width = 56  # narrow enough that both disposable hints cannot fit

    not_scrolling = [key for key, _ in _footer_hints(width, scrolls=False)]
    scrolling = [key for key, _ in _footer_hints(width, scrolls=True)]

    assert "pgup/pgdn" not in not_scrolling, "paging is a no-op on a list that fits"
    assert "pgup/pgdn" in scrolling, "paging must survive when the list actually pages"
    assert "type" not in scrolling

    # A card at its widest keeps both: the reorder is a shed policy, not a
    # removal, so nothing is lost when there is room for everything.
    wide = [key for key, _ in _footer_hints(74, scrolls=True)]
    assert "pgup/pgdn" in wide and "type" in wide


def test_the_paging_hint_survives_on_a_plain_scrolling_list() -> None:
    """The same rule on the UNMARKED branch, which is the common one.

    The first version of this fix consulted ``scrolls`` only inside
    ``if has_marked:``, so a plain scrolling picker still shed ``pgup/pgdn``
    first at 60 and 66 columns. Uncapping the store makes the bare scrolling
    picker the DEFAULT state, so that was the usual case rather than an edge.
    """
    for width in (56, 60, 66):
        scrolling = [key for key, _ in _footer_hints(width, scrolls=True)]
        assert "pgup/pgdn" in scrolling, f"paging hint dropped at width {width}: {scrolling}"

    # A list that fits on one page may still shed it first: there is nothing to
    # page through, so the hint is the least useful thing on the row.
    settled = [key for key, _ in _footer_hints(60, scrolls=False)]
    assert "pgup/pgdn" not in settled


def test_the_empty_state_footer_offers_only_what_works() -> None:
    """A filter that matched nothing has no rows to move through, page, or
    resume, so advertising those keys names actions that do nothing.

    `backspace` is what widens the query and is the key a user in this state is
    already reaching for; `esc` stays because leaving is still available.

    `enter` ALSO STAYS, because it does something here: it closes the picker
    (``action_choose`` dismisses with ``None`` on an empty result set). That is
    deliberate, and it was the one thing the row did not say — pressing Enter to
    "accept" a filter looked like a silent crash (UX round 1, U8).
    """
    hints = _footer_hints(74, empty=True)
    keys = [key for key, _ in hints]
    assert keys == ["backspace", "enter", "esc"], keys

    # Fits the narrow cards too, where the tally has already shed.
    for width in (50, 56, 60):
        assert [key for key, _ in _footer_hints(width, empty=True)] == [
            "backspace",
            "enter",
            "esc",
        ]

    # And the populated footer is untouched.
    populated = [key for key, _ in _footer_hints(74, empty=False)]
    assert "↑↓" in populated and "enter" in populated


def test_the_running_marker_animates_and_idle_is_not_the_attached_glyph() -> None:
    """D1 and D6, the two findings a still frame caught and no test did.

    D1: `render_rows` was correct for every frame index, but the ONE call site
    never passed one, so the marker sat on frame 0 forever. A frozen braille
    dot does not read as "busy" — it reads as a static bullet, which is the
    marker for a different state, collapsing the one distinction the picker
    exists to make.

    D6: `attached` and `idle` shared `○`, separated only by muted-vs-dim ink at
    1.90:1. Two different facts (someone else is watching / nobody is, it is
    just warm) need two different glyphs, not two brightnesses of one.
    """
    from local_operator.tui.terminal_title import SPINNER_FRAMES
    from local_operator.tui.widgets.session_picker import (
        ATTACHED_MARKER,
        IDLE_MARKER,
        row_state_mark,
    )

    busy = _row("busy00000001", "a running session")._replace(live_state="busy")
    glyphs = {row_state_mark(busy, frame)[0] for frame in range(len(SPINNER_FRAMES))}
    assert glyphs == set(SPINNER_FRAMES), "the running marker does not cover its cycle"

    attached = _row("attach000002", "watched elsewhere")._replace(live_state="attached")
    idle = _row("idle00000003", "an idle session")._replace(live_state="idle")
    assert row_state_mark(attached, 0)[0] == ATTACHED_MARKER
    assert row_state_mark(idle, 0)[0] == IDLE_MARKER
    assert row_state_mark(attached, 0)[0] != row_state_mark(idle, 0)[0]


def test_every_state_marker_is_exactly_one_cell() -> None:
    """The column reserves one cell plus a separator.

    A two-cell glyph eats the separator and the name starts flush against it —
    how `⏰` was caught. Asserted over the whole marker set so a new one cannot
    reintroduce it, including the idle glyph D6 added.
    """
    from rich.cells import cell_len

    from local_operator.tui.terminal_title import SPINNER_FRAMES
    from local_operator.tui.widgets.session_picker import (
        ATTACHED_MARKER,
        IDLE_MARKER,
        NEEDS_YOU_MARKER,
        WAKE_MARKER,
        WEDGED_MARKER,
    )

    markers = (
        NEEDS_YOU_MARKER,
        WEDGED_MARKER,
        ATTACHED_MARKER,
        IDLE_MARKER,
        WAKE_MARKER,
        *SPINNER_FRAMES,
    )
    assert {cell_len(marker) for marker in markers} == {1}


def test_the_painted_cursor_and_enter_agree_after_a_reorder() -> None:
    """The row under the cursor must be the row Enter resumes.

    `_tick` reassigns `self._all` from a refresh that REORDERS (needs-you
    sorts first) and `_selected` is an index into that order. Skipping the
    repaint therefore left the SCREEN in the old order while Enter resolved
    against the new one: the cursor sat on `alpha` and Enter resumed `beta`
    (round 3, D10). That fires on this release's headline event — a detached
    session parking on a gate, with nothing spinning.

    Asserted against what `_repaint` actually pushed to the body, NOT against
    `render_lines_for_test`: that helper recomputes the card from current
    state, so it can never show a stale frame and cannot catch this class.
    """
    import time

    from local_operator.resume import SessionRow
    from local_operator.tui.widgets.session_picker import SessionPickerScreen

    now = time.time()
    rows = [
        SessionRow(id="aaaaaaaaaaa1", mtime=now, name="alpha the top row", live_state="idle"),
        SessionRow(id="bbbbbbbbbbb2", mtime=now, name="beta", live_state="idle"),
    ]
    parked = {"yes": False}

    def refresh(current: list[SessionRow]) -> list[SessionRow]:
        out = list(current)
        if parked["yes"]:
            out = [r._replace(pending="approval") if r.id == "bbbbbbbbbbb2" else r for r in out]
            out.sort(key=lambda r: (getattr(r, "pending", None) is None,))
        return out

    screen = SessionPickerScreen(rows, now, refresh_live_state=refresh)
    screen._selected = 0

    painted: dict[str, list[str]] = {}

    class _Body:
        is_mounted = True

        def update(self, text: object) -> None:
            lines = text.split("\n")  # type: ignore[attr-defined]
            painted["lines"] = [line.plain for line in lines]

    screen._results = _Body()  # type: ignore[assignment]
    screen._repaint()
    assert "alpha" in next(ln for ln in painted["lines"] if ln.strip().startswith("❯"))

    # beta parks while the picker is open. Nothing is busy — the state the
    # previous guard returned early on.
    parked["yes"] = True
    screen._tick()

    cursor_line = next(ln for ln in painted["lines"] if ln.strip().startswith("❯"))
    would_resume = screen._all[screen._selected].id
    assert "beta" in cursor_line, "the screen kept the pre-reorder frame"
    assert would_resume == "bbbbbbbbbbb2"


def test_a_marker_change_with_nothing_busy_still_repaints() -> None:
    """Every non-busy marker transition froze too (D10's second half).

    idle→wedged, idle→attached, record-gone and wake-armed all changed the
    row's meaning while the screen kept the old glyph, because the repaint
    was tied to the spinner rather than to the data.
    """
    import time

    from local_operator.resume import SessionRow
    from local_operator.tui.widgets.session_picker import SessionPickerScreen

    now = time.time()
    rows = [SessionRow(id="aaaaaaaaaaa1", mtime=now, name="only", live_state="idle")]
    state = {"live": "idle"}

    def refresh(current: list[SessionRow]) -> list[SessionRow]:
        return [r._replace(live_state=state["live"]) for r in current]

    screen = SessionPickerScreen(rows, now, refresh_live_state=refresh)
    repaints = {"n": 0}

    class _Body:
        is_mounted = True

        def update(self, text: object) -> None:
            repaints["n"] += 1

    screen._results = _Body()  # type: ignore[assignment]

    screen._tick()
    assert repaints["n"] == 0, "an unchanged settled store must stay cheap"

    state["live"] = "wedged"
    # The live-marker read is on a BOUNDED cadence now (`LIVE_REFRESH_INTERVAL_S`
    # in the widget), so this test drives the refresh explicitly instead of
    # assuming the frame rate is it. What is pinned here is the
    # repaint-on-visible-change property; the cadence has its own test
    # (`test_the_overlay_is_not_re_run_on_every_tick`).
    screen._live_refreshed_at = float("-inf")
    screen._tick()
    assert repaints["n"] == 1, "a marker transition must reach the screen"


def test_the_overlay_is_not_re_run_on_every_tick(monkeypatch) -> None:
    """The overlay's cost was per FRAME; the design says per visible change.

    `_tick` runs at `SPINNER_INTERVAL_S` (12.5 Hz) and called `refresh`
    unconditionally — one `registry.scan()` (a glob, a JSON parse per record, a
    `ps` per quiet-heartbeat record) plus `read_index()` — while its own
    docstring claimed the refresh was "skipped entirely when no row is
    animating". An open picker over a store of cold sessions therefore
    re-scanned the whole store twelve times a second to discover that nothing
    had changed.

    Driven with a fake clock so the cadence is asserted EXACTLY rather than
    inferred from how fast the test machine happened to run: the property is a
    ratio of refreshes to frames, and nothing here measures a duration.
    """
    import time as real_time

    from local_operator.resume import SessionRow
    from local_operator.tui.widgets import session_picker as picker_module
    from local_operator.tui.widgets.session_picker import (
        LIVE_REFRESH_INTERVAL_S,
        SPINNER_INTERVAL_S,
        SessionPickerScreen,
    )

    clock = {"t": 0.0}

    class _Clock:
        # Patched onto the widget's own module reference, not onto `time`
        # itself: only this picker sees it, and pytest's own timing is untouched.
        @staticmethod
        def monotonic() -> float:
            return clock["t"]

    monkeypatch.setattr(picker_module, "time", _Clock)

    now = real_time.time()
    rows = [SessionRow(id="aaaaaaaaaaa1", mtime=now, name="only", live_state="idle")]
    refreshes = {"n": 0}

    def refresh(current: list[SessionRow]) -> list[SessionRow]:
        refreshes["n"] += 1
        return list(current)

    screen = SessionPickerScreen(rows, now, refresh_live_state=refresh)

    class _Body:
        is_mounted = True

        def update(self, _text: object) -> None:
            return None

    screen._body = _Body()  # type: ignore[assignment]

    # One second of frames (0.00 through 0.96 s) — one refresh, not thirteen.
    for tick in range(13):
        clock["t"] = tick * SPINNER_INTERVAL_S
        screen._tick()
    assert refreshes["n"] == 1, "the overlay was re-run once per frame"

    # A CADENCE, not a one-off: the next interval refreshes again, which is what
    # keeps a row that REORDERS (a session parking on a gate, nothing animating)
    # from freezing on screen — the property the D10 tests above pin.
    clock["t"] += LIVE_REFRESH_INTERVAL_S
    screen._tick()
    assert refreshes["n"] == 2

    # ...and the frame counter still runs at the FRAME rate while a row is
    # busy, because that is motion and it is the honest thing for a running
    # marker to show. The data stays on the slower cadence.
    screen._all = [row._replace(live_state="busy") for row in screen._all]
    before_frames = screen._frame
    for _ in range(5):
        clock["t"] += SPINNER_INTERVAL_S
        screen._tick()
    assert screen._frame == before_frames + 5, "the spinner stopped animating"


def test_a_pure_reorder_with_identical_content_still_repaints() -> None:
    """The gate handoff: two rows SWAP marker state, so the multiset is equal.

    This is the permutation the round-3 D10 fix did not catch. Its signature
    read `session_id` — a field `SessionRow` does not have — so `getattr`'s
    default made identity the empty string on every row, and a reorder that
    preserves the multiset of `(live_state, pending, wakes)` tuples compared
    EQUAL. `_tick` returned early and the screen kept the pre-reorder frame
    while Enter resolved against the new order (round 4, D10).

    Both existing D10 tests pass with that bug live, because each changes
    tuple CONTENT (a pending appears, idle->wedged). Only a content-preserving
    permutation needs identity in the comparison, which is why this test
    exists alongside them.
    """
    import time

    from local_operator.resume import SessionRow
    from local_operator.tui.widgets.session_picker import SessionPickerScreen

    now = time.time()
    # alpha holds the gate; beta is idle.
    rows = [
        SessionRow(
            id="aaaaaaaaaaa1", mtime=now, name="alpha", live_state="idle", pending="approval"
        ),
        SessionRow(id="bbbbbbbbbbb2", mtime=now, name="beta", live_state="idle"),
    ]
    handed_off = {"yes": False}

    def refresh(current: list[SessionRow]) -> list[SessionRow]:
        if not handed_off["yes"]:
            return list(current)
        # alpha's gate is answered and beta parks one: the SAME multiset of
        # marker tuples, in the opposite order, needs-you sorted first.
        swapped = [
            r._replace(pending="approval" if r.id == "bbbbbbbbbbb2" else None) for r in current
        ]
        swapped.sort(key=lambda r: (r.pending is None,))
        return swapped

    screen = SessionPickerScreen(rows, now, refresh_live_state=refresh)
    screen._selected = 0
    painted: dict[str, list[str]] = {}

    class _Body:
        is_mounted = True

        def update(self, text: object) -> None:
            lines = text.split("\n")  # type: ignore[attr-defined]
            painted["lines"] = [line.plain for line in lines]

    screen._results = _Body()  # type: ignore[assignment]
    screen._repaint()
    assert "alpha" in next(ln for ln in painted["lines"] if ln.strip().startswith("❯"))

    handed_off["yes"] = True
    screen._tick()

    cursor_line = next(ln for ln in painted["lines"] if ln.strip().startswith("❯"))
    would_resume = screen._all[screen._selected].id
    assert would_resume == "bbbbbbbbbbb2"
    assert "beta" in cursor_line, (
        "a content-preserving reorder must still repaint — the cursor is an "
        "index into the order Enter resolves against"
    )


def test_the_signature_names_only_real_row_fields() -> None:
    """A misspelled field name silently drops a column from the comparison.

    `getattr(row, "session_id", "")` returns the default forever rather than
    raising, which is exactly how D10 survived its own fix. The import-time
    assertion in the screen guards this; this test pins it so a rename in
    `resume.SessionRow` cannot quietly re-open the hole.
    """
    from local_operator.resume import SessionRow
    from local_operator.tui.widgets.session_picker import SessionPickerScreen

    unknown = set(SessionPickerScreen._SIGNATURE_FIELDS) - set(SessionRow._fields)
    assert not unknown, f"signature names fields SessionRow does not have: {sorted(unknown)}"
    # Identity must be in it, or a pure reorder compares equal.
    assert "id" in SessionPickerScreen._SIGNATURE_FIELDS


def test_an_armed_wake_outranks_idle_but_not_attached() -> None:
    """The reported ask: "show the wake symbol if it's just scheduled wakes".

    Under the original precedence the wake glyph was UNREACHABLE for any live
    session: a session with a wake armed is by definition resident, so
    `idle`/`attached` matched first and the wake was only ever visible on a
    cold row — one with no runtime to fire it. That is backwards, because the
    cold row is the one where the schedule means least.

    `idle` is the least informative thing true of a row (every resident session
    has it); "this will act on its own later" is a fact nothing else conveys.

    `attached` is NOT that, and this test asserts the distinction because the
    first cut of the change got it wrong (round 1, D2): `○` means *a terminal
    is watching this session*, which on a list the user is scanning is the one
    mark that answers "where am I?". A wake outranks bare residency and loses
    to presence.
    """
    from local_operator.tui.widgets.session_picker import (
        ATTACHED_MARKER,
        IDLE_MARKER,
        WAKE_MARKER,
        row_state_mark,
    )

    idle = _row("wake00000001", "armed")._replace(live_state="idle", wakes=2)
    assert row_state_mark(idle, 0)[0] == WAKE_MARKER

    watched = _row("wake00000003", "armed and watched")._replace(live_state="attached", wakes=2)
    assert row_state_mark(watched, 0)[0] == ATTACHED_MARKER

    # A cold row with an armed wake still shows it — that never regressed.
    cold = _row("wake00000004", "cold armed")._replace(wakes=1)
    assert row_state_mark(cold, 0)[0] == WAKE_MARKER

    # And with no wake armed, presence still renders exactly as before.
    plain = _row("idle00000002", "no wake")._replace(live_state="idle")
    assert row_state_mark(plain, 0)[0] == IDLE_MARKER


def test_a_dormant_wake_stays_below_presence() -> None:
    """A stopped session's schedule is not going to fire, so it must not win.

    Promoting it would advertise a future that is not coming over a runtime
    that is genuinely here. On a COLD row it still renders, dimmed, because
    there it is the last thing worth saying about the session.
    """
    from local_operator.tui.widgets.session_picker import (
        IDLE_MARKER,
        WAKE_MARKER,
        row_state_mark,
    )

    live = _row("dorm00000001", "stopped but resident")._replace(
        live_state="idle", wakes=1, wakes_dormant=True
    )
    assert row_state_mark(live, 0)[0] == IDLE_MARKER

    cold = _row("dorm00000002", "cold")._replace(wakes=1, wakes_dormant=True)
    glyph, ink = row_state_mark(cold, 0)
    assert glyph == WAKE_MARKER
    assert ink == "dim", "a dormant wake must read as quieter than an armed one"


def test_urgency_still_outranks_an_armed_wake() -> None:
    """Promoting wakes must not have demoted anything above them.

    A person waiting and a broken runtime are both more urgent than a schedule,
    and a live turn is what the spinner exists for.
    """
    from local_operator.tui.terminal_title import SPINNER_FRAMES
    from local_operator.tui.widgets.session_picker import (
        NEEDS_YOU_MARKER,
        WEDGED_MARKER,
        row_state_mark,
    )

    base = dict(wakes=3, wakes_dormant=False)
    pending = _row("urg000000001", "waiting")._replace(pending="approval", **base)
    assert row_state_mark(pending, 0)[0] == NEEDS_YOU_MARKER

    wedged = _row("urg000000002", "broken")._replace(live_state="wedged", **base)
    assert row_state_mark(wedged, 0)[0] == WEDGED_MARKER

    busy = _row("urg000000003", "working")._replace(live_state="busy", **base)
    assert row_state_mark(busy, 0)[0] in set(SPINNER_FRAMES)


# --- the two-pane telescope layout ------------------------------------------


@pytest.mark.asyncio
async def test_the_picker_draws_more_than_ten_rows_on_a_tall_terminal() -> None:
    """``PAGE_ROWS_MAX`` drew 10 rows out of 140 at EVERY terminal height — 7%,
    while a 60-row terminal has room for 41. Fails on ``origin/main``.
    """
    rows = [_row(f"{index:012x}", f"session {index}") for index in range(60)]
    app = _PickerHost(rows)
    async with app.run_test(size=(160, 50)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        drawn = screen.render_lines_for_test()
    assert len(drawn) > 10, f"only {len(drawn)} rows drawn on a 50-row terminal"


@pytest.mark.asyncio
async def test_the_layout_flips_at_the_breakpoint_and_back(tmp_path: Path) -> None:
    """THE LIVE-RESIZE CRITERION. A mode-only restyle guard fails this: the
    preview's scroll offset and the cursor must survive the round trip, which
    means the widget tree is restyled in place and never rebuilt.
    """
    rows = [_row(f"{index:012x}", f"session {index}") for index in range(40)]
    # A real transcript on the row the cursor lands on, so the preview has
    # something to SCROLL — without it `ctrl+d` is a no-op and the round trip
    # would prove nothing.
    _write_transcript(
        tmp_path,
        f"{3:012x}",
        [_message("user", f"turn {index} " + "body text " * 12) for index in range(40)],
    )
    app = _PickerHost(rows)
    async with app.run_test(size=(180, 50)) as pilot:
        screen = await app.open_picker()
        screen.use_previews_for_test(tmp_path / "sessions")
        await pilot.pause()
        assert screen.layout_mode_for_test() == "side-by-side"

        await pilot.press("down", "down", "down")
        await pilot.press("ctrl+d")
        await pilot.pause()
        cursor = screen.selected_index
        offset = screen.preview_offset_for_test()
        assert offset > 0, "ctrl+d did not scroll the preview, so the round trip proves nothing"

        await pilot.resize_terminal(120, 35)
        await pilot.pause()
        assert screen.layout_mode_for_test() == "stacked"

        await pilot.resize_terminal(180, 50)
        await pilot.pause()
        assert screen.layout_mode_for_test() == "side-by-side"
        assert screen.selected_index == cursor
        assert screen.preview_offset_for_test() == offset


@pytest.mark.asyncio
async def test_ctrl_e_toggles_verbose_and_the_mode_is_sticky_across_rows() -> None:
    """A mode that resets as the cursor moves is a mode the user re-sets on every row."""
    rows = [_row(f"{index:012x}", f"session {index}") for index in range(10)]
    app = _PickerHost(rows)
    async with app.run_test(size=(160, 45)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        assert screen.preview_mode_for_test() == "condensed", "condensed is the DEFAULT"

        await pilot.press("ctrl+e")
        await pilot.pause()
        assert screen.preview_mode_for_test() == "verbose"

        await pilot.press("down")
        await pilot.pause()
        assert screen.preview_mode_for_test() == "verbose", "the toggle is not sticky"

        await pilot.press("ctrl+e")
        await pilot.pause()
        assert screen.preview_mode_for_test() == "condensed"


@pytest.mark.asyncio
async def test_ctrl_u_and_ctrl_d_scroll_the_preview_without_moving_the_cursor(
    tmp_path: Path,
) -> None:
    """The preview scrolls under a stationary cursor, or paging it costs the row."""
    session_id = "aa11bb22cc33"
    _write_transcript(
        tmp_path,
        session_id,
        [_message("user", f"turn {index} " + "body text " * 12) for index in range(40)],
    )
    rows = [_row(session_id, "long conversation")]
    app = _PickerHost(rows)
    async with app.run_test(size=(160, 45)) as pilot:
        screen = await app.open_picker()
        screen.use_previews_for_test(tmp_path / "sessions")
        await pilot.pause()
        before_index = screen.selected_index
        before = screen.render_preview_for_test()

        await pilot.press("ctrl+d")
        await pilot.pause()
        assert screen.selected_index == before_index
        after = screen.render_preview_for_test()
        assert after != before, "ctrl+d did not scroll the preview"

        await pilot.press("ctrl+u")
        await pilot.pause()
        assert screen.selected_index == before_index
        assert screen.render_preview_for_test() == before


@pytest.mark.asyncio
async def test_ctrl_g_jumps_the_preview_to_the_newest_turn(tmp_path: Path) -> None:
    session_id = "bb22cc33dd44"
    _write_transcript(
        tmp_path,
        session_id,
        [_message("user", f"turn {index} " + "body text " * 12) for index in range(40)],
    )
    app = _PickerHost([_row(session_id, "long conversation")])
    async with app.run_test(size=(160, 45)) as pilot:
        screen = await app.open_picker()
        screen.use_previews_for_test(tmp_path / "sessions")
        await pilot.pause()
        assert screen.preview_offset_for_test() == 0

        await pilot.press("ctrl+g")
        await pilot.pause()
        jumped = screen.preview_offset_for_test()
        assert jumped > 0, "ctrl+g did not move to the newest turn"
        # Clamped to the last full screen, never past the end.
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert screen.preview_offset_for_test() == jumped


@pytest.mark.asyncio
async def test_typing_still_filters_and_a_chord_never_types_into_the_filter() -> None:
    """Every new affordance is a CHORD: printable keys belong to the filter,
    and that stays true.
    """
    rows = [_row("aaa111aaa111", "asteroids game"), _row("bbb222bbb222", "parser crash")]
    app = _PickerHost(rows)
    async with app.run_test(size=(160, 45)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        for char in "parser":
            await pilot.press(char)
        await pilot.pause()
        assert screen.filter_query == "parser"
        assert [row.id for row in screen.visible_rows] == ["bbb222bbb222"]

        for chord in ("ctrl+e", "ctrl+u", "ctrl+d", "ctrl+g"):
            await pilot.press(chord)
            await pilot.pause()
            assert screen.filter_query == "parser", f"{chord} typed into the filter"


@pytest.mark.asyncio
async def test_the_filter_row_carries_the_query_the_counters_and_the_chords() -> None:
    """Replaces ``test_the_card_ends_on_quiet_ground_then_its_meta_in_both_states``.

    The card's ``lines[-1]/[-2]/[-3]`` grammar was a property of the single
    ``Static``. The filter row now carries the same FACTS in one line: the
    position is stated, the keys are stated, and they do not collide.
    """
    rows = [_row(f"{index:012x}", f"session {index}") for index in range(80)]
    app = _PickerHost(rows)
    async with app.run_test(size=(160, 45)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        footer = screen.render_footer_for_test()
        assert f"{len(rows):,}" in footer
        assert "enter" in footer and "esc" in footer
        assert "ctrl+e" in footer

        for char in "session 1":
            await pilot.press(char)
        await pilot.pause()
        assert "session 1" in screen.render_footer_for_test()


@pytest.mark.asyncio
async def test_the_panes_fit_the_terminal_at_every_height_on_the_real_stylesheet() -> None:
    """Replaces ``test_the_footer_survives_at_every_height_on_the_real_stylesheet``.

    The hazard it guarded is NOT gone — Textual still clips silently — so this
    mounts the REAL ``OperatorApp`` (``_PickerHost`` declares no ``CSS_PATH``
    and so agreed with the original bug) and asserts the filter row is drawn
    and every pane's composed line count is within its region height.
    """
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    rows = [_row(f"{index:012x}", f"session {index}") for index in range(60)]
    for height in (14, 16, 18, 20, 23, 30, 48):
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, height)) as pilot:
            screen = SessionPickerScreen(rows, NOW)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.pause()

            footer = screen.render_footer_for_test()
            assert footer.strip(), f"filter row empty at height {height}"

            results = screen.query_one("#session-picker-results")
            preview = screen.query_one("#session-picker-preview")
            assert len(screen.render_lines_for_test()) <= results.size.height, height
            assert len(screen.render_preview_for_test()) <= preview.size.height, height


@pytest.mark.asyncio
async def test_a_narrow_filter_row_keeps_the_active_query() -> None:
    """Replaces ``test_a_narrow_painted_header_keeps_the_active_filter``.

    Same rule, new home: the query is the user's only receipt that typing
    reached the modal, so it is shed LAST.
    """
    rows = [_row("aaa111aaa111", "asteroids game"), _row("bbb222bbb222", "parser crash")]
    app = _PickerHost(rows)
    async with app.run_test(size=(50, 24)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        for char in "asteroid":
            await pilot.press(char)
        await pilot.pause()
        assert "asteroid" in screen.render_footer_for_test()


@pytest.mark.asyncio
async def test_the_way_out_is_stated_at_every_width_the_picker_supports() -> None:
    """The footer is the only place the picker says how to get out.

    The module docstring has said so since the card era, and the shed ladder is
    what has to honour it: counters, legends and chord GLOSSES all go before
    the keys, and ``esc`` goes last of all. A review found the row shedding
    ``esc`` entirely at 30 and 24 columns — a silent no-exit state, on a modal
    whose only other exit is the mouse.
    """
    rows = [_row(f"{index:012d}", f"session {index}") for index in range(60)]
    for width in (24, PICKER_MIN_WIDTH, 40, 50, 74, 100):
        screen = SessionPickerScreen(rows, NOW)
        screen._layout = lambda width=width: plan_layout(width, 30)  # type: ignore[method-assign]
        footer = screen.render_footer_for_test()
        assert "esc" in footer, f"no way out stated at {width} cols: {footer!r}"
        assert cell_len(footer) <= plan_layout(width, 30).screen_width, (width, footer)


@pytest.mark.asyncio
async def test_a_filter_that_fits_one_page_reports_the_match_count() -> None:
    """D35: the footer reported the whole store's size over a filtered list.

    The position counter only appears when the list scrolls, and the branch it
    falls through to was answering "how many sessions are there" — true of an
    unfiltered picker, and plainly wrong beside eleven rows the user filtered
    down to.
    """
    rows = [_row(f"{index:012d}", f"session {index}") for index in range(63)]
    screen = SessionPickerScreen(rows, NOW)
    screen._layout = lambda: plan_layout(120, 30)  # type: ignore[method-assign]

    # Unfiltered, 63 rows scroll at this height, so the POSITION counter is
    # what shows — and its denominator is the store total, correctly.
    assert f"of {len(rows):,}" in screen.render_footer_for_test()

    screen.set_query("session 1")
    matches = len(screen.visible_rows)
    assert 0 < matches <= plan_layout(120, 30).list_rows, "fixture must fit one page"

    footer = screen.render_footer_for_test()
    # ...and the noun is MATCHES while a filter is active (design round 5, D4):
    # over a filtered list, "session(s)" reads as a statement about the store —
    # which is how the zero-match frame came to say `0 sessions` beside
    # `no session matches that filter`.
    assert f"{matches:,} match" in footer, footer
    assert f"{len(rows):,} sessions" not in footer, footer


# --- remediation round: the preview's window, its affordances, and the mouse --
#
# Four streams batched into one round (agent review 3, design 5, UX 1, QA 1).
# Every test here fails on the head they reviewed; each names the finding it
# closes so a later round can tell what is still load-bearing.


@pytest.mark.asyncio
async def test_the_preview_opens_on_the_true_first_turn_and_marks_the_unread_middle(
    tmp_path: Path,
) -> None:
    """UX U1 = QA Q2, the MAJOR this round exists for.

    The pane used to read the TAIL window and open on whatever turn it found
    there, drawing it with a clean role gutter: a 557 KB / 39-turn transcript
    showed a header naming ``burst 0`` over a first body line of ``burst 22``,
    with the opening turns unreachable (``preview_offset_for_test() == 0`` after
    400 ``ctrl+u``) and nothing on screen saying so. The pane reads BOTH ends
    now, so the first body turn is the session's real one and the turns in
    neither window are stated.
    """
    filler = "y" * 8_000
    entries = [
        _message("user", f"burst {index}: open the file {filler}", ts=float(index))
        for index in range(80)
    ]
    _write_transcript(tmp_path, "aa0000000001", entries)
    transcript = tmp_path / "sessions" / "aa0000000001" / "transcript.jsonl"
    assert transcript.stat().st_size > 2 * PREVIEW_TAIL_BYTES, "fixture is not over-window"

    app = _PickerHost([_row("aa0000000001", "burst 0: open the file")])
    async with app.run_test(size=(120, 36)) as pilot:
        screen = await app.open_picker()
        screen.use_previews_for_test(tmp_path / "sessions")
        await pilot.pause()
        pane = screen.render_preview_for_test()

        body = [line.strip() for line in pane if line.strip()]
        first_turn = next(index for index, line in enumerate(body) if line.startswith("▸"))
        # The FIRST body turn is the session's opening message, not a
        # mid-session one wearing the same gutter.
        assert body[first_turn + 1].startswith("burst 0:"), body[first_turn : first_turn + 3]

        # ...and the turns this bounded read could not reach are STATED rather
        # than silently absent or, worse, presented as a beginning. The head
        # window is thousands of wrapped lines here, so the marker is driven
        # into view rather than looked for in the frame the cursor opens on.
        lines = screen._preview_lines()
        marker = next(index for index, (kind, _) in enumerate(lines) if kind == "marker")
        assert lines[marker][1] == GAP_TEXT
        screen._pane_top = marker
        screen._repaint()
        await pilot.pause()
        scrolled = screen.render_preview_for_test()
    assert any("not read" in line for line in scrolled), scrolled


@pytest.mark.asyncio
async def test_the_clipped_pane_states_its_position_and_names_the_chords(tmp_path: Path) -> None:
    """Design D1 = UX U2: an overflowing pane with no affordance at all.

    A 202-line conversation through an 8-line window drew no ellipsis, no
    "more below", no position and no scrollbar, and the three chords that
    scroll it — ``ctrl+u``/``ctrl+d``/``ctrl+g``, bound ``show=False`` so
    Textual's footer cannot reveal them either — appeared in **0 of 36**
    rendered footers.
    """
    _write_transcript(
        tmp_path,
        "cc0000000001",
        [_message("user", f"turn {index} " + "conversation body " * 12) for index in range(40)],
    )
    app = _PickerHost([_row("cc0000000001", "a long conversation")])
    async with app.run_test(size=(120, 36)) as pilot:
        screen = await app.open_picker()
        screen.use_previews_for_test(tmp_path / "sessions")
        await pilot.pause()
        pane = screen.render_preview_for_test()
        assert len(screen._preview_lines()) > screen._pane_height(), "fixture is not clipped"
        # ...the chords are named ON SCREEN, in the pane that they move...
        assert any("ctrl+u/ctrl+d scroll" in line for line in pane), pane
        assert any("ctrl+g newest" in line for line in pane), pane
        # ...and so is where you are in it.
        assert any(re.search(r"\d+–\d+ of \d+", line) for line in pane), pane

        # The marker costs one body row and is reserved for the SESSION, not for
        # the offset: scrolling does not add or remove it.
        await pilot.press("ctrl+d")
        await pilot.pause()
        scrolled = screen.render_preview_for_test()
        assert any("ctrl+u/ctrl+d scroll" in line for line in scrolled), scrolled


@pytest.mark.asyncio
async def test_the_wheel_over_the_preview_scrolls_the_preview_not_the_list(tmp_path: Path) -> None:
    """UX U3: the wheel over the new pane moved the LIST cursor.

    At 120x36 one notch over the preview took the selection 1 → 2 (and at
    200x50 it reset the pane's offset 50 → 0), so the gesture a mouse user
    reaches for on the pane CHANGED WHICH CONVERSATION WAS BEING PREVIEWED.
    """
    _write_transcript(
        tmp_path,
        "dd0000000001",
        [_message("user", f"turn {index} " + "body text " * 12) for index in range(40)],
    )
    rows = [_row("dd0000000001", "first"), _row("ee0000000001", "second")]
    app = _PickerHost(rows)
    async with app.run_test(size=(120, 36)) as pilot:
        screen = await app.open_picker()
        screen.use_previews_for_test(tmp_path / "sessions")
        await pilot.pause()

        class _Wheel:
            def __init__(self, x: int, y: int) -> None:
                self.screen_x = x
                self.screen_y = y
                self.stopped = False

            def stop(self) -> None:
                self.stopped = True

        preview = screen._preview.region
        before_cursor = screen.selected_index
        before_offset = screen.preview_offset_for_test()
        event = _Wheel(preview.x + 2, preview.y + 2)
        screen.on_mouse_scroll_down(event)
        assert event.stopped, "the gesture must not also scroll the transcript behind"
        assert screen.selected_index == before_cursor, "the wheel moved the LIST cursor"
        assert screen.preview_offset_for_test() > before_offset, "the wheel did not scroll the pane"

        # The same gesture over the LIST still moves the cursor: one gesture,
        # one meaning, decided by where the pointer is.
        results = screen.query_one("#session-picker-results").region
        screen.on_mouse_scroll_down(_Wheel(results.x + 2, results.y + 1))
        assert screen.selected_index == before_cursor + 1

        # A click in the pane is inert — and stopped, so it cannot fall through
        # to the transcript behind the modal.
        chosen: list[str | None] = []
        screen.dismiss = chosen.append  # type: ignore[method-assign]
        click = _Wheel(preview.x + 2, preview.y + 2)
        click.button = 1  # type: ignore[attr-defined]
        screen.on_click(click)
        assert click.stopped and chosen == [], "a click in the preview must choose nothing"


@pytest.mark.asyncio
async def test_the_painted_name_field_equals_the_plan_across_the_id_gap_band() -> None:
    """QA Q1 = design D2, asserted on the plane the defect lived on: the TEXT.

    At 52-71 columns every row reserved 14 cells for an id nobody drew, so the
    painted name field went 29 (48 cols) → **17** (52) → 36 (71) → 37 (72, the
    id appears) while ``plan_layout.name_width`` rose monotonically throughout,
    and 18 cells sat blank at the right edge of every row. The existing sweep
    asserts ``plan_layout`` alone and is monotone across the band, so it could
    not see the plane the user reads.

    WHERE THIS IS AND IS NOT A FRAME ASSERTION. ``render_lines_for_test`` is the
    widget's own ``Text``, not the compositor, so this bites for the defect it
    targets — the row is built four cells too wide and the arithmetic fails — but
    it cannot see a row the compositor then wraps or drops. It is not wrong to
    say this is the plane the defect lived on (the row's own construction is
    where the id went missing) and it IS wrong to call it the paint, which an
    earlier version of this docstring did (agent review round 4, MINOR).
    """
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.session_picker import GUTTER_CELLS
    from local_operator.tui.widgets.session_picker import plan_layout as planner
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    # 64 cells, no digits and no "ago", so nothing in the name can be confused
    # with the age or the id when the painted row is measured.
    name = ("alpha tenant " * 6).strip()
    rows = [_row(f"aa{index:010d}", name) for index in range(4)]
    painted: list[tuple[int, int]] = []
    for width in (44, 48, 52, 56, 60, 64, 68, 71, 72, 80, 100):
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(width, 30)) as pilot:
            await pilot.pause()
            screen = SessionPickerScreen(rows, NOW)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.pause()
            plan = planner(width, 30)
            assert plan.mode == "stacked"
            row0 = screen.render_lines_for_test()[0]
            # The name field, measured off the PAINTED row: everything to the
            # right of it is the two-cell gap plus the age column, plus the id
            # column on the widths where the plan draws one.
            trailing = 2 + plan.age_width + (2 + len(rows[0].id) if plan.show_id else 0)
            name_cells = cell_len(row0) - trailing - GUTTER_CELLS
            painted.append((width, name_cells))
            assert name_cells == plan.name_width, (
                width,
                name_cells,
                plan.name_width,
                row0,
            )
            assert cell_len(row0) <= screen._usable(), (width, cell_len(row0), screen._usable())

    # THE INVARIANT, AT THE PAINTED LEVEL: the name field never narrows as the
    # terminal grows, with the id flip as the one pinned exception.
    for (width, cells), (next_width, next_cells) in zip(painted, painted[1:]):
        if next_cells < cells:
            assert width == 71 and next_width == 72, (width, next_width, cells, next_cells)


def test_no_word_on_the_filter_row_is_painted_at_faint() -> None:
    """Design D5: the row the legends were moved onto was painted at 1.49:1.

    ``render_rows``' own docstring cites 1.49:1 as why the ids, the ages and the
    keys were moved off ``faint`` on this raised ground; the keys moved and the
    words explaining them did not, so every frame at every size carried ink the
    module explicitly rejects. Separators stay ``faint`` — that is the step's
    stated job — and every WORD is asserted to be at least ``dim``.
    """
    rows = [_row(f"{index:012x}", f"session {index}") for index in range(40)]
    screen = SessionPickerScreen(rows, NOW)
    screen.set_query("session")
    text = screen._filter_text()
    faint_ink = theme_mod.semantic_color("faint")
    faint = (faint_ink if isinstance(faint_ink, Color) else Color.parse(faint_ink)).get_truecolor()

    def painted_faint(style: Style | str) -> bool:
        # `Text.spans` carries `Style | str`, and the string form is a style
        # NAME — no colour to compare, so it cannot be the step under test.
        if not isinstance(style, Style) or style.color is None:
            return False
        colour = style.color
        return (
            colour if isinstance(colour, Color) else Color.parse(colour)
        ).get_truecolor() == faint

    offenders = [
        text.plain[span.start : span.end]
        for span in text.spans
        if painted_faint(span.style)
        and any(character.isalnum() for character in text.plain[span.start : span.end])
    ]
    assert not offenders, offenders


def test_the_preview_mode_status_survives_where_the_row_has_room() -> None:
    """Design D3: the status was dropped by exactly one cell, 4 states in 36.

    The key row prefixes its first hint with the same three cells it joins them
    with, and ``_footer_hints`` measures only the hints — so a block that
    exactly filled ``room`` was ``room + 3`` wide and the mode hint's fit test
    compared 55 cells against 54. Measured effect: at 100x30 with ONE match the
    row read ``/ asteroids   1 session   ↑↓ move · type to filter · enter resume
    · esc cancel`` with 18 cells idle and no statement of the preview mode
    anywhere on screen.
    """
    rows = [_row(f"{index:012x}", f"session {index}") for index in range(40)]
    rows[0] = _row(f"{0:012x}", "asteroids game")
    screen = SessionPickerScreen(rows, NOW)
    screen.set_query("asteroids")
    assert len(screen.visible_rows) == 1, "fixture must be the one-match state"
    footer = screen.render_footer_for_test()
    assert "ctrl+e condensed" in footer, footer
    assert cell_len(footer) <= plan_layout(100, 30).screen_width, footer


def test_a_zero_match_filter_says_matches_and_names_the_filter_in_the_pane() -> None:
    """Design D4: one fact, three vocabularies, one of them about the store.

    The zero-match frame read ``no session matches that filter`` / ``0
    sessions`` / preview ``no session`` — the last of which is the same
    "there are no sessions" misreading ``RESUME_EMPTY_NOTICE`` exists to avoid,
    and the middle of which is a statement about the store beside a filtered
    list.
    """
    rows = [_row(f"{index:012x}", f"session {index}") for index in range(5)]
    screen = SessionPickerScreen(rows, NOW)
    screen.set_query("qqqzzz")
    assert screen.visible_rows == []
    footer = screen.render_footer_for_test()
    assert "0 matches" in footer, footer
    assert "0 sessions" not in footer, footer
    pane = "\n".join(screen.render_preview_for_test())
    assert "no session" not in pane, pane
    assert "qqqzzz" in pane, pane


def test_a_soft_only_match_is_glossed_as_fuzzy() -> None:
    """UX U4: the new ``~`` glyph shipped with no legend at all.

    A fuzzy query drew a tilde on both matched rows while the footer glossed only
    ``” matched inside``, so the one mark meaning "there is no literal substring
    to show you" was the one mark nothing explained.
    """
    rows = [_row("aaa111aaa111", "the classifier work"), _row("bbb222bbb222", "unrelated")]
    digests = {"aaa111aaa111": "a digest about classifier tuning and the classifier"}
    app = _PickerHost(rows, digests)
    screen = SessionPickerScreen(rows, NOW, digests)
    screen.set_query("classifer")  # one transposition: a soft hit, not a substring
    assert [row.id for row in screen.visible_rows] == ["aaa111aaa111"], "fixture must be soft-only"
    assert screen.body_matched_ids - screen._body_matches == {"aaa111aaa111"}
    footer = screen.render_footer_for_test()
    assert "~ fuzzy" in footer, footer
    assert app is not None


def test_an_unreadable_transcript_says_so_instead_of_no_prose(tmp_path: Path) -> None:
    """Design D6 = UX U6: a permission error reported as an empty conversation.

    A transcript at mode 000 rendered ``(no prose in this transcript)`` — a claim
    about prose nobody could read — under a header showing the bare hex id. A
    permission problem, a file deleted at that moment and a genuinely empty
    conversation are three different states.
    """
    directory = tmp_path / "sessions" / "aa0000000009"
    directory.mkdir(parents=True)
    transcript = directory / "transcript.jsonl"
    transcript.write_text(json.dumps(_message("user", "readable once")) + "\n", encoding="utf-8")
    transcript.chmod(0o000)
    try:
        screen = SessionPickerScreen([_row("aa0000000009", "unreadable")], NOW)
        screen.use_previews_for_test(tmp_path / "sessions")
        pane = "\n".join(screen.render_preview_for_test())
    finally:
        transcript.chmod(0o600)
    assert "could not be read" in pane, pane
    assert "no prose" not in pane, pane


def test_a_name_match_draws_no_context_line_under_it() -> None:
    """Design D7: the pane printed the row's own name twice.

    A row's name IS its opening user message, so a query in the name is in the
    body digest too: the frame drew the name as the row and again as a quote
    beneath it, with no ``”`` marker and no legend — correctly, since the row
    did not match *inside* the conversation. The quote cost a line and taught
    nothing.
    """
    rows = [_row("bb0000000002", "Make an asteroids game in pygame")]
    digests = {"bb0000000002": "Make an asteroids game in pygame draw the ship, then the rocks"}
    screen = SessionPickerScreen(rows, NOW, digests)
    screen.set_query("asteroids")
    assert screen.visible_rows, "the name match must still be offered"
    assert screen._context_for(rows[0]) is None
    assert screen._row_costs() == [1], "a context line was drawn for a name-only match"


def test_a_short_terminal_drops_the_preview_and_gives_the_rows_to_the_list() -> None:
    """UX U5: four rows spent on a header and zero on the conversation.

    At 30x12 the pane drew its name, both clocks and its rule, then nothing —
    while ``plan_layout``'s own short-height branch said it existed precisely so
    the preview would not collapse that way. Below the height at which it can
    draw a header and one body line, the preview is not drawn at all and the
    list takes the rows.
    """
    from local_operator.tui.widgets.session_picker import PREVIEW_DRAW_MIN

    assert PREVIEW_DRAW_MIN == 5
    short = plan_layout(30, 12)
    assert short.preview_rows == 0
    # The rows the preview would have taken are the LIST's: chrome is still
    # reserved first, so the list gets everything above the filter row.
    assert short.list_rows == 12 - 3
    assert plan_layout(40, 14).preview_rows == PREVIEW_DRAW_MIN


@pytest.mark.asyncio
async def test_the_short_terminal_hides_the_pane_in_the_real_app() -> None:
    """The plan saying zero is only half of it: the widget must not be drawn."""
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    rows = [_row(f"{index:012x}", f"session {index}") for index in range(20)]
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(30, 12)) as pilot:
        await pilot.pause()
        screen = SessionPickerScreen(rows, NOW)
        app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        assert screen._preview.display is False
        # ...and the list, not the pane, holds the rows that freed.
        assert len(screen.render_lines_for_test()) > plan_layout(30, 12).preview_rows


@pytest.mark.asyncio
async def test_the_context_line_carries_the_query_in_both_layouts() -> None:
    """Agent review R-MINOR-1: the drawn context line's CONTENT was unasserted.

    The one test that touched it asserted its COST (two lines), so truncating the
    line to 12 cells rendered ``'    …xxxxxxxxxx…'`` with no query in it and 136
    tests stayed green — the PR's headline claim ("the filter's matches quoted
    in place") guarded by nothing.
    """
    body = "the incident began quietly and then " + "retention window " * 4
    rows = [_row("cc0000000001", "an unrelated title")]
    digests = {"cc0000000001": body}
    for size in ((100, 30), (200, 50)):
        app = _PickerHost(rows, digests)
        async with app.run_test(size=size) as pilot:
            screen = await app.open_picker()
            await pilot.pause()
            screen.set_query("retention")
            await pilot.pause()
            drawn = screen.render_lines_for_test()
            context = [line for line in drawn[1:] if "retention" in line]
            assert context, (size, drawn)
            assert any("retention" in line for line in context), (size, context)
            # ...AND IT STILL FITS THE PANE. The context line is built from the
            # indent plus ``_context_width()``; dropping the indent from that
            # subtraction — the D17 contract broken the other way — leaves the
            # drawn line four cells wider than the pane, Textual wraps it and
            # every row below shifts, while the content assertions above stay
            # green. A line that cannot fit has to fail, so the width is asserted
            # on the same plane the content is (agent review round 4, MINOR).
            for line in drawn:
                assert cell_len(line) <= screen._usable(), (size, cell_len(line), line)


def _real_app() -> App[None]:
    """The REAL ``OperatorApp`` — ``_PickerHost`` declares no ``CSS_PATH``.

    Imported here rather than at module scope for the reason the other tests in
    this file import it in their bodies: the app module pulls the whole TUI in,
    and collection pays for it once per session instead of once per file.
    """
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    return OperatorApp(lambda: _factory(FakeSession()))


def _pane_rows(app: App[None], screen) -> list[tuple[int, str]]:
    """``(strip index, text)`` for every painted row of the preview pane's region.

    THE COMPOSITOR, blanks included — ``painted_rows`` drops blank rows, and the
    row budget this round is about is exactly a question of blank rows: the pane
    painting seven of its eight content rows and leaving the last bare is a
    defect made of a blank.
    """
    preview = screen.query_one("#session-picker-preview")
    strips = list(app.screen._compositor.render_strips())
    region = preview.region
    rows = []
    for y in range(max(0, region.y), min(len(strips), region.bottom)):
        rows.append((y, strips[y].crop(max(0, region.x), max(0, region.right)).text.rstrip()))
    return rows


def _status_row(app: App[None], screen) -> tuple[int, str] | None:
    """The pane's status row — its strip index and its text — or ``None``.

    Found by the ARITHMETIC it carries rather than by position, so a pane whose
    status row is missing, blanked or painted on the wrong row all read as
    ``None``/wrong row here.
    """
    for y, text in _pane_rows(app, screen):
        if re.search(r"\d+–\d+ of \d+", text) or text.strip() == "⋮":
            return y, text
    return None


def _chrome(text: str, name: str) -> bool:
    """Is this painted pane row chrome rather than conversation?

    Every chrome row has a shape no body line has: the rule rows are all
    ``─``, the name row carries the row's name, the clock row opens
    ``started``, and the status row carries the arithmetic or the bare ``⋮``.
    Classifying by chrome rather than by the body's own text is what lets the
    same predicate read the frame at the TOP (whose first body line is the
    session's opening message) and at ``ctrl+g`` (whose last body line is the
    tail of the newest reply, and says nothing identifiable).
    """
    stripped = text.strip()
    if not stripped or stripped == "⋮" or set(stripped) == {"─"}:
        return True
    if name in stripped or stripped.startswith("started "):
        return True
    return bool(re.search(r"\d+–\d+ of \d+", stripped)) or "not read" in stripped


async def _open_with_transcript(screen, pilot, sessions: Path) -> None:
    screen.use_previews_for_test(sessions)
    await pilot.pause()


def _long_transcript(root: Path, session_id: str, turns: int = 40) -> None:
    _write_transcript(
        root,
        session_id,
        [_message("user", f"turn {index} " + "conversation body " * 12) for index in range(turns)],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 14), (100, 16), (40, 15), (40, 16), (45, 17), (60, 20)])
async def test_the_pane_never_paints_without_a_line_of_conversation(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    """Agent review round 4 MAJOR = UX round 2 U5, at the heights that defeat them.

    ``max(1, height - 1)`` emitted a body line AND a status line into a one-row
    budget. Textual clips the LAST line, so the loser was the affordance the
    round before had just added: at 100x14 the opening frame painted chrome plus
    ``1–0 of 299`` — an inverted range — with no conversation at all, and one
    scroll further the body line painted and the status row vanished. At 40x15
    and 40x16 the pane is drawn by the plan too (``PREVIEW_DRAW_MIN`` rows) and
    painted the same nothing, because a clipped row's body still had to absorb
    the reserved status row and could lose its only line to the D30 gutter guard.

    These are the sizes the plan draws the pane at and no existing test covered:
    the delta's two pane guards sit at 120x36 (drawn, mid-size) and 30x12 (pane
    hidden), so nothing exercised the band where the pane is drawn AND its
    status row cannot fit.
    """
    _long_transcript(tmp_path, "cc0000000001")
    app = _real_app()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = SessionPickerScreen([_row("cc0000000001", "a long conversation")], NOW)
        app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        await _open_with_transcript(screen, pilot, tmp_path / "sessions")
        preview = screen.query_one("#session-picker-preview")
        assert preview.display, f"the plan draws the pane at {size}"

        for label in ("top=0", "ctrl+g"):
            if label == "ctrl+g":
                screen.action_pane_end()
                await pilot.pause()
            rows = _pane_rows(app, screen)
            painted = [text for _, text in rows if text.strip()]
            # THE WHOLE BAND, in one assertion: a pane that is on screen says
            # something about the conversation it is a window onto.
            conversation = [text for text in painted if not _chrome(text, "a long conversation")]
            assert conversation, (label, painted)
            # ...and no row states a position whose end precedes its start.
            for text in painted:
                match = re.search(r"(\d+)–(\d+) of (\d+)", text)
                if match:
                    assert int(match.group(2)) >= int(match.group(1)), (label, text)

        # The pane's own text never exceeds the rows it is given — the silent
        # clip is what put the affordance on the floor in the first place.
        assert len(screen.render_preview_for_test()) <= preview.size.height


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 14), (40, 14), (30, 13)])
async def test_the_status_row_is_spent_only_where_there_is_one_to_spare(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    """The reservation's rule, from both sides, on the painted frame.

    At one body row the pane has room for a line of conversation OR for the
    status row, never both, and the CONVERSATION wins: the pane is a window onto
    a session, and a position marker over a body nobody can read is how the
    MAJOR above presented itself. One row higher there is room for both, and
    both are painted. Below the plan's floor there is room for neither and the
    pane is not drawn at all (UX U5) — the third case here, so the band is
    closed from both ends rather than only from the middle.
    """
    _long_transcript(tmp_path, "cc0000000001")
    app = _real_app()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = SessionPickerScreen([_row("cc0000000001", "a long conversation")], NOW)
        app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        await _open_with_transcript(screen, pilot, tmp_path / "sessions")
        preview = screen.query_one("#session-picker-preview")
        if not preview.display:
            # THE FLOOR: no pane at all — and its clock row, which only the pane
            # draws, is nowhere on the frame either. A bare header painted where
            # the pane would have been IS the state U5 is about, so the assertion
            # is on the whole frame rather than on the hidden widget's region.
            assert screen._layout().preview_rows == 0, size
            frame = [strip.text for strip in app.screen._compositor.render_strips()]
            assert not any(text.strip().startswith("started ") for text in frame), (size, frame)
            return

        painted = [text for _, text in _pane_rows(app, screen) if text.strip()]
        has_conversation = any(not _chrome(text, "a long conversation") for text in painted)
        status = _status_row(app, screen)

        assert has_conversation, (size, painted)
        if screen._pane_height() == 1:
            # one row: the conversation line, and no marker claiming a range
            assert status is None, (size, status)
        else:
            assert status is not None, (size, painted)
            # ...and the body under it is not one row short of what the pane
            # reserved: the marker is the pane's LAST content row, which is the
            # row the D9 shortfall left bare.
            assert status[0] == preview.content_region.bottom - 1, (size, status, painted)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 16), (100, 20), (100, 30), (120, 36)])
async def test_the_stacked_pane_spends_every_content_row_it_reserves(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    """Agent review round 4 MINOR: the stacked ``-1`` over-subtracted a border row.

    ``content_region.height`` already excludes the pane's ``border-top`` — its
    docstring says "inside the widget's own border and padding" — and the pane
    subtracted it a second time, so every stacked preview painted ONE
    CONVERSATION LINE FEWER than it had room for (measured: 100x20 painted 7 of
    8 rows with a blank tail, 100x30 8 of 9, 100x16 5 of 7 with the inverted
    ``1–0``). At 100x16 the row it cost is what tipped the pane into the state
    where the status row had nowhere to go at all.

    Pinned as ROWS, from the frame: the status row is the pane's last content
    row, so no reserved row is left bare below the text the pane draws.
    """
    _long_transcript(tmp_path, "cc0000000001")
    app = _real_app()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = SessionPickerScreen([_row("cc0000000001", "a long conversation")], NOW)
        app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        await _open_with_transcript(screen, pilot, tmp_path / "sessions")
        preview = screen.query_one("#session-picker-preview")

        assert len(screen._preview_lines()) > screen._pane_height(), "fixture is not clipped"
        # The pane's budget is its CONTENT rows, and the widget's own text fills
        # them: header + window + status, with nothing left over.
        assert len(screen.render_preview_for_test()) == preview.content_region.height, size
        status = _status_row(app, screen)
        assert status is not None, size
        assert status[0] == preview.content_region.bottom - 1, (
            size,
            status,
            _pane_rows(app, screen),
        )


@pytest.mark.asyncio
async def test_ctrl_g_lands_on_the_newest_line_not_one_short_of_it(tmp_path: Path) -> None:
    """Design round 5 D9, the scroll half: the chord named "newest" stopped short.

    At 100x30 ``ctrl+g`` reported ``offset 198`` and painted ``199–201 of 202``
    while the newest line was the 202nd, because the clamp used the pane's
    RESERVED height while the paint cut the window from a budget one row smaller
    — ``ctrl+d`` once more reached ``200–202 of 202``, so it was a landing error
    rather than an unreachable state, and only the counter revealed it.
    """
    _write_transcript(
        tmp_path,
        "cc0000000001",
        [_message("user", f"turn {index} " + "conversation body " * 12) for index in range(40)],
    )
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = SessionPickerScreen([_row("cc0000000001", "a long conversation")], NOW)
        app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        await _open_with_transcript(screen, pilot, tmp_path / "sessions")
        total = len(screen._preview_lines())
        assert total > screen._pane_height(), "fixture is not clipped"

        screen.action_pane_end()
        await pilot.pause()
        status = _status_row(app, screen)
        assert status is not None, _pane_rows(app, screen)
        match = re.search(r"(\d+)–(\d+) of (\d+)", status[1])
        assert match, status
        assert int(match.group(3)) == total
        # THE NEWEST LINE IS SHOWN, which is the whole claim the chord makes.
        assert int(match.group(2)) == total, (status[1], total)


@pytest.mark.asyncio
async def test_the_bounded_read_says_so_at_the_top_of_the_pane(tmp_path: Path) -> None:
    """UX round 2 U12: the "middle was not read" statement was thousands of lines down.

    On the 509 KB repro the marker is the 3540th wrapped line of 7065 — 704
    ``ctrl+u`` presses from ``ctrl+g``, counted by driving the keys — so a reader
    at the very top of the pane had no hint that the transcript being drawn is a
    bounded read at all. It is painted as the pane's first body row now, in the
    row the pane already spends on chrome whenever it clips, and only where
    there is a row to spare (the line of conversation wins at one row).
    """
    filler = "y" * 8_000
    _write_transcript(
        tmp_path,
        "aa0000000001",
        [
            _message("user", f"burst {index}: open the file {filler}", ts=float(index))
            for index in range(80)
        ],
    )
    transcript = tmp_path / "sessions" / "aa0000000001" / "transcript.jsonl"
    assert transcript.stat().st_size > 2 * PREVIEW_TAIL_BYTES, "fixture is not over-window"

    app = _PickerHost([_row("aa0000000001", "burst 0: open the file")])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        screen.use_previews_for_test(tmp_path / "sessions")
        await pilot.pause()
        pane = screen.render_preview_for_test()
        body = [line.strip() for line in pane if line.strip()]
        header_end = next(index for index, line in enumerate(body) if line.startswith("─"))
        assert GAP_TEXT in body[header_end + 1], body[header_end : header_end + 3]
        # ...and the conversation is still drawn under it, so the statement
        # tells the reader what it is looking at rather than replacing it.
        assert any(line.startswith("burst 0:") for line in body[header_end + 2 :]), body

    # A file inside one read window is not a bounded read and says nothing.
    _write_transcript(tmp_path, "bb0000000002", [_message("user", "a short one")])
    app = _PickerHost([_row("bb0000000002", "a short one")])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        screen.use_previews_for_test(tmp_path / "sessions")
        await pilot.pause()
        assert not any("not read" in line for line in screen.render_preview_for_test())


def _checkpoint_entry(model: str, cwd: str, ts: float = 0.0) -> dict[str, object]:
    """A frontend-state checkpoint as the picker's own reader expects it.

    The discriminator is ``payload["custom_type"]``, NOT ``payload["kind"]`` —
    that is ``None`` on these entries, which is why the reader has to look at
    ``type == "custom"`` (see ``preview.CHECKPOINT_CUSTOM_TYPE``).
    """
    return {
        "id": "c1",
        "ts": ts,
        "type": "custom",
        "payload": {
            "custom_type": "frontend_state_checkpoint_v1",
            "details": {
                "state": {
                    "effective_model": {"model_id": f"anthropic/{model}"},
                    "cwd": cwd,
                }
            },
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 14), (100, 15), (40, 15), (100, 16)])
async def test_the_optional_header_row_is_shed_before_the_conversation_line(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    """UX round 2 U5's last row: the pane on screen, costing its rows, saying nothing.

    ``PREVIEW_DRAW_MIN`` is the SHORTEST header plus one body line, and the plan
    draws the pane from it whatever the selected row holds. A row carrying a
    checkpoint makes the header one row longer (D7's ``model · cwd`` line), and
    reserving that row unconditionally left the pane painting its name, both
    clocks, the model line and its rule and NOT ONE LINE of the conversation —
    measured against the finished frame at 100x14 and 40x14, exactly the band
    the round-4 MAJOR lives in. The row is the header's optional one, so it is
    the one that goes: the name and the two clocks are what identify the row.
    """
    _write_transcript(
        tmp_path,
        "dd0000000001",
        [
            _checkpoint_entry("claude-opus-5", "/Users/example/workspace"),
            *[
                _message("user", f"turn {index} " + "conversation body " * 12)
                for index in range(40)
            ],
        ],
    )
    app = _real_app()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = SessionPickerScreen([_row("dd0000000001", "a checkpointed conversation")], NOW)
        app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        await _open_with_transcript(screen, pilot, tmp_path / "sessions")
        preview = screen.query_one("#session-picker-preview")
        assert preview.display, size
        assert screen._selected_checkpoint(), "the fixture's checkpoint was not read"

        painted = [text for _, text in _pane_rows(app, screen) if text.strip()]
        assert any("turn 0" in text for text in painted), (size, painted)
        # ...and the model line is drawn only where there is a row under it.
        shed = screen._content_rows() - 4 < 1
        assert any("claude-opus-5" in text for text in painted) is not shed, (size, painted)
        assert len(screen.render_preview_for_test()) <= preview.size.height, size


def test_the_position_marker_never_states_a_range_that_ends_before_it_starts() -> None:
    """Agent review round 4 MAJOR, the arithmetic half: ``1–0 of 299`` on screen.

    A window of zero drawn lines printed an end BEFORE its own start. The pane
    now guarantees itself a drawn line, so this pins the ARITHMETIC rather than
    a reachable frame: the state is what the round's frame tests cover, and this
    is what stops a future caller — a different pane shape, a different budget —
    from painting the inverted range again. A guard on a resolved contract,
    which is the only thing a helper this small can honestly hold.
    """
    screen = SessionPickerScreen([_row("aaa111aaa111", "one")], NOW)
    for drawn in range(4):
        text = screen._preview_pane_status(40, 0, drawn, 299).plain
        match = re.search(r"(\d+)–(\d+) of (\d+)", text)
        assert match, (drawn, text)
        assert int(match.group(2)) >= int(match.group(1)), (drawn, text)
    # One drawn line reads as the one line it is at, not as nothing.
    assert screen._preview_pane_status(40, 0, 0, 299).plain.startswith("1–1 of 299")

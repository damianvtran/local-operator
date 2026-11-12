"""The second expand level of a write/edit card: the WHOLE diff, in the TUI.

Reported from a subagent page: a reviewer opening a child's write could see
40 preview rows and ``… N more diff lines`` and nothing past them. The inline
record holds 200 rows; the rest lives only in the redacted full-diff archive
(``tools/diff_archive``, #3762), which no terminal reader opened. These tests
drive the archive a REAL loop wrote (redaction seam included) through the real
app, in the parent transcript and on a subagent page, and pin the fallbacks:
a refused read never paints anything the card was not already given.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from rich.color import ColorTriplet

from local_operator.tools import diff_archive
from local_operator.tools.builtin import DIFF_TRUNCATED_ROWS_KEY
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.subagent_view import SubagentView
from local_operator.tui.widgets.tool_card import (
    COLLAPSE_HINT,
    EXPAND_HINT,
    EXPAND_MAX_LINES,
    ToolCard,
)
from tests.unit.harness.test_full_diff_archive import SESSION_ID, _big, _details, _run

from .test_band_panels import FakeSession, _async_factory, _fake_jobs
from .test_subagent_view import _job_with, _open, _wait_history


@pytest.fixture(autouse=True)
def _config(tmp_path, monkeypatch):
    # The config root IS ``tmp_path``: the loop's transcript lands at
    # ``tmp_path/sessions/<id>`` (``_run``'s layout), which is exactly where the
    # app resolves the parent conversation, and the archive under it.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))


async def _archived_write(tmp_path: Path, rows: int = 500) -> dict[str, Any]:
    """A real ``write`` over the inline cap; its persisted (redacted) details."""
    target = tmp_path / "big.py"
    session, _events = await _run(
        tmp_path, [("w-big", "write", {"path": str(target), "content": _big(rows)})]
    )
    details = _details(session, "w-big")
    assert details[DIFF_TRUNCATED_ROWS_KEY] > 0
    assert details[diff_archive.DIFF_FULL_KEY]["state"] == "archived"
    return details


def _diff_rows(card: ToolCard, width: int = 100) -> list[str]:
    plain = card._build_content(width).plain.split("\n")
    return [line.strip() for line in plain[1:]]


def _expect(semantic: str) -> ColorTriplet:
    hexv = theme_mod.semantic_color(semantic)
    return ColorTriplet(*[int(hexv[i : i + 2], 16) for i in (1, 3, 5)])


def _ink_of(card: ToolCard, needle: str) -> ColorTriplet | None:
    content = card._build_content(100)
    at = content.plain.index(needle)
    for span in content.spans:
        if span.start <= at < span.end and not isinstance(span.style, str):
            color = span.style.color
            if color is not None:
                return color.get_truecolor()
    return None


async def _settle_full(pilot: Any, card: ToolCard) -> None:
    for _ in range(200):
        await pilot.pause()
        if not card._full_diff_loading and card._full_diff_shown:
            return
    raise AssertionError("the full-diff read never settled")


# -- the card on its own ------------------------------------------------------


def test_complete_inline_diff_past_the_preview_opens_whole_without_io() -> None:
    """A 120-row diff the inline record holds whole: level 2 is a pure repaint."""
    diff = ["--- ", "+++ ", "@@ -0,0 +1,120 @@", *(f"+line {n}" for n in range(120))]
    card = ToolCard("t", "write", {"path": "a.py"})
    card.mark_done("Wrote a.py", {"path": "a.py", "added": 120, "removed": 0, "diff": diff})
    card.diff_call_id = ""  # no archive identity at all: nothing can be read
    assert card.toggle_expanded() is True
    rows = _diff_rows(card)
    assert len(rows) == EXPAND_MAX_LINES + 1
    # The marker offers the next level with the header's own hint (U2).
    assert rows[-1] == f"… {121 - EXPAND_MAX_LINES} more diff lines {EXPAND_HINT}"
    assert card._full_diff_offered()
    # Second press: every row, no marker, still expanded.
    assert card.toggle_expanded() is True
    rows = _diff_rows(card)
    assert rows[0] == "@@ -0,0 +1,120 @@" and rows[-1] == "+line 119"
    assert len(rows) == 121 and not any(r.startswith("…") for r in rows)
    assert not card._full_diff_offered()
    # Third press collapses; reopening starts at the preview again.
    assert card.toggle_expanded() is False
    assert card.toggle_expanded() is True
    assert len(_diff_rows(card)) == EXPAND_MAX_LINES + 1


def test_short_diff_keeps_the_two_state_toggle() -> None:
    diff = ["--- ", "+++ ", "@@ -1 +1 @@", "-a", "+b"]
    card = ToolCard("t", "edit", {"path": "a.py"})
    card.mark_done("Edited a.py", {"path": "a.py", "added": 1, "removed": 1, "diff": diff})
    assert card.toggle_expanded() is True
    assert not card._full_diff_offered()
    assert card.toggle_expanded() is False


def test_load_all_rows_rejoins_split_rows_and_refuses_a_foreign_session(tmp_path) -> None:
    body = "x" * 300_000  # larger than one 120 KiB page: served as fragments
    rows = ["--- ", "+++ ", "@@ -1 +1,2 @@", " ctx", "+" + body, "-" + body[::-1]]
    desc = diff_archive.write_archive(rows, session_id=SESSION_ID, call_id="j", tool_name="write")
    session_dir = tmp_path / "sessions" / SESSION_ID
    session_dir.mkdir(parents=True)
    payload = {
        "role": "tool",
        "tool_call_id": "j",
        "tool_name": "write",
        "provider_payload": {"details": {"diff_full": desc, DIFF_TRUNCATED_ROWS_KEY: 1}},
    }
    (session_dir / "transcript.jsonl").write_text(
        json.dumps({"id": "m-j", "ts": 0, "type": "message", "payload": payload}) + "\n"
    )
    assert diff_archive.load_all_rows(session_dir, SESSION_ID, "j") == rows
    # The artifact names its session: the same row under another id is refused.
    assert diff_archive.load_all_rows(session_dir, "fd0000000009", "j") is None
    assert diff_archive.load_all_rows(session_dir, SESSION_ID, "nope") is None


# -- the parent transcript ------------------------------------------------------


class _ParentSession(FakeSession):
    @property
    def session_id(self) -> str:
        return SESSION_ID


@pytest.mark.asyncio
async def test_parent_card_pages_the_complete_archived_diff_in_git_colours(tmp_path) -> None:
    details = await _archived_write(tmp_path)
    app = OperatorApp(_async_factory(_ParentSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        workers = [w for w in app.workers if w.group == "session"]
        if workers:
            await app.workers.wait_for_complete(workers)
        card = ToolCard("w-big", "write", {"path": "big.py"})
        card.mark_done("Wrote big.py", details)
        app._append_block(card)
        await pilot.pause()
        card.focus()
        await pilot.press("enter")
        await pilot.pause()
        preview = _diff_rows(card)
        assert len(preview) == EXPAND_MAX_LINES + 1
        # The marker counts the rows the inline record omitted, not only the
        # ones the preview skipped: 500 added lines + one hunk header.
        assert preview[-1] == f"… {501 - EXPAND_MAX_LINES} more diff lines {EXPAND_HINT}"
        card._hovered = True
        assert EXPAND_HINT in card._build_content(100).plain.split("\n")[0]
        await pilot.press("enter")
        await _settle_full(pilot, card)
        rows = _diff_rows(card)
        assert rows[0] == "@@ -0,0 +1,500 @@"
        assert rows[1] == "+FIRST_SENTINEL = 1" and rows[-1] == "+FINAL_SENTINEL = 3"
        assert len(rows) == 501 and "+MIDDLE_SENTINEL = 2" in rows
        assert not any(row.startswith("…") for row in rows)
        assert COLLAPSE_HINT in card._build_content(100).plain.split("\n")[0]
        assert _ink_of(card, "+FINAL_SENTINEL") == _expect("success")
        # The card really grew on screen to hold every row.
        await pilot.pause()
        assert card.size.height == 1 + 501
        await pilot.press("enter")
        await pilot.pause()
        assert not card.expanded and card.size.height == 1


@pytest.mark.asyncio
async def test_removed_rows_past_the_preview_paint_in_danger(tmp_path) -> None:
    target = tmp_path / "big.py"
    before = "".join(f"old_{n} = {n}\n" for n in range(300))
    target.write_text(before)
    session, _events = await _run(
        tmp_path, [("w-del", "write", {"path": str(target), "content": "only = 1\n"})]
    )
    details = _details(session, "w-del")
    app = OperatorApp(_async_factory(_ParentSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        workers = [w for w in app.workers if w.group == "session"]
        if workers:
            await app.workers.wait_for_complete(workers)
        card = ToolCard("w-del", "write", {"path": "big.py"})
        card.mark_done("Wrote big.py", details)
        app._append_block(card)
        await pilot.pause()
        card.activate()
        card.activate()
        await _settle_full(pilot, card)
        rows = _diff_rows(card)
        assert "-old_299 = 299" in rows and "+only = 1" in rows
        assert _ink_of(card, "-old_299") == _expect("danger")
        assert _ink_of(card, "+only = 1") == _expect("success")


@pytest.mark.asyncio
async def test_evicted_archive_falls_back_to_the_whole_inline_record(tmp_path) -> None:
    details = await _archived_write(tmp_path)
    for path in (tmp_path / diff_archive.DIFF_ARCHIVE_DIRNAME).iterdir():
        path.unlink()
    app = OperatorApp(_async_factory(_ParentSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        workers = [w for w in app.workers if w.group == "session"]
        if workers:
            await app.workers.wait_for_complete(workers)
        card = ToolCard("w-big", "write", {"path": "big.py"})
        card.mark_done("Wrote big.py", details)
        app._append_block(card)
        await pilot.pause()
        card.activate()
        card.activate()
        await _settle_full(pilot, card)
        rows = _diff_rows(card)
        inline = details["diff"][2:]
        assert rows[:-1] == inline, "every inline row, and only those"
        assert rows[-1] == f"… {details[DIFF_TRUNCATED_ROWS_KEY]} more diff lines"
        assert "+FINAL_SENTINEL = 3" not in rows


# -- the subagent page ----------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_page_card_pages_the_childs_archived_diff(tmp_path) -> None:
    """The reported surface: a finished child's write, read on its page."""
    await _archived_write(tmp_path)
    child_dir = tmp_path / "sessions" / SESSION_ID
    job = _job_with([], status="completed")
    session = FakeSession()  # the PARENT is another conversation ("sess")
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: child_dir}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        assert isinstance(view, SubagentView)
        cards = [c for c in view.query(ToolCard) if c.tool_name == "write"]
        assert len(cards) == 1
        card = cards[0]
        # Bound to the CHILD's call and directory, not the parent's session.
        assert card.diff_call_id == "w-big"
        assert card._diff_archive_directory() == child_dir
        card.activate()
        await pilot.pause()
        assert _diff_rows(card)[-1] == f"… {501 - EXPAND_MAX_LINES} more diff lines {EXPAND_HINT}"
        card.activate()
        await _settle_full(pilot, card)
        rows = _diff_rows(card)
        assert len(rows) == 501 and rows[-1] == "+FINAL_SENTINEL = 3"
        assert _ink_of(card, "+MIDDLE_SENTINEL") == _expect("success")


# -- review round 1 remediation ---------------------------------------------------


async def _mounted_card(app: OperatorApp, pilot: Any, details: dict[str, Any]) -> ToolCard:
    workers = [w for w in app.workers if w.group == "session"]
    if workers:
        await app.workers.wait_for_complete(workers)
    card = ToolCard("w-big", "write", {"path": "big.py"})
    card.mark_done("Wrote big.py", details)
    app._append_block(card)
    await pilot.pause()
    return card


def _marker_line(card: ToolCard) -> int:
    lines = card._build_content(card.size.width or 100).plain.split("\n")
    return next(i for i, line in enumerate(lines) if line.strip().startswith("… "))


def test_load_all_rows_proves_once_and_enforces_the_row_bound(tmp_path, monkeypatch) -> None:
    """One journal walk per read, not one per 200-row page; the docstring's bound holds."""
    rows = ["--- ", "+++ ", "@@ -0,0 +1,700 @@", *(f"+r{n}" for n in range(700))]
    desc = diff_archive.write_archive(rows, session_id=SESSION_ID, call_id="b", tool_name="write")
    session_dir = tmp_path / "sessions" / SESSION_ID
    session_dir.mkdir(parents=True)
    payload = {
        "role": "tool",
        "tool_call_id": "b",
        "tool_name": "write",
        "provider_payload": {"details": {"diff_full": desc, DIFF_TRUNCATED_ROWS_KEY: 500}},
    }
    (session_dir / "transcript.jsonl").write_text(
        json.dumps({"id": "m-b", "ts": 0, "type": "message", "payload": payload}) + "\n"
    )
    walks: list[str] = []
    real = diff_archive.find_tool_details

    def counting(*args: Any, **kwargs: Any) -> Any:
        walks.append(args[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(diff_archive, "find_tool_details", counting)
    assert diff_archive.load_all_rows(session_dir, SESSION_ID, "b") == rows
    assert walks == ["b"], "the membership proof runs once per read"
    # An artifact claiming more rows than the writer can produce is refused.
    monkeypatch.setattr(diff_archive, "ARCHIVE_MAX_ROWS", len(rows) - 1)
    assert diff_archive.load_all_rows(session_dir, SESSION_ID, "b") is None


@pytest.mark.asyncio
async def test_opening_the_whole_diff_keeps_the_marker_row_in_place(tmp_path) -> None:
    """U1: Enter at the end of a preview grows the card where the reader looks."""
    details = await _archived_write(tmp_path)
    app = OperatorApp(_async_factory(_ParentSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        card = await _mounted_card(app, pilot, details)
        view = card.parent
        card.focus()
        await pilot.press("enter")
        await pilot.pause()
        view.scroll_end(animate=False, immediate=True)  # type: ignore[union-attr]
        await pilot.pause()
        marker = _marker_line(card)
        before = card.virtual_region.y + marker - round(view.scroll_y)  # type: ignore[union-attr]
        assert 0 <= before < view.scrollable_content_region.height  # type: ignore[union-attr]
        await pilot.press("enter")
        await _settle_full(pilot, card)
        for _ in range(3):
            await pilot.pause()
        after = card.virtual_region.y + marker - round(view.scroll_y)  # type: ignore[union-attr]
        assert after == before, "the first new row lands on the marker's screen row"
        line = card._build_content(card.size.width).plain.split("\n")[marker].strip()
        assert line == details["diff"][2:][EXPAND_MAX_LINES].strip()


@pytest.mark.asyncio
async def test_marker_row_offers_the_next_level_and_a_click_on_it_opens_it(tmp_path) -> None:
    """U2: the existing hint sits on the marker; clicking that row activates the card."""
    details = await _archived_write(tmp_path)
    app = OperatorApp(_async_factory(_ParentSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        card = await _mounted_card(app, pilot, details)
        card.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert _diff_rows(card)[-1].endswith(EXPAND_HINT)
        card.parent.scroll_end(animate=False, immediate=True)  # type: ignore[union-attr]
        await pilot.pause()
        await pilot.click(card, offset=(6, _marker_line(card)))
        await _settle_full(pilot, card)
        rows = _diff_rows(card)
        assert len(rows) == 501 and not any(EXPAND_HINT in row for row in rows)


@pytest.mark.asyncio
async def test_a_press_while_the_read_is_in_flight_collapses(tmp_path, monkeypatch) -> None:
    """The second press used to be swallowed; the hint now says what it does."""
    import threading

    details = await _archived_write(tmp_path)
    gate = threading.Event()
    real = diff_archive.load_all_rows

    def slow(*args: Any, **kwargs: Any) -> Any:
        gate.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(diff_archive, "load_all_rows", slow)
    app = OperatorApp(_async_factory(_ParentSession()))
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            card = await _mounted_card(app, pilot, details)
            card.focus()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("enter")  # the read starts and blocks
            await pilot.pause()
            assert card._full_diff_loading
            # Honest hint: the next press collapses, so the header offers that.
            header = card._build_content(100).plain.split("\n")[0]
            assert COLLAPSE_HINT in header and EXPAND_HINT not in header
            await pilot.press("enter")
            await pilot.pause()
            assert card.expanded is False, "a press during the read collapses the card"
            gate.set()
            for _ in range(200):
                await pilot.pause()
                if not card._full_diff_loading:
                    break
            await pilot.pause()
            assert card.expanded is False and not card._full_diff_shown
            # The rows landed in the cache: reopening needs no second read.
            assert card._full_diff_rows is not None
    finally:
        gate.set()


@pytest.mark.asyncio
async def test_a_paint_fault_after_the_read_does_not_close_the_tui(tmp_path, monkeypatch) -> None:
    details = await _archived_write(tmp_path)

    def broken(self: ToolCard) -> None:
        raise RuntimeError("paint fault")

    app = OperatorApp(_async_factory(_ParentSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        card = await _mounted_card(app, pilot, details)
        card.focus()
        await pilot.press("enter")
        await pilot.pause()
        monkeypatch.setattr(ToolCard, "_show_full_diff", broken)
        await pilot.press("enter")
        for _ in range(200):
            await pilot.pause()
            if not card._full_diff_loading:
                break
        for _ in range(3):
            await pilot.pause()
        assert app.return_code is None and app.is_running


@pytest.mark.asyncio
async def test_a_header_refresh_on_the_whole_diff_does_not_repaint_the_body(monkeypatch) -> None:
    """Q1: hovering an 8 000-row whole diff re-renders the rows on screen, not all of them.

    Measured on the base: every hover re-rendered all 8 002 rows (0.35-0.95 s
    of a frozen loop on this host). Pinned two ways: the rows the card renders
    for one header refresh are bounded by the viewport, and that refresh costs
    less than ONE full render of the same card on the same machine.
    """
    import time

    from textual.content import Content
    from textual.visual import Visual

    total = 8000
    rows = ["--- ", "+++ ", f"@@ -0,0 +1,{total} @@", *(f"+v_{n} = {n}" for n in range(total))]
    app = OperatorApp(_async_factory(FakeSession()))
    async with app.run_test(size=(110, 40)) as pilot:
        card = ToolCard("huge", "write", {"path": "x"})
        card.mark_done(
            "Wrote x",
            {"path": "x", "added": total, "removed": 0, "diff": rows[:203],
             DIFF_TRUNCATED_ROWS_KEY: total - 200},
        )
        app._append_block(card)
        await pilot.pause()
        card._full_diff_rows = rows  # the read's result, without the I/O
        card.focus()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(5):
            await pilot.pause()
        assert card.size.height == total + 2
        card._focused = False  # a hover repaints only an unfocused card
        painted: list[int] = []
        real = Visual.to_strips.__func__  # type: ignore[attr-defined]

        def counting(cls: Any, widget: Any, *args: Any, **kwargs: Any) -> Any:
            strips = real(cls, widget, *args, **kwargs)
            if widget is card:
                painted.append(len(strips))
            return strips

        monkeypatch.setattr(Visual, "to_strips", classmethod(counting))
        started = time.perf_counter()
        card._set_hovered(True)
        for _ in range(3):
            await pilot.pause()
        hover = time.perf_counter() - started
        assert EXPAND_HINT not in card._build_content(110).plain.split("\n")[0]
        assert 0 < sum(painted) <= 40, f"rendered {sum(painted)} rows for one hint"
        monkeypatch.setattr(Visual, "to_strips", classmethod(real))
        started = time.perf_counter()
        whole = Content.from_rich_text(card._build_content(card.size.width))
        Visual.to_strips(card, whole, card.size.width, None, card.visual_style)
        full_render = time.perf_counter() - started
        assert hover < full_render, (hover, full_render)

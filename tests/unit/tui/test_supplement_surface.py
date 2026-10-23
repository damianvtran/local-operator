"""The turn-supplements TUI surface (memo §2.7/§2.8/§3.2 T1–T5).

Three layers, in one file because they are one surface: the pure mapping
helpers (copy, row formatting, ``display_text``), the block widget as a
builder (no app), and the app seams that mount it — the live
``supplement_progress`` beat, the cold lazy read, ``/highlights``, the
settings rows and a live theme switch. Layout claims that need the real
``local_operator.tcss`` are driven through ``OperatorApp`` + pilot, the same
harness ``tests/unit/tui/test_settings_view.py`` documents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from rich.cells import cell_len
from rich.style import Style
from rich.text import Text

from local_operator.tui import supplements as supplement_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import SupplementProgress
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.supplement_block import SupplementBlock
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _factory

ANCHOR = "msg-final"
JOB = "abcdef1234567890"


def _file(path: str, name: str | None = None, size: int = 4120) -> dict[str, Any]:
    return {
        "path": path,
        "name": name or path.rsplit("/", 1)[-1],
        "size_bytes": size,
        "mtime": 1.0,
        "why": "written by write",
    }


def _rows(
    files: list[dict[str, Any]] | None = None,
    *,
    state: str = "done",
    more: list[str] | None = None,
    files_more: int | None = None,
    components: list[dict[str, Any]] | None = None,
    sounds: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "anchor": ANCHOR,
        "job": JOB,
        "version": 1,
        "state": state,
        "files": list(files or []),
        "components": list(components or []),
        "error": "",
        "at": 1.0,
    }
    if more is not None:
        details["more"] = list(more)
    if files_more is not None:
        details["files_more"] = files_more
    if sounds is not None:
        details["sounds"] = list(sounds)
    return details


def _block(details: dict[str, Any], **kwargs: Any) -> SupplementBlock:
    kwargs.setdefault("fold_width", 96)
    kwargs.setdefault("hyperlinks", False)
    return SupplementBlock(details, **kwargs)


# ---------------------------------------------------------------------------
# The pure helpers
# ---------------------------------------------------------------------------


def test_display_text_strips_controls_and_collapses_whitespace() -> None:
    assert supplement_mod.display_text("a\x1b[31mb\nc  d") == "ab c d"
    assert supplement_mod.display_text("x\x1b]8;;https://e\x07y") == "xy"
    assert supplement_mod.display_text(None) == ""
    assert supplement_mod.display_text(7) == ""


def test_row_helpers_match_the_copy_table() -> None:
    assert supplement_mod.files_summary_line(["a.md"]) == "1 file: a.md"
    assert supplement_mod.files_summary_line(["a.md", "b.csv"]) == "2 files: a.md, b.csv"
    assert supplement_mod.more_text(1) == "1 more file"
    assert supplement_mod.more_text(3) == "3 more files"
    assert supplement_mod.file_size_text(912) == "912 B"
    assert supplement_mod.file_size_text(4120) == "4 KB"
    assert supplement_mod.file_size_text(1_200_000) == "1.1 MB"
    assert supplement_mod.file_size_text(None) == ""
    name, path, size = supplement_mod.file_row_parts(_file("reports/q3.md", size=913))
    assert (name, path, size) == ("q3.md", "reports/q3.md", "913 B")
    # A file at the session root does not print its name twice.
    assert supplement_mod.file_row_parts(_file("report.md"))[1] == ""


# ---------------------------------------------------------------------------
# The block, as a pure builder
# ---------------------------------------------------------------------------


def test_block_files_only_is_the_one_line() -> None:
    block = _block(_rows([_file("reports/q3.md"), _file("data/bench.csv")]))
    assert _painted(block).plain == "  2 files: q3.md, bench.csv"


def test_block_rows_carry_label_path_size_and_the_more_row() -> None:
    details = _rows(
        [_file("reports/q3.md"), _file("data/bench.csv")],
        more=["reports/q4.md", "reports/q5.md", "data/raw.csv"],
        files_more=3,
        components=[{"attachment": "d" * 32, "mime": "text/html"}],
    )
    block = _block(details)
    assert block.disposition() == "block"
    assert _painted(block).plain.splitlines() == [
        "  q3.md  reports/q3.md  4 KB",
        "  bench.csv  data/bench.csv  4 KB",
        "  3 more files",
    ]
    # The more row is the click target; collapsing records its line.
    assert block._more_line == 2


def _sound(n: int) -> dict[str, Any]:
    return {
        "attachment": f"{n:032x}",
        "mime": "audio/mpeg",
        "title": f"Storm swell {n}",
        "duration_s": 2.0,
    }


def test_every_sound_gets_its_own_row_after_the_files() -> None:
    """14 sounds from one batch are 14 terminal rows, in generation order.

    The defect was a three-cap all the way down; a count or a single summary row
    would hide the same eleven sounds one layer higher.
    """
    block = _block(_rows([_file("reports/q3.md")], sounds=[_sound(n) for n in range(14)]))
    lines = _painted(block).plain.splitlines()
    assert len(lines) == 15
    # The files half keeps its own one-line arm (one file, no component); the sounds
    # still get a row each underneath it.
    assert lines[0] == "  1 file: q3.md"
    assert lines[1:] == [f"  ♪ Storm swell {n} · 2.0 s" for n in range(14)]
    # A sound-only row still paints: existence is never conditional on files.
    sounds_only = _block(_rows([], sounds=[_sound(0)]))
    assert _painted(sounds_only).plain == "  ♪ Storm swell 0 · 2.0 s"


def test_sound_row_is_text_and_never_claims_a_length_it_does_not_know() -> None:
    assert (
        supplement_mod.sound_row_text({"title": "Storm swell", "duration_s": 2.0})
        == "♪ Storm swell · 2.0 s"
    )
    # 0.0 is the trigger's "no usable length" sentinel: print no clock at all.
    assert supplement_mod.sound_row_text({"title": "Storm swell", "duration_s": 0}) == (
        "♪ Storm swell"
    )
    assert supplement_mod.sound_row_text({}) == "♪ Generated audio"
    assert len(supplement_mod.sound_rows([_sound(n) for n in range(14)])) == 14


def test_block_more_row_expands_in_place() -> None:
    details = _rows(
        [_file("reports/q3.md")],
        more=["reports/q4.md", "reports/q5.md"],
        files_more=2,
        components=[{"attachment": "d" * 32, "mime": "text/html"}],
    )
    block = _block(details)
    assert block._more_line == 1
    block.expand_more()
    assert _painted(block).plain.splitlines() == [
        "  q3.md  reports/q3.md  4 KB",
        "  q4.md  reports/q4.md",
        "  q5.md  reports/q5.md",
    ]
    assert block._more_line is None
    # Two rows at width 40: the size column is the part that gives way.
    narrow = _block(details)
    narrow._expanded = True
    narrow._rebuild(width=40)
    assert all(len(line) <= 40 for line in _painted(narrow).plain.splitlines())


def test_an_overflowing_row_keeps_its_unstyled_head() -> None:
    """Design round 1, D1: an overflow must eat the tail, never the name.

    With no OSC 8 link the indent and the file NAME are unstyled — they carry
    no span — and the spans-only fit emitted just the styled path/size, so an
    overflowing row lost exactly the characters that identify the file and
    started flush at the block's left edge. The fit now walks ``row.plain``
    from char 0: the head survives whole and the path is what ellipsizes.
    """
    block = _block(
        _rows(
            [_file("reports/experiments/2026-10-09/summary-of-results.md")],
            more=["reports/q3.md"],
            files_more=1,
            components=[{"attachment": "d" * 32, "mime": "text/html"}],
        )
    )
    block._rebuild(width=40)
    first, second = _painted(block).plain.splitlines()
    assert first == "  summary-of-results.md  reports/experi…"
    assert cell_len(first) == 40
    assert second == "  1 more file"


def test_block_skips_html_components_with_no_placeholder() -> None:
    """The operator's requirement: no stub, no box, no note — nothing at all."""
    block = _block(
        _rows(
            [_file("reports/q3.md")],
            components=[
                {"attachment": "d" * 32, "title": "latency chart", "mime": "text/html"},
                {"attachment": "e" * 32, "title": "bench grid", "mime": "text/html"},
            ],
        )
    )
    plain = _painted(block).plain
    assert plain.splitlines() == ["  q3.md  reports/q3.md  4 KB"]
    assert "latency chart" not in plain and "bench grid" not in plain
    # ...and a row whose ONLY contents are HTML components renders no block at
    # all (the app removes it; the builder's own answer is empty text).
    empty = _block(_rows([], components=[{"attachment": "d" * 32, "mime": "text/html"}]))
    assert _painted(empty).plain == ""


def test_block_preparing_line_leads_with_the_shimmer_canvas() -> None:
    block = _block(
        _rows([_file("reports/q3.md")], state="decided"),
        job_live=True,
    )
    assert block.disposition() == "files_preparing"
    lines = _painted(block).plain.splitlines()
    assert lines[0] == "  1 file: q3.md"
    assert lines[1].endswith(" Preparing highlights…")
    # The lead is `progress_graphic(None)`, not a literal glyph.
    from local_operator.tui import imagegen as imagegen_mod

    assert lines[1].startswith(imagegen_mod.progress_graphic(None, 0.0).plain)


def test_block_cancelled_and_failed_lines() -> None:
    cancelled = _block(_rows([_file("reports/q3.md")], state="cancelled"))
    assert cancelled.disposition() == "files_cancelled_retry"
    assert _painted(cancelled).plain.splitlines() == [
        "  1 file: q3.md",
        "  Highlights cancelled",
    ]
    failed = _block(_rows([], state="failed"))
    assert failed.disposition() == "failed_retry"
    assert _painted(failed).plain == "  Couldn't prepare highlights"
    # A cold non-terminal row is the stale-row rule, NOT the preparing line.
    cold = _block(_rows([_file("reports/q3.md")], state="decided"), job_live=False)
    assert cold.disposition() == "files_cancelled_retry"


def test_block_failed_line_names_the_typed_cause_when_present() -> None:
    """A typed cause says WHICH wall the retry would meet (operator, 2026-10-17); the bare
    sentence above stays byte-identical for rows without one."""
    quota = _rows([], state="failed")
    quota["cause"] = "provider_quota"
    assert _painted(_block(quota)).plain == (
        "  Couldn't prepare highlights — provider limit reached"
    )
    # An UNKNOWN cause string (a newer build's vocabulary) renders no phrase: inventing
    # prose for a vocabulary this build does not know is worse than the bare sentence.
    unknown = _rows([], state="failed")
    unknown["cause"] = "provider_from_the_future"
    assert _painted(_block(unknown)).plain == "  Couldn't prepare highlights"
    # The pure helper: absent, unknown and ``provider_unknown`` (the catch-all folds
    # ``aborted`` in) all map to "".
    assert supplement_mod.cause_phrase(None) == "" and supplement_mod.cause_phrase("") == ""
    assert supplement_mod.cause_phrase("provider_unknown") == ""
    assert supplement_mod.cause_phrase("provider_quota") == "provider limit reached"


def test_block_superseded_and_dismissed_render_nothing() -> None:
    for state, extra in (("cancelled", {"error": "superseded"}), ("skipped", {})):
        details = _rows([_file("reports/q3.md")], state=state)
        details.update(extra)
        block = _block(details)
        assert block.disposition() == "nothing"
        assert _painted(block).plain == ""


def test_block_links_only_files_the_affordance_admits(tmp_path: Path) -> None:
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "q3.md").write_text("report")
    # THREE files: the ≤ 2 files arm is the one-liner, and this test is about
    # the ROWS arm (each row is where a link is painted).
    details = _rows(
        [
            _file("reports/q3.md"),
            _file("missing.md"),
            _file("also-missing.csv"),
        ],
        files_more=0,
    )
    linked = _block(details, cwd=str(tmp_path), hyperlinks=True)
    uris = [
        span.style.link
        for span in _painted(linked).spans
        if isinstance(span.style, Style) and span.style.link
    ]
    assert uris and all(uri.startswith("file://") for uri in uris)
    assert any("q3.md" in uri for uri in uris)
    # The missing file paints as plain text: no link span names it.
    assert not any("missing.md" in uri for uri in uris)
    # And with the terminal not known to render links, nothing carries one.
    plain = _block(details, cwd=str(tmp_path), hyperlinks=False)
    assert not [
        span for span in _painted(plain).spans if isinstance(span.style, Style) and span.style.link
    ]


def test_more_row_click_maps_to_its_painted_line() -> None:
    details = _rows(
        [_file("reports/q3.md")],
        more=["reports/q4.md"],
        files_more=1,
        components=[{"attachment": "d" * 32, "mime": "text/html"}],
    )
    block = _block(details)
    assert block._more_line == 1

    class Event:
        def __init__(self, y: int) -> None:
            self.x = 4
            self.y = padding + y
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    padding = block.styles.padding.top if block.styles.padding else 0
    # link_at needs a composed cell; nothing is mounted, so it answers None and
    # the more-row branch is what runs. A click elsewhere changes nothing.
    other = Event(0)
    block.on_click(other)  # type: ignore[arg-type]
    assert block._expanded is False
    hit = Event(1)
    block.on_click(hit)  # type: ignore[arg-type]
    assert block._expanded is True and hit.stopped


# ---------------------------------------------------------------------------
# The app seams
# ---------------------------------------------------------------------------


async def _boot(app: OperatorApp, pilot: Any, session: Any) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is session:
            return
    raise AssertionError("session never adopted")


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _mount_answer(app: OperatorApp, anchor: str) -> AssistantBlock:
    block = AssistantBlock()
    block.completion_anchor_id = anchor
    app._append_block(block)
    return block


def _supplement_blocks(app: OperatorApp) -> list[SupplementBlock]:
    return [
        block for block in app._transcript_view().blocks() if isinstance(block, SupplementBlock)
    ]


def _painted(block: SupplementBlock) -> Text:
    """The block's painted ``Text``.

    ``Widget.renderable`` is typed as the whole ``RenderableType`` union, and
    every assertion in this file rests on a supplement block painting a
    ``Text`` (``_build`` returns one). Asserting it here means a block that
    started returning something else fails HERE and says so, rather than four
    tests down on a missing attribute — the same shape as
    ``test_user_block``'s helper.
    """
    renderable = block.renderable
    assert isinstance(renderable, Text)
    return renderable


@pytest.mark.asyncio
async def test_the_live_beat_mounts_under_its_answer_and_swaps_in_place() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 32)) as pilot:
        await _boot(app, pilot, session)
        answer = _mount_answer(app, ANCHOR)
        await pilot.pause()
        files = [_file("reports/q3.md")]
        app.post_message(SupplementProgress(ANCHOR, JOB, 1, "decided", files=files, components=[]))
        await pilot.pause()
        mounted = _supplement_blocks(app)
        assert len(mounted) == 1
        block = mounted[0]
        lines = _painted(block).plain.splitlines()
        assert lines[0] == "  1 file: q3.md"
        assert _painted(block).plain.splitlines()[1].endswith("Preparing highlights…")
        # Directly under the answer, by position.
        blocks = app._transcript_view().blocks()
        assert blocks[blocks.index(answer) + 1] is block
        # A repeat beat is idempotent; the terminal beat swaps the line for the
        # settled rows IN PLACE — same widget, no second block.
        app.post_message(SupplementProgress(ANCHOR, JOB, 1, "running", stage="generating"))
        app.post_message(SupplementProgress(ANCHOR, JOB, 1, "done", files=files, components=[]))
        await pilot.pause()
        assert _supplement_blocks(app) == [block]
        assert _painted(block).plain == "  1 file: q3.md"


@pytest.mark.asyncio
async def test_a_live_beat_paints_every_sound_and_a_later_beat_keeps_them() -> None:
    """Sounds ride the wire beat, and a beat that omits them must not erase them.

    The 14-row fixture again: the beat that carries the batch paints all 14, and
    the ``running`` beat (which carries no lists by design) merges over the row
    instead of replacing it -- the same rule the file callouts already follow.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 32)) as pilot:
        await _boot(app, pilot, session)
        _mount_answer(app, ANCHOR)
        await pilot.pause()
        sounds = [_sound(n) for n in range(14)]
        expected = [f"  ♪ Storm swell {n} · 2.0 s" for n in range(14)]
        app.post_message(
            SupplementProgress(ANCHOR, JOB, 1, "decided", files=[], components=[], sounds=sounds)
        )
        await pilot.pause()
        block = _supplement_blocks(app)[0]
        lines = _painted(block).plain.splitlines()
        assert lines[:14] == expected
        assert lines[14].endswith("Preparing highlights…")
        app.post_message(SupplementProgress(ANCHOR, JOB, 1, "running", stage="generating"))
        app.post_message(SupplementProgress(ANCHOR, JOB, 1, "done", files=[], components=[]))
        await pilot.pause()
        assert _supplement_blocks(app) == [block]
        assert _painted(block).plain.splitlines() == expected


@pytest.mark.asyncio
async def test_an_html_only_done_mounts_no_block() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 32)) as pilot:
        await _boot(app, pilot, session)
        _mount_answer(app, ANCHOR)
        await pilot.pause()
        app.post_message(SupplementProgress(ANCHOR, JOB, 1, "decided", files=[], components=[]))
        await pilot.pause()
        assert len(_supplement_blocks(app)) == 1  # the preparing line
        app.post_message(
            SupplementProgress(
                ANCHOR,
                JOB,
                1,
                "done",
                files=[],
                components=[{"attachment": "d" * 32, "mime": "text/html"}],
            )
        )
        await pilot.pause()
        assert _supplement_blocks(app) == [], "an HTML-only row leaves no frame"


@pytest.mark.asyncio
async def test_superseded_lifts_the_block() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 32)) as pilot:
        await _boot(app, pilot, session)
        _mount_answer(app, ANCHOR)
        app.post_message(
            SupplementProgress(ANCHOR, JOB, 1, "decided", files=[_file("a.md")], components=[])
        )
        await pilot.pause()
        assert len(_supplement_blocks(app)) == 1
        app.post_message(SupplementProgress(ANCHOR, JOB, 1, "cancelled", error="superseded"))
        await pilot.pause()
        assert _supplement_blocks(app) == []


@pytest.mark.asyncio
async def test_the_cold_read_mounts_a_journaled_row_once() -> None:
    """A row the live stream never carried (resume) arrives via the lazy read."""

    class Readable(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.reads: list[list[str]] = []

        async def supplements_for(self, anchors: list[str]) -> dict[str, dict[str, Any]]:
            self.reads.append(list(anchors))
            return {anchor: _rows([_file("reports/q3.md")], state="done") for anchor in anchors}

    session = Readable()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 32)) as pilot:
        await _boot(app, pilot, session)
        _mount_answer(app, ANCHOR)
        await pilot.pause()
        app._request_supplement_reads()
        for _ in range(20):
            await pilot.pause()
            if _supplement_blocks(app):
                break
        assert len(_supplement_blocks(app)) == 1, session.reads
        assert _painted(_supplement_blocks(app)[0]).plain == "  1 file: q3.md"
        assert session.reads and session.reads[0] == [ANCHOR]
        # The anchor that a read answered for is not asked twice.
        app._request_supplement_reads()
        await pilot.pause()
        assert session.reads == [[ANCHOR]]


@pytest.mark.asyncio
async def test_a_beat_for_an_unmounted_answer_re_arms_at_the_mount() -> None:
    """The deferral the early return promises is delivered by the projection.

    A beat can land while its answer is outside the rendered bound: the row is
    held, no block mounts, and the terminal beat's own read sits below that
    early return, so it never asks. When the answer's page mounts, the
    projection read must re-arm for the live, blockless anchor — scoping the
    live-skip to anchors with a MOUNTED block is what makes the early return's
    promise ("a later projection re-reads and mounts it then") true.
    """

    class Readable(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.reads: list[list[str]] = []

        async def supplements_for(self, anchors: list[str]) -> dict[str, dict[str, Any]]:
            self.reads.append(list(anchors))
            return {anchor: _rows([_file("reports/q3.md")], state="done") for anchor in anchors}

    session = Readable()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 32)) as pilot:
        await _boot(app, pilot, session)
        # The beat lands with no answer mounted: the row is held, no block, and
        # the beat path's read is never asked (the early return sits above it).
        app.post_message(SupplementProgress(ANCHOR, JOB, 1, "done", files=[_file("reports/q3.md")]))
        for _ in range(5):
            await pilot.pause()
        assert _supplement_blocks(app) == []
        assert session.reads == []
        # The answer's page mounts and the next projection asks; the read
        # mounts the held row under it, and is not asked again.
        _mount_answer(app, ANCHOR)
        app._request_supplement_reads()
        for _ in range(20):
            await pilot.pause()
            if _supplement_blocks(app):
                break
        assert len(_supplement_blocks(app)) == 1, session.reads
        assert _painted(_supplement_blocks(app)[0]).plain == "  1 file: q3.md"
        assert session.reads == [[ANCHOR]]
        app._request_supplement_reads()
        await pilot.pause()
        assert session.reads == [[ANCHOR]]


@pytest.mark.asyncio
async def test_a_live_beat_wins_over_a_stale_read() -> None:
    class Readable(FakeSession):
        async def supplements_for(self, anchors: list[str]) -> dict[str, dict[str, Any]]:
            return {ANCHOR: _rows([_file("reports/q3.md")], state="decided")}

    session = Readable()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 32)) as pilot:
        await _boot(app, pilot, session)
        _mount_answer(app, ANCHOR)
        app.post_message(SupplementProgress(ANCHOR, JOB, 1, "done", files=[_file("reports/q3.md")]))
        await pilot.pause()
        app._request_supplement_reads()
        for _ in range(20):
            await pilot.pause()
        # The settled row stands: the read (a non-terminal row, cold) never
        # repaints it back to the stale-cancelled line.
        assert _painted(_supplement_blocks(app)[0]).plain == "  1 file: q3.md"


@pytest.mark.asyncio
async def test_a_terminal_beat_upgrades_itself_with_the_full_row() -> None:
    """The event carries the FEATURED files; the read serves the ROW.

    ``files_more``/``more`` ride the journaled row only (memo §2.7), so the app
    asks for one read after a terminal beat — and applies it because it is the
    SAME (version, rank): that is what makes the live block identical to the
    replayed one instead of the "N more files" disclosure appearing only after
    a reload.
    """
    full = _rows(
        [_file("reports/q3.md"), _file("reports/q4.md")],
        more=["data/raw.csv", "reports/appendix.md", "data/summary.json"],
        files_more=3,
        components=[{"attachment": "f" * 32, "mime": "text/html", "title": "chart"}],
    )

    class Readable(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.reads: list[list[str]] = []

        async def supplements_for(self, anchors: list[str]) -> dict[str, dict[str, Any]]:
            self.reads.append(list(anchors))
            return {ANCHOR: dict(full)}

    session = Readable()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 32)) as pilot:
        await _boot(app, pilot, session)
        _mount_answer(app, ANCHOR)
        # The event's own shape: featured files only, no disclosure keys.
        app.post_message(
            SupplementProgress(
                ANCHOR,
                JOB,
                1,
                "done",
                files=[_file("reports/q3.md"), _file("reports/q4.md")],
            )
        )
        for _ in range(20):
            await pilot.pause()
            if "3 more files" in _painted(_supplement_blocks(app)[0]).plain:
                break
        assert session.reads, "the terminal beat never asked for its row"
        plain = _painted(_supplement_blocks(app)[0]).plain
        assert "3 more files" in plain, plain
        # The HTML component is still skipped, placeholder-free.
        assert "chart" not in plain


@pytest.mark.asyncio
async def test_highlights_command_receipts_and_refusals() -> None:
    class Ops(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.cancels: list[tuple[str, str]] = []
            self.restarts: list[tuple[str, str]] = []
            self.steers: list[tuple[str, str, str]] = []

        async def supplement_cancel(self, anchor: str, job: str) -> str:
            self.cancels.append((anchor, job))
            return "cancelled"

        async def supplement_restart(self, anchor: str, job: str) -> str:
            self.restarts.append((anchor, job))
            return "restarting"

        async def supplement_steer(self, anchor: str, job: str, text: str) -> str:
            self.steers.append((anchor, job, text))
            return "steering"

    session = Ops()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 32)) as pilot:
        await _boot(app, pilot, session)
        _mount_answer(app, ANCHOR)
        await pilot.pause()

        app._run_slash_command("/highlights")
        await pilot.pause()
        assert any("Usage: /highlights" in text for text in _notices(app))
        app._run_slash_command("/highlights cancel")
        await pilot.pause()
        assert not session.cancels, "nothing was rendered, so nothing can be cut"

        app.post_message(SupplementProgress(ANCHOR, JOB, 1, "decided", files=[_file("a.md")]))
        await pilot.pause()
        app._run_slash_command("/highlights cancel")
        for _ in range(20):
            await pilot.pause()
            if session.cancels:
                break
        assert session.cancels == [(ANCHOR, JOB)]
        assert any("Highlights cancelled" in text for text in _notices(app))

        # The runner's own beat is what moves the app's row to `cancelled`; the
        # fake receipt does not, so the retry below targets what a real flow
        # would have: the settled row.
        app.post_message(SupplementProgress(ANCHOR, JOB, 1, "cancelled"))
        await pilot.pause()
        app._run_slash_command("/highlights retry")
        for _ in range(20):
            await pilot.pause()
            if session.restarts:
                break
        assert session.restarts == [(ANCHOR, JOB)]
        assert any("Preparing highlights again" in text for text in _notices(app))

        app._run_slash_command("/highlights adjust make it a table")
        for _ in range(20):
            await pilot.pause()
            if session.steers:
                break
        assert session.steers == [(ANCHOR, JOB, "make it a table")]


@pytest.mark.asyncio
async def test_settings_shows_the_highlights_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The section renders with the shipped defaults, scoped like its peers."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    from local_operator.tui.settings import settings_reload

    settings_reload()
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(app, pilot, session)
        app._run_slash_command("/settings")
        await pilot.pause()
        from local_operator.tui.widgets.settings_view import SettingsView

        view = app.query_one(SettingsView)
        keys = {
            row.setting.key: row
            for row in view._rows
            if getattr(row, "kind", "") == "setting" and row.setting is not None
        }
        for key in (
            "supplements.enabled",
            "supplements.files",
            "supplements.graphics",
            "supplements.model",
            "supplements.maxCostUsd",
            "supplements.timeoutS",
        ):
            assert key in keys, f"{key} missing from /settings"
        enabled = keys["supplements.enabled"].setting
        assert enabled is not None and enabled.default is True
        graphics = keys["supplements.graphics"].setting
        assert graphics is not None and graphics.default is True
        cost = keys["supplements.maxCostUsd"].setting
        assert cost is not None and cost.default == 1.00
        # The section is scoped to new sessions, like `monitor` beside it.
        from local_operator import settings_io

        section = next(entry for entry in settings_io.SECTIONS if entry.name == "supplements")
        assert section.scope.name == "NEW_SESSIONS"
        assert section.title == "Highlights"


@pytest.mark.asyncio
async def test_a_live_theme_switch_re_inks_the_block() -> None:
    from local_operator.tui import theme as theme_mod

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 32)) as pilot:
        await _boot(app, pilot, session)
        _mount_answer(app, ANCHOR)
        app.post_message(SupplementProgress(ANCHOR, JOB, 1, "done", files=[_file("reports/q3.md")]))
        await pilot.pause()
        block = _supplement_blocks(app)[0]

        def ink(name: str) -> list[str]:
            colors = sorted(
                {
                    span.style.color.get_truecolor().hex
                    for span in _painted(block).spans
                    if isinstance(span.style, Style) and span.style.color
                }
            )
            assert colors, "the block resolved no tokens"
            expected = {theme_mod.semantic_color(token, name).lower() for token in ("dim", "muted")}
            assert (
                set(colors) <= expected
            ), "the block painted ink that is not one of its own tokens"
            return colors

        app._apply_theme("dark")
        await pilot.pause()
        dark = ink("dark")
        app._apply_theme("light")
        await pilot.pause()
        light = ink("light")
        assert dark != light, "the block kept its pre-switch ramp"


@pytest.mark.asyncio
async def test_image_entries_mount_through_the_existing_transport(tmp_path: Path) -> None:
    """A row's ``images`` mount as real ImageBlocks, right under the block.

    The engine emits no ``images`` yet (the generator produces HTML components),
    so this drives the app's own apply seam with a contract-shaped row — the
    path the first lane to emit them will reach — and asserts the transport is
    the existing ``ImageBlock`` (one renderer), positioned after the supplement
    block rather than at the tail.
    """
    import base64

    from local_operator.session import attachments as attachments_mod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    try:
        raw = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
            "h6FO1AAAAABJRU5ErkJggg=="
        )
        ref = attachments_mod.AttachmentStore().put_bytes(raw, "image/png")
        assert ref is not None
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 32)) as pilot:
            await _boot(app, pilot, session)
            _mount_answer(app, ANCHOR)
            details = _rows([_file("reports/q3.md")])
            details["images"] = [
                {"attachment": ref.digest, "mime": "image/png", "title": "latency"}
            ]
            app._apply_supplement(ANCHOR, details, job_live=False)
            await pilot.pause()
            kinds = [type(block).__name__ for block in app._transcript_view().blocks()]
            assert "SupplementBlock" in kinds and "ImageBlock" in kinds, kinds
            assert kinds.index("ImageBlock") == kinds.index("SupplementBlock") + 1
            # A repeat apply (the idempotent path) never doubles it.
            app._apply_supplement(ANCHOR, details, job_live=False)
            await pilot.pause()
            assert (
                kinds.count("ImageBlock")
                == [type(block).__name__ for block in app._transcript_view().blocks()].count(
                    "ImageBlock"
                )
                == 1
            )
    finally:
        monkeypatch.undo()

"""``display.rail``: the assistant gutter rail, and turning it off.

The rail gives the model's answer the same left-edge delineation the user's
prompt has carried since ``UserBlock`` grew one. PR #1229's maintainer asked
for it to be configurable — "on my end I'd probably turn it off" — so it is a
BOOL registry setting defaulting to ON.

**The promise this file exists to keep is that OFF is not "the rail, unpainted".**
It is the pre-rail build: the rail costs two cells of the lane, and a gate that
skipped only the PAINT would leave the prose folded two cells narrower than its
box with nothing in the space it left — an indent the reader did not ask for
and cannot explain. So the flag is read at five places that all have to agree:
the paint in ``_apply_rows``, the fold in ``_flat_width`` and ``authored_width``,
the ``copy_gutter`` the clipboard strips, and the de-rail slice plus the three
selection compensations in ``get_selection``. ``_rail_cols()`` is the single
answer they share, read at PAINT RATE rather than cached, which is also what
makes a mid-session flip reach blocks already on screen.

``T2-a`` is the test that makes the promise honest: it renders through a clean
export of the actual pre-rail commit (``bf67bf69``) and compares row lists
string for string, rather than asserting in-tree properties that would pass
against a build that had drifted.

**The promise is bounded at lane >= 8, deliberately and measurably.** Pre-rail
``AssistantBlock`` had no ``MIN_BODY`` floor — ``_flat_width`` was a bare
``return self.fold_width(FALLBACK_WIDTH)`` — and no ``authored_width`` or
``copy_gutter`` override at all. The floor arrived WITH the rail, for the reason
``UserBlock`` records: folding prose into the last two or three cells turns a
sentence into a column of single characters, so the rows are built wider than
the frame and Rich clips them instead. With the rail off the floor still
applies, so a lane below ``MIN_BODY`` folds at 8 where the pre-rail build folded
at the lane. That divergence is asserted here explicitly
(``test_rail_off_below_the_min_body_floor_diverges_from_pre_rail``) rather than
avoided by only testing wide frames: it is reachable — a terminal 9 columns or
narrower gets there — and the floor is better behaviour than what it replaced,
so the claim is narrowed to match the code instead of the code being bent to
match the claim.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from textual.content import Content

from local_operator import settings_io
from local_operator.tui.widgets.assistant import (
    DEFAULT_RAIL,
    MIN_BODY,
    QUOTE_BAR,
    RAIL,
    RAIL_COLS,
    AssistantBlock,
)
from local_operator.tui.widgets.transcript import TranscriptView
from tests.unit.tui.conftest import StyledTranscriptApp

from .test_assistant_rail import MIXED_CONSTRUCTS, THREE_PARAGRAPHS

#: The commit the rail landed in. Its PARENT is the pre-rail tree ``T2-a``
#: compares against, resolved through git rather than written out so the export
#: cannot drift from the branch it is supposed to be the baseline for.
RAIL_COMMIT = "d52f28f6"


@pytest.fixture
def _rail_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``display.rail`` OFF for one test.

    Monkeypatched on the CONSUMING module rather than by writing a config file,
    the ``_markers_on`` idiom from ``test_copy_markdown.py``: the production
    code reads a module-level ``settings_get`` binding, so patching it there is
    the same seam the app uses. Every other key delegates to the real reader, so
    a test that flips the rail does not silently flatten the rest of the
    settings.
    """
    import local_operator.tui.widgets.assistant as _assistant

    real = _assistant.settings_get
    monkeypatch.setattr(
        _assistant,
        "settings_get",
        lambda key, default=None: (False if key == "display.rail" else real(key, default)),
    )


@pytest.fixture
def _rail_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``display.rail`` ON, the same way — used where a test pairs ON
    against OFF and neither side should depend on the ambient default."""
    import local_operator.tui.widgets.assistant as _assistant

    real = _assistant.settings_get
    monkeypatch.setattr(
        _assistant,
        "settings_get",
        lambda key, default=None: (True if key == "display.rail" else real(key, default)),
    )


async def _rows(text: str, size: tuple[int, int]) -> list[str]:
    """The rows a finalized block paints at ``size``.

    Two pauses after the text, as ``test_assistant_rail._block`` does it: the
    first mounts and lays the block out, the second lets the resize re-fold it
    against its real width.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=size) as pilot:
        view = app.query_one(TranscriptView)
        block = AssistantBlock()
        view.append_block(block)
        await pilot.pause()
        block.update_text(text)
        block.finalize_text()
        await pilot.pause()
        await pilot.pause()
        visual = block._render()
        assert isinstance(visual, Content)
        return visual.plain.split("\n")


# --------------------------------------------------------------------------
# On
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("_rail_on")
async def test_the_rail_is_painted_when_the_setting_is_on() -> None:
    """ON is the shipped behaviour: every row carries the rail."""
    rows = await _rows(MIXED_CONSTRUCTS, (60, 24))
    missing = [(i, row) for i, row in enumerate(rows) if row.strip() and not row.startswith(RAIL)]
    assert not missing, f"rows without the rail: {missing!r}"


# --------------------------------------------------------------------------
# Off — the honest promise
# --------------------------------------------------------------------------


def _pre_rail_export() -> Path | None:
    """A clean tree at the pre-rail commit, or ``None`` if git cannot supply it.

    ``git archive`` into a directory this function created — never ``cp -R`` of
    a worktree, whose ``.git`` is a pointer sharing the LIVE INDEX, and never
    ``git stash``, which is repo-global and would reach across every worktree on
    the machine.
    """
    repo = Path(__file__).resolve().parents[3]
    try:
        parent = subprocess.run(
            ["git", "rev-parse", f"{RAIL_COMMIT}^"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None

    dest = Path(tempfile.mkdtemp(prefix="pre-rail-"))
    try:
        archive = subprocess.run(
            ["git", "archive", parent], cwd=repo, capture_output=True, check=True
        ).stdout
        subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    except (subprocess.CalledProcessError, OSError):
        shutil.rmtree(dest, ignore_errors=True)
        return None
    return dest


def _pre_rail_rows(tree: Path, text: str, size: tuple[int, int]) -> list[str]:
    """Render ``text`` at ``size`` through the pre-rail tree, in a subprocess.

    A subprocess because the two trees define the same module path: importing
    both into one interpreter would have them fight over
    ``local_operator.tui.widgets.assistant`` in ``sys.modules`` and silently
    compare a tree against itself, which is the failure mode that would make
    this test pass while proving nothing.
    """
    script = """
import asyncio, json, sys
from textual.content import Content
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.transcript import TranscriptView

text, cols, rows_n = json.loads(sys.argv[1])

# The export's OWN copy of the shipped-CSS harness, so both sides of the
# comparison are folded under the same stylesheet and the same lane. Rendering
# the pre-rail tree under an ad-hoc App instead makes the rows differ by their
# trailing pad — the block gets a different width — which looks exactly like the
# fold regression this test exists to catch.
from tests.unit.tui.conftest import StyledTranscriptApp as Probe

async def main():
    app = Probe()
    async with app.run_test(size=(cols, rows_n)) as pilot:
        view = app.query_one(TranscriptView)
        block = AssistantBlock()
        view.append_block(block)
        await pilot.pause()
        block.update_text(text)
        block.finalize_text()
        await pilot.pause()
        await pilot.pause()
        visual = block._render()
        assert isinstance(visual, Content)
        print(json.dumps(visual.plain.split("\\n")))

asyncio.run(main())
"""
    import json

    out = subprocess.run(
        [sys.executable, "-c", script, json.dumps([text, size[0], size[1]])],
        cwd=str(tree),
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(tree), "PATH": "/usr/bin:/bin", "HOME": str(tree)},
    )
    if out.returncode != 0:
        raise RuntimeError(f"pre-rail render failed: {out.stderr[-2000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.asyncio
@pytest.mark.usefixtures("_rail_off")
@pytest.mark.parametrize("size", [(100, 30), (60, 40)])
@pytest.mark.parametrize("text", [THREE_PARAGRAPHS, MIXED_CONSTRUCTS])
async def test_rail_off_renders_byte_identical_rows_to_the_pre_rail_build(
    text: str, size: tuple[int, int]
) -> None:
    """T2-a. OFF is the pre-rail build, string for string, for lane >= 8.

    The necessity test for the whole slice, and the reason the gate is at five
    sites rather than one. Compared against a clean ``git archive`` export of
    the actual pre-rail commit rather than against in-tree properties, because
    the claim is about a BUILD, and only the build can answer it.

    Faithful revert shape: remove the gate from ``_flat_width`` and
    ``authored_width`` only, leaving ``_apply_rows`` gated. The rows then differ
    because the prose folded two cells narrower than the pre-rail build — the
    exact defect this test exists for, and one a single-site mutation of the
    paint would not catch.
    """
    tree = _pre_rail_export()
    if tree is None:
        pytest.skip("git archive of the pre-rail commit is unavailable here")
    try:
        expected = _pre_rail_rows(tree, text, size)
    finally:
        shutil.rmtree(tree, ignore_errors=True)

    got = await _rows(text, size)
    assert got == expected, (
        "rail OFF must render the pre-rail build exactly.\n"
        f"got:      {got!r}\n"
        f"pre-rail: {expected!r}"
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("_rail_off")
async def test_rail_off_below_the_min_body_floor_diverges_from_pre_rail() -> None:
    """The KNOWN divergence, asserted rather than avoided — lane < ``MIN_BODY``.

    Pre-rail ``_flat_width`` was ``return self.fold_width(FALLBACK_WIDTH)``, with
    no floor: at lane 6 it folded at 6. The floor arrived with the rail and
    still applies when the rail is off, so the same lane folds at 8 here.

    This is the one place "OFF restores pre-rail rendering" is FALSE, and it is
    reachable: a terminal 9 columns or narrower produces a lane under 8
    (measured — at terminal width 6 the lane is 6). It is documented and pinned
    here rather than quietly excluded by only testing wide frames, because an
    undocumented divergence is what turns an honest promise into a false one.

    The floor is kept deliberately: folding into two or three cells is worse
    than letting Rich clip. Making it conditional on a display setting — a floor
    that exists only when the rail is on — would be a worse defect than the
    divergence it removed, so the CLAIM is narrowed to lane >= 8 instead.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(6, 20)) as pilot:
        view = app.query_one(TranscriptView)
        block = AssistantBlock()
        view.append_block(block)
        await pilot.pause()
        block.update_text("alpha beta gamma")
        block.finalize_text()
        await pilot.pause()
        await pilot.pause()

        lane = block.fold_width(80)
        assert lane < MIN_BODY, f"this test needs a lane under the floor; got {lane}"
        # The divergence, stated exactly: the floor, not the lane, and not the
        # lane less the rail either — the rail is off, so RAIL_COLS is not in it.
        assert block._flat_width() == MIN_BODY, block._flat_width()
        assert block._flat_width() != lane, (block._flat_width(), lane)
        assert block._rail_cols() == 0, "the rail is off; the gutter must cost nothing"


@pytest.mark.asyncio
@pytest.mark.usefixtures("_rail_off")
async def test_the_quote_bar_still_renders_when_the_rail_is_off() -> None:
    """T2-b. Turning the rail off must not take Rich's blockquote bar with it.

    The interaction the maintainer's two requests create: slice 1 made a quote
    row carry both marks, so an over-broad OFF that suppressed "the bar on a
    quote row" would remove the model's content rather than the block's chrome.
    """
    rows = await _rows(MIXED_CONSTRUCTS, (60, 24))
    assert all(RAIL not in row for row in rows), rows

    quoted = [row for row in rows if QUOTE_BAR in row]
    assert quoted, f"the blockquote lost its bar entirely: {rows!r}"
    for row in quoted:
        # Bare column 0: with no gutter painted, the bar opens the row.
        assert row.startswith(QUOTE_BAR), row


@pytest.mark.asyncio
async def test_copy_is_clean_with_the_rail_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """T2-c. The clipboard is the same document either way.

    The de-rail slice is the thing most likely to be off by two: with the rail
    off the rows carry no gutter, so a constant ``row[RAIL_COLS:]`` would eat
    the first two characters of real content. Asserted as ON == OFF rather than
    against a written-out string, because the claim is that the setting is
    invisible to the clipboard.
    """
    from textual.geometry import Offset
    from textual.selection import Selection

    import local_operator.tui.widgets.assistant as _assistant

    real = _assistant.settings_get

    async def _copied(flag: bool) -> str:
        monkeypatch.setattr(
            _assistant,
            "settings_get",
            lambda key, default=None: (flag if key == "display.rail" else real(key, default)),
        )
        app = StyledTranscriptApp()
        async with app.run_test(size=(60, 24)) as pilot:
            view = app.query_one(TranscriptView)
            block = AssistantBlock()
            view.append_block(block)
            await pilot.pause()
            block.update_text(MIXED_CONSTRUCTS)
            block.finalize_text()
            await pilot.pause()
            await pilot.pause()
            visual = block._render()
            assert isinstance(visual, Content)
            rows = visual.plain.split("\n")
            selection = Selection(Offset(0, 0), Offset(len(rows[-1]), len(rows) - 1))
            got = block.get_selection(selection)
            assert got is not None
            return got[0]

    on = await _copied(True)
    off = await _copied(False)
    assert on == off, f"the flag changed the clipboard.\non:  {on!r}\noff: {off!r}"
    assert RAIL not in on and RAIL not in off, (on, off)
    # The quote survives as markdown in both, which is what the copy is for.
    assert "> a quoted line" in off, off


@pytest.mark.asyncio
async def test_flipping_the_setting_repaints_a_mounted_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2-d. A mid-session flip reaches blocks already on screen.

    Pins the live-apply claim through the REAL seam rather than by asserting a
    handler was called: ``display.*`` changes run ``_repaint_themed_widgets``,
    which calls ``retheme`` on every block, which re-enters ``_apply_rows``,
    which is where the flag is read. Because the read happens at paint time and
    is never cached on the instance, that path needs no new live-apply code —
    and this test is what would notice if someone cached it.
    """
    import local_operator.tui.widgets.assistant as _assistant

    real = _assistant.settings_get
    state = {"on": True}
    monkeypatch.setattr(
        _assistant,
        "settings_get",
        lambda key, default=None: (state["on"] if key == "display.rail" else real(key, default)),
    )

    app = StyledTranscriptApp()
    async with app.run_test(size=(60, 24)) as pilot:
        view = app.query_one(TranscriptView)
        block = AssistantBlock()
        view.append_block(block)
        await pilot.pause()
        block.update_text(THREE_PARAGRAPHS)
        block.finalize_text()
        await pilot.pause()
        await pilot.pause()

        visual = block._render()
        assert isinstance(visual, Content)
        assert all(row.startswith(RAIL) for row in visual.plain.split("\n") if row.strip())

        state["on"] = False
        block.retheme()
        await pilot.pause()

        visual = block._render()
        assert isinstance(visual, Content)
        after = visual.plain.split("\n")
        assert all(RAIL not in row for row in after), after
        # ...and the fold widened to reclaim the two cells, or the flip left an
        # indent behind: the paint and the fold must move together.
        assert block._built_width == block.fold_width(80), (
            block._built_width,
            block.fold_width(80),
        )


def test_the_registry_default_matches_the_render_edge_constant() -> None:
    """T2-e. One value, two homes, pinned in both directions.

    ``settings_io`` is what the config UI and the CLI write; ``DEFAULT_RAIL`` is
    what the render edge falls back to when no config is readable. They must
    agree or an untouched install renders one way and a default-resolved read
    says the other. Asserted both ways round so changing EITHER alone goes red,
    which is what makes flipping the default the one-token change the MR offers
    the maintainer.
    """
    entry = settings_io.BY_KEY["display.rail"]
    assert entry.default is DEFAULT_RAIL
    assert DEFAULT_RAIL is entry.default
    assert entry.kind is settings_io.Kind.BOOL
    # The flat-dotted path, not a nested mapping nothing reads.
    assert entry.path == ("display.rail",)


def test_the_rail_constant_is_the_painted_geometry_not_the_runtime_answer() -> None:
    """``RAIL_COLS`` stays 2 with the flag off; ``_rail_cols()`` is the answer.

    The distinction the 34 constant-sliced tests in ``test_transcript_selection``,
    ``test_subagent_view`` and ``test_rendered_history_paging`` depend on: they
    slice ``row[RAIL_COLS:]`` and offset selections by it, and they run with the
    rail ON. ``RAIL_COLS`` is the width of a rail that IS painted — geometry —
    so it must not become 0 when the setting is off, or those tests would be
    asserting against a constant that moved under them.
    """
    assert RAIL_COLS == 2

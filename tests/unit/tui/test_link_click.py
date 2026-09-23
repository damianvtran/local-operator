"""A click on a painted URL in the transcript opens it — through the real app.

"Hyperlinks don't work in the lop TUI under Ghostty" had two independent causes,
and this file drives the real app past both of them:

* **The driver claims the mouse, and nothing answered the click.** Textual's
  driver writes ``\\x1b[?1000h`` / ``\\x1b[?1003h`` at startup, and a terminal
  reporting mouse events to an application does not run its own click-to-open
  gesture; Textual's dispatch routes ``@click`` action meta and a rich ``link=``
  style carries none. So ``TranscriptBlock.on_click`` resolves the clicked cell's
  link and asks the app to open it. The markdown-link and autolink tests below
  are the shapes that already carried a span and still did nothing.
* **A bare URL carried no span at all.** Rich sets ``link=`` for
  ``[label](target)`` and ``<target>`` and nothing for ``https://a.test/x``
  written plainly, so there was nothing under the pointer to find.
  :func:`~local_operator.tui.link_markup.autolink_bare_urls` promotes it before
  the render, and the bare-URL test is its regression: with the promotion
  reverted that test goes red while the other two stay green.

Two properties of the click route are asserted beyond "something opened":

* the URL handed to the opener is EXACTLY the one clicked — a click that opens a
  near neighbour is indistinguishable from a working one at a glance;
* prose opens nothing, on the same rendered row as a link, because the click
  resolution reads the cell's style and an off-by-one that reached the adjacent
  link would otherwise pass every test above.

The opener is stubbed through ``monkeypatch`` on ``local_operator.mcp.auth`` —
the module ``OperatorApp._open_link`` imports it from — so no test launches a
browser on the developer's machine. The open runs in a worker that awaits a child
process, so every click is followed by polling the recorder for frames; a single
``pause`` is not enough and the shapes were measured failing that way.

No timing assertions: the polls are counted frames, not a clock (AGENTS.md,
"Timing, flakes, and how to assert that something is fast").
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

import local_operator.mcp.auth as auth_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.transcript import TranscriptView, UserBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory

MD_URL = "https://example.com/merge_requests/412"
AUTO_URL = "https://docs.example.com/ci/yaml/"
BARE_URL = "https://example.com/pipelines/88213"

#: Longer than the 96-cell body column at the pilot's 100-column width, so rich
#: folds it across two rows. The fold is asserted structurally rather than
#: assumed, so a width change fails loudly instead of quietly testing one row.
WRAPPED_URL = (
    "https://example.com/very/long/path/that/goes/on/and/on/for/a/while"
    "/until/it/definitely/wraps/somewhere/tail"
)


def _recorder(opened: list[str]):
    """An async stand-in for the browser boundary, recording what it was handed.

    Installed on ``local_operator.mcp.auth`` because that is where ``_open_link``
    imports ``open_browser_quietly`` from, inside the call — patching the app
    module instead would leave the real opener in place.
    """

    async def _open(url: str) -> bool:
        opened.append(url)
        return True

    return _open


@asynccontextmanager
async def _painted_answer(message: str, monkeypatch, *, width: int = 100):
    """Boot the real app holding one finalized answer, plus the recorder.

    The prompt above the answer is the shape the repro script drives: a click has
    to survive the transcript's own focus handling on a conversation with more
    than one block in it.
    """
    opened: list[str] = []
    monkeypatch.setattr(auth_mod, "open_browser_quietly", _recorder(opened))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(width, 30)) as pilot:
        await pilot.pause()
        block = AssistantBlock()
        block.update_text(message)
        block.finalize_text()
        app._append_block(UserBlock("where is that report?", fold_width=width))
        app._append_block(block)
        await pilot.pause()
        await pilot.pause()

        transcript = app.query_one(TranscriptView)
        answer = [b for b in transcript.blocks() if isinstance(b, AssistantBlock)][0]
        yield pilot, answer, opened


def _painted_rows(block: AssistantBlock) -> list[str]:
    """The block's painted text, row by row."""
    return getattr(block.renderable, "plain", "").split("\n")


def _cell(block: AssistantBlock, needle: str) -> tuple[int, int]:
    """The block-relative ``(x, y)`` of the first cell of ``needle``.

    Raises rather than returning ``None``: a needle that was never painted IS the
    failure these tests exist for, and a click at a guessed offset would land on
    prose and pass for the wrong reason.
    """
    for y, row in enumerate(_painted_rows(block)):
        x = row.find(needle)
        if x >= 0:
            return x, y
    raise AssertionError(f"{needle!r} is not painted in {_painted_rows(block)!r}")


async def _click(pilot, block: AssistantBlock, cell: tuple[int, int], opened: list[str]) -> None:
    """Click ``cell``, then give the open worker the frames it needs.

    The offset is ``x + 1`` because the cell is indexed from the painted TEXT
    while the widget's first column carries the block's spine. The poll is 40
    frames and gives up when nothing has been recorded — which is what makes it
    equally usable for the prose test, where the assertion is that the recorder
    is still empty after waiting as long as an open could take.
    """
    x, y = cell
    await pilot.click(block, offset=(x + 1, y))
    for _ in range(40):
        await pilot.pause()
        if opened:
            return


# --- one click per shape -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_click_on_a_markdown_link_opens_exactly_that_url(monkeypatch) -> None:
    """Rich already painted this span; before the fix the click did nothing."""
    message = f"Start with [the pull request]({MD_URL}) and read on.\n"
    async with _painted_answer(message, monkeypatch) as (pilot, answer, opened):
        await _click(pilot, answer, _cell(answer, "the pull request"), opened)
    assert opened == [MD_URL]


@pytest.mark.asyncio
async def test_a_click_on_a_bare_url_opens_exactly_that_url(monkeypatch) -> None:
    """The regression for the second cause: no span, so nothing to find.

    Revert ``autolink_bare_urls`` to the identity and this is the test that goes
    red, while the markdown-link and autolink tests above stay green.
    """
    message = f"The run that produced it is {BARE_URL} — the logs are there.\n"
    async with _painted_answer(message, monkeypatch) as (pilot, answer, opened):
        await _click(pilot, answer, _cell(answer, BARE_URL), opened)
    assert opened == [BARE_URL]


@pytest.mark.asyncio
async def test_a_click_on_an_autolink_opens_exactly_that_url(monkeypatch) -> None:
    message = f"Read <{AUTO_URL}> before changing the pipeline.\n"
    async with _painted_answer(message, monkeypatch) as (pilot, answer, opened):
        await _click(pilot, answer, _cell(answer, AUTO_URL), opened)
    assert opened == [AUTO_URL]


@pytest.mark.asyncio
async def test_a_click_on_prose_opens_nothing(monkeypatch) -> None:
    """Prose on the same painted row as a link, so an off-by-one is caught.

    The click route resolves a cell's style, and a resolution that reached one
    cell too far would open the neighbouring URL on every test above.
    """
    message = f"Deployed at {BARE_URL} on Tuesday.\n"
    async with _painted_answer(message, monkeypatch) as (pilot, answer, opened):
        cell = _cell(answer, "Tuesday")
        assert cell[1] == _cell(answer, BARE_URL)[1], "prose and link must share a row"
        await _click(pilot, answer, cell, opened)
    assert opened == []


@pytest.mark.asyncio
async def test_a_wrapped_url_opens_whole_from_either_row(monkeypatch) -> None:
    """A folded link is ONE link, on both of its rows.

    Rich splits the span at the fold and repeats the FULL target on each half
    (measured in this tree), so the second row must open the whole URL rather
    than the fragment it displays — the failure mode a fix on the clicked row
    alone would leave behind. Two clicks in one app is deliberate: the recorder
    is asserted after each, so a second click that silently stopped landing
    fails here instead of passing on the first one's evidence.
    """
    message = f"{WRAPPED_URL}\n"
    async with _painted_answer(message, monkeypatch) as (pilot, answer, opened):
        rows = [row.rstrip() for row in _painted_rows(answer)]
        # Located by SEARCH rather than by `startswith`, and clicked at the
        # URL's own column rather than at column 0: an assistant row carries a
        # gutter (`▎ `), so neither the head of the URL nor its continuation
        # begins its row. Asserting the layout here would make this test fail
        # whenever the gutter changes width, which is a fact about the spine
        # and not about whether a folded link opens whole.
        head = WRAPPED_URL[:24]
        first = next(i for i, row in enumerate(rows) if head in row)
        start = rows[first].index(head)
        # The two halves still reconstitute the whole URL, gutter removed.
        assert rows[first][start:] + rows[first + 1].lstrip("▎ ") == WRAPPED_URL, rows

        await _click(pilot, answer, (start, first), opened)
        assert opened == [WRAPPED_URL]
        tail_start = len(rows[first + 1]) - len(rows[first + 1].lstrip("▎ "))
        await _click(pilot, answer, (tail_start, first + 1), opened)
        assert opened == [WRAPPED_URL, WRAPPED_URL]

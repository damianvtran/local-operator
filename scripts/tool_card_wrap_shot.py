"""Capture the expanded tool-card body, collapsed and expanded, at any width.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/tool_card_wrap_shot.py OUT.svg [COLSxROWS] [CASE]

``CASE`` selects which card's body is painted, because the wrap budget this
script exists to justify has to be judged against both shapes at once:

    fail   (default)  a FAILED web_search — the card from the issue, whose
                      failure sentence is longer than the body at 80 and 100
                      columns. The whole sentence is the card's OWN prose, so it
                      is the shape a wrap budget must carry in full.
    stdout            a SUCCEEDING bash call whose captured output is long
                      LINES rather than many lines. This is the shape the crop
                      is load-bearing for: wrapping every block would let one
                      400-cell stdout line spend five rows, and 40 of them a
                      fifth of a screen. The same script proves the budget did
                      not move this case's row count.

Why a script rather than an assertion: the bug is a painted crop, and only a
frame shows WHERE the cut lands. The script prints the geometry too — the row
and column of each probe phrase in the built body, or ABSENT when the crop has
eaten it — because the still shows the symptom and the numbers show the cause
(AGENTS.md, "Visual validation"). Run the SAME script against a checkout that
predates the change to take the before frame; the fixtures below are composed
from the shipped builders, never re-typed, so both frames argue about the same
sentence.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Clear every multiplexer identifier BEFORE any application import: a headless
# pilot must not rename the operator's real workspace through inherited CMUX IDs.
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# A stable cwd for the capture. The status band paints ``os.getcwd()``, so a
# before frame taken from a checkout of the base revision and an after frame
# taken from this worktree would differ in that band and nowhere else — and a
# pair that differs in an unrelated row cannot be read as "the card changed".
# chdir rather than shorten the row: the band is not what this script is about.
_CAPTURE_CWD = "/tmp/lo-tool-card-wrap-shot"
os.makedirs(_CAPTURE_CWD, exist_ok=True)
os.chdir(_CAPTURE_CWD)

from scripts.visual_capture import (  # noqa: E402
    isolate_capture,
    save_capture,
    settle_status_line,
)

isolate_capture()

from rich.cells import cell_len  # noqa: E402
from rich.text import Text  # noqa: E402

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.tool_card import OUTPUT_INDENT, ToolCard  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: The phrases the geometry print locates in the body. The head is on the row
#: either way; the cause is the half the collapsed status cap can never carry,
#: and it is what the crop eats.
PROBES = (
    "Fetch a page directly",
    "keyed Sonar",
    "refused this search",
    "only provider tried",
)

#: One stdout line, 400 cells: a minified payload / a long log line, the shape a
#: blanket wrap would spend five rows on. Repeated to the expansion's line cap.
STDOUT_LINE = ("payload=" + '{"k":"v",' * 42 + '"end":true}')[:400]

STDOUT_LINES = 40


def _answer(text: str) -> AssistantBlock:
    """A SETTLED answer — an unsettled block keeps its live-turn ink."""
    block = AssistantBlock()
    block.update_text(text)
    block.finalize_text()
    return block


def search_failure_text() -> str:
    """The exact sentence ``WebSearchService`` raises for a walled search.

    Composed from the provider's own builder and the service's own prefix, so a
    change to the refusal's wording moves this frame rather than leaving the
    artifact arguing about a sentence the app no longer produces.
    """
    from local_operator.web_search.providers import _perplexity_authwall

    detail = _perplexity_authwall(
        {"upsell_information": {"name": "fraud_authwall_upsell", "upsell_type": "LOGIN"}}
    )
    assert detail is not None, "the authwall fixture stopped being a refusal"
    return f"Web search failed: {detail} ('perplexity' was the only provider tried)"


def _pin_clock(card: ToolCard) -> None:
    """Freeze the card's duration so a before/after pair differs only in its body.

    The settled duration is the card's own wall clock (``mark_failed`` reads
    ``_elapsed``), so the same card paints ``<0.1s`` on an idle machine and
    ``0.1s`` under load — a difference in a column this work does not touch, in
    a pair whose entire claim is "only the body changed".
    """
    card._duration = 0.4
    card._refresh_row()


def _seed(app: OperatorApp, case: str) -> ToolCard:
    """Put the card in a realistic turn so the body's indent is judged against
    the ledger spine rather than floating alone on an empty screen."""
    if case == "stdout":
        app._append_block(UserBlock("dump the raw payload for that webhook"))
        app._append_block(_answer("Fetching it now."))
        card = ToolCard("t1", "bash", {"command": "curl -s https://api.example.com/hook | jq -c ."})
        app._append_block(card)
        card.mark_done("\n".join([STDOUT_LINE] * STDOUT_LINES), None, measured_s=0.4)
        _pin_clock(card)
        app._append_block(_answer("That is the payload."))
        return card

    app._append_block(UserBlock("search the web for the current rate limit"))
    app._append_block(_answer("Searching."))
    card = ToolCard("t1", "web_search", {"query": "openai rate limits"})
    app._append_block(card)
    text = search_failure_text()
    # The APP's own settle call (``app.py``): the collapsed row carries the first
    # line, the expansion carries the whole result. Driven through it rather than
    # through ``mark_failed(error=text)`` so the fixture cannot be a shape the
    # app never builds.
    card.mark_failed(text.splitlines()[0], text, None, measured_s=0.4)
    _pin_clock(card)
    app._append_block(_answer("The search provider refused; here is why."))
    return card


def _body_probe(app: OperatorApp, card: ToolCard, width: int) -> None:
    """The geometry behind the still: row count and where each probe lands.

    Read off the widget's PAINTED renderable rather than a fresh
    ``_build_content``, so the columns reported are the ones the frame shows:
    the transcript hands the card its own lane (76 cells inside an 80-column
    frame), and measuring against the terminal width would report a row the
    card never paints.
    """
    renderable = card.renderable
    body = (renderable if isinstance(renderable, Text) else Text()).plain.splitlines()
    if not body:
        body = card._build_content(width).plain.splitlines()
    print(f"  body_rows={len(body)}", file=sys.stderr)
    for index, line in enumerate(body):
        print(f"    [{index:>2}] cells={cell_len(line):>3} {line!r}", file=sys.stderr)
    for probe in PROBES:
        found = [(index, line.index(probe)) for index, line in enumerate(body) if probe in line]
        # A probe that straddles a wrap boundary is PRESENT and on no single
        # row, so the joined reading is reported beside the per-row one: at 200
        # columns the reason's last two words land on the row after "only",
        # and a per-row-only probe would report the sentence cropped when it is
        # whole. Only the wrap can split them — nothing else joins these rows.
        joined = " ".join(line.strip() for line in body)
        print(
            f"  probe {probe!r}: rows={found or 'ABSENT'} "
            f"joined={'yes' if probe in joined else 'NO'}",
            file=sys.stderr,
        )


async def main() -> None:
    out = sys.argv[1]
    size = (80, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    case = sys.argv[3] if len(sys.argv) > 3 else "fail"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        card = _seed(app, case)
        await pilot.pause()
        # Both states, from the same card: the collapsed row is the cap the issue
        # says can never carry the cause, and the expanded body is where it has
        # to become reachable.
        await settle_status_line(pilot, app)
        await pilot.pause()
        print(f"collapsed at {size[0]} columns, case={case}", file=sys.stderr)
        _body_probe(app, card, size[0])
        save_capture(app, str(Path(out).with_name(Path(out).stem + "-collapsed.svg")))

        card.toggle_expanded()
        await pilot.pause()
        await pilot.pause()
        print(f"expanded at {size[0]} columns, case={case}", file=sys.stderr)
        screen = app.screen
        print(
            f"  size={screen.size} virtual={screen.virtual_size} "
            f"vscroll={screen.show_vertical_scrollbar} card_size={card.size} "
            f"indent={OUTPUT_INDENT}",
            file=sys.stderr,
        )
        _body_probe(app, card, size[0])
        save_capture(app, out)


asyncio.run(main())

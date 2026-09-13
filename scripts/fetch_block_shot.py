"""Capture the web_fetch tool card for a BLOCKED fetch (bot-protection refusal).

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/fetch_block_shot.py OUT.svg [COLSxROWS] [CASE]

``CASE`` selects which result the card is painted from, because the change to
the blocked body text has to be judged against the shapes it must NOT alter:

    blocked   (default)  a confirmed Akamai refusal — the case whose body text
                         changes (challenge markup replaced by the origin
                         statement, the reference id and the `browser` next
                         step). Carries the two-attempt/escalated shape, so the
                         ``Rendered:`` row's attempt summary is in the frame.
    missing              a 404, which keeps inlining its body verbatim: the
                         §5.2 narrowing applies to the ``blocked`` class only,
                         and a frame is what proves the other classes are
                         untouched
    stall                a TERMINAL failure (no response ever arrived), which
                         carries no render_method/final_url — the shape that
                         used to fall through to the generic output body
    ok                   a plain 200, the regression baseline

Why a script rather than an assertion: the card's structured rows are unchanged
by this work, but its painted BODY is not, and a passing test cannot show that
the replacement text wraps, aligns under the ledger spine, and stays inside the
danger treatment the error row establishes. The two frames (this tree vs. a
checkout predating the change, same script, same fixture) differ in the body
text and in nothing else.

The fixtures are REAL captures taken through the live path on 2026-09-12 —
the shoppersdrugmart.ca Akamai body with its ``Reference #`` id, and the
medium.com Cloudflare interstitial — not invented markup, so the frame shows
the number of rows an actual block costs.
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

from scripts.visual_capture import (  # noqa: E402
    isolate_capture,
    save_capture,
    settle_status_line,
)

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

BLOCKED_URL = "https://www.shoppersdrugmart.ca/?lang=en&query=power+bar"

#: The terminal-failure fixture's URL. A host that accepts the connection and
#: then never answers — the case whose card used to be indistinguishable from a
#: bash error (design review round 1, D5).
STALL_URL = "https://www.canadiantire.ca/en/pdp/some-product.html"

#: What the card painted BEFORE this change: the Akamai interstitial rendered as
#: if it were content, under the warning lead. Captured verbatim from the live
#: CLI on 2026-09-12 (`fetch test` against the URL above).
BEFORE_BLOCKED_TEXT = (
    "⚠ HTTP 403 Forbidden — this is an error/block page, not page content. "
    f"{BLOCKED_URL}\n"
    "markdownify · text/html · cache miss\n"
    "(The body below is the error response, not the requested page.)\n"
    "\n"
    "Access Denied\n"
    "\n"
    "# Access Denied\n"
    "\n"
    'You don\'t have permission to access "http://www.shoppersdrugmart.ca/?" on this server.\n'
    "\n"
    "Reference #18.4b182117.1789250183.345ea6ea\n"
    "\n"
    "https://errors.edgesuite.net/18.4b182117.1789250183.345ea6ea"
)

#: The 404 fixture. Its body is inlined before AND after, which is the property
#: the `missing` frame exists to hold: a page that explains itself still does.
MISSING_TEXT = (
    "⚠ HTTP 404 Not Found — this is an error/block page, not page content. "
    "https://docs.example.com/guide/old-page\n"
    "markdownify · text/html · cache miss\n"
    "(The body below is the error response, not the requested page.)\n"
    "\n"
    "# Page moved\n"
    "\n"
    "This guide moved to /guide/new-page in the 3.0 release. Update your bookmarks."
)

OK_TEXT = (
    "[200] https://example.com/\n"
    "text · text/html · cache miss\n"
    "\n"
    "# Example Domain\n"
    "\n"
    "This domain is for use in illustrative examples in documents."
)


def _answer(text: str) -> AssistantBlock:
    """A SETTLED answer — an unsettled block keeps its live-turn ink, which is
    not the state the operator reads a finished fetch in."""
    block = AssistantBlock()
    block.update_text(text)
    block.finalize_text()
    return block


def _blocked_details() -> dict[str, object]:
    """``details`` for the blocked card.

    Built by hand rather than imported so the SAME script runs against a
    checkout that predates the new keys — the additive keys are simply ignored
    by an older ``_fetch_result_output``, which is itself the §5.3 claim under
    test.
    """
    return {
        "url": BLOCKED_URL,
        "final_url": BLOCKED_URL,
        "status": 403,
        "content_type": "text/html",
        # NOT "markdownify": the body below the error row is OUR statement, not
        # a rendering of the origin's markup (``service.py`` sets this to the
        # rendered-content method, and the replacement body is plain text). The
        # card prints this field, so a wrong value here is a frame that lies.
        "render_method": "text",
        "cache": "miss",
        "bytes": 382,
        "lines": 9,
        "ok": False,
        "http_error": True,
        "attempts": 2,
        # The escalation took a refusal to a 200 on medium.com; on this origin it
        # was refused too. Either way the count and the identity happen to be the
        # facts a reader cannot get from the row without them (review round 1,
        # D1).
        "profiles": ["default", "browser"],
        "failure_kind": "blocked",
        "block_vendor": "akamai",
        "block_reference": "18.4b182117.1789250183.345ea6ea",
        "profile": "browser",
        "suggested_tool": "browser",
    }


def _stall_details() -> dict[str, object]:
    """``details`` for a TERMINAL stall: no response, so no render/final_url.

    This is the shape ``tool.py``'s ``except FetchError`` branch builds, and the
    one the card used to mistake for a bash row (design review round 1, D5).
    """
    return {
        "url": STALL_URL,
        "cache": "miss",
        "failure_kind": "stall",
        "attempts": 2,
        "profiles": ["default", "browser"],
        "suggested_tool": "browser",
    }


def _stall_text() -> str:
    """The terminal stall preview, composed by the engine's own ``describe``."""
    from local_operator.web_fetch.failure import FetchFailure, describe

    failure = FetchFailure(
        kind="stall",
        retryable=False,
        detail=(
            "read timed out after 20.0s — the origin accepted the connection but "
            "never sent a response"
        ),
    )
    return describe(failure, attempts=2, profiles=("default", "browser"), url=STALL_URL)


def _seed(app: OperatorApp, case: str) -> ToolCard:
    """Put the fetch card in a realistic turn: a question, an answer, and tool
    rows on both sides, so the body's indentation can be judged against the
    ledger spine rather than floating alone on an empty screen."""
    if case == "missing":
        app._append_block(UserBlock("read the old setup guide and tell me what changed"))
    else:
        app._append_block(UserBlock("what does a power bar cost at shoppers?"))
    app._append_block(_answer("Checking the product page directly."))

    if case == "missing":
        url = "https://docs.example.com/guide/old-page"
        text = MISSING_TEXT
        details: dict[str, object] = {
            "url": url,
            "final_url": url,
            "status": 404,
            "content_type": "text/html",
            "render_method": "markdownify",
            "cache": "miss",
            "bytes": 512,
            "lines": 3,
            "ok": False,
            "http_error": True,
            "attempts": 1,
            "failure_kind": "client",
        }
    elif case == "ok":
        url = "https://example.com/"
        text = OK_TEXT
        details = {
            "url": url,
            "final_url": url,
            "status": 200,
            "content_type": "text/html",
            "render_method": "text",
            "cache": "miss",
            "bytes": 1256,
            "lines": 4,
            "ok": True,
            "http_error": False,
            "attempts": 1,
        }
    elif case == "stall":
        url = STALL_URL
        text = _stall_text()
        details = _stall_details()
    else:
        url = BLOCKED_URL
        text = _blocked_after_text()
        details = _blocked_details()

    card = ToolCard("t1", "web_fetch", {"url": url})
    app._append_block(card)
    # Settle the way the APP settles it, which is the whole point of this
    # script: a non-2xx fetch returns ``is_error=True`` and the session settles
    # the card as an ERROR (``session_presentation.py`` → ``restore(state=
    # "error", error=_first_line(result_text))``). Capturing a 403 with
    # ``mark_done`` painted ``✓ <0.1s`` over a refusal — a frame that showed the
    # opposite of what the app does.
    if details.get("http_error") or details.get("failure_kind"):
        card.mark_failed(
            text.splitlines()[0] if text else "error",
            result_text=text,
            details=details,
            measured_s=0.4,
        )
    else:
        card.mark_done(text, details, measured_s=0.4)

    app._append_block(_answer("The origin refused the request; here is what it said."))
    return card


def _blocked_after_text() -> str:
    """The blocked preview as this tree builds it, or the pre-change capture.

    Composed from the engine's OWN ``_header_line`` and ``describe``, not
    re-typed here: the previous revision hand-wrote the lead and got it wrong in
    two ways a reader could see — it printed the ``(The body below is the error
    response…)`` note that the blocked class deliberately no longer emits, and it
    kept ``markdownify`` as the render method. A capture script whose fixture
    disagrees with the shipped code produces a frame that argues for a shape the
    app does not have. A checkout predating the change has no such module, and
    falls back to the verbatim BEFORE capture — which is what lets one script
    take both frames.
    """
    try:
        from local_operator.web_fetch.failure import FetchFailure, describe
        from local_operator.web_fetch.tool import _header_line
    except ImportError:
        return BEFORE_BLOCKED_TEXT
    failure = FetchFailure(
        kind="blocked",
        retryable=False,
        vendor="akamai",
        reference="18.4b182117.1789250183.345ea6ea",
        status=403,
    )
    details = _blocked_details()
    lead = _header_line(dict(details))
    return (
        lead
        + "\n\n"
        + describe(failure, attempts=2, profiles=("default", "browser"), url=BLOCKED_URL)
    )


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    case = sys.argv[3] if len(sys.argv) > 3 else "blocked"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        card = _seed(app, case)
        await pilot.pause()
        # The body is what changed, and it is only painted when the card is
        # open — a collapsed row shows the one-line summary either way.
        card.toggle_expanded()
        await pilot.pause()

        # A second settled frame: a first paint that differs from this one is a
        # reflow the user sees as motion (AGENTS.md, "Animation and multi-frame
        # changes"). The status band is waited on so two captures of the same
        # tree differ in the card and in nothing else.
        await pilot.pause()
        await settle_status_line(pilot, app)
        screen = app.screen
        print(
            f"size={screen.size} virtual={screen.virtual_size} "
            f"vscroll={screen.show_vertical_scrollbar} "
            f"card_size={card.size} "
            f"card_lines={len(card._build_content(size[0]).plain.splitlines())}",
            file=sys.stderr,
        )
        save_capture(app, out)


asyncio.run(main())

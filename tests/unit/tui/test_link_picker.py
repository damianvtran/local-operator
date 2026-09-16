"""``/links``: which URL reaches the browser, and what the card shows.

The command exists because the gesture users reach for cannot reach the app.
Textual claims the mouse at startup, and a terminal reporting mouse events to an
application does not run its own click-to-open — so the OSC-8 hyperlink the
transcript paints is correct on screen and unclickable (Ghostty's shift+click
bypass is the terminal's own and is documented as undetectable by the program).
These tests therefore drive the typed command, which is the route that works on
every terminal, and assert the two halves a plausible implementation gets wrong
in opposite directions:

* the URL that reaches the opener must be the WHOLE one, never the truncated
  string the card paints. A row that opens what it displays looks right and opens
  the wrong thing.
* the refusals must SPEAK. An openable-looking row that silently does nothing
  reads as a broken command, which is the whole defect being fixed.

The opener is a spy in every test — a browser launched from the suite would be a
side effect on the developer's machine — and the spy is installed on
``local_operator.mcp.auth``, the module the app imports it from.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.events import (
    AssistantDelta,
    AssistantMessageEnd,
    AssistantMessageStart,
)
from local_operator.tui.link_targets import LinkTarget, build_link_targets
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.link_picker import LinkPickerScreen
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView, UserBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: A message holding every shape the extractor must find: a markdown link, a
#: bare URL that Rich marks as nothing at all, and an explicit autolink.
ANSWER = (
    "Read [the docs](https://example.com/docs) or the raw one "
    "https://bare.test/x and the autolink <https://auto.test/y>.\n"
)


class _Opener:
    """A spy standing in for the browser, recording what it was handed."""

    def __init__(self, *, ok: bool = True) -> None:
        self.urls: list[str] = []
        self._ok = ok

    async def __call__(self, url: str) -> bool:
        self.urls.append(url)
        return self._ok


def _spy(opener: _Opener):
    return patch("local_operator.mcp.auth.open_browser_quietly", opener)


def _answer(text: str, *, finalized: bool = True) -> AssistantBlock:
    block = AssistantBlock()
    block.update_text(text)
    if finalized:
        block.finalize_text()
    return block


def _targets(*urls: str) -> list[LinkTarget]:
    return [LinkTarget(url=url, sender="agent", rank=index + 1) for index, url in enumerate(urls)]


async def _boot(pilot, app: OperatorApp) -> None:
    """Settle until the session exists, so ``/links`` is not answered by the
    no-session path while the test reads like it tested the real one."""
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _stream(pilot, app: OperatorApp, text: str) -> None:
    """Paint one agent message through the real event path."""
    app.post_message(AssistantMessageStart())
    await pilot.pause()
    app.post_message(AssistantDelta(text))
    await pilot.pause()
    app.post_message(AssistantMessageEnd(text))
    await pilot.pause()
    await pilot.pause()


async def _submit(pilot, app: OperatorApp, text: str) -> None:
    """Type a line into the real editor and press Enter."""
    editor = app.query_one(Editor)
    editor.text = text
    await pilot.pause()
    if editor._picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


def _body_region(screen: LinkPickerScreen):
    """The card body's screen region — the origin every row offset is measured
    from. Named because ``_body`` is typed optional and ``region`` is not."""
    body = screen._body
    assert body is not None
    return body.region


def _picker(app: OperatorApp) -> LinkPickerScreen | None:
    """The open picker, or ``None`` — never an ``isinstance`` assert at the
    call site, so a test expecting NO picker reads as plainly as one that does."""
    screen = app.screen
    return screen if isinstance(screen, LinkPickerScreen) else None


def _real_app():
    return OperatorApp(lambda: _factory(FakeSession()))


async def _open_picker(app: OperatorApp, targets: list[LinkTarget], pilot) -> LinkPickerScreen:
    """Push the card directly — for the geometry tests, which are about the card
    and not about which links the transcript yields."""
    screen = LinkPickerScreen(targets)
    app.push_screen(screen)
    await pilot.pause()
    await pilot.pause()
    return screen


# --- the card ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_card_lists_the_urls_in_order_under_a_title_and_footer() -> None:
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await _open_picker(app, _targets("https://a.test/1", "https://b.test/2"), pilot)
        lines = screen.render_lines_for_test()
        assert lines[0].startswith("Open a link")
        assert "https://a.test/1" in lines[2]
        assert "https://b.test/2" in lines[3]
        # The sibling cards' three clauses, in the sibling cards' order.
        assert lines[-1].startswith("↑↓ move · enter open · esc cancel"), lines[-1]


@pytest.mark.asyncio
async def test_arrows_wrap_because_the_card_is_an_overlay() -> None:
    """AGENTS.md's rule, and the reason is the one recorded there: a deliberate
    arrow press on a short list over a screen the user has not left is a shortcut
    to a row already visible. ``/move`` and ``/settings`` clamp because their list
    is the whole page."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await _open_picker(app, _targets("https://a.test/1", "https://b.test/2"), pilot)
        await pilot.press("up")
        assert screen._selected == 1, "up from the first row wraps to the last"
        await pilot.press("down")
        assert screen._selected == 0, "down from the last row wraps to the first"


@pytest.mark.asyncio
async def test_a_page_clamps_where_an_arrow_wraps() -> None:
    """A page is a scroll gesture: wrapping it reads as the list resetting."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        targets = _targets(*[f"https://a.test/{n}" for n in range(3)])
        screen = await _open_picker(app, targets, pilot)
        await pilot.press("pagedown")
        assert screen._selected == 2
        await pilot.press("pagedown")
        assert screen._selected == 2, "a second page at the end must not wrap"


@pytest.mark.asyncio
async def test_a_long_list_windows_and_says_so() -> None:
    """The card must not grow past its box: twelve rows plus chrome at a 14-row
    terminal is a card with its footer clipped off the bottom, which is the one
    thing it must never do."""
    app = _real_app()
    urls = [f"https://a.test/{n}" for n in range(20)]
    async with app.run_test(size=(80, 14)) as pilot:
        screen = await _open_picker(app, _targets(*urls), pilot)
        lines = screen.render_lines_for_test()
        assert len(lines) <= 14 - 2, f"card overflowed its box: {len(lines)} rows"
        assert lines[-1].startswith("↑↓"), "the footer must survive at the bottom"
        assert any(line.startswith("showing ") for line in lines)


@pytest.mark.asyncio
async def test_a_windowed_card_keeps_the_cursor_on_screen() -> None:
    app = _real_app()
    urls = [f"https://a.test/{n:02d}" for n in range(20)]
    async with app.run_test(size=(80, 14)) as pilot:
        screen = await _open_picker(app, _targets(*urls), pilot)
        for _ in range(19):
            await pilot.press("down")
        visible = screen.render_lines_for_test()
        assert any("https://a.test/19" in line and line.startswith("❯") for line in visible)


@pytest.mark.asyncio
async def test_a_terminal_too_small_draws_the_notice_instead_of_an_empty_card() -> None:
    """A dimmed screen with nothing on it reads as a crash; the notice says why
    and keeps ``esc`` on screen."""
    app = _real_app()
    async with app.run_test(size=(30, 8)) as pilot:
        targets = _targets(*[f"https://a.test/{n}" for n in range(5)])
        screen = await _open_picker(app, targets, pilot)
        assert screen.render_lines_for_test() == []
        notice = screen.query_one("#link-picker-too-small")
        assert notice.display is True


@pytest.mark.asyncio
async def test_escape_closes_the_card_without_opening_anything() -> None:
    opener = _Opener()
    app = _real_app()
    with _spy(opener):
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            await _stream(pilot, app, ANSWER)
            await _submit(pilot, app, "/links")
            assert _picker(app) is not None
            await pilot.press("escape")
            await pilot.pause()
            await pilot.pause()
            # Read INSIDE the pilot: a stopped app has no screen stack to ask.
            assert _picker(app) is None
    assert opener.urls == []


@pytest.mark.asyncio
async def test_a_click_opens_the_row_it_landed_on() -> None:
    """The gesture that DOES work under mouse capture.

    The terminal's own click-to-open is suppressed while the app reports mouse
    events, but the app receives the click — so a click on a ROW (not on the
    URL's cells, which would depend on where in the string the pointer landed)
    is a working route to the same outcome.
    """
    opener = _Opener()
    app = _real_app()
    with _spy(opener):
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            await _stream(pilot, app, ANSWER)
            await _submit(pilot, app, "/links")
            screen = _picker(app)
            assert screen is not None
            # `Pilot.click` without a selector takes SCREEN coordinates, so the
            # offset has to be the body's own origin plus the row — a bare (4, y)
            # lands on the backdrop to the card's left and hits nothing.
            region = _body_region(screen)
            await pilot.click(offset=(region.x + 4, region.y + 3))  # title, rule, then row 1
            await pilot.pause()
            await pilot.pause()
    assert opener.urls == ["https://bare.test/x"], opener.urls


# --- the command ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_picker_opens_on_the_newest_message_first_url() -> None:
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "old https://old.test/a")
        await _stream(pilot, app, ANSWER)
        await _submit(pilot, app, "/links")
        screen = _picker(app)
        assert screen is not None
        assert screen._selected == 0
        lines = screen.render_lines_for_test()
        assert "❯ https://example.com/docs" in lines[2]


@pytest.mark.asyncio
async def test_a_bare_url_is_openable_end_to_end() -> None:
    """The requirement that decides the whole extraction layer: a plain URL in
    prose has no Rich link span at all, so it can only be found in the source."""
    opener = _Opener()
    app = _real_app()
    with _spy(opener):
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            await _stream(pilot, app, "deployed at https://bare.test/x today")
            await _submit(pilot, app, "/links")
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
    assert opener.urls == ["https://bare.test/x"]


@pytest.mark.asyncio
async def test_the_whole_url_reaches_the_opener_even_when_the_row_is_cut() -> None:
    """A row that opens what it DISPLAYS looks right and opens the wrong thing."""
    long_url = "https://example.com/" + "a" * 120
    opener = _Opener()
    app = _real_app()
    with _spy(opener):
        async with app.run_test(size=(60, 20)) as pilot:
            await _boot(pilot, app)
            await _stream(pilot, app, f"the report is at {long_url} today")
            await _submit(pilot, app, "/links")
            screen = _picker(app)
            assert screen is not None
            painted = screen.render_lines_for_test()[2]
            assert long_url not in painted, "the row was not cut, so this proves nothing"
            assert long_url[:20] in painted, "the head of the url is what the row keeps"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
    assert opener.urls == [long_url]


@pytest.mark.asyncio
async def test_a_user_prompt_and_a_tool_notice_are_treated_differently() -> None:
    """A URL the user pasted is a link they were reading; one a tool card
    printed is not."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        transcript = app.query_one(TranscriptView)
        app._append_block(UserBlock("look at https://pasted.test/x", fold_width=80))
        app._append_block(NoticeBlock("curl https://tool.test/y failed", fold_width=80))
        await pilot.pause()
        targets = build_link_targets(transcript.blocks())
    assert [t.url for t in targets] == ["https://pasted.test/x"]


@pytest.mark.asyncio
async def test_no_links_says_so_rather_than_no_opping() -> None:
    """Two sentences for two states: a typed command that opens nothing and says
    nothing reads as broken."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "no addresses in here")
        await _submit(pilot, app, "/links")
        assert _picker(app) is None
        text = _notice_text(app)
    assert "no links" in text


@pytest.mark.asyncio
async def test_links_still_arriving_says_the_first_answer_is_coming() -> None:
    """The same two-case rule ``/copy`` records: telling a user to wait for
    something that is not coming, or to stop waiting for something that is,
    are different failures with different fixes."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(AssistantMessageStart())
        await pilot.pause()
        app.post_message(AssistantDelta("still writing https://half.test/x"))
        await pilot.pause()
        await _submit(pilot, app, "/links")
        assert _picker(app) is None
        text = _notice_text(app)
    assert "still coming" in text


@pytest.mark.asyncio
async def test_links_writes_no_user_row() -> None:
    """Nothing here reaches the model, so a row would be a keystroke log."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, ANSWER)
        await _submit(pilot, app, "/links")
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        blocks = app.query_one(TranscriptView).blocks()
    assert not any(isinstance(block, UserBlock) for block in blocks)


@pytest.mark.asyncio
async def test_the_opener_is_told_when_there_is_no_browser() -> None:
    """The one failure the user can work around, so the receipt names the URL."""
    opener = _Opener(ok=False)
    app = _real_app()
    with _spy(opener):
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            await _stream(pilot, app, ANSWER)
            await _submit(pilot, app, "/links")
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            text = _notice_text(app)
    assert opener.urls == ["https://example.com/docs"], "the opener was still tried"
    assert "no browser" in text
    assert "https://example.com/docs" in text


@pytest.mark.asyncio
async def test_a_non_http_target_never_reaches_the_opener() -> None:
    """The guard at the BOUNDARY, not only at extraction.

    ``_open_link`` is the single place in the app that hands a string to a
    browser, so it re-checks the scheme rather than trusting the list it was
    given — a route added later that does not go through the extractor must not
    be able to skip the rule.
    """
    opener = _Opener()
    said: list[tuple[str, str]] = []
    app = _real_app()
    with _spy(opener):
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            await app._open_link(
                LinkTarget(url="file:///etc/passwd", sender="agent", rank=1),
                lambda body, kind="info": said.append((body, kind)),
            )
            await pilot.pause()
    assert opener.urls == []
    assert any("only http and https" in text for text, _ in said), said


def _notice_text(app: OperatorApp) -> str:
    """Every notice row's text, joined — what the user reads after a command."""
    return "\n".join(
        block.text()
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    )

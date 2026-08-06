"""Welcome view — geometry, degradation, and lifecycle through a real pilot.

Two layers, because the widget has two independent failure modes:

- the PURE geometry (``build_welcome_lines``): what the view draws for a given
  box and a given set of facts. Tested without an app, so every width and
  height tier is exhaustive and instant.
- the WIRING: the view exists on boot, retires on the first transcript block,
  returns on ``/clear``, and contributes zero rows while hidden. Tested
  through ``App.run_test`` against the real stylesheet, because ``display``,
  ``1fr`` shares, and the clear hook only exist once the layout engine runs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from rich.cells import cell_len
from rich.text import Text

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import TranscriptView, UserBlock
from local_operator.tui.widgets.welcome import (
    HINT_KEY_WIDTH_TIGHT,
    HINTS,
    LOGO_FULL_MIN_WIDTH,
    LOGO_MARK,
    MARK_WIDTH,
    MODEL_PENDING,
    WelcomeInfo,
    WelcomeView,
    WORDMARK,
    WORDMARK_SPACED,
    build_welcome_lines,
)

TCSS = Path(__file__).parent.parent.parent.parent / "local_operator" / "tui" / "local_operator.tcss"
_HEX_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b")


def _lines(info: WelcomeInfo, width: int, height: int) -> list[str]:
    """Plain strings from the builder: geometry is what these tests read."""
    return [line.plain.rstrip("\n") for line in build_welcome_lines(info, width, height)]


def _has_mark(lines: list[str]) -> bool:
    """The mark is present when any row carries its signature top-left glyph."""
    return any(LOGO_MARK[0] in row for row in lines)


def _has_spaced_wordmark(lines: list[str]) -> bool:
    return any(WORDMARK_SPACED in row for row in lines)


def _has_plain_wordmark(lines: list[str]) -> bool:
    return any(WORDMARK in row for row in lines)


def _has_hints(lines: list[str]) -> bool:
    return any("command picker" in row for row in lines) or any(
        row.strip() in {"/", "/help", "ctrl+d"} for row in lines
    )


def _has_any_status(lines: list[str], info: WelcomeInfo) -> bool:
    candidates = [f"v{info.version}", info.model_label or MODEL_PENDING]
    return any(any(c in row for c in candidates) for row in lines)


def _info(missing: str | None = "openrouter") -> WelcomeInfo:
    return WelcomeInfo(
        version="0.15.10",
        model_label="openrouter/deepseek/deepseek-chat",
        cwd="/Users/damian/local-operator",
        missing_credential=missing,
    )


#: A box every section fits in; the degradation tests measure the SAME facts
#: at smaller boxes, so a section disappearing can only mean the box shrank.
ROOMY_W, ROOMY_H = 77, 23


# --- pure geometry: the lockup is exact -------------------------------------


def test_logo_lockup_widths_are_exact() -> None:
    """The mark and both wordmarks are the three fixed widths the lockup is
    drawn against. A single off-by-one cell would silently break every
    centering offset, so the widths are pinned here rather than implied."""
    assert all(cell_len(row) == MARK_WIDTH for row in LOGO_MARK)
    assert cell_len(WORDMARK) == MARK_WIDTH
    assert cell_len(WORDMARK_SPACED) == LOGO_FULL_MIN_WIDTH
    assert MARK_WIDTH == 14
    assert LOGO_FULL_MIN_WIDTH == 27


# --- pure geometry: width tiers ----------------------------------------------


def test_width_tier_full_lockup() -> None:
    """At 27 cells and up: the mark, a blank row, the letterspaced wordmark."""
    lines = _lines(_info(), ROOMY_W, ROOMY_H)
    assert _has_mark(lines)
    assert _has_spaced_wordmark(lines)


def test_width_tier_tight_lockup() -> None:
    """14-26 cells: the mark sits DIRECTLY over the plain wordmark — both are
    exactly 14 cells, so they lock flush and the separating blank row would
    only loosen a lockup that no longer has the room to be loose."""
    lines = _lines(_info(), 21, ROOMY_H)
    assert _has_mark(lines)
    assert _has_plain_wordmark(lines)
    assert not _has_spaced_wordmark(lines)


def test_width_tier_wordmark_only() -> None:
    """Below the mark's own width (14 cells) the plain wordmark alone remains.
    Every box in this tier is narrower than the wordmark itself, so it always
    truncates — what is pinned is that the mark is gone while no OTHER section
    is."""
    lines = _lines(_info(), 13, ROOMY_H)
    assert not _has_mark(lines)
    assert not _has_spaced_wordmark(lines)
    assert any(row.startswith(WORDMARK[:6]) for row in lines)  # truncated wordmark
    assert _has_hints(lines) and _has_any_status(lines, _info())

    narrow = _lines(_info(), 9, ROOMY_H)
    assert not _has_mark(narrow)
    assert _has_hints(narrow) and _has_any_status(narrow, _info())


def test_no_line_exceeds_the_box_width() -> None:
    """Every tier at a range of widths: nothing drawn is wider than the box.
    A row over the width would wrap or clip under the stylesheet's `no-wrap`
    assumption, so the builder must have already absorbed the fit."""
    info = _info()
    for width in (9, 13, 14, 21, 26, 27, 37, 77):
        lines = build_welcome_lines(info, width, ROOMY_H)
        for line in lines:
            assert cell_len(line.plain) <= width, f"{width}: {line.plain!r}"


def test_hint_descriptions_never_truncate_into_nonsense() -> None:
    """The narrow tiers keep the description intact or drop to keys only —
    `command pi…` is worse than `command picker` and worse than `/` alone."""
    tight_block = HINT_KEY_WIDTH_TIGHT + max(cell_len(desc) for _, desc in HINTS)
    for width in (tight_block, tight_block - 1, tight_block - 2):
        lines = _lines(_info(), width, ROOMY_H)
        for row in lines:
            assert "picker" not in row or "command picker" in row
        if width >= tight_block:
            assert _has_hints(lines)
        else:
            # Keys only: every affordance still named, no description at all.
            keys = {key for key, _ in HINTS}
            assert keys <= {row.strip() for row in lines}
            assert not any("command picker" in row or "all commands" in row for row in lines)


# --- pure geometry: height tiers ---------------------------------------------


def test_height_tiers_drop_logo_then_hints_then_status() -> None:
    """The degradation order, pinned at three heights of the SAME facts:

    1. a box with room: logo, status, hints — everything;
    2. a box missing one section: the LOGO is what goes, and nothing else;
    3. a box missing two sections: hints go next, status stays.

    Decoration sheds before teaching; teaching sheds before information.
    """
    info = _info()
    # The natural block height, measured the way the widget measures it: at a
    # box taller than the content, the builder top-pads, so the content extent
    # is the run from the first to the last drawn row.
    roomy = _lines(info, ROOMY_W, 99)
    drawn = [i for i, row in enumerate(roomy) if row.strip()]
    full_h = drawn[-1] - drawn[0] + 1
    # 6 logo (mark 4 + blank + wordmark) + 1 blank + 4 status + 1 blank + 3 hints
    assert full_h == 15

    # One section short (13 < 15): the LOGO is what goes, nothing else — the
    # hints and every status row are still on screen.
    mid = _lines(info, ROOMY_W, full_h - 2)
    assert not _has_mark(mid)
    assert _has_hints(mid)
    assert _has_any_status(mid, info)
    assert f"v{info.version}" in "\n".join(mid)

    # Two sections short (7): hints go next; the status block stays whole.
    short = _lines(info, ROOMY_W, 7)
    assert not _has_mark(short)
    assert not _has_hints(short)
    assert _has_any_status(short, info)
    assert f"v{info.version}" in "\n".join(short)


def test_status_rows_shed_lowest_priority_first_and_the_warning_never() -> None:
    """When even the status block alone is too tall, rows shed from the bottom
    of the priority stack — version, then cwd, then model — so the credential
    warning, the only row that changes what the user must DO, is the last one
    standing."""
    info = _info()

    three = build_welcome_lines(info, ROOMY_W, 3)
    assert len(three) == 3
    assert _warning_body(three) and info.model_label in plain(three)
    assert f"v{info.version}" not in plain(three)

    two = build_welcome_lines(info, ROOMY_W, 2)
    assert len(two) == 2
    assert _warning_body(two) and info.model_label in plain(two)

    one = build_welcome_lines(info, ROOMY_W, 1)
    assert len(one) == 1
    assert _warning_body(one)


def _warning_body(lines: list[Text]) -> bool:
    return any("not logged in" in line.plain for line in lines)


def plain(lines: list[Text]) -> str:
    return "\n".join(line.plain for line in lines)


def test_empty_box_renders_nothing() -> None:
    assert build_welcome_lines(_info(), 0, 10) == []
    assert build_welcome_lines(_info(), 10, 0) == []


# --- pilot wiring --------------------------------------------------------------


class FakeSession:
    """Satisfies SessionProtocol including the naming members StatusSlice
    added to the protocol; the welcome tests never drive naming itself."""

    def __init__(self, model_label: str = "openrouter/deepseek/deepseek-chat") -> None:
        self.prompts: list[str] = []
        self.aborts: list[str] = []
        self.disposed = False
        self.model_label = model_label
        self.conversation_name = ""
        self._handlers: list[Any] = []

    @property
    def session_id(self) -> str:
        return "sess"

    @property
    def agent_id(self) -> str:
        return "agent"

    @property
    def is_streaming(self) -> bool:
        return False

    def set_conversation_name(self, name: str) -> None:
        self.conversation_name = name

    async def complete_once(self, system: str, prompt: str) -> str:
        # Parameter names AND order must match SessionProtocol.complete_once:
        # this fake had them reversed, so the first test to drive naming through
        # it positionally would have swapped the system and user prompts and
        # still passed.
        return ""

    async def prompt(self, text: str, attachments: list[Any] | None = None) -> None:
        self.prompts.append(text)

    def steer(self, text: str) -> None:
        pass

    def abort(self, reason: str = "interrupted") -> None:
        self.aborts.append(reason)

    def subscribe(self, handler: Any) -> Any:
        self._handlers.append(handler)
        return lambda: None

    async def dispose(self) -> None:
        self.disposed = True


class FakeProviders:
    """A credential facade with a programmable answer per provider."""

    def __init__(self, missing: dict[str, bool]) -> None:
        self.missing = missing

    def has_any_credential(self, provider_id: str) -> bool:
        return not self.missing.get(provider_id, False)


class ExplodingProviders:
    """A credential store that blows up on every read: the welcome view must
    degrade to NO warning rather than take the app's first frame down."""

    def has_any_credential(self, provider_id: str) -> bool:
        raise RuntimeError("credential store unavailable")


def _make_app(
    session: FakeSession,
    providers: Any | None = None,
) -> OperatorApp:
    async def factory() -> FakeSession:
        return session

    return OperatorApp(lambda: factory(), provider_controller=providers)


def _welcome(app: OperatorApp) -> WelcomeView:
    return app.query_one(WelcomeView)


@pytest.mark.asyncio
async def test_visible_on_boot() -> None:
    """Fresh session: the welcome view is mounted, visible, and centered in the
    transcript region while the transcript holds no blocks."""
    app = _make_app(FakeSession())
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        welcome = _welcome(app)
        assert app.query_one(TranscriptView).blocks() == []
        assert welcome.display is True

        lines = [
            line.plain.rstrip()
            for line in build_welcome_lines(welcome._info, welcome.size.width, welcome.size.height)
        ]
        assert _has_mark(lines)
        assert _has_spaced_wordmark(lines)
        assert _has_hints(lines)

        # Centered: the mark's row carries the SAME left padding as every other
        # mark row, and the wordmark sits below it at the block's horizontal
        # center (the spaced wordmark is 27 cells in a 97-cell box).
        top_pad = next(i for i, row in enumerate(lines) if row.strip())
        assert top_pad > 0  # not jammed at the top
        mark_row = next(row for row in lines if LOGO_MARK[0] in row)
        left = cell_len(mark_row) - len(mark_row.lstrip(" "))
        assert left == (welcome.size.width - MARK_WIDTH) // 2


@pytest.mark.asyncio
async def test_hidden_after_the_first_transcript_block() -> None:
    """The first block lands and the welcome view retires — display goes none
    and its geometry stops occupying any row of the region."""
    session = FakeSession()
    app = _make_app(session)
    async with app.run_test(size=(80, 26)) as pilot:
        await pilot.pause()
        assert _welcome(app).display is True

        app._append_block(UserBlock("hello"))
        await pilot.pause()

        welcome = _welcome(app)
        assert welcome.display is False
        # Zero footprint is proven by the layout, not by a size field that the
        # engine may hold stale: the first real block sits at the top of the
        # region, exactly where it would sit with NO welcome widget present.
        transcript = app.query_one(TranscriptView)
        block = transcript.blocks()[0]
        assert block.region.y == transcript.content_region.y
        assert block.region.height == 1


@pytest.mark.asyncio
async def test_returns_after_clear() -> None:
    """``/clear`` empties the transcript, so the welcome view comes back —
    through the transcript's ``set_on_clear`` hook, not a second mechanism.
    The "history is untouched" receipt lands under it without overflowing."""
    app = _make_app(FakeSession())
    async with app.run_test(size=(80, 26)) as pilot:
        await pilot.pause()
        app._append_block(UserBlock("hello"))
        await pilot.pause()
        assert _welcome(app).display is False

        await pilot.press("slash", "c", "l", "e", "a", "r", "enter")
        await pilot.pause()

        welcome = _welcome(app)
        assert welcome.display is True
        assert app.query_one(TranscriptView).blocks()  # the receipt sits under it


@pytest.mark.asyncio
async def test_no_credential_warning_appears() -> None:
    """No stored credential for the active provider: the warning names the
    provider and the command that fixes it."""
    app = _make_app(FakeSession(), FakeProviders(missing={"openrouter": True}))
    async with app.run_test(size=(80, 26)) as pilot:
        await pilot.pause()
        welcome = _welcome(app)
        welcome._poll()  # the factory has resolved by now; force one read
        body = plain(build_welcome_lines(welcome._info, welcome.size.width, welcome.size.height))
        assert "! not logged in — /login openrouter" in body


@pytest.mark.asyncio
async def test_no_warning_when_credentials_are_configured() -> None:
    app = _make_app(FakeSession(), FakeProviders(missing={}))
    async with app.run_test(size=(80, 26)) as pilot:
        await pilot.pause()
        welcome = _welcome(app)
        welcome._poll()
        body = plain(build_welcome_lines(welcome._info, welcome.size.width, welcome.size.height))
        assert "not logged in" not in body
        assert "/login" not in body


@pytest.mark.asyncio
async def test_no_warning_when_there_is_no_provider_facade() -> None:
    """No controller (embedded TUI): the warning stays off — silence is the
    correct read when the answer is UNKNOWN, never a guess at logged-out."""
    app = _make_app(FakeSession(), None)
    async with app.run_test(size=(80, 26)) as pilot:
        await pilot.pause()
        welcome = _welcome(app)
        welcome._poll()
        assert welcome._info.missing_credential is None


@pytest.mark.asyncio
async def test_no_warning_when_credential_store_explodes() -> None:
    app = _make_app(FakeSession(), ExplodingProviders())
    async with app.run_test(size=(80, 26)) as pilot:
        await pilot.pause()
        welcome = _welcome(app)
        welcome._poll()
        assert welcome._info.missing_credential is None


@pytest.mark.asyncio
async def test_model_label_polls_in_after_boot() -> None:
    """The label lands via the session worker AFTER mount; the view must pick
    it up on its own timer rather than be pushed it."""
    import asyncio

    app = _make_app(FakeSession())
    async with app.run_test(size=(80, 26)) as pilot:
        await pilot.pause()
        # The timer's first tick (250 ms) is the thing under test: wait it out
        # rather than force the poll, so a broken timer fails this test.
        await asyncio.sleep(0.6)
        await pilot.pause()
        welcome = _welcome(app)
        assert welcome._info.model_label == "openrouter/deepseek/deepseek-chat"
        assert welcome._timer is None  # retired once the label arrived


# --- the stylesheet carries no literal hex in the welcome region ---------------


def test_stylesheet_region_has_no_literal_hex() -> None:
    """The whole sheet is hex-free (test_minimalism pins that); this pins the
    welcome region specifically so a regression is blamed on the rule it lives
    in, and so the region is asserted to use ONLY ``$lo-*`` tokens."""
    text = TCSS.read_text()
    marker = "/* ---- welcome view"
    region = text[text.index(marker) :]
    assert not _HEX_RE.search(region), f"literal hex in welcome region: {_HEX_RE.findall(region)}"
    colors = re.findall(r"(?:color|background|border\w*)\s*:\s*([^;]+);", region)
    assert colors, "the region declares no colors at all?"
    for value in colors:
        for part in value.split():
            assert part.startswith("$lo-"), f"non-token color in welcome region: {value!r}"

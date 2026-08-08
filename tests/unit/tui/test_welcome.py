"""Welcome view — geometry, degradation, and lifecycle through a real pilot.

Two layers, because the widget has two independent failure modes:

- the PURE geometry (``build_welcome_lines``): what the view draws for a given
  box and a given set of facts. Tested without an app, so every width and
  height tier is exhaustive and instant.
- the WIRING: the view exists on boot, retires on the first transcript block,
  returns on ``/clear``, and contributes zero rows while hidden. Tested
  through ``App.run_test`` against the real stylesheet, because ``display``,
  the measured content height, and the clear hook only exist once the layout
  engine runs. The LAYOUT that height feeds — the centred boot card and the
  class that selects it — is pinned in ``test_boot_layout.py``.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any

import pytest
from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.color import Color

from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import TranscriptView, UserBlock
from local_operator.tui.widgets.welcome import (
    HINT_KEY_WIDTH_TIGHT,
    HINTS,
    LOGO_FULL_MIN_WIDTH,
    LOGO_MARK,
    MARK_PULSE_DEPTH,
    MARK_PULSE_INTERVAL_S,
    MARK_PULSE_PERIOD_S,
    MARK_WIDTH,
    MODEL_PENDING,
    WORDMARK,
    WORDMARK_SPACED,
    WelcomeInfo,
    WelcomeView,
    build_welcome_lines,
    mark_pulse_color,
    mark_pulse_phase,
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
ROOMY_W, ROOMY_H = 77, 27


# --- pure geometry: the lockup is exact -------------------------------------


def test_logo_lockup_widths_are_exact() -> None:
    """The mark and both wordmarks are the fixed widths the lockup is drawn
    against. A single off-by-one cell would silently break every centering
    offset, so the widths are pinned here rather than implied.

    The mark is NOT the same width as the plain wordmark. The old hand-drawn
    monogram happened to be 14 cells like ``WORDMARK`` and a previous version of
    this test asserted that coincidence as though it were the lockup's contract.
    The real mark's aspect ratio fixes it at 15 for eight rows, and squeezing it
    to 14 would turn every ring into an ellipse.
    """
    assert all(cell_len(row) == MARK_WIDTH for row in LOGO_MARK)
    assert MARK_WIDTH == 22
    assert cell_len(WORDMARK) == 14
    assert cell_len(WORDMARK_SPACED) == LOGO_FULL_MIN_WIDTH
    assert LOGO_FULL_MIN_WIDTH == 27
    # Ten rows is what keeps all three of the mark's rings reading as
    # rings rather than quantising into squares.
    assert len(LOGO_MARK) == 10


# --- pure geometry: width tiers ----------------------------------------------


def test_width_tier_full_lockup() -> None:
    """At 27 cells and up: the mark, a blank row, the letterspaced wordmark."""
    lines = _lines(_info(), ROOMY_W, ROOMY_H)
    assert _has_mark(lines)
    assert _has_spaced_wordmark(lines)


def test_width_tier_tight_lockup() -> None:
    """MARK_WIDTH..26 cells: the mark sits DIRECTLY over the plain wordmark, no
    separating blank row — at this width the lockup no longer has the room to be
    loose."""
    lines = _lines(_info(), 24, ROOMY_H)
    assert _has_mark(lines)
    assert _has_plain_wordmark(lines)
    assert not _has_spaced_wordmark(lines)


def test_width_tier_wordmark_only() -> None:
    """Below the mark's own width the plain wordmark alone remains.
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


def test_height_tiers_shed_in_cost_to_the_user_order() -> None:
    """The degradation order, pinned at four heights of the SAME facts.

    The order is by what each step COSTS THE USER, which is deliberately NOT
    plain decoration-before-information:

    1. a box with room: everything, with the lockup's breathing row;
    2. one row short: the lockup goes FLUSH — one row of air is the cheapest
       thing on the screen;
    3. tighter: the VERSION row goes. It is the least actionable fact here, and
       spending it to keep the mark is a better trade than losing the product's
       identity on the one screen that exists to show it. At a 28-row terminal
       this single row is exactly the difference;
    4. tighter still: the mark goes, then the hints last, because the hints are
       a first-time user's way in.

    The credential warning survives all of it — see the test below.
    """
    info = _info()
    # The natural block height. The builder returns the block and nothing else —
    # no padding rows — so at a box taller than the content the row count IS the
    # content extent, and the widget reports exactly this as its height.
    roomy = _lines(info, ROOMY_W, 99)
    drawn = [i for i, row in enumerate(roomy) if row.strip()]
    full_h = drawn[-1] - drawn[0] + 1
    # 12 logo (mark 10 + blank + wordmark) + 1 blank + 4 status + 1 blank + 3 hints
    assert full_h == 21
    assert len(roomy) == full_h, "no padding rows: the block is all the builder draws"

    # A box of exactly the natural height keeps everything: the budget is what
    # the block may SPEND, and nothing is held back for a gap the input panel's
    # own top padding row already provides.
    exact = _lines(info, ROOMY_W, full_h)
    assert _has_mark(exact)
    assert _has_hints(exact)
    assert f"v{info.version}" in "\n".join(exact)
    assert len(exact) == full_h

    # One row short: the lockup goes flush — one row of air is the cheapest thing
    # on the screen, and every fact survives.
    flush = _lines(info, ROOMY_W, full_h - 1)
    assert _has_mark(flush)
    assert _has_hints(flush)
    assert f"v{info.version}" in "\n".join(flush)

    # One row tighter: the version row is spent to keep the mark.
    traded = _lines(info, ROOMY_W, full_h - 2)
    assert _has_mark(traded), "the mark is worth more than the version number"
    assert f"v{info.version}" not in "\n".join(traded)
    assert _has_hints(traded)

    # One row tighter again: there is nothing cheap left, so the mark goes.
    mid = _lines(info, ROOMY_W, full_h - 3)
    assert not _has_mark(mid)
    assert _has_hints(mid)
    assert _has_any_status(mid, info)

    # Far shorter: hints go LAST, after the weak status rows have already been
    # spent — at seven rows the version row buys them, and only a box too short
    # for the hint block plus a single status row finally drops them.
    keeps_hints = _lines(info, ROOMY_W, 7)
    assert not _has_mark(keeps_hints)
    assert _has_hints(keeps_hints)

    short = _lines(info, ROOMY_W, 6)
    assert not _has_mark(short)
    assert not _has_hints(short)
    assert _has_any_status(short, info)


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


# --- pure pulse: the mark breathes and nothing else moves ----------------------


def _mark_styles(lines: list[Text]) -> list[Style]:
    """The style each mark row is drawn in, one per glyph row."""
    out: list[Style] = []
    for line in lines:
        if not any(glyph in line.plain for glyph in ("█", "▄", "▀")):
            continue
        # The centring pad is a span of its own; the glyphs carry the tint.
        out.append(next(span.style for span in line.spans if isinstance(span.style, Style)))
    return out


def _row_styles(lines: list[Text]) -> list[list[Any]]:
    """Every row's spans, so a style-only difference is visible to a compare."""
    return [[(span.start, span.end, str(span.style)) for span in line.spans] for line in lines]


def _relative_luminance(color: Color) -> float:
    """WCAG relative luminance — what the eye reads, not what the hex says."""

    def channel(value: int) -> float:
        srgb = value / 255.0
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(color.r) + 0.7152 * channel(color.g) + 0.0722 * channel(color.b)


def _contrast(a: Color, b: Color) -> float:
    high, low = sorted((_relative_luminance(a), _relative_luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_the_pulse_rests_at_the_marks_own_dim() -> None:
    """Phase zero is the mark's historical tint, not a sample of the animation.

    This is what makes the still frame the OLD frame: with the animation gated
    off the view never overrides the colour at all, and the first tick of a
    running pulse starts from exactly the same place.
    """
    assert mark_pulse_phase(0.0) == 0.0
    assert mark_pulse_phase(MARK_PULSE_PERIOD_S) == pytest.approx(0.0, abs=1e-9)
    dim = theme_mod.semantic_color("dim")
    assert mark_pulse_color(0.0).lower() == dim.lower()


def test_the_pulse_swings_both_ways_and_stays_inside_its_ramp_neighbours() -> None:
    """Up towards ``muted``, down towards ``faint``, a quarter step each way.

    The bound is the point: an excursion that reached either neighbour outright
    would be the flat ``muted`` mark the lockup rejected, or a mark that fades
    to ``faint`` and reads as dropping out.
    """
    peak = mark_pulse_phase(MARK_PULSE_PERIOD_S / 4)
    trough = mark_pulse_phase(3 * MARK_PULSE_PERIOD_S / 4)
    assert peak == pytest.approx(1.0)
    assert trough == pytest.approx(-1.0)

    dim = Color.parse(theme_mod.semantic_color("dim"))
    muted = Color.parse(theme_mod.semantic_color("muted"))
    faint = Color.parse(theme_mod.semantic_color("faint"))
    assert mark_pulse_color(1.0) == dim.blend(muted, MARK_PULSE_DEPTH).hex
    assert mark_pulse_color(-1.0) == dim.blend(faint, MARK_PULSE_DEPTH).hex
    # Brighter at the peak, darker at the trough — a pulse, not a colour cycle.
    assert Color.parse(mark_pulse_color(1.0)).brightness > dim.brightness
    assert Color.parse(mark_pulse_color(-1.0)).brightness < dim.brightness

    # Breathing, not flashing — and this is the assertion that says so, because
    # the two above only restate the constant. The excursion is bounded as a
    # CONTRAST RATIO between its extremes: the full `dim`->`faint` swing that
    # was rejected measures 2.30:1 and reads as the logo dropping out, and
    # reaching both neighbours outright measures 4.37:1. The shipped amplitude
    # is 1.44:1, and the mark stays legible on the ground at either end.
    high, low = Color.parse(mark_pulse_color(1.0)), Color.parse(mark_pulse_color(-1.0))
    ground = Color.parse(theme_mod.semantic_color("bg"))
    assert _contrast(high, low) < 1.6
    assert _contrast(low, ground) > 3.0
    assert high.hex.lower() != muted.hex.lower()
    assert low.hex.lower() != faint.hex.lower()


def test_a_pulse_frame_moves_the_marks_style_and_nothing_else() -> None:
    """Two frames of the breath: identical text, identical row count, and the
    ONLY styles that differ are the mark's.

    Geometry is what the boot composition is measured from, so a pulse that
    could change a row count or a pad would move the splash on the card twelve
    times a second. Asserting the text is not enough — a style-only change is
    invisible to a plain-text dump, which is exactly why this compares spans.
    """
    peak = build_welcome_lines(_info(), ROOMY_W, ROOMY_H, mark_color=mark_pulse_color(1.0))
    trough = build_welcome_lines(_info(), ROOMY_W, ROOMY_H, mark_color=mark_pulse_color(-1.0))
    rest = build_welcome_lines(_info(), ROOMY_W, ROOMY_H)

    assert len(peak) == len(trough) == len(rest)
    assert plain(peak) == plain(trough) == plain(rest)
    assert len(_mark_styles(rest)) == len(LOGO_MARK)

    peak_rows, trough_rows = _row_styles(peak), _row_styles(trough)
    differing = [i for i in range(len(peak)) if peak_rows[i] != trough_rows[i]]
    glyph_rows = [i for i, line in enumerate(peak) if any(g in line.plain for g in ("█", "▄", "▀"))]
    # EVERY mark row moves and ONLY the mark rows move: one leaked span would
    # mean a row of the wordmark or the status stack breathing along with it.
    assert differing == glyph_rows == list(range(len(LOGO_MARK)))
    assert _mark_styles(peak) != _mark_styles(trough)


def test_the_pulse_cannot_change_the_blocks_height_at_any_size() -> None:
    """The degradation ladder is blind to the tint.

    Checked across the sizes where the ladder actually fires, because a pulse
    that shifted a height would do it at the threshold — the one row of budget
    that decides whether the mark is drawn at all.
    """
    for width in (20, MARK_WIDTH, LOGO_FULL_MIN_WIDTH, ROOMY_W):
        for height in range(1, ROOMY_H + 1):
            rest = build_welcome_lines(_info(), width, height)
            for phase in (-1.0, -0.5, 0.5, 1.0):
                frame = build_welcome_lines(
                    _info(), width, height, mark_color=mark_pulse_color(phase)
                )
                assert len(frame) == len(rest), f"{width}x{height} moved at phase {phase}"
                assert plain(frame) == plain(rest)


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

    @property
    def model(self) -> Any:
        return None

    def set_model(self, model: Any) -> None:
        pass

    @property
    def goal(self) -> str:
        return ""

    def set_goal(self, text: str) -> str:
        return text

    async def seed_history(self, messages: list[Any]) -> None:
        pass

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        self.conversation_name = text
        return text

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

    def set_approval_handler(self, handler: object | None) -> None:
        # The TUI installs its own approval gate on boot (the stdin gate
        # deadlocks under a full-screen app); fakes only need to accept it.
        self.approval_handler = handler

    def abort(self, reason: str = "interrupted") -> None:
        self.aborts.append(reason)

    def subscribe(self, handler: Any) -> Any:
        self._handlers.append(handler)
        return lambda: None

    async def dispose(self) -> None:
        self.disposed = True


class FakeProviders:
    """A credential facade with a programmable answer per provider.

    Answers `is_usable`, which is what the splash asks: an ENVIRONMENT key is a
    working credential, and the narrower stored-credential question told those
    users "not logged in" on the first screen.
    """

    def __init__(self, missing: dict[str, bool]) -> None:
        self.missing = missing

    def is_usable(self, provider_id: str) -> bool:
        return not self.missing.get(provider_id, False)


class ExplodingProviders:
    """A credential store that blows up on every read: the welcome view must
    degrade to NO warning rather than take the app's first frame down."""

    def is_usable(self, provider_id: str) -> bool:
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


def _frame(app: OperatorApp) -> list[tuple[str, list[tuple[str, str]]]]:
    """The composed frame as (row text, [(segment style, segment text)]).

    Styles and not just text, because the pulse is a STYLE-ONLY change: a dump
    of ``strip.text`` is byte-identical across every frame of the breath, so a
    text comparison would pass an animation that had stopped working.
    """
    return [
        (strip.text, [(str(segment.style), segment.text) for segment in strip._segments])
        for strip in app.screen._compositor.render_strips()
    ]


@pytest.mark.asyncio
async def test_visible_on_boot() -> None:
    """Fresh session: the welcome view is mounted, visible, content-sized, and
    resting on the input card while the transcript holds no blocks."""
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

        # CONTENT-SIZED, and that is the whole of the boot composition: the widget
        # is exactly as tall as the block it draws, so the rows the block does not
        # need stay above it and its last row is the row above the input card. A
        # widget taller than its block would put an arbitrary gap there and the
        # splash would read as floating over a bar instead of resting on a card.
        assert len(lines) == welcome.size.height
        assert welcome.region.bottom == app.query_one("#input-dock").region.y

        # Centered horizontally: the mark's first row sits at the block's
        # centring offset.
        #
        # Measured as leading SPACES, not as `cell_len(row) - len(lstrip)` — that
        # earlier form subtracted a stripped length from the row's full width and
        # only agreed with the offset while the mark had no indent of its own. The
        # real mark's first row begins with several spaces, so the expected leading
        # run is the centring pad PLUS the mark's own indent.
        mark_row = next(row for row in lines if LOGO_MARK[0] in row)
        own_indent = len(LOGO_MARK[0]) - len(LOGO_MARK[0].lstrip(" "))
        leading = len(mark_row) - len(mark_row.lstrip(" "))
        assert leading == (welcome.size.width - MARK_WIDTH) // 2 + own_indent


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


# --- pulse lifecycle: a timer that only exists while it can be seen ------------


async def _settled_welcome(pilot: Any) -> WelcomeView:
    """Pause until the splash has stopped changing for reasons of its OWN.

    The model label lands on the poll timer a fraction of a second into every
    boot and re-centres the whole block, so a frame captured before that is not
    a still frame and comparing two of them measures the poll, not the pulse.
    The poll timer retiring IS the settled edge.
    """
    welcome = pilot.app.query_one(WelcomeView)
    for _ in range(24):
        await pilot.pause()
        if welcome._timer is None:
            break
        await asyncio.sleep(WelcomeView.POLL_INTERVAL_S)
    await pilot.pause()
    return welcome


@pytest.fixture
def animation_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the suite-wide shimmer pin for the tests that need real motion.

    The autouse fixture sets ``LOCAL_OPERATOR_NO_SHIMMER`` so every other test
    reads a deterministic still frame; the pulse's own lifecycle can only be
    observed with the gate open.
    """
    monkeypatch.delenv("LOCAL_OPERATOR_NO_SHIMMER", raising=False)


@pytest.mark.asyncio
async def test_the_pulse_is_a_no_op_when_animation_is_disabled() -> None:
    """``LOCAL_OPERATOR_NO_SHIMMER`` (set by this suite's autouse fixture) buys
    a STILL frame, not a slow one: no timer is created, no colour is overridden,
    and the composed frame is byte-identical across a second of wall clock.

    Deterministic stills are a hard requirement here — the SVG goldens are
    captured from exactly this path — so "the pulse is paused" would not be
    good enough. Nothing may be scheduled at all.
    """
    app = _make_app(FakeSession())
    async with app.run_test(size=(100, 30)) as pilot:
        welcome = await _settled_welcome(pilot)
        assert welcome._pulse_timer is None
        assert welcome._mark_color is None

        before = _frame(app)
        await asyncio.sleep(1.0)
        await pilot.pause()
        assert _frame(app) == before
        assert welcome._pulse_timer is None


@pytest.mark.asyncio
async def test_the_pulse_runs_only_while_the_splash_is_on_screen(animation_on: None) -> None:
    """Visible: breathing. Hidden by the first transcript block: stopped, and
    back to rest. Returned by ``/clear``: breathing again, from rest.

    The stopped timer is checked through the Timer OBJECT rather than through
    the view's attribute: dropping the reference is not stopping it, and a
    Textual interval whose widget is gone keeps waking the event loop.
    """
    app = _make_app(FakeSession())
    async with app.run_test(size=(100, 30)) as pilot:
        welcome = await _settled_welcome(pilot)
        running = welcome._pulse_timer
        assert running is not None

        app._append_block(UserBlock("hello"))
        await pilot.pause()
        assert welcome.display is False
        assert welcome._pulse_timer is None
        assert running._task is None, "the timer was dereferenced but never stopped"
        assert welcome._mark_color is None, "a hidden splash kept the phase it paused at"

        app.query_one(TranscriptView).clear_blocks()
        await pilot.pause()
        assert welcome.display is True
        assert welcome._pulse_timer is not None
        assert welcome._pulse_timer is not running


@pytest.mark.asyncio
async def test_both_timers_stop_when_the_view_is_unmounted(animation_on: None) -> None:
    """Teardown with the splash still up — the commonest exit there is, since
    every boot that quits without a prompt takes it — leaves nothing running."""
    app = _make_app(FakeSession())
    async with app.run_test(size=(100, 30)) as pilot:
        welcome = await _settled_welcome(pilot)
        pulse, poll = welcome._pulse_timer, welcome._timer
        assert pulse is not None

        await welcome.remove()
        await pilot.pause()

        assert welcome._pulse_timer is None
        assert welcome._timer is None
        assert pulse._task is None
        assert poll is None or poll._task is None


@pytest.mark.asyncio
async def test_a_pulse_frame_repaints_the_mark_and_moves_no_row(animation_on: None) -> None:
    """The composed frame, not the builder: two phases of the breath restyle the
    mark's rows and leave every row's TEXT and the frame's height untouched.

    Sampled at the two extrema, where the sine is flat, so the assertion does
    not depend on how long the pilot's pause actually took. A text-only dump
    cannot see this change at all, which is the trap this test exists to avoid:
    the comparison is over rendered SEGMENT STYLES.
    """
    app = _make_app(FakeSession())
    async with app.run_test(size=(100, 30)) as pilot:
        welcome = await _settled_welcome(pilot)

        welcome._pulse_origin = time.monotonic() - MARK_PULSE_PERIOD_S / 4
        welcome._pulse_tick()
        await pilot.pause()
        peak = _frame(app)

        welcome._pulse_origin = time.monotonic() - 3 * MARK_PULSE_PERIOD_S / 4
        welcome._pulse_tick()
        await pilot.pause()
        trough = _frame(app)

        assert len(peak) == len(trough)
        assert [text for text, _ in peak] == [text for text, _ in trough]
        differing = [i for i in range(len(peak)) if peak[i][1] != trough[i][1]]
        glyphs = [i for i, (text, _) in enumerate(peak) if any(g in text for g in ("█", "▄", "▀"))]
        assert differing == glyphs
        assert len(glyphs) == len(LOGO_MARK), "a mark row went missing between frames"


@pytest.mark.asyncio
async def test_a_tick_never_re_measures_and_skips_the_colours_it_already_drew(
    animation_on: None,
) -> None:
    """Two properties of one tick, both about cost.

    A tick may NOT ask for layout: ``refresh(layout=True)`` re-runs the height
    degradation ladder, and doing that twelve times a second on a boot frame
    sitting one row from the threshold that drops the mark is a splash that
    twitches while the user reads it.

    And a tick that resolves to the colour already on screen must do nothing:
    the ramp quantises to about two dozen hexes across 40 ticks, so repainting
    the widget to draw identical bytes is the waste this cadence exists to avoid.
    """
    app = _make_app(FakeSession())
    async with app.run_test(size=(100, 30)) as pilot:
        welcome = await _settled_welcome(pilot)
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        original = welcome.refresh

        def recording_refresh(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        welcome.refresh = recording_refresh  # type: ignore[method-assign]

        # Land on the peak: a colour a long way from rest, so the tick repaints.
        welcome._pulse_origin = time.monotonic() - MARK_PULSE_PERIOD_S / 4
        welcome._pulse_tick()
        assert len(calls) == 1, "a moved colour did not repaint"
        assert not any(kwargs.get("layout") for _, kwargs in calls)

        # Immediately again: at 12.5 fps the sine cannot have left the flat top,
        # so the second tick must cost nothing.
        welcome._pulse_tick()
        assert len(calls) == 1, "an unchanged colour still repainted the widget"
        assert MARK_PULSE_INTERVAL_S <= 0.1, "the pulse is budgeted at 10-15 fps"
        assert 2.5 <= MARK_PULSE_PERIOD_S <= 4.0, "the breath left its slow band"


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

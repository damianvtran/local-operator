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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.color import Color

from local_operator.harness.types import AgentMessage, ImageContent
from local_operator.session.naming import ConversationName
from local_operator.session.protocol import CompactionOutcome
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import SLASH_COMMANDS, OperatorApp
from local_operator.tui.widgets.transcript import TranscriptView, UserBlock
from local_operator.tui.widgets.welcome import (
    HINT_KEY_WIDTH_TIGHT,
    HINTS,
    LOGO_FULL_MIN_WIDTH,
    LOGO_MARK,
    MARK_PULSE_DEPTH,
    MARK_PULSE_INTERVAL_S,
    MARK_PULSE_PERIOD_S,
    MARK_PULSE_SWELL_S,
    MARK_WIDTH,
    MODEL_PENDING,
    TIP_GLYPH,
    TIP_MIN_WIDTH,
    TIP_ROTATE_INTERVAL_S,
    TIPS,
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


def _has_tip(lines: list[str]) -> bool:
    """A tip is present when a row opens with the tip glyph.

    Matched on the PREFIX and not on a tip's text, so the same helper answers for
    a truncated tip as for a whole one. Nothing else on the splash opens with
    that glyph — the credential warning uses ``!``.
    """
    return any(row.lstrip().startswith(f"{TIP_GLYPH} ") for row in lines)


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


def test_height_tiers_only_add_sections_as_the_budget_grows() -> None:
    """The height ladder is monotonic, not a set of reversible concessions.

    The old "refund when the mark is dropped" special case made 80x24 show the
    version and tip, 80x25 lose both in exchange for the mark, and 80x27 regain
    only the tip. A ladder promises that more room means at least the content
    already visible. Pin that contract across EVERY height, then pin the order
    in which the optional sections first become affordable:

    keys → tip → version → wordmark → mark.
    """
    info = _info()
    roomy = _lines(info, ROOMY_W, 99)
    drawn = [index for index, row in enumerate(roomy) if row.strip()]
    full_height = drawn[-1] - drawn[0] + 1
    assert full_height == 23
    assert len(roomy) == full_height, "the block returns content, never canvas padding"

    def sections(height: int) -> frozenset[str]:
        lines = _lines(info, ROOMY_W, height)
        text = "\n".join(lines)
        visible: set[str] = set()
        if _has_hints(lines):
            visible.add("keys")
        if _has_tip(lines):
            visible.add("tip")
        if f"v{info.version}" in text:
            visible.add("version")
        if _has_spaced_wordmark(lines):
            visible.add("wordmark")
        if _has_mark(lines):
            visible.add("mark")
        return frozenset(visible)

    previous: frozenset[str] = frozenset()
    first_seen: dict[str, int] = {}
    for height in range(1, full_height + 1):
        current = sections(height)
        assert previous <= current, (height, previous - current, previous, current)
        for section in current - previous:
            first_seen[section] = height
        previous = current

    order = ("keys", "tip", "version", "wordmark", "mark")
    thresholds = [first_seen[name] for name in order]
    assert all(left < right for left, right in zip(thresholds, thresholds[1:]))
    assert previous == frozenset(order)

    # The last decorative concession is still one row of air inside the lockup:
    # exact height draws it; one row short keeps every CONTENT section and merely
    # flushes the mark against the wordmark.
    exact = _lines(info, ROOMY_W, full_height)
    flush = _lines(info, ROOMY_W, full_height - 1)
    assert sections(full_height) == sections(full_height - 1)
    assert len(exact) == len(flush) + 1


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


def test_a_harness_notice_sheds_before_the_credential_warning() -> None:
    """A quota fallback is news; a missing login is an action. When the
    box can hold only one of them, the action stays."""
    info = WelcomeInfo(
        version="0.15.10",
        model_label="anthropic/claude-opus-5",
        cwd="/tmp",
        missing_credential="anthropic",
        notice="anthropic quota low — falling back to zai/glm-5.3",
    )
    one = build_welcome_lines(info, ROOMY_W, 1)
    assert len(one) == 1
    assert _warning_body(one)
    assert "quota low" not in plain(one)

    two = build_welcome_lines(info, ROOMY_W, 2)
    assert "quota low" in plain(two)
    assert _warning_body(two)


def _warning_body(lines: list[Text]) -> bool:
    return any("not logged in" in line.plain for line in lines)


def plain(lines: list[Text]) -> str:
    return "\n".join(line.plain for line in lines)


def test_empty_box_renders_nothing() -> None:
    assert build_welcome_lines(_info(), 0, 10) == []
    assert build_welcome_lines(_info(), 10, 0) == []


# --- pure glow: the mark strobes and nothing else moves ------------------------


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


def test_the_glow_rests_at_the_marks_own_dim() -> None:
    """Phase zero is the mark's historical tint, not a sample of the animation.

    This is what makes the still frame the OLD frame: with the animation gated
    off the view never overrides the colour at all, and the first tick of a
    running glow starts from exactly the same place.
    """
    assert mark_pulse_phase(0.0) == 0.0
    assert mark_pulse_phase(MARK_PULSE_PERIOD_S) == pytest.approx(0.0, abs=1e-9)
    dim = theme_mod.semantic_color("dim")
    assert mark_pulse_color(0.0).lower() == dim.lower()


def test_the_glow_only_ever_adds_light_and_spends_most_of_its_cycle_at_rest() -> None:
    """A strobe, not a breath: never below rest, and mostly not running at all.

    Two separate claims, and both are what "subtle" means here. A cycle that
    dipped towards ``faint`` would read as the logo guttering rather than as
    light passing over it, and a cycle that was always moving would nag — the
    eye keeps returning to motion that never finishes.
    """
    # The swell is a raised cosine: rest at both ends, peak in the middle.
    assert mark_pulse_phase(MARK_PULSE_SWELL_S / 2) == pytest.approx(1.0)
    assert mark_pulse_phase(MARK_PULSE_SWELL_S) == pytest.approx(0.0, abs=1e-9)

    # Sampled across a whole cycle at the timer's own cadence: nothing negative,
    # and the majority of ticks are exactly rest.
    ticks = [
        mark_pulse_phase(i * MARK_PULSE_INTERVAL_S)
        for i in range(int(MARK_PULSE_PERIOD_S / MARK_PULSE_INTERVAL_S))
    ]
    assert min(ticks) == 0.0, "the glow went below the mark's resting dim"
    assert max(ticks) == pytest.approx(1.0)
    at_rest = sum(1 for level in ticks if level == 0.0)
    assert at_rest > len(ticks) / 2, "the mark is moving more often than it is still"
    assert MARK_PULSE_SWELL_S < MARK_PULSE_PERIOD_S / 2


def test_the_glow_peaks_inside_one_step_of_the_marks_resting_ramp_value() -> None:
    """Up towards ``muted`` and never as far as it.

    The bound is the point: an excursion that reached the neighbour outright
    would be the flat ``muted`` mark the lockup rejected for burying the
    wordmark it is supposed to sit behind.
    """
    dim = Color.parse(theme_mod.semantic_color("dim"))
    muted = Color.parse(theme_mod.semantic_color("muted"))
    assert mark_pulse_color(1.0) == dim.blend(muted, MARK_PULSE_DEPTH).hex
    assert Color.parse(mark_pulse_color(1.0)).brightness > dim.brightness
    assert mark_pulse_color(1.0).lower() != muted.hex.lower()

    # Subtlety as a CONTRAST RATIO, because the assertions above only restate
    # the constant. Rest sits at 4.55:1 on the ground and the peak at 5.57:1 —
    # 1.22:1 peak to rest, against 1.90:1 for the full step to `muted`. Anything
    # past 1.35:1 stops reading as a glow and starts reading as a second theme.
    peak = Color.parse(mark_pulse_color(1.0))
    ground = Color.parse(theme_mod.semantic_color("bg"))
    assert _contrast(peak, dim) < 1.35
    assert _contrast(peak, ground) > _contrast(dim, ground)
    assert _contrast(peak, ground) < _contrast(muted, ground)


def test_a_glow_frame_moves_the_marks_style_and_nothing_else() -> None:
    """Two frames of the glow: identical text, identical row count, and the
    ONLY styles that differ are the mark's.

    Geometry is what the boot composition is measured from, so a glow that
    could change a row count or a pad would move the splash on the card twelve
    times a second. Asserting the text is not enough — a style-only change is
    invisible to a plain-text dump, which is exactly why this compares spans.
    """
    peak = build_welcome_lines(_info(), ROOMY_W, ROOMY_H, mark_color=mark_pulse_color(1.0))
    mid = build_welcome_lines(_info(), ROOMY_W, ROOMY_H, mark_color=mark_pulse_color(0.5))
    rest = build_welcome_lines(_info(), ROOMY_W, ROOMY_H)

    assert len(peak) == len(mid) == len(rest)
    assert plain(peak) == plain(mid) == plain(rest)
    assert len(_mark_styles(rest)) == len(LOGO_MARK)

    peak_rows, mid_rows = _row_styles(peak), _row_styles(mid)
    differing = [i for i in range(len(peak)) if peak_rows[i] != mid_rows[i]]
    glyph_rows = [i for i, line in enumerate(peak) if any(g in line.plain for g in ("█", "▄", "▀"))]
    # EVERY mark row moves and ONLY the mark rows move: one leaked span would
    # mean a row of the wordmark or the status stack glowing along with it.
    assert differing == glyph_rows == list(range(len(LOGO_MARK)))
    assert _mark_styles(peak) != _mark_styles(mid)


def test_the_glow_cannot_change_the_blocks_height_at_any_size() -> None:
    """The degradation ladder is blind to the tint.

    Checked across the sizes where the ladder actually fires, because a glow
    that shifted a height would do it at the threshold — the one row of budget
    that decides whether the mark is drawn at all. Every phase the timer can
    actually produce is sampled, not just the extremes.
    """
    levels = [
        mark_pulse_phase(i * MARK_PULSE_INTERVAL_S)
        for i in range(int(MARK_PULSE_PERIOD_S / MARK_PULSE_INTERVAL_S))
    ]
    for width in (20, MARK_WIDTH, LOGO_FULL_MIN_WIDTH, ROOMY_W):
        for height in range(1, ROOMY_H + 1):
            rest = build_welcome_lines(_info(), width, height)
            for phase in levels:
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
        self._handlers: list[Any] = []
        self.asides: list[list[Any]] = []
        self.adopted: list[list[Any]] = []

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

    @property
    def effective_model(self) -> Any:
        return self.model

    @property
    def effective_model_label(self) -> str:
        return self.model_label

    def set_model(self, model: Any, *, explicit: bool = False) -> None:
        pass

    @property
    def goal(self) -> str:
        return ""

    def set_goal(self, text: str) -> str:
        return text

    async def seed_history(self, messages: list[Any]) -> None:
        pass

    @property
    def conversation_name(self) -> str:
        return self.conversation_name_state.text

    @property
    def conversation_name_state(self) -> ConversationName:
        # The real holder, created on first read: `user_set` precedence (a
        # human rename outranks every generated title, forever) is behaviour
        # the TUI reads before it spends a re-title call, so a fake that
        # reimplemented it as a bare string would hide a regression in it.
        state = getattr(self, "_name_state", None)
        if state is None:
            state = self._name_state = ConversationName()
        return state

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        return self.conversation_name_state.set(text, user_set=user_set)

    async def complete_once(self, system: str, prompt: str) -> str:
        # Parameter names AND order must match SessionProtocol.complete_once:
        # this fake had them reversed, so the first test to drive naming through
        # it positionally would have swapped the system and user prompts and
        # still passed.
        return ""

    def history(self) -> list[AgentMessage]:
        return []

    async def prompt(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        self.prompts.append(text)

    def steer(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        pass

    def set_approval_handler(self, handler: object | None) -> None:
        # The TUI installs its own approval gate on boot (the stdin gate
        # deadlocks under a full-screen app); fakes only need to accept it.
        self.approval_handler = handler

    def set_ask_handler(self, handler: object | None) -> None:
        # The TUI installs the `ask` tool's picker surface on boot, and that
        # install is what makes the tool exist; fakes only need to accept it.
        self.ask_handler = handler

    def abort(self, reason: str = "interrupted") -> None:
        self.aborts.append(reason)

    def cancel_subagents(self, reason: str = "interrupted") -> int:
        """No subagents in this fake; the protocol requires the method."""
        return 0

    def running_subagents(self) -> int:
        """No subagents in this fake; the protocol requires the method."""
        return 0

    def subscribe(self, handler: Any) -> Any:
        self._handlers.append(handler)
        return lambda: None

    async def dispose(self) -> None:
        self.disposed = True

    async def complete_aside(
        self,
        turns: list[Any],
        *,
        on_delta: Callable[[str], None] | None = None,
        on_usage: Callable[[Any], None] | None = None,
    ) -> str:
        # Recorded, not answered: the aside's no-trace contract is proven
        # against the real Session in tests/unit/session/test_aside.py. Here
        # the only thing that must hold is that the app can call it.
        self.asides.append(list(turns))
        return ""

    async def adopt_aside(self, messages: list[Any]) -> None:
        self.adopted.append(list(messages))

    async def compact_now(self) -> CompactionOutcome:
        # No history to compact: this fake never carries a conversation, which
        # is the state a real session answers with the same refusal.
        return CompactionOutcome(
            ran=False, reason="nothing_to_compact", detail="nothing to compact"
        )


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

    Styles and not just text, because the glow is a STYLE-ONLY change: a dump
    of ``strip.text`` is byte-identical across every frame of it, so a
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


def test_a_harness_notice_is_drawn_on_the_splash() -> None:
    """A quota fallback is a status row, same glyph as the login warning."""
    info = WelcomeInfo(
        version="0.15.10",
        model_label="anthropic/claude-opus-5",
        cwd="/tmp",
        notice="anthropic quota low — falling back to zai/glm-5.3",
    )
    body = plain(build_welcome_lines(info, ROOMY_W, ROOMY_H))
    assert "! anthropic quota low — falling back to zai/glm-5.3" in body


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


# --- glow lifecycle: a timer that only exists while it can be seen -------------


async def _settled_welcome(pilot: Any) -> WelcomeView:
    """Pause until the splash has stopped changing for reasons of its OWN.

    The model label lands on the poll timer a fraction of a second into every
    boot and re-centres the whole block, so a frame captured before that is not
    a still frame and comparing two of them measures the poll, not the glow.
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
    reads a deterministic still frame; the glow's own lifecycle can only be
    observed with the gate open.
    """
    monkeypatch.delenv("LOCAL_OPERATOR_NO_SHIMMER", raising=False)


@pytest.mark.asyncio
async def test_the_glow_is_a_no_op_when_animation_is_disabled() -> None:
    """``LOCAL_OPERATOR_NO_SHIMMER`` (set by this suite's autouse fixture) buys
    a STILL frame, not a slow one: no timer is created, no colour is overridden,
    and the composed frame is byte-identical across a second of wall clock.

    Deterministic stills are a hard requirement here — the SVG goldens are
    captured from exactly this path — so "the glow is paused" would not be
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
async def test_the_glow_runs_only_while_the_splash_is_on_screen(animation_on: None) -> None:
    """Visible: glowing. Hidden by the first transcript block: stopped, and
    back to rest. Returned by ``/clear``: glowing again, from rest.

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
async def test_a_glow_frame_repaints_the_mark_and_moves_no_row(animation_on: None) -> None:
    """The composed frame, not the builder: two phases of the glow restyle the
    mark's rows and leave every row's TEXT and the frame's height untouched.

    Sampled at the swell's peak and at its rest, both of which are flat, so the
    assertion does not depend on how long the pilot's pause actually took. A
    text-only dump cannot see this change at all, which is the trap this test
    exists to avoid: the comparison is over rendered SEGMENT STYLES.
    """
    app = _make_app(FakeSession())
    async with app.run_test(size=(100, 30)) as pilot:
        welcome = await _settled_welcome(pilot)

        welcome._pulse_origin = time.monotonic() - MARK_PULSE_SWELL_S / 2
        welcome._pulse_tick()
        await pilot.pause()
        peak = _frame(app)

        # Deep inside the hold, where the mark is at its resting `dim`.
        welcome._pulse_origin = time.monotonic() - (MARK_PULSE_PERIOD_S + MARK_PULSE_SWELL_S) / 2
        welcome._pulse_tick()
        await pilot.pause()
        resting = _frame(app)

        assert len(peak) == len(resting)
        assert [text for text, _ in peak] == [text for text, _ in resting]
        differing = [i for i in range(len(peak)) if peak[i][1] != resting[i][1]]
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
    two thirds of the cycle is held at rest and the swell quantises to ten
    hexes across twenty ticks, so repainting the widget to draw identical bytes
    is the waste this cadence exists to avoid.
    """
    app = _make_app(FakeSession())
    async with app.run_test(size=(100, 30)) as pilot:
        welcome = await _settled_welcome(pilot)
        # The glow's own interval timer is a CONCURRENT writer to `_mark_color`
        # (it fires every 80 ms while animation is on). Left running, a tick
        # landing between the origin-set and the manual tick below either
        # consumes the colour change (the manual tick then sees an unchanged
        # colour and repaints nothing) or adds a repaint of its own — under a
        # slow CI runner that turned this pin into a flake. The two properties
        # under test are about what ONE tick does, so stop the timer and let the
        # manual ticks be the only driver. Stopping also resets `_mark_color` to
        # None, which is the deterministic starting state the first tick needs.
        welcome._stop_pulse_timer()
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        original = welcome.refresh

        def recording_refresh(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        welcome.refresh = recording_refresh  # type: ignore[method-assign]

        # Land on the peak: a colour a long way from rest, so the tick repaints.
        welcome._pulse_origin = time.monotonic() - MARK_PULSE_SWELL_S / 2
        welcome._pulse_tick()
        assert len(calls) == 1, "a moved colour did not repaint"
        assert not any(kwargs.get("layout") for _, kwargs in calls)

        # Immediately again: at 12.5 fps the raised cosine cannot have left the
        # flat top, so the second tick must cost nothing.
        welcome._pulse_tick()
        assert len(calls) == 1, "an unchanged colour still repainted the widget"
        assert MARK_PULSE_INTERVAL_S <= 0.1, "the glow is budgeted at 10-15 fps"
        assert 1.0 <= MARK_PULSE_SWELL_S <= 2.0, "the swell left its slow band"
        assert MARK_PULSE_PERIOD_S >= 3 * MARK_PULSE_SWELL_S / 2, "the mark barely rests"


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


# --- the rotating tip ----------------------------------------------------------
#
# The load-bearing property is the ROW COUNT. This view is content-sized and the
# boot layout rests it on the input card, so a tip that could be one row for one
# entry and two (or none) for the next would shove the whole splash up and down
# the screen every TIP_ROTATE_INTERVAL_S. Every test here is ultimately about
# that, from either the pure or the composed side.


def _tip_rows(rows: list[str]) -> list[str]:
    """Every rendered row that opens with the tip glyph — expected to be one."""
    return [row for row in rows if row.lstrip().startswith(f"{TIP_GLYPH} ")]


def _tip_styles(lines: list[Text]) -> list[Style]:
    """The tip row's spans, glyph first then body."""
    line = next(line for line in lines if line.plain.lstrip().startswith(f"{TIP_GLYPH} "))
    return [span.style for span in line.spans if isinstance(span.style, Style)]


def test_a_tip_is_drawn_as_the_blocks_last_row() -> None:
    """One tip, glyph-prefixed, one blank row below the hints it sits under.

    Last on purpose: it is the only row here that teaches something the user did
    not ask about, so it reads after the facts and after the way in.
    """
    lines = _lines(_info(), ROOMY_W, ROOMY_H)
    rows = _tip_rows(lines)
    assert len(rows) == 1
    assert rows[0].strip() == f"{TIP_GLYPH} {TIPS[0]}"
    # One blank row joins it to the hints, the way every section of this block is
    # joined — and the hint table's last row is what sits above that blank.
    assert lines[-1] == rows[0]
    assert not lines[-2].strip()
    assert "ctrl+d" in lines[-3]


def test_the_tip_is_quieter_than_the_hints_it_sits_under() -> None:
    """Subordinate is measured as CONTRAST against the ground, not as a token
    name, so the assertion holds on either theme: the sentence sits below the
    hints' description ink, and the glyph below the sentence — a bullet the eye
    can skip over prose it can still read.
    """
    ground = Color.parse(theme_mod.semantic_color("bg"))
    hint_desc = Color.parse(theme_mod.semantic_color("muted"))
    glyph_style, body_style = _tip_styles(build_welcome_lines(_info(), ROOMY_W, ROOMY_H))
    assert glyph_style.color is not None and body_style.color is not None
    glyph = Color.parse(glyph_style.color.get_truecolor().hex)
    body = Color.parse(body_style.color.get_truecolor().hex)
    assert _contrast(body, ground) < _contrast(hint_desc, ground)
    assert _contrast(glyph, ground) < _contrast(body, ground)


def test_advancing_the_rotation_changes_the_tip_and_never_the_row_count() -> None:
    """THE constraint. Every tip, at every size, draws the same number of rows.

    Swept over the whole pool rather than sampled, because the failure this
    guards against is one long entry among twelve: the block is measured once and
    then repainted per rotation, so a single tip that needed a second row would
    make the splash jump every time the ring came round to it.
    """
    for width in (TIP_MIN_WIDTH, TIP_MIN_WIDTH + 8, 60, ROOMY_W):
        for height in range(1, ROOMY_H + 1):
            base = build_welcome_lines(_info(), width, height)
            for index in range(1, len(TIPS)):
                frame = build_welcome_lines(_info(), width, height, tip_index=index)
                assert len(frame) == len(base), f"{width}x{height} moved on tip {index}"

    # And the text really does turn over where there is room to read it: one row
    # per tip, a different sentence in each, and the pool walked exactly once.
    seen: list[str] = []
    for index in range(len(TIPS)):
        block = build_welcome_lines(_info(), ROOMY_W, 99, tip_index=index)
        rows = [line.plain.rstrip() for line in block]
        tip = _tip_rows(rows)
        assert len(tip) == 1
        seen.append(tip[0].strip())
    assert len(set(seen)) == len(TIPS)


def test_the_index_wraps_so_a_caller_can_keep_a_monotonic_counter() -> None:
    """The view advances an int forever; the pool is what makes it a ring."""
    base = build_welcome_lines(_info(), ROOMY_W, ROOMY_H)
    wrapped = build_welcome_lines(_info(), ROOMY_W, ROOMY_H, tip_index=len(TIPS))
    assert plain(wrapped) == plain(base)


def test_a_narrow_terminal_omits_the_tip_rather_than_wrapping_it() -> None:
    """Below the threshold there is no tip row at all; at it there is exactly
    one, truncated into the box.

    Omitted on WIDTH and never on the current tip's own length — that is what
    keeps the row count the same answer for all twelve, which the rotation
    depends on.
    """
    for width in range(1, TIP_MIN_WIDTH):
        assert not _tip_rows(_lines(_info(), width, 99)), f"a tip survived at {width} cells"
    for index in range(len(TIPS)):
        rows = [
            line.plain.rstrip()
            for line in build_welcome_lines(_info(), TIP_MIN_WIDTH, 99, tip_index=index)
        ]
        tip = _tip_rows(rows)
        assert len(tip) == 1, f"tip {index} did not fit in one row at the threshold"
        assert cell_len(tip[0]) <= TIP_MIN_WIDTH


def test_every_tip_names_something_this_build_answers() -> None:
    """A splash advertising a command the app rejects is worse than a blank row,
    and this is the one screen a first-run user reads word for word — so every
    leading ``/token`` is checked against the real command table."""
    known = {command.name for command in SLASH_COMMANDS}
    known |= {alias for command in SLASH_COMMANDS for alias in command.aliases}
    for tip in TIPS:
        for token in re.findall(r"/[a-z]+", tip):
            assert token[1:] in known, f"{tip!r} names a command that does not exist: {token}"


def test_the_pool_stays_a_readable_size_and_fits_a_60_column_terminal() -> None:
    """Bounds on the three numbers that make this feature pleasant or annoying.

    A pool much past a dozen dilutes the odds of ever meeting the entries that
    change how the app is used; duplicates waste a slot outright. 60 cells is the
    narrowest terminal a tip should survive whole in, and the interval has a band
    of its own (see ``TIP_ROTATE_INTERVAL_S``): under 8 s the row turns over
    while it is being read, over 15 s a short session only ever sees one.
    """
    assert 8 <= len(TIPS) <= 12
    assert len(set(TIPS)) == len(TIPS)
    assert max(cell_len(f"{TIP_GLYPH} {tip}") for tip in TIPS) <= 60
    assert 8.0 <= TIP_ROTATE_INTERVAL_S <= 15.0


@pytest.mark.asyncio
async def test_the_rotation_is_a_no_op_when_animation_is_disabled() -> None:
    """``LOCAL_OPERATOR_NO_SHIMMER`` (this suite's autouse fixture) holds the row
    at the FIRST tip with no timer scheduled at all.

    Same gate as the pulse, and for a sharper reason: a row of text on a clock
    would make every still frame a sample of whichever tip the wall clock
    happened to be holding, so the SVG goldens could never be regenerated twice.
    """
    app = _make_app(FakeSession())
    async with app.run_test(size=(100, 30)) as pilot:
        welcome = await _settled_welcome(pilot)
        assert welcome._tip_timer is None
        assert welcome._tip_index == 0

        before = _frame(app)
        shown = [row.strip() for row in _tip_rows([text for text, _ in before])]
        assert shown == [f"{TIP_GLYPH} {TIPS[0]}"]
        await asyncio.sleep(1.0)
        await pilot.pause()
        assert _frame(app) == before
        assert welcome._tip_timer is None
        assert welcome._tip_index == 0


@pytest.mark.asyncio
async def test_a_rotation_tick_turns_the_tip_over_and_moves_no_other_row(
    animation_on: None,
) -> None:
    """The composed frame, not the builder: one tick replaces the sentence and
    leaves the widget's height and every other row of the screen alone.

    Driven by calling the tick rather than by waiting out the interval — twelve
    seconds of wall clock in a unit test buys nothing the tick does not prove.
    """
    app = _make_app(FakeSession())
    async with app.run_test(size=(100, 30)) as pilot:
        welcome = await _settled_welcome(pilot)
        assert welcome._tip_timer is not None
        before = [text for text, _ in _frame(app)]
        before_tip = _tip_rows(before)
        assert len(before_tip) == 1
        height = welcome.size.height

        welcome._tip_tick()
        await pilot.pause()
        after = [text for text, _ in _frame(app)]
        after_tip = _tip_rows(after)

        assert len(after_tip) == 1
        assert after_tip != before_tip, "the tick did not turn the tip over"
        assert len(after) == len(before)
        assert welcome.size.height == height, "the splash was re-measured by a rotation"
        # Every row that is not the tip is byte-identical, which is the whole
        # claim: the block did not shift on the card.
        assert [row for row in before if row not in before_tip] == [
            row for row in after if row not in after_tip
        ]


@pytest.mark.asyncio
async def test_the_rotation_walks_the_whole_ring_and_never_repeats_itself(
    animation_on: None,
) -> None:
    """A tip that came up twice running reads as a broken app, and one the ring
    never reaches might as well not be written.

    The walk starts one tick in, because the FIRST entry of an appearance is
    pinned to ``TIPS[0]`` and the ring resumes at a drawn point after it — so the
    lap being checked here is the one the rotation actually runs, not the handoff
    into it.
    """
    app = _make_app(FakeSession())
    async with app.run_test(size=(100, 30)) as pilot:
        welcome = await _settled_welcome(pilot)
        assert welcome._tip_index == 0, "a launch opened on an arbitrary tip"
        welcome._tip_tick()
        assert welcome._tip_index != 0, "the handoff repainted the same sentence"

        seen = [welcome._tip_index]
        for _ in range(len(TIPS)):
            welcome._tip_tick()
            seen.append(welcome._tip_index)
        assert all(a != b for a, b in zip(seen, seen[1:])), "a tip repeated back to back"
        assert set(seen) == set(range(len(TIPS))), "the ring cannot reach every tip"
        assert seen[-1] == seen[0], "a full lap did not come back round"


@pytest.mark.asyncio
async def test_the_rotation_runs_only_while_the_splash_is_on_screen(animation_on: None) -> None:
    """Visible: rotating. Hidden by the first transcript block: stopped, and back
    to the first tip. Returned by ``/clear``: a new timer.

    Checked through the Timer OBJECT as well as the attribute, because dropping
    the reference is not stopping it — an interval whose widget is gone keeps
    waking the event loop.
    """
    app = _make_app(FakeSession())
    async with app.run_test(size=(100, 30)) as pilot:
        welcome = await _settled_welcome(pilot)
        running = welcome._tip_timer
        assert running is not None

        app._append_block(UserBlock("hello"))
        await pilot.pause()
        assert welcome.display is False
        assert welcome._tip_timer is None
        assert running._task is None, "the timer was dereferenced but never stopped"
        assert welcome._tip_index == 0, "a hidden splash kept the tip it paused on"

        app.query_one(TranscriptView).clear_blocks()
        await pilot.pause()
        assert welcome.display is True
        assert welcome._tip_timer is not None
        assert welcome._tip_timer is not running


@pytest.mark.asyncio
async def test_the_tip_timer_stops_when_the_view_is_unmounted(animation_on: None) -> None:
    """Teardown with the splash still up — every boot that quits without a prompt
    takes that exit — leaves no rotation running behind a screen that is gone."""
    app = _make_app(FakeSession())
    async with app.run_test(size=(100, 30)) as pilot:
        welcome = await _settled_welcome(pilot)
        rotation = welcome._tip_timer
        assert rotation is not None

        await welcome.remove()
        await pilot.pause()

        assert welcome._tip_timer is None
        assert rotation._task is None


# --- the defects the design round found in the tip ------------------------------


def test_no_width_the_row_is_drawn_at_ever_truncates_a_tip() -> None:
    """D11. The threshold admitted the row at 32 cells and then handed it to the
    shared ellipsis pass, so every terminal from 32 to 55 columns read half a
    sentence: `· /resume picks up a recent ses…`.

    Swept over the whole pool at every width from the threshold to 120, because
    the failure mode is one long entry among twelve — a pool that fits on average
    is a pool that truncates on rotation.
    """
    for width in range(TIP_MIN_WIDTH, 121):
        for index, tip in enumerate(TIPS):
            rows = [
                line.plain.rstrip()
                for line in build_welcome_lines(_info(), width, 99, tip_index=index)
            ]
            drawn = _tip_rows(rows)
            assert len(drawn) == 1, f"tip {index} is not one row at {width} cells"
            assert drawn[0].strip() == f"{TIP_GLYPH} {tip}", f"truncated at {width} cells"


def test_the_threshold_is_the_width_the_pool_actually_needs() -> None:
    """The constant is DERIVED, so rewording a tip cannot leave it stale — which
    is how it came to admit a row 24 cells narrower than its longest entry."""
    assert TIP_MIN_WIDTH == max(cell_len(f"{TIP_GLYPH} {tip}") for tip in TIPS)
    # One cell under it there is no row at all, rather than a fragment of one.
    assert not _tip_rows(_lines(_info(), TIP_MIN_WIDTH - 1, 99))


@pytest.mark.asyncio
async def test_a_24_row_terminal_still_gets_a_tip() -> None:
    """D16. 80x24 is the classic default and a common split pane, and it saw no
    tip at any width: the ladder spent the row on a budget that then went on to
    give up the mark's twelve rows anyway.

    Asserted through the REAL app rather than the builder, because the thing that
    was broken is the budget the view is handed, not the block it builds. Two
    settled frames, so what is asserted is the frame that stays.
    """
    for width in (80, 100, 120, 190):
        app = _make_app(FakeSession())
        async with app.run_test(size=(width, 24)) as pilot:
            await _settled_welcome(pilot)
            first = [text for text, _ in _frame(app)]
            await pilot.pause()
            second = [text for text, _ in _frame(app)]
        assert _tip_rows(first), f"no tip at {width}x24"
        assert first == second, f"the splash was still moving at {width}x24"


@pytest.mark.asyncio
async def test_a_fresh_view_opens_on_the_first_tip_and_only_then_varies(
    animation_on: None,
) -> None:
    """D12. Measured over twelve launches the old start was random ten of those
    times, so the first thing a first-run user read was an arbitrary entry —
    "compaction runs itself when the context window fills", before they have a
    context. The pool is ordered; the row now opens on the entry it is ordered
    for, and the draw moves to where the ring RESUMES."""
    starts = set()
    resumes = set()
    for _ in range(12):
        app = _make_app(FakeSession())
        async with app.run_test(size=(100, 30)) as pilot:
            welcome = await _settled_welcome(pilot)
            starts.add(welcome._tip_index)
            resumes.add(welcome._tip_resume)
            shown = [row.strip() for row in _tip_rows([text for text, _ in _frame(app)])]
            assert shown == [f"{TIP_GLYPH} {TIPS[0]}"]
    assert starts == {0}, "a launch opened on an arbitrary tip"
    # …and the variation did not simply go away with it: the resume point is
    # still drawn, and never zero, so the first tick always turns the row over.
    assert len(resumes) > 1, "the ring resumes at a fixed point, so the pool is unreachable"
    assert 0 not in resumes


def test_the_splash_names_the_model_the_way_the_band_does() -> None:
    """D10, restated for display names.

    The splash and the status band six rows below it must answer "which model" with
    the same string. They have already disagreed once — the splash printed
    `openrouter/deepseek/deepseek-…` while the band printed `deepseek-chat-v3.1` —
    and giving the band a display name while leaving the splash on the raw selector
    would be that defect again with a nicer-looking half.
    """
    from local_operator.tui.widgets.status_line import format_model_label

    info = WelcomeInfo(
        version="0.15.10",
        model_label="anthropic/claude-opus-5",
        model_name="Claude Opus 5",
        cwd="/Users/damian/local-operator",
    )
    rows = plain(build_welcome_lines(info, ROOMY_W, ROOMY_H))
    band_text = format_model_label(info.model_label, short=False, name=info.model_name)
    assert band_text == "Claude Opus 5"
    assert band_text in rows, rows
    assert info.model_label not in rows, rows


def test_the_splash_falls_back_to_the_selector_when_nothing_names_the_model() -> None:
    """A local Ollama tag has no name anywhere, and the string the operator typed
    beats an invented abbreviation of it — the same fallback the band takes."""
    info = WelcomeInfo(version="0.15.10", model_label="ollama/qwen3:32b", cwd="/tmp")
    rows = plain(build_welcome_lines(info, ROOMY_W, ROOMY_H))
    assert "ollama/qwen3:32b" in rows, rows

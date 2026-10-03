"""Wake surfaces: the transcript delivery receipt and the composer band panel.

A wake fires with no keystroke, which used to leave two gaps: the transcript
showed the agent starting to work with no record of WHY (no delivery line),
and the composer band had no standing answer to "does this session wake on its
own". These cover the two blocks that close them — ``WakeBlock`` (the
expandable receipt) and ``WakePanel`` (one row per schedule, hidden when
empty) — plus the session event that carries a live fire to the front end.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.text import Text
from textual.app import App
from textual.selection import Selection

from local_operator.harness.types import WakeDeliveredEvent
from local_operator.harness.wake import WakeSchedule
from local_operator.tui.glyphs import tool_icon
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.tool_card import (
    COLLAPSE_HINT,
    EXPAND_HINT,
    OUTPUT_INDENT,
    ROW_INDENT,
    ToolCard,
)
from local_operator.tui.widgets.transcript import (
    GAP_CLASS,
    ExpandableActionBlock,
    MonitorDeltaBlock,
    NoticeBlock,
    TranscriptView,
    WakeBlock,
)
from local_operator.tui.widgets.wake_panel import WakePanel
from tests.unit.tui.conftest import StyledTranscriptApp
from tests.unit.tui.test_tool_card import _ComposerApp

LIVE_TEXT = (
    "(alarm) Scheduled wake w3 (3/8, every 1h) — "
    'cancel with wake({op:"cancel",id:"w3"}) once its goal is met.\n\ncheck the build'
)
CATCHUP_TEXT = (
    "(alarm) The session resumed after being closed; the following scheduled wake(s) "
    "came due while it was down.\n\n- w1 (due 09:00): missed while the session was down.\n"
    "  Message: check the backup"
)


def _wake_text(block: WakeBlock) -> Text:
    """The applied Rich text, narrowed once for pyright and every assertion."""
    rendered = block.renderable
    assert isinstance(rendered, Text)
    return rendered


class TestWakeBlock:
    def test_collapsed_line_names_the_wake_and_hides_the_cancel_howto(self) -> None:
        block = WakeBlock(LIVE_TEXT)
        rendered = block._build_row(80).plain
        assert "w3" in rendered
        assert "wake" in rendered  # the name column, matching the tool ledger
        assert "check the build" not in rendered  # the body stays collapsed
        # The cancel instruction is for the model, not the user reading the line.
        assert "cancel with wake(" not in rendered
        # The envelope prefix is the model's framing; the card's fill already
        # says this is a delivery, so repeating "(alarm)" on the summary is
        # the dim single-line the user could not find.
        assert "(alarm)" not in rendered
        # At rest the expand hint is silent — the fill and the icon are the
        # affordance, the same contract as a settled tool row.
        assert EXPAND_HINT not in rendered
        assert "message" not in rendered

    def test_expand_reveals_the_full_message(self) -> None:
        block = WakeBlock(LIVE_TEXT)
        assert block.expanded is False
        assert block.toggle_expanded() is True
        expanded = block._build_content(80).plain
        assert "check the build" in expanded
        assert "w3" in expanded  # the summary row stays

    def test_catchup_bullet_wraps_with_a_hanging_indent(self) -> None:
        """A wrapped continuation hangs two cells deeper than the next
        bullet's dash, so at narrow widths a fold never reads as a new
        schedule line (design review round 1, D3)."""
        long_message = "report the state of the build and every failing test"
        text = (
            "(alarm) Scheduled wake w1 (1).\n\n"
            "- w1 (due 09:00): " + long_message + "\n"
            "- w2 (due 10:00): next"
        )
        block = WakeBlock(text, catchup=True)
        block.toggle_expanded()
        lines = block._build_content(40).plain.splitlines()
        bullet_rows = [i for i, line in enumerate(lines) if line.strip().startswith("- ")]
        assert len(bullet_rows) == 2
        continuation = lines[bullet_rows[0] + 1]
        assert not continuation.strip().startswith("- ")
        assert continuation.startswith(" " * 4)

    def test_catchup_line_marks_the_folded_misses(self) -> None:
        block = WakeBlock(CATCHUP_TEXT, catchup=True)
        rendered = block._build_row(80).plain
        assert "catch-up" in rendered
        assert "1 missed wake" in rendered
        assert "w1" in rendered
        # The model-facing "(alarm) The session resumed…" preamble must NOT
        # leak into the user-facing headline (review round 3, m3).
        assert "(alarm)" not in rendered
        block.toggle_expanded()
        expanded = block._build_content(80).plain
        assert "check the backup" in expanded
        assert "- w1" in expanded

    def test_catchup_line_labels_engine_armed_ids(self) -> None:
        """A folded Aida row reads as hers, not as the raw schedule id.

        The catch-up branch composes its own headline from the bullet ids, so
        it never went through ``wake_receipt_headline``'s label map: a fresh
        install's first receipt read ``catch-up — 1 missed wake
        (aida-greeting)`` while the live seat beside it already said ``Aida's
        introduction`` (UX review round 2, U5 / QA Q3). One shared lookup now
        covers both shapes; a user-made wake keeps naming itself.
        """
        text = (
            "(alarm) The session resumed after being closed; the following "
            "scheduled wake(s) came due while it was down.\n\n"
            "- aida-greeting (due 09:00): missed while the session was down.\n"
            "- w2 (due 10:00): next"
        )
        block = WakeBlock(text, catchup=True)
        rendered = block._build_row(120).plain
        assert "Aida's introduction" in rendered
        assert "aida-greeting" not in rendered
        assert "w2" in rendered

    def test_catchup_line_labels_follow_a_rename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D4 on the folded shape: both receipt shapes read the CONFIGURED name.

        The catch-up composition builds its own headline from the bullet ids,
        so it gets the same lookup as the live seat — a rename reaches both
        shapes or neither. The literal froze the packaged default: the receipt
        said "Aida's introduction" over a body signed with the new name (design
        review round 3, D4).
        """
        from tests.unit.aida.conftest import isolated_root_path, write_config

        root = isolated_root_path(tmp_path, monkeypatch)
        write_config(root, {"aida": {"name": "Sovereign"}})

        text = (
            "(alarm) The session resumed after being closed; the following "
            "scheduled wake(s) came due while it was down.\n\n"
            "- aida-greeting (due 09:00): missed while the session was down.\n"
            "- w2 (due 10:00): next"
        )
        block = WakeBlock(text, catchup=True)
        rendered = block._build_row(120).plain
        assert "Sovereign's introduction" in rendered
        assert "Aida's introduction" not in rendered
        assert "aida-greeting" not in rendered
        assert "w2" in rendered

    def test_activate_toggles_like_the_tool_ledger(self) -> None:
        """``activate`` returns True when it toggled, matching ToolCard —
        both expand and collapse are the row's one action, so both report
        success. The old return was the new expanded state, which made a
        collapse look like a no-op to any caller that checked the bool."""
        block = WakeBlock(LIVE_TEXT)
        assert block.activate() is True
        assert block.expanded is True
        assert block.activate() is True
        assert block.expanded is False

    def test_hint_appears_only_when_pointed_at_or_focused(self) -> None:
        """Same two-pointer contract as ToolCard: at rest the fill is the
        whole affordance; the hint lights under the pointer or the keyboard."""
        block = WakeBlock(LIVE_TEXT)
        assert EXPAND_HINT not in block._build_row(80).plain

        block._set_hovered(True)
        assert EXPAND_HINT in block._build_row(80).plain

        block._set_hovered(False)
        block._set_focused(True)
        assert EXPAND_HINT in block._build_row(80).plain

        block.toggle_expanded()
        row = block._build_row(80).plain
        assert COLLAPSE_HINT in row and EXPAND_HINT not in row

    def test_the_pointer_leaving_does_not_put_out_a_focused_rows_hint(self) -> None:
        block = WakeBlock(LIVE_TEXT)
        block._set_focused(True)
        block._set_hovered(True)
        block._set_hovered(False)
        assert EXPAND_HINT in block._build_row(80).plain

    def test_collapsed_card_is_one_row_and_expanded_is_taller(self) -> None:
        block = WakeBlock(LIVE_TEXT)
        assert block.spans_multiple_rows() is False
        block.toggle_expanded()
        assert block.spans_multiple_rows() is True


@pytest.mark.asyncio
async def test_real_pointer_hover_and_click_use_the_tool_trace_contract() -> None:
    """Exercise the actual Textual event path under the production sheet.

    Unit calls to ``_set_hovered`` prove the row builder; this proves the
    terminal receives a hand pointer, the real hover event reveals the hint,
    and a click grows/collapses the card without losing the one-row gap below.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 24)) as pilot:
        view = app.query_one(TranscriptView)
        wake = WakeBlock(LIVE_TEXT)
        below = NoticeBlock("trying another account", "warning")
        view.append_block(wake)
        view.append_block(below)
        await pilot.pause()
        await pilot.pause()

        assert wake.size.height == 1
        assert below.has_class(GAP_CLASS)
        assert below.region.y - wake.region.y == 2
        assert EXPAND_HINT not in _wake_text(wake).plain

        landed = await pilot.hover(wake)
        assert landed, "hover missed the wake card"
        await pilot.pause()
        assert app.screen._pointer_shape == "pointer"
        assert EXPAND_HINT in _wake_text(wake).plain

        await pilot.click(wake)
        await pilot.pause()
        assert wake.expanded is True
        assert wake.size.height == 2
        assert COLLAPSE_HINT in _wake_text(wake).plain
        assert below.region.y - wake.region.y == 3

        await pilot.click(wake)
        await pilot.pause()
        assert wake.expanded is False
        assert wake.size.height == 1
        assert below.region.y - wake.region.y == 2


def test_retheme_is_the_shared_finalized_re_entry_point() -> None:
    """A theme switch must repaint settled ledger rows in the new ink, and
    both row kinds must share the one sanctioned hook (agent review round 1,
    M1): the override lives on the base, not on each subclass."""
    assert WakeBlock.retheme is ExpandableActionBlock.retheme
    from local_operator.tui.widgets.tool_card import ToolCard

    assert ToolCard.retheme is ExpandableActionBlock.retheme

    block = WakeBlock(LIVE_TEXT)
    builds = {"n": 0}
    original = WakeBlock._refresh_row
    block._refresh_row = (  # type: ignore[assignment]
        lambda: (builds.__setitem__("n", builds["n"] + 1), original(block))[1]
    )
    block.retheme()
    assert builds["n"] == 1


def test_a_height_only_resize_rebuilds_nothing() -> None:
    """Same guard as ToolCard's (TUI-017): an expansion raises a Resize back
    into on_resize, and rebuilding an identical row was a third of all builds
    on a measured replay."""
    block = WakeBlock(LIVE_TEXT)
    width = block._built_width
    builds = {"n": 0}
    original = WakeBlock._refresh_row
    block._refresh_row = (  # type: ignore[assignment]
        lambda: (builds.__setitem__("n", builds["n"] + 1), original(block))[1]
    )
    block.on_resize(SimpleNamespace(size=SimpleNamespace(width=width)))
    assert builds["n"] == 0


@pytest.mark.asyncio
async def test_an_expanded_wake_copy_strips_the_icon_and_keeps_the_prompt() -> None:
    """The wake row is a ledger row for selection too: the icon is chrome and
    never reaches the clipboard, the summary row keeps its receipt, and the
    expansion's prompt rows copy verbatim (agent review round 1, m2)."""
    app = StyledTranscriptApp()
    async with app.run_test(size=(80, 24)) as pilot:
        view = app.query_one(TranscriptView)
        wake = WakeBlock(LIVE_TEXT)
        view.append_block(wake)
        await pilot.pause()
        wake.toggle_expanded()
        await pilot.pause()

        # The summary's gutter is the icon field PLUS the row's left inset:
        # the shared ledger spine, so it leaves with the copy for the same
        # reason the icon does.
        assert wake.copy_gutter(0) == ROW_INDENT + ToolCard.ICON_COLS
        assert wake.copy_gutter(1) == OUTPUT_INDENT

        app.screen.selections = {wake: Selection(None, None)}
        copied = app.screen.get_selected_text()
        assert copied is not None
        rows = copied.split("\n")
        assert rows[0].startswith("wake")
        assert tool_icon("wake") not in copied
        assert any("check the build" in row for row in rows)


@pytest.mark.asyncio
async def test_typing_on_a_focused_wake_reaches_the_composer_intact() -> None:
    """The passthrough is inherited, not copied — but the contract still needs
    a wake-row pin: a printable key on a focused wake must land in the
    composer, first character included (agent review round 1, m2)."""
    app = _ComposerApp()
    async with app.run_test(size=(90, 12)) as pilot:
        view = app.query_one(TranscriptView)
        wake = WakeBlock(LIVE_TEXT)
        view.append_block(wake)
        wake.focus()
        await pilot.pause()
        assert app.focused is wake

        await pilot.press(*"check")
        await pilot.pause()
        editor = app.query_one(Editor)
        assert editor.text == "check"
        assert app.focused is editor


@pytest.mark.asyncio
async def test_focused_wake_expands_and_collapses_on_enter_and_space() -> None:
    """The keyboard half of the affordance is not optional: a terminal may
    have no mouse reporting, and focus has to reveal what Enter will do."""
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 24)) as pilot:
        view = app.query_one(TranscriptView)
        wake = WakeBlock(LIVE_TEXT)
        view.append_block(wake)
        await pilot.pause()

        wake.focus()
        await pilot.pause()
        assert wake.has_focus
        assert EXPAND_HINT in _wake_text(wake).plain

        await pilot.press("enter")
        await pilot.pause()
        assert wake.expanded is True
        assert COLLAPSE_HINT in _wake_text(wake).plain

        await pilot.press("space")
        await pilot.pause()
        assert wake.expanded is False
        assert EXPAND_HINT in _wake_text(wake).plain


def _schedule(wake_id: str, message: str, every_ms: int | None = None, **kw) -> WakeSchedule:
    import time as _time

    now = int(_time.time() * 1000)
    return WakeSchedule(
        id=wake_id,
        message=message,
        next_due_at=now + 3_600_000,
        every_ms=every_ms,
        created_at=now,
        **kw,
    )


class _FakeScheduler:
    def __init__(self, schedules: list[WakeSchedule]) -> None:
        self.schedules = tuple(schedules)


class _FakeMonitorRuntime:
    def __init__(self, spec: SimpleNamespace, counters: dict[str, Any]) -> None:
        self.spec = spec
        self.counters = counters


class _FakeMonitorScheduler:
    """The slice ``WakePanel.sync`` reads: ``monitors`` + ``runtime(id)``."""

    def __init__(self, rows: list[tuple[SimpleNamespace, dict[str, Any]]]) -> None:
        self._rows = rows

    @property
    def monitors(self) -> tuple[SimpleNamespace, ...]:
        return tuple(spec for spec, _ in self._rows)

    def runtime(self, monitor_id: str) -> _FakeMonitorRuntime | None:
        for spec, counters in self._rows:
            if spec.id == monitor_id:
                return _FakeMonitorRuntime(spec, dict(counters))
        return None


def _monitor(
    monitor_id: str,
    name: str,
    *,
    every_ms: int = 60_000,
    due_in_ms: int | None = 30_000,
    disabled: bool = False,
    reason: str = "",
    failures: int = 0,
) -> tuple[SimpleNamespace, dict[str, Any]]:
    """(spec-ish, counters) — what the scheduler's two read surfaces return."""
    now = int(time.time() * 1000)
    spec = SimpleNamespace(id=monitor_id, name=name, tool="bash", every_ms=every_ms)
    counters: dict[str, Any] = {
        "next_due_at": None if due_in_ms is None else now + due_in_ms,
        "last_check_at": now - 1_000,
        "checks": 3,
        "consecutive_failures": failures,
        "disabled": disabled,
        "disabled_reason": reason,
    }
    return spec, counters


class _FakeSession:
    def __init__(
        self,
        schedules: list[WakeSchedule],
        monitors: list[tuple[SimpleNamespace, dict[str, Any]]] | None = None,
    ) -> None:
        self.wake_scheduler = _FakeScheduler(schedules)
        # ``None`` (not an empty scheduler) for the pre-monitor shape: a
        # session whose host has no monitor scheduler is a real production
        # state, and the panel must read it as "no rows".
        self.monitor_scheduler = _FakeMonitorScheduler(monitors) if monitors is not None else None


class _PanelHost(App[None]):
    """A minimal app that mounts only a WakePanel, so ``Static.update`` and the
    screen-geometry probes run against a live Textual app (they raise
    ``NoActiveAppError`` / ``NoScreen`` off-app, which is what the panel's
    guards are for)."""

    def compose(self):  # type: ignore[override]
        yield WakePanel()


async def _paint(
    schedules: list[WakeSchedule],
    monitors: list[tuple[SimpleNamespace, dict[str, Any]]] | None = None,
    *,
    size: tuple[int, int] = (100, 30),
) -> tuple[bool, str]:
    """Sync a mounted panel; return (was_displayed, painted_text).

    ``display`` is read INSIDE the running app and returned as a plain bool:
    it is a Textual reactive that app shutdown resets, so reading it on the
    panel after ``run_test`` exits reports False even for a panel that painted.
    """
    app = _PanelHost()
    async with app.run_test(size=size) as pilot:
        panel = app.query_one(WakePanel)
        panel.sync(_FakeSession(schedules, monitors))
        await pilot.pause()
        return bool(panel.display), str(panel._body.content)


class TestWakePanel:
    @pytest.mark.asyncio
    async def test_hidden_when_no_wakes(self) -> None:
        displayed, out = await _paint([])
        assert displayed is False
        assert out == ""

    @pytest.mark.asyncio
    async def test_one_row_per_schedule_not_per_occurrence(self) -> None:
        """A wake that fires every hour for a week is still ONE schedule, so
        it gets one line — the recurrence is stated once on it, never re-listed
        per trigger."""
        displayed, out = await _paint(
            [
                _schedule("w1", "check the backup", every_ms=3_600_000),
                _schedule("w2", "poll the queue"),
            ]
        )
        assert displayed is True
        assert "Wakes · 2 scheduled" in out
        assert "w1" in out and "every 1h" in out
        assert "w2" in out and "once" in out
        # Each schedule appears exactly once even though w1 recurs.
        assert out.count("w1") == 1

    @pytest.mark.asyncio
    async def test_message_snippet_is_shown(self) -> None:
        _, out = await _paint([_schedule("w1", "check the backup")])
        assert "check the backup" in out

    @pytest.mark.asyncio
    async def test_equality_guard_skips_identical_repaints(self) -> None:
        app = _PanelHost()
        async with app.run_test(size=(100, 30)):
            panel = app.query_one(WakePanel)
            session = _FakeSession([_schedule("w1", "x")])
            panel.sync(session)
            first = panel._shown
            panel.sync(session)
            assert panel._shown is first  # same object — no repaint happened

    @pytest.mark.asyncio
    async def test_survives_a_session_without_a_scheduler(self) -> None:
        app = _PanelHost()
        async with app.run_test(size=(100, 30)):
            panel = app.query_one(WakePanel)
            panel.sync(object())  # no wake_scheduler attribute
            assert panel.display is False

    @pytest.mark.asyncio
    async def test_overflow_marker_survives_a_short_screen(self) -> None:
        """On a screen short enough to floor the row budget, the "… N more"
        marker must still paint (review round 2, m1): silently dropping it
        leaves one visible wake beside a hidden count. The fix reserves the
        marker row even when that costs the last visible wake."""
        app = _PanelHost()
        async with app.run_test(size=(100, 12)):
            panel = app.query_one(WakePanel)
            session = _FakeSession([_schedule(f"w{i}", f"wake {i}") for i in range(1, 6)])
            panel.sync(session)
            out = str(panel._body.content)
            # Five schedules can never fit a floored budget; the marker must
            # report the hidden ones rather than vanish.
            assert "more wakes" in out


#: One material monitor delta as the session hands it over: the model-facing
#: envelope, the description line, the cancel hint, and the bounded diff.
MONITOR_TEXT = (
    "(monitor) 'watch the deploy queue' m1: 2 changes at 09:31 — check 12.\n"
    "Watching for: the deploy queue to drain\n"
    'Cancel with monitor({op:"cancel",id:"m1"}) once its goal is met.\n'
    "\n"
    "Diff vs the previous check:\n"
    "+ running 3 -> 5"
)


class TestMonitorBandStates:
    """Remediation round 1, design D1/D2/D3/D5/D8: what the band can say about a
    monitor in each state, at the widths people actually use."""

    @pytest.mark.asyncio
    async def test_a_stalled_monitor_shows_the_state_word_and_hides_a_stale_count(self) -> None:
        """D3: an unavailable episode outranks the failure count it froze."""
        now = int(time.time() * 1000)
        spec, counters = _monitor("m1", "ner-gpu-fleet-guard", failures=3)
        counters["unavailable_since"] = now - 42 * 60 * 1000

        _, out = await _paint([], [(spec, counters)])
        assert "stalled" in out
        assert "tool unavailable since" in out
        assert "3 failed" not in out

    @pytest.mark.asyncio
    async def test_an_idle_monitor_is_not_drawn_as_a_healthy_one(self) -> None:
        """D2: the state this change exists for was the one the band could not
        show — an overdue-by-hours row read exactly like a fresh one."""
        spec, counters = _monitor("m1", "watch the deploy queue", due_in_ms=-3_600_000)
        counters["last_check_at"] = int(time.time() * 1000) - 3_600_000

        _, out = await _paint([], [(spec, counters)])
        # The state word is the band's whole message for a dormant row: the
        # "overdue by 3h — session not open" tail is the CLI's, and carrying it
        # here cost the row its name at 100 columns (D1).
        assert "idle" in out.splitlines()[1]
        assert "watch the deploy queue" in out

    @pytest.mark.asyncio
    async def test_the_health_sentence_survives_a_narrow_band(self) -> None:
        """D1: at 60 columns the sentence after the name was the first thing
        truncation took, so the state word and the hint now precede it."""
        spec, counters = _monitor("m1", "watch the deploy queue")
        counters["checks"], counters["deliveries"] = 12, 0

        _, out = await _paint([], [(spec, counters)], size=(60, 20))
        assert "0 deliveries" in out.splitlines()[1]

    @pytest.mark.asyncio
    async def test_the_visible_two_rows_are_the_ones_that_need_reading(self) -> None:
        """D5: the cap is two rows, so order worst-first."""
        now = int(time.time() * 1000)
        healthy_spec, healthy = _monitor("m1", "watch the deploy queue", due_in_ms=30_000)
        healthy["deliveries"] = 2
        dead_spec, dead = _monitor("m2", "ner-gpu-fleet-guard", disabled=True, reason="boom")
        stalled_spec, stalled = _monitor("m3", "ner-gpu-fleet-dd")
        stalled["unavailable_since"] = now - 42 * 60 * 1000
        zero_spec, zero = _monitor("m4", "issue-tracker")
        zero["checks"], zero["deliveries"] = 12, 0

        _, out = await _paint(
            [],
            [
                (healthy_spec, healthy),
                (zero_spec, zero),
                (stalled_spec, stalled),
                (dead_spec, dead),
            ],
        )
        body = out.splitlines()[1:]
        assert "m2" in body[0]
        assert "m3" in body[1]
        assert "… 2 more monitors not shown" in out

    @pytest.mark.asyncio
    async def test_a_never_checked_watch_is_not_drawn_as_a_dormant_one(self) -> None:
        """D10: a watch that has never run a check is overdue BY DEFINITION, so
        an idle-first branch order rendered the one state this PR exists to
        surface as a plain dormant row. The pair below is the whole finding —
        same counters apart from the check count."""
        now = int(time.time() * 1000)
        spec, counters = _monitor("m1", "watch the deploy queue", due_in_ms=-7_200_000)
        counters["checks"], counters["deliveries"] = 0, 0
        spec.created_at = now - 7_200_000

        _, out = await _paint([], [(spec, counters)])
        assert "never checked" in out
        assert "idle" not in out

        counters["checks"] = 12
        _, out = await _paint([], [(spec, counters)])
        assert "idle" in out and "never checked" not in out

    @pytest.mark.asyncio
    async def test_a_muted_hint_does_not_repaint_the_due_clock(self) -> None:
        """D11: one ``tone`` was applied to both the label and the health
        clause, so the neutral 0-deliveries hint promoted the due clock from
        ``dim`` to ``muted`` — measured on the frames by sampling the clock
        cells, where a quiet row's boilerplate outranked a healthy row's."""
        from rich.style import Style

        from local_operator.tui import theme as theme_mod

        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        warning = Style(color=theme_mod.semantic_color("warning"))

        spec, counters = _monitor("m1", "watch the deploy queue")
        counters["checks"], counters["deliveries"] = 12, 0
        row = WakePanel._monitor_fingerprint(spec, counters)
        assert row[4] == "muted"
        # The label is the clock, not a state word.
        assert row[1] not in ("disabled", "stalled", "idle", "never checked")

        panel = WakePanel.__new__(WakePanel)
        text = WakePanel._monitor_section(
            panel, (row,), room=4, dim=dim, muted=muted, warning=warning
        )[1]

        def style_of(needle: str) -> Any:
            start = text.plain.index(needle)
            for span in text.spans:
                if span.start <= start and span.end >= start + len(needle):
                    return span.style
            return None

        assert style_of(row[1]) == dim  # the clock keeps the dim ink
        assert style_of("12 checks, 0 deliveries") == muted  # the hint keeps its own

    @pytest.mark.asyncio
    async def test_a_disabled_row_leads_with_a_cause_a_person_can_read(self) -> None:
        """U4: the health slot carried the tool's raw argument banner, so the
        one glance surface for a broken watch led with developer text."""
        spec, counters = _monitor("m1", "watch the error budget", disabled=True)
        counters["disabled_reason"] = "invalid arguments:\n- path: Extra inputs are not permitted"
        counters["consecutive_failures"] = 5

        _, out = await _paint([], [(spec, counters)])
        assert "5 consecutive failed checks" in out
        assert "invalid arguments" not in out

    @pytest.mark.asyncio
    async def test_a_disabled_row_keeps_a_reason_that_already_reads_as_a_sentence(self) -> None:
        """The other half of U4: only a raw banner is replaced — the
        informative reasons a person can act on stay verbatim."""
        spec, counters = _monitor("m1", "watch the queue", disabled=True)
        counters["disabled_reason"] = 'monitor can\'t watch "mcp__x": it is not in this tool set.'
        counters["consecutive_failures"] = 5

        _, out = await _paint([], [(spec, counters)])
        assert "it is not in this tool set." in out
        assert "consecutive failed checks" not in out

    @pytest.mark.asyncio
    async def test_the_header_never_says_watching_above_a_broken_row(self) -> None:
        """D7: "1 watching" above "disabled" contradicts the row itself."""
        spec, counters = _monitor("m1", "x", disabled=True, reason="boom")
        _, out = await _paint([], [(spec, counters)])
        assert "Monitors · 1 of 1 needs attention" in out
        assert "watching" not in out


class TestMonitorBand:
    """The monitor section of the same band (design §12)."""

    @pytest.mark.asyncio
    async def test_monitor_rows_join_the_same_band(self) -> None:
        displayed, out = await _paint([], [_monitor("m1", "watch the deploy queue")])
        assert displayed is True
        assert "Monitors · 1 monitor armed" in out
        assert "m1" in out and "watch the deploy queue" in out and "every 1m" in out

    @pytest.mark.asyncio
    async def test_wake_rows_come_first_then_the_monitor_section(self) -> None:
        _, out = await _paint(
            [_schedule("w1", "check the backup")], [_monitor("m1", "watch the queue")]
        )
        assert out.index("Wakes · 1 scheduled") < out.index("Monitors · 1 monitor armed")
        assert out.index("w1") < out.index("m1")

    @pytest.mark.asyncio
    async def test_hidden_when_both_are_empty(self) -> None:
        displayed, out = await _paint([], [])
        assert displayed is False
        assert out == ""

    @pytest.mark.asyncio
    async def test_a_disabled_monitor_shows_state_and_reason(self) -> None:
        _, out = await _paint(
            [],
            [
                _monitor(
                    "m2",
                    "watch the issue tracker",
                    due_in_ms=None,
                    disabled=True,
                    reason="the tool stopped being read-only after 5 failed checks",
                    failures=5,
                )
            ],
        )
        assert "disabled" in out
        assert "stopped being read-only" in out

    @pytest.mark.asyncio
    async def test_monitor_overflow_marker_counts_the_hidden(self) -> None:
        _, out = await _paint([], [_monitor(f"m{i}", f"watch {i}") for i in range(1, 5)])
        assert "more monitors" in out

    @pytest.mark.asyncio
    async def test_a_short_screen_keeps_the_wake_rows_whole(self) -> None:
        """At the floor budget the wake section wins whole: the monitor
        section is dropped rather than two half-sections each claiming rows
        they cannot show (the design's ordering, made explicit)."""
        app = _PanelHost()
        async with app.run_test(size=(100, 12)):
            panel = app.query_one(WakePanel)
            panel.sync(_FakeSession([_schedule("w1", "x")], [_monitor("m1", "y")]))
            out = str(panel._body.content)
            assert "Wakes · 1 scheduled" in out and "w1" in out
            assert "Monitors" not in out

    @pytest.mark.asyncio
    async def test_equality_guard_covers_monitors_too(self) -> None:
        app = _PanelHost()
        async with app.run_test(size=(100, 30)):
            panel = app.query_one(WakePanel)
            session = _FakeSession([], [_monitor("m1", "x")])
            panel.sync(session)
            first = panel._shown
            panel.sync(session)
            assert panel._shown is first  # same object — no repaint happened


class TestMonitorDeltaBlock:
    """The monitor receipt: the WakeBlock's twin (design §12)."""

    def test_collapsed_line_strips_the_model_facing_prefix(self) -> None:
        block = MonitorDeltaBlock(MONITOR_TEXT)
        # AT A FITTING WIDTH (round 1 review, F1): at width 80 the row
        # truncated before the envelope's cancel sentence, so the old pin
        # passed while the sentence was still in the string — it reached the
        # user at fold widths ≥160. 200 cells fits the whole collapsed line,
        # which makes "no cancel anywhere" a real assertion.
        rendered = block._build_row(200).plain
        assert "m1" in rendered
        assert "monitor" in rendered  # the name column
        assert "(monitor)" not in rendered
        assert "cancel" not in rendered.lower()
        assert "Watching for: the deploy queue to drain" in rendered
        assert "Diff vs" not in rendered  # the delta stays collapsed

    def test_expand_shows_the_envelope_and_the_bounded_delta(self) -> None:
        block = MonitorDeltaBlock(MONITOR_TEXT)
        assert block.expanded is False
        assert block.toggle_expanded() is True
        expanded = block._build_content(80).plain
        assert "Diff vs the previous check:" in expanded
        assert "+ running 3 -> 5" in expanded
        assert "m1" in expanded  # the summary row stays
        # The collapse's polish applies here too (design review round 1, D2):
        # the expansion's first line no longer keeps the model-facing prefix
        # the collapsed line strips — but the cancel HINT it is meant to
        # audit stays (only the prefix is polished).
        assert "(monitor)" not in expanded
        assert 'Cancel with monitor({op:"cancel",id:"m1"})' in expanded

    def test_the_body_strip_removes_only_the_prefix(self) -> None:
        """``monitor_receipt_body`` is a prefix strip, not a rewrite: the
        expansion keeps the cancel hint and the delta it is meant to audit."""
        from local_operator.harness.rows import monitor_receipt_body

        body = monitor_receipt_body(MONITOR_TEXT)
        assert body.startswith("'watch the deploy queue' m1:")
        assert 'Cancel with monitor({op:"cancel",id:"m1"})' in body
        assert "Diff vs the previous check:" in body
        # A text that never carried the prefix is returned unchanged.
        assert monitor_receipt_body("plain text") == "plain text"

    def test_a_delta_and_a_lifecycle_notice_are_told_apart(self) -> None:
        """D4: the frame drew a disabled watch and a delta identically — same
        glyph lane, same warm-grey ink — and the one notice with ``notify=True``
        was the least distinguished row on screen."""
        from local_operator.harness.rows import monitor_notice_kind
        from local_operator.monitors.delivery import (
            MonitorNotice,
            format_monitor_notice_text,
        )

        delta = MonitorDeltaBlock(MONITOR_TEXT)
        assert delta._summary_ink() == "dim"
        assert monitor_notice_kind(MONITOR_TEXT) is None

        disabled = MonitorDeltaBlock(
            format_monitor_notice_text(
                MonitorNotice(
                    monitor_id="m1",
                    name="ner-gpu-fleet-dd",
                    tool="mcp__datadog_search_datadog_hosts",
                    kind="disabled",
                    at_ms=1_756_000_000_000,
                    checks=7,
                    deliveries=0,
                    failures=5,
                    detail="boom",
                )
            )
        )
        assert disabled._summary_ink() == "warning"
        assert monitor_notice_kind(disabled._text) == "disabled"

        # Restored is good news and keeps the neutral ink — its distinction is
        # the headline, which is the notice's own first line.
        restored = MonitorDeltaBlock(
            format_monitor_notice_text(
                MonitorNotice(
                    monitor_id="m1",
                    name="w",
                    tool="bash",
                    kind="restored",
                    at_ms=1_756_000_000_000,
                    checks=7,
                    deliveries=1,
                )
            )
        )
        assert restored._summary_ink() == "dim"

    def test_a_notice_row_shows_the_news_and_not_the_whole_notice(self) -> None:
        from local_operator.monitors.delivery import (
            MonitorNotice,
            format_monitor_notice_text,
        )

        text = format_monitor_notice_text(
            MonitorNotice(
                monitor_id="m1",
                name="ner-gpu-fleet-dd",
                tool="mcp__datadog_search_datadog_hosts",
                kind="disabled",
                at_ms=1_756_000_000_000,
                checks=7,
                deliveries=0,
                failures=5,
                detail="boom",
            )
        )
        row = MonitorDeltaBlock(text)._build_row(200).plain
        assert "was DISABLED" in row
        assert "no longer watching" in row
        # The collapsed row was the ENTIRE 5-line notice (529 chars) with the
        # news cut off; the record stays in the expansion.
        assert "Last error" not in row
        assert "To restore" not in row

    def test_a_notice_card_tells_the_human_what_they_can_do(self) -> None:
        """U1: the notice's "what next" was the model's tool syntax. The card
        body now carries the same remedy in the reader's words, while the
        model-facing text keeps its own shape."""
        from local_operator.harness.rows import (
            MONITOR_HUMAN_REMEDY,
            monitor_receipt_body,
        )
        from local_operator.monitors.delivery import (
            MonitorNotice,
            format_monitor_notice_text,
        )

        text = format_monitor_notice_text(
            MonitorNotice(
                monitor_id="m2",
                name="ner-gpu-fleet-guard",
                tool="bash",
                kind="disabled",
                at_ms=1_756_000_000_000,
                checks=7,
                deliveries=0,
                failures=5,
                detail="boom",
            )
        )
        body = monitor_receipt_body(text)
        assert MONITOR_HUMAN_REMEDY in body
        assert "lop monitor cancel <session> <id>" in body
        # The wire text the model was handed is untouched: the sentence is added
        # by the card layer only.
        assert MONITOR_HUMAN_REMEDY not in text

    def test_the_collapsed_headline_survives_the_consequence(self) -> None:
        """R12 / QA Q1: the property U5+D13 fixed, pinned at the card's REAL
        widths (the cited test did not exist, and the nearest one ran at 200
        where the whole envelope fits and nothing can be cut).

        Boundaries are MEASURED for this name and clock, not hoped for: the
        whole phrase at 96 cells and up (the card's default box), ``no longer
        wat…`` at 80, and the news alone at 60. Literal survival at 60/80 is
        out of reach with this prefix — the point of the finding was which HALF
        survives a cut, and that is what these assertions pin.
        """
        from local_operator.monitors.delivery import (
            MonitorNotice,
            format_monitor_notice_text,
        )

        text = format_monitor_notice_text(
            MonitorNotice(
                monitor_id="m2",
                name="ner-gpu-fleet-dd",
                tool="mcp__datadog_search_datadog_hosts",
                kind="disabled",
                at_ms=1_756_000_000_000,
                checks=7,
                deliveries=0,
                failures=5,
                detail="boom",
            )
        )
        assert "was DISABLED" in MonitorDeltaBlock(text)._build_row(60).plain
        narrow = MonitorDeltaBlock(text)._build_row(80).plain
        assert "no longer wat" in narrow
        assert "no longer watching" not in narrow
        for width in (96, 110, 200):
            assert "no longer watching" in MonitorDeltaBlock(text)._build_row(width).plain, width

    def test_the_row_wears_the_monitors_own_name_and_glyph(self) -> None:
        """The name column says ``monitor`` and the glyph table knows it — not
        the generic fallback, which would read as an unknown tool."""
        from local_operator.tui.glyphs import NERD_TOOL_ICONS, PLAIN_TOOL_ICONS

        assert MonitorDeltaBlock.tool_name == "monitor"
        assert "monitor" in NERD_TOOL_ICONS and "monitor" in PLAIN_TOOL_ICONS


class TestWakeDeliveredEvent:
    def test_live_fire_and_catchup_are_distinguished(self) -> None:
        live = WakeDeliveredEvent(text=LIVE_TEXT, catchup=False)
        catchup = WakeDeliveredEvent(text=CATCHUP_TEXT, catchup=True)
        assert live.type == "wake_delivered" and live.catchup is False
        assert catchup.catchup is True

    def test_carries_the_full_text_for_the_expansion(self) -> None:
        event = WakeDeliveredEvent(text=LIVE_TEXT)
        assert "check the build" in event.text


@pytest.mark.asyncio
async def test_live_wake_fire_emits_a_delivery_receipt(tmp_path) -> None:
    """The session emits wake_delivered BEFORE spawning the turn, so the front
    end can paint the expandable line ahead of the work the wake triggered."""
    import time as _time

    from local_operator.harness.types import StreamEndEvent, StreamTextDelta
    from local_operator.harness.wake import DueWake, WakeSchedule
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from tests.unit.session.test_session import MODEL, ScriptedStream

    stream = ScriptedStream([[StreamTextDelta(delta="ack"), StreamEndEvent(stop_reason="stop")]])
    session = Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
    )
    events = []
    session.subscribe(events.append)

    schedule = WakeSchedule(
        id="w1", message="wake up", next_due_at=int(_time.time() * 1000), created_at=0
    )
    due = DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)
    await session._deliver_wake(due)

    receipts = [e for e in events if getattr(e, "type", None) == "wake_delivered"]
    assert len(receipts) == 1
    assert receipts[0].catchup is False
    assert "wake up" in receipts[0].text
    await session.dispose()


class TestWakePanelOwedState:
    """The panel is where the incident's operator was looking (design D1).

    Before this, an owed wake, a wake owed after five failed attempts and a
    healthy one painted as three indistinguishable dim rows: nothing under
    ``local_operator/tui/`` read the supervisor's ledger, so the one surface
    that names a session's wakes could not say a wake was not firing.
    """

    @staticmethod
    def _session(schedules: list[WakeSchedule]) -> _FakeSession:
        session = _FakeSession(schedules)
        session.session_id = "sess"  # type: ignore[attr-defined]
        return session

    async def _paint_owed(
        self, tmp_path: Path, schedules: list[WakeSchedule], *, state: str = "retrying"
    ) -> str:
        import json
        import time as _time

        from local_operator.wakes import deliveries

        now = int(_time.time() * 1000)
        due = schedules[0].next_due_at
        record = {
            "schema": 1,
            "session_id": "sess",
            "occurrence_ms": due,
            "state": state,
            "attempts": 1 if state == "retrying" else 5,
            "first_attempt_ms": now - 9 * 86_400_000,
            "last_attempt_ms": now - 60_000,
            "next_attempt_ms": now + 240_000,
            "last_error": "could not reach a runtime for session sess within 180s",
        }
        directory = deliveries.deliveries_dir(tmp_path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "sess.json").write_text(json.dumps(record), encoding="utf-8")

        app = _PanelHost()
        async with app.run_test(size=(100, 30)) as pilot:
            panel = app.query_one(WakePanel)
            panel.sync(self._session(schedules))
            await pilot.pause()
            return str(panel._body.content)

    @pytest.mark.parametrize("state", ["retrying", "undelivered"])
    @pytest.mark.parametrize(
        ("age_ms", "labels"),
        [
            (1_234, ("1s", "2s", "3s")),
            (61_234, ("1m", "1m", "1m")),
            (3_661_234, ("1h", "1h", "1h")),
            (9 * 86_400_000 + 61_234, ("9d", "9d", "9d")),
        ],
    )
    def test_owed_age_stays_human_across_two_ticks(
        self, state: str, age_ms: int, labels: tuple[str, str, str]
    ) -> None:
        """Status age is approximate, unlike round-trippable schedule syntax.

        A whole-second compound age can still need three terms; that previously
        collapsed into raw milliseconds on the very next poll after a capture.
        """
        from local_operator.tui.widgets.wake_panel import WakePanel

        first = 1_700_000_000_000
        owed = {"state": state, "first_attempt_ms": first}
        for tick, label in zip((0, 1_200, 2_400), labels, strict=True):
            assert WakePanel._owed_label(owed, first + age_ms + tick) == f"{state} · owed {label}"

    def test_a_fresh_live_stall_ages_from_its_own_stamp(self) -> None:
        """F2: the state whose age matters most rendered as a bare word.

        A fresh live-stall record carries no attempt stamp — nothing was
        attempted — so the panel must fall back to ``first_stalled_ms``, the
        same anchor the CLI's ``_owed_age_s`` uses; without it the one state
        that separates "stalled seconds ago" from "stranded last week" showed
        no age while the CLI read "owed 23h" for the same record. A stall
        carried over a retry run keeps dating from the attempt, exactly as the
        CLI does it.
        """
        from local_operator.tui.widgets.wake_panel import WakePanel

        first = 1_700_000_000_000
        fresh = {"state": "live-stalled", "first_stalled_ms": first}
        assert WakePanel._owed_label(fresh, first + 82_800_000) == "live-stalled · owed 23h"

        carried = {
            "state": "live-stalled",
            "first_stalled_ms": first,
            "first_attempt_ms": first - 86_400_000,
        }
        assert WakePanel._owed_label(carried, first) == "live-stalled · owed 1d"

        # And a record with neither stamp still gets its state word.
        assert WakePanel._owed_label({"state": "live-stalled"}, first) == "live-stalled"

    @pytest.mark.asyncio
    async def test_an_owed_fire_carries_its_state_and_age_and_a_healthy_one_does_not(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
        now = int(time.time() * 1000)
        owed = _schedule("w1", "stale but owed")
        owed.next_due_at = now - 9 * 86_400_000
        healthy = _schedule("w2", "fresh, due in 5m")
        healthy.next_due_at = now + 300_000

        out = await self._paint_owed(tmp_path, [owed, healthy])

        owed_row = next(line for line in out.splitlines() if line.startswith("- w1"))
        healthy_row = next(line for line in out.splitlines() if line.startswith("- w2"))
        assert "retrying · owed 9d" in owed_row, owed_row
        assert "retrying" not in healthy_row and "owed" not in healthy_row, healthy_row
        # One row per schedule, and the band keeps its height: the state word and
        # the age replace the due label rather than adding a row.
        assert len([line for line in out.splitlines() if line.startswith("- ")]) == 2, out

    @pytest.mark.asyncio
    async def test_the_undelivered_word_is_the_ledger_state_not_a_second_vocabulary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
        now = int(time.time() * 1000)
        owed = _schedule("w1", "many failures")
        owed.next_due_at = now - 600_000

        out = await self._paint_owed(tmp_path, [owed], state="undelivered")

        assert "undelivered · owed" in out, out

    @pytest.mark.asyncio
    async def test_the_ledger_is_read_on_change_not_on_every_tick(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Memoised on the ledger DIRECTORY's mtime: this runs at 1 Hz.

        Every mutation goes through ``os.replace``/``unlink`` inside that
        directory, so its ``st_mtime_ns`` moves with it — the poll pays a stat,
        not a read per owed wake.
        """
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
        from local_operator.wakes import deliveries

        reads: list[Path] = []
        real = deliveries.read_deliveries

        def counting(root: Path):  # noqa: ANN202
            reads.append(root)
            return real(root)

        monkeypatch.setattr(deliveries, "read_deliveries", counting)
        # The directory must exist for the stat to succeed: a store with no
        # ledger at all is the cheaper path (one failing stat, no read), and
        # that path has its own test below.
        deliveries.deliveries_dir(tmp_path).mkdir(parents=True, exist_ok=True)
        session = self._session([_schedule("w1", "x")])

        app = _PanelHost()
        async with app.run_test(size=(100, 30)) as pilot:
            panel = app.query_one(WakePanel)
            panel.sync(session)
            await pilot.pause()
            assert len(reads) == 1, reads
            panel.sync(session)
            await pilot.pause()
            assert len(reads) == 1, f"the ledger was re-read for an unchanged store: {reads}"

            deliveries.note_failure(
                tmp_path,
                "sess",
                session.wake_scheduler.schedules[0].next_due_at,
                error="unreachable",
                now_ms=int(time.time() * 1000),
            )
            panel.sync(session)
            await pilot.pause()
            assert len(reads) == 2, f"a written record did not invalidate the read: {reads}"

    @pytest.mark.asyncio
    async def test_a_session_with_no_id_or_no_store_still_paints(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A delivery mark is an addition to the row, never a precondition for it."""
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "missing"))
        app = _PanelHost()
        async with app.run_test(size=(100, 30)) as pilot:
            panel = app.query_one(WakePanel)
            panel.sync(_FakeSession([_schedule("w1", "no id, no store")]))  # no session_id
            await pilot.pause()
            assert panel.display is True
            assert "w1" in str(panel._body.content)


class TestMonitorBandHealth:
    """§D6 in the band: the two states the band could not show at all."""

    @pytest.mark.asyncio
    async def test_a_never_checked_monitor_is_flagged(self) -> None:
        spec, counters = _monitor("m1", "watch the queue")
        counters.update({"checks": 0, "deliveries": 0})
        spec.created_at = int(time.time() * 1000) - 3_600_000

        _, out = await _paint([], [(spec, counters)])

        # The band shows the hint's STATE clause only (D1): the explanation
        # pushed the row's own name out at 100 columns and the CLI — which has
        # the room — pins the whole sentence (test_monitor_cli.py).
        assert "never checked" in out
        assert "watch the queue" in out

    @pytest.mark.asyncio
    async def test_an_unavailable_episode_is_flagged(self) -> None:
        spec, counters = _monitor("m1", "watch the queue")
        counters.update({"unavailable_since": int(time.time() * 1000) - 600_000})

        _, out = await _paint([], [(spec, counters)])

        assert "tool unavailable since" in out

    @pytest.mark.asyncio
    async def test_a_quiet_watch_with_no_deliveries_stays_neutral(self) -> None:
        """A watch that has seen nothing may simply be watching something
        quiet, so its hint is NEUTRAL: present, but not the warning ink a
        stalled or disabled row earns.
        """
        spec, counters = _monitor("m1", "watch the queue")
        counters.update({"checks": 9, "deliveries": 0})

        _, out = await _paint([], [(spec, counters)])

        assert "9 checks, 0 deliveries" in out
        assert "never checked" not in out

    @pytest.mark.asyncio
    async def test_a_healthy_monitor_shows_no_hint(self) -> None:
        spec, counters = _monitor("m1", "watch the queue")
        counters.update({"checks": 5, "deliveries": 2})

        _, out = await _paint([], [(spec, counters)])

        assert "checks, 0 deliveries" not in out
        assert "unavailable" not in out

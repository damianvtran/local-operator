"""Startup toast — one slot, one summary, and a timer that always dies.

Three properties are asserted here rather than eyeballed:

- **Single slot.** A second toast replaces the first and cancels its timer.
  Stacking would march a column of cards over the transcript, one per failed
  server, which is the exact failure the coalesced summary exists to prevent.
- **No timer survives.** Dismissal and unmount both stop it. A Textual timer
  outliving its widget is a shutdown warning and an intermittent test failure,
  and this suite has already been debugged for that once.
- **Silence when unused.** A machine with no MCP config gets no toast at all.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.containers import Container

from local_operator.session.mcp_status import McpStartupOutcome
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.status_line import ICON_MCP
from local_operator.tui.widgets.toast import (
    TOAST_DEFAULT_MS,
    TOAST_FAILURE_MS,
    TOAST_MAX_WIDTH,
    TOAST_MIN_WIDTH,
    Toast,
    format_mcp_startup,
    toast_max_width,
)
from tests.unit.tui.conftest import TCSS_PATH

CLEAN = McpStartupOutcome(
    configured=("github", "linear"),
    connected=("github", "linear"),
    tool_count=31,
)
PARTIAL = McpStartupOutcome(
    configured=("github", "linear", "slack"),
    connected=("github", "linear"),
    failures={"slack": "command not found: slack-mcp"},
    tool_count=31,
)
MULTI = McpStartupOutcome(
    configured=("github", "linear", "slack"),
    connected=("github",),
    failures={"slack": "command not found", "linear": "handshake timed out"},
    tool_count=12,
)


class ToastApp(App[None]):
    """A toast under the REAL stylesheet, so the zero-row hidden slot and the
    ``height: auto`` card are the shipped rules rather than defaults."""

    CSS_PATH = TCSS_PATH

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables.update(theme_mod.tcss_variable_map())
        return variables

    def compose(self) -> ComposeResult:
        with Container(id="toast-host"):
            yield Toast()


# -- message content ---------------------------------------------------------


def test_a_clean_startup_names_what_loaded() -> None:
    payload = format_mcp_startup(CLEAN)
    assert payload is not None
    text, duration = payload
    assert text.plain == f"{ICON_MCP} MCP ready: 2 servers, 31 tools"
    assert duration == TOAST_DEFAULT_MS


def test_a_failure_names_the_server_and_its_error_and_holds_longer() -> None:
    """The error text is the actionable part — "command not found" usually IS
    the fix — and a user who has to read and act on it needs longer than the
    5 s a courtesy summary gets.

    The line SAYS failed. Without the verb the reader had to infer which of the
    configured servers the head line meant, and the name appeared twice in
    seventeen characters while never stating the state."""
    payload = format_mcp_startup(PARTIAL)
    assert payload is not None
    text, duration = payload
    lines = text.plain.split("\n")
    assert lines[0] == f"{ICON_MCP} MCP: 2 of 3 servers up, 31 tools"
    assert lines[1] == "failed: slack — command not found: slack-mcp"
    assert duration == TOAST_FAILURE_MS


def test_one_failure_and_many_share_an_opening() -> None:
    """Both variants of the second line start on the same word, so the line
    reads as a failure list at a glance either way."""
    one = format_mcp_startup(PARTIAL)
    many = format_mcp_startup(MULTI)
    assert one is not None and many is not None
    assert one[0].plain.split("\n")[1].startswith("failed: ")
    assert many[0].plain.split("\n")[1].startswith("failed: ")


def test_several_failures_coalesce_into_one_two_line_message() -> None:
    """ONE toast, never one per server: it is an overlay over the user's work.
    The per-server error text lives in the transcript notice and ``/mcp``, both
    of which survive the dismissal."""
    payload = format_mcp_startup(MULTI)
    assert payload is not None
    text, _duration = payload
    assert text.plain.count("\n") == 1
    assert text.plain.split("\n")[1] == "failed: linear, slack"


def _fills(text) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """``{span text: hex fill}`` for every styled span in a rendered message."""
    return {
        text.plain[span.start : span.end]: span.style.color.triplet.hex.lower()
        for span in text.spans
        if getattr(span.style, "color", None) is not None and span.style.color.triplet is not None
    }


def test_the_failure_line_is_tinted_danger_and_the_glyph_is_the_lamp() -> None:
    """The toast's lamp is derived through the band's own rule, so the two
    surfaces cannot disagree about the state they are reporting."""
    payload = format_mcp_startup(PARTIAL)
    assert payload is not None
    text, _duration = payload
    fills = _fills(text)
    danger = theme_mod.semantic_color("danger").lower()
    assert fills[f"{ICON_MCP} "] == danger
    assert fills["failed: slack — command not found: slack-mcp"] == danger


def test_a_clean_startup_lamp_spends_no_green_at_all() -> None:
    """The lamp used to be `success` #57c785, which is 5.08 dE2000 from the
    accent #38c96a — indistinguishable at one cell, and the accent is reserved
    for "a turn is live". A healthy toast is neutral: the card's only coloured
    thing is the failure line, when there is one."""
    payload = format_mcp_startup(CLEAN)
    assert payload is not None
    text, _duration = payload
    fills = _fills(text)
    assert fills[f"{ICON_MCP} "] == theme_mod.semantic_color("muted").lower()
    greens = {theme_mod.semantic_color(name).lower() for name in ("accent", "success")}
    assert not greens & set(fills.values())


def test_nothing_configured_produces_no_toast_at_all() -> None:
    """The whole feature has to be invisible on a machine with no ``.mcp.json``."""
    assert format_mcp_startup(McpStartupOutcome()) is None


def test_servers_still_connecting_past_the_gate_stay_quiet() -> None:
    """The startup gate leaves slow connects in flight on every launch.
    Announcing "0 connected" then would be both alarming and wrong; the band's
    count covers that window and ticks up when the connect lands."""
    pending = McpStartupOutcome(configured=("github",))
    assert format_mcp_startup(pending) is None


def test_a_settling_snapshot_with_a_failure_stays_quiet() -> None:
    """A failure in a SETTLING snapshot must not be toasted: servers deferred
    past the gate are still connecting, and one of them failing fast would
    otherwise flash "N of M up — failed: X" a beat before the slow OAuth
    servers land. The manager re-reports a settled outcome once the round
    drains, and THAT one is what the user sees."""
    settling = McpStartupOutcome(
        configured=("notion", "linear"),
        connected=(),
        failures={"notion": "needs authorization"},
        settling=True,
    )
    assert format_mcp_startup(settling) is None


def test_the_same_failure_toasts_once_the_round_has_settled() -> None:
    """The settled (``settling=False``) re-report is the one that surfaces."""
    settled = McpStartupOutcome(
        configured=("notion", "linear"),
        connected=("linear",),
        failures={"notion": "needs authorization"},
        settling=False,
    )
    payload = format_mcp_startup(settled)
    assert payload is not None
    text, _duration = payload
    lines = text.plain.split("\n")
    assert lines[0] == f"{ICON_MCP} MCP: 1 of 2 servers up"
    assert lines[1] == "failed: notion — needs authorization"


def test_a_hard_discovery_failure_does_not_invent_a_server_tally() -> None:
    """The config layer never produced a server list on that path, so "0 of 0
    servers up" would be meaningless and quietly wrong about what broke."""
    payload = format_mcp_startup(
        McpStartupOutcome(failures={"discovery": "config unreadable"}),
    )
    assert payload is not None
    text, duration = payload
    lines = text.plain.split("\n")
    assert lines[0] == f"{ICON_MCP} MCP discovery failed"
    assert lines[1] == "failed: discovery — config unreadable"
    assert duration == TOAST_FAILURE_MS


def test_the_message_is_clamped_to_the_cells_it_was_given() -> None:
    """A long error must not wrap a two-line note into three."""
    long_error = McpStartupOutcome(
        configured=("github",),
        failures={"github": "x" * 200},
    )
    payload = format_mcp_startup(long_error, max_cells=30)
    assert payload is not None
    text, _duration = payload
    for line in text.plain.split("\n"):
        assert cell_len(line) <= 30, line


# -- width -------------------------------------------------------------------


def test_the_card_is_capped_on_wide_terminals_and_floored_on_narrow_ones() -> None:
    """``min(60, width - 6)`` from the reference, plus a floor: a 200-cell toast
    reads as a banner, and a 24-cell terminal must still get a card rather than
    a negative clamp."""
    assert toast_max_width(200) == TOAST_MAX_WIDTH
    assert toast_max_width(50) == 44
    assert toast_max_width(24) == TOAST_MIN_WIDTH


def test_the_floor_never_outgrows_the_screen_it_paints_on() -> None:
    """The floor is clamped by the terminal's own content box. A card wider than
    the screen is hard-clipped by the compositor, which eats the ellipsis
    ``truncate_cells`` put there — so the one cue that text was cut is the first
    thing lost. Below 22 cells the floor loses to the box."""
    assert toast_max_width(21) == 19
    assert toast_max_width(20) == 18
    assert toast_max_width(16) == 14
    # Never negative, never zero, whatever the terminal claims.
    assert toast_max_width(1) == 1
    assert toast_max_width(0) == 1


# -- lifecycle ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_empty_slot_claims_no_rows() -> None:
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        assert toast.display is False
        assert pilot.app.query_one("#toast-host").size.height == 0


@pytest.mark.asyncio
async def test_a_second_toast_replaces_the_first_and_cancels_its_timer() -> None:
    """Cap one, and the old timer must be DEAD before the new one is armed.

    Asserted by outcome rather than by poking at the timer's internals: the
    first toast is given a 1 ms life, the second a minute. If ``show`` left the
    first timer running, its expiry would hide the replacement — so a still
    visible second card after pumping the loop is the proof.
    """
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("first", duration_ms=1)
        first_timer = toast._timer
        toast.show("second", duration_ms=60_000)
        for _ in range(10):
            await pilot.pause()
        assert toast._timer is not first_timer
        assert toast.display is True
        assert toast.message == "second"


@pytest.mark.asyncio
async def test_dismissal_hides_the_card_and_leaves_no_timer_behind() -> None:
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("hello", duration_ms=5000)
        await pilot.pause()
        assert toast.display is True
        toast.dismiss_toast()
        await pilot.pause()
        assert toast.display is False
        assert toast._timer is None
        # Idempotent: a timer firing after a manual dismiss must be a no-op.
        toast.dismiss_toast()
        assert toast._timer is None


@pytest.mark.asyncio
async def test_the_timer_actually_dismisses_the_toast() -> None:
    """Driven with a 1 ms duration rather than by sleeping out the real 5 s:
    the property under test is that the timer fires and calls the dismissal,
    not how long Textual waits."""
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("bye", duration_ms=1)
        assert toast.display is True
        for _ in range(20):
            await pilot.pause()
            if not toast.display:
                break
        assert toast.display is False
        assert toast._timer is None


@pytest.mark.asyncio
async def test_unmount_stops_a_live_timer() -> None:
    """A timer outliving its widget is a shutdown warning and a test flake."""
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("still up", duration_ms=60_000)
        timer = toast._timer
        assert timer is not None
        await toast.remove()
        await pilot.pause()
        assert toast._timer is None


@pytest.mark.asyncio
async def test_the_host_is_never_wider_than_the_card_it_holds() -> None:
    """The host owns a REGION, and Textual blanks all of it. A full-width host
    (`width: 1fr`, as shipped) painted 35 cells of notice and erased the rest of
    every row it covered — see the A/B in test_app_pilot. Hugging the card is
    what confines the damage to the cells the notice occupies, and the offset is
    what keeps it against the screen's right inset."""
    async with ToastApp().run_test(size=(96, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        host = pilot.app.query_one("#toast-host")
        toast.show("⊙ MCP ready: 2 servers, 9 tools", duration_ms=60_000)
        await pilot.pause()
        await pilot.pause()
        assert host.region == toast.region
        # Flush against the screen's own one-cell inset, not the terminal edge.
        assert toast.region.right == 96 - 1


@pytest.mark.asyncio
async def test_the_card_stays_inside_a_terminal_narrower_than_the_floor() -> None:
    """At 20 cells the floor (20) beat the box (18) and the card painted two
    cells past the screen, where the compositor clipped the ellipsis off the
    truncated head."""
    for width in (16, 20, 21):
        async with ToastApp().run_test(size=(width, 12)) as pilot:
            toast = pilot.app.query_one(Toast)
            toast.show("⊙ MCP ready: 2 servers, 9 tools", duration_ms=60_000)
            await pilot.pause()
            await pilot.pause()
            assert toast.region.right <= width - 1, width
            assert toast.region.x >= 1, width


@pytest.mark.asyncio
async def test_a_click_dismisses_the_card_early() -> None:
    """The timer is a floor on how long the notice is readable, not a sentence:
    the failure variant holds ten seconds over the transcript."""
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("⊙ MCP: 1 of 2 servers up", duration_ms=60_000)
        await pilot.pause()
        assert toast.display is True
        await pilot.click(Toast)
        await pilot.pause()
        assert toast.display is False
        assert toast._timer is None
        assert toast.message == ""


# -- the slot's two guards ----------------------------------------------------
#
# Both exist because a routine editing gesture (a drag over the composer)
# became able to write this card. A copy receipt is a COURTESY — the user did
# the thing and can see the result — while an MCP failure names a server and an
# error they have not read yet. So a courtesy card declines the slot rather
# than taking it, and `generation` lets a caller withdraw its own card later
# without being able to touch anyone else's.


@pytest.mark.asyncio
async def test_a_courtesy_card_declines_a_slot_an_actionable_notice_holds() -> None:
    """The failure the user must act on outranks the receipt for what they did."""
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("⊙ MCP failed: github", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()

        toast.show("copied 9 characters", yield_to_actionable=True)
        await pilot.pause()

        assert toast.message == "⊙ MCP failed: github"


@pytest.mark.asyncio
async def test_a_courtesy_card_takes_a_slot_an_ordinary_notice_holds() -> None:
    """The deference is to ACTIONABILITY, not to whatever happens to be showing.

    Without this the test above would also pass if a courtesy card had simply
    stopped being able to show at all.
    """
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("⊙ MCP ready: 2 servers", duration_ms=TOAST_DEFAULT_MS)
        await pilot.pause()

        toast.show("copied 9 characters", yield_to_actionable=True)
        await pilot.pause()

        assert toast.message == "copied 9 characters"


@pytest.mark.asyncio
async def test_an_actionable_notice_is_never_refused_the_slot() -> None:
    """`yield_to_actionable` is opt-in; a failure always gets through."""
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("⊙ MCP failed: github", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()

        toast.show("⊙ MCP failed: gitlab", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()

        assert toast.message == "⊙ MCP failed: gitlab"


@pytest.mark.asyncio
async def test_dismissing_clears_the_actionable_hold() -> None:
    """A failure that has been read must not lock the slot for the session."""
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("⊙ MCP failed: github", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()
        toast.dismiss_toast()
        await pilot.pause()

        toast.show("copied 9 characters", yield_to_actionable=True)
        await pilot.pause()

        assert toast.message == "copied 9 characters"


@pytest.mark.asyncio
async def test_the_generation_names_the_card_that_is_showing() -> None:
    """Each clause of `generation`'s contract, separately.

    It is load-bearing for the copy receipt: the app holds the generation of
    the card its own copy raised and dismisses only while that is still what is
    on screen. Every clause below is independently falsifiable, and the third
    is the one a real bug turned on — a copy made while a failure was up
    adopted the FAILURE's generation and withdrew it on the next keystroke.
    """
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        # 0 before anything has ever been shown.
        assert toast.generation == 0

        # +1 per ACCEPTED show.
        toast.show("first", duration_ms=TOAST_DEFAULT_MS)
        await pilot.pause()
        first = toast.generation
        assert first == 1
        toast.show("second", duration_ms=TOAST_DEFAULT_MS)
        await pilot.pause()
        assert toast.generation == 2

        # UNCHANGED by a show that declined the slot: no card was raised, so
        # there is nothing for a caller to name.
        toast.show("⊙ MCP failed: github", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()
        held = toast.generation
        toast.show("copied 9 characters", yield_to_actionable=True)
        await pilot.pause()
        assert toast.generation == held

        # Unchanged by dismissal, so a held value can never come to match a
        # later card by accident. Dismissed twice: the first frees the slot and
        # lets the deferred receipt take its turn (which is a new card, and so
        # a new generation), the second retires that.
        toast.dismiss_toast()
        await pilot.pause()
        assert toast.message == "copied 9 characters"
        after_deferred = toast.generation
        assert after_deferred == held + 1
        toast.dismiss_toast()
        await pilot.pause()
        assert toast.generation == after_deferred


@pytest.mark.asyncio
async def test_a_deferred_courtesy_card_gets_its_turn_when_the_slot_frees() -> None:
    """Held, not dropped: the acknowledgement is late rather than lost.

    Deferring fixed the eviction, but it left the copy with no feedback at all
    — the failure was dismissed and nothing ever said the text had been taken
    (design round 2, D9). The user had dragged, seen nothing, and the notice
    they were reading disappeared.
    """
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("⊙ MCP failed: github", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()

        toast.show("copied 9 characters", yield_to_actionable=True)
        await pilot.pause()
        assert toast.message == "⊙ MCP failed: github"

        toast.dismiss_toast()
        await pilot.pause()

        assert toast.message == "copied 9 characters"
        assert toast.display


@pytest.mark.asyncio
async def test_only_the_latest_deferred_card_is_kept() -> None:
    """Superseded news is not owed a turn.

    Three drags behind one failure notice must not march three cards down the
    screen afterwards — that is the stacking the single slot exists to prevent.
    """
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("⊙ MCP failed: github", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()
        for message in ("copied 3 characters", "copied 5 characters", "copied 9 characters"):
            toast.show(message, yield_to_actionable=True)
            await pilot.pause()

        toast.dismiss_toast()
        await pilot.pause()
        assert toast.message == "copied 9 characters"

        # ...and exactly one card is owed. The next dismissal ends it rather
        # than uncovering another receipt.
        toast.dismiss_toast()
        await pilot.pause()
        assert toast.message == ""
        assert toast.display is False


@pytest.mark.asyncio
async def test_a_deferred_card_does_not_defer_to_itself() -> None:
    """The hold is released before the deferred card is shown.

    Dismissal clears `_actionable` first, so the card taking its turn cannot
    find the slot still held and re-defer — which would either lose it after
    all or, worse, recurse.
    """
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("⊙ MCP failed: github", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()
        toast.show("copied 9 characters", yield_to_actionable=True)
        await pilot.pause()

        toast.dismiss_toast()
        await pilot.pause()

        assert toast.message == "copied 9 characters"
        assert toast._deferred is None
        # A courtesy card is not itself actionable, so the slot is takeable.
        toast.show("copied 4 characters", yield_to_actionable=True)
        await pilot.pause()
        assert toast.message == "copied 4 characters"


@pytest.mark.asyncio
async def test_a_card_that_actually_showed_clears_the_deferred_one() -> None:
    """A receipt that got its own turn is not also owed a later one.

    Without clearing the hold on an accepted `show`, a copy deferred behind a
    failure would resurface after some unrelated notice minutes later — a
    receipt for a copy the user made long ago, arriving with no gesture behind
    it.
    """
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("⊙ MCP failed: github", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()
        toast.show("copied 9 characters", yield_to_actionable=True)
        await pilot.pause()
        assert toast._deferred is not None

        # The failure is replaced by an ordinary notice rather than dismissed,
        # so the slot never passes through `dismiss_toast`.
        toast.show("⊙ MCP ready: 2 servers", duration_ms=TOAST_DEFAULT_MS)
        await pilot.pause()
        assert toast._deferred is None

        toast.dismiss_toast()
        await pilot.pause()
        assert toast.message == ""
        assert toast.display is False


# -- withdrawal, by owner -----------------------------------------------------
#
# The slot is shared, so "retire my card" has to mean *mine*. Four bugs in this
# PR (F5, D8, D14, F14) were all one signal reaching a card it did not own,
# patched at four different layers; `withdraw(owner)` asks the question once.
# These pin it at the widget, where the ownership actually lives — the
# end-to-end tests in `test_transcript_selection.py` only ever exercise two
# owners, so the cases that make ownership MEAN something are unpinned there
# (review round 5, F16).

#: Two distinct callers, as the app's own sentinels are.
OWNER_A = object()
OWNER_B = object()


@pytest.mark.asyncio
async def test_withdrawing_retires_a_showing_card_of_your_own() -> None:
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("mine", owner=OWNER_A)
        await pilot.pause()

        toast.withdraw(OWNER_A)
        await pilot.pause()

        assert toast.message == ""
        assert toast.display is False


@pytest.mark.asyncio
async def test_withdrawing_leaves_someone_else_s_showing_card_alone() -> None:
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("theirs", owner=OWNER_B)
        await pilot.pause()

        toast.withdraw(OWNER_A)
        await pilot.pause()

        assert toast.message == "theirs"
        assert toast.display


@pytest.mark.asyncio
async def test_withdrawing_leaves_someone_else_s_held_card_alone() -> None:
    """...and it still gets its turn when the slot frees."""
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("⊙ MCP failed: github", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()
        toast.show("theirs", yield_to_actionable=True, owner=OWNER_B)
        await pilot.pause()

        toast.withdraw(OWNER_A)
        await pilot.pause()
        toast.dismiss_toast()
        await pilot.pause()

        assert toast.message == "theirs"


@pytest.mark.asyncio
async def test_withdrawing_drops_the_hold_without_touching_the_card_above_it() -> None:
    """One owner showing, another held — the shipped failure-plus-copy state."""
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("⊙ MCP failed: github", duration_ms=TOAST_FAILURE_MS, owner=OWNER_B)
        await pilot.pause()
        toast.show("mine", yield_to_actionable=True, owner=OWNER_A)
        await pilot.pause()

        toast.withdraw(OWNER_A)
        await pilot.pause()
        assert toast.message == "⊙ MCP failed: github"

        # The slot frees to nothing: the held card was withdrawn, not promoted.
        toast.dismiss_toast()
        await pilot.pause()
        assert toast.message == ""
        assert toast.display is False


@pytest.mark.asyncio
async def test_withdrawing_the_card_above_promotes_the_hold_beneath_it() -> None:
    """The same state, withdrawn from the other side."""
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("⊙ MCP failed: github", duration_ms=TOAST_FAILURE_MS, owner=OWNER_B)
        await pilot.pause()
        toast.show("mine", yield_to_actionable=True, owner=OWNER_A)
        await pilot.pause()

        toast.withdraw(OWNER_B)
        await pilot.pause()

        assert toast.message == "mine"
        assert toast.display


@pytest.mark.asyncio
async def test_withdrawing_drops_the_hold_before_dismissing_the_card() -> None:
    """The order inside `withdraw` is load-bearing, in a reachable state.

    `dismiss_toast` PROMOTES whatever is held. So if `withdraw` dismissed
    first, it would raise the very card it is withdrawing and then find the
    hold already consumed — leaving the claim on screen (review round 5, F15).

    The state is constructed directly because no caller reaches it yet: holding
    requires the showing card to be actionable, and the only owned card the app
    raises is a copy receipt at `TOAST_DEFAULT_MS`. That is the point — the
    order is correct defensively, and this pins it before the first owner that
    would make the hazard live (round 6, F19).

    Swapping the two statements leaves every other test in the suite passing,
    which is exactly why this one exists.
    """
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("mine, actionable", duration_ms=TOAST_FAILURE_MS, owner=OWNER_A)
        await pilot.pause()
        toast.show("mine, held", yield_to_actionable=True, owner=OWNER_A)
        await pilot.pause()

        toast.withdraw(OWNER_A)
        await pilot.pause()

        assert toast.message == ""
        assert toast.display is False
        assert toast._deferred is None


@pytest.mark.asyncio
async def test_withdrawing_nothing_is_silent() -> None:
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)

        toast.withdraw(OWNER_A)
        await pilot.pause()

        assert toast.message == ""
        assert toast.display is False


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["withdraw", "drop_deferred"])
async def test_none_is_not_an_owner(method: str) -> None:
    """`None` tags every card that named no owner, so it cannot address one.

    Nothing calls it that way today, but it is a one-word mistake that
    typechecks (`object` includes `None`) and would silently retire an unread
    MCP failure — D2 for the third time (review round 5, F17).
    """
    async with ToastApp().run_test(size=(80, 24)) as pilot:
        toast = pilot.app.query_one(Toast)
        toast.show("⊙ MCP failed: github", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()

        with pytest.raises(ValueError):
            getattr(toast, method)(None)

        assert toast.message == "⊙ MCP failed: github"
        assert toast.display

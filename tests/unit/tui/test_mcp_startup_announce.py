"""The MCP startup toast announces once per session, per distinct outcome.

Reported: "MCP ready: 12 servers, 425 tools" fired on EVERY session attach,
including every click in the session sidebar. ``session.mcp_startup`` is a
frozen BOOT SNAPSHOT and ``_report_mcp_startup`` runs on every adoption, so a
sidebar switch re-announced a round that happened minutes ago — and, through
``RemoteSession``'s rehydration of the owner's outcome, sometimes a round this
process never ran at all.

These tests pin the rule the fix implements: each SESSION remembers the last
sentence it was told and stays silent while its own outcome is unchanged, so
clicking away and back is quiet (A→B→A→B across sessions) while a server that
breaks, recovers and breaks again is announced every time (A→B→A on one
session). A session seeing a sentence that is CURRENTLY on screen inherits
that record, which is what keeps one shared MCP set to one toast. The identity
is the UNTRUNCATED sentence — the card still shows the width-fitted one — so
terminal width cannot re-arm or swallow an announce. The record is taken when
the card is DISPLAYED, not when ``show`` is called, and is released by
eviction for every session that card spoke for. The durable failure notice is
deduped per session AND per failure instead. The status band is asserted
alongside, because the suppression is only safe while the band keeps
re-stating live MCP state on every attach.
"""

from __future__ import annotations

import os

import pytest

from local_operator.session.mcp_status import McpStartupOutcome
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.toast import Toast
from tests.unit.tui.test_app_pilot import (
    FakeMcpManager,
    McpSession,
    _band,
    _factory,
    _transcript_text,
)


@pytest.fixture(autouse=True)
def isolate_sources(tmp_path, monkeypatch):
    # A headless pilot that inherits the operator's CMUX_* variables can rename
    # their real cmux workspaces; HOME is redirected too because the config dir
    # alone leaves the cache pointed at the real home (AGENTS.md, "Isolating a
    # run").
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda self: None)


def _outcome(**overrides) -> McpStartupOutcome:
    """The reported shape, shrunk: several servers up, a tool tally, no failure."""
    fields = {
        "configured": ("github", "linear", "slack"),
        "connected": ("github", "linear", "slack"),
        "tool_count": 425,
    }
    fields.update(overrides)
    return McpStartupOutcome(**fields)  # type: ignore[arg-type]


class _IdentifiedMcpSession(McpSession):
    """``McpSession`` with a settable id.

    ``FakeSession.session_id`` is a fixed property returning ``"sess"``, so two
    fakes are indistinguishable to the per-session notice key — which is
    exactly the distinction these tests are about.
    """

    def __init__(self, manager, startup, session_id: str) -> None:
        super().__init__(manager, startup)
        self._session_id = session_id

    @property
    def session_id(self) -> str:
        return self._session_id


def _session(outcome: McpStartupOutcome, *, session_id: str) -> _IdentifiedMcpSession:
    manager = FakeMcpManager(list(outcome.configured), list(outcome.connected))
    return _IdentifiedMcpSession(manager, outcome, session_id)


#: The reported failure, as a whole outcome: slack down, the other two up.
_SLACK_DOWN = dict(
    connected=("github", "linear"),
    failures={"slack": "command not found: slack-mcp"},
    tool_count=310,
)

#: Two errors whose FULL sentences differ but whose 50-column renders are
#: byte-identical (both ellipsize to ``failed: slack — command not found:
#: slack-…``). The pair review and UX both measured collapsing into one
#: announce on the round-2 head, swallowing the second failure (R2-2, U2-1).
_TRUNCATION_COLLISION_A = "command not found: slack-mcp-stdio-bridge"
_TRUNCATION_COLLISION_B = "command not found: slack-mcp-oauth-refresh-expired"

#: An error long enough to fit the 100-column card WHOLE but truncate at 46,
#: so a resize between attaches changed the round-2 head's key for an
#: unchanged outcome (R2-1).
_RESIZE_SENSITIVE_ERROR = "command not found: slack-mcp-stdio-bridge-v2-bin"


async def _until(pilot, predicate) -> bool:  # type: ignore[no-untyped-def]
    """Pause until ``predicate()`` holds, or the budget runs out.

    Polled rather than a fixed tick count: a wall-clock-shaped wait is what
    made the evidence script flake ~18% of the time under load (UX round 1,
    U6), and AGENTS.md's timing section says to wait on the event, not the
    clock. Returns whether it held, so a caller asserting the NEGATIVE (a toast
    that must stay silent) can still spend the full budget looking for it.

    The budget is 200 ``pause()`` calls, not seconds: a pause is one event-loop
    tick, and the observed settle for this surface is single-digit ticks (the
    announce is synchronous in ``show``; only the posted ``Toast.Evicted`` ever
    needs a drain). 200 is roughly two orders of magnitude above that need
    while still costing milliseconds — tight enough that a genuine hang fails
    the test quickly, loose enough that CI under load cannot turn a pass into
    a flake (review round 2, R2-5).
    """
    for _ in range(200):
        await pilot.pause()
        if predicate():
            return True
    return False


async def _quiet(pilot) -> None:  # type: ignore[no-untyped-def]
    """Give a toast that must NOT appear every chance to appear anyway."""
    for _ in range(20):
        await pilot.pause()


@pytest.mark.asyncio
async def test_boot_announces_the_startup_outcome() -> None:
    """First loadup is exactly what the operator wants announced."""
    app = OperatorApp(lambda: _factory(_session(_outcome(), session_id="a")))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        assert "MCP ready: 3 servers, 425 tools" in toast.message


@pytest.mark.asyncio
async def test_attaching_a_second_session_with_the_same_outcome_stays_silent() -> None:
    """THE REPORTED DEFECT. MCP servers are process-wide and shared, so every
    sidebar click re-announced a tally the user was already told. The band still
    carries the live count, which is what makes the silence safe."""
    first = _session(_outcome(), session_id="a")
    second = _session(_outcome(), session_id="b")
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        toast.dismiss_toast()
        await pilot.pause()

        app._adopt_session(second, replay_history=False)
        await _quiet(pilot)
        assert toast.display is False, "a re-attach re-announced a boot snapshot"
        # The surface that legitimately re-states MCP state per attach.
        assert "\u2299 3 MCP" in _band(app)


@pytest.mark.asyncio
async def test_four_sidebar_clicks_onto_one_mcp_set_announce_once() -> None:
    """The operator's actual complaint, at the count they reported it at.

    A\u2192A\u2192A\u2192A\u2192A: the sentence never changes, so only the first announces.
    Pinned at four attaches rather than one because "only the CURRENT announce
    is compared" would still be satisfied by a rule that alternated, and this
    is the case that must not regress while U1 is being fixed.
    """
    sessions = [_session(_outcome(), session_id=f"s{i}") for i in range(5)]
    shown: list[str] = []
    app = OperatorApp(lambda: _factory(sessions[0]))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        original = toast.show

        def _spy(text, **kwargs):  # type: ignore[no-untyped-def]
            shown.append(text.plain if hasattr(text, "plain") else str(text))
            return original(text, **kwargs)

        assert await _until(pilot, lambda: toast.display)
        assert toast.message.startswith("\u2299 MCP ready")
        toast.show = _spy  # type: ignore[method-assign]

        for session in sessions[1:]:
            toast.dismiss_toast()
            await pilot.pause()
            app._adopt_session(session, replay_history=False)
            await _quiet(pilot)
            assert toast.display is False

        assert shown == [], f"a re-attach re-announced: {shown}"


@pytest.mark.asyncio
async def test_alternating_clicks_between_two_mcp_sets_announce_once_each() -> None:
    """QA round 2, Q2-1 — the reported defect had returned at full strength.

    The sidebar catalogue is not cwd-scoped (``load_catalog`` scans the whole
    store) while MCP is wired per session cwd, so alternating between a repo
    with a ``.mcp.json`` and one without is ordinary A→B→A→B. One global
    sentence slot cannot tell "the world changed" from "I clicked elsewhere
    and came back": every single click re-announced — 20 announces in 20
    clicks, identical to the unfixed tree. Per session, each side of the
    alternation is unchanged since its own last announce, so only the first
    click per distinct outcome speaks.

    QA's own shape: the real ``_adopt_session``, the card dismissed between
    clicks so eviction is not the variable, and announces counted at the real
    ``Toast.show``. The boot session carries a third outcome so neither click
    target has been told yet — the expected count is then exactly one per
    distinct session outcome.
    """
    boot = _session(
        _outcome(
            configured=("github", "linear"),
            connected=("github", "linear"),
            tool_count=180,
        ),
        session_id="boot",
    )
    repo = _session(_outcome(), session_id="repo")
    plain = _session(
        _outcome(configured=("github",), connected=("github",), tool_count=40),
        session_id="plain",
    )
    app = OperatorApp(lambda: _factory(boot))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        toast.dismiss_toast()
        await pilot.pause()

        announced = 0
        for index in range(20):
            app._adopt_session(repo if index % 2 == 0 else plain, replay_history=False)
            await _quiet(pilot)
            if toast.display:
                announced += 1
                toast.dismiss_toast()
                await pilot.pause()
        assert announced == 2, (
            f"{announced} announces in 20 alternating clicks between two MCP "
            "sets — expected exactly one per distinct session outcome"
        )


@pytest.mark.asyncio
async def test_re_adopting_the_same_session_stays_silent() -> None:
    """A sidebar click back onto the session already attached is the same
    snapshot a second time; nothing about it is news."""
    session = _session(_outcome(), session_id="a")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        toast.dismiss_toast()
        await pilot.pause()
        app._adopt_session(session, replay_history=False)
        await _quiet(pilot)
        assert toast.display is False


@pytest.mark.asyncio
async def test_a_changed_tally_announces_again() -> None:
    """A session in another cwd with a genuinely different server set IS news,
    and the key is the rendered sentence — so it announces once of its own."""
    first = _session(_outcome(), session_id="a")
    second = _session(
        _outcome(
            configured=("github", "linear", "slack", "notion"),
            connected=("github", "linear", "slack", "notion"),
            tool_count=511,
        ),
        session_id="b",
    )
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        toast.dismiss_toast()
        await pilot.pause()

        app._adopt_session(second, replay_history=False)
        assert await _until(pilot, lambda: toast.display)
        assert "MCP ready: 4 servers, 511 tools" in toast.message


@pytest.mark.asyncio
async def test_connect_order_alone_does_not_re_announce() -> None:
    """``connected`` is a set with an incidental order \u2014 the connect race
    reorders it run to run. An order-sensitive key would re-toast an outcome
    whose content is identical; the rendered sentence is inherently immune."""
    first = _session(_outcome(), session_id="a")
    second = _session(
        _outcome(connected=("slack", "github", "linear")),
        session_id="b",
    )
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        toast.dismiss_toast()
        await pilot.pause()
        app._adopt_session(second, replay_history=False)
        await _quiet(pilot)
        assert toast.display is False


@pytest.mark.asyncio
async def test_the_same_failure_recurring_after_a_recovery_announces_again() -> None:
    """UX round 1, U1 \u2014 the regression the ever-growing ledger introduced.

    A ``lop`` window lives for days, so a server that dies, recovers and dies
    AGAIN is the operator's normal shape. Keyed by "every sentence ever shown",
    the second death was silent forever because its sentence had been retired
    hours earlier \u2014 suppressing a RECURRING FAILURE, which is the single most
    interruption-worthy thing this surface reports and the exact opposite of
    the intent. Keyed against the CURRENT announce only, A\u2192B\u2192A announces on
    the return, and it keeps announcing however often the server flaps.
    """
    session = _session(_outcome(), session_id="a")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        assert "MCP ready: 3 servers" in toast.message

        for round_number in range(1, 4):
            # It breaks.
            toast.dismiss_toast()
            await pilot.pause()
            session.mcp_startup = _outcome(**_SLACK_DOWN)
            app._report_mcp_startup(session)
            assert await _until(pilot, lambda: toast.display), (
                f"breakage {round_number} was silent \u2014 a recurring failure "
                "must always announce"
            )
            assert "2 of 3 servers up" in toast.message
            assert "slack" in toast.message

            # It recovers. Also news: pre-fix this announced too.
            toast.dismiss_toast()
            await pilot.pause()
            session.mcp_startup = _outcome()
            app._report_mcp_startup(session)
            assert await _until(pilot, lambda: toast.display), f"recovery {round_number} was silent"
            assert "MCP ready: 3 servers" in toast.message


@pytest.mark.asyncio
async def test_a_flap_on_one_session_announces_through_sibling_clicks() -> None:
    """U1 and Q2-1 together — the two shapes UX walked in one flow.

    The sibling clicks must stay silent (nothing changed for either session)
    while the break→recover→break on ONE session announces every transition:
    per-session records are what let both hold at once, where the single
    global slot could only ever trade one for the other.
    """
    # The sibling carries a THIRD sentence, distinct from both of a's states:
    # the point is that a's transitions announce on their own merits, not that
    # they coincide with words the user just read on the sibling's card.
    broken = _session(_outcome(**_SLACK_DOWN), session_id="a")
    sibling = _session(
        _outcome(
            configured=("notion", "sentry", "stripe"),
            connected=("notion", "sentry", "stripe"),
            tool_count=512,
        ),
        session_id="b",
    )
    app = OperatorApp(lambda: _factory(broken))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        assert "2 of 3 servers up" in toast.message  # a breaks (announce 1)

        toast.dismiss_toast()
        await pilot.pause()
        app._adopt_session(sibling, replay_history=False)
        assert await _until(pilot, lambda: toast.display)  # b's first (2)
        assert "3 servers" in toast.message and "512 tools" in toast.message

        toast.dismiss_toast()
        await pilot.pause()
        app._adopt_session(broken, replay_history=False)
        await _quiet(pilot)
        assert toast.display is False, "clicking back to an unchanged outcome re-announced"

        toast.dismiss_toast()
        await pilot.pause()
        app._adopt_session(sibling, replay_history=False)
        await _quiet(pilot)
        assert toast.display is False, "the sibling's unchanged outcome re-announced"

        broken.mcp_startup = _outcome()  # a recovers (announce 3)
        app._report_mcp_startup(broken)
        assert await _until(pilot, lambda: toast.display)
        assert "MCP ready: 3 servers" in toast.message

        toast.dismiss_toast()
        await pilot.pause()
        app._adopt_session(sibling, replay_history=False)
        await _quiet(pilot)
        assert toast.display is False

        broken.mcp_startup = _outcome(**_SLACK_DOWN)  # a breaks AGAIN (4)
        app._report_mcp_startup(broken)
        assert await _until(
            pilot, lambda: toast.display
        ), "a recurring failure after a recovery went silent — U1 must stay closed"
        assert "2 of 3 servers up" in toast.message


@pytest.mark.asyncio
async def test_an_announce_evicted_before_it_is_read_is_not_spent() -> None:
    """UX round 1, U2 \u2014 the announce is consumed on being SEEN, not on being
    shown.

    The single toast slot is shared, and ``yield_to_actionable`` only protects
    an incumbent whose duration is ``TOAST_FAILURE_MS``. The healthy announce is
    ``TOAST_DEFAULT_MS``, so a routine copy receipt inside those 5 s replaces
    it. Recording at ``show`` time spent the user's ONE announce on a card they
    never read, with no second chance now the rule is one-shot.
    """
    first = _session(_outcome(), session_id="a")
    second = _session(_outcome(), session_id="b")
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        assert "MCP ready: 3 servers" in toast.message

        # The documented eviction: a routine copy receipt, which asks to yield
        # but is not made to, because the announce is not actionable.
        toast.show("copied 12 lines to clipboard", yield_to_actionable=True)
        assert await _until(pilot, lambda: toast.message.startswith("copied"))
        assert "MCP" not in toast.message, "the announce really was evicted"

        # The next attach owes the user the announce they never got to read.
        toast.dismiss_toast()
        await pilot.pause()
        app._adopt_session(second, replay_history=False)
        assert await _until(
            pilot, lambda: toast.display
        ), "an evicted announce was still counted as told"
        assert "MCP ready: 3 servers, 425 tools" in toast.message


@pytest.mark.asyncio
async def test_an_eviction_releases_every_session_the_card_spoke_for() -> None:
    """U2's release, per session now — the evicted card spoke for more than
    its raiser.

    A session that stayed silent because the words were ALREADY on screen
    inherited its record from that card, so the card being thrown away unread
    owes that session its announce back too. A session told by an earlier,
    delivered card keeps its record: it was told by a card that retired
    normally. Pins the per-session release; passes on the round-2 head as
    well, where the single slot coincided with this behaviour.
    """
    first = _session(_outcome(), session_id="a")
    second = _session(_outcome(), session_id="b")
    third = _session(_outcome(), session_id="c")
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)

        # b and c inherit the LIVE card's record: same words, on screen now.
        app._adopt_session(second, replay_history=False)
        await _quiet(pilot)
        app._adopt_session(third, replay_history=False)
        await _quiet(pilot)
        assert toast.display, "the boot card is still showing"

        # A routine copy receipt throws that card away unread. Everyone it
        # spoke for — its raiser and its inheritors — is owed the announce.
        toast.show("copied 12 lines to clipboard", yield_to_actionable=True)
        assert await _until(pilot, lambda: toast.message.startswith("copied"))

        toast.dismiss_toast()
        await pilot.pause()
        app._adopt_session(third, replay_history=False)
        assert await _until(
            pilot, lambda: toast.display
        ), "an inheritor's record survived the eviction of the card that spoke for it"


@pytest.mark.asyncio
async def test_an_announce_the_user_actually_saw_is_not_re_announced() -> None:
    """The other half of U2: eviction is not the same as expiry.

    A card that ran its timer out, or that the user clicked away, WAS delivered
    \u2014 they had their chance to read it. Only a card thrown away mid-display
    releases the record. Without this, `on_toast_evicted` would re-arm on every
    normal dismissal and the reported defect would return.
    """
    first = _session(_outcome(), session_id="a")
    second = _session(_outcome(), session_id="b")
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        toast.dismiss_toast()  # read and dismissed, the ordinary path
        await pilot.pause()
        app._adopt_session(second, replay_history=False)
        await _quiet(pilot)
        assert toast.display is False


@pytest.mark.asyncio
async def test_a_flap_inside_one_card_does_not_release_the_live_announce() -> None:
    """Review round 2, R2-3 — a stale ``Evicted`` released a live announce.

    ``Evicted`` is posted, so it arrives after further reports have run. A
    flap A→B→A inside one card's lifetime queues ``Evicted(text=A)`` from the
    FIRST card; matching the release on the WORDS made it clear a record that
    had since come back to A, re-arming the announce for a sentence the user
    was reading right then. The release matches the card's GENERATION, which
    cannot alias. Sentences here are short so truncation is not the variable.
    """
    session = _session(_outcome(), session_id="a")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        assert "MCP ready: 3 servers" in toast.message  # A takes the slot

        # No dismissal: each report evicts the previous card, queueing its
        # Evicted behind the pump.
        session.mcp_startup = _outcome(**_SLACK_DOWN)
        app._report_mcp_startup(session)  # B evicts A
        session.mcp_startup = _outcome()
        app._report_mcp_startup(session)  # A evicts B — the flap closes

        # The queued Evicted(text=A) is delivered here and must not release
        # the record: the card on screen IS A, and the user is reading it.
        await pilot.pause()
        await _quiet(pilot)
        assert toast.display, "the live announce card vanished"
        assert "MCP ready: 3 servers" in toast.message

        toast.dismiss_toast()
        await pilot.pause()
        app._report_mcp_startup(session)
        await _quiet(pilot)
        assert (
            toast.display is False
        ), "the A→B→A flap re-armed the announce for a sentence already on screen"


@pytest.mark.asyncio
async def test_an_actionable_failure_announce_is_held_not_evicted() -> None:
    """The failure announce is ``TOAST_FAILURE_MS`` and therefore actionable, so
    a courtesy receipt defers to it rather than evicting it. It must stay
    recorded: nothing was thrown away, so a re-attach is still a repeat."""
    session = _session(_outcome(**_SLACK_DOWN), session_id="a")
    other = _session(_outcome(**_SLACK_DOWN), session_id="b")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        assert "2 of 3 servers up" in toast.message

        toast.show("copied 12 lines to clipboard", yield_to_actionable=True)
        await _quiet(pilot)
        assert (
            "2 of 3 servers up" in toast.message
        ), "the actionable announce should have held the slot"

        toast.dismiss_toast()
        await _until(pilot, lambda: toast.message.startswith("copied"))
        toast.dismiss_toast()
        await pilot.pause()
        app._adopt_session(other, replay_history=False)
        await _quiet(pilot)
        assert toast.display is False, "a held announce was never lost, so it is a repeat"


@pytest.mark.asyncio
async def test_two_server_sets_rendering_one_sentence_announce_once() -> None:
    """Review round 1, R1-2 \u2014 the key is what the user would READ.

    Two disjoint server sets of the same size and tool tally render a
    byte-identical sentence. Keyed structurally, the user saw the same words
    twice for no reason; keyed on the sentence, it is one piece of news.
    """
    first = _session(_outcome(), session_id="a")
    second = _session(
        _outcome(
            configured=("notion", "sentry", "stripe"),
            connected=("notion", "sentry", "stripe"),
        ),
        session_id="b",
    )
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        sentence = toast.message
        toast.dismiss_toast()
        await pilot.pause()

        app._adopt_session(second, replay_history=False)
        await _quiet(pilot)
        assert toast.display is False, f"the same sentence {sentence!r} was announced twice"


@pytest.mark.asyncio
async def test_a_terminal_resize_does_not_re_announce_news_the_user_read() -> None:
    """Review round 2, R2-1 / UX round 2, U2-2 — width leaked into identity.

    The round-2 head keyed the TRUNCATED sentence, so the same outcome keyed
    differently at two widths: announce at 100 columns, narrow the pane,
    re-attach, and the identical outcome announced again. The key is now the
    untruncated sentence; the card still shows the width-fitted renderable.
    This error fits a 100-column card whole but truncates at 46, so the two
    keys genuinely differed on the round-2 head.
    """
    session = _session(
        _outcome(
            connected=("github", "linear"),
            failures={"slack": _RESIZE_SENSITIVE_ERROR},
            tool_count=310,
        ),
        session_id="a",
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        # The card truncates at BOTH widths — differently, which is the point:
        # the round-2 head keyed this rendered text, so the resize changed the
        # key for an unchanged outcome.
        at_wide = toast.message
        assert "slack" in at_wide
        toast.dismiss_toast()
        await pilot.pause()

        await pilot.resize_terminal(46, 24)
        await pilot.pause()
        app._report_mcp_startup(session)
        await _quiet(pilot)
        assert toast.display is False, "a resize re-announced an unchanged outcome"


@pytest.mark.asyncio
async def test_two_failures_that_truncate_identically_both_announce() -> None:
    """Review round 2, R2-2 / UX round 2, U2-1 — the serious half.

    Truncation is lossy, so two genuinely different failures whose 50-column
    renders are byte-identical collapsed into one key on the round-2 head and
    the SECOND failure was silently swallowed — the one case this surface
    exists to raise. Keying the untruncated sentence, both announce. The card
    may still SHOW the same ellipsized text; what matters is that it is raised.
    """
    first = _session(
        _outcome(
            connected=("github", "linear"),
            failures={"slack": _TRUNCATION_COLLISION_A},
            tool_count=310,
        ),
        session_id="a",
    )
    second = _session(
        _outcome(
            connected=("github", "linear"),
            failures={"slack": _TRUNCATION_COLLISION_B},
            tool_count=310,
        ),
        session_id="b",
    )
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(50, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        assert "slack-" in toast.message  # truncated, but raised
        rendered = toast.message
        toast.dismiss_toast()
        await pilot.pause()

        app._adopt_session(second, replay_history=False)
        assert await _until(pilot, lambda: toast.display), (
            "a genuinely different failure was swallowed because it truncates "
            f"identically to the last one ({rendered!r})"
        )


@pytest.mark.asyncio
async def test_the_settled_outcome_still_announces_after_a_silent_gate() -> None:
    """The 250 ms gate snapshot is ``settling`` and unreportable, so it records
    nothing; the settled round that follows renders its own sentence and is the
    one the user actually sees. The fix must not consume the gate pass."""
    settling = McpStartupOutcome(
        configured=("github", "linear", "slack"),
        settling=True,
    )
    session = _session(settling, session_id="a")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        await _quiet(pilot)
        assert toast.display is False, "a settling snapshot must stay quiet"

        # What the manager's settle callback does: the factory rebuilds
        # ``session.mcp_startup`` with the final tally, then the app re-reports.
        session.mcp_startup = _outcome()
        app._report_mcp_startup(session)
        assert await _until(pilot, lambda: toast.display)
        assert "MCP ready: 3 servers, 425 tools" in toast.message


@pytest.mark.asyncio
async def test_a_failure_notice_is_written_once_per_session() -> None:
    """A durable failure record belongs in the transcript of the session it
    describes \u2014 so a SECOND session gets its own copy \u2014 but the repeat on the
    same session is the duplicate the re-attach used to append."""
    first = _session(_outcome(**_SLACK_DOWN), session_id="a")
    second = _session(_outcome(**_SLACK_DOWN), session_id="b")
    app = OperatorApp(lambda: _factory(first))
    async with app.run_test(size=(100, 24)) as pilot:
        assert await _until(pilot, lambda: _transcript_text(app).count("MCP slack failed") == 1)

        # Same session again: the transcript already carries the record.
        app._adopt_session(first, replay_history=False)
        await _quiet(pilot)
        assert _transcript_text(app).count("MCP slack failed") == 1

        # A different session with the same failure: its own transcript has
        # never carried it, so it gets the record even though the toast \u2014 a
        # process-wide interruption \u2014 stays silent.
        toast = app.query_one(Toast)
        toast.dismiss_toast()
        await pilot.pause()
        app._adopt_session(second, replay_history=False)
        assert await _until(pilot, lambda: _transcript_text(app).count("MCP slack failed") == 2)
        assert toast.display is False


@pytest.mark.asyncio
async def test_a_new_failure_does_not_drag_its_neighbours_back() -> None:
    """Review round 1, R1-1 \u2014 reproduced there as ``slack=2, github=1``.

    Keyed by ``(session, whole outcome)``, a later round that ADDS a failure
    changed the key, so the loop re-emitted EVERY failure in the outcome \u2014
    including ones already in the transcript. Reachable via ``/mcp reload``,
    which re-enters ``_connect_round`` and re-arms the settle. Keyed per
    ``(session, server, error)``, each distinct failure lands once per session
    and the new one arrives alone.
    """
    session = _session(_outcome(**_SLACK_DOWN), session_id="a")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        assert await _until(pilot, lambda: _transcript_text(app).count("MCP slack failed") == 1)

        # A reload round: github breaks too, slack is still broken.
        session.mcp_startup = _outcome(
            connected=("linear",),
            failures={
                "slack": "command not found: slack-mcp",
                "github": "command not found: gh",
            },
            tool_count=120,
        )
        app._report_mcp_startup(session)
        assert await _until(pilot, lambda: _transcript_text(app).count("MCP github failed") == 1)
        transcript = _transcript_text(app)
        assert (
            transcript.count("MCP slack failed") == 1
        ), "the already-recorded failure was re-emitted alongside the new one"


@pytest.mark.asyncio
async def test_a_failure_whose_error_changes_is_recorded_again() -> None:
    """The error text is part of the key because it is the actionable half.

    ``slack: command not found`` and ``slack: connection refused`` are different
    problems with different fixes, and a key on the server name alone would
    swallow the second."""
    session = _session(_outcome(**_SLACK_DOWN), session_id="a")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        assert await _until(pilot, lambda: "command not found: slack-mcp" in _transcript_text(app))
        session.mcp_startup = _outcome(
            connected=("github", "linear"),
            failures={"slack": "connection refused"},
            tool_count=310,
        )
        app._report_mcp_startup(session)
        assert await _until(pilot, lambda: "connection refused" in _transcript_text(app))
        assert _transcript_text(app).count("MCP slack failed") == 2


@pytest.mark.asyncio
async def test_the_failure_notice_points_at_the_standing_answer() -> None:
    """UX round 1, U4 \u2014 ``/mcp`` is the standing answer and it was named
    nowhere.

    It matters more now the announce is one-shot: "I'll read it next time" no
    longer works. The pointer goes on the DURABLE notice, where there is room,
    and not on the toast, whose width budget is already tight enough to
    truncate.
    """
    session = _session(_outcome(**_SLACK_DOWN), session_id="a")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        assert await _until(pilot, lambda: "MCP slack failed" in _transcript_text(app))
        assert "/mcp for details" in _transcript_text(app)
        # NOT on the toast: it is width-budgeted and already two lines.
        assert "/mcp" not in app.query_one(Toast).message


@pytest.mark.asyncio
async def test_the_announce_keeps_its_semantic_lamp() -> None:
    """Comparing the plain text must not cost the toast its colour.

    ``format_mcp_startup`` returns a ``Text`` carrying the lamp tint derived
    through the band's own rule, precisely so the two surfaces cannot disagree.
    The dedupe compares ``.plain`` — words, not styling — but what is SHOWN has
    to stay the renderable, or a failure announce paints in the healthy colour.
    """
    from textual.geometry import Region

    from local_operator.tui import theme as theme_mod

    session = _session(_outcome(**_SLACK_DOWN), session_id="a")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        # The PAINTED cells, which is the only place the lamp is observable:
        # `Toast` stores the renderable through Textual's internal content
        # accessor, so the frame is what a caller is entitled to inspect.
        painted = {
            segment.style.color.triplet.hex.lower()
            for strip in toast.render_lines(Region(0, 0, toast.size.width, toast.size.height))
            for segment in strip
            if segment.style is not None
            and segment.style.color is not None
            and segment.style.color.triplet is not None
        }
        assert theme_mod.semantic_color("danger").lower() in painted, (
            "the failure announce lost its danger lamp — it was shown as a "
            f"bare string. painted: {sorted(painted)}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("columns", [100, 80, 70, 60, 50, 44, 40])
async def test_the_notice_pointer_wraps_rather_than_truncating(columns: int) -> None:
    """The pointer is only worth adding if the user can actually read it.

    The toast was excluded from U4 precisely because it truncates; the notice
    was chosen because it WRAPS. That distinction is the whole justification
    for where the text went, so it is asserted against the painted cell grid
    at the widths the band was driven at \u2014 the rendered frame, not the block's
    source string, because only the frame shows what wrapping did to it.
    """
    session = _session(_outcome(**_SLACK_DOWN), session_id="a")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(columns, 24)) as pilot:

        def _grid() -> str:
            return "\n".join(
                "".join(segment.text for segment in strip).rstrip()
                for strip in app.screen._compositor.render_strips()
            )

        assert await _until(pilot, lambda: "MCP slack" in _grid())
        lines = [line.strip() for line in _grid().splitlines() if line.strip()]
        start = next(index for index, line in enumerate(lines) if "MCP slack" in line)
        # Three rows is the deepest this wraps at 40 columns, the narrowest
        # width the status band itself was driven at.
        painted = " ".join(lines[start : start + 3])
        assert "/mcp for details" in painted, f"the pointer was clipped at {columns}"


@pytest.mark.asyncio
async def test_the_announce_record_is_bounded_by_sessions_not_outcomes() -> None:
    """UX round 1, U5 / review round 1, R1-3 \u2014 the record's growth, re-stated
    for the per-session shape.

    The round-1 ledger grew per distinct OUTCOME: forty outcomes left forty
    entries behind, for the life of the process. The record is now one entry
    per ATTACHED session \u2014 outcome churn on one session rewrites that
    session's entry, and a re-attach adds nothing \u2014 so the bound is the
    session count (276 B measured per entry), not the outcome count. This
    asserts the shape, so a future change back to per-outcome growth fails
    here rather than in a memory profile.
    """
    session = _session(_outcome(), session_id="a")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        toast = app.query_one(Toast)
        assert await _until(pilot, lambda: toast.display)
        for index in range(40):
            toast.dismiss_toast()
            await pilot.pause()
            session.mcp_startup = _outcome(tool_count=100 + index)
            app._report_mcp_startup(session)
            assert await _until(pilot, lambda: toast.display)
        assert list(app._announced_mcp_startup) == ["a"], (
            "forty distinct outcomes on one session left more than that " "session's entry behind"
        )
        assert f"{100 + 39} tools" in app._announced_mcp_startup["a"].sentence

        # The other direction of the bound: sessions grow the record, attaches
        # do not. Five sessions is five entries; re-adopting all of them again
        # is still five.
        others = [_session(_outcome(), session_id=f"s{i}") for i in range(4)]
        for other in others:
            toast.dismiss_toast()
            await pilot.pause()
            app._adopt_session(other, replay_history=False)
            await _quiet(pilot)
        assert sorted(app._announced_mcp_startup) == ["a", "s0", "s1", "s2", "s3"]
        for again in [session, *others]:
            app._adopt_session(again, replay_history=False)
        await _quiet(pilot)
        assert sorted(app._announced_mcp_startup) == ["a", "s0", "s1", "s2", "s3"]

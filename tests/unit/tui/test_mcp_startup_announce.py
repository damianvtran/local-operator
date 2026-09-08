"""The MCP startup toast announces once per process, per distinct outcome.

Reported: "MCP ready: 12 servers, 425 tools" fired on EVERY session attach,
including every click in the session sidebar. ``session.mcp_startup`` is a
frozen BOOT SNAPSHOT and ``_report_mcp_startup`` runs on every adoption, so a
sidebar switch re-announced a round that happened minutes ago — and, through
``RemoteSession``'s rehydration of the owner's outcome, sometimes a round this
process never ran at all.

These tests pin the rule the fix implements: the toast announces only what
differs from the sentence CURRENTLY announced, so a re-attach saying the same
thing is silent (A→A) while a server that breaks, recovers and breaks again is
announced every time (A→B→A). The record is taken when the card is DISPLAYED,
not when ``show`` is called, so an announce evicted unread is not spent. The
durable failure notice is deduped per session AND per failure instead. The
status band is asserted alongside, because the suppression is only safe while
the band keeps re-stating live MCP state on every attach.
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


async def _until(pilot, predicate) -> bool:  # type: ignore[no-untyped-def]
    """Pause until ``predicate()`` holds, or the budget runs out.

    Polled rather than a fixed tick count: a wall-clock-shaped wait is what
    made the evidence script flake ~18% of the time under load (UX round 1,
    U6), and AGENTS.md's timing section says to wait on the event, not the
    clock. Returns whether it held, so a caller asserting the NEGATIVE (a toast
    that must stay silent) can still spend the full budget looking for it.
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
async def test_the_announce_record_is_one_slot_not_a_ledger() -> None:
    """UX round 1, U5 \u2014 the toast ledger grew for the life of the process.

    Forty distinct outcomes left 41 entries behind. Holding only the CURRENT
    sentence there is no growth to bound: this asserts the shape, so a future
    change back to a set fails here rather than in a memory profile.
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
        assert isinstance(app._announced_mcp_startup, str)
        assert f"{100 + 39} tools" in app._announced_mcp_startup

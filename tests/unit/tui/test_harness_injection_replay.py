"""Replay must not paint a harness-injected row as the user's own words.

A row the harness minted from a ``CustomMessage`` — the transient failover
notice a compaction pass used to bake into the transcript is the reported case —
is a plain ``Message(role="user")`` carrying the ``harness_injected`` stamp on
its ``provider_payload``. The live path never paints one (the failover moment
has its own receipt: the retry notice, the splash toast, the band), so replay
skipping it is live/replay parity rather than a second opinion — the same
doctrine the three continuation prompts follow.

This drives the TUI fold through the real app. The decision itself lives in
``harness/rows.py``; what these tests pin is that the fold ASKS it, because a
shared helper nobody calls is the drift the convergence review found.
"""

from __future__ import annotations

import pytest

from local_operator.compaction.cutpoint import (
    PRESERVED_USER_TURN_KEY,
    RENDERED_INJECTION_KEY,
)
from local_operator.harness.types import Message, TextContent
from local_operator.incidents import format_model_switch_message
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory, _transcript_text


def _injected(text: str) -> Message:
    """Exactly what ``_injected_user_message`` mints at render time."""
    return Message(
        role="user",
        content=[TextContent(text=text)],
        provider_payload={RENDERED_INJECTION_KEY: True},
    )


#: The legacy shape: a notice written before the stamp existed, carried into a
#: compaction marker's preserved block as a "user turn" and re-seated on every
#: replay with ``compaction_preserved`` and no stamp. The text is the only
#: surviving evidence of what wrote it.
_LEGACY_NOTICE = (
    "[model switch] You are now running as zai/glm-5.3 (was anthropic/claude-opus-5).\n"
    "Reason: provider failure"
)


def _carried_notice(text: str = _LEGACY_NOTICE) -> Message:
    return Message(
        role="user",
        content=[TextContent(text=text)],
        provider_payload={PRESERVED_USER_TURN_KEY: True},
    )


@pytest.mark.asyncio
async def test_a_carried_notice_copy_mounts_no_user_bubble() -> None:
    """QA Q1: the row a pre-stamp build's marker replays is not the user's.

    A real session carries eight of these, and the missing stamp is exactly why
    the reported symptom survived the stamp-scoped fix. The same negative
    control as the injected case applies: a row with the same WORDING and no
    carried marker is the operator's own (a pasted notice), and still paints.
    """
    session = FakeSession()
    session._history = [
        Message.user("why does the resume picker look empty?"),
        _carried_notice(),
    ]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._project_settled_rows(list(session._history))
        await pilot.pause()
        shown = _transcript_text(app)

    assert "why does the resume picker look empty?" in shown
    assert "[model switch]" not in shown


@pytest.mark.asyncio
async def test_a_pasted_notice_is_hidden_on_display_but_kept_in_the_transcript() -> None:
    """The documented LIMIT of the notice rule, pinned rather than discovered later.

    A person who pastes a harness notice verbatim — to ask about one, say — loses
    their display row, because the audit phase serves stored rows that carry no
    provenance at all and the text is the only thing left to decide from. The row
    is not lost anywhere else: it is still in the journal and still in the model's
    context. ``is_harness_chrome`` trades the same way for the three continuation
    prompts, which a person can equally paste.
    """
    pasted = Message.user(_LEGACY_NOTICE)
    session = FakeSession()
    session._history = [pasted]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._project_settled_rows(list(session._history))
        await pilot.pause()
        shown = _transcript_text(app)

    assert "[model switch] You are now running as zai/glm-5.3" not in shown
    # Not lost: the row itself is untouched, which is the half that matters.
    assert pasted.text.startswith("[model switch] You are now running as zai/glm-5.3")


@pytest.mark.asyncio
async def test_a_stored_notice_row_mounts_no_user_bubble() -> None:
    """QA round 2 Q1, at the fold: the AUDIT phase serves stored rows.

    A stored pre-stamp notice has no payload at all — nothing to stamp, nothing
    to carry — and the audit phase replays it verbatim, so the four such rows on
    the operator's session were painted as his own words as soon as the carried
    copies were shed. A row with no provenance is the case the text rule is for.
    """
    session = FakeSession()
    session._history = [
        Message.user("why did the model change?"),
        Message(role="user", content=[TextContent(text=_LEGACY_NOTICE)]),
    ]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._project_settled_rows(list(session._history))
        await pilot.pause()
        shown = _transcript_text(app)

    assert "why did the model change?" in shown
    assert "[model switch]" not in shown


@pytest.mark.asyncio
async def test_a_harness_injected_row_mounts_no_user_bubble() -> None:
    notice = format_model_switch_message(
        "zai/glm-5.3",
        "anthropic/claude-opus-5",
        reason="anthropic quota exhausted (0% remaining)",
        transient=True,
    )
    session = FakeSession()
    session._history = [
        _injected(notice),
        Message.user("why does the resumed transcript look empty?"),
    ]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._project_settled_rows(list(session._history))
        await pilot.pause()
        shown = _transcript_text(app)

    # The operator's own row replays exactly as it lived…
    assert "why does the resumed transcript look empty?" in shown
    # …and the notice the user never typed mounts nothing at all.
    assert "[model switch]" not in shown
    assert "You are now running as zai/glm-5.3" not in shown


@pytest.mark.asyncio
async def test_the_mcp_unavailable_row_replays_on_the_warning_tier() -> None:
    """The replay row's tier, asserted as PIXELS, and what it does NOT say.

    ``session_mcp_unavailable`` is persisted, so a resumed session folds it
    through ``project_settled_rows``. The tier is deliberate: the row is a state
    the operator must act on (``/mcp reauth <server>`` is theirs to run), which
    is the role table's own definition of ``warning`` — an earlier revision
    painted it ``note``, the receipt tier, and the design round rejected that
    (D1) because the one actionable row in the frame read as bookkeeping.

    Both rows are folded in one app on purpose: the difference between this row
    and a genuine incident is now in the TEXT, not the tier, so the test asserts
    the shared tier AND the absent false tail side by side. Asserted on token and
    glyph rather than the class name, because those are what the reader sees.
    """
    from local_operator.harness.message_types import (
        SESSION_INCIDENT_MESSAGE_TYPE,
        SESSION_MCP_UNAVAILABLE_MESSAGE_TYPE,
    )
    from local_operator.harness.types import CustomMessage
    from local_operator.incidents import (
        format_incident_message,
        format_mcp_unavailable_message,
    )
    from local_operator.tui.widgets.transcript import (
        NOTICE_GLYPHS,
        NoticeBlock,
        TranscriptView,
    )

    session = FakeSession()
    session._history = [
        CustomMessage(
            custom_type=SESSION_MCP_UNAVAILABLE_MESSAGE_TYPE,
            attribution="system",
            details={
                "text": format_mcp_unavailable_message(
                    "minerva-qa", "/mcp reauth minerva-qa — sign-in expired"
                )
            },
        ),
        CustomMessage(
            custom_type=SESSION_INCIDENT_MESSAGE_TYPE,
            attribution="system",
            details={"text": format_incident_message("429 too many requests", "test", "m")},
        ),
    ]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        # No manual fold: the app's own boot replays ``settled_rows()``, and
        # calling ``_project_settled_rows`` here as well would fold the same two
        # rows twice (the duplicate is the fixture's, not the product's).
        await pilot.pause()
        notices = [
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock)
        ]
        shown = _transcript_text(app)

    assert len(notices) == 2, f"the fold dropped a notice row: {[b._text for b in notices]}"
    warning, incident = notices
    assert warning._text.startswith("[session warning] ")
    assert warning._glyph == NOTICE_GLYPHS["warning"] == "!"
    assert warning._token == "warning", "the MCP-unavailable row is not on the warning tier"
    assert incident._token == "warning" and incident._glyph == "!"
    assert "MCP server 'minerva-qa' is unavailable" in shown
    # The frame carries the incident's false tail — for the row that IS a failed
    # turn. The capability warning must not, which is asserted on its own text
    # rather than on the frame, because both rows are painted here on purpose.
    assert "previous turn ended" not in warning._text
    assert "previous turn ended" in incident._text
    assert "until it reconnects" not in shown


@pytest.mark.asyncio
async def test_a_held_child_report_replays_as_a_warning_and_a_delivered_one_does_not() -> None:
    """UX round 1, U1/U2/U6: the held row's ARRIVAL SIGNAL, and only for held rows.

    A row written by ``Session._hold_job_results_for_next_turn`` opened no turn,
    published no outcome and painted nothing — so a resumed session read "idle,
    turn complete" while a report the operator had delegated was owed to a turn
    that had not happened yet, which is indistinguishable from the children having
    reported nothing. The phone already showed it (its generic custom-message
    fallback paints the row); the TUI's branches did not, because a
    ``job_result`` custom message with no role fell past every arm.

    BOTH ROWS ARE HERE ON PURPOSE, and the difference is the finding: the held row
    must paint, and a DELIVERED one must not — the delivered row was acknowledged
    by the turn it opened, whose answer is already in the frame, so painting it
    too would duplicate every ordinary child delivery. The two rows differ by
    exactly one flag, which is why the assertion is on the pair rather than on
    each alone.
    """
    from local_operator.harness.jobs import JOB_RESULT_MESSAGE_TYPE
    from local_operator.harness.types import CustomMessage
    from local_operator.tui.widgets.transcript import (
        NOTICE_GLYPHS,
        NoticeBlock,
        TranscriptView,
    )

    def row(job_id: str, *, held: bool) -> CustomMessage:
        details: dict[str, object] = {
            "job_id": job_id,
            "text": f"background job '{job_id}' completed:\nthe report",
        }
        if held:
            details["held"] = True
        return CustomMessage(
            custom_type=JOB_RESULT_MESSAGE_TYPE, attribution="user", details=details
        )

    session = FakeSession()
    session._history = [row("qa-r2", held=True), row("rev-r6", held=False)]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        notices = [
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock)
        ]
        shown = _transcript_text(app)

    assert len(notices) == 1, f"exactly the held row paints: {[b._text for b in notices]}"
    held = notices[0]
    assert held._token == "warning", (
        "a report waiting for a turn nobody has run is a state the operator must "
        "know about, not a receipt"
    )
    assert held._glyph == NOTICE_GLYPHS["warning"] == "!"
    # The child's own text is carried unchanged, and the notice line is what
    # distinguishes it from a delivered row.
    assert "background job 'qa-r2' completed" in held._text
    assert "held for your next turn" in held._text
    # The delivered row's own arrival is the answer it bought, which is not in
    # this history — so nothing licenses painting it here.
    assert "rev-r6" not in shown

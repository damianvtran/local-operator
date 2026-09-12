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

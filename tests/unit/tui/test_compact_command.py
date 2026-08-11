"""``/compact`` — the command that has to actually compact.

The owner's report: "/compact doesn't actually run compaction, it just echoes a
description". It printed "compaction runs automatically when the context fills
up" and changed nothing, and its registry description ("Explain context
compaction") documented that as intended.

So the tests here are about what a command owes its user: it RUNS the pass (the
same one the automatic gate runs, reached through
``SessionProtocol.compact_now``), it SAYS what the pass achieved, and when it
cannot run it says so out loud — a silent refusal is the bug wearing a different
hat.
"""

from __future__ import annotations

import pytest

from local_operator.harness.types import CompactionEndEvent, CompactionStartEvent
from local_operator.session.protocol import CompactionOutcome
from local_operator.tui.app import OperatorApp, compaction_receipt
from local_operator.tui.events import CompactionEnded
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from local_operator.tui.widgets.welcome import WelcomeView
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_slash_echo import _boot, _submit


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


class RanCompaction(FakeSession):
    """A session whose ``/compact`` succeeds, reporting a real reduction.

    It also emits the two events the real pass emits, because that pair is what
    the receipt is rendered from — a fake that only returned the outcome would
    let the notice regress unnoticed.
    """

    def __init__(self, strategy: str = "snapcompact") -> None:
        super().__init__()
        self.compact_outcome = CompactionOutcome(
            ran=True, strategy=strategy, tokens_before=128_400, tokens_after=21_900
        )

    async def compact_now(self) -> CompactionOutcome:
        outcome = self._answer_compaction()
        self.emit(CompactionStartEvent(reason="manual"))
        self.emit(
            CompactionEndEvent(
                reason="manual",
                success=True,
                strategy=outcome.strategy,
                tokens_before=outcome.tokens_before,
                tokens_after=outcome.tokens_after,
            )
        )
        return outcome


class RefusedCompaction(FakeSession):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__()
        self.compact_outcome = CompactionOutcome(ran=False, reason=reason, detail=detail)


class BrokenCompaction(FakeSession):
    async def compact_now(self) -> CompactionOutcome:
        self._answer_compaction()
        raise RuntimeError("transcript is locked")


@pytest.mark.asyncio
async def test_it_runs_the_pass_instead_of_explaining_it() -> None:
    """The reported bug. Submitting the command must reach the session's
    compaction, and the ledger must carry the pass's own notices — not a
    sentence about how compaction works."""
    session = RanCompaction()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/compact")
        for _ in range(20):
            await pilot.pause()
            if session.compactions:
                break
        await pilot.pause()
        notices = _notices(app)

    assert session.compactions == 1
    assert "compacting context…" in notices
    assert any(text.startswith("context compacted ·") for text in notices)
    # The old behaviour, gone: no explanation of the automatic trigger.
    assert not any("runs automatically" in text for text in notices)


@pytest.mark.asyncio
async def test_the_receipt_names_the_saving_and_the_strategy() -> None:
    """Compaction is slow and invisible, so the receipt has to be concrete: the
    tokens either side of the pass, the percentage that makes them readable, and
    which of the two mechanisms produced them."""
    snap = compaction_receipt(CompactionEnded("manual", True, "snapcompact", 128_400, 21_900))
    assert snap == "context compacted · 128.4k → 21.9k tokens (83% smaller), via snapcompact"

    language = compaction_receipt(CompactionEnded("manual", True, "context-full", 60_000, 30_000))
    # `context-full` is the config spelling; the receipt names the mechanism.
    assert language == "context compacted · 60.0k → 30.0k tokens (50% smaller), via summary"


@pytest.mark.asyncio
async def test_the_receipt_falls_back_when_there_are_no_figures() -> None:
    """A host that reported no tokens (or a pass that grew the history, which
    would mean the numbers are wrong) gets the bare line rather than a receipt
    quoting a saving that did not happen."""
    assert compaction_receipt(CompactionEnded("context-window", True)) == "context compacted"
    assert (
        compaction_receipt(CompactionEnded("manual", True, "snapcompact", 100, 400))
        == "context compacted"
    )


@pytest.mark.asyncio
async def test_the_band_reading_drops_by_what_the_pass_saved() -> None:
    """The band reports the next request's size, and a compaction changes only
    the history — so the saving transfers exactly. Left alone the reading would
    sit at the pre-compaction figure until the next turn, which is the "it did
    nothing" frame this command exists to stop showing."""
    session = RanCompaction()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        assert app._status is not None
        app._status.update(context_tokens=150_000, context_is_estimate=False)
        await _submit(pilot, app, "/compact")
        for _ in range(20):
            await pilot.pause()
            if session.compactions:
                break
        await pilot.pause()
        tokens = app._status.context_tokens
        estimated = app._status.context_is_estimate

    # 150 000 - (128 400 - 21 900): the delta is measured in history tokens,
    # which is the only term a compaction moves.
    assert tokens == 43_500
    # Demoted: the subtrahend came from a local estimator, not the wire.
    assert estimated is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "detail"),
    [
        (
            "turn_running",
            "a turn is still running — compaction rewrites the history the turn is holding",
        ),
        (
            "nothing_to_compact",
            "nothing to compact: the whole conversation is ~412 tokens and the most recent "
            "20,000 are kept verbatim",
        ),
        ("nothing_to_compact", "nothing left to compact: everything older is already summarized"),
        ("disabled", "compaction is switched off in config (values.compaction)"),
    ],
)
async def test_every_refusal_is_said_out_loud(reason: str, detail: str) -> None:
    """Each state a manual trigger can be pressed in that the automatic gate
    never sees. The command must not run — and must not look like it did
    nothing, which is the failure being fixed: the session's sentence is printed
    verbatim, so one wording serves every host."""
    session = RefusedCompaction(reason, detail)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/compact")
        for _ in range(20):
            await pilot.pause()
            if session.compactions:
                break
        await pilot.pause()
        notices = _notices(app)

    assert detail in notices
    # No pass notices: nothing ran, so nothing may claim it did.
    assert "compacting context…" not in notices
    assert not any(text.startswith("context compacted") for text in notices)


@pytest.mark.asyncio
async def test_it_refuses_before_the_session_exists() -> None:
    """Pressed as the first action, while boot is still in flight: there is no
    context to compact, and saying so must leave the boot composition standing —
    a command that changed nothing has not started the conversation."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._session = None
        app._run_slash_command("/compact")
        await pilot.pause()
        notices = _notices(app)
        welcome_visible = app.query_one(WelcomeView).display

    assert notices == ["no session yet — there is no context to compact"]
    assert welcome_visible is True


@pytest.mark.asyncio
async def test_a_crashing_pass_is_reported_not_swallowed() -> None:
    """``compact_now`` raising must reach the user as an error, not vanish into a
    worker traceback — the whole point of the command is that its effect is
    visible."""
    session = BrokenCompaction()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/compact")
        for _ in range(20):
            await pilot.pause()
            if session.compactions:
                break
        await pilot.pause()
        notices = _notices(app)

    assert "compaction failed: transcript is locked" in notices


@pytest.mark.asyncio
async def test_a_second_press_is_refused_by_the_session_not_by_cancelling_the_first() -> None:
    """Concurrency is the SESSION's call (``already_running``), and the second
    press renders that refusal.

    The tempting alternative — an exclusive worker — cancels the running pass,
    which would leave ``compacting context…`` on screen with no end event to
    retire it. So both presses reach the session, and the loser is told why.
    """
    session = RefusedCompaction("already_running", "a compaction is already running")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._cmd_compact()
        app._cmd_compact()
        for _ in range(20):
            await pilot.pause()
        notices = _notices(app)

    assert session.compactions == 2
    assert notices.count("a compaction is already running") == 2

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

import asyncio

import pytest
from textual import events

from local_operator.harness.types import CompactionEndEvent, CompactionStartEvent
from local_operator.session.protocol import CompactionOutcome
from local_operator.tui.app import OperatorApp, compaction_receipt
from local_operator.tui.events import (
    CompactionEnded,
    TurnBoundaryEnd,
    TurnEnded,
    TurnStarted,
)
from local_operator.tui.widgets.editor import resolve_markers
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView, UserBlock
from local_operator.tui.widgets.welcome import WelcomeView
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_slash_echo import _boot, _submit
from tests.unit.tui.test_working_line import _started


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


class FailedCompaction(FakeSession):
    """A pass that STARTED and blew up: end event, then ``reason="failed"``.

    Distinct from :class:`BrokenCompaction`, where ``compact_now`` itself
    raises. This is the real shape — ``Session._run_compaction`` catches, emits
    ``CompactionEndEvent(success=False)`` and returns the outcome — and it is
    the one that produced two notices for one event.
    """

    def __init__(self) -> None:
        super().__init__()
        self.compact_outcome = CompactionOutcome(
            ran=False, reason="failed", detail="compaction failed: summarizer returned nothing"
        )

    async def compact_now(self) -> CompactionOutcome:
        outcome = self._answer_compaction()
        self.emit(CompactionStartEvent(reason="manual"))
        self.emit(CompactionEndEvent(reason="manual", success=False))
        return outcome


class SlowCompaction(FakeSession):
    """Announces the pass, then waits — the multi-minute window a user types in."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.compact_outcome = CompactionOutcome(
            ran=True, strategy="context-full", tokens_before=90_000, tokens_after=30_000
        )

    async def compact_now(self) -> CompactionOutcome:
        outcome = self._answer_compaction()
        self.emit(CompactionStartEvent(reason="manual"))
        await self.release.wait()
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


@pytest.mark.asyncio
async def test_a_prompt_typed_during_a_pass_is_held_and_then_sent() -> None:
    """The pass holds the turn lock, and ``is_streaming`` is False for it — so
    without the hold the submit path called ``prompt()``, which raises while the
    history is being rewritten: the user's row sat in the ledger as if sent and
    the text was gone. Same failure the steer branch prevents for a running
    turn, one lock holder over.
    """
    session = SlowCompaction()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._cmd_compact()
        for _ in range(30):
            await pilot.pause()
            if app._compacting:
                break
        await _submit(pilot, app, "and now analyse the parser")
        held = app._prompt_held_for_compaction
        queued = _notices(app)
        sent_during = list(session.prompts)

        session.release.set()
        for _ in range(60):
            await pilot.pause()
            if session.prompts:
                break
        sent_after = list(session.prompts)
        rows = [
            block.text()
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, UserBlock)
        ]

    assert held == "and now analyse the parser"
    assert "queued — sends when compaction finishes" in queued
    assert sent_during == []  # nothing reached the session mid-pass
    assert sent_after == ["and now analyse the parser"]  # and nothing was lost
    # Exactly ONE user row: the hold must not write the text into the ledger twice.
    assert rows.count("and now analyse the parser") == 1


@pytest.mark.asyncio
async def test_a_reload_during_a_pass_does_not_brick_the_app() -> None:
    """``/reload`` mid-compaction must not leave the hold armed forever.

    ``_compacting`` was cleared in exactly one place, ``on_compaction_ended``
    — which can never run after a reload, because ``_reload_session`` disposes
    the controller first, by design, so the dying session's terminal events are
    dropped. Slash commands are not gated on ``_compacting`` and a manual pass
    is minutes long, so this is an ordinary thing to do. Left set, the flag
    stuck True for the life of the app: every later prompt was swallowed into
    the hold and answered "queued", and nothing ever sent it. A fully booted
    session could no longer reach the model at all.
    """
    session = SlowCompaction()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._cmd_compact()
        for _ in range(30):
            await pilot.pause()
            if app._compacting:
                break
        assert app._compacting, "the pass never started, so this proves nothing"

        # ARM the hold. Without this the test only covers `_compacting` and the
        # held-prompt assertion below is vacuously true — which is exactly how
        # the first version of this test passed with that line removed.
        await _submit(pilot, app, "summarise the parser work above")
        assert app._prompt_held_for_compaction == "summarise the parser work above"
        assert session.prompts == [], "the hold did not intercept it"

        # The dying session never emits its end event: disposal drops it.
        replacement = SlowCompaction()
        app._session_factory = lambda: _factory(replacement)  # type: ignore[assignment]
        await app._reload_session()
        for _ in range(30):
            await pilot.pause()

        assert not app._compacting, "the hold outlived the session that armed it"
        assert app._prompt_held_for_compaction == "", "the held prompt survived"
        # Handed back, not destroyed: the app had already promised the user it
        # was queued, and `clear_blocks` took both that promise and their echo.
        assert app._editor().text == "summarise the parser work above"

        # Run a pass in the REPLACEMENT session. If the stale hold survived,
        # its `on_compaction_ended` drains it and sends the old text here.
        app._cmd_compact()
        for _ in range(30):
            await pilot.pause()
            if app._compacting:
                break
        replacement.release.set()
        for _ in range(40):
            await pilot.pause()
        assert replacement.prompts == [], "text from the dead conversation was sent"

        # And the real proof: the app can still talk to the model.
        app._editor().load_text("")
        await _submit(pilot, app, "hello again")
        for _ in range(60):
            await pilot.pause()
            if replacement.prompts:
                break

    assert replacement.prompts == ["hello again"], "the prompt was swallowed"


@pytest.mark.asyncio
async def test_a_reload_does_not_carry_an_interrupted_count_into_the_next_session() -> None:
    """The per-turn card count must not suppress the NEXT session's notice.

    ``_interrupted_cards`` exists so an aborted turn does not print a
    standalone "interrupted" row on top of cards that each already say so. It
    is cleared only by ``on_turn_ended``, which the dying session never
    reaches — so a reload while cards were interrupted carried the count into
    the replacement, and its first aborted turn silently printed nothing on
    the strength of cards belonging to a conversation the user cannot see.
    """
    session = SlowCompaction()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        # A turn with a live tool call, stopped: the count arms.
        app.post_message(TurnStarted())
        app.post_message(_started("c0", "bash", command="sleep 600"))
        await pilot.pause()
        app.post_message(TurnBoundaryEnd())
        await pilot.pause()
        assert app._interrupted_cards > 0, "no cards were interrupted, so this proves nothing"

        replacement = SlowCompaction()
        app._session_factory = lambda: _factory(replacement)  # type: ignore[assignment]
        await app._reload_session()
        for _ in range(20):
            await pilot.pause()

        # An aborted turn in the NEW session with nothing in flight: the
        # standalone notice is the only thing that can say it stopped.
        app.post_message(TurnStarted())
        await pilot.pause()
        app.post_message(TurnEnded(aborted=True, error=None))
        for _ in range(20):
            await pilot.pause()

        assert "interrupted" in _notices(app), "the stale count swallowed the notice"


@pytest.mark.asyncio
async def test_the_automatic_pass_gets_the_same_receipt_and_band_update() -> None:
    """The property the whole design claims: the two triggers differ only in what
    starts them. An automatic pass emits the same events, so it gets the same
    receipt line and the same band correction — reached without ``/compact``
    being typed at all."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        assert app._status is not None
        app._status.update(context_tokens=150_000, context_is_estimate=False)
        session.emit(CompactionStartEvent(reason="context-window"))
        session.emit(
            CompactionEndEvent(
                reason="context-window",
                success=True,
                strategy="context-full",
                tokens_before=120_000,
                tokens_after=40_000,
            )
        )
        for _ in range(20):
            await pilot.pause()
        notices = _notices(app)
        tokens = app._status.context_tokens

    assert "compacting context…" in notices
    assert "context compacted · 120.0k → 40.0k tokens (67% smaller), via summary" in notices
    assert tokens == 70_000  # 150 000 - 80 000
    assert session.compactions == 0  # nobody asked; the gate fired


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
async def test_the_band_reading_never_falls_below_the_history_that_is_left() -> None:
    """A live run made this necessary, twice: the band read 12.3k on a vision
    session and ZERO on a text one.

    Compaction's local ruler runs above a provider's own count, so the estimated
    saving can exceed the provider figure it is subtracted from — and a band
    claiming an empty context immediately after a compaction is a worse lie than
    a stale one. The kept history is a floor the next request cannot go under.
    """
    session = RanCompaction()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        assert app._status is not None
        # Smaller than the 106 500-token saving the pass reports.
        app._status.update(context_tokens=33_000, context_is_estimate=False)
        await _submit(pilot, app, "/compact")
        for _ in range(20):
            await pilot.pause()
            if session.compactions:
                break
        await pilot.pause()
        tokens = app._status.context_tokens

    assert tokens == 21_900  # the pass's own tokens_after, not 0


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
async def test_a_failed_pass_says_why_once_and_says_it_as_an_error() -> None:
    """One event, one notice, at the severity the event deserves.

    A ``reason="failed"`` outcome went through the same ``not outcome.ran``
    branch as a refusal, so it printed ``warning`` — UNDER the end event's bare
    ``compaction failed`` in ``error``. Two notices for one failure, the empty
    one louder than the one carrying the reason, and both saying the same thing
    with different urgency. ``failed`` is not a refusal: the pass ran.
    """
    session = FailedCompaction()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/compact")
        for _ in range(20):
            await pilot.pause()
            if session.compactions:
                break
        await pilot.pause()
        blocks = [
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock) and "compaction failed" in block._text
        ]

    assert [block._text for block in blocks] == ["compaction failed: summarizer returned nothing"]
    assert blocks[0]._token == NoticeBlock._KIND_TOKENS["error"]


@pytest.mark.asyncio
async def test_an_automatic_failure_still_says_so() -> None:
    """The manual case is narrated by the worker; the automatic one has nobody
    else, so suppressing the end event's line for BOTH would have swallowed the
    only report a background pass ever makes."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app.on_compaction_ended(CompactionEnded(reason="context-window", success=False))
        await pilot.pause()
        notices = _notices(app)

    assert "compaction failed" in notices


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


@pytest.mark.asyncio
async def test_a_prompt_held_for_a_pass_keeps_its_attachments(tmp_path) -> None:
    """Review round 19, P1. The hold has to capture the images, not re-read them.

    `Editor._submit` clears the composer synchronously right after posting and
    Textual delivers on a later tick, so a handler that asked the widget for
    its attachments got an empty map. The app announced "queued - sends when
    compaction finishes" and then sent the words alone - which the comment at
    that very branch calls worse than not queueing at all.
    """
    from PIL import Image

    from local_operator.tui.widgets.editor import Editor

    path = tmp_path / "a.png"
    Image.new("RGB", (30, 40), "red").save(path)

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.25)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        app.post_message(events.Paste(str(path)))
        await pilot.pause()
        await pilot.pause()
        assert len(editor.referenced_images()) == 1, "the fixture never attached"

        editor.insert("look at this")
        app._compacting = True
        await pilot.press("enter")
        await pilot.pause()

        assert app._prompt_held_for_compaction, "the prompt was not held"
        assert app._images_held_for_compaction, "the hold dropped the screenshot"
        held = resolve_markers(app._prompt_held_for_compaction, app._images_held_for_compaction)
        assert len(held) == 1, "the held prompt would have sent no image"

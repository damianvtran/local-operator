"""Real styled TUI geometry with a controlled host-probe boundary.

These are render-policy tests, not evidence that the OS focused a real window.
Native focus evidence is captured separately using the read-only host protocol.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest
from textual.events import AppBlur, AppFocus
from textual.screen import Screen

from local_operator.harness.types import Message, TextContent
from local_operator.session.attention import AttentionStore
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory


class ReceiptSession(FakeSession):
    def __init__(self, path: Path, *, long: bool = False) -> None:
        super().__init__()
        self.store = AttentionStore(path)
        self.token = str(uuid.uuid4())
        text = "Finished result\n\n" + (
            "A long result line\n\n" * 90 if long else "The final answer is visible."
        )
        self.result = Message(role="assistant", content=[TextContent(text=text)])
        self._history = [self.result]
        self.store.publish("session/sess", self.token, self.result.id, "complete")

    async def refresh_attention(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.store.state, "session/sess")

    async def acknowledge_attention(self, token: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.store.acknowledge, "session/sess", token)


@pytest.mark.asyncio
async def test_default_focus_is_not_proof_and_rendered_focus_acknowledges(
    tmp_path, monkeypatch
) -> None:
    session = ReceiptSession(tmp_path / "attention.db")
    probes: list[bool] = []
    monkeypatch.setattr(
        "local_operator.tui.attention.terminal_is_foreground", lambda: probes.append(True) or True
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        # Boot history is mounted by a worker, then laid out on a later frame.
        # Wait on the actual geometry, not a fixed wall-clock delay.
        for _ in range(50):
            await pilot.pause()
            if app._completion_anchor_visible(session.result.id):
                break
        assert app._completion_anchor_visible(session.result.id)
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["unseen"]
        assert not probes
        app.on_app_focus(AppFocus())
        await app._poll_completion_attention()
        assert not session.store.state("session/sess")["unseen"]
        count = len(probes)
        await app._poll_completion_attention()
        assert len(probes) == count


class FencedReceiptSession(ReceiptSession):
    """A surface stuck on a SUPERSEDED completion, on a backend that no-ops.

    The live shape from the findings file: the receipt is pinned to an older
    completion while a newer one is unseen, and the surface still renders the
    OLDER one (its owner's projection has not caught up), so the token it holds
    is stale. `acknowledge_attention` answers the way the shipped 0.54.43 daemon
    did -- a resolved call whose state still says `unseen` -- which is the answer
    a client must not believe.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.stale = self.token
        self.newer = Message(role="assistant", content=[TextContent(text="The newer result.")])
        self.fresh = str(uuid.uuid4())
        self._history = [self.result, self.newer]
        self.store.publish("session/sess", self.fresh, self.newer.id, "complete")
        self.acked: list[str] = []
        self.serve_stale = True
        # Set when the answer should move PAST the token the surface sent: the
        # one reading that is evidence of a lost receipt rather than ambiguity.
        self.ack_moves_past = False

    async def refresh_attention(self) -> dict[str, Any]:
        state = await asyncio.to_thread(self.store.state, "session/sess")
        if self.serve_stale:
            # The projection behind the surface has not caught up: it still
            # renders the older completion, which is how a real client came to
            # send a superseded token at all.
            return {
                **state,
                "completion_token": self.stale,
                "anchor_id": self.result.id,
                "unseen": True,
            }
        return state

    async def acknowledge_attention(self, token: str) -> dict[str, Any]:
        self.acked.append(token)
        if token == self.stale:
            state = await asyncio.to_thread(self.store.state, "session/sess")
            return {
                **state,
                "completion_token": self.fresh if self.ack_moves_past else self.stale,
                "anchor_id": self.newer.id if self.ack_moves_past else self.result.id,
                "unseen": True,
            }
        return await asyncio.to_thread(self.store.acknowledge, "session/sess", token)


async def _settle(app: Any, pilot: Any, anchor: str) -> None:
    """Pump until the anchored block is laid out and hit-testable."""
    for _ in range(50):
        await pilot.pause()
        if app._completion_anchor_visible(anchor):
            return


@pytest.mark.asyncio
async def test_a_rendered_result_is_receipted_without_a_focus_report(tmp_path, monkeypatch) -> None:
    """The reported TUI defect: a terminal that never tells us it has focus.

    Textual learns focus from the terminal's own focus REPORTS, and a terminal
    that was already focused when those reports were enabled sends none -- so an
    app that starts focused never sets the latch `on_app_focus` owns, and the
    honest attempt (`_completion_anchor_visible` says every guard passes) was
    refused by a gate nothing could open. Where the host probe actually MEASURES
    focus, the same evidence may be re-taken on a cadence instead of an edge.
    """
    session = ReceiptSession(tmp_path / "attention.db")
    probes: list[bool] = []
    monkeypatch.setattr(
        "local_operator.tui.attention.focus_is_measurable", lambda *a, **k: True, raising=False
    )
    monkeypatch.setattr(
        "local_operator.tui.attention.terminal_is_foreground",
        lambda *a, **k: probes.append(True) or True,
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot, session.result.id)
        assert app._completion_anchor_visible(session.result.id), "the result is not on screen"
        assert not getattr(app, "_attention_focus_observed", False), "no report has arrived"

        assert session.store.state("session/sess")["unseen"]
        await app._poll_completion_attention()
        await pilot.pause()

        assert probes, "the poll refused without asking the host for evidence"
        assert not session.store.state("session/sess")[
            "unseen"
        ], "a rendered result on a foreground terminal was not receipted"


@pytest.mark.asyncio
async def test_a_terminal_that_cannot_measure_focus_still_waits_for_a_report(
    tmp_path, monkeypatch
) -> None:
    """The fence, where the probe is not a measurement (a plain terminal).

    There the probe's True says only that no `CMUX_*` variable is set, so it
    carries no information about what the user is looking at. Startup's
    optimistic Textual focus is not evidence either, so the receipt still waits
    for a real focus report -- and still takes it when one arrives.
    """
    session = ReceiptSession(tmp_path / "attention.db")
    probes: list[bool] = []
    monkeypatch.setattr(
        "local_operator.tui.attention.focus_is_measurable", lambda *a, **k: False, raising=False
    )
    monkeypatch.setattr(
        "local_operator.tui.attention.terminal_is_foreground",
        lambda *a, **k: probes.append(True) or True,
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot, session.result.id)
        assert app._completion_anchor_visible(session.result.id)
        for _ in range(3):
            await app._poll_completion_attention()
            await pilot.pause()
        assert session.store.state("session/sess")[
            "unseen"
        ], "an unmeasurable terminal was receipted without any focus report"
        assert not probes, "an unmeasurable probe was asked for focus evidence"

        app.on_app_focus(AppFocus())
        await app._poll_completion_attention()
        await pilot.pause()
        assert not session.store.state("session/sess")["unseen"]


@pytest.mark.asyncio
async def test_input_on_a_terminal_that_cannot_measure_focus_receipts_the_result(
    tmp_path, monkeypatch
) -> None:
    """The reported symptom, on a terminal the host probe cannot measure.

    The PORTABLE half of the fix, and the terminal the TUI defect was reported
    on. The fence above deliberately refuses to trust the probe where it is not
    a measurement, and a terminal that was already focused when Textual enabled
    focus reporting never sends the edge that would open it -- so the receipt
    waited forever for a report that could not arrive. A key or mouse-down this
    app RECEIVES can only have been delivered to a focused terminal, which is
    the same proof, available everywhere: it is what opens the gate here, with
    no subprocess and no cmux.
    """
    session = ReceiptSession(tmp_path / "attention.db")
    probes: list[bool] = []
    monkeypatch.setattr(
        "local_operator.tui.attention.focus_is_measurable", lambda *a, **k: False, raising=False
    )
    monkeypatch.setattr(
        "local_operator.tui.attention.terminal_is_foreground",
        lambda *a, **k: probes.append(True) or True,
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot, session.result.id)
        assert app._completion_anchor_visible(session.result.id)

        # Control: no input, no report, nothing measurable -> nothing is read.
        for _ in range(3):
            await app._poll_completion_attention()
            await pilot.pause()
        assert session.store.state("session/sess")["unseen"]

        # One real input edge: the ordinary act of opening the conversation.
        await pilot.press("a")
        assert getattr(
            app, "_attention_input_at", 0.0
        ), "a key the app received did not stamp an input edge"
        await app._poll_completion_attention()
        await pilot.pause()
        assert not session.store.state("session/sess")[
            "unseen"
        ], "input observed on an unmeasurable terminal did not receipt the rendered result"
        assert not probes, "the input edge must not need the host probe"


@pytest.mark.asyncio
async def test_input_evidence_does_not_stand_forever(tmp_path, monkeypatch) -> None:
    """The shelf life, stated because the fence depends on it.

    A terminal that cannot report focus cannot report blur either, so an
    unbounded input latch would keep marking LATER completions read with nobody
    at the screen -- trading a false negative for a false positive. The window
    is what bounds it: evidence that has expired is not evidence, and one act
    (a keystroke, a click) re-arms it.
    """
    from local_operator.tui.attention import ATTENTION_INPUT_EVIDENCE_S

    session = ReceiptSession(tmp_path / "attention.db")
    monkeypatch.setattr(
        "local_operator.tui.attention.focus_is_measurable", lambda *a, **k: False, raising=False
    )
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda *a, **k: True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot, session.result.id)
        await pilot.press("a")
        # Age the edge past its window: an old act is not evidence about now.
        app._attention_input_at -= ATTENTION_INPUT_EVIDENCE_S + 1
        await app._poll_completion_attention()
        await pilot.pause()
        assert session.store.state("session/sess")[
            "unseen"
        ], "expired input evidence was still treated as a watchful terminal"

        await pilot.press("b")
        await app._poll_completion_attention()
        await pilot.pause()
        assert not session.store.state("session/sess")["unseen"]


@pytest.mark.asyncio
async def test_the_focus_probe_is_bounded_to_one_per_window(tmp_path, monkeypatch) -> None:
    """The cost claim the fence is justified by, pinned (agent review R5).

    Two polls inside :data:`ATTENTION_FOCUS_REFRESH_S` may not each shell out to
    the host, or "one probe per half minute" is not true and a per-tick
    regression would pass the whole suite. A due tick must ask AGAIN, which is
    what keeps a refusal from shutting the gate for good.
    """
    from local_operator.tui.app import ATTENTION_FOCUS_REFRESH_S

    session = ReceiptSession(tmp_path / "attention.db")
    probes: list[bool] = []
    monkeypatch.setattr(
        "local_operator.tui.attention.focus_is_measurable", lambda *a, **k: True, raising=False
    )
    monkeypatch.setattr(
        "local_operator.tui.attention.terminal_is_foreground",
        lambda *a, **k: probes.append(True) or False,
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot, session.result.id)
        for _ in range(3):
            await app._poll_completion_attention()
            await pilot.pause()
        assert len(probes) == 1, f"an unlatched poll probed {len(probes)} times inside one window"
        assert session.store.state("session/sess")["unseen"], "a background terminal was receipted"

        # The window elapses (the probe's own stamp is the clock): asked again.
        app._attention_focus_probe_at -= ATTENTION_FOCUS_REFRESH_S + 1
        await app._poll_completion_attention()
        await pilot.pause()
        assert len(probes) == 2, "a due tick did not re-ask, so a refusal would never heal"


@pytest.mark.asyncio
async def test_a_background_terminal_is_never_receipted(tmp_path, monkeypatch) -> None:
    """Measured and NOT in front: nothing is read, however long the poll runs."""
    session = ReceiptSession(tmp_path / "attention.db")
    monkeypatch.setattr(
        "local_operator.tui.attention.focus_is_measurable", lambda *a, **k: True, raising=False
    )
    monkeypatch.setattr(
        "local_operator.tui.attention.terminal_is_foreground", lambda *a, **k: False
    )
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot, session.result.id)
        assert app._completion_anchor_visible(session.result.id)
        for _ in range(3):
            await app._poll_completion_attention()
            await pilot.pause()
        # Even a real focus report does not override the host's own answer that
        # this terminal is not the frontmost one.
        app.on_app_focus(AppFocus())
        for _ in range(3):
            await app._poll_completion_attention()
            await pilot.pause()
        assert session.store.state("session/sess")["unseen"]


@pytest.mark.asyncio
async def test_a_lagging_projection_is_inconclusive_rather_than_a_verdict(
    tmp_path, monkeypatch, caplog
) -> None:
    """A follower's own answer cannot prove a receipt was lost (agent review R4).

    On the transport the operator's TUI actually runs on -- an ATTACHED session
    -- the owner publishes its projection asynchronously while the op replies
    out of the follower's last-applied state, so "still unseen, still my token"
    is equally consistent with one tick of push lag (the honest path) and with
    an owner that did nothing. The client must not slander the honest one: no
    verdict, the attempt repeats on the next tick, and the receipt lands as soon
    as the projection names the current token -- all against a backend that
    really does answer a no-op with success, so "it did not latch" is the
    property under test rather than "it converged by luck".
    """
    session = FencedReceiptSession(tmp_path / "attention.db")
    monkeypatch.setattr(
        "local_operator.tui.attention.focus_is_measurable", lambda *a, **k: True, raising=False
    )
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda *a, **k: True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot, session.result.id)
        assert app._completion_anchor_visible(session.result.id)
        # The focus edge, deliberately: this test is about the VERIFICATION of
        # what an acknowledgement returns, so the focus path must not be what it
        # depends on.
        app.on_app_focus(AppFocus())
        await pilot.pause()

        with caplog.at_level("DEBUG", logger="local_operator.tui.app"):
            await app._poll_completion_attention()
            await pilot.pause()

        assert session.acked == [session.stale], "the stale token was not the one attempted"
        assert session.store.state("session/sess")[
            "unseen"
        ], "a no-op acknowledgement was taken as a read"
        assert "completion receipt" not in caplog.text, (
            "an answer that still named the token we sent produced a receipt verdict; push lag and "
            "a no-op are indistinguishable on a follower, so there is nothing to say about it"
        )

        # The projection catches up. The receipt advances on the token it now
        # names, without any new focus evidence and without a restart.
        session.serve_stale = False
        await app._poll_completion_attention()
        await pilot.pause()
        assert session.acked[-1] == session.fresh
        assert not session.store.state("session/sess")["unseen"]


@pytest.mark.asyncio
async def test_a_receipt_that_raced_a_newer_completion_says_so(
    tmp_path, monkeypatch, caplog
) -> None:
    """The one answer that IS evidence: the state moved past the token we sent.

    Here the reason is narrow and nameable -- the completion receipted is no
    longer the one the conversation asks about -- and the log line says exactly
    that instead of accusing the transport of losing a receipt.
    """
    session = FencedReceiptSession(tmp_path / "attention.db")
    session.ack_moves_past = True
    monkeypatch.setattr(
        "local_operator.tui.attention.focus_is_measurable", lambda *a, **k: True, raising=False
    )
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda *a, **k: True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle(app, pilot, session.result.id)
        app.on_app_focus(AppFocus())
        await pilot.pause()

        with caplog.at_level("DEBUG", logger="local_operator.tui.app"):
            await app._poll_completion_attention()
            await pilot.pause()

        assert session.acked == [session.stale]
        assert (
            "raced a newer completion" in caplog.text
        ), "an answer that had moved past the rendered token passed silently"
        assert session.store.state("session/sess")["unseen"]


@pytest.mark.asyncio
@pytest.mark.parametrize("follow", [False, True])
@pytest.mark.parametrize("already_read", [False, True])
async def test_old_failure_is_not_inserted_at_a_new_retry_tail(
    tmp_path, follow, already_read
) -> None:
    from local_operator.harness.types import StreamEndEvent
    from local_operator.paths import config_dir
    from local_operator.session.attached import AttachedSession
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import ServingSessionHandle
    from local_operator.tui.widgets.transcript import NoticeBlock, UserBlock
    from tests.unit.session.test_session import make_session

    calls = 0
    started, release = asyncio.Event(), asyncio.Event()

    async def stream(request, signal):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield StreamEndEvent(stop_reason="error", error="Old failure")
            return
        started.set()
        await release.wait()
        # A quiet settled retry leaves the older unread outcome authoritative;
        # it still must not acquire a newly appended historical-error marker.
        yield StreamEndEvent(stop_reason="stop")

    session = make_session(tmp_path, stream)
    await session.prompt("Old request")
    old = await session.refresh_attention()
    if already_read:
        await session.acknowledge_attention(old["completion_token"])
    runtime = RuntimeServer(
        ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path)), kind="daemon"
    )
    await runtime.start_in_process()
    retry = asyncio.create_task(session.prompt("New retry is running"))
    await asyncio.wait_for(started.wait(), 10)
    source: Any = session

    async def never(*args, **kwargs):
        raise AssertionError("live fixture must not take over")

    try:
        if follow:
            source = await AttachedSession.connect(
                runtime._record, session.session_id, config_dir=config_dir(), takeover_factory=never
            )

        async def factory():
            return source

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(50):
                await pilot.pause()
                if app._session is source and len(app._transcript_view().query(UserBlock)) > 0:
                    break
            assert source.is_streaming
            await app._poll_completion_attention()
            await pilot.pause()
            assert not any(
                block.completion_anchor_id == old["anchor_id"]
                for block in app._transcript_view().query(NoticeBlock)
            )
            release.set()
            await retry
            for _ in range(50):
                await pilot.pause()
                if not source.is_streaming:
                    break
            await app._poll_completion_attention()
            assert not any(
                block.completion_anchor_id == old["anchor_id"]
                for block in app._transcript_view().query(NoticeBlock)
            )
    finally:
        release.set()
        await asyncio.gather(retry, return_exceptions=True)
        if follow:
            await source.dispose()
        await runtime.aclose()
        await session.dispose()


@pytest.mark.asyncio
async def test_overlay_scrollback_and_blur_do_not_acknowledge(tmp_path, monkeypatch) -> None:
    session = ReceiptSession(tmp_path / "attention.db", long=True)
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(50):
            await pilot.pause()
            if app._completion_anchor_visible(session.result.id):
                break
        assert app._completion_anchor_visible(session.result.id)
        app.on_app_focus(AppFocus())
        app._transcript_view().scroll_home(animate=False)
        await pilot.pause()
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["unseen"]
        app._transcript_view().scroll_end(animate=False)
        app.push_screen(Screen())
        await pilot.pause()
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["unseen"]
        app.on_app_blur(AppBlur())
        app.pop_screen()
        await pilot.pause()
        await app._poll_completion_attention()
        assert session.store.state("session/sess")["unseen"]


class InterruptSession(FakeSession):
    """An attention-backed fake with nothing published until a test says so.

    ``ReceiptSession`` publishes a COMPLETE outcome in its constructor. The
    duplicate-row defect needs the opposite order — the app paints its own
    live row first, and the durable ``interrupted`` outcome is published
    afterwards, exactly as a real session does (it mints
    ``completion-<token>`` when it publishes, which is strictly after the turn
    has ended on screen).
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.store = AttentionStore(path)
        self.identity = "session/interrupted"

    def publish_interrupted(self) -> str:
        token = str(uuid.uuid4())
        self.store.publish(self.identity, token, f"completion-{token}", "interrupted")
        return token

    async def refresh_attention(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.store.state, self.identity)

    async def acknowledge_attention(self, token: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.store.acknowledge, self.identity, token)


async def _interrupt_a_turn(app: OperatorApp, pilot: Any) -> None:
    """Run a turn to the point where the app has painted its own abort row."""
    from local_operator.tui.events import TurnEnded, TurnStarted
    from local_operator.tui.widgets.editor import Editor

    for _ in range(200):
        if app._session is not None:
            break
        await pilot.pause()
        await asyncio.sleep(0.01)
    editor = app.query_one(Editor)
    editor.focus()
    editor.text = "run the long job"
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    app.post_message(TurnStarted())
    await pilot.pause()
    app.post_message(TurnEnded(True, None))
    await pilot.pause()
    await pilot.pause()


def _notice_texts(app: OperatorApp) -> list[str]:
    from local_operator.tui.widgets.transcript import NoticeBlock

    return [
        block._text for block in app._transcript_view().blocks() if isinstance(block, NoticeBlock)
    ]


@pytest.mark.asyncio
async def test_one_interruption_is_stated_once(tmp_path) -> None:
    """Two producers, one outcome, one row.

    ``_finalize_turn`` announces the abort live; the attention poller reads the
    same interruption back out of the durable store a tick later and appended a
    SECOND row, because its only dedupe is ``completion_anchor_id`` and the
    live row cannot carry one (the anchor does not exist until the session
    publishes). The user saw ``! interrupted`` above ``· Interrupted``.
    """
    from local_operator.tui.widgets.transcript import NoticeBlock

    session = InterruptSession(tmp_path / "attention.db")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _interrupt_a_turn(app, pilot)
        assert _notice_texts(app) == ["interrupted"], "the turn states its own outcome"

        # Now the session publishes that same interruption durably.
        token = session.publish_interrupted()
        await app._poll_completion_attention()
        await pilot.pause()

        assert _notice_texts(app) == ["interrupted"], "and it is not restated"
        # The live row ADOPTED the anchor rather than being shadowed by a
        # second one, which is what lets looking at it mark the outcome read.
        anchor = session.store.state(session.identity)["anchor_id"]
        assert [
            block.completion_anchor_id
            for block in app._transcript_view().query(NoticeBlock)
            if block.completion_anchor_id
        ] == [anchor]
        assert token


@pytest.mark.asyncio
async def test_an_interruption_this_app_never_painted_still_gets_a_row(tmp_path) -> None:
    """The poller keeps the case it exists for: a session that stopped away.

    Proves the fix suppresses a DUPLICATE, not the attention notice itself —
    with no live row to adopt, a user returning to the session must still be
    told it was interrupted.
    """
    session = InterruptSession(tmp_path / "attention.db")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(200):
            if app._session is not None:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        # No turn ran in THIS app; the outcome was produced elsewhere.
        session.publish_interrupted()
        await app._poll_completion_attention()
        await pilot.pause()

        assert _notice_texts(app) == ["Interrupted"]


@pytest.mark.asyncio
async def test_a_later_outcome_never_adopts_an_earlier_turns_row(tmp_path) -> None:
    """The held row is turn-scoped: a new turn makes it unadoptable.

    Otherwise a second interruption would stamp its anchor onto the FIRST
    turn's row — marking an old outcome read while the new one gets no row at
    all.
    """
    from local_operator.tui.events import TurnStarted

    session = InterruptSession(tmp_path / "attention.db")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _interrupt_a_turn(app, pilot)
        assert _notice_texts(app) == ["interrupted"]

        # A NEW turn opens, and only then does an outcome get published.
        app.post_message(TurnStarted())
        await pilot.pause()
        session.publish_interrupted()
        await app._poll_completion_attention()
        await pilot.pause()

        assert _notice_texts(app) == ["interrupted", "Interrupted"]


@pytest.mark.asyncio
async def test_the_adopted_row_can_be_acknowledged(tmp_path, monkeypatch) -> None:
    """Adoption stamps a real anchor, so looking at the row marks it read.

    Not incidental to the duplicate fix but the other half of it. The receipt
    is cleared by `_completion_anchor_visible` finding the ANCHORED block in
    the viewport; suppressing the poller's row without stamping the live one
    would leave an interruption the user is looking at permanently unseen, and
    the sidebar flagging a session whose outcome is on screen.
    """
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    session = InterruptSession(tmp_path / "attention.db")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _interrupt_a_turn(app, pilot)
        session.publish_interrupted()
        app.on_app_focus(AppFocus())
        await app._poll_completion_attention()
        await pilot.pause()

        anchor = session.store.state(session.identity)["anchor_id"]
        assert app._completion_anchor_visible(anchor), "the adopted row is the anchor"
        # Wait on the acknowledgement the poll publishes, not on a clock.
        for _ in range(20):
            await app._poll_completion_attention()
            await pilot.pause()
            if not session.store.state(session.identity)["unseen"]:
                break
        assert not session.store.state(session.identity)["unseen"]


@pytest.mark.asyncio
async def test_a_failed_adoption_does_not_burn_the_held_row(tmp_path) -> None:
    """A poll that cannot see the row must not consume the reference.

    Round-1 review MAJOR-2. `self._own_interrupt_notice = None` ran BEFORE the
    mounted-membership test, so a poll that legitimately could not adopt —
    a sidebar switch parks the row in the other conversation's
    `TranscriptView` — destroyed the reference anyway. On switching back the
    poller then appended its `Interrupted` under the live `interrupted`,
    restoring the duplicate row the adoption exists to remove.
    """
    session = InterruptSession(tmp_path / "attention.db")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _interrupt_a_turn(app, pilot)
        held = app._own_interrupt_notice
        assert held is not None

        # The row leaves the CURRENT view — what a sidebar switch does to it,
        # and what `/clear` does permanently. Either way this poll cannot see
        # it, and the question is whether the reference survives that.
        app._transcript_view().remove_block(held)
        await pilot.pause()
        session.publish_interrupted()
        anchor = session.store.state(session.identity)["anchor_id"]
        assert app._adopt_own_interrupt_notice("interrupted", anchor) is False

        assert app._own_interrupt_notice is held, "the reference survives a failed adoption"
        assert not held.completion_anchor_id, "and nothing was stamped onto it"


@pytest.mark.asyncio
async def test_the_held_interrupt_row_rides_a_sidebar_switch(tmp_path) -> None:
    """The row belongs to its conversation, so the presentation carries it.

    Round-1 review MAJOR-2, second half. `SessionPresentation` did not carry
    `own_interrupt_notice` at all, so a switch away dropped it even once the
    burn above was fixed.
    """
    session = InterruptSession(tmp_path / "attention.db")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _interrupt_a_turn(app, pilot)
        held = app._own_interrupt_notice
        assert held is not None

        captured = app._capture_sidebar_presentation()
        assert captured.own_interrupt_notice is held, "the switch carries it out"

        # Whatever the app does while away, coming back restores the row.
        app._own_interrupt_notice = None
        app._apply_sidebar_presentation(captured)
        assert app._own_interrupt_notice is held, "and back in"

        # ...and it still adopts, so the duplicate never returns.
        session.publish_interrupted()
        await app._poll_completion_attention()
        await pilot.pause()
        assert _notice_texts(app) == ["interrupted"]

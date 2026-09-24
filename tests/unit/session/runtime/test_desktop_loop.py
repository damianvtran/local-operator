"""Owner loop invariants supplemental to the assembled HTTP/runtime exercise."""

import asyncio
from typing import Any

import pytest

from local_operator.session.frontend_state import (
    FRONTEND_CHECKPOINT_CUSTOM_TYPE,
    FrontendSessionState,
    FrontendStateStore,
)
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.runtime.test_serving import FakeSession


class LoopSession(FakeSession):
    def __init__(self):
        super().__init__()
        self.goal = "Finish the fixture"
        self.aborted = 0
        self._frontend_state_store: Any = None
        #: The `append_custom` seam `FrontendStateStore.checkpoint` writes through.
        #: Declared (as ``None``) so the durability test can substitute a recorder
        #: and the type checker still sees the attribute on the real shape.
        self._transcript: Any = None

    def abort(self, reason="cancelled"):
        self.aborted += 1
        self.prompt_release.set()


async def until(predicate):
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_cancelling_queued_loop_does_not_abort_manual_turn(tmp_path):
    session = LoopSession()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        await handle.prompt("manual")
        await until(lambda: session.prompt_calls == ["manual"])
        driver = handle._loop_driver()
        driver.start("2", session.goal)
        await until(lambda: len(handle._prompt_queue) == 2)
        await driver.cancel()
        assert driver.state["status"] == "cancelled"
        assert session.aborted == 0
        assert len(handle._prompt_queue) == 1
        session.prompt_release.set()
        await until(lambda: not handle._prompt_queue)
        assert session.prompt_calls == ["manual"]
    finally:
        await handle.dispose()


@pytest.mark.asyncio
async def test_replacement_owner_marks_active_loop_interrupted(tmp_path):
    session = LoopSession()
    session._frontend_state_store = FrontendStateStore(
        FrontendSessionState(
            session_id="abcdef123456", epoch="old", loop={"status": "running", "completed": 2}
        )
    )
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        assert session._frontend_state_store.state.loop["status"] == "interrupted"
        driver = handle._loop_driver()
        assert not driver.running
        assert driver.state == {"status": "interrupted", "completed": 2}
        assert not session.prompt_calls
    finally:
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["0", "26", "3e", "-1", "1.5"])
async def test_invalid_loop_does_not_start_work(tmp_path, argument):
    session = LoopSession()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        result = await handle.run_slash_authoritative("loop", argument, [])
        assert result["kind"] == "error"
        assert result["data"]["code"] == "loop_invalid"
        assert not handle._loop_driver().running
        assert not session.prompt_calls
    finally:
        await handle.dispose()


class _RecordingTranscript:
    """The `append_custom` seam `FrontendStateStore.checkpoint` writes through."""

    def __init__(self) -> None:
        self.customs: list[tuple[str, Any]] = []

    async def append_custom(self, custom_type: str, details: Any) -> None:
        self.customs.append((custom_type, details))


def _finished_loop_session() -> LoopSession:
    """A session whose published loop state is a FINISHED run.

    The state a dismissed desktop surface is left holding: a terminal status
    with the goal and reason of the run that produced it, which is exactly what
    a merge (`publish(**values)`) would carry into the cleared snapshot.
    """
    session = LoopSession()
    session._frontend_state_store = FrontendStateStore(
        FrontendSessionState(
            session_id="abcdef123456",
            epoch="e1",
            loop={
                "status": "done",
                "completed": 3,
                "goal": "Finish the fixture",
                "reason": "the judge saw the goal met",
            },
        )
    )
    session._transcript = _RecordingTranscript()
    return session


@pytest.mark.asyncio
async def test_loop_clear_flag_resets_a_finished_loops_published_state(tmp_path):
    """`/loop --clear` leaves ``{"status": "idle", "completed": 0}`` published.

    REPLACED, not merged: the previous run's goal and reason must not survive,
    or the surface the clear exists to dismiss would still describe an objective
    nobody is looping toward.
    """
    session = _finished_loop_session()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        result = await handle.run_slash_authoritative("loop", "--clear", [])
        assert result["kind"] == "block"
        assert result["data"]["type"] == "loop"
        assert result["data"]["status"] == "idle"
        assert result["data"]["completed"] == 0
        assert session._frontend_state_store.state.loop == {"status": "idle", "completed": 0}
        assert not session.prompt_calls
    finally:
        await handle.dispose()


@pytest.mark.asyncio
async def test_loop_clear_is_durable_in_the_frontend_checkpoint(tmp_path):
    """The clear rides the same checkpoint as every other loop transition.

    Without it the reset would hold only until something re-checkpointed the
    session, and a replaced runtime — which restores `store.state.loop` — would
    bring the dismissed run back.
    """
    session = _finished_loop_session()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        await handle.run_slash_authoritative("loop", "--clear", [])
        written = [
            details
            for custom_type, details in session._transcript.customs
            if custom_type == FRONTEND_CHECKPOINT_CUSTOM_TYPE
        ]
        assert written, "the cleared state never reached a checkpoint"
        assert written[-1]["state"]["loop"] == {"status": "idle", "completed": 0}
    finally:
        await handle.dispose()


@pytest.mark.asyncio
async def test_loop_clear_is_refused_while_a_loop_runs(tmp_path):
    """`--clear` never cancels live work, and names the word that does."""
    session = LoopSession()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        driver = handle._loop_driver()
        driver.start("2", session.goal)
        # Numeric mode iterates toward the STANDING goal with the shared loop
        # prompt, so the first iteration is in flight once the fake has been
        # handed ANY turn — and it stays in flight on the release gate, which
        # is what makes `driver.running` a real observation here.
        await until(lambda: len(session.prompt_calls) == 1)
        result = await handle.run_slash_authoritative("loop", "--clear", [])
        assert result["kind"] == "error"
        assert result["data"]["code"] == "loop_running"
        assert "/loop --stop" in result["text"]
        # Still driving: the refusal is a refusal, not a quiet cancel.
        assert driver.running
        assert session.aborted == 0
        await driver.cancel()
        session.prompt_release.set()
        await until(lambda: not handle._prompt_queue)
    finally:
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["--stop", "stop", "cancel", "abort"])
async def test_loop_stop_forms_cancel_the_driver(tmp_path, argument):
    """The flag and the three bare words all reach the same cancel."""
    session = LoopSession()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        driver = handle._loop_driver()
        driver.start("2", session.goal)
        # Numeric mode iterates toward the STANDING goal with the shared loop
        # prompt, so the first iteration is in flight once the fake has been
        # handed ANY turn — and it stays in flight on the release gate, which
        # is what makes `driver.running` a real observation here.
        await until(lambda: len(session.prompt_calls) == 1)
        result = await handle.run_slash_authoritative("loop", argument, [])
        assert result["kind"] == "block"
        assert not driver.running
        assert driver.state["status"] == "cancelled"
        session.prompt_release.set()
        await until(lambda: not handle._prompt_queue)
    finally:
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["--stop ", "--stop\t", "cancel "])
async def test_a_trailing_space_still_stops_the_loop(tmp_path, argument):
    """The wire carries `Command.args` verbatim, and the flags match WHOLE.

    Unstripped, `/loop --stop ` fell through the flag comparisons to the count
    parser and answered `loop_busy` while the driver kept running — a stop that
    silently did nothing, on the one host where the loop is real work (round 1,
    reviewer MAJOR-2). The TUI strips the same argument before its handler, so
    the two hosts disagreed about one word.
    """
    session = LoopSession()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        driver = handle._loop_driver()
        driver.start("2", session.goal)
        await until(lambda: len(session.prompt_calls) == 1)
        result = await handle.run_slash_authoritative("loop", argument, [])
        assert result["kind"] == "block", result
        assert not driver.running
        session.prompt_release.set()
        await until(lambda: not handle._prompt_queue)
    finally:
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["--clear ", "--clear\t"])
async def test_a_trailing_space_on_clear_never_starts_a_loop(tmp_path, argument):
    """Unstripped this started a PAID unbounded goal-mode loop toward `--clear`.

    That is the worst shape of the same defect: a word that promises to tidy a
    snapshot away, answered with a loop toward its own literal text.
    """
    session = LoopSession()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        result = await handle.run_slash_authoritative("loop", argument, [])
        assert result["kind"] == "block", result
        assert result["data"]["status"] == "idle"
        assert not handle._loop_driver().running
        # The idle snapshot is the replaced one, not a loop's state.
        assert result["data"] == {"type": "loop", "status": "idle", "completed": 0}
    finally:
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["--stopx", "--clearx", "--clera"])
async def test_a_bare_unknown_flag_is_refused_instead_of_starting_a_loop(tmp_path, argument):
    """Two flag vocabularies are taught, so a mixed-up one is the expected typo.

    The documented whole-argument rule sent anything else down the GOAL branch,
    so `/loop --stopx` started an unbounded paid loop toward the literal flag
    text (round 1, reviewer NIT-5 / UX U6).
    """
    session = LoopSession()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        result = await handle.run_slash_authoritative("loop", argument, [])
        assert result["kind"] == "error", result
        assert result["data"]["code"] == "loop_invalid"
        assert f"unknown flag {argument}" in result["text"]
        assert "/loop --stop cancels" in result["text"]
        assert not handle._loop_driver().running
        assert session.prompt_calls == []
    finally:
        await handle.dispose()


@pytest.mark.asyncio
async def test_the_loops_own_turn_is_stamped_as_harness_chrome(tmp_path):
    """Agent review round 3: the loop's working turn is chrome, on the wire.

    Driven through THIS route's own driver over a REAL session, because the
    marker's whole job is to survive into the durable transcript: the cell reads
    the persisted rows back rather than the arguments a call was made with.

    Why it matters: ``LOOP_GOAL_PROMPT`` is in neither ``harness_chrome_prompts()``
    nor a producer-side recogniser, and ``docs/DESKTOP_API.md`` names "the goal
    loop's own prompt" as a row that MUST carry
    ``provider_payload.harness_injected`` — the desktop is marker-only by
    contract. Unstamped, a goal loop's turn replayed on every surface as the
    USER's own words. Measured before the fix, through this handle: the row read
    ``stamp=no, chrome-recognised=False``.

    The negative control is in the same transcript: the person's own turn is not
    stamped, which is what stops the marker from hiding a human's words.
    """
    from local_operator.harness.types import Message
    from local_operator.session.goal_loop import LOOP_GOAL_PROMPT
    from tests.e2e.harness import (
        ScriptedStream,
        build_session,
        dispose_quietly,
        text_turn,
    )

    stream = ScriptedStream(
        [text_turn("advanced")] + [text_turn("VERDICT: ACHIEVED\nthe work is done")] * 5
    )
    session = build_session(tmp_path / "loop", stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        await handle.prompt("Ship it now", wait_complete=True)
        driver = handle._loop_driver()
        driver.start("Verify the fixture goal", "")
        async with asyncio.timeout(60):
            while driver.running:
                await asyncio.sleep(0.01)
        # ``isinstance(Message)`` is the repo's narrowing here rather than a
        # ``getattr`` guard: ``history()`` is a union that also carries
        # ``CustomMessage``, which has no ``text``/``provider_payload`` at all,
        # and this cell reads the persisted ROW.
        rows = [m for m in session.history() if isinstance(m, Message) and m.role == "user"]
    finally:
        await handle.dispose()
        await dispose_quietly(session)

    by_text = {m.text: (m.provider_payload or {}) for m in rows}
    assert stream.exhausted_at is None, (
        "the loop settled on this script: a short tape does not fail the test, it "
        "answers the next call from the wrong turn"
    )
    assert by_text.get("Ship it now") == {}, "a person's own turn is never stamped"
    loop_turn = LOOP_GOAL_PROMPT.format(goal="Verify the fixture goal")
    assert loop_turn in by_text, f"the loop's own turn is on the record: {list(by_text)}"
    assert by_text[loop_turn] == {"harness_injected": True}

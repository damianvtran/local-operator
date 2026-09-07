"""`/move` driven through the REAL app: the editor, the submit handler, the band.

Calling ``_cmd_move`` directly would skip the editor and the submit handler,
which is the pair a user actually goes through — so everything here types into
the real composer and presses Enter.

The band assertions are the load-bearing ones. The failure this feature must
not have is the one AGENTS.md names for `/reload`: the screen showing one
directory while the session works in another. So every path that changes the
directory is checked against what the band says, and every path that refuses is
checked against the band NOT having changed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.move_picker import MovePickerScreen
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _factory


class MovableSession(FakeSession):
    """A session that can actually be moved, recording what it was asked.

    ``FakeSession`` deliberately has no ``set_working_directory``: the app must
    refuse a facade that cannot be moved rather than reporting a move that did
    not happen, and that refusal is asserted below too.
    """

    def __init__(self, cwd: str = "/tmp", outcome: str = "cold") -> None:
        super().__init__()
        self._cwd = cwd
        self._outcome = outcome
        self.moves: list[str] = []
        self.error: Exception | None = None
        # ``FakeSession`` pins ``session_id`` to a constant, but the eval-latch
        # guards need TWO distinguishable conversations to drive a real
        # replacement through ``_adopt_session`` — one id for every double
        # cannot express a swap. Overridden as a property (not a plain
        # attribute) so it stays type-compatible with the base.
        self._session_id = "sess"
        #: What ``is_cold`` reads. The app must NOT consult it to decide
        #: whether to narrate a move — a viewer with an engage in flight reads
        #: cold while its move joins that engage and then retires the runtime
        #: it produces (review MAJOR-1, design U6). It stays on this double
        #: precisely so a test can set it to the reading that used to suppress
        #: the line and prove the app is indifferent to it.
        self.is_cold: bool = True

    def move_will_wait(self) -> bool:
        """DERIVED from the outcome, so this double cannot lie about the wait.

        The round-1 guard let the gate's input and the move's outcome be set
        independently, so a gate reading an unrelated predicate could disagree
        with what the move actually did and the test still passed — which is
        how the ``is_cold`` gate shipped green beside the blocker that
        established ``is_cold`` is unsound. Tying the two together here makes
        the disagreement unrepresentable: a rebind is exactly the move that
        retires a runtime, which is exactly the move that makes the user wait.
        """
        return self._outcome == "rebound"

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._session_id = value

    async def set_working_directory(self, cwd: str) -> str:
        self.moves.append(cwd)
        if self.error is not None:
            raise self.error
        self._cwd = cwd
        return self._outcome


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _band(app: OperatorApp) -> str:
    status = app._status
    assert status is not None
    return status.render_text(200).plain


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _submit(pilot, app: OperatorApp, text: str) -> None:
    editor = app.query_one(Editor)
    editor.text = text
    await pilot.pause()
    if editor._picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    for _ in range(6):
        await pilot.pause()


@pytest.mark.asyncio
async def test_moving_a_cold_session_updates_the_band_and_says_so(tmp_path: Path) -> None:
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, f"/move {destination}")

        assert session.moves == [str(destination)]
        assert str(destination) in _band(app)
        assert any("moved to" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_a_rebound_session_says_its_runtime_restarted(tmp_path: Path) -> None:
    """The user must be told the runtime was replaced — it is a visible pause
    and a real event, even though the conversation is untouched."""
    session = MovableSession(cwd=str(tmp_path), outcome="rebound")
    app = OperatorApp(lambda: _factory(session))
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, f"/move {destination}")
        assert any("runtime restarted" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_a_tilde_path_is_expanded(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "project").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/move ~/project")
        assert session.moves == [str(home / "project")]


@pytest.mark.asyncio
async def test_a_relative_path_resolves_against_the_SESSIONS_directory(tmp_path: Path) -> None:
    (tmp_path / "child").mkdir()
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/move child")
        assert session.moves == [str(tmp_path / "child")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suffix,expected",
    [("nope", "no such directory"), ("file.txt", "not a directory")],
)
async def test_an_invalid_target_is_refused_without_moving(
    tmp_path: Path, suffix: str, expected: str
) -> None:
    """A clear notice, never a traceback and never a half-applied state."""
    (tmp_path / "file.txt").write_text("x")
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        before = _band(app)
        await _submit(pilot, app, f"/move {tmp_path / suffix}")

        assert session.moves == []
        assert any(expected in text for text in _notices(app))
        assert _band(app) == before


@pytest.mark.asyncio
async def test_moving_to_the_directory_youre_already_in_is_a_no_op(tmp_path: Path) -> None:
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, f"/move {tmp_path}")
        assert session.moves == []
        assert any("already in" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_a_refusal_from_the_session_reaches_the_user_verbatim(tmp_path: Path) -> None:
    """A busy session's refusal is the receipt; the band must not move."""
    session = MovableSession(cwd=str(tmp_path))
    session.error = RuntimeError("this session is working right now")
    app = OperatorApp(lambda: _factory(session))
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        before = _band(app)
        await _submit(pilot, app, f"/move {destination}")

        assert any("working right now" in text for text in _notices(app))
        assert _band(app) == before


@pytest.mark.asyncio
async def test_a_session_that_cannot_be_moved_is_refused_not_silently_ignored() -> None:
    """``FakeSession`` has no ``set_working_directory``; reporting success
    would tell the user a move happened that did not."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/move /tmp")
        assert any("cannot be moved" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_a_bare_move_opens_the_picker_on_the_current_directory(tmp_path: Path) -> None:
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/move")
        for _ in range(4):
            await pilot.pause()

        screen = app.screen
        assert isinstance(screen, MovePickerScreen)
        assert screen.visible_rows
        assert screen.visible_rows[0].path == str(tmp_path)
        assert screen.visible_rows[0].kind == "current"


@pytest.mark.asyncio
async def test_escaping_the_picker_leaves_the_session_where_it_was(tmp_path: Path) -> None:
    """A cancelled picker is not an event worth a transcript line."""
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        before = _band(app)
        await _submit(pilot, app, "/move")
        for _ in range(4):
            await pilot.pause()
        await pilot.press("escape")
        for _ in range(4):
            await pilot.pause()

        assert session.moves == []
        assert _band(app) == before


@pytest.mark.asyncio
async def test_choosing_a_row_in_the_picker_moves_there(tmp_path: Path) -> None:
    """The whole point of the card, driven the way a user drives it."""
    child = tmp_path / "child"
    child.mkdir()
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/move")
        for _ in range(4):
            await pilot.pause()

        screen = app.screen
        assert isinstance(screen, MovePickerScreen)
        # Row 0 is the current directory, so move down to a real destination.
        index = next(i for i, row in enumerate(screen.visible_rows) if row.path != str(tmp_path))
        for _ in range(index):
            await pilot.press("down")
            await pilot.pause()
        chosen = screen.selected_path()
        await pilot.press("enter")
        for _ in range(6):
            await pilot.pause()

        assert session.moves == [chosen]
        assert str(chosen) in _band(app)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_cold",
    [
        # A settled bound runtime: the case that already worked.
        False,
        # An engage IN FLIGHT: reads cold, joins, retires, rebinds. The move
        # takes seconds and the round-1 gate said nothing for all of them.
        True,
    ],
)
async def test_every_move_that_waits_says_so_before_it_waits(tmp_path: Path, is_cold: bool) -> None:
    """THE INVARIANT, not the gate's input: a move that ends in ``rebound``
    made the user wait, so it must have narrated before it did.

    The round-1 guard set ``is_cold = False`` itself and asserted the line
    appeared — so it pinned the branch that already worked and was
    structurally incapable of seeing the one that shipped broken beside it.
    The double returns ``rebound`` regardless of ``is_cold``, so the gate and
    the outcome could disagree freely and the test still passed. Gating on
    ``is_cold`` then left a joined mount engage silent for a measured 1.94 s —
    the feature's primary case, and the same predicate this PR's own blocker
    established is unsound (review MAJOR-1, design U6).

    Parametrised over both readings of ``is_cold`` precisely because the
    invariant must not depend on it: the double's ``move_will_wait`` is
    derived from the outcome, so ``is_cold`` is free to say anything and a
    gate that consults it goes red on the ``True`` case.
    """
    session = MovableSession(cwd=str(tmp_path), outcome="rebound")
    session.is_cold = is_cold
    app = OperatorApp(lambda: _factory(session))
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, f"/move {destination}")

        assert session.moves == [str(destination)], "the move must actually have run"
        texts = _notices(app)
        assert any("restarting the runtime" in t for t in texts), (
            f"a move that returned 'rebound' (is_cold={is_cold}) never told the "
            f"user it was going to wait: {texts}"
        )


@pytest.mark.asyncio
async def test_a_move_that_does_not_wait_stays_quiet(tmp_path: Path) -> None:
    """The other half of the invariant, so the fix cannot be "always narrate".

    A genuinely cold viewer settles within the frame, so an in-flight line
    would be contradicted by its own receipt a moment later. ``is_cold`` is
    left FALSE here — the reading that used to force the line — so this half
    is red for an "always narrate" fix and for the old gate alike.
    """
    session = MovableSession(cwd=str(tmp_path), outcome="cold")
    session.is_cold = False
    app = OperatorApp(lambda: _factory(session))
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, f"/move {destination}")

        texts = _notices(app)
        assert not any("restarting the runtime" in t for t in texts), texts
        assert any("moved to" in t for t in texts), texts


@pytest.mark.asyncio
async def test_choosing_the_current_row_says_you_are_already_there(tmp_path: Path) -> None:
    """Selecting the row labelled `current` closed the picker in silence —
    byte-identical to Esc, on the one row whose purpose is to answer "where am
    I?" — while typing the same path said `already in …` (UX U4)."""
    session = MovableSession(cwd=str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/move")
        for _ in range(4):
            await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MovePickerScreen)
        assert screen.visible_rows[0].kind == "current"
        await pilot.press("enter")
        for _ in range(6):
            await pilot.pause()

        assert session.moves == [], "the current row must not issue a move"
        assert any("already in" in t for t in _notices(app)), _notices(app)


@pytest.mark.asyncio
async def test_a_rebind_warns_that_the_eval_kernel_was_lost(tmp_path: Path) -> None:
    """A move that restarts the runtime destroys the persistent `eval`
    namespace, and the user must be TOLD rather than find out from a
    `NameError` two turns later.

    `tools/eval.py` caches one interpreter per session in `_KERNELS`, and
    `Session` registers `close_session_kernel` as a dispose hook — so retiring
    the runtime takes every variable, import and function built up in `eval`
    with it, while the conversation, transcript and session id all survive.
    That survival is exactly what makes the loss surprising: nothing else on
    screen changes. Reported by a peer session during round 3, verified in
    source.
    """
    session = MovableSession(cwd=str(tmp_path), outcome="rebound")
    app = OperatorApp(lambda: _factory(session))
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        # A real `eval` call in the conversation: the signal the receipt reads.
        app._append_block(ToolCard(tool_call_id="c1", tool_name="eval"))
        await pilot.pause()
        await _submit(pilot, app, f"/move {destination}")

        texts = _notices(app)
        assert any(
            "set up in eval" in t for t in texts
        ), f"a rebind destroyed the eval namespace without saying so: {texts}"
        # The move still HAPPENS: this is narration, not a refusal.
        assert session.moves == [str(destination)]
        assert str(destination) in _band(app)


@pytest.mark.asyncio
async def test_a_session_that_never_used_eval_is_not_warned_about_it(tmp_path: Path) -> None:
    """The other half, so the warning cannot become noise on every rebind.

    A user who has never touched `eval` has no kernel to lose, and a receipt
    that mentions one invents a consequence that did not happen.
    """
    session = MovableSession(cwd=str(tmp_path), outcome="rebound")
    app = OperatorApp(lambda: _factory(session))
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, f"/move {destination}")

        texts = _notices(app)
        assert any("runtime restarted" in t for t in texts), texts
        assert not any("set up in eval" in t for t in texts), texts


@pytest.mark.asyncio
async def test_a_COLD_move_never_mentions_the_eval_kernel(tmp_path: Path) -> None:
    """A cold move retires nothing, so no kernel is disposed — even in a
    session that has used `eval`. Warning there would be a lie about a loss
    that did not occur."""
    session = MovableSession(cwd=str(tmp_path), outcome="cold")
    app = OperatorApp(lambda: _factory(session))
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._append_block(ToolCard(tool_call_id="c1", tool_name="eval"))
        await pilot.pause()
        await _submit(pilot, app, f"/move {destination}")

        texts = _notices(app)
        assert any("moved to" in t for t in texts), texts
        assert not any("set up in eval" in t for t in texts), texts


@pytest.mark.asyncio
async def test_the_eval_warning_survives_a_cleared_transcript(tmp_path: Path) -> None:
    """`/clear` empties the VIEW while the session, the runtime and the `eval`
    kernel all survive — and `/clear`'s own receipt promises "history is
    untouched", so that user has every reason to believe their state is
    intact. Deriving "did this session use eval" from `blocks()` therefore
    lost the answer in the one direction that costs work, which the original
    docstring ruled out as impossible (review round 3, MAJOR-1).

    CLEARS THE TRANSCRIPT and still expects the warning. The three guards that
    shipped with the feature all pass while the defect is live, because none
    of them clears or bounds the transcript — the decoration shape AGENTS.md
    names.
    """
    session = MovableSession(cwd=str(tmp_path), outcome="rebound")
    session.is_cold = False
    app = OperatorApp(lambda: _factory(session))
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._append_block(ToolCard(tool_call_id="c1", tool_name="eval"))
        for _ in range(3):
            await pilot.pause()

        # The kernel is untouched by this; only the screen is emptied.
        app._transcript_view().clear_blocks()
        for _ in range(3):
            await pilot.pause()
        assert app._session_used_eval(), "the latch did not survive /clear"

        await _submit(pilot, app, f"/move {destination}")
        texts = _notices(app)

    assert any(
        "set up in eval" in t for t in texts
    ), f"a rebind destroyed the eval namespace with no warning after /clear: {texts}"


@pytest.mark.asyncio
async def test_the_eval_record_follows_the_conversation_across_a_round_trip(
    tmp_path: Path,
) -> None:
    """A -> B -> A: the record belongs to the conversation, not the viewer.

    THE THREE STATES IN ONE WALK, because closing either direction alone is
    what three review rounds each did. Leaving A must not carry its record to
    B (a conversation told its namespace was destroyed when it never had one,
    MAJOR-4.1); returning to A must not have lost it (a live kernel destroyed
    in silence, MAJOR-5.1 — the sidebar return path re-adopts the SAME
    session, whose kernel never died). A single flag cannot satisfy both, which
    is why the record is keyed by session id.

    EVERY ADOPTED SESSION GETS ITS OWN EMPTY VIEW, and that is load-bearing
    rather than tidiness: `_session_used_eval` falls back to the transcript,
    so a probe that adopts B while A's `eval` card is still on screen has the
    fallback answer for it and passes over a broken record. That masking is
    what made a simulation of the round-5 remedy report success — and what made
    an earlier repro of MAJOR-5.1 fail to reproduce it.
    """
    session_a = MovableSession(cwd=str(tmp_path), outcome="rebound")
    session_a.session_id = "conversation-a"
    session_b = MovableSession(cwd=str(tmp_path), outcome="rebound")
    session_b.session_id = "conversation-b"
    app = OperatorApp(lambda: _factory(session_a))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._append_block(ToolCard(tool_call_id="c1", tool_name="eval"))
        for _ in range(3):
            await pilot.pause()
        assert app._session_used_eval(), "the session that ran eval is not recorded"

        # A -> B, with B on its own empty view as production gives it.
        app._adopt_session(session_b, replay_history=False, reuse_controller=True)
        app._transcript_view().clear_blocks()
        for _ in range(3):
            await pilot.pause()
        assert not app._session_used_eval(), (
            "conversation B inherited A's eval record and would be warned "
            "about a namespace it never had"
        )

        # B -> A. The same live session returns; its kernel never died, so the
        # warning it is owed must still be there.
        app._adopt_session(session_a, replay_history=False, reuse_controller=True)
        for _ in range(3):
            await pilot.pause()
        assert app._session_used_eval(), (
            "returning to conversation A lost its eval record, so a move would "
            "destroy a live kernel in silence"
        )


@pytest.mark.asyncio
async def test_the_eval_record_answers_only_for_the_session_that_ran_it(
    tmp_path: Path,
) -> None:
    """Membership, asserted directly rather than through a swap's side effects.

    The state is per conversation, so the storage is too: another session's id
    in the record cannot answer for this one. That is the property which makes
    a swap path added later unable to reintroduce either direction — there is
    nothing for it to remember to clear.
    """
    session = MovableSession(cwd=str(tmp_path), outcome="rebound")
    session.session_id = "the-current-one"
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._transcript_view().clear_blocks()
        for _ in range(3):
            await pilot.pause()

        # Somebody else's conversation is recorded: the shape a leak takes.
        app._sessions_that_used_eval.add("a-conversation-the-user-left")
        assert not app._session_used_eval(), "another session's record answered"

        # This conversation's own id does answer.
        app._sessions_that_used_eval.add("the-current-one")
        assert app._session_used_eval()


def test_the_eval_record_is_bounded() -> None:
    """The set is capped: a viewer can walk through many conversations in one
    process, and an unbounded id set is a slow leak for a signal that only
    decorates one receipt.

    Asserted through the recorder rather than by reading the constant, so the
    bound is checked where it is enforced. The empty-id case is here too: an
    unidentified session cannot be matched on read, so storing one would grow
    the set without ever answering a question.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))

    app._remember_session_used_eval("")
    assert not app._sessions_that_used_eval, "an unidentifiable session was stored"

    for index in range(app._EVAL_MEMORY_MAX + 50):
        app._remember_session_used_eval(f"session-{index}")
    assert len(app._sessions_that_used_eval) <= app._EVAL_MEMORY_MAX
    # The most recent writer is always present: eviction may drop an older id,
    # never the one being recorded.
    assert f"session-{app._EVAL_MEMORY_MAX + 49}" in app._sessions_that_used_eval

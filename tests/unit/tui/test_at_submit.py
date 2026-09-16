"""The ``@path`` submit exits: what the MODEL receives, and what the row shows.

Split from the picker half (``test_at_picker.py``) because the two regress
independently. This file asserts nothing about when a list opens; it is about
the text that leaves the composer.

Two properties carry the feature and are asserted separately, because they can
break separately: the model gets the file's content, and the transcript keeps
the short line the operator typed. A change that expanded the row as well would
pass every "the model got it" assertion and still be wrong.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_operator.references import REFERENCE_BLOCK_OPEN
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor, EditorSubmitted, PastedText

from .test_app_pilot import FakeSession, _factory
from .test_slash_echo import _boot, _notice_texts, _user_rows

#: The content the fixture writes, unique enough that finding it in a payload
#: cannot be a coincidence with anything else the app puts there.
FILE_BODY = "def login():\n    return MARKER_CONTENT_42\n"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "auth.py").write_text(FILE_BODY)
    (tmp_path / "README.md").write_text("readme\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


async def _submit(pilot, app: OperatorApp, text: str) -> None:
    """Type a line and press Enter — through the editor, not around it.

    The picker is dismissed first when it is showing: Enter on an open list
    COMPLETES the highlighted row instead of submitting, so a draft ending in a
    ``@`` token would otherwise never reach the submit handler. Same reason and
    same shape as ``test_slash_echo._submit``.
    """
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    editor.load_text(text)
    editor.move_cursor(editor._end_of_buffer())
    await pilot.pause()
    if editor.picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    for _ in range(40):
        await pilot.pause()


async def _await_prompt(pilot, session: FakeSession) -> None:
    for _ in range(200):
        await pilot.pause()
        if session.prompts:
            return


# --- Exit 1: the main prompt -------------------------------------------------


@pytest.mark.asyncio
async def test_the_main_prompt_exit_hands_the_token_to_the_session(workspace) -> None:
    """Exit 1 passes the token through; `Session.prompt` is what expands it.

    NOT a weakening of the feature — a relocation of it, and the safety property
    is the reason. Expanding in this handler cannot carry the approval gate: the
    pump awaits each message handler to completion and the approval card is
    mounted and answered through that same pump, so awaiting one here freezes
    the composer (probed: no card ever mounts, Enter never returns).
    `Session.prompt` expands with `None if self._yolo else self._request_approval`
    from OUTSIDE the pump, where the card works.

    So what this exit owes is an unexpanded, unmangled token. The end-to-end
    property — that the model receives the file — is asserted against the real
    resolver below, over exactly the text the TUI handed over.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "what does @auth.py do?")
        await _await_prompt(pilot, session)
    assert session.prompts, "nothing reached the model"
    handed_over = session.prompts[0]
    assert handed_over == "what does @auth.py do?"
    assert REFERENCE_BLOCK_OPEN not in handed_over

    from local_operator.references import expand_references

    result = await expand_references(handed_over, str(workspace))

    assert REFERENCE_BLOCK_OPEN in result.sent
    assert "MARKER_CONTENT_42" in result.sent


@pytest.mark.asyncio
async def test_the_transcript_row_shows_the_typed_line_not_the_block(workspace) -> None:
    """The display/sent split: the operator sees what they typed.

    Without it a one-line question about a file paints as the whole file.

    NAMING is deliberately not cited as a second reason. It does not read this
    text: `_submit_prompt` titles from `named = typed or text` (`app.py:23987`)
    and nothing in `session/naming.py` calls `user_row_text`. The transcript-row
    reason is sufficient on its own, and a load-bearing comment that states
    something false is a defect in this repo, not a nit.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "what does @auth.py do?")
        await _await_prompt(pilot, session)
        rows = _user_rows(app)
    assert rows, "no user row was painted"
    assert "@auth.py" in rows[0]
    assert "MARKER_CONTENT_42" not in rows[0]
    assert REFERENCE_BLOCK_OPEN not in rows[0]


@pytest.mark.asyncio
async def test_an_unresolved_token_is_sent_verbatim_and_SILENTLY_at_exit_1(
    workspace,
) -> None:
    """A token that resolves to nothing is PROSE, and the text still goes.

    Swallowing the request would be the worse half of the trade — the same
    bargain ``_expand_invocation`` strikes for an unreadable skill body — and
    that half still holds.

    THE NOTICE DOES NOT, and this records it as a known v1 limitation rather
    than leaving it to be rediscovered. Exit 1 no longer expands in the TUI (see
    the exit-1 test above for why), and `Session.prompt` has no channel back to
    a UI for its notices. So `@nope.py` is sent as written with nothing on
    screen to say why. Exits 3 and 4 have always behaved this way, so this is
    exit 1 becoming CONSISTENT rather than newly broken.

    The fix, if it is wanted, is expanding exit 1 in a worker — which disturbs
    the submit ordering `on_editor_submitted` calls load-bearing, and so is its
    own change. The ASIDE keeps its notices and is tested for them below.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "ask @nope.py about it")
        await _await_prompt(pilot, session)
        notices = _notice_texts(app)
    assert session.prompts, "the turn was swallowed"
    assert "@nope.py" in session.prompts[0]
    assert REFERENCE_BLOCK_OPEN not in session.prompts[0]
    assert not any("nope.py" in notice for notice in notices), (
        "exit 1 painted a reference notice — if this is now wanted, the worker "
        "change landed and this test should assert the notice instead"
    )


# --- Exit 2: the aside, the hole this feature had to close -------------------


@pytest.mark.asyncio
async def test_the_aside_exit_expands(workspace) -> None:
    """``/btw what does @auth.py do?`` must not reach the aside model bare.

    The exit most likely to be missed BECAUSE it does not go through
    ``_expand_invocation`` and so does not look like the others. It is also the
    one exit the session layer cannot cover: the aside is a separate model call
    that never reaches ``Session.prompt``, so if the TUI does not expand here,
    nothing does.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        # Opens the card.
        await _submit(pilot, app, "/btw what does @auth.py do?")
        for _ in range(80):
            await pilot.pause()
            if session.asides:
                break
    assert session.asides, "the aside was never asked"
    asked = "\n".join(str(turn) for turn in session.asides[-1])
    assert "MARKER_CONTENT_42" in asked, "the aside model got the bare token"


# --- Exits 3 and 4: covered by the session, asserted rather than assumed -----


@pytest.mark.asyncio
async def test_the_slash_command_prompt_exit_expands(workspace) -> None:
    """``/goal <request with @auth.py>`` reaches the model expanded.

    THE LOAD-BEARING TEST for the decision not to expand TUI-side at
    ``_submit_command_prompt``. That method is synchronous and every path into
    it is synchronous, while ``expand_references`` is a coroutine; expansion is
    left to ``Session.prompt``, which awaits it before taking the turn lock.

    "The session covers it" is an assertion until something proves it, and this
    is the proof: it drives exit 3 (``_run_slash_command``) into exit 4 through
    a real ``consumes_prompt`` command and reads what the session was handed.
    If this fails, the decision was wrong — the fix is making that method async
    as its own refactor, never a detached task from inside it.
    """
    from local_operator.references import expand_references

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/goal fix what @auth.py does")
        await _await_prompt(pilot, session)
    assert session.prompts, "the command never submitted a prompt"
    request = session.prompts[0]
    # The TUI hands this through unexpanded BY DESIGN; `Session.prompt` is what
    # expands it. `FakeSession` records the text instead of running the real
    # `prompt`, so the expansion is asserted on the same text the real session
    # would have received — which is exactly the input that method acts on.
    assert "@auth.py" in request
    result = await expand_references(request, str(workspace))
    assert result.expanded is True, f"the session could not expand it: {result.notices}"
    assert "MARKER_CONTENT_42" in result.sent


# --- Ordering and layering ---------------------------------------------------


@pytest.mark.asyncio
async def test_references_expand_before_the_invocation_splice(workspace, tmp_path) -> None:
    """A ``@path`` in a ``$skill`` REQUEST expands; the SKILL.md body does not.

    The property is unchanged by exit 1's expansion moving to the session; WHERE
    it is asserted is. The TUI hands `Session.prompt` the spliced payload —
    skill body and request together — and the session expands that, so the
    guarantee now rests on the resolver deduplicating by RESOLVED PATH rather
    than on expansion happening before the splice.

    It still holds, and this drives the real resolver over the real payload to
    say so: one block, and the body's `@auth.py` left as the literal text the
    skill author wrote. A SKILL.md that happens to contain an ``@word`` is not
    making a reference, and must not grow a second expansion.
    """
    root = tmp_path / "skills" / "research"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: research\ndescription: Investigate.\n---\n\nSee @auth.py for the shape.\n"
    )
    import os

    os.environ["LOCAL_OPERATOR_SKILL_EXTRA_ROOTS"] = str(tmp_path / "skills")
    try:
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            await _submit(pilot, app, "$research look at @auth.py")
            await _await_prompt(pilot, session)
    finally:
        os.environ.pop("LOCAL_OPERATOR_SKILL_EXTRA_ROOTS", None)
    assert session.prompts, "the invocation never submitted"
    payload = session.prompts[0]
    # The TUI's half: the body was spliced in, and NOTHING was expanded here.
    assert "See @auth.py for the shape." in payload
    assert REFERENCE_BLOCK_OPEN not in payload, "the TUI expanded; the gated pass cannot re-run"

    # The session's half, through the real resolver over the real payload.
    from local_operator.references import expand_references

    result = await expand_references(payload, str(workspace))

    assert result.expanded
    # Exactly once, deduplicated by resolved path — the body's token and the
    # request's token name the same file and must not produce two blocks.
    assert result.sent.count(REFERENCE_BLOCK_OPEN) == 1
    assert "MARKER_CONTENT_42" in result.sent
    # ...and the body's `@auth.py` is still the literal text the skill author
    # wrote, not a substitution.
    assert "See @auth.py for the shape." in result.sent


@pytest.mark.asyncio
async def test_expansion_is_awaited_outside_the_turn_lock(workspace) -> None:
    """STRUCTURAL, not timed: the lock is unheld while expansion runs.

    An approval can park on a human indefinitely, and the lock that serialises
    turns also serialises compaction — so awaiting a human decision inside it is
    a deadlock shape rather than a slow turn.

    THE INSTRUMENT IS PROVEN BEFORE ITS READING IS TRUSTED. This test read
    `getattr(session, "_turn_lock", None)` and recorded False whenever it was
    not an `asyncio.Lock` — and `FakeSession` HAS no `_turn_lock`, so it
    recorded False for "no lock exists" and passed with nothing observed. It
    would have stayed green if expansion were awaited inside a held lock, which
    is the one thing it exists to catch. The fake is given a real lock, and the
    positive control below asserts the detector can report True, so a later
    refactor cannot quietly return it to measuring nothing.
    """
    session = FakeSession()
    # The real session's lock is `asyncio.Lock` (`session/session.py:2327`); the
    # fake needs the same TYPE, not a stand-in, because the detector's
    # `isinstance` check is what decides whether a reading is taken at all.
    # Set on the FAKE rather than declared on it: `test_app_pilot.FakeSession`
    # is shared by the whole TUI suite, and this is the test's instrument, not
    # part of the shape the app requires of a session.
    session._turn_lock = asyncio.Lock()  # type: ignore[attr-defined]
    app = OperatorApp(lambda: _factory(session))
    observed: list[bool] = []

    import local_operator.tui.app as app_mod

    real_expand = app_mod.expand_references

    async def _spy(text: str, cwd: str, **kwargs):
        # `_turn_lock` is the session's; record whether it is held at the moment
        # expansion runs, which is the property under test.
        lock = getattr(session, "_turn_lock", None)
        observed.append(bool(lock.locked()) if isinstance(lock, asyncio.Lock) else False)
        return await real_expand(text, cwd, **kwargs)

    app_mod.expand_references = _spy
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            # THE ASIDE, because it is now the TUI's only expansion site: exit 1
            # hands its text to `Session.prompt` unexpanded. The property is the
            # same one and it still has teeth here — `_aside_worker` awaits this
            # expansion, and the aside runs while a turn may be in flight.
            await _submit(pilot, app, "/btw what does @auth.py do?")
            for _ in range(120):
                await pilot.pause()
                if session.asides:
                    break
    finally:
        app_mod.expand_references = real_expand

    assert observed, "expansion never ran on the aside path"
    assert not any(observed), "expansion awaited while the turn lock was held"

    # POSITIVE CONTROL: the same detector, with the lock deliberately held, must
    # report True. Without this the assertions above are satisfied by an
    # instrument that reads False unconditionally — which is exactly how this
    # test passed while measuring nothing.
    control: list[bool] = []
    async with session._turn_lock:  # type: ignore[attr-defined]
        lock = getattr(session, "_turn_lock", None)
        control.append(bool(lock.locked()) if isinstance(lock, asyncio.Lock) else False)
    assert control == [True], "the detector cannot report a held lock; its readings mean nothing"


# --- References and pastes in ONE draft --------------------------------------


async def _submit_with_paste(pilot, app: OperatorApp, text: str, pasted: PastedText) -> None:
    """Post the submit the editor posts for a draft carrying a collapsed paste.

    Through ``on_editor_submitted`` with a real :class:`PastedText`, rather than
    by calling the expansion helpers directly: the defect this covers lives in
    how that handler COMBINES the two expansions, so a test that called them
    itself would pass while the composer stayed broken.
    """
    app.post_message(EditorSubmitted(text, attachments={1: pasted}))
    for _ in range(60):
        await pilot.pause()


@pytest.mark.parametrize(
    "draft",
    [
        "[Paste #1, 3 lines] look at @auth.py",
        "look at @auth.py then [Paste #1, 3 lines]",
    ],
    ids=["paste-first", "reference-first"],
)
@pytest.mark.asyncio
async def test_a_draft_with_BOTH_a_paste_and_a_reference_sends_both(workspace, draft) -> None:
    """The natural way to use this feature, and it was the broken one.

    "Here's the traceback, compare it with ``@src/auth.py``" carries a chip and
    an ``@path`` in one draft. When exit 1 expanded references it set ``sent``
    to the CHIP text — never paste-expanded — and the `expand_pastes` branch
    below it was skipped, so the model received the literal
    ``[Paste #1, 3 lines]`` and the pasted payload was lost entirely. A
    regression against the pre-feature behaviour, where ordinary prose left with
    ``sent is None`` and the splice happened.

    It is closed by a DELETION: exit 1 no longer expands, so ``sent`` stays
    ``None`` for ordinary prose and the original splice is live again. Both
    halves are asserted — the payload goes, and the ``@`` token survives it
    intact so the session's gated pass can still expand it.

    BOTH ORDERS, because the splice is by span and a token sitting after the
    chip is the case where an off-by-one would not show.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    pasted = PastedText("PASTED_PAYLOAD_77", "[Paste #1, 3 lines]")
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit_with_paste(pilot, app, draft, pasted)
        await _await_prompt(pilot, session)
        rows = _user_rows(app)
    assert session.prompts, "nothing reached the model"
    sent = session.prompts[0]
    assert "PASTED_PAYLOAD_77" in sent, "the pasted payload was dropped"
    assert "[Paste #1, 3 lines]" not in sent, "the model got the chip text instead of the paste"
    # The token survives the splice, so the session can still expand it. Losing
    # it here would trade one half of the draft for the other.
    assert "@auth.py" in sent
    assert REFERENCE_BLOCK_OPEN not in sent, "the TUI expanded; the gated pass cannot re-run"
    # R4 on the hardest row shape: the operator sees the token they typed, not
    # the file and not the pasted payload.
    assert rows, "no user row was painted"
    assert "@auth.py" in rows[0]
    assert "MARKER_CONTENT_42" not in rows[0]
    assert REFERENCE_BLOCK_OPEN not in rows[0]


@pytest.mark.asyncio
async def test_a_paste_with_NO_reference_still_reaches_the_model(workspace) -> None:
    """The control for the case above, and the behaviour that already worked.

    Asserted so a fix aimed at the reference case cannot quietly break the
    commonest paste path — which carries no ``@`` at all and is how most pastes
    are sent.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    pasted = PastedText("PASTED_PAYLOAD_77", "[Paste #1, 3 lines]")
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit_with_paste(pilot, app, "[Paste #1, 3 lines] hello", pasted)
        await _await_prompt(pilot, session)
    assert session.prompts, "nothing reached the model"
    sent = session.prompts[0]
    assert "PASTED_PAYLOAD_77" in sent
    assert REFERENCE_BLOCK_OPEN not in sent, "prose with no @token grew a block"


# --- The containment gate: who asks, who declines, who never asks ------------


@pytest.mark.asyncio
async def test_the_main_exit_hands_a_sensitive_token_over_UNEXPANDED(workspace) -> None:
    """Exit 1 does not expand, so the SESSION's gated pass is what sees `@.env`.

    THE TEST THAT WOULD HAVE CAUGHT THE ORIGINAL DEFECT. The TUI used to expand
    here with no `request_approval`, and `_approved` auto-approves a `None` gate
    by contract — so a deny-listed in-workspace file was read with no prompt and
    no notice, and the session's own pass could not catch it because idempotence
    skips any token already named in a block.

    `FakeSession` does not expand (it is a fake), which is what lets this
    distinguish "the TUI expanded it" from "the TUI passed it on".
    """
    (workspace / ".env").write_text("API_KEY=sk-NOT-FOR-THE-MODEL\n")
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "check @.env please")
        await _await_prompt(pilot, session)
    assert session.prompts, "the turn was swallowed"
    sent = session.prompts[0]
    assert "sk-NOT-FOR-THE-MODEL" not in sent, "a deny-listed file was read without approval"
    assert REFERENCE_BLOCK_OPEN not in sent, "the TUI expanded; the gated pass cannot re-run"
    assert "@.env" in sent, "the token must reach the session for its gated pass to see it"


@pytest.mark.asyncio
async def test_the_aside_DECLINES_a_sensitive_path_and_still_asks_the_question(
    workspace,
) -> None:
    """The aside cannot raise a card, so it refuses — and says so.

    `_expand_references` passes a gate that DECLINES rather than the interactive
    one, because awaiting the interactive gate from the aside worker cancels
    that worker: `request_tool_approval` closes the aside (`app.py:20177`) and
    `_close_aside` cancels the group the worker runs in (`app.py:34020`). The
    failure that produced was a question discarded in silence, so the property
    asserted FIRST here is that the question survives at all.
    """
    (workspace / ".env").write_text("API_KEY=sk-NOT-FOR-THE-MODEL\n")
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/btw what is in @.env ?")
        for _ in range(120):
            await pilot.pause()
            if session.asides:
                break
        notices = _notice_texts(app)
    assert session.asides, "the question was discarded — the worker was cancelled"
    asked = "\n".join(str(turn) for turn in session.asides[-1])
    assert "sk-NOT-FOR-THE-MODEL" not in asked, "a declined path reached the model anyway"
    assert "@.env" in asked, "the token is prose when it is not included"
    assert any(".env" in notice for notice in notices), notices
    assert app._approval is None, "the aside must not mount a card; that cancels its own worker"


@pytest.mark.asyncio
async def test_the_aside_still_expands_an_ORDINARY_file(workspace) -> None:
    """The regression guard on the decline being too broad.

    An in-workspace file that trips no deny-list rule never consults the gate at
    all (`_approved` returns True before it is reached), so the commonest aside
    — "what does this file do?" — must be untouched by the refusal above.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/btw what does @auth.py do?")
        for _ in range(120):
            await pilot.pause()
            if session.asides:
                break
    assert session.asides, "the aside was never asked"
    asked = "\n".join(str(turn) for turn in session.asides[-1])
    assert "MARKER_CONTENT_42" in asked, "an ordinary file was refused by the decline gate"

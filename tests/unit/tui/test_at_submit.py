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
from local_operator.tui.widgets.editor import Editor

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
async def test_the_main_prompt_exit_expands(workspace) -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "what does @auth.py do?")
        await _await_prompt(pilot, session)
    assert session.prompts, "nothing reached the model"
    sent = session.prompts[0]
    assert REFERENCE_BLOCK_OPEN in sent
    assert "MARKER_CONTENT_42" in sent


@pytest.mark.asyncio
async def test_the_transcript_row_shows_the_typed_line_not_the_block(workspace) -> None:
    """The display/sent split: the operator sees what they typed.

    Without it a one-line question about a file paints as the whole file — and,
    because a session is titled from its first user turn, titles the thread
    after the file too.
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
async def test_an_unresolved_token_paints_a_notice_and_sends_the_text_verbatim(
    workspace,
) -> None:
    """A token that resolves to nothing is PROSE, and is said out loud.

    Swallowing the request would be the worse half of the trade — the same
    bargain ``_expand_invocation`` strikes for an unreadable skill body.
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
    assert any("nope.py" in notice for notice in notices), notices


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

    The order is the whole point. ``_expand_invocation`` splices into the
    request ALONE, so references must already be in the request by the time it
    runs — expanding afterwards would reach the body, and a SKILL.md that
    happens to contain an ``@word`` is not making a reference.
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
    sent = session.prompts[0]
    # The REQUEST's reference expanded exactly once...
    assert sent.count(REFERENCE_BLOCK_OPEN) == 1
    assert "MARKER_CONTENT_42" in sent
    # ...and the body's `@auth.py` is still the literal text the skill author
    # wrote, not a second expansion.
    assert "See @auth.py for the shape." in sent


@pytest.mark.asyncio
async def test_expansion_is_awaited_outside_the_turn_lock(workspace) -> None:
    """STRUCTURAL, not timed: the lock is unheld while expansion runs.

    An approval can park on a human indefinitely, and the lock that serialises
    turns also serialises compaction — so awaiting a human decision inside it is
    a deadlock shape rather than a slow turn.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    observed: list[bool] = []

    real_expand = None

    async def _spy(text: str, cwd: str, **kwargs):
        # `_turn_lock` is the session's; record whether it is held at the moment
        # expansion runs, which is the property under test.
        lock = getattr(session, "_turn_lock", None)
        observed.append(bool(lock.locked()) if isinstance(lock, asyncio.Lock) else False)
        return await real_expand(text, cwd, **kwargs)

    import local_operator.tui.app as app_mod

    real_expand = app_mod.expand_references
    app_mod.expand_references = _spy
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(pilot, app)
            await _submit(pilot, app, "what does @auth.py do?")
            await _await_prompt(pilot, session)
    finally:
        app_mod.expand_references = real_expand

    assert observed, "expansion never ran on the submit path"
    assert not any(observed), "expansion awaited while the turn lock was held"

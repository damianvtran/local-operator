"""The ``$skill`` composer prefix: expansion at submit, and the picker.

Two properties carry the feature and are asserted separately here, because
they can regress independently: what the MODEL receives (the skill body plus
the request) and what the TRANSCRIPT shows (the short line the user typed).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.command_picker import (
    CompletionMode,
    PickerMode,
    completion_for,
    skill_token,
)
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import TranscriptView

from .test_app_pilot import FakeSession, _factory, _transcript_text


@pytest.fixture
def skill_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A skills root wired in as the ONLY root, via the documented env var."""
    root = tmp_path / "skills"
    for name, description, body, hide in [
        ("research", "Investigate a question.", "Read primary sources.", False),
        ("code-review", "Review a merge request.", "Check the diff.", False),
        ("secret-audit", "Audit credentials.", "Scan for secrets.", True),
    ]:
        skill_dir = root / name
        skill_dir.mkdir(parents=True)
        front = f"---\nname: {name}\ndescription: {description}\n"
        if hide:
            front += "hide: true\n"
        front += "---\n\n"
        (skill_dir / "SKILL.md").write_text(front + body)
    monkeypatch.setenv("LOCAL_OPERATOR_SKILL_EXTRA_ROOTS", str(root))
    # cwd walk-up must not pick up a real project root on the dev machine.
    monkeypatch.chdir(tmp_path)
    return root


async def _submit(app: OperatorApp, pilot, text: str) -> None:
    """Load a draft the way the suite does and send it.

    ``move_cursor(_end_of_buffer())`` is not decoration: ``load_text`` leaves
    the caret at offset 0, and every picker parse in this codebase is
    caret-anchored, so without it the caret sits INSIDE the ``$`` token and
    Enter completes the row instead of submitting — the same reason the
    existing pilot tests move the caret after loading a ``/`` draft.
    """
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    editor.load_text(text)
    editor.move_cursor(editor._end_of_buffer())
    await pilot.pause()
    await pilot.press("enter")


async def _await_prompt(pilot, session: FakeSession) -> None:
    for _ in range(200):
        await pilot.pause()
        if session.prompts:
            return


@pytest.mark.asyncio
async def test_invocation_sends_body_and_request(skill_root) -> None:
    """The model gets the SKILL.md body and the request; one prompt, not two."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _submit(app, pilot, "$research fix the login bug")
        await _await_prompt(pilot, session)
    assert len(session.prompts) == 1
    sent = session.prompts[0]
    assert "Read primary sources." in sent
    assert "fix the login bug" in sent
    assert "`research`" in sent


@pytest.mark.asyncio
async def test_transcript_shows_what_was_typed_not_the_body(skill_root) -> None:
    """The row is the user's line; the injected body never lands in the ledger."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _submit(app, pilot, "$research fix the login bug")
        await _await_prompt(pilot, session)
        await pilot.pause()
        shown = _transcript_text(app)
        assert "$research fix the login bug" in shown
        assert "Read primary sources." not in shown
        assert len(app.query_one(TranscriptView).blocks()) == 1


@pytest.mark.asyncio
async def test_bare_invocation_sends_the_skill_alone(skill_root) -> None:
    """`$research` + Enter completes the row; a second Enter sends it bare.

    Two keystrokes, deliberately, and the same shape ``/team <name>`` has: the
    first Enter acts on the open list (the name is a ROW, and acting on the
    highlighted row is what Enter means while a list is up), the second submits
    the buffer. Sending on the first would make the list unusable for its main
    purpose \u2014 choosing a skill and then typing the request.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _submit(app, pilot, "$research")
        await pilot.pause()
        editor = app.query_one(Editor)
        assert editor.text == "$research "
        assert session.prompts == []
        await pilot.press("enter")
        await _await_prompt(pilot, session)
    assert "Read primary sources." in session.prompts[0]


@pytest.mark.asyncio
async def test_hidden_skill_is_invocable_by_name(skill_root) -> None:
    """`hide` blocks semantic routing, not a user naming it outright."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _submit(app, pilot, "$secret-audit check the repo")
        await _await_prompt(pilot, session)
    assert "Scan for secrets." in session.prompts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    ["$100 for the redesign", "$unknown do the thing", "just a normal message"],
)
async def test_non_invocations_are_sent_verbatim(skill_root, text) -> None:
    """The money guard: an unmatched `$` token is prose and is not rewritten."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _submit(app, pilot, text)
        await _await_prompt(pilot, session)
    assert session.prompts == [text]


@pytest.mark.asyncio
async def test_picker_opens_on_the_sigil_and_lists_skills(skill_root) -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await pilot.press("dollar_sign")
        for _ in range(50):
            await pilot.pause()
            if editor.picker.is_open():
                break
        assert editor.picker.mode is PickerMode.SKILL
        names = [name for name, _ in editor.picker._matches]
        assert "research" in names
        # Hidden skills are offered: a name you can type but cannot see is a
        # worse secret than a listed one.
        assert "secret-audit" in names


@pytest.mark.asyncio
async def test_picker_closes_once_the_request_starts(skill_root) -> None:
    """Word phase only — the terminating space hands over to the request."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.load_text("$research ")
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        assert not editor.picker.is_open()


@pytest.mark.asyncio
async def test_enter_on_a_skill_row_completes_without_submitting(skill_root) -> None:
    """A completed `$skill ` opens a prompt; it is not itself a runnable thing."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.load_text("$resea")
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        for _ in range(50):
            await pilot.pause()
            if editor.picker.is_open():
                break
        await pilot.press("enter")
        await pilot.pause()
        assert editor.text == "$research "
        assert session.prompts == []


class TestSkillTokenParser:
    """The pure parse, independent of any app."""

    def test_bare_sigil_is_an_empty_query(self):
        token = skill_token("$")
        assert token is not None and token.query == ""

    def test_partial_name(self):
        token = skill_token("$res")
        assert token is not None and token.query == "res"

    def test_space_closes_the_token(self):
        assert skill_token("$research go") is None

    def test_only_at_the_buffer_start(self):
        """Mid-draft `$` is money or a shell variable, never an invocation."""
        assert skill_token("a $research") is None
        assert skill_token("costs $5") is None

    def test_caret_outside_the_token_closes_it(self):
        assert skill_token("$research go", 11) is None

    def test_completion_adds_the_terminating_space(self):
        assert completion_for("$res", 4, CompletionMode.SKILL, "research", ()) == (
            "$research ",
            10,
        )

    def test_completion_preserves_a_following_request(self):
        result = completion_for("$res fix bug", 4, CompletionMode.SKILL, "research", ())
        assert result is not None
        assert result[0] == "$research fix bug"

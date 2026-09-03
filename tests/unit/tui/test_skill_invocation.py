"""The ``$skill`` composer prefix: expansion at submit, and the picker.

Two properties carry the feature and are asserted separately here, because
they can regress independently: what the MODEL receives (the skill body plus
the request) and what the TRANSCRIPT shows (the short line the user typed).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.command_picker import (
    ArgumentChoice,
    CompletionMode,
    PickerMode,
    completion_for,
    ghost_for,
    skill_suggestions,
    skill_token,
    skill_token_is_leading,
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


@pytest.fixture
def shell_lookalike_skills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A vocabulary that COLLIDES with common shell variables, both ways.

    Separate from ``skill_root`` on purpose, and the names are the whole point:
    a shell-variable test run against a vocabulary the query cannot reach passes
    for a reason that has nothing to do with the gate under test, and reports a
    live defect as fixed. Two collision classes have to be reachable here:

    - SUBSEQUENCE — `LANG` reaches `planning` (l-a-n-g).
    - PREFIX, case-folded — `DEBUG`/`RESEARCH`/`HOME`/`PATH` are exact
      case-insensitive prefixes of `debug`/`research`/`home`/`path`. This class
      survived the first fix, because `"DEBUG".lower()` IS `"debug"`.

    `home` and `path` are named after environment variables deliberately: they
    are the worst realistic case, and the accepted lowercase caveat is asserted
    against `path` rather than left implicit.
    """
    root = tmp_path / "skills"
    for name, description, body in [
        ("planning", "Plan a piece of work.", "Break it into slices."),
        ("research", "Investigate a question.", "Read primary sources."),
        ("debug", "Diagnose a failure.", "Reproduce it first."),
        ("home", "Tidy the home directory.", "List what is there."),
        ("path", "Inspect the executable path.", "Print each entry."),
    ]:
        skill_dir = root / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n\n{body}"
        )
    monkeypatch.setenv("LOCAL_OPERATOR_SKILL_EXTRA_ROOTS", str(root))
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


class TestHandBackPathsReturnTheTypedLine:
    """Every path that hands a held prompt BACK must return what was typed.

    The `sent` split means an invocation's queued/held payload is the expanded
    SKILL.md body. Three paths consume those strings, and they do not all want
    the same half: the ones that SEND want the payload, the ones that hand the
    text back to the composer want the user's line. Getting it backwards makes
    the user delete a whole skill body by hand to recover one sentence, which
    is what these pin.
    """

    @pytest.mark.asyncio
    async def test_esc_recall_of_a_queued_steer_restores_the_typed_line(self, skill_root) -> None:
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one(Editor)
            # Mid-turn, so the submit takes the STEER branch.
            session.streaming = True
            await _submit(app, pilot, "$research fix the login bug")
            for _ in range(60):
                await pilot.pause()
                if app._held_steer_blocks:
                    break
            # The QUEUE carries the payload...
            queued = app._held_steer_blocks[0][0]
            assert "Read primary sources." in queued.text
            # ...and the recall gives back the line.
            app._recall_queued_steers()
            await pilot.pause()
            assert editor.text == "$research fix the login bug"
            assert "Read primary sources." not in editor.text

    @pytest.mark.asyncio
    async def test_compaction_hold_keeps_both_halves(self, skill_root) -> None:
        """The hold sends the body; the hand-back returns the line."""
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._compacting = True
            await _submit(app, pilot, "$research fix the login bug")
            await pilot.pause()
            assert "Read primary sources." in app._prompt_held_for_compaction
            assert app._typed_held_for_compaction == "$research fix the login bug"

    @pytest.mark.asyncio
    async def test_reload_during_a_hold_hands_back_the_typed_line(self, skill_root) -> None:
        """The `/reload` teardown returns the draft, so it must return the LINE.

        Driven through the real teardown rather than the two fields, because
        the defect was which of them that code path read.
        """
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one(Editor)
            app._compacting = True
            await _submit(app, pilot, "$research fix the login bug")
            await pilot.pause()
            assert not editor.text
            await app._reload_session()
            await pilot.pause()
            assert editor.text == "$research fix the login bug"
            assert "Read primary sources." not in editor.text

    @pytest.mark.asyncio
    async def test_ordinary_prompt_holds_no_separate_typed_line(self, skill_root) -> None:
        """`""` stays the sentinel for "the two strings are the same"."""
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._compacting = True
            await _submit(app, pilot, "just a normal message")
            await pilot.pause()
            assert app._prompt_held_for_compaction == "just a normal message"
            assert app._typed_held_for_compaction == ""

    @pytest.mark.asyncio
    async def test_ordinary_steer_recall_is_unchanged(self, skill_root) -> None:
        """The non-invocation path must behave exactly as it did before."""
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one(Editor)
            session.streaming = True
            await _submit(app, pilot, "please also check the logs")
            for _ in range(60):
                await pilot.pause()
                if app._held_steer_blocks:
                    break
            app._recall_queued_steers()
            await pilot.pause()
            assert editor.text == "please also check the logs"


class TestReviewRegressions:
    """Defects found in independent review of the first commit."""

    @pytest.mark.asyncio
    async def test_compaction_resume_orders_images_by_the_typed_citations(self, skill_root) -> None:
        """`resolve_markers` must read the TYPED line, not the payload.

        Images are ordered by WHERE the citation sits. A `[Image #N]` anywhere
        in the SKILL.md body (a skill about screenshots is the obvious case)
        shifts those positions, so resolving against the expanded payload sent
        the model the attachments in the wrong order — silently.

        Driven through the real `on_compaction_ended` call site, not through
        `resolve_markers` directly: the defect was WHICH STRING that call site
        passed, so a test that calls the helper itself cannot catch it.
        """
        from local_operator.tui.events import CompactionEnded
        from local_operator.tui.widgets.editor import Attachment

        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            # A skill whose BODY cites an image, which is what shifts the order.
            app._skills_by_name = None
            app._compacting = True
            editor = app.query_one(Editor)
            editor.focus()
            await pilot.pause()
            # Hold a prompt the way a mid-compaction submit does, with the
            # payload and the typed line diverging exactly as they do live.
            typed = "$research compare [Image #2] against [Image #1]"
            app._prompt_held_for_compaction = (
                'lead\n<skill name="research" invocation="x">See [Image #1].</skill>\n' + typed
            )
            app._typed_held_for_compaction = typed
            app._images_held_for_compaction = {
                1: Attachment(cast(Any, "AAA"), "[Image #1]"),
                2: Attachment(cast(Any, "BBB"), "[Image #2]"),
            }
            app.on_compaction_ended(CompactionEnded("manual", True, "snapcompact", 10, 5))
            for _ in range(200):
                await pilot.pause()
                if session.prompts:
                    break
            # Cited #2 first, so #2's image must be sent first.
            assert session.prompt_images[0] == ["BBB", "AAA"]

    @pytest.mark.asyncio
    async def test_replay_repaints_the_typed_line_not_the_body(self, skill_root) -> None:
        """A resumed session must show what the live session showed.

        Driven through `_replay_history`, the real call site: the defect was
        that replay painted the persisted payload verbatim.
        """
        from local_operator.skills.api import default_skill_roots
        from local_operator.skills.discovery import discover_skills
        from local_operator.skills.invoke import parse_invocation, render_invocation

        skills, _ = discover_skills(default_skill_roots(Path(os.getcwd())))
        invocation = parse_invocation("$research fix the login bug", {s.name: s for s in skills})
        assert invocation is not None
        payload = render_invocation(invocation, "Read primary sources.")

        session = FakeSession()
        # `history` is a read-only property backed by `_history` on the fake.
        session._history = [SimpleNamespace(role="user", text=payload, content=[])]
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._project_settled_rows(list(session._history))
            await pilot.pause()
            shown = _transcript_text(app)
            assert "$research fix the login bug" in shown
            assert "Read primary sources." not in shown

    @pytest.mark.asyncio
    async def test_empty_skill_body_warns_instead_of_silently_sending_prose(
        self, tmp_path, monkeypatch
    ) -> None:
        """The user must not believe a skill fired when it did not."""
        root = tmp_path / "skills"
        (root / "stub").mkdir(parents=True)
        (root / "stub" / "SKILL.md").write_text("---\nname: stub\ndescription: A stub.\n---\n")
        monkeypatch.setenv("LOCAL_OPERATOR_SKILL_EXTRA_ROOTS", str(root))
        monkeypatch.chdir(tmp_path)

        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await _submit(app, pilot, "$stub do the thing")
            await _await_prompt(pilot, session)
            await pilot.pause()
            # Sent as written, and SAID SO.
            assert session.prompts == ["$stub do the thing"]
            assert "empty body" in _transcript_text(app)

    @pytest.mark.asyncio
    async def test_indented_draft_offers_the_list_it_will_act_on(self, skill_root) -> None:
        """The picker and the submit parser must agree about leading spaces."""
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one(Editor)
            editor.focus()
            await pilot.pause()
            editor.load_text("  $resea")
            editor.move_cursor(editor._end_of_buffer())
            await pilot.pause()
            for _ in range(50):
                await pilot.pause()
                if editor.picker.is_open():
                    break
            assert editor.picker.is_open(), "indented draft expanded but offered no list"
            await pilot.press("enter")
            await pilot.pause()
            assert editor.text == "  $research "


class TestInlineInvocation:
    """`$` reaches the picker anywhere `/` does, and reassembles like it.

    The composer, not the submit parser, is where the position rule now lives:
    `parse_invocation` stays anchored (see the last test in this class), and
    accepting an inline row moves the token to the front so it still sees a
    prefix.
    """

    async def _draft(self, app: OperatorApp, pilot, text: str) -> Editor:
        """Type ``text`` into a focused composer with the caret at its end."""
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.load_text(text)
        editor.move_cursor(editor._end_of_buffer())
        for _ in range(50):
            await pilot.pause()
            if editor.picker.is_open():
                break
        return editor

    @pytest.mark.asyncio
    async def test_a_token_after_prose_opens_the_list(self, skill_root) -> None:
        """The case the feature exists for: remembering the skill afterwards."""
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, "fix the flaky login test $res")
            assert editor.picker.is_open(), "inline `$` offered no list"
            assert editor.picker.mode is PickerMode.SKILL
            assert "research" in [name for name, _ in editor.picker._matches]

    @pytest.mark.asyncio
    async def test_a_token_opening_a_later_line_opens_the_list(self, skill_root) -> None:
        """Dropping the sigil on its own line below the draft works too."""
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, "fix the flaky login test\n$res")
            assert editor.picker.is_open()
            assert editor.picker.mode is PickerMode.SKILL

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", ["a$res", "costs$5"])
    async def test_a_glued_sigil_never_opens_the_list(self, skill_root, text) -> None:
        """The boundary rule, at the surface the user actually sees."""
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, text)
            assert not editor.picker.is_open()

    @pytest.mark.asyncio
    async def test_money_at_a_boundary_opens_nothing_because_nothing_matches(
        self, skill_root
    ) -> None:
        """`costs $5` parses as a token and closes on the EMPTY match set.

        The parser-level half is asserted in `TestSkillTokenParser`; this is the
        half the user sees, and it is why no digit guard is needed: the
        vocabulary decides, exactly as it always has.
        """
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, "costs $5")
            assert not editor.picker.is_open()

    @pytest.mark.asyncio
    async def test_a_sigil_inside_an_engaged_command_is_plain_text(self, skill_root) -> None:
        """`/team ops $research` is a request about a skill, not an invocation.

        The arbitration case that only became possible when `$` went inline.
        """
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, "/team ops $res")
            assert editor.picker.mode is not PickerMode.SKILL

    @pytest.mark.asyncio
    async def test_enter_reassembles_the_draft_and_stages_it(self, skill_root) -> None:
        """Accepting an inline row stages `$research <draft> ` — and sends nothing.

        The whole inline contract in one keystroke: the token moves to the front
        so the anchored submit parser can read it, the draft becomes the request
        rather than being consumed as a name, and the buffer is STAGED for the
        user to read and send themselves.
        """
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, "fix the flaky login test $res")
            assert editor.picker.is_open()
            await pilot.press("enter")
            await pilot.pause()
            assert editor.text == "$research fix the flaky login test "
            assert session.prompts == [], "a skill row must never submit"
            # Caret at the end, where the request continues.
            assert editor._caret_offset() == len(editor.text)

    @pytest.mark.asyncio
    async def test_the_staged_line_then_sends_as_an_invocation(self, skill_root) -> None:
        """Enter on the staged line fires the skill with the draft as its request.

        Drives the whole path — type inline, engage, send — because the point of
        reassembly is that the ANCHORED submit parser reads what it produced.
        """
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await self._draft(app, pilot, "fix the flaky login test $res")
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("enter")
            await _await_prompt(pilot, session)
        assert len(session.prompts) == 1
        sent = session.prompts[0]
        assert "Read primary sources." in sent
        assert "fix the flaky login test" in sent

    @pytest.mark.asyncio
    async def test_a_mid_message_sigil_submitted_raw_is_still_prose(self, skill_root) -> None:
        """`parse_invocation` stays ANCHORED; only the composer went inline.

        The property the anchoring buys, asserted at the submit path rather than
        at the parser: a `$research` that was never engaged through the picker —
        a pasted document, a sentence about a skill — must not fire one.
        """
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        text = "we should use $research for this"
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await _submit(app, pilot, text)
            await _await_prompt(pilot, session)
        assert session.prompts == [text]

    @pytest.mark.asyncio
    async def test_a_bare_prefix_still_behaves_exactly_as_before(self, skill_root) -> None:
        """No draft outside the token means no reassembly: unchanged behaviour."""
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, "$resea")
            await pilot.press("enter")
            await pilot.pause()
            assert editor.text == "$research "
            assert session.prompts == []


class TestInlineShellVariablesAreNotInvocations:
    """An inline `$VAR` must never take the Enter that was meant to send.

    The boundary rule alone does not tell a shell variable from an invocation,
    and a matching row turns that gap into DATA LOSS: a non-empty match set keeps
    the SKILL list open, an open SKILL list makes Enter complete a row instead of
    submitting, and the draft is rewritten to `$<skill> <prose> ` with nothing
    sent and no undo.

    Two rounds of review found two distinct classes reaching a row:

    - SUBSEQUENCE — `translate to $LANG` scored `LANG` against `planning`.
    - PREFIX COLLISION — `echo $DEBUG` against `debug`, which survived a
      case-INSENSITIVE prefix gate because `"DEBUG".lower()` IS `"debug"`.

    Case-sensitive prefix matching inline removes both, on the convention that
    shell variables are UPPERCASE and skill names lowercase. Leading-position
    behaviour is untouched: a `$` typed first is unambiguous, so fuzzy AND
    case-insensitive matching both survive there.

    Every case below runs against `shell_lookalike_skills`, whose vocabulary
    CONTAINS the colliding names — the previous version of these tests passed
    against a vocabulary that could not reach them, which is why it reported a
    live defect as fixed.
    """

    async def _draft(self, app: OperatorApp, pilot, text: str) -> Editor:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.load_text(text)
        editor.move_cursor(editor._end_of_buffer())
        for _ in range(50):
            await pilot.pause()
            if editor.picker.is_open():
                break
        return editor

    @pytest.mark.asyncio
    async def test_enter_on_an_inline_shell_variable_sends_the_prose(
        self, shell_lookalike_skills
    ) -> None:
        """THE regression assertion: the prompt goes out verbatim.

        Asserted on what was SENT rather than on the picker's state, because the
        defect's cost was the send that never happened. A test that only checked
        `is_open()` would pass against a build that still swallowed the Enter.
        """
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        text = "echo $DEBUG"
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, text)
            assert not editor.picker.is_open(), "an inline `$VAR` opened a skill list"
            await pilot.press("enter")
            await _await_prompt(pilot, session)
        assert session.prompts == [text]

    @pytest.mark.asyncio
    async def test_enter_on_a_bare_trailing_sigil_sends_the_prose(
        self, shell_lookalike_skills
    ) -> None:
        """`the price is $` + Enter must SEND, not open the catalogue.

        The empty-query hole: it returned the whole vocabulary inline, so Enter
        completed row 0 and the draft became `$debug the price is `. Asserted on
        the send, like its siblings, because the cost was the send that never
        happened.
        """
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        text = "the price is $"
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, text)
            assert not editor.picker.is_open(), "a bare inline `$` opened the catalogue"
            await pilot.press("enter")
            await _await_prompt(pilot, session)
        assert session.prompts == [text]

    @pytest.mark.asyncio
    async def test_the_list_arrives_one_keystroke_into_an_inline_name(
        self, shell_lookalike_skills
    ) -> None:
        """`$` shows nothing inline; `$r` shows the matches.

        The deliberate cost of closing the empty-query hole, pinned so the
        one-keystroke delay is a decision rather than a surprise. The feature
        still works — the list appears long before the name is finished.
        """
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, "fix this $")
            assert not editor.picker.is_open()
            editor.load_text("fix this $r")
            editor.move_cursor(editor._end_of_buffer())
            for _ in range(50):
                await pilot.pause()
                if editor.picker.is_open():
                    break
            assert editor.picker.is_open(), "the list never arrived for an inline name"
            assert "research" in [name for name, _ in editor.picker._matches]

    @pytest.mark.asyncio
    async def test_a_bare_leading_sigil_still_browses_the_catalogue(
        self, shell_lookalike_skills
    ) -> None:
        """`$` at position 0 is an unambiguous ask — existing behaviour."""
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, "$")
            assert editor.picker.is_open(), "a leading `$` stopped browsing the catalogue"
            assert len(editor.picker._matches) == 5

    @pytest.mark.asyncio
    async def test_enter_on_a_fuzzy_cousin_also_sends_the_prose(
        self, shell_lookalike_skills
    ) -> None:
        """The subsequence class, asserted on the send for the same reason."""
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        text = "translate to $LANG"
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, text)
            assert not editor.picker.is_open()
            await pilot.press("enter")
            await _await_prompt(pilot, session)
        assert session.prompts == [text]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text",
        [
            # Prefix collisions: each query IS the skill name once case-folded,
            # so every one of these opened a list before the case rule. No `=1`
            # or other suffix — that made the query miss for reasons unrelated
            # to the gate, which is how this defect shipped as fixed.
            "echo $DEBUG",
            "unset $DEBUG",
            "use $Debug",
            "run with $RESEARCH",
            "echo $PATH",
            "echo $HOME",
            # Subsequence: `LANG` against `planning`.
            "translate to $LANG",
        ],
    )
    async def test_no_list_opens_for_an_inline_shell_variable(
        self, shell_lookalike_skills, text
    ) -> None:
        """Each query CAN reach a skill in this vocabulary, and must not."""
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, text)
            assert not editor.picker.is_open()

    @pytest.mark.asyncio
    async def test_an_inline_prefix_still_opens_the_list(self, shell_lookalike_skills) -> None:
        """The case the inline feature exists for is untouched by the gate."""
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, "fix this $re")
            assert editor.picker.is_open()
            assert editor.picker.mode is PickerMode.SKILL
            assert [name for name, _ in editor.picker._matches] == ["research"]

    @pytest.mark.asyncio
    async def test_a_leading_token_keeps_fuzzy_matching(self, shell_lookalike_skills) -> None:
        """`$rsrch` at position 0 is unambiguous, so typo tolerance survives."""
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, "$rsrch")
            assert editor.picker.is_open(), "the leading position lost fuzzy matching"
            assert "research" in [name for name, _ in editor.picker._matches]

    @pytest.mark.asyncio
    async def test_a_leading_token_keeps_case_insensitive_matching(
        self, shell_lookalike_skills
    ) -> None:
        """`$Research` at a sentence start must keep working.

        The case rule is scoped to INLINE precisely so this stays true: at the
        leading position the sigil's own position is the signal, so case carries
        no information and a capitalised sentence start is ordinary typing.
        """
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, "$Research")
            assert editor.picker.is_open(), "the leading position lost case-insensitivity"
            assert "research" in [name for name, _ in editor.picker._matches]

    def test_the_gate_is_ranked_then_filtered_not_rescored(self):
        """Prefix rows keep the shared scorer's ORDER; the gate only cuts."""
        choices = [ArgumentChoice(name, "") for name in ["planning", "research", "debug"]]
        # Inline: `LANG` reaches `planning` fuzzily and must be excluded...
        assert skill_suggestions("LANG", choices, inline=True) == []
        # ...as must `DEBUG`, which reaches `debug` by an exact case-folded
        # prefix rather than by subsequence — the class the first fix missed.
        assert skill_suggestions("DEBUG", choices, inline=True) == []
        # ...while the leading position still finds them by the same queries.
        assert [name for name, _ in skill_suggestions("rsrch", choices, inline=False)] == [
            "research"
        ]
        assert [name for name, _ in skill_suggestions("DEBUG", choices, inline=False)] == ["debug"]
        # A bare `$` INLINE now offers nothing — see
        # `test_an_empty_inline_query_offers_nothing` for why an empty query
        # carries no evidence. It remains the whole vocabulary when LEADING.
        assert skill_suggestions("", choices, inline=True) == []
        assert len(skill_suggestions("", choices, inline=False)) == 3

    def test_a_lowercase_collision_is_the_accepted_caveat(self):
        """A lowercase token that PREFIXES a skill name matches — by design.

        Pinned at its TRUE WIDTH rather than as "a skill named exactly `path`":
        this is prefix matching, so `pathfinder` catches `$path` and
        `language-tutor` catches `$lang`. Lowercase environment variables like
        `$http_proxy` are genuinely in this class. Accepted, not unnoticed —
        closing it would need a rule outranking the user's own vocabulary.
        """
        wide = [ArgumentChoice(name, "") for name in ["pathfinder", "language-tutor"]]
        assert [n for n, _ in skill_suggestions("path", wide, inline=True)] == ["pathfinder"]
        assert [n for n, _ in skill_suggestions("lang", wide, inline=True)] == ["language-tutor"]
        # The UPPERCASE form a shell actually uses is what the gate stops.
        assert skill_suggestions("PATH", wide, inline=True) == []

    def test_an_empty_inline_query_offers_nothing(self):
        """A bare trailing `$` in prose is money, not a request for the list.

        It returned the WHOLE vocabulary before, so Enter completed row 0 and
        `the price is $` became `$planning the price is `. No case rule can
        catch this: an empty query has no case to inspect, which is why the gate
        tests for POSITIVE lowercase evidence instead of for uppercase.
        """
        choices = [ArgumentChoice(name, "") for name in ["planning", "research"]]
        assert skill_suggestions("", choices, inline=True) == []
        # Leading keeps browsing: `$` typed first is an unambiguous ask.
        assert len(skill_suggestions("", choices, inline=False)) == 2

    def test_caseless_tokens_offer_nothing_inline(self):
        """Digits and underscores carry no case, so they carry no evidence.

        `costs $5` against a skill named `5things` is the money case that
        justifies this whole grammar being narrow, and it is closed by the same
        clause as the empty query rather than by a special case.

        `$_private` is NOT in this class and must not be asserted into it: it
        contains lowercase letters, so it is the accepted prefix caveat above.
        Only the genuinely caseless prefix `$_` is closed here.
        """
        choices = [ArgumentChoice(name, "") for name in ["5things", "_private"]]
        assert skill_suggestions("5", choices, inline=True) == []
        assert skill_suggestions("_", choices, inline=True) == []
        assert skill_suggestions("_PRIVATE", choices, inline=True) == []
        # The lowercase-bearing form is the documented caveat, not a defect.
        assert [n for n, _ in skill_suggestions("_private", choices, inline=True)] == ["_private"]

    def test_an_uppercase_named_skill_does_not_void_the_guard(self):
        """A skill really named `DEBUG` must not make `$DEBUG` match again.

        The rule reads evidence off the QUERY, never off the vocabulary, because
        nothing enforces lowercase names: `discovery` takes frontmatter `name`
        or the directory name with only `.strip()`. Resting the guard on a
        convention the code does not enforce is what this pins shut.
        """
        choices = [ArgumentChoice(name, "") for name in ["DEBUG", "AWS_PROFILE_SWITCHER"]]
        assert skill_suggestions("DEBUG", choices, inline=True) == []
        assert skill_suggestions("AWS_PROFILE", choices, inline=True) == []
        # Leading position is unchanged: position is the signal there.
        assert [n for n, _ in skill_suggestions("DEBUG", choices, inline=False)] == ["DEBUG"]

    def test_unicode_hazards_stay_closed(self):
        """`startswith` does no case folding, so the folding traps cannot arise.

        Recorded as a REASON, not a coincidence: `casefold` would map `ß` to
        `ss` and the Turkish dotless `ı`/`İ` across the ASCII `i`, which is how
        a "tidier" comparison would reopen this. Do not swap it.

        The `Café`/`café-audit` pair is the one that actually DISCRIMINATES the
        mutation, and it is the whole point of this test. The mixed-case query
        subsequence-matches, so the scorer surfaces the row and it reaches the
        gate; `Café` carries a lowercase letter, so condition 1 lets it through
        to the comparison; `"café-audit".startswith("Café")` is False, so the
        row is dropped. A `casefold` comparison would map both sides to lower and
        reopen it — verified: swapping `startswith` for a folding compare turns
        this assertion red, where the `İ`/`STRASSE` cases below would stay green
        either way because they carry no lowercase letter and condition 1 closes
        them before the comparison is ever reached.
        """
        # The discriminating case: the scorer surfaces the row, condition 1 lets
        # it reach the comparison, and only `startswith` (not `casefold`) keeps
        # it closed. This is the assertion the casefold mutation fails.
        assert skill_suggestions("Café", [ArgumentChoice("café-audit", "")], inline=True) == []
        # These carry no lowercase letter, so condition 1 closes them before the
        # comparison — included as the shape of the hazard, not as its guard.
        closed_before_comparison = [ArgumentChoice(name, "") for name in ["istanbul", "strasse"]]
        for query in ["İ", "STRASSE"]:
            assert skill_suggestions(query, closed_before_comparison, inline=True) == []

    def test_leading_is_whitespace_tolerant(self):
        """An indented `  $rsrch` is still the buffer's first token.

        Pins the predicate both the fuzzy gate and the reassembly read, so they
        cannot drift apart on the definition of "leading".
        """
        token = skill_token("  $rsrch")
        assert token is not None
        assert skill_token_is_leading("  $rsrch", token)
        inline_token = skill_token("fix $rsrch")
        assert inline_token is not None
        assert not skill_token_is_leading("fix $rsrch", inline_token)


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

    def test_inline_after_prose_opens_the_token(self):
        """A `$` at a word boundary mid-draft IS a token now.

        This used to assert ``None``: the token was anchored at the buffer
        start, so reaching for a skill after writing the request meant retyping
        the draft around the sigil. The submit-side parser is still anchored —
        what changed is that the COMPOSER reassembles the token to the front
        before Enter, so the picker can open wherever a boundary allows.
        """
        token = skill_token("a $research")
        assert token is not None and token.query == "research"

    def test_the_boundary_rule_is_what_replaced_the_position_rule(self):
        """A `$` glued to a word is punctuation, not a sigil.

        The guard that used to be "offset 0 only" is now the same word-boundary
        rule `/` has always used, and it is what keeps inline detection safe to
        run on every keystroke of ordinary prose.
        """
        assert skill_token("a$b") is None
        assert skill_token("costs$5") is None

    def test_money_after_a_space_is_left_to_the_vocabulary(self):
        """`costs $5` opens a token whose query matches NO skill.

        Deliberately closed at the PICKER rather than at the parser. `5` is a
        boundary token like any other, and the module's standing rule is that
        the vocabulary decides what is an invocation — a digit guard here would
        be a second, weaker rule saying the same thing. The picker test below
        pins the user-visible half: no list opens.
        """
        token = skill_token("costs $5")
        assert token is not None and token.query == "5"

    def test_a_terminated_command_claims_the_rest_of_its_line(self):
        """`$` inside an engaged command's argument is plain text.

        The arbitration that replaced "a `$` is anchored at offset 0, so it
        cannot overlap a slash construct". `/team ops $research` is possible now,
        and the claiming rule `_active_slash` already implements for nested
        slashes decides it: a recognised, TERMINATED command owns the rest of its
        line as its argument.
        """
        commands = frozenset({"team", "model"})
        assert skill_token("/team ops $research", None, commands) is None
        # The claim needs BOTH halves. An UNRECOGNISED word claims nothing, so
        # the `$` after it is an ordinary boundary token.
        token = skill_token("/teem $research", None, commands)
        assert token is not None and token.query == "research"
        # And with no vocabulary at all — the pure-parser default — nothing can
        # be recognised, so nothing claims.
        assert skill_token("/team ops $research") is not None

    def test_a_token_opening_a_later_line_is_found(self):
        """A `$` at the start of line 2 is a boundary token on ITS line."""
        token = skill_token("fix the bug\n$res")
        assert token is not None and token.query == "res"

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

    def test_inline_completion_reassembles_the_draft_to_the_front(self):
        """The token moves to the front with the surviving draft as its request.

        The `$` twin of `/team`'s inline engage, and the reason the submit-side
        parser can stay anchored: by the time Enter is pressed the token IS the
        prefix. Staged, not submitted — the caret lands at the end so the user
        can keep writing.
        """
        text = "fix this $res"
        result = completion_for(text, len(text), CompletionMode.SKILL, "research", ())
        assert result == ("$research fix this ", len("$research fix this "))

    def test_inline_completion_across_lines_reassembles_too(self):
        """A `$` opening line 2 collapses onto one staged line."""
        text = "fix this\n$res"
        result = completion_for(text, len(text), CompletionMode.SKILL, "research", ())
        assert result is not None
        assert result[0] == "$research fix this "

    def test_a_leading_token_is_not_reassembled(self):
        """Already the prefix, so the request is preserved where it was typed.

        Reassembly is only for prose BEFORE the token — the shape an anchored
        submit parser cannot read. Rebuilding a leading token's request would
        churn a buffer that is already correct.
        """
        text = "$res fix bug"
        result = completion_for(text, 4, CompletionMode.SKILL, "research", ())
        assert result == ("$research fix bug", len("$research") + 1)

    def test_the_reassembled_ghost_is_withheld(self):
        """A reordering is not an append, so no honest ghost describes it.

        The consequence `completion_for` already documents for NAME_ARGUMENT,
        asserted for SKILL: `ghost_for`'s `startswith` rule declines of its own
        accord rather than painting characters Tab does not produce.
        """
        text = "fix this $res"
        completion = completion_for(text, len(text), CompletionMode.SKILL, "research", ())
        assert ghost_for(completion, text) == ""
        # The bare-prefix case IS an append, and still previews.
        bare = completion_for("$res", 4, CompletionMode.SKILL, "research", ())
        assert ghost_for(bare, "$res") == "earch "

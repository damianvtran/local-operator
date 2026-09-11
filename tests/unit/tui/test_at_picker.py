"""The ``@path`` composer sigil: when the picker opens, and what accepting inserts.

Split from the submit half (``test_at_submit.py``) because the two regress
independently: this file is about the COMPOSER — which keystrokes open a list,
which never do, and what lands in the buffer when a row is chosen — and asserts
nothing about what the model receives.

The defence that matters most here is the negative one. An ``@`` is an ordinary
character in an email address, a decorator and a ``glab`` flag, so the tests
that assert the picker does NOT open carry as much weight as the ones that
assert it does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.autocomplete import ArgumentChoice
from local_operator.tui.widgets.command_picker import (
    CompletionMode,
    PickerMode,
    completion_for,
    file_suggestions,
    ghost_for,
)
from local_operator.tui.widgets.editor import Editor

from .test_app_pilot import FakeSession, _factory


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A small tree with the shapes the tests need, as the process cwd.

    Two levels deep on purpose: ``test_deepening_a_second_level_relists`` needs
    ``src/sub/`` to hold a name that appears nowhere in ``src/``, or a stale
    listing would still contain the row it looks for and the test would pass
    against the very bug it exists to catch.

    The name with a space is here rather than in its own fixture because the
    quoting rule is a property of this same listing — a directory where one
    entry needs quoting and its neighbours do not.
    """
    (tmp_path / "src" / "sub").mkdir(parents=True)
    (tmp_path / "src" / "app.py").write_text("app\n")
    (tmp_path / "src" / "sub" / "deep.py").write_text("deep\n")
    (tmp_path / "README.md").write_text("readme\n")
    (tmp_path / "my file.txt").write_text("spaces\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


async def _draft(app: OperatorApp, pilot, text: str, caret: int | None = None) -> Editor:
    """Put ``text`` in the composer with the caret at ``caret`` (default: end).

    The caret move is not decoration: ``load_text`` leaves the caret at offset 0
    and every picker parse in this codebase is caret-anchored, so a draft loaded
    without it parses as though the user had typed nothing. Same reason the
    ``$`` and ``/`` pilot tests move the caret after loading.

    ``caret`` is a whole-buffer OFFSET, converted through the editor's own
    ``_location_at_offset`` because ``move_cursor`` takes a ``(row, column)``
    location — the same conversion ``test_command_picker.py:582`` uses.
    """
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    editor.load_text(text)
    editor.move_cursor(
        editor._end_of_buffer() if caret is None else editor._location_at_offset(caret)
    )
    await pilot.pause()
    # The app answers `FileQueryOpened` one message-loop tick later, so a single
    # pause would observe the picker mid-fill — open in principle, empty in fact.
    await pilot.pause()
    return editor


def _rows(editor: Editor) -> list[str]:
    return [name for name, _ in editor.picker.suggestions()]


# --- Opening, and the far more important not-opening -------------------------


@pytest.mark.asyncio
async def test_a_boundary_at_opens_the_picker(workspace) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "look at @RE")
        assert editor.picker.mode is PickerMode.FILE
        assert editor.picker.is_open()
        assert "README.md" in _rows(editor)


@pytest.mark.asyncio
async def test_an_email_address_never_opens_the_picker(workspace) -> None:
    """At EVERY caret position, not just the end.

    An email address is the single most common ``@`` in ordinary prose, and the
    boundary rule is the whole defence. Sweeping every caret is what makes this
    a real guard: the picker could be closed at the end of the word and open
    with the caret parked just after the ``@``.
    """
    text = "mail ben@host.com now"
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        for caret in range(len(text) + 1):
            editor = await _draft(app, pilot, text, caret)
            assert (
                editor.picker.mode is not PickerMode.FILE or not editor.picker.is_open()
            ), f"picker opened on an email address with the caret at {caret}"


@pytest.mark.asyncio
async def test_at_glued_to_a_word_never_opens_the_picker(workspace) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "a@b")
        assert not editor.picker.is_open() or editor.picker.mode is not PickerMode.FILE


@pytest.mark.asyncio
async def test_at_me_closes_when_nothing_in_the_directory_matches(workspace) -> None:
    """Design R1's case, asserted against what the MATCHER actually does.

    ``@me`` is a boundary ``@``, so the list opens; whether it then closes is a
    question about the fuzzy scorer, not about the ``@`` grammar. The picker
    routes through ``argument_suggestions``, whose documented short-query rule
    is ``prefixed or matches`` — so a SUBSEQUENCE hit keeps the list open when
    no prefix matches. In a directory holding ``README.md``, ``@me`` is such a
    hit (m…e), and the list staying open is the ranking behaviour every other
    list in the app has, not a failure of the ``@`` mitigation.

    What R1 actually promises is that the list closes when the directory holds
    nothing the query can reach, which is asserted here against a directory
    that does. The prose-safety property that matters — ``@me`` never inserting
    anything or submitting on its own — is covered by the tests above and by
    the no-submit test below.
    """
    empty = workspace / "empty"
    empty.mkdir()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "glab mr create --assignee @empty/me")
        assert not editor.picker.is_open()


# --- Listing, and deepening --------------------------------------------------


@pytest.mark.asyncio
async def test_a_bare_at_lists_the_cwd(workspace) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@")
        assert editor.picker.mode is PickerMode.FILE
        assert "README.md" in _rows(editor)
        assert "src/" in _rows(editor)


@pytest.mark.asyncio
async def test_typing_a_segment_deepens_into_that_directory(workspace) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@src/")
        assert "app.py" in _rows(editor)


@pytest.mark.asyncio
async def test_deepening_a_second_level_relists(workspace) -> None:
    """``@src/`` then ``@src/sub/`` lists ``sub``, not ``src``.

    The test for the directory re-arm. A latch keyed on the TOKEN — which is
    all the ``$skill`` list needs, because a session has one skill vocabulary —
    posts ``FileQueryOpened`` once and then never again while the token stands,
    so the app keeps answering with the FIRST directory it was asked about and
    the user types deeper into a listing of the parent, forever.

    ``deep.py`` exists only in ``src/sub/``, so a stale listing cannot contain
    it by accident.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@src/")
        assert "app.py" in _rows(editor)
        # Same token, deeper directory — the case a token-keyed latch misses.
        editor = await _draft(app, pilot, "@src/sub/")
        rows = _rows(editor)
        assert "deep.py" in rows
        assert "app.py" not in rows


@pytest.mark.asyncio
async def test_escape_latches_the_dismissal(workspace) -> None:
    """Esc closes the list, and a LATER ``@`` opens a fresh one.

    Both halves matter: a dismissal that does not latch reopens on the next
    keystroke, and one that latches too hard makes the feature unreachable for
    the rest of the draft.

    The second half TYPES the rest of the draft rather than reloading the
    buffer. ``load_text`` replaces the text wholesale, which is not a keystroke
    and does not re-sync the dismissal the way a typed character does; asserting
    against it would be asserting about the test harness rather than about what
    a user experiences.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@RE")
        assert editor.picker.is_open()
        await pilot.press("escape")
        await pilot.pause()
        assert not editor.picker.is_open()
        for key in ("space", "a", "n", "d", "space", "at", "s", "r"):
            await pilot.press(key)
        await pilot.pause()
        await pilot.pause()
        assert editor.picker.is_open(), "a later @ did not open a fresh list"
        assert editor.picker._query == "sr"


# --- Accepting ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_ghost_previews_the_completion(workspace) -> None:
    """And ``buffer + ghost == buffer_after_tab``.

    The invariant ``completion_for`` exists to make structural rather than to
    maintain by agreement between two string builders: the ghost is derived
    from the same call Tab commits.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@READ")
        ghost = editor._ghost_completion()
        assert ghost
        before = editor.text
        await pilot.press("tab")
        await pilot.pause()
        assert before + ghost == editor.text


@pytest.mark.asyncio
async def test_arrows_wrap_at_both_ends(workspace) -> None:
    """Arrows WRAP (the wheel clamps, which is a different surface).

    ``command_picker.move`` wraps for every other list and must keep wrapping
    for this one; a mode that clamped would be a rule the user has to learn per
    list.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@")
        assert len(editor.picker.suggestions()) > 1
        first = editor.picker.highlighted_name()
        await pilot.press("up")
        await pilot.pause()
        assert editor.picker.highlighted_name() != first, "up from row 0 did not wrap"
        await pilot.press("down")
        await pilot.pause()
        assert editor.picker.highlighted_name() == first


@pytest.mark.asyncio
async def test_accepting_inserts_a_literal_at_path_token(workspace) -> None:
    """Exactly ``@<path>``, and NO trailing space.

    The space is withheld because a path segment may continue; adding it would
    terminate the token and close the list the next keystroke needs.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "read @READ")
        await pilot.press("tab")
        await pilot.pause()
        assert editor.text == "read @README.md"
        assert not editor.text.endswith(" ")


@pytest.mark.asyncio
async def test_accepting_a_path_with_spaces_emits_the_quoted_form(workspace) -> None:
    """``@"my file.txt"`` — the form ``at_token`` reads back.

    A bare space would end the token at the gap and leave the rest of the
    filename sitting in the draft as prose.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        # Driven through the app so the picker is really in FILE mode, but the
        # completion is called directly: the query stops at the space, so no row
        # the fuzzy matcher can reach from `@my fi` would exercise the quoting.
        await _draft(app, pilot, "@my fi")
        completed = completion_for(
            "@my",
            3,
            CompletionMode.FILE,
            "my file.txt",
            (),
        )
        assert completed is not None
        assert completed[0] == '@"my file.txt"'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("buffer", "row", "expected"),
    [
        ("@src/", "app.py", "@src/app.py"),
        ("@src/ap", "app.py", "@src/app.py"),
        ("@src/sub/", "deep.py", "@src/sub/deep.py"),
    ],
)
async def test_a_reference_accepted_in_a_subdirectory_resolves(
    workspace, buffer: str, row: str, expected: str
) -> None:
    """END TO END: the buffer the composer writes is one the resolver can read.

    Deliberately NOT two tests. The defect this pins passed every top-level
    picker assertion and every resolver assertion separately — ``scan_directory``
    correctly returns BARE names (``app.py``), the resolver correctly resolves
    real paths, and the composer correctly replaced the token span. The bug
    lived in the seam: replacing the whole span with a bare row name dropped the
    directory, so ``@src/`` + ``app.py`` produced ``@app.py``, which does not
    exist and reached submit as a "no such path" notice.

    So this asserts BOTH halves in one test — the text that lands in the buffer,
    and that ``expand_references`` on that exact text resolves. Depth 2 is
    covered because the general form of the bug is "every segment before the
    last is dropped", which a depth-1 test alone cannot see.
    """
    from local_operator.references import expand_references

    completed = completion_for(buffer, len(buffer), CompletionMode.FILE, row, ())
    assert completed is not None
    assert completed[0] == expected
    result = await expand_references(completed[0], str(workspace))
    assert result.expanded is True, f"{completed[0]!r} did not resolve: {result.notices}"
    assert result.notices == []


@pytest.mark.asyncio
async def test_accepting_a_directory_row_deepens_instead_of_terminating(workspace) -> None:
    """``@src/`` + ``sub/`` → ``@src/sub/``, and the list re-lists the deeper one.

    A directory row is the one completion that must NOT close the token: the
    user is still naming a path. The caret lands after the trailing slash and
    the re-arm (keyed on the directory part, not on a transition edge) asks the
    app for the deeper listing.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@src/")
        rows = _rows(editor)
        assert "sub/" in rows, f"fixture did not reach the directory row: {rows}"
        editor.picker.choose(rows.index("sub/"))
        await pilot.pause()
        await pilot.pause()
        assert editor.text == "@src/sub/"
        # The deeper directory is now the one being listed, which is the re-arm
        # working: a token-keyed latch would still be showing `src/`.
        assert "deep.py" in _rows(editor)


@pytest.mark.asyncio
async def test_a_quoted_path_in_a_subdirectory_round_trips(workspace) -> None:
    """``@"src/my file.txt"`` — quotes wrap the FULL path, not the bare name.

    The first place the two halves of this feature actually meet: the composer
    emits the quoted form and the resolver's parser reads it back. They were
    written against a frozen contract by different hands, so agreement is
    asserted rather than assumed.
    """
    from local_operator.references import at_token, expand_references

    (workspace / "src" / "my file.txt").write_text("spaces\n")
    completed = completion_for("@src/", 5, CompletionMode.FILE, "my file.txt", ())
    assert completed is not None
    assert completed[0] == '@"src/my file.txt"'
    token = at_token(completed[0], completed[1])
    assert token is not None
    assert token.query == "src/my file.txt", "the quoted form did not round-trip"
    result = await expand_references(completed[0], str(workspace))
    assert result.expanded is True, f"quoted path did not resolve: {result.notices}"


@pytest.mark.asyncio
async def test_neither_tab_nor_enter_submits_on_a_file_row(workspace) -> None:
    """A completed ``@src/`` is very often mid-path.

    Submitting on the one-match case would send a reference to a directory the
    user was still typing past, and a dispatched turn has no undo.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@READ")
        await pilot.press("enter")
        await pilot.pause()
        assert session.prompts == [], "Enter on a file row submitted the turn"
        assert "@README.md" in editor.text


@pytest.mark.asyncio
async def test_clicking_a_file_row_inserts_the_path(workspace) -> None:
    """The MOUSE path, which is completely separate from the key dispatch.

    ``choose()`` is what ``on_click`` reaches, so driving it here exercises the
    same callback a real click does.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "read @READ")
        assert editor.picker.mode is PickerMode.FILE
        editor.picker.choose(0)
        await pilot.pause()
        assert editor.text == "read @README.md"
        assert session.prompts == [], "a clicked file row submitted the turn"


@pytest.mark.asyncio
async def test_clicking_a_file_row_does_not_fall_through_to_the_command_vocabulary(
    workspace,
) -> None:
    """The regression that makes the click test above non-vacuous.

    Without the FILE arm in ``_apply_command`` a clicked row falls through to
    the COMMAND completion, which looks ``src/app.py`` up in the command
    vocabulary, gets ``None``, and returns at the ``completed is None`` guard —
    leaving the buffer UNTOUCHED with no exception and no message. This asserts
    the buffer changed, which is the only observable difference.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@src/ap")
        assert editor.picker.mode is PickerMode.FILE
        before = editor.text
        assert "app.py" in _rows(editor), "fixture did not reach the row under test"
        editor.picker.choose(_rows(editor).index("app.py"))
        await pilot.pause()
        assert editor.text != before, "the click silently did nothing"
        assert editor.text == "@src/app.py"


# --- The totality guard ------------------------------------------------------

#: One buffer+caret per mode that actually PUTS the picker in that mode. The
#: table is keyed by mode so the assertion below can compare its keys against
#: the enum; a list of cases would not have that property.
CASES: dict[PickerMode, tuple[str, int]] = {
    PickerMode.COMMAND: ("/te", 3),
    PickerMode.ARGUMENT: ("/theme ro", 9),
    PickerMode.SKILL: ("$res", 4),
    PickerMode.FILE: ("@src", 4),
}


@pytest.mark.asyncio
async def test_every_picker_mode_has_a_live_phase(workspace) -> None:
    """Every mode the enum declares reaches a non-``None`` picker phase.

    The highest-value test in this change, and assertion 1 is why. ``_picker_phase``
    derives its answer from the BUFFER, never from a mode, and
    ``_latched_picker_at_caret`` asks ``is not None`` rather than listing phases —
    so a fifth mode added without wiring the phase machinery would fall silently
    out of every latch and lose its Esc. Comparing the table's KEYS against
    ``set(PickerMode)`` refuses to let a mode exist untested; it needs to know
    nothing about what the new mode does.

    Assertion 2 keeps the table honest: a fixture that does not actually reach
    its mode would make this decorative.
    """
    assert set(CASES) == set(PickerMode), (
        "every PickerMode needs a fixture here — a new mode must not be able to "
        "join the enum without someone deciding what buffer puts the picker in it"
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        for mode, (text, caret) in CASES.items():
            editor = await _draft(app, pilot, text, caret)
            assert editor.picker.mode is mode, f"{text!r} did not reach {mode}"
            assert editor._picker_phase() is not None, f"{mode} has no live phase"


# --- Pure units --------------------------------------------------------------


def test_file_suggestions_rank_without_a_skill_style_gate() -> None:
    """An uppercase name and a bare query both survive.

    ``skill_suggestions``' gate would reject both — it demands a lowercase
    letter as evidence and case-sensitive prefix matching — and a path list
    needs neither. This pins that ``file_suggestions`` did not inherit it.
    """
    choices = [ArgumentChoice("README.md"), ArgumentChoice("src/")]
    assert [name for name, _ in file_suggestions("", choices)] == ["README.md", "src/"]
    assert [name for name, _ in file_suggestions("read", choices)] == ["README.md"]


def test_the_ghost_is_an_append_for_a_file_row() -> None:
    """``ghost_for``'s ``startswith`` rule gives FILE a ghost for free.

    A FILE completion is a pure span replacement with nothing moved to the
    front of the buffer, so it IS an append whenever the row extends what was
    typed — which is what makes a ghost branch unnecessary rather than missing.
    """
    completed = completion_for("@READ", 5, CompletionMode.FILE, "README.md", ())
    assert completed is not None
    assert completed[0].startswith("@READ")
    assert ghost_for(completed, "@READ") == "ME.md"

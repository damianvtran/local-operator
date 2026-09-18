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
async def test_at_me_never_opens_the_picker_at_all(workspace) -> None:
    """``@me`` in a POPULATED directory: no list, no rewrite, prose intact.

    This used to assert only the empty-directory case, on the reasoning that
    whether the list closes is a question about the fuzzy scorer, not about the
    ``@`` grammar — ``argument_suggestions``' ``prefixed or matches`` kept a
    SUBSEQUENCE hit open (m…e against ``README.md``) and that was accepted as
    "the ranking behaviour every other list has".

    It is not: the resolver's governing rule calls that token prose
    (``@me — no such path; sent as written``), and an open FILE list OWNS Enter,
    so the accepted behaviour was a silent rewrite of prose into a filename the
    operator never typed, with the message never sent (QA round 1, Q-2,
    measured against the real composer). ``file_suggestions`` now requires
    prefix evidence, which is what makes the populated case behave like the
    empty one. Both are asserted here because the harm was always in the
    populated one.

    The negative is the point of this file: an ``@`` is an ordinary character
    in ``--assignee @me``, so this is the assertion that keeps prose prose.
    """
    empty = workspace / "empty"
    empty.mkdir()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        populated = await _draft(app, pilot, "glab mr create --assignee @me")
        assert (
            not populated.picker.is_open()
        ), "a subsequence row opened for a token the resolver calls prose"
        assert populated.text == "glab mr create --assignee @me"

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
async def test_enter_completes_an_unfinished_row_and_a_second_enter_sends(workspace) -> None:
    """A completed ``@src/`` is very often mid-path — so the FIRST Enter completes.

    Submitting on the one-match case would send a reference to a directory the
    user was still typing past, and a dispatched turn has no undo. That is the
    rule this test has always pinned, and it still holds: ``@READ`` is not what
    the row says, so Enter inserts the row and nothing is submitted.

    What the rule needed was a TERMINATION CONDITION, because the FILE list
    inserts no trailing space and therefore re-opens on the very name it just
    completed. Without one, Enter completed the same row forever: a draft
    ending in a reference could never be sent at all (QA round 1, Q-1). So the
    SECOND Enter — by which point the row IS what the buffer holds and there is
    nothing left to accept — submits, which is the assertion the old version of
    this test could not make and the reason its name claimed "neither key ever
    submits".
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@READ")
        assert editor.picker.is_open(), "fixture never reached an open file list"
        await pilot.press("enter")
        await pilot.pause()
        assert session.prompts == [], "Enter on an unfinished file row submitted the turn"
        assert editor.text == "@README.md"
        await pilot.press("enter")
        await pilot.pause()
        assert len(session.prompts) == 1, (
            "the token never terminated: Enter on a row the buffer already holds "
            "must send, or the draft is unsendable"
        )


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


def test_file_suggestions_offer_a_path_only_on_prefix_evidence() -> None:
    """The one rule a path list has of its own: a typed segment must PREFIX it.

    Everything the catalogue behaviour was argued for survives — ``@ap`` still
    reaches ``app.py``, and the comparison is case-INSENSITIVE so ``@read``
    still reaches ``README.md`` (a letter-for-letter rule would drop every
    uppercase name). What does not survive is a SUBSEQUENCE match, which is what
    ``argument_suggestions`` would have returned: ``@me`` in a directory holding
    ``README.md`` reached a row, the resolver calls that token prose, and an
    open FILE list owns Enter — so the row was an offer to rewrite prose
    silently (QA round 1, Q-2).

    The empty query is still a real answer here: a bare ``@`` is an explicit
    "what is here", so the whole directory lists.
    """
    choices = [ArgumentChoice("README.md"), ArgumentChoice("src/"), ArgumentChoice("app.py")]
    assert [name for name, _ in file_suggestions("", choices)] == [
        "README.md",
        "src/",
        "app.py",
    ]
    assert [name for name, _ in file_suggestions("read", choices)] == ["README.md"]
    assert [name for name, _ in file_suggestions("READ", choices)] == [
        "README.md"
    ], "a case-sensitive prefix rule would reject every uppercase entry name"
    assert [name for name, _ in file_suggestions("ap", choices)] == ["app.py"]
    assert [
        name for name, _ in file_suggestions("me", choices)
    ] == [], "a subsequence hit reached a row for a token the resolver leaves as prose"
    # The cost of the gate, pinned rather than discovered later: a typo no
    # longer reaches its near-miss row. Fail-closed is the deliberate direction
    # — a shut list leaves the draft as typed, an open one can silently change
    # it into a file the model will then read.
    assert [name for name, _ in file_suggestions("appp.py", choices)] == []


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


@pytest.mark.asyncio
async def test_the_pending_window_covers_FILE_not_just_ARGUMENT(workspace) -> None:
    """``is_pending`` must answer True for a FILE list still being filled.

    THE ARM HAD NO TEST THAT COULD GO RED. Replacing ``is_pending``'s body with
    the base ARGUMENT-only rule left all three composer test files green, so
    nothing held the FILE branch in place: a later reader had no way to learn it
    was load-bearing, and deleting it would have looked free.

    What it buys is the Esc in the one-tick fill window. The editor posts
    ``FileQueryOpened`` and the app answers with ``set_choices`` a tick later
    (``app.on_file_query_opened``); in between, the picker is in FILE mode
    holding nothing — closed by ``is_open()``, but a list the user has just
    opened as far as they are concerned. ``Editor._on_key`` routes Esc on
    ``is_pending()`` alone, so without the FILE arm the key is dropped and the
    user watches the list they dismissed appear anyway.

    Reached WITHOUT pumping the message loop, which is the whole trick: the
    state under test is the tick BEFORE the app answers, so a test that paused
    would be testing the filled list instead. ``_sync_picker`` is called
    directly to derive the mode from the buffer, exactly as the keystroke path
    does, and the app's answer is never awaited.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.load_text("@")
        editor.move_cursor(editor._end_of_buffer())
        editor._sync_picker()

        picker = editor.picker
        assert picker.mode is PickerMode.FILE, "the buffer did not put the picker in FILE mode"
        assert not picker.suggestions(), "the app already answered; this is not the fill window"
        assert not picker.is_open(), "an empty list must still read as closed"
        # The assertion the ARGUMENT-only rule fails. It is the only one here
        # that distinguishes the two implementations, so it is the one that
        # proves the arm exists.
        assert picker.is_pending(), "a FILE list being filled must read as pending"


@pytest.mark.asyncio
async def test_an_empty_directory_notice_SURVIVES_the_next_keystroke(workspace) -> None:
    """The notice has to outlive the frame it was painted on to be readable.

    `_apply`'s notice-hold branch enumerated ARGUMENT only, so a FILE notice was
    dropped by the very next re-derivation: "nothing to reference in empty/"
    painted for one frame and vanished as the user kept typing. A notice nobody
    can read is not an answer.

    It is still true while they type, which is the point — the editor re-posts
    `FileQueryOpened` only when the DIRECTORY changes, so every keystroke inside
    an empty directory re-derives against the same empty listing.
    """
    (workspace / "empty").mkdir()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@empty/")
        picker = editor.picker
        assert picker._notice, "the empty directory produced no notice at all"
        held = picker._notice
        assert picker.display, "the list closed instead of holding the notice"

        # One more character INSIDE the same directory: a re-derivation, not a
        # new query, which is exactly the window the notice used to die in.
        editor.insert("x")
        for _ in range(6):
            await pilot.pause()

        assert picker._notice == held, "the FILE notice was dropped on the next keystroke"
        assert picker.display, "the list closed and took the notice with it"


# ---------------------------------------------------------------------------
# The design round's fix-forward (design review round 1 on #1220, D1–D5).
# Every test below FAILS against 59c435739 and is written from that round's own
# reproductions, because the four defects it found are all invisible to a test
# that only asks whether the feature works: the shape that shipped worked, and
# offered the wrong directory while it did.
# ---------------------------------------------------------------------------


def _rendered(editor: Editor, width: int = 80) -> list[str]:
    """The picker's painted rows + notice, as the user reads them."""
    return editor.picker.render_text(width).plain.split("\n")


def _gutter(row: str) -> str:
    """The 3-cell selection gutter of a rendered row, stripped."""
    return row[:3].strip()


async def _click_at(editor: Editor, pilot, offset: int) -> None:
    """One real mouse click on whole-buffer ``offset`` in the composer.

    The gesture, not the handler: D1's second reproduction is a plain click, and
    a test that called ``_sync_picker_if_phase_changed`` directly would prove the
    comparison is right while proving nothing about whether anything CALLS it.
    """
    row, column = editor._location_at_offset(offset)
    x, y = editor.wrapped_document.location_to_offset((row, column))
    await pilot.click(Editor, offset=(editor.gutter_width + x, y))
    for _ in range(3):
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_paste_into_the_second_reference_lists_THAT_tokens_directory(workspace) -> None:
    """D1, the paste-shaped arrival: the rows must describe the CARET's token.

    `load_text` then a caret move to the end is one plain paste, and before the
    fix it left the caret in `@src/` while the rows were the FIRST token's
    (`README.md`) — the phase went `file -> file`, so the phase-only gate never
    re-derived. ``dir_latch`` is the witness: it still named the old directory.

    The consequence asserted here is the harmful one rather than the cosmetic
    one. Enter on a stale row cannot be accepted at this caret
    (`_file_row_is_already_in_the_buffer` is False), so it fell through to the
    ordinary submit — and with the root's row highlighted, ``_complete_file``
    wrote **@src/README.md**, a path that does not exist, into the draft,
    silently, with nothing sent and no notice.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "@README.md and also @src/")
        assert editor.picker.mode is PickerMode.FILE, "premise: the list is live"
        assert editor._file_choices_requested == "src/", "premise: the caret left `src/`"
        assert "app.py" in _rows(editor), "the rows are the FIRST token's directory"
        assert "README.md" not in _rows(editor), "a row from the other token's directory"

        await pilot.press("enter")
        for _ in range(6):
            await pilot.pause()

        assert editor.text == "@README.md and also @src/app.py", "the row was not accepted"
        assert editor.text.count("@") == 2, "a reference was duplicated"
        assert session.prompts == [], "the draft went out instead of completing"


@pytest.mark.asyncio
async def test_a_click_into_an_earlier_reference_relists_its_directory(workspace) -> None:
    """D1, the mouse shape: the list follows the caret, not the last directory.

    Driven with real keys and one real click, because that is the reproduction:
    typing ends in `@src/`, and clicking back into the earlier `@README.md` used
    to leave `src/`'s four rows on screen under a ROOT-directory token, with a
    highlighted row that could not be accepted at this caret.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "explain @README.md and also @src/")
        assert "app.py" in _rows(editor), "premise: the list is on `src/`"

        await _click_at(editor, pilot, 15)  # inside the earlier `@README.md`

        assert 8 < editor._caret_offset() <= 18, "premise: the click landed in the token"
        assert editor._file_choices_requested == "", "the list still latches the other token"
        assert "README.md" in _rows(editor), "the root's own row is missing"
        assert "app.py" not in _rows(editor), "a row from the token the caret LEFT"


@pytest.mark.asyncio
async def test_a_reference_is_painted_with_the_reference_ink(workspace) -> None:
    """D2: the `@path` token must not be indistinguishable from the prose.

    Measured before the fix: `/help` painted `#6ea8d8` and
    ` explain @README.md to me` came back as ONE run of `#e9e5db`, so the token
    against the words beside it was **1.00:1** — not faint, identical. The rule
    the sheet already stated for an object of exactly this kind (the attachment
    chip: "the ramp's file/reference hue") had simply never been written for it.

    Read off the FINISHED strip `render_line` produces, which is what the
    terminal is sent, and both halves of the ruling are asserted: a token that
    RESOLVES gets the reference ink and a token the resolver would call prose
    (`@me`) does not. The second assertion is the one that keeps the ink from
    claiming an expansion that will not happen.
    """
    from local_operator.tui import theme as theme_mod

    signal = theme_mod.semantic_color("signal").lower()
    prose = theme_mod.semantic_color("fg").lower()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "/help explain @README.md to me")
        cells = {
            segment.text: (
                segment.style.color.get_truecolor().hex.lower()
                if segment.style and segment.style.color
                else None
            )
            for segment in editor.render_line(0)._segments
        }
        assert cells.get("/help") == signal, "premise: the command word's own ink"
        assert cells.get("@README.md") == signal, "the reference token has no ink of its own"
        assert cells.get(" explain ") == prose, "the prose either side changed instead"

        editor = await _draft(app, pilot, "glab mr create --assignee @me")
        cells = [
            (
                segment.text,
                (
                    segment.style.color.get_truecolor().hex.lower()
                    if segment.style and segment.style.color
                    else None
                ),
            )
            for segment in editor.render_line(0)._segments
        ]
        assert "@me" in editor.text, "premise: the token is on this row"
        assert signal not in [ink for _text, ink in cells], "prose was painted as a reference"


@pytest.mark.asyncio
async def test_a_no_match_query_holds_a_notice_and_stays_closed(workspace) -> None:
    """D3: `@me` answered with silence, one keystroke from `@srx/` explaining itself.

    A notice is not a match, which is the property Q-2 rests on: the fuzzy
    near-miss stays unoffered, ``is_open()`` stays False and every key still
    reaches the buffer. What changes is that the surface now says why.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "glab mr create --assignee @me")
        picker = editor.picker
        assert not picker.is_open(), "a no-match query must not own Enter"
        assert not picker.suggestions(), "a notice is not a row"
        assert picker._notice == "nothing here matches `me`", "the no-match state is silent"
        assert picker.display, "the notice was not held on screen"
        assert "nothing here matches `me`" in chr(10).join(_rendered(editor)), "not painted"

        # The copy quotes the query, so it has to move with it: holding the
        # first one left `@zz` reading "nothing here matches `z`".
        editor.insert("z")
        for _ in range(3):
            await pilot.pause()
        assert picker._notice == "nothing here matches `mez`", "a stale copy of the query"

        # And the buffer still owns the keys.
        before = editor.text
        editor.insert("!")
        for _ in range(3):
            await pilot.pause()
        assert editor.text == before + "!", "the notice swallowed a keystroke"


@pytest.mark.asyncio
async def test_the_row_enter_would_send_carries_the_send_mark(workspace) -> None:
    """D5: nothing before the press said whether Enter would complete or SEND.

    The rows were pixel-identical in the two states — same gutter glyph, same
    ink, same ground, same detail cell — and only
    ``_file_row_is_already_in_the_buffer`` differed, invisibly. The mark is asked
    of a predicate at PAINT time rather than read from a flag, which is why the
    two rows below can be rendered straight off the widget with no app at all.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        # COMPLETES: `@README` is a partial token, so the accept key appends.
        completing = await _draft(app, pilot, "explain @README")
        assert completing.picker.highlighted_name() == "README.md"
        assert not completing._file_row_is_already_in_the_buffer("README.md")
        assert _gutter(_rendered(completing)[0]) == "❯", "an accept key that completes"

        # SENDS: the buffer already holds the row, so there is nothing to accept.
        sending = await _draft(app, pilot, "explain @README.md")
        assert sending.picker.highlighted_name() == "README.md"
        assert sending._file_row_is_already_in_the_buffer("README.md")
        assert _gutter(_rendered(sending)[0]) == "↵", "an accept key that SENDS"

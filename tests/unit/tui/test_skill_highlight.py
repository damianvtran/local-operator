"""The ``$name`` composer ink: which tokens are painted, and with which claim.

The ``$`` token is the last composer sigil without an ink of its own, and the
states it has to keep apart are the whole of the work: a token that WILL fire,
a prefix in progress under an open list, a typo that will be sent as prose, and
the money/shell shapes that must stay prose because they always were. The
negative cases carry as much weight as the positive ones — ``$5`` and ``$PATH``
are ordinary characters in a sentence, and ``costs$5`` is money, not a skill.

Two layers are asserted separately because they regress independently:

- the GRAMMAR (:func:`skill_token_spans`): every boundary ``$name`` on a line,
  with the RESOLVER's extent — pinned against ``skills.invoke._INVOCATION_RE``
  itself, so the ink cannot claim a token the submit-side parser would read
  differently;
- the RENDER PATH (``Editor._skill_runs`` and the strip ``render_line``
  produces): the state gates on the real ``OperatorApp``, read off the finished
  strip the terminal is sent.
"""

from __future__ import annotations

import pytest

from local_operator.skills.invoke import _INVOCATION_RE
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.command_picker import PickerMode, skill_token_spans
from local_operator.tui.widgets.editor import Editor

from .test_app_pilot import FakeSession, _factory
from .test_skill_invocation import skill_root  # noqa: F401  (pytest fixture)

SKILL = "text-area--skill"
SKILL_UNKNOWN = "text-area--skill-unknown"


# --- the grammar -------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        # The resolver's extent: `$` then [A-Za-z0-9][A-Za-z0-9._-]*.
        ("$research", [(0, 9, "research")]),
        ("  $research", [(2, 11, "research")]),
        # A comma is OUTSIDE the class and stays request text...
        ("$research, fix it", [(0, 9, "research")]),
        # ...a dot is INSIDE it, so it joins the name (and will not resolve).
        ("$research. next", [(0, 10, "research.")]),
        ("$a.b_c-d", [(0, 8, "a.b_c-d")]),
        ("$5", [(0, 2, "5")]),
        ("echo $PATH", [(5, 10, "PATH")]),
        # Several tokens on one line, each with its own extent.
        ("$a $b", [(0, 2, "a"), (3, 5, "b")]),
        ("x $a $b", [(2, 4, "a"), (5, 7, "b")]),
        # The boundary rule: glued `$` is punctuation inside a word.
        ("costs$5 for it", []),
        ("a$b", []),
        ("$$research", []),
        # A `$` with nothing usable after it opens no span, like the resolver.
        ("$", []),
        ("$-x", []),
        ("$_private", []),
        # Non-ASCII heads are outside the ASCII class the parser reads.
        ("$ßeta", []),
        ("$研究", []),
        ("($research)", []),
    ],
)
def test_the_scan_reads_every_boundary_token(
    line: str, expected: list[tuple[int, int, str]]
) -> None:
    """The scan is pure grammar: it decides spans, never worth painting."""
    assert skill_token_spans(line) == expected


@pytest.mark.parametrize(
    "name",
    [
        "research",
        "code-review",
        "a.b_c-d",
        "research.",
        "research,",
        "5",
        "PATH",
        "Research",
        "a b",
        "a$b",
        "ßeta",
        "_private",
        "-x",
        "研究",
    ],
)
def test_the_scan_agrees_with_the_submit_resolver(name: str) -> None:
    """The extent is the RESOLVER's, pinned against the resolver itself.

    ``command_picker`` does not import ``skills.invoke`` — that module pulls
    the discovery tree in, and the widget tree is on the app's boot path — so
    the name class is reproduced there. This is the pin that keeps the two
    from drifting: whatever ``_INVOCATION_RE`` will read on submit, the scan
    spans the same characters.
    """
    line = "$" + name
    match = _INVOCATION_RE.match(line)
    spans = skill_token_spans(line)
    if match is None:
        assert spans == [], "the scan claimed a token the resolver rejects"
        return
    assert spans, "the resolver reads a token the scan missed"
    start, end, read = spans[0]
    assert (start, end, read) == (match.start(), match.end(), match.group(1))


# --- the state matrix (no I/O: the gate reads only its own inputs) -----------


def _editor(text: str, names: frozenset[str] | set[str] | None) -> Editor:
    editor = Editor()
    editor.text = text
    if names is not None:
        editor.set_skill_names(frozenset(names))
    return editor


@pytest.mark.parametrize(
    ("text", "names", "expected"),
    [
        # RESOLVED: exact case-insensitive membership, whole buffer scannable.
        ("$research fix this", {"research"}, {0: [(0, 9, SKILL)]}),
        ("$Research", {"research"}, {0: [(0, 9, SKILL)]}),
        ("$research,", {"research"}, {0: [(0, 9, SKILL)]}),
        # ATTEMPTED: leading, whole trimmed draft, lowercase, unresolved.
        ("$zzz", {"research"}, {0: [(0, 4, SKILL_UNKNOWN)]}),
        ("$research.", {"research"}, {0: [(0, 10, SKILL_UNKNOWN)]}),
        # An EMPTY-but-settled vocabulary resolves nothing and still dims.
        ("$zzz", set(), {0: [(0, 4, SKILL_UNKNOWN)]}),
        # ...but a vocabulary that was never pushed claims NOTHING at all.
        ("$research", None, {}),
        ("$zzz", None, {}),
        # No lowercase evidence: money and shell stay prose.
        ("$5", {"research"}, {}),
        ("$PATH", {"research"}, {}),
        # Not the whole draft: the muted sentence would be about the wrong thing.
        ("$zzz fix this", {"research"}, {}),
        ("$research $other", {"research"}, {0: [(0, 9, SKILL)]}),
        # INLINE: only the buffer's first token can be leading.
        ("fix this $research", {"research"}, {}),
        ("\n$research", {"research"}, {1: [(0, 9, SKILL)]}),
        # Glued and headless forms open no span at all.
        ("costs$5 for it", {"research"}, {}),
        ("$_private", {"research"}, {}),
    ],
)
def test_the_state_matrix(
    text: str,
    names: frozenset[str] | set[str] | None,
    expected: dict[int, list[tuple[int, int, str]]],
) -> None:
    """One table over the four states: resolved, inert, neutral, unknown.

    Host-free and I/O-free on purpose — the classification reads the text, the
    snapshot and the picker's own flags, so the matrix can pin every corner
    without booting an app (the pilot tests below cover the live states the
    flags produce). The expected value is the RUN MAP: which document line
    carries which spans, which is the shape the paint consumes.
    """
    assert _editor(text, names)._compute_skill_runs() == expected


def test_a_resolved_token_on_a_later_line_still_resolves() -> None:
    """No first-content-line restriction, unlike the slash pass.

    The `$` token may open a LATER line; leading is a whole-BUFFER question, so
    blank lines above it do not disqualify it — where any non-whitespace before
    it would (the matrix's inline case).
    """
    runs = _editor("\n\n$research", {"research"})._compute_skill_runs()
    assert runs == {2: [(0, 9, SKILL)]}


# --- the render path (real OperatorApp) --------------------------------------


async def _draft(app: OperatorApp, pilot, text: str, caret: int | None = None) -> Editor:
    """Put ``text`` in the composer with the caret at ``caret`` (default: end).

    Two pauses, not one: the app answers ``SkillQueryOpened`` one message-loop
    tick behind the keystroke, so a single pause observes a picker mid-fill —
    the same reason ``test_at_picker._draft`` double-pauses.
    """
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    editor.load_text(text)
    editor.move_cursor(
        editor._end_of_buffer() if caret is None else editor._location_at_offset(caret)
    )
    for _ in range(6):
        await pilot.pause()
    return editor


def _segment_inks(editor: Editor, y: int = 0) -> list[tuple[str, str | None]]:
    """``(text, ink)`` for every segment of the finished strip on row ``y``."""
    return [
        (
            segment.text,
            (
                segment.style.color.get_truecolor().hex.lower()
                if segment.style and segment.style.color
                else None
            ),
        )
        for segment in editor.render_line(y)._segments
    ]


def _ink_on(segments: list[tuple[str, str | None]], needle: str) -> str | None:
    for text, ink in segments:
        if needle in text:
            return ink
    return None


@pytest.mark.usefixtures("skill_root")
@pytest.mark.asyncio
async def test_a_resolved_token_takes_the_signal_ink() -> None:
    """The must-have: `$research` is not indistinguishable from the prose.

    Read off the FINISHED strip, and both halves asserted: the token carries
    the structured-token ink and the words beside it are untouched.
    """
    signal = theme_mod.semantic_color("signal").lower()
    prose = theme_mod.semantic_color("fg").lower()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "$research fix this")
        segments = _segment_inks(editor)
    assert _ink_on(segments, "$research") == signal, segments
    assert _ink_on(segments, " fix this") == prose, segments


@pytest.mark.usefixtures("skill_root")
@pytest.mark.asyncio
async def test_typing_the_full_name_lights_it_while_the_list_is_open() -> None:
    """A full name is an ANSWER, not a prefix: it paints even while open.

    The two frames a user passes through on the way to `$research`: `$researc`
    is in progress (its row is offered, so no ink), and one keystroke later the
    token names a discovered skill — from there the ink holds regardless of the
    list, which is the difference between "in progress" and "resolved".
    """
    signal = theme_mod.semantic_color("signal").lower()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "$researc")
        assert editor.picker.mode is PickerMode.SKILL, "premise: the skill list is live"
        assert editor.picker.is_open(), "premise: a row is offered for the prefix"
        assert _ink_on(_segment_inks(editor), "$researc") != signal, "a prefix in progress"

        await pilot.press("h")
        for _ in range(4):
            await pilot.pause()
        assert editor.text == "$research"
        assert editor.picker.is_open(), "premise: the row for the full name is still up"
        assert _ink_on(_segment_inks(editor), "$research") == signal, "a resolved name"


@pytest.mark.usefixtures("skill_root")
@pytest.mark.asyncio
async def test_a_prefix_under_the_open_list_is_not_painted_then_dims_on_escape() -> None:
    """The suppression, and the close that ends it.

    `$res` offers the `research` row, so the token is "in progress, not wrong"
    and paints nothing — mirroring `text-area--slash-unknown`. Esc closes the
    list on the same buffer, and THEN the muted ink is the true sentence: this
    word, alone in the draft, will be sent as prose.
    """
    dim = theme_mod.semantic_color("dim").lower()
    prose = theme_mod.semantic_color("fg").lower()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "$res")
        assert editor.picker.is_open(), "premise: a row is offered for the prefix"
        # The TOKEN's own ink, not a blanket scan of the row: the strip also
        # carries the completion ghost (`earch `), which is legitimately dim.
        assert _ink_on(_segment_inks(editor), "$res") == prose, "the open list owns the token"

        await pilot.press("escape")
        for _ in range(8):
            await pilot.pause()
        assert not editor.picker.is_open(), "premise: Esc closed the list"
        assert _ink_on(_segment_inks(editor), "$res") == dim, "the inert ink after close"


@pytest.mark.usefixtures("skill_root")
@pytest.mark.asyncio
async def test_the_closed_list_ink_reaches_the_composed_screen() -> None:
    """The flip must be ON SCREEN, not only in the widget's own strip.

    The suppression lifts on an Esc that changes no text, so the ink moves on a
    buffer that is otherwise identical — exactly the shape where a cached strip
    could stay stale on screen until the next keystroke. Read off the
    COMPOSITOR's own style for the token's cells (what the terminal shows),
    with no forced refresh in between.
    """
    dim = theme_mod.semantic_color("dim").lower()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "$res")
        await pilot.press("escape")
        for _ in range(8):
            await pilot.pause()

        compositor = app.screen._compositor
        x, y = editor.region.x, editor.region.y
        cell_inks: set[str] = set()
        for column in range(x, x + 8):
            style = compositor.get_style_at(column, y)
            if style is not None and style.color is not None:
                cell_inks.add(style.color.get_truecolor().hex.lower())
    assert dim in cell_inks, f"the inert ink never reached the screen: {cell_inks}"


@pytest.mark.usefixtures("skill_root")
@pytest.mark.asyncio
async def test_an_unresolved_word_dims_once_the_word_stops_matching() -> None:
    """`$zzz` before any Escape: the TUI closes the list itself.

    A query miss closes the skill list silently (long-standing, and explicitly
    out of this lane), so there is no "miss shown" state for the ink to wait
    on — unlike the desktop the dim lands without a dismissal. The premise
    asserts the silent close, so this test fails loudly if that changes.
    """
    dim = theme_mod.semantic_color("dim").lower()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "$zzz")
        assert not editor.picker.is_open(), "premise: a skill miss closes the list"
        assert _ink_on(_segment_inks(editor), "$zzz") == dim, "the inert ink"


@pytest.mark.usefixtures("skill_root")
@pytest.mark.asyncio
async def test_money_and_shell_stay_prose_in_the_real_composer() -> None:
    """The guards, swept on the live render path: no ink of either kind."""
    dim = theme_mod.semantic_color("dim").lower()
    signal = theme_mod.semantic_color("signal").lower()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        for text in ("$5 for the redesign", "echo $PATH", "costs$5 for it", "a$b"):
            editor = await _draft(app, pilot, text)
            inks = [ink for _text, ink in _segment_inks(editor)]
            assert signal not in inks and dim not in inks, f"{text!r} was inked: {inks}"


@pytest.mark.usefixtures("skill_root")
@pytest.mark.asyncio
async def test_an_inline_token_gets_nothing_until_acceptance() -> None:
    """`fix this $res` is not the anchored shape yet — acceptance makes it one.

    Enter on the offered row reassembles the token to the FRONT (the existing
    picker contract: the submit parser is anchored), and that is the moment the
    resolved ink becomes true.
    """
    signal = theme_mod.semantic_color("signal").lower()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "fix this $res")
        assert editor.picker.is_open(), "premise: the inline evidence gate opened the list"
        assert signal not in [ink for _text, ink in _segment_inks(editor)], "not yet"

        await pilot.press("enter")
        for _ in range(6):
            await pilot.pause()
        assert editor.text == "$research fix this ", editor.text
        assert _ink_on(_segment_inks(editor), "$research") == signal, "after acceptance"


@pytest.mark.usefixtures("skill_root")
@pytest.mark.asyncio
async def test_a_slash_claimed_token_stays_prose() -> None:
    """`/model $5`: the command owns its line, and the token is not leading.

    Two independent reasons the `$` gets no ink here, and the assertion is on
    the OUTCOME both of them exist for: a command's argument never wears the
    skill inks.
    """
    prose = theme_mod.semantic_color("fg").lower()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "/model $5")
        assert editor.picker.mode is not PickerMode.SKILL, "premise: no skill list"
        # Scoped to the `$` token: `/model` itself takes the slash-command ink
        # (that is its OWN pass), so a blanket scan would flag the command word.
        assert _ink_on(_segment_inks(editor), "$5") == prose, _segment_inks(editor)

        # The TUI's RELAXED claim (a prompt command past its name slot) makes
        # `$research` a live skill token — and the ink still stays off it,
        # because a leading token this is not.
        editor = await _draft(app, pilot, "/team delivery $research")
        assert editor.picker.mode is PickerMode.SKILL, "premise: the relaxed claim keeps it one"
        assert _ink_on(_segment_inks(editor), "$research") == prose, _segment_inks(editor)


@pytest.mark.asyncio
async def test_an_empty_vocabulary_resolves_nothing_but_still_dims(tmp_path, monkeypatch) -> None:
    """Settled-empty is an ANSWER: no skill can fire, and `$zzz` says so.

    The fixture wires in an EMPTY skills root as the only root, so discovery
    really answers "none" — the state a fresh machine boots in (where the bare
    `$` arm shows its own notice; that surface predates this ink and is out of
    scope here).
    """
    dim = theme_mod.semantic_color("dim").lower()
    root = tmp_path / "skills"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_SKILL_EXTRA_ROOTS", str(root))
    monkeypatch.chdir(tmp_path)

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "$zzz")
        assert editor._skill_names == frozenset(), "premise: the empty answer landed"
        assert _ink_on(_segment_inks(editor), "$zzz") == dim, "the inert ink"


@pytest.mark.usefixtures("skill_root")
@pytest.mark.asyncio
async def test_the_ink_follows_the_snapshot_the_app_pushes() -> None:
    """The editor's vocabulary IS the pushed snapshot — one push moves the ink.

    `set_skill_names` is the app's whole side of the contract, so a later push
    (a rescan after a skill was added or removed) has to move the ink without
    any keystroke: the refresh-on-change the setter documents, read at strip
    level. `$research` alone is the demo shape: a member paints resolved, and
    once the same word is no longer a member it falls to the inert ink.
    """
    signal = theme_mod.semantic_color("signal").lower()
    dim = theme_mod.semantic_color("dim").lower()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _draft(app, pilot, "$research")
        assert _ink_on(_segment_inks(editor), "$research") == signal, "the app's snapshot"
        assert editor._skill_names is not None

        # The push moves the token OFF resolved at once. It is unresolved now,
        # but its list is still open (the rows come from the picker's choices,
        # not from the snapshot), so it paints nothing; the muted ink then
        # needs the list closed, which Esc does on the same buffer.
        editor.set_skill_names(frozenset({"code-review"}))
        for _ in range(4):
            await pilot.pause()
        assert _ink_on(_segment_inks(editor), "$research") != signal, "a changed snapshot"

        await pilot.press("escape")
        for _ in range(8):
            await pilot.pause()
        assert _ink_on(_segment_inks(editor), "$research") == dim, "the fall to inert"


def test_the_skill_ink_is_exactly_the_two_agreed_classes() -> None:
    """A contract guard: two treatments, and nothing else.

    The tcss rules are written for exactly a resolved ink and an inert one; a
    third class arriving without its own copy and contrast round is the silent
    scope creep this pins out.
    """
    classes = {name for name in Editor.COMPONENT_CLASSES if name.startswith("text-area--skill")}
    assert classes == {SKILL, SKILL_UNKNOWN}

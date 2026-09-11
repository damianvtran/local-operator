"""`$skill` inside a PROMPT-carrying slash command's argument.

The composer half of the contract. A command whose argument is free text bound
for the model (``consumes_prompt``) gives up its claim on that argument PAST the
name slot, so ``/team delivery $cte-explainer explain the MR`` opens the picker
and accepting a row rewrites the ARGUMENT rather than the buffer.

Three properties carry it and regress independently, so they are asserted apart:
WHICH carets open a token (the floor), WHAT the picker offers there (the inline
lowercase gate, which applies unchanged in this position), and WHERE an accepted
row lands (the argument's front, not the buffer's — the submit parser is anchored
at offset 0 of the argument string).

The negative half — where the claim still holds totally — stays in
`test_skill_invocation.py::test_a_sigil_stays_plain_text_where_the_claim_holds`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.slash_commands import SLASH_COMMANDS
from local_operator.tui.app import OperatorApp
from local_operator.tui.autocomplete import ArgumentChoice, ArgumentMode
from local_operator.tui.widgets.command_picker import (
    CompletionMode,
    PickerMode,
    _skill_argument_floor,
    _skill_argument_region,
    completion_for,
    ghost_for,
    skill_suggestions,
    skill_token,
)
from local_operator.tui.widgets.editor import Editor

from .test_app_pilot import FakeSession, _factory
from .test_skill_invocation import skill_root  # noqa: F401  (pytest fixture)

# The registry-derived vocabularies the composer passes. Built here exactly as
# `Editor.set_commands` builds them, so a drift between the two shows up as a
# failure rather than as a test that quietly stops describing the product.
COMMAND_NAMES = frozenset(name.lower() for c in SLASH_COMMANDS for name in c.names)
PROMPT_COMMANDS = frozenset(
    name.lower() for c in SLASH_COMMANDS if c.consumes_prompt for name in c.names
)
NAME_COMMANDS = frozenset(
    name.lower()
    for c in SLASH_COMMANDS
    if c.consumes_prompt and c.arguments is not ArgumentMode.NONE
    for name in c.names
)
ARGUMENT_COMMANDS = tuple(
    name for c in SLASH_COMMANDS if c.arguments is not ArgumentMode.NONE for name in c.names
)


def _token(text: str, caret: int | None = None):
    """``skill_token`` as the COMPOSER asks it, with both vocabularies."""
    return skill_token(
        text,
        len(text) if caret is None else caret,
        COMMAND_NAMES,
        PROMPT_COMMANDS,
        NAME_COMMANDS,
    )


def _complete(text: str, row_name: str, caret: int | None = None):
    """``completion_for`` in SKILL mode as the composer asks it."""
    return completion_for(
        text,
        len(text) if caret is None else caret,
        CompletionMode.SKILL,
        row_name,
        ARGUMENT_COMMANDS,
        COMMAND_NAMES,
        PROMPT_COMMANDS,
        NAME_COMMANDS,
    )


class TestTheFloorDecidesWhereATokenOpens:
    """Which carets inside a claimed line may open a `$` token at all."""

    def test_skill_token_opens_in_team_request_slot(self):
        """The case the slice exists for: past the name, the rest is a request."""
        token = _token("/team delivery $cte")
        assert token is not None and token.query == "cte"

    def test_skill_token_declines_in_team_name_slot(self):
        """`/team $cte` must not hijack the roster list.

        The ordering answer: while the name slot is open there is no skill token,
        so `slash_argument` resolves first and the roster owns the caret.
        """
        assert _token("/team $cte") is None

    def test_skill_token_declines_on_bare_team_space(self):
        assert _token("/team $") is None

    def test_skill_token_declines_in_partial_team_name(self):
        """`/team del$` is still inside the unterminated name token."""
        assert _token("/team del$") is None

    @pytest.mark.parametrize("text", ["/team  $x", "/team   $x", "/agent  $x"])
    def test_skill_token_declines_on_doubled_space_before_name(self, text: str):
        """Extra spaces before the name still leave the name slot OPEN.

        `partition(" ")` splits on the FIRST of two spaces, so a doubled space
        reports an empty-but-"terminated" name. Reading that as a finished name
        handed the caret to the skill picker and dropped the roster list the
        floor exists to protect.
        """
        assert _token(text) is None

    def test_skill_token_opens_in_goal_argument(self):
        """`/goal` offers no name list, so its argument starts at the request."""
        token = _token("/goal $cte")
        assert token is not None and token.query == "cte"

    @pytest.mark.parametrize("text", ["/btw $cte", "/loop $cte", "/fork $cte"])
    def test_skill_token_opens_in_btw_and_loop_arguments(self, text: str):
        """The other three no-name-slot prompt commands behave as `/goal` does."""
        token = _token(text)
        assert token is not None and token.query == "cte"

    @pytest.mark.parametrize("text", ["/model $cte", "/theme $cte", "/login $cte"])
    def test_skill_token_declines_in_enum_tail_argument(self, text: str):
        """An enum-tail argument is a fixed vocabulary, so the claim stands whole."""
        assert _token(text) is None

    def test_skill_token_declines_in_team_chart_slot(self):
        """`/team chart <name>`'s second slot is another name list, not free text."""
        assert _token("/team chart $cte") is None

    def test_skill_token_opens_after_agent_name(self):
        token = _token("/agent reviewer $cte")
        assert token is not None and token.query == "cte"

    def test_skill_token_claim_total_without_prompt_sets(self):
        """Omitting the new sets keeps the claim TOTAL — the parser-level default.

        The guarantee every caller outside the composer relies on, including the
        two `app.py` slash-argument sites and every pure-parser test: the
        relaxation is opt-in, carried by the vocabularies the composer passes.
        """
        assert skill_token("/team delivery $cte", None, COMMAND_NAMES) is None


class TestTheInlineGateAppliesUnchanged:
    """A `$` in an argument is never buffer-leading, so the inline gate applies.

    No code makes this true: `skill_token_is_leading` is already False in this
    position, so `sync_skills` sets `_skill_inline` and the lowercase-evidence
    rule holds for free. Asserted here because it is load-bearing for the
    founding money case — `/team delivery review the $5 invoice` must stay prose.
    """

    CHOICES = [
        ArgumentChoice("cte-explainer", "Explain a CTE."),
        ArgumentChoice("gitlab", "Work a merge request."),
    ]

    def _rows(self, text: str) -> list[str]:
        token = _token(text)
        if token is None:
            return []
        return [name for name, _ in skill_suggestions(token.query, self.CHOICES, inline=True)]

    def test_bare_dollar_in_command_argument_offers_nothing(self):
        assert self._rows("/team delivery $") == []

    def test_money_in_command_argument_offers_nothing(self):
        assert self._rows("/team delivery review the $5 invoice") == []

    def test_shell_variable_in_command_argument_offers_nothing(self):
        """`$PATH` carries no lowercase evidence, so it stays prose."""
        assert self._rows("/goal deploy with $PATH") == []

    def test_lowercase_prefix_in_command_argument_offers_rows(self):
        assert "cte-explainer" in self._rows("/team delivery $ct")


class TestAcceptingARowRewritesTheArgument:
    """An accepted row becomes the ARGUMENT's prefix, never the buffer's.

    The submit-side parser is anchored at offset 0 of the argument string, so a
    reassembly to the buffer start would produce `$gitlab /team delivery …` —
    a line neither the dispatcher nor the skill parser can read.
    """

    def test_completion_moves_skill_to_argument_front(self):
        assert _complete("/team delivery explain this $git", "gitlab") == (
            "/team delivery $gitlab explain this ",
            23,
        )

    def test_completion_preserves_tail_when_skill_leads_argument(self):
        """No trailing space here: the separator before the request already exists.

        The token already leads the region, so the plain span replacement is the
        whole answer and a space appended after the user's own sentence would be
        a stray character no other completion in this codebase adds.
        """
        assert _complete("/team delivery $gi fix this", "gitlab", caret=18) == (
            "/team delivery $gitlab fix this",
            23,
        )

    def test_completion_in_goal_argument(self):
        """The case that could not work while the region came from `ArgumentMode`."""
        assert _complete("/goal ship it $git", "gitlab") == ("/goal $gitlab ship it ", 14)

    @pytest.mark.parametrize(
        "text,expected,caret",
        [
            ("/btw ask it $git", "/btw $gitlab ask it ", 13),
            ("/loop drive it $git", "/loop $gitlab drive it ", 14),
            ("/fork start on $git", "/fork $gitlab start on ", 14),
        ],
    )
    def test_completion_in_btw_loop_fork_arguments(self, text: str, expected: str, caret: int):
        """The three remaining no-name-slot prompt commands rebuild in place."""
        assert _complete(text, "gitlab") == (expected, caret)

    def test_completion_outside_command_still_reassembles_buffer(self):
        """The pre-existing whole-buffer path, unregressed."""
        assert _complete("fix this $git", "gitlab") == ("$gitlab fix this ", 17)

    def test_dollar_on_second_line_of_goal_draft_is_unchanged(self):
        """LOCKED AS UNCHANGED, not as desirable.

        Line 2 carries no claiming `/`, so the floor is never consulted and the
        token takes the whole-buffer path from 92fe6402. The result is odd, and
        it is odd on origin/main too; fixing it is out of this slice's scope.
        """
        assert _complete("/goal ship it\nfix the $git", "gitlab") == (
            "$gitlab /goal ship it\nfix the ",
            30,
        )

    @pytest.mark.parametrize(
        "text,caret",
        [
            ("/team delivery explain this $git", None),
            ("/goal ship it $git", None),
            ("/agent reviewer look at this $git", None),
            ("/btw ask it $git", None),
            ("/loop drive it $git", None),
            ("/fork start on $git", None),
            ("/team delivery $gi fix this", 18),
        ],
    )
    def test_argument_reassembly_never_precedes_the_command_word(
        self, text: str, caret: int | None
    ):
        """The command word stays at index 0 in every rebuild case."""
        completed = _complete(text, "gitlab", caret=caret)
        assert completed is not None, text
        new_text, _ = completed
        assert new_text.startswith("/")
        assert new_text.index("$") > new_text.index(" ")

    def test_completion_after_agent_name(self):
        assert _complete("/agent reviewer look at this $git", "gitlab") == (
            "/agent reviewer $gitlab look at this ",
            24,
        )

    def test_ghost_withheld_for_argument_reassembly(self):
        """The result is not an append, so `ghost_for`'s startswith rule withholds it."""
        text = "/team delivery explain this $git"
        assert ghost_for(_complete(text, "gitlab"), text) == ""


class TestTheRegionCannotDriftFromTheFloor:
    """One helper answers "where does the argument start", for gate and edit alike."""

    @pytest.mark.parametrize(
        "text",
        [
            "/team delivery explain this $git",
            "/goal ship it $git",
            "/agent reviewer look at this $git",
            "/btw ask it $git",
            "/loop drive it $git",
            "/fork start on $git",
        ],
    )
    def test_region_start_equals_floor(self, text: str):
        region = _skill_argument_region(
            text, len(text), COMMAND_NAMES, PROMPT_COMMANDS, NAME_COMMANDS
        )
        assert region is not None
        floor = _skill_argument_floor(text, 0, PROMPT_COMMANDS, NAME_COMMANDS)
        assert floor is not None
        # Single-line buffers, so line_start is 0 and the two are directly comparable.
        assert region[0] == floor
        assert region[1] == len(text)


class TestPhaseDisjointness:
    """At most one picker phase may answer for any caret — the highest risk here.

    Before this slice the CLAIM delivered disjointness. It is now partial, so the
    floor delivers it instead: the phases stay mutually exclusive by construction
    and `_picker_phase`'s order still decides only who is ASKED first.
    """

    def _editor(self) -> Editor:
        editor = Editor()
        editor.set_commands(list(SLASH_COMMANDS))
        return editor

    async def _phase(self, buffer: str) -> str | None:
        """``_picker_phase()`` with the caret at the END of ``buffer``.

        Driven through a mounted app rather than a bare ``Editor``: the phase
        reads ``_caret_offset()``, which is 0 until the widget is mounted and
        the caret actually moved, so an unmounted editor would answer for column
        0 — the ``/`` — and every case here would pass for the wrong reason.
        """
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one(Editor)
            editor.focus()
            await pilot.pause()
            editor.load_text(buffer)
            editor.move_cursor(editor._end_of_buffer())
            await pilot.pause()
            return editor._picker_phase()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("buffer", ["/team ", "/team del"])
    async def test_picker_phase_is_argument_in_team_name_slot(self, buffer: str):
        assert await self._phase(buffer) == "argument"

    @pytest.mark.asyncio
    async def test_picker_phase_is_skill_in_team_request_slot(self):
        assert await self._phase("/team delivery $ct") == "skill"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "buffer",
        [
            "/team ",
            "/team del",
            "/team $",
            "/team delivery $ct",
            "/team chart $ct",
            "/model $ct",
            "/goal $ct",
            "/agent reviewer $ct",
        ],
    )
    async def test_picker_phase_never_two_answers(self, buffer: str):
        """Exactly one phase owns the caret, and a name slot is never the skill.

        NOTE ON FORMULATION — this deliberately does NOT assert that at most one
        of `skill_token` / `slash_argument` / `slash_context` is non-None, which
        is how the property was first written. That version cannot hold, and its
        failure is the slice working as designed: `slash_argument` answers for
        the WHOLE argument tail of a name-list command, `$` or no `$`
        (`/team delivery hello` already answers it on origin/main). It was alone
        there only because the total claim suppressed `skill_token`; relaxing
        that claim past the name slot — the entire point of this slice —
        necessarily makes both answer for `/team delivery $ct`. Asserting raw
        exclusivity would mean un-building the feature.

        What must actually hold, and what this asserts, is the user-visible
        contract D2 is about: `_picker_phase` resolves to exactly ONE phase, and
        a `$` is never allowed to open inside a still-open NAME slot. The
        short-circuit order then decides only who is asked first among parsers
        that may legitimately overlap — never whether a roster list can be
        hijacked, which the floor decides and the name-slot cases above pin.
        """
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one(Editor)
            editor.focus()
            await pilot.pause()
            editor.load_text(buffer)
            editor.move_cursor(editor._end_of_buffer())
            await pilot.pause()
            phase = editor._picker_phase()
            assert phase in (None, "skill", "argument", "command"), (buffer, phase)
            caret = editor._caret_offset()
            token = _token(buffer, caret)
            if token is not None:
                # A token that opened must own the phase outright...
                assert phase == "skill", (buffer, phase)
                # ...and must sit at or past the floor, never in the name slot.
                region = _skill_argument_region(
                    buffer, caret, COMMAND_NAMES, PROMPT_COMMANDS, NAME_COMMANDS
                )
                assert region is None or region[0] <= token.start, (buffer, region)

    def test_name_prompt_commands_match_name_argument_commands(self):
        """The registry-derived set must equal the hand-written one it describes.

        A guard, not a tautology: if a `consumes_prompt` command grows an argument
        list that is not a team/agent roster, the two-set formulation stops
        describing the registry and the floor's name-slot rule needs rethinking.
        """
        editor = self._editor()
        assert editor._name_prompt_commands == frozenset(Editor.NAME_ARGUMENT_COMMANDS)


class TestTheComposerOpensTheListInARequestSlot:
    """The composer-level half: the picker really opens, end to end in the app."""

    async def _draft(self, app, pilot, text: str) -> Editor:
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
    async def test_a_sigil_in_a_prompt_command_request_opens_the_skill_list(
        self, skill_root: Path  # noqa: F811
    ) -> None:
        """The inverse of the enum-tail case.

        Past a terminated name slot the request is free text, so `$res` is an
        invocation and not prose. This is the coverage the old
        `test_a_sigil_inside_an_engaged_command_is_plain_text` gave up when its
        buffer became the very case the slice exists to open.
        """
        session = FakeSession()
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = await self._draft(app, pilot, "/team ops $res")
            assert editor.picker.mode is PickerMode.SKILL

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
    SKILL_SUBMIT_COMMANDS,
    CompletionMode,
    PickerMode,
    SkillScanRegion,
    _skill_argument_floor,
    completion_for,
    ghost_for,
    skill_scan_region,
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
# The registry's `name_argument` FLAG, which is the fact `Editor.set_commands`
# and `CommandPicker.set_commands` both read now. Deliberately not
# `arguments is not ArgumentMode.NONE`: those two stopped being the same
# question when `/goal` and `/loop` gained a value list with no name slot, and
# reading the proxy here would assert the floor against a vocabulary the
# composer no longer uses — the drift this fixture exists to catch.
NAME_COMMANDS = frozenset(
    name.lower()
    for c in SLASH_COMMANDS
    if c.consumes_prompt and c.name_argument
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


class TestTheScanRegionIsTheSurfaceRule:
    """Which cells the ink may scan, mirroring the submit path's own exits."""

    def test_a_plain_message_scans_the_whole_buffer(self):
        text = "fix $research now"
        assert skill_scan_region(text, COMMAND_NAMES, PROMPT_COMMANDS, NAME_COMMANDS) == (
            SkillScanRegion(0, len(text), None)
        )

    def test_an_unknown_leading_slash_is_prose(self):
        """`/tmp/test` is dispatched by nothing, so submit scans it as a prompt."""
        text = "/tmp/test $research"
        region = skill_scan_region(text, COMMAND_NAMES, PROMPT_COMMANDS, NAME_COMMANDS)
        assert region is not None and region.start == 0

    @pytest.mark.parametrize(
        "text",
        [
            "/btw $research",
            "/help $research",
            "/model $research",
            "/loop $research",
            "/fork $research",
        ],
    )
    def test_commands_that_never_expand_scan_nothing(self, text: str):
        assert skill_scan_region(text, COMMAND_NAMES, PROMPT_COMMANDS, NAME_COMMANDS) is None, text

    @pytest.mark.parametrize("text", ["!echo $research", "! $research"])
    def test_a_bang_line_scans_nothing(self, text: str):
        assert (
            skill_scan_region(text, COMMAND_NAMES, PROMPT_COMMANDS, NAME_COMMANDS, shell=True)
            is None
        )

    def test_shell_mode_scans_nothing(self):
        assert (
            skill_scan_region(
                "echo $research", COMMAND_NAMES, PROMPT_COMMANDS, NAME_COMMANDS, shell=True
            )
            is None
        )

    @pytest.mark.parametrize(
        "text",
        [
            "/team backend fix $research",
            "/agent reviewer fix $research",
            "/goal fix $research",
            "/aida use $research",
        ],
    )
    def test_a_skill_bearing_argument_tail_is_the_region(self, text: str):
        region = skill_scan_region(text, COMMAND_NAMES, PROMPT_COMMANDS, NAME_COMMANDS)
        assert region is not None
        assert region.start <= text.index("$research")
        assert region.end == len(text)
        assert region.command in SKILL_SUBMIT_COMMANDS

    def test_a_team_name_slot_scans_nothing(self):
        """`/team backend` with no request yet: no argument tail to scan."""
        assert (
            skill_scan_region("/team backend", COMMAND_NAMES, PROMPT_COMMANDS, NAME_COMMANDS)
            is None
        )

    def test_argument_mode_scans_the_whole_string(self):
        """`_submit_command_prompt` already holds the tail; its first token leads."""
        text = "fix $RESEARCH"
        assert skill_scan_region(text, argument=True) == SkillScanRegion(0, len(text), None)

    def test_a_draft_that_opens_with_a_blank_line_still_finds_its_command(self):
        """The claim is read from the line it sits on, not from line 0.

        A leading newline used to make the region `None` while submit still
        dispatched `/goal` and expanded the tail — an under-claim, but one the
        ink rule exists to avoid (review round 2, n1).
        """
        text = "\n/goal fix $research"
        region = skill_scan_region(text, COMMAND_NAMES, PROMPT_COMMANDS, NAME_COMMANDS)
        assert region is not None
        assert region.start == text.index("fix")
        assert region.end == len(text)
        assert region.command == "goal"

    def test_the_skill_bearing_set_is_a_subset_of_prompt_commands(self):
        """The ink may only claim command tails whose handler expands a skill.

        `/aida` reaches `_submit_command_prompt` through `_cmd_aida`, so it is
        IN the set; the three prompt-consuming commands that start their own
        flow are out (review round 2, m1).
        """
        prompt_names = {name for c in SLASH_COMMANDS if c.consumes_prompt for name in c.names}
        assert SKILL_SUBMIT_COMMANDS <= prompt_names
        assert {"btw", "loop", "fork"}.isdisjoint(SKILL_SUBMIT_COMMANDS)
        assert "aida" in SKILL_SUBMIT_COMMANDS


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


class TestAcceptingARowReplacesOnlyTheToken:
    """An accepted row edits the token in place, wherever it sits.

    The submit-side trigger scans the whole argument, so there is nothing to
    move: the command word, its name slot and every other character survive
    byte-for-byte, and only `$<query>` becomes `$<name>` plus the terminating
    space when the next character is not whitespace.
    """

    @staticmethod
    def _expected(text: str, name: str = "gitlab") -> tuple[str, int]:
        start = text.index("$")
        tail = text[text.index(" ", start) :] if " " in text[start:] else ""
        return f"{text[:start]}${name} {tail}" if not tail else f"{text[:start]}${name}{tail}", (
            start + len(name) + 2
        )

    def test_completion_mid_argument_keeps_every_earlier_word(self):
        text = "/team delivery explain this $git"
        assert _complete(text, "gitlab") == self._expected(text)

    def test_completion_preserves_tail_when_skill_leads_argument(self):
        """A request already after the token survives with its separator."""
        text = "/team delivery $gi fix this"
        assert _complete(text, "gitlab", caret=18) == (
            "/team delivery $gitlab fix this",
            23,
        )

    def test_completion_in_goal_argument(self):
        text = "/goal ship it $git"
        assert _complete(text, "gitlab") == self._expected(text)

    def test_completion_in_btw_loop_fork_arguments(self):
        for text in ("/btw ask it $git", "/loop drive it $git", "/fork start on $git"):
            assert _complete(text, "gitlab") == self._expected(text), text

    def test_completion_outside_command_is_the_same_edit(self):
        text = "fix this $git"
        assert _complete(text, "gitlab") == self._expected(text)

    def test_dollar_on_second_line_keeps_the_line_break(self):
        text = "/goal ship it\nfix the $git"
        assert _complete(text, "gitlab") == self._expected(text)

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
    def test_the_edit_never_moves_the_command_word(self, text: str, caret: int | None):
        completed = _complete(text, "gitlab", caret=caret)
        assert completed is not None, text
        new_text, _ = completed
        assert new_text.startswith("/")
        assert new_text[: text.index("$")] == text[: text.index("$")]

    def test_completion_after_agent_name(self):
        text = "/agent reviewer look at this $git"
        assert _complete(text, "gitlab") == self._expected(text)

    def test_ghost_previews_the_span_replacement(self):
        text = "/team delivery explain this $git"
        assert ghost_for(_complete(text, "gitlab"), text) == "lab "


class TestTheFloorDecidesWhereTheArgumentStarts:
    """One helper answers "where may a `$` open inside a claim"."""

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
    def test_floor_is_at_or_before_the_token(self, text: str):
        floor = _skill_argument_floor(text, 0, PROMPT_COMMANDS, NAME_COMMANDS)
        assert floor is not None
        assert floor <= text.index("$")


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
                # A token that opened must own the phase outright. The floor's
                # own half (never inside the name slot) is pinned by
                # `TestTheFloorDecidesWhereATokenOpens`.
                assert phase == "skill", (buffer, phase)

    def test_name_prompt_commands_match_name_argument_commands(self):
        """The registry-derived set must equal the hand-written one it describes.

        A guard, not a tautology: the composer's set is derived from the
        registry's ``name_argument`` flag and this tuple is hand-written, so the
        two agree only while the flag is set on exactly the commands the slate
        names. `test_slash_goal_loop_flags` pins the flag itself against the
        registry, which is the half this assertion cannot see.
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

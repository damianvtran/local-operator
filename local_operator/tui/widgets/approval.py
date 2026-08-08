"""The in-TUI tool-approval prompt.

Why this exists at all: the harness gates write/exec tier tools behind an
approval callback, and the default callback is an ``input()`` on stdin. That
works for the plain REPL and it DEADLOCKS under the TUI — Textual owns the
terminal in raw mode and consumes every keystroke, so the thread blocked on
``input()`` never receives a line, the awaiting turn never resumes, and the
session hangs forever with its tool cards stuck on "running". That is not a
hypothetical: it is the reported freeze this module fixes, reproduced as two
`bash` cards that never completed while the working line kept animating.

So the TUI answers approvals ITSELF. The app installs
:meth:`OperatorApp.request_tool_approval` as the session's approval handler and
this block is the surface: one focused card holding the question, resolved by a
keystroke into an :class:`asyncio.Future` the engine is awaiting.

Design constraints:

- The card takes FOCUS. The alternative — app-level bindings while the editor
  keeps focus — cannot work, because the answers are plain letters and the
  editor is a text buffer that would swallow them as input.
- Focus is RESTORED to whatever held it, so answering a prompt does not silently
  move the user out of the composer.
- Every exit path resolves the future exactly once (:meth:`resolve`). A dropped
  future is the same hang this module exists to remove, so the app also resolves
  it on abort, on transcript clear, and on unmount.
- ``a`` (allow all) is scoped to the SESSION and is the app's flag, not this
  widget's: the widget reports the answer, the app decides what to remember.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from rich.cells import cell_len
from rich.console import Group, RenderableType
from rich.style import Style
from rich.text import Text
from textual.events import Key

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.transcript import SPINE_INDENT, TranscriptBlock

#: What each key does, in the order the row prints them. Kept as data so the
#: hint row and the key handler cannot disagree about what is on offer.
#:
#: "Allow all" is ``A``, not ``a``: it disarms the approval gate for the whole
#: session, and this block takes focus, so a user who did not notice that and
#: kept typing their next instruction would hit the most common letter in
#: English and permanently disable the gate with no receipt. Every per-call
#: answer stays a single keystroke, because those are per-call.
#: Ordered refuse -> allow -> stop -> allow-all, and the order does two jobs.
#:
#: It opens on the refusal rather than the yes-word, so the row reads as a
#: decision rather than a prompt to agree. And because the hint row sheds WHOLE
#: choices from the right, the order is also the priority: the session-wide
#: switch stops being advertised first, then the global stop key (which means the
#: same thing everywhere in the app and is not this prompt's to teach), leaving
#: the two per-call ANSWERS — the ones a user cannot proceed without — surviving
#: to the narrowest terminals. A strict permissiveness gradient would have put
#: `esc stop` last and cost `y allow` at 24 columns, which is worse: a prompt
#: that only tells you how to refuse is a prompt you cannot get past.
CHOICES: tuple[tuple[str, str], ...] = (
    ("n", "deny"),
    ("y", "allow"),
    ("esc", "stop"),
    ("A", "allow all"),
)

#: The hazard clause, in words rather than the parser's own bracket token.
HAZARD_WORDS = "outside the workspace — "
#: Its narrow-width form. The hazard must never be what costs the row its target,
#: so below the width where both fit it degrades to this and the path survives.
HAZARD_MARKER = "! "
#: The prompt glyph a LIVE hazardous ask uses when no clause fits (the ladder's
#: floor). Deliberately the same character as the marker and as the transcript's
#: warning notice: one symbol for "this needs your attention", wherever it lands.
HAZARD_GLYPH = "!"
#: Cells of target that must still fit for the full words to be worth their room.
HAZARD_MIN_TARGET = 12
#: The two cells between the tool name and the detail. Named because the hazard
#: has to reserve its room BEFORE this is spent, not out of what it leaves.
_SEPARATOR_CELLS = 2
#: Cells below which the target says nothing worth its room. `…s` is not a
#: shorter answer to "which file", it is a different question; the row is more
#: honest ending after the tool name than ending on a one-character stub.
TARGET_MIN_USEFUL = 4

#: Marker prefix the builtin tools put on a description whose target sits
#: outside the workspace. Surfaced as its own tinted clause because "outside
#: the workspace" is the single most decision-relevant fact in the prompt.
OUTSIDE_MARKER = "[outside workspace]"

#: Glyph opening the question row. Deliberately not the tool ledger's marker:
#: this row is a QUESTION, and it must not read as one more completed action.
PROMPT_GLYPH = "?"


def fg_style() -> Style:
    """The key ink. A function, not a constant: the theme can change at runtime."""
    return Style(color=theme_mod.semantic_color("fg"))


def fit_tail(text: str, width: int) -> str:
    """``text`` reduced to ``width`` cells, keeping its TAIL and shortening $HOME.

    The opposite end from :func:`truncate_cells`, because the two are used on
    different kinds of string. A tool summary's meaning is front-loaded; a PATH's
    is at the end — the basename is what distinguishes `~/.ssh/authorized_keys`
    from `~/Documents/notes.md`, and both share a leading `/Users/<name>/` that a
    right-truncation spends the whole budget on.

    ``$HOME`` collapses to ``~`` first, which is free legibility: the shell's own
    shorthand, and it buys back the cells the prefix was costing. The same two
    moves the welcome view already makes for a path that matters less.
    """
    shortened = _shorten_home(text)
    if cell_len(shortened) <= width:
        return shortened
    if width <= 1:
        return "…"[:width]
    kept: list[str] = []
    used = 1  # the leading ellipsis
    for char in reversed(shortened):
        size = cell_len(char)
        if used + size > width:
            break
        kept.append(char)
        used += size
    return "…" + "".join(reversed(kept))


def _shorten_home(text: str) -> str:
    """Collapse the home directory to ``~`` (the shell's own shorthand)."""
    home = str(Path.home())
    return text.replace(home, "~") if home and home != "/" else text


class ApprovalBlock(TranscriptBlock):
    """One pending approval: a focused, amber-tinted question row.

    Lifecycle: constructed with the tool name and the resolved description,
    mounted, focused. :meth:`resolve` settles the future and repaints the row as
    a receipt (what was asked, what was answered) so the transcript keeps the
    decision instead of the question vanishing without a trace.
    """

    #: Always give the question a blank row above it: it interrupts whatever
    #: was happening, and a question flush against a tool row reads as output.
    SPACING_LEAD = True
    SPACING_KIND = "approval"

    #: Focusable so the answer keys reach it rather than the editor's buffer.
    can_focus = True

    #: Escape is NOT bound here. It means "stop" everywhere else in the app, and
    #: a key that stops the run must not quietly become "answer one question" —
    #: with two concurrent prompts that cost one Esc per prompt before the run
    #: actually stopped. It bubbles to the app, which denies every queued prompt
    #: and aborts; ``n`` is the answer that refuses just this tool.
    BINDINGS = [
        ("y", "answer('y')", "Allow"),
        ("n", "answer('n')", "Deny"),
        ("A", "answer('a')", "Allow all"),
    ]

    #: The answer keys, excluded from the typing passthrough below. Spelled out
    #: rather than derived from BINDINGS: that attribute is typed as accepting
    #: Binding objects as well as the bare tuples used above, so reading `[0]`
    #: off it does not type-check. The set is pinned against BINDINGS by a test
    #: instead, which catches drift without a cast here.
    _ANSWER_KEYS = frozenset({"y", "n", "A"})

    def __init__(
        self,
        tool_name: str,
        description: str,
        on_answer: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.add_class("approval-card")
        self.tool_name = tool_name
        self.description = description
        self._answer: str | None = None
        # `get_running_loop`, not `get_event_loop`: the future must belong to the
        # loop that will await it, and stating that precondition turns a future
        # construction from a sync context into an immediate error instead of a
        # future nobody resolves. (3.14 removes the implicit-loop fallback.)
        self._future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._on_answer = on_answer
        self._restore_focus: object | None = None
        self._refresh_row()

    # -- the awaited half ----------------------------------------------------
    def wait(self) -> asyncio.Future[bool]:
        """The future the engine awaits. Resolved exactly once."""
        return self._future

    def resolve(self, approved: bool, *, answer: str | None = None) -> None:
        """Settle the prompt (idempotent) and repaint it as a receipt.

        Idempotent because several paths can end one prompt — the keystroke, an
        abort, a transcript clear, unmount — and a second ``set_result`` on a
        settled future raises. Losing that race must not take the app down.
        """
        if self._answer is not None:
            return
        self._answer = answer or ("y" if approved else "n")
        # BEFORE `set_result`, and a direct call rather than a posted message:
        # the app owns what an answer MEANS beyond this call (latching "allow
        # all"), and a queued asker wakes the moment the future resolves. Routed
        # through the message pump, the flag landed several pump hops LATER, so
        # the waiter read a stale policy and asked the user again for the second
        # tool of the same batch — immediately after they pressed "allow all".
        if self._on_answer is not None:
            self._on_answer(self._answer)
        if not self._future.done():
            self._future.set_result(approved)
        self.remove_class("approval-pending")
        self._refresh_row()
        self.finalize()

    @property
    def answered(self) -> bool:
        return self._answer is not None

    # -- keys ---------------------------------------------------------------
    def action_answer(self, key: str) -> None:
        """Answer from a keystroke: ``y`` once, ``a`` for the session, ``n`` deny."""
        self.resolve(key in ("y", "a"), answer=key)

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        """Typing while the prompt holds focus goes to the COMPOSER, not nowhere.

        The block takes focus so its answer keys are not typed into the buffer,
        which leaves a user who did not notice typing into a widget that ignores
        them: every keystroke of a sentence vanished. Any printable key that is
        not an answer hands focus to the composer and is re-posted there, so the
        sentence survives and the user learns where the caret went.

        Runs BEFORE ``BINDINGS`` (Textual dispatches the focused widget's
        handlers first), so the answer keys are excluded by hand.

        The prompt stays PENDING — the question is still on screen with its hint
        row, and clicking it (see :meth:`on_click`) brings focus back. It is not
        dismissed, because the engine is still waiting on the answer.
        """
        if event.key in self._ANSWER_KEYS or not event.is_printable:
            return
        composer = self._composer()
        if composer is None:
            return
        composer.focus()
        composer.post_message(Key(event.key, event.character))
        event.stop()
        event.prevent_default()

    def _composer(self):  # type: ignore[no-untyped-def]
        """The app's one text input, or None when there is not one.

        Imported lazily and queried defensively for the same reason the tool card
        does it: this block is mounted in harnesses that host a transcript and
        nothing else, where a missing composer must degrade to "the key does
        nothing" rather than raise out of a key handler.
        """
        from local_operator.tui.widgets.editor import Editor

        screen = self.screen
        if screen is None:
            return None
        try:
            return screen.query_one(Editor)
        except Exception:
            return None

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """Clicking an unanswered prompt takes focus back so the keys work again.

        The way back after the passthrough above moved focus to the composer —
        without it the question would be on screen with no discoverable way to
        answer it.
        """
        if self.answered:
            return
        self.focus()
        event.stop()

    def on_mount(self) -> None:
        """Take focus, remembering what had it so it can be handed back."""
        self.add_class("approval-pending")
        screen = self.screen
        self._restore_focus = screen.focused if screen is not None else None
        self.focus()

    def restore_focus(self) -> None:
        """Return focus to whatever held it when the prompt appeared."""
        widget = self._restore_focus
        self._restore_focus = None
        if widget is not None and getattr(widget, "is_attached", False):
            widget.focus()  # type: ignore[attr-defined]

    # -- rendering ----------------------------------------------------------
    def _refresh_row(self) -> None:
        """Rebuild the card at its own width (same discipline as the tool row).

        Bypasses the finalize guard on purpose: a settled prompt still has to
        re-fit on resize, and the content is a pure function of state.
        """
        was_finalized = self._finalized
        self._finalized = False
        try:
            self.set_content(self._build())
        finally:
            self._finalized = was_finalized

    def on_resize(self, event: object) -> None:
        self._refresh_row()

    def _build(self) -> RenderableType:
        width = max((self.size.width or 80) - 2, 10)
        # Three rungs of concession, each cheaper than the next thing it protects.
        # The word `allow` goes first because it is the one token on the row said
        # twice already (the `?` asks the question and the hint row answers it,
        # `n deny · y allow`). Below that the hazard moves INTO the glyph: `!` in
        # place of `?` costs nothing, so the narrowest terminal still tells the two
        # asks apart instead of clipping the tool name — which is the one string a
        # security prompt may never abbreviate.
        question, hazard_shown = self._compose_question(width, verb=True, glyph_hazard=False)
        if not hazard_shown and self._answer is None and self._detail()[1]:
            for verb, glyph_hazard in ((False, False), (False, True)):
                candidate, shown = self._compose_question(
                    width, verb=verb, glyph_hazard=glyph_hazard
                )
                question = candidate
                if shown:
                    break

        if self._answer is not None:
            # A Group even with one child: rich honours ``Text.no_wrap`` for a
            # Group's children but NOT for a bare Text handed to a Static, so the
            # answered receipt wrapped onto column zero — the composer's own
            # gutter — at narrow widths while the pending two-row form did not.
            return Group(question)
        return Group(question, self._hint_row(width))

    def _compose_question(self, width: int, *, verb: bool, glyph_hazard: bool) -> tuple[Text, bool]:
        """The question row, and whether the hazard reached the user at all.

        ``glyph_hazard`` moves the warning from a clause into the prompt glyph, for
        the widths where no clause fits. It is the ladder's floor because it is
        free: the glyph cell is already spent and already carries warning ink.
        """
        warning = Style(color=theme_mod.semantic_color("warning"))
        fg = Style(color=theme_mod.semantic_color("fg"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        danger = Style(color=theme_mod.semantic_color("danger"))

        question = Text(" " * SPINE_INDENT, no_wrap=True, overflow="ellipsis")
        hazard_shown = False
        if self._answer is None:
            # The question glyph is the ONE place the prompt spends warning ink
            # while it is live, so the hazard clause below can outrank it — unless
            # the glyph IS the hazard, in which case it takes the bold weight the
            # clause would have had.
            if glyph_hazard:
                question.append(f"{HAZARD_GLYPH} ", style=warning + Style(bold=True))
                hazard_shown = True
            else:
                question.append(f"{PROMPT_GLYPH} ", style=warning)
            if verb:
                question.append("allow ", style=muted)
        else:
            allowed = self._answer in ("y", "a")
            question.append("✓ " if allowed else "✗ ", style=fg if allowed else danger)
            # Never shed on a receipt: `allowed`/`denied` IS the outcome, which is
            # the only reason the settled row is still on screen.
            question.append("allowed " if allowed else "denied ", style=muted)
        # The RAW tool name, always whole, where the ledger row two lines below
        # shows the shortened `display_name`. The divergence is intended and is
        # not a consistency bug to fix: a security prompt must name exactly what
        # is being authorised (`mcp__linear_create_issue`, server and all), while
        # the ledger optimises for scanning a column of settled actions.
        question.append(self.tool_name, style=fg)

        detail, outside = self._detail()
        hazard_shown = False
        if detail:
            # The hazard's room is reserved BEFORE anything else is spent, and it
            # is the LAST thing shed. Below ~32 columns the two cells carrying
            # `!` and the two cells carrying the tail of a path are competing for
            # the same room, and a one-character path stub (`…s`) tells the user
            # nothing that helps them answer, while `!` tells them the one fact
            # that changes the answer. Shedding in the other order made the
            # dangerous prompt paint BYTE-IDENTICAL to the safe one, which is the
            # version of "says less when it matters more" the eye cannot catch.
            #
            # On a settled receipt the hazard drops to plain dim: the decision is
            # made, and a permanent alarm in the transcript trains the eye to
            # ignore the live one.
            hazard_style = warning + Style(bold=True) if self._answer is None else dim
            spare = width - question.cell_len - _SEPARATOR_CELLS
            hazard = ""
            if outside:
                if spare - cell_len(HAZARD_WORDS) >= HAZARD_MIN_TARGET:
                    hazard = HAZARD_WORDS
                elif spare >= cell_len(HAZARD_MARKER):
                    hazard = HAZARD_MARKER
            question.append("  ", style=dim)
            if hazard:
                question.append(hazard, style=hazard_style)
                hazard_shown = True
            budget = width - question.cell_len
            if budget >= TARGET_MIN_USEFUL:
                target_verb, target = self._split_target(detail)
                if target_verb and budget > cell_len(target_verb) + TARGET_MIN_USEFUL:
                    question.append(target_verb, style=dim)
                    budget = width - question.cell_len
                else:
                    target = detail
                # The target is the string the whole prompt exists to have read,
                # so it gets the brightest ink and keeps its TAIL. A resolved
                # absolute path truncated from the right spends the entire narrow
                # budget on `/Users/<name>/` boilerplate, which made
                # `~/.ssh/authorized_keys` and `~/Documents/notes.md` render
                # identically — the two asks a user most needs told apart.
                question.append(fit_tail(target, budget), style=fg)
        return question, hazard_shown

    def _hint_row(self, width: int) -> Text:
        """The key hints, shedding WHOLE choices rather than truncating one.

        Sheds from the right, which is what makes :data:`CHOICES`' order a
        priority list as well as a reading order: the session-wide switch stops
        being advertised first, then the global stop key, leaving the two per-call
        answers longest. Truncating instead left rows ending `A allow all …` and
        then `A …`, offering a key whose consequence had been cut off.
        """
        indent = " " * (SPINE_INDENT + 2)
        for count in range(len(CHOICES), 1, -1):
            row = self._render_hints(indent, CHOICES[:count])
            if cell_len(row.plain) <= width:
                return row
        # Above the single labelled choice, not below it. `n/y` costs 5 cells and
        # advertises BOTH answers; `n deny` costs 10 and advertises one — so the
        # labelled rung was winning at widths where the legend fits, leaving a
        # prompt that only says how to refuse. Placed here the legend is reachable
        # (24 down to 14) instead of dead code below a rung that always fits.
        legend = self._render_legend(indent)
        if cell_len(legend.plain) <= width:
            return legend
        return self._render_hints(indent, CHOICES[:1])

    def _render_legend(self, indent: str) -> Text:
        """``n/y`` — the keys alone, both answers, no labels."""
        dim = Style(color=theme_mod.semantic_color("dim"))
        row = Text(indent, no_wrap=True, overflow="ellipsis")
        for index, (key, _) in enumerate(CHOICES[:2]):
            if index:
                row.append("/", style=dim)
            row.append(key, style=fg_style())
        return row

    def _render_hints(self, indent: str, choices: tuple[tuple[str, str], ...]) -> Text:
        row = Text(indent, no_wrap=True, overflow="ellipsis")
        dim = Style(color=theme_mod.semantic_color("dim"))
        for index, (key, label) in enumerate(choices):
            if index:
                row.append(" · ", style=dim)
            # Keys in `fg`, not warning: a tint worn by the hazard AND by the key
            # that REFUSES distinguishes nothing. The keys are affordances, so
            # they read as the brightest plain ink and leave warning to the alarm.
            row.append(key, style=fg_style())
            row.append(f" {label}", style=dim)
        return row

    def _split_target(self, detail: str) -> tuple[str, str]:
        """``("write: ", "/path")`` when the description carries a verb prefix.

        The builtin tools emit ``<action>: <resolved path>``. Splitting lets the
        verb stay quiet while the target takes the bright ink and the tail-aware
        truncation; a description with no ``: `` is returned whole as the target.
        """
        head, separator, tail = detail.partition(": ")
        if not separator or not tail:
            return "", detail
        return f"{head}: ", tail

    def _detail(self) -> tuple[str, bool]:
        """``(description, outside_workspace)`` with the marker lifted out."""
        text = " ".join(self.description.split())
        if text.startswith(OUTSIDE_MARKER):
            return text[len(OUTSIDE_MARKER) :].strip(), True
        return text, False

    def spans_multiple_rows(self) -> bool:
        # Two rows while pending (question + hints), one once answered. Answered
        # from state so the spacing rule never renders the block to find out.
        return self._answer is None

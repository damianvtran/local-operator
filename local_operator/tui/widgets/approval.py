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
from dataclasses import dataclass
from pathlib import Path

from rich.cells import cell_len
from rich.console import Group, RenderableType
from rich.style import Style
from rich.text import Text
from textual.events import Key

from local_operator.ansi import strip_control_sequences
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.tool_card import truncate_cells
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
#: Cells the target must keep for a `verb:` label to be worth its own room. A
#: label plus a stub names a category and hides the instance, which on this row
#: means two different authorisations paint the same text.
TARGET_LABELLED_MIN = 10

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


#: Description verbs whose target is a PATH. Everything else keeps its head.
#: Declared rather than sniffed: `run: /usr/local/bin/deploy --env prod` starts
#: with a slash and is not a path, and treating it as one truncated away the
#: binary (`run: …od --force`) and rewrote `$HOME` inside a command string —
#: a prompt showing a command that is not the command that will run.
PATH_VERBS = frozenset({"write", "edit", "read", "grep", "append"})


def fit_target(text: str, width: int, *, is_path: bool) -> str:
    """``text`` reduced to ``width`` cells, keeping the end that carries meaning.

    A PATH's meaning is at its tail: `/Users/<name>/` is boilerplate every path
    on the machine shares, and the basename is the whole difference between the
    ask a user must refuse and the one they can wave through. A COMMAND's is at
    its head: `rm -rf` decides the answer and the flags after it rarely change
    it, and a command truncated from the left reads as a fragment of something
    unknown rather than as a shortened version of itself.

    Detected from the string rather than declared by the caller, because the
    approval description is a rendered sentence by the time the UI sees it. The
    test is deliberately narrow — an absolute or home-relative path — so anything
    ambiguous keeps its head, which is the safer default for prose.
    """
    if is_path:
        return fit_tail(text, width)
    return truncate_cells(text, width)


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


@dataclass(frozen=True)
class _Row:
    """One candidate rendering of the question row, with what it managed to say.

    Built so the rungs can be COMPARED rather than ordered by hand: `fits` is
    whether the tool name survived un-clipped, `hazard` whether the risk reached
    the user, `target` how many cells of the actual subject are on screen.
    """

    text: Text
    #: 2 = the full clause, 1 = the `!` marker or glyph, 0 = the risk is not on
    #: the row. A rank rather than a bool because degrading the 24-cell clause to
    #: two cells is a REAL loss that has to compete with what those 22 cells buy;
    #: scored as a bool it looked free, and the wordier rung won a six-column band
    #: where the narrower frame said more.
    hazard: int
    target: int
    prefix: int
    #: Whether this rendering kept the two-cell spine indent, so the hint row
    #: below it can start in the same column as the question above it.
    spine: bool

    def score(self, width: int) -> tuple[int, int, int, int, int]:
        """Higher is better, compared left to right.

        The last term is ZERO for any row that fits, so it can only decide
        between rows that are all overflowing — a terminal so narrow that even
        the glyph and the tool name do not fit. There the fewest cells spent
        before the name is the most name the user gets to read. Left ungated it
        inverted the whole ladder: at 100 columns, where the detail fits either
        way, "shorter prefix" quietly outranked "say the word `allowed`".
        """
        fits = cell_len(self.text.plain) <= width
        # Three separate questions, in the order they matter:
        #
        # 1. Is the risk on the row AT ALL (`min(hazard, 1)`)? This outranks the
        #    fit: at one width the safety glyph was the single cell breaking the
        #    fit, so the ladder traded it and the dangerous receipt painted
        #    exactly like the safe one. Overflowing by a cell costs the last
        #    character of a name the row repeats two words earlier.
        # 2. Does the row fit, and how much of the SENTENCE does it show?
        # 3. Only then, is the risk spelled out in full (`hazard` == 2)?
        #
        # Ranking the FORM first instead made it dominate rather than compete:
        # over a ten-column band the 24-cell clause won against the whole path,
        # and the row showed less of the target than the same prompt one column
        # narrower — the exact non-monotonicity this scoring replaced a hand-
        # ordered ladder to prevent.
        return (
            min(self.hazard, 1),
            int(fits),
            self.target,
            self.hazard,
            0 if fits else -self.prefix,
        )


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
        # Both strings are MODEL-CONTROLLED and both reach a real terminal, so
        # both are stripped here — the same discipline `ToolCard` applies to every
        # untrusted string it renders. This prompt is the surface where it matters
        # most: the description now carries the tool's own rendering of its
        # arguments (a shell command, a URL) where it used to carry a JSON dump,
        # whose escaping made control bytes inert by accident. Without this, a
        # command argument containing CSI can erase the row above the prompt and
        # repaint a forged receipt over it, mis-measure the width ladder (cell_len
        # counts the escape bytes), and be cut mid-sequence by truncation.
        self.tool_name = strip_control_sequences(tool_name)
        self.description = strip_control_sequences(description)
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
        # Every rung is BUILT and the best one is chosen, rather than walking the
        # ladder until something fits. Walking produced a row that was not
        # monotonic in the width: between 30 and 33 columns the clause fitted on
        # the first rung, so the loop never ran, so the row kept six cells of
        # `allow` and showed no target at all — while the SAME prompt at 29
        # columns, one rung down, showed the target. A narrower frame that says
        # more is a bug the user experiences as the UI flickering between two
        # designs, and no amount of rung ordering fixes it: the rungs are not
        # totally ordered by width, so the choice has to be made by measuring.
        #
        # Preference order, applied to every variant: the row must FIT (the tool
        # name is the one string this prompt exists to name, and Rich clipping it
        # is not a concession the ladder chose), then the hazard must be visible,
        # then as much of the target as possible. Ties go to the earliest rung,
        # which is the most explicit wording.
        shapes: list[tuple[bool, bool]] = [(True, False), (False, False)]
        if self._detail()[1]:
            shapes.append((False, True))
        # The spine indent is the LAST thing offered up, and only because at 16
        # columns it is 12% of the row spent on alignment with blocks that are
        # scrolled off screen. A rung, not a special case, so it competes on the
        # same terms as everything else and never fires while the row fits.
        rungs = [(verb, glyph, True) for verb, glyph in shapes]
        rungs += [(verb, glyph, False) for verb, glyph in shapes]
        best: _Row | None = None
        for verb, glyph_hazard, spine in rungs:
            row = self._compose_question(width, verb=verb, glyph_hazard=glyph_hazard, spine=spine)
            if best is None or row.score(width) > best.score(width):
                best = row
        assert best is not None
        question = best.text

        if self._answer is not None:
            # A Group even with one child: rich honours ``Text.no_wrap`` for a
            # Group's children but NOT for a bare Text handed to a Static, so the
            # answered receipt wrapped onto column zero — the composer's own
            # gutter — at narrow widths while the pending two-row form did not.
            return Group(question)
        # The hint row follows the question's SPINE decision. Hard-coding the
        # indent let the block's two rows disagree by four cells whenever the
        # spine rung fired — one row at column 0 and its own answer keys at
        # column 4, which reads as two unrelated things rather than one prompt.
        return Group(question, self._hint_row(width, spine=best.spine))

    def _hazard_ink(self) -> Style:
        """Amber and bold while the question is LIVE; plain dim once it is not.

        One helper, because the three places that paint the hazard drifted: the
        clause honoured this rule and the two glyph forms did not, so below ~53
        columns a scrollback of settled receipts wore the live alarm's amber
        permanently. A transcript where four consecutive rows are alarms trains
        the eye to ignore the one that is asking for an answer.
        """
        if self._answer is None:
            return Style(color=theme_mod.semantic_color("warning"), bold=True)
        return Style(color=theme_mod.semantic_color("dim"))

    def _compose_question(
        self, width: int, *, verb: bool, glyph_hazard: bool, spine: bool = True
    ) -> _Row:
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

        indent = SPINE_INDENT if spine else 0
        question = Text(" " * indent, no_wrap=True, overflow="ellipsis")
        hazard_rank = 0
        if self._answer is None:
            # The question glyph is the ONE place the prompt spends warning ink
            # while it is live, so the hazard clause below can outrank it — unless
            # the glyph IS the hazard, in which case it takes the bold weight the
            # clause would have had.
            if glyph_hazard:
                question.append(f"{HAZARD_GLYPH} ", style=self._hazard_ink())
                hazard_rank = 1
            else:
                question.append(f"{PROMPT_GLYPH} ", style=warning)
            if verb:
                question.append("allow ", style=muted)
        else:
            allowed = self._answer in ("y", "a")
            question.append("✓" if allowed else "✗", style=fg if allowed else danger)
            if glyph_hazard:
                # The receipt's floor rung. The outcome glyph cannot be given up —
                # it is the whole reason a settled row is still on screen — so the
                # hazard rides BESIDE it in the space that was there anyway, for
                # one cell instead of the clause's three. Outcome first, because
                # the record answers "what happened" before "how risky was it".
                question.append(HAZARD_GLYPH, style=self._hazard_ink())
                hazard_rank = 1
            question.append(" ")
            if verb:
                # The word repeats the glyph beside it, so it is shed on the same
                # rung as the live ask's `allow` — and only ever to keep the
                # hazard, which is the one thing on the row the glyph does NOT
                # already say. `allowed ` costs 8 cells where `allow ` costs 6,
                # which is why the receipt used to lose the clause two columns
                # before the live ask did.
                question.append("allowed " if allowed else "denied ", style=muted)
        # The RAW tool name, always whole, where the ledger row two lines below
        # shows the shortened `display_name`. The divergence is intended and is
        # not a consistency bug to fix: a security prompt must name exactly what
        # is being authorised (`mcp__linear_create_issue`, server and all), while
        # the ledger optimises for scanning a column of settled actions.
        prefix_cells = question.cell_len
        question.append(self.tool_name, style=fg)

        detail, outside = self._detail()
        target_cells = 0
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
            hazard_style = self._hazard_ink()
            spare = width - question.cell_len - _SEPARATOR_CELLS
            hazard = ""
            # H-11: the glyph already IS the hazard on that rung; repeating it as
            # a clause paints `! write_file  ! ` and spends a separator on a
            # duplicate. The rung's whole premise is that no clause fits.
            if outside and not glyph_hazard:
                if spare - cell_len(HAZARD_WORDS) >= HAZARD_MIN_TARGET:
                    hazard = HAZARD_WORDS
                elif spare >= cell_len(HAZARD_MARKER):
                    hazard = HAZARD_MARKER

            # What is left for the subject once the separator and the hazard are
            # paid for. Computed BEFORE anything is appended, because a separator
            # spent on nothing is what painted a phantom `…`: Rich clipped two
            # trailing spaces and the prompt claimed a truncation that never
            # happened.
            target_verb, target = self._split_target(detail)
            budget = width - question.cell_len - _SEPARATOR_CELLS - cell_len(hazard)
            with_verb = budget - cell_len(target_verb)
            # The verb needs to leave more than the bare minimum, or it is
            # spending its cells to say a category while hiding the instance:
            # `schedule: e…` was byte-identical for a wake firing eight times and
            # one that never stops, because ten cells went to the word `schedule`
            # and one to the thing that differed.
            if target_verb and with_verb >= TARGET_LABELLED_MIN:
                body, body_budget = target_verb, with_verb
            else:
                # The verb is dropped whole rather than fitted: `edit: /etc/…`
                # shortened to `edi…` names nothing, and the target alone still
                # says which file — the tool name two words to the left already
                # says what is being done to it.
                body, body_budget = "", budget

            # A trailing `!` with no target behind it is a one-column island —
            # two cells of separator and a dangling marker — where the fused
            # glyph form says the same thing for less and reads as one row. The
            # marker is dropped here so THAT rung wins the comparison on rank.
            if hazard == HAZARD_MARKER and body_budget < TARGET_MIN_USEFUL:
                hazard = ""
            if hazard or body_budget >= TARGET_MIN_USEFUL:
                question.append("  ", style=dim)
            if hazard:
                question.append(hazard, style=hazard_style)
                hazard_rank = 2 if hazard == HAZARD_WORDS else 1
            if body_budget >= TARGET_MIN_USEFUL:
                if body:
                    question.append(body, style=dim)
                # The subject gets the brightest ink and the end that carries its
                # meaning — a path keeps its tail, a command keeps its head.
                shown = fit_target(target, body_budget, is_path=self._target_is_path(detail))
                question.append(shown, style=fg)
                # The verb prefix counts as detail too. Scoring only the target
                # made the wordier rung win at 34 columns — `allow` survived and
                # bought eight cells of path, where shedding it would have bought
                # seven cells of path AND the `write:` that says what happens to
                # it. The comparison is "how much of the SENTENCE is on screen".
                target_cells = cell_len(body) + cell_len(shown)
        return _Row(question, hazard_rank, target_cells, prefix_cells, spine)

    def _hint_row(self, width: int, *, spine: bool = True) -> Text:
        """The key hints, shedding WHOLE choices rather than truncating one.

        Sheds from the right, which is what makes :data:`CHOICES`' order a
        priority list as well as a reading order: the session-wide switch stops
        being advertised first, then the global stop key, leaving the two per-call
        answers longest. Truncating instead left rows ending `A allow all …` and
        then `A …`, offering a key whose consequence had been cut off.
        """
        indent = " " * ((SPINE_INDENT if spine else 0) + 2)
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

    def _target_is_path(self, detail: str) -> bool:
        """Does this description's target name a FILE?

        Read from the verb the describer chose, not from the string: only the
        tool knows whether its argument is a path, and it already said so by
        picking `write:`/`edit:`/`read:`/`grep:` over `run:`/`browse:`/`schedule:`.
        """
        verb, target = self._split_target(detail)
        if verb.rstrip(": ").strip().lower() not in PATH_VERBS:
            return False
        # The verb alone is not enough: `read` is a filesystem verb AND a browser
        # action, so `read: accounts.google.com/settings` was being tail-truncated
        # into `…s.google.com/settings` — host gone — and `$HOME`-rewritten inside
        # a URL. A resolved path always starts at the root or at `~`; nothing the
        # describers produce for a URL ever does.
        return target.startswith("/") or target.startswith("~")

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

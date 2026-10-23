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
this module is the surface, in TWO widgets with one job each:

- :class:`ApprovalPrompt` is the live question, anchored in the dock above the
  composer. It is where the user answers.
- :class:`ApprovalBlock` is the transcript RECEIPT — what was asked, what was
  answered — so the decision survives in the conversation after the question
  has gone.

The split is a fix, not a refactor. One widget used to do both, mounted in the
transcript, and that made two defects inevitable:

- **The question scrolled away.** A transcript block moves with the transcript,
  so a prompt raised during a busy turn ended up above the fold, and the turn
  was parked on an answer the user could not see they owed.
- **Its keys typed into the composer.** The block took focus to receive `y`/`n`,
  but any click or keystroke in the composer took focus back, and from then on
  the advertised answer keys went into the prompt buffer as text. The user
  pressed `y`, watched a `y` appear in the composer, and the tool went on
  waiting — reported as "the key press is misleading".

Anchored in the dock, the question cannot scroll away, and it presents its
answers as a SELECTABLE LIST (arrows plus Enter, or a click) rather than as
letters the user must aim at a focused widget. The letter keys still work for
anyone who knows them; they are no longer the only way in.

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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from rich.cells import cell_len
from rich.console import Group, RenderableType
from rich.style import Style
from rich.text import Text
from textual.app import SkipAction
from textual.events import Key

from local_operator.ansi import strip_control_sequences
from local_operator.i18n.keys import tui_small as copy
from local_operator.i18n.text import tr
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.ask_picker import AskPickerScreen
from local_operator.tui.widgets.tool_card import truncate_cells
from local_operator.tui.widgets.transcript import SPINE_INDENT, TranscriptBlock

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.harness.types import AskQuestion

#: The slice's catalogued copy, resolved at each render point. This module holds
#: no copy CONSTANT, so it registers no locale refresher (text.py: only
#: constants re-materialise when the active locale moves).
_t = tr("tui.small")

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
#: The OTHER hazard's clause. A target that could not be resolved escalates
#: identically, but it is a different sentence: a path visibly under the
#: workspace root described as "outside the workspace" argues with itself, and a
#: clause the user can see is wrong is one they learn to ignore on the genuine
#: escape. Only this rung differs — at the widths where the clause collapses to
#: `!` the marker makes no location claim and is already honest for both.
HAZARD_WORDS_UNRESOLVABLE = "unresolvable — "
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
#: Row width at or above which the two-cell spine indent is kept. Below it the
#: indent is 12% of everything the row has, and the whole transcript gives it up
#: together — an alignment edge that only some blocks honour is not an edge.
SPINE_FLOOR_WIDTH = 24

#: Marker prefix the builtin tools put on a description whose target sits
#: outside the workspace. Surfaced as its own tinted clause because "outside
#: the workspace" is the single most decision-relevant fact in the prompt.
OUTSIDE_MARKER = "[outside workspace]"
#: The other marker the tools emit: the target could not be resolved, so nothing
#: can be said about where it is. Stripped here like the one above — a bracket
#: sentinel that reached the sentence would be the parser's token leaking into
#: the prose, which is the thing the clause constants exist to prevent.
UNRESOLVABLE_MARKER = "[unresolvable]"

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
            # Fewest cells before the name wins in the two cases where nothing
            # else separates the rungs: when no row fits, and when none of them
            # can show any of the detail. Without the second case the wordier
            # form came BACK at the narrowest widths — `allow` shed at 40 and
            # returning at 20 — because with no target to compare, the rungs tied
            # and the tie fell to the most verbose one.
            0 if (fits and self.target) else -self.prefix,
        )


class ApprovalBlock(TranscriptBlock):
    """One approval decision, as a row in the transcript.

    Since the live question moved into the dock (:class:`ApprovalPrompt`), the
    normal way this block reaches the screen is :meth:`receipt`: already
    answered, recording what was asked and what was decided so the conversation
    keeps the decision after the card is gone.

    It can still be constructed PENDING and resolved in place, and that path is
    kept deliberately rather than deleted: it is how a host that has no dock
    (the reduced test harnesses, and any embedder mounting a transcript on its
    own) can still show and settle an approval. Under the full app nothing takes
    it, because a question that lives in the transcript is a question that
    scrolls away from the turn it is blocking.
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

    @classmethod
    def receipt(cls, tool_name: str, description: str, answer: str) -> "ApprovalBlock":
        """An already-settled block recording a decision the dock card took.

        Built settled rather than constructed-then-resolved so it never spends a
        frame as a pending question: this block is appended to the transcript
        AFTER the answer is in, and a row that painted itself as an unanswered
        prompt first would flash a question the user has already answered.

        Its future is resolved on the spot and nobody awaits it — the engine was
        parked on the dock card's future, which is settled by the time this
        exists. Resolving it anyway keeps the invariant this module is built on
        (every future it creates is resolved exactly once) true by construction
        rather than by argument.
        """
        block = cls(tool_name, description)
        block.resolve(answer in ("y", "a", "A"), answer=answer)
        return block

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
        """Take focus while PENDING, remembering what had it.

        An already-answered block takes nothing: the normal way this widget
        reaches the screen now is :meth:`receipt`, appended after the dock card
        settled, and a receipt that grabbed focus would take the caret out of
        the composer the instant the user answered — measured as the composer
        losing focus immediately after `y`, so the next thing typed went
        nowhere. Only a live question, in a host with no dock, still needs the
        keys.
        """
        if self.answered:
            return
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
        self.refit_width(self.size.width)

    def refit_width(self, width: int) -> None:
        """Rebuild the prompt's row when the lane moved.

        Reached by the container's lane walk
        (:meth:`TranscriptView._refit_authored_blocks`). ``_build`` reads the
        width off this block's own reconciled size — which is already the NEW
        lane in the frame the walk runs in — so this rebuilds exactly as
        ``on_resize`` does, and ``_refresh_row`` is unconditional here for the
        same reason it is there: one row, rebuilt from state.
        """
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
        # then as much of the target as possible.
        # Two lists, not one with an index convention. `shapes` are PEERS that
        # compete on measurement; `floor_shapes` are last resorts consulted only
        # when no peer can carry the hazard at all. Expressing that as a slice of
        # one ordered list made the outcome depend on the order the rungs happen
        # to be written in: reordering did not bring back the tie this replaced,
        # but it silently deleted the marker rung across ~30 columns, turning a
        # row that leads with `?` and carries a distinct `!` into a bare `!`.
        shapes: list[tuple[bool, bool]] = [(False, False)]
        floor_shapes: list[tuple[bool, bool]] = []
        if self._detail()[1]:
            floor_shapes.append((False, True))
        # The spine is decided by the FRAME, not by this row. As a rung it was
        # chosen per block, so four prompts on one screen sat at three different
        # left edges — each having individually traded the transcript's
        # structural edge for one or two more cells of target. That edge is worth
        # more than the cells precisely because it is SHARED, and a rule only one
        # block obeys is not an edge. Below the width where a tool name stops
        # fitting beside it, every block gives it up together.
        spine = width >= SPINE_FLOOR_WIDTH

        # The full wording wins whenever it costs the user NOTHING — it fits and
        # it hides no part of the sentence. Everywhere else the rungs compete on
        # measurement. Deciding it this way rather than by a tiebreak is what
        # makes the grammar flip exactly once as the frame narrows: "shows the
        # whole sentence" can only go from true to false as width shrinks, so
        # `allow` cannot come back after it has been shed. A pure argmax could
        # not promise that — two rungs carrying identical information tie, and
        # whichever way the tie is broken it flips back at some width.
        # The threshold is COMPUTED, not discovered by trial. Asking the verbose
        # rung whether it happens to show everything was not monotone: the hazard
        # clause has its own ladder inside that rung, so at one width the clause
        # still fitted and squeezed the target, and one column narrower the
        # clause collapsed to `!` and the target jumped past the yardstick — the
        # predicate ran false, true, false, and `allow` came back. A width the
        # row needs for its fullest form is a constant, so comparing against it
        # can only flip once.
        if width >= self._verbose_min_width():
            question = self._compose_question(
                width, verb=True, glyph_hazard=False, spine=spine
            ).text
        else:
            # The verbose rung is not a candidate here. Seeding the comparison
            # with it let it win ties below the threshold, which is how `allow`
            # came back at a narrower width after the threshold had already
            # ruled it out — the same defect one level down from the predicate
            # that used to be non-monotone.
            def pick(candidates: list[tuple[bool, bool]]) -> _Row | None:
                chosen: _Row | None = None
                for verb, glyph_hazard in candidates:
                    row = self._compose_question(
                        width, verb=verb, glyph_hazard=glyph_hazard, spine=spine
                    )
                    if chosen is None or row.score(width) > chosen.score(width):
                        chosen = row
                return chosen

            # The glyph rung is the ladder's FLOOR, not a peer. Scored against the
            # marker rung it tied on every term but `target` — and `target` is not
            # monotone in the width, because the clause has its own ladder inside
            # it — so the winner alternated marker, glyph, marker as the frame
            # narrowed and the hazard visibly hopped between two slots. So the
            # peers are settled FIRST and the floor is consulted only if none of
            # them can carry the risk at all, which makes the outcome independent
            # of the order the rungs are declared in.
            best = pick(shapes)
            if best is None or not best.hazard:
                floor = pick(floor_shapes)
                if floor is not None and (best is None or floor.score(width) > best.score(width)):
                    best = floor
            assert best is not None
            question = best.text

        if self._answer is not None:
            # A Group even with one child: rich honours ``Text.no_wrap`` for a
            # Group's children but NOT for a bare Text handed to a Static, so the
            # answered receipt wrapped onto column zero — the composer's own
            # gutter — at narrow widths while the pending two-row form did not.
            return Group(question)
        # The hint row uses the SAME frame-wide spine decision as the question,
        # so the block's two rows always start in the same column.
        return Group(question, self._hint_row(width, spine=spine))

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

    def _verbose_min_width(self) -> int:
        """Row width at which the fullest wording hides nothing.

        Counted rather than measured by rendering: indent, glyph, the verb, the
        tool name, the separator, the hazard clause in its longest form, and the
        whole description. Above this the explicit form is free; below it, no
        arrangement of the explicit form can show everything, so the rungs
        compete on measurement instead.
        """
        detail, outside = self._detail()
        verb_cells = cell_len("allowed ") if self._answer is not None else cell_len("allow ")
        total = SPINE_INDENT + 2 + verb_cells + cell_len(self.tool_name)
        if detail:
            total += _SEPARATOR_CELLS + cell_len(detail)
            if outside:
                # The clause's own floor counts too. `_compose_question` upgrades
                # `!` to the full words only when HAZARD_MIN_TARGET cells of
                # target survive, so a threshold that ignored it sat just BELOW
                # the width where the upgrade happens: the clause grew, the budget
                # shrank past the label's floor, and a wider frame showed less of
                # the sentence across a four-column band.
                #
                # Only the SHORTFALL, because `cell_len(detail)` above already
                # counts the target in full. Adding the floor flat double-counted
                # it and pushed the threshold 12 columns too high, so a hazardous
                # prompt shed `allow` while the explicit form still fitted with
                # cells to spare — the same class of defect one direction over.
                # `outside` IS the clause this row spells out, so the threshold
                # measures the sentence that will actually be painted rather than
                # assuming the longer of the two.
                total += cell_len(outside) + max(0, HAZARD_MIN_TARGET - cell_len(detail))
            target_verb, _target = self._split_target(detail)
            if target_verb:
                # Same reasoning for the `verb:` label, which `_compose_question`
                # keeps only while TARGET_LABELLED_MIN cells of target remain.
                total += max(0, TARGET_LABELLED_MIN - cell_len(_target))
        return total

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
        # Dropping the spine must not buy the TARGET room, only the tool name.
        # Left free, the two reclaimed cells went to the path, so the frame one
        # column narrower than the breakpoint said MORE than the frame above it —
        # the same non-monotonicity the scored ladder replaced a hand-ordered one
        # to prevent, reintroduced by the fix for the ragged left edge.
        budget_width = width - (SPINE_INDENT - indent)
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
                # The catalogued verb, trailing space included: the row reads
                # `allow <target>` once the caller appends the target.
                question.append(_t(copy.approval_allow_verb()), style=muted)
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
                question.append(
                    _t(
                        copy.approval_decision_allowed()
                        if allowed
                        else copy.approval_decision_denied()
                    ),
                    style=muted,
                )
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
            spare = budget_width - question.cell_len - _SEPARATOR_CELLS
            hazard = ""
            # H-11: the glyph already IS the hazard on that rung; repeating it as
            # a clause paints `! write_file  ! ` and spends a separator on a
            # duplicate. The rung's whole premise is that no clause fits.
            if outside and not glyph_hazard:
                # `outside` carries the WORDS; the marker and the glyph are shared
                # by both hazards because at these widths the row makes no
                # location claim and is already honest for either reason.
                if spare - cell_len(outside) >= HAZARD_MIN_TARGET:
                    hazard = outside
                elif spare >= cell_len(HAZARD_MARKER):
                    hazard = HAZARD_MARKER

            # What is left for the subject once the separator and the hazard are
            # paid for. Computed BEFORE anything is appended, because a separator
            # spent on nothing is what painted a phantom `…`: Rich clipped two
            # trailing spaces and the prompt claimed a truncation that never
            # happened.
            target_verb, target = self._split_target(detail)
            budget = budget_width - question.cell_len - _SEPARATOR_CELLS - cell_len(hazard)
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
                # Rank 2 is "spelled out in words", whichever clause it was.
                # Currently INERT — measured, not assumed: `shapes` holds one peer
                # rung and the floor is only consulted when no peer carries the
                # hazard, so nothing compares two hazardous rows and both spellings
                # paint identically at every width 90-20. Written this way anyway
                # because the old form tested against ONE clause by name, so a
                # second words-clause scored as a bare marker; that is a trap for
                # the next rung, not a bug today, and it is not worth a test that
                # would assert nothing.
                hazard_rank = 1 if hazard == HAZARD_MARKER else 2
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

    def _detail(self) -> tuple[str, str]:
        """``(description, hazard_clause)`` with the marker lifted out.

        The second element is the WORDS this hazard spells out, or `""` for no
        hazard — so it stays truthy exactly where the old boolean was, and the
        rungs that spell the clause out get the right sentence without having to
        know which markers exist.
        """
        text = " ".join(self.description.split())
        if text.startswith(OUTSIDE_MARKER):
            return text[len(OUTSIDE_MARKER) :].strip(), HAZARD_WORDS
        if text.startswith(UNRESOLVABLE_MARKER):
            return text[len(UNRESOLVABLE_MARKER) :].strip(), HAZARD_WORDS_UNRESOLVABLE
        return text, ""

    def spans_multiple_rows(self) -> bool:
        # Two rows while pending (question + hints), one once answered. Answered
        # from state so the spacing rule never renders the block to find out.
        return self._answer is None


#: The three answers the gate can take, in the order the card lists them.
#:
#: Ordered allow, deny, allow-all: the reverse of the hint row's refuse-first
#: order, and deliberately so. That row sheds choices from the right under
#: width pressure, so its order is a PRIORITY ranking; this is a LIST the user
#: reads top to bottom, where the ordinary answer belongs first and the
#: session-wide one last. "Allow all" sits at the bottom because it is the only
#: row here that outlives the call: a list that put it under the cursor would
#: make disarming the gate for the whole session the thing Enter does by
#: reflex.
#:
#: Each row carries the CONSEQUENCE as its description, because that is what
#: the user is actually deciding between, and the letter it answers to, so the
#: keys stay discoverable now that they are no longer the only way in.
#: The Allow consequence under a gate that will prompt again. Also the string
#: :meth:`ApprovalPrompt._row_description` swaps for the auto wording below.
ALLOW_ASK_CONSEQUENCE = "run this call and ask again next time"
#: The same consequence once the gate is auto: telling the reader to expect a
#: prompt that will not come is the stale-copy defect this pair removes.
ALLOW_AUTO_CONSEQUENCE = "run this call; later calls will not ask"

APPROVAL_CHOICES: tuple[tuple[str, str, str], ...] = (
    ("Allow", "y", ALLOW_ASK_CONSEQUENCE),
    ("Deny", "n", "refuse this call; the turn continues"),
    ("Allow all", "A", "stop asking for this session"),
)

#: Answer key -> the label the picker hands back, so the two ways of answering
#: (a letter, or a selected row) converge on ONE value before anything acts on
#: it. Built from the table above rather than repeated, because a letter that
#: disagreed with its row would be a gate that approves what the user refused.
APPROVAL_KEY_BY_LABEL: dict[str, str] = {label: key for label, key, _ in APPROVAL_CHOICES}

#: The KEY the user presses for allow-all, and the ANSWER that decision is
#: recorded as. They differ on purpose and the difference has bitten once.
#:
#: The key is capital ``A`` so that a user typing an ordinary sentence into a
#: focused card cannot disarm the session's approval gate with the most common
#: letter in English. The answer is lowercase ``a`` because that is the token
#: the app's latch (:meth:`OperatorApp._latch_approval_answer`) and the
#: receipt's own wording already match on. Reported as ``A``, the latch simply
#: did not fire: "allow all" answered the one call and the very next tool asked
#: again, which is the opposite of what the row promises.
ALLOW_ALL_KEY = "A"
ALLOW_ALL_ANSWER = "a"


def approval_question(tool_name: str, description: str) -> "AskQuestion":
    """The approval gate as a question the anchored picker can put to the user.

    Reusing the ``ask`` picker rather than growing a second list widget is what
    makes the two prompts behave identically: one card, one set of movement
    keys, one hit-test, one row budget that keeps the conversation on screen.
    Both surfaces are the same interaction — the agent needs an answer before it
    can go on — and the app had two different answers to how that should look.

    No free-text row (``allow_free_text=False`` where this is mounted): the gate
    returns a boolean plus a scope, so a typed answer would be an invitation to
    write prose that nothing can read.
    """
    from local_operator.harness.types import AskOption, AskQuestion

    detail = " ".join(strip_control_sequences(description).split())
    for marker in (OUTSIDE_MARKER, UNRESOLVABLE_MARKER):
        if detail.startswith(marker):
            # The hazard is stated in the QUESTION, where it belongs, rather
            # than left as the parser's bracket token in front of the target.
            words = HAZARD_WORDS if marker == OUTSIDE_MARKER else HAZARD_WORDS_UNRESOLVABLE
            detail = f"{words}{detail[len(marker):].strip()}"
            break
    question = f"Allow {strip_control_sequences(tool_name)}? {detail}".rstrip()
    return AskQuestion(
        id="approval",
        question=question,
        options=[
            AskOption(label=label, description=consequence)
            for label, _key, consequence in APPROVAL_CHOICES
        ],
    )


#: What the card says about the approval mode in force, and where it came from.
#:
#: The card is the one place a user decides whether to loosen the gate, so it
#: has to state the CURRENT rule as well as offer the two destinations — a
#: destination with no statement of where the user stands is a switch they
#: cannot read before pressing it. ``provenance`` is ``"global"`` (the file),
#: ``"session"`` (this conversation's own recorded choice) or ``"pin"`` (a
#: ``--yolo``/``--no-yolo`` flag on this run).
#:
#: ``saving`` and ``status`` share ONE rendering slot: the callout's state row
#: is reserved from mount through the saved and refused states, so a save
#: acknowledgement replaces a sentence instead of adding a row (the row budget
#: is what keeps the conversation on screen — see ApprovalPrompt.set_note).
@dataclass(frozen=True)
class ApprovalPolicy:
    mode: str = "ask"
    provenance: str = "global"
    saving: str = ""
    status: str = ""
    detail: str = ""


#: The result of one setting gesture, returned to the card so it can acknowledge
#: INLINE. ``kind`` separates the three outcomes the copy distinguishes:
#: ``ok`` (the mode moved and was recorded), ``refused`` (this window holds no
#: authority to loosen — nothing was written) and ``error`` (the write failed;
#: the mode may or may not have moved). The reason is already safe to paint: it
#: names a failure class, never a credential or a stack.
@dataclass(frozen=True)
class ApprovalSettingOutcome:
    kind: str = "ok"
    mode: str = "ask"
    provenance: str = "global"
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.kind == "ok"


#: The two secondary destinations, as data: the key that reaches each one and
#: the scope word the app layer receives. Held here rather than spelled inline
#: in the key handler and the renderer, so the letter printed beside a target
#: and the callback it invokes cannot disagree.
SETTING_KEYS: dict[str, str] = {"S": "session", "G": "global"}

#: The mode words as a human reads them. ``auto`` in the command vocabulary is
#: "Auto" in prose; the same capitalisation the band and the receipts use.
_MODE_WORD = {"ask": "Ask", "auto": "Auto"}


class ApprovalPrompt(AskPickerScreen):
    """The live approval question, anchored in the dock above the composer.

    Subclasses the ``ask`` picker so the two prompts are the same surface: the
    same row budget that keeps the conversation on screen, the same movement
    keys, the same hit-test, the same focus discipline. What it adds is the
    approval gate's own contract:

    - it answers an :class:`asyncio.Future` with a BOOLEAN, not a label list;
    - the answer letters (``y``/``n``/``A``) still work, because muscle memory
      exists and the hint row has advertised them for as long as this prompt
      has;
    - Escape DENIES rather than skips. There is no such thing as an unanswered
      approval: the tool is either allowed to run or it is not, and a question
      that quietly went away would leave the engine parked forever.

    The transcript keeps a separate receipt (:class:`ApprovalBlock`), so the
    decision survives in the conversation after this card is gone.
    """

    #: Kept in sync with the letters offered in the footer. A letter that
    #: answered here without appearing there would be undiscoverable; one that
    #: appeared there without answering here is the defect this rework exists
    #: to remove.
    _ANSWER_KEYS = frozenset(key for _label, key, _why in APPROVAL_CHOICES)

    #: This card's REMOTE hint speaks the approval vocabulary: an allow from
    #: the origin viewer is refused by the owner's runtime by design (the
    #: origin holds no key a peer can verify), a deny is ordinary. The ask
    #: card's hint says the different truth its own gate has (see
    #: ``AskPickerScreen.set_remote_device``).
    REMOTE_HINT_KIND = "approval"

    #: Escape must reach the APP, which stops the run, rather than answering
    #: this one question.
    #:
    #: On the ``ask`` picker Escape means "leave this question unanswered",
    #: which is a real outcome the tool reports. On an approval there is no such
    #: outcome — the tool either runs or it does not — and Escape means STOP
    #: everywhere else in this app. Answering here instead cost the user one
    #: press per queued prompt before the run actually stopped, with each press
    #: looking like it had done nothing (the defect the transcript prompt
    #: recorded in its own BINDINGS note).
    #:
    #: Filtering the inherited ``BINDINGS`` list does NOT achieve this: Textual
    #: merges bindings up the MRO (``_merged_bindings``), so the parent's
    #: ``escape`` survives whatever this class's own list says — measured, with
    #: the card still answering ``n`` and ``action_stop`` never running. The
    #: binding is left in place and its ACTION is refused instead, so the key
    #: bubbles to the app, which denies every queued prompt and aborts the turn.
    #: The footer still says `esc deny`, because denial is what that stop does
    #: to this particular call.
    def action_cancel(self) -> None:
        """Refuse the key so it reaches the app's stop. See the note above."""
        raise SkipAction()

    def _labels_must_all_fit(self) -> bool:
        """Every consequence's LABEL stays on screen; descriptions yield first.

        The authorisation contract, and the one place this gate diverges from
        the ``ask`` picker's OMP-style windowing. On the ask card a long list
        SHOULD scroll its labels with descriptions kept (the coexist headline,
        ask_picker design §0). On an approval gate a user who cannot SEE that
        *Allow all* exists cannot weigh it: hiding an answer's label behind a
        scroll is the same C3/D1 defect as a card that never names what it is
        asking — worse, because it looks answerable with a permissive option in
        view. So when the full 2-line-clamp list overflows a tight budget (the
        BOOT layout, dock 8, where an approval can be the first thing a session
        shows), the allocator drops the consequence lines to keep every label
        visible rather than windowing one off, and windows only if even the
        labels alone do not fit. The three consequences are each one line at
        every width down to 44 columns, so in practice the gate drops to bare
        labels exactly as it did before the line-granular rework — the frame the
        prior three rounds signed off on.
        """
        return True

    def __init__(
        self,
        tool_name: str,
        description: str,
        on_answer: Callable[[str], None] | None = None,
        *,
        policy: ApprovalPolicy | None = None,
        on_set_session_auto: Callable[[], ApprovalSettingOutcome | None] | None = None,
        on_set_global_auto: Callable[[], ApprovalSettingOutcome | None] | None = None,
    ) -> None:
        super().__init__(
            [approval_question(tool_name, description)],
            self._answered,
            allow_free_text=False,
            # A UNIQUE id per prompt, not the bare "approval-prompt".
            #
            # Approvals are serialized, but the handover is not instantaneous:
            # a queued asker wakes the moment the front prompt's future
            # resolves, which is BEFORE the awaiting frame reaches its `finally`
            # and unmounts that prompt. For those few pump hops both widgets are
            # attached, and a fixed id makes Textual refuse the mount with
            # `DuplicateIds` — observed as two concurrent `bash` approvals
            # crashing the mount of the second.
            widget_id=f"approval-prompt-{id(self):x}",
            title="the agent needs your approval",
            # Not "skip": escaping an approval denies the call and stops the
            # turn. A footer that offered to skip a gate would be describing a
            # third outcome that does not exist.
            exit_hint=("esc", "deny"),
        )
        self.tool_name = strip_control_sequences(tool_name)
        self.description = strip_control_sequences(description)
        self._answer: str | None = None
        # `get_running_loop`, not `get_event_loop`: the future must belong to
        # the loop that will await it, and stating that precondition turns a
        # future built from a sync context into an immediate error rather than
        # a future nobody resolves. (3.14 removes the implicit-loop fallback.)
        self._future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._on_answer = on_answer
        self._policy = policy or ApprovalPolicy()
        self._on_set_session_auto = on_set_session_auto
        self._on_set_global_auto = on_set_global_auto
        #: Painted spans, in BODY-LINE coordinates: ``(row, start, end, scope)``
        #: for the callout, and ``(row, start, end, answer_index)`` for the
        #: compact rail. Both are re-recorded on every paint from the same plan
        #: that drew them; a stale span would make a target activate something
        #: other than what it shows.
        self._callout_spans: list[tuple[int, int, int, str]] = []
        self._rail_spans: list[tuple[int, int, int, int]] = []
        #: Where the injected chrome starts and how many rows it owns. Read by
        #: the hit-test so a click on it never falls through to the base list's
        #: row map (which only covers the lines the BASE drew).
        self._chrome_start = 0
        self._chrome_rows = 0
        #: Whether the dense padding is currently applied. Assigning padding is
        #: a layout invalidation, so the resize handler compares first.
        self._dense_padded: bool | None = None

    # -- the mode callout ----------------------------------------------------
    #
    # WHY THIS IS ON THE CARD AND NOT A POPUP: the two handles only make sense
    # beside the decision they govern. A user deciding whether to allow a call
    # is the one moment they can judge "asking me every time is noise" or "I
    # want this session gated"; another screen would lose the context that
    # motivated the choice. They are NOT answer rows — an answer settles this
    # request, a setting changes the rule for the NEXT one — so they are painted
    # below the choices, with their own callbacks, and never touch the future.

    #: Full-width copy thresholds. The handle forms are whole units rather than
    #: truncated fragments: "sess…" teaches nothing, and a destination the
    #: reader cannot name is one they will not press.
    _HANDLES_FULL_MIN = 63
    _HANDLES_MEDIUM_MIN = 53
    _HANDLES_SHORT_MIN = 40
    #: The combined row's roomier form is 55 cells at its longest (a locked
    #: `Auto(session):locked` prefix), so it needs that much content width.
    _COMBINED_FULL_MIN = 55
    #: The dense card: the measured 24x16 geometry, where the content width is
    #: 20 cells. This is the ONLY size at which the card reclaims its vertical
    #: padding — the spec's single padding exception, derived from measured
    #: frames rather than promised for every crowded dock.
    _DENSE_WIDTH = 20

    def set_policy(self, policy: ApprovalPolicy) -> None:
        """Replace the effective state the callout paints.

        Used when the app learns the mode moved for a reason the card did not
        initiate (a config write, an owner's acknowledgement). It repaints an
        existing statement instead of appending a transcript notice, which is
        what keeps one event to one sentence.
        """
        self._policy = policy
        # The Allow consequence is policy-dependent (`run this call; later
        # calls will not ask`), and the per-row wrap cache keys only on
        # (index, width) — a mode move that kept the cache would repaint the
        # new status above the old consequence.
        self._invalidate_description_wraps()
        self._repaint()

    def show_setting_outcome(self, scope: str, outcome: ApprovalSettingOutcome) -> None:
        """Acknowledge a setting gesture in the callout's reserved slot.

        ``scope`` is the destination the USER pressed, not the scope the app
        settled on: a global write that moved this conversation from its own
        override clears the override, and the acknowledgement should still read
        as the gesture the reader made ("global default"). The effective state
        painted beside it comes from the outcome, never from the gesture.

        A reply that arrives after the card was ANSWERED is discarded: the card
        is a receipt by then, and the spec explicitly forbids resurrecting it.
        """
        if self._answer is not None:
            return
        if outcome.kind == "error":
            status = "error"
        elif outcome.kind == "refused":
            status = "refused"
        elif outcome.kind == "applied-unsaved":
            # The live mode moved but the record did not. Neither "saved" nor
            # "unchanged" is true, so it gets its own sentence.
            status = "applied-unsaved"
        else:
            status = "saved-session" if scope == "session" else "saved-global"
        self._policy = ApprovalPolicy(
            mode=outcome.mode,
            provenance=outcome.provenance,
            status=status,
            detail=outcome.reason,
        )
        self._invalidate_description_wraps()
        self._repaint()

    def _request_setting(self, scope: str) -> None:
        """Run one destination's callback, then paint its outcome.

        Deliberately NOT through :meth:`_answer_with`: the base's resolve path
        completes the future, and a setting change must leave this request
        parked (the operator's brief: a mode change must NEVER answer a pending
        prompt). The in-flight guard is the widget's own because the callbacks
        can write a file; a second press while the first is running would be
        two writes of one gesture.
        """
        if self._answer is not None or self._policy.saving:
            return
        callback = self._on_set_session_auto if scope == "session" else self._on_set_global_auto
        if callback is None:
            return
        self._policy = replace(self._policy, saving=scope, status="", detail="")
        self._repaint()
        try:
            outcome = callback()
        except Exception as error:  # noqa: BLE001 — a write must not take the card down
            outcome = ApprovalSettingOutcome(
                kind="error",
                mode=self._policy.mode,
                provenance=self._policy.provenance,
                reason=error.__class__.__name__,
            )
        if outcome is None:
            # Routed to the process that owns the gate. The card keeps its
            # "Saving…" state until that process acknowledges; optimistic copy
            # here would claim a mode nobody has confirmed.
            return
        self.show_setting_outcome(scope, outcome)

    # -- layout and paint ----------------------------------------------------

    def _dense(self, width: int) -> bool:
        """The measured 24x16 geometry, where the card reclaims its padding."""
        return width <= self._DENSE_WIDTH

    def _body_rows(self, question_lines: int) -> int:
        """Reclaim the two vertical padding rows at the dense geometry.

        The spec's ONE padding exception: at 24x16 the card's content region is
        three rows with one padding row above and below, and the five things
        that must be visible there (question, two rail rows, two callout rows)
        need five. Without this the card would have to shed a destination or an
        answer label, which the spec forbids. Bounded to the measured shape: a
        shorter dock keeps the base budget and sheds by the ordinary order
        rather than overdrawing a region it cannot paint.
        """
        budget = super()._body_rows(question_lines)
        if self._dense(self._card_width()) and 3 <= budget < 5:
            # No terminal-height guard here: at a 16-row terminal the SCREEN
            # CONTENT box is 14, so a `height >= 16` test never engaged the
            # dense plan the spec derives for that exact frame — the rail then
            # painted 30 cells into 20 and `A.Allow all` clipped to a bare `A`
            # that still answered (agent/QA/design round 1, M2/Q2/D1). The
            # budget already encodes the physical room; removing the two
            # padding rows is what makes the two extra rows affordable.
            budget = 5
        # Remembered for `_layout`: the plan has to know how many rows it may
        # spend before it decides whether this frame can afford the callout at
        # all. `_layout` is the only caller, so this is a parameter passed
        # through an attribute rather than a second, drifting calculation.
        self._last_body_budget = budget
        return budget

    def _apply_dense_padding(self) -> None:
        """Give the dense card back the two rows its slot spends on padding.

        Only at :attr:`_DENSE_WIDTH`. Assigned here rather than in the sheet
        because the exception is a fact about this card at one width, not about
        every ask picker at that width; compared before assigning because each
        assignment invalidates layout and the resize handler runs on every
        frame the dock moves.
        """
        if not self.size.width:
            # Not laid out yet: `_card_width` would answer from a fallback and
            # could flash the unpadded shape before the first real resize.
            return
        dense = self._dense(self._card_width())
        if dense == self._dense_padded:
            return
        self._dense_padded = dense
        self.styles.padding = (0, 1) if dense else (1, 1)

    def on_mount(self) -> None:
        self._apply_dense_padding()
        super().on_mount()

    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        self._apply_dense_padding()
        super().on_resize(event)

    def _layout(self, *, reveal: bool | None = None):  # type: ignore[no-untyped-def]
        """The base plan, with the callout and compact rail bought from it.

        The allocator's shape is reused rather than re-derived: the callout
        spends exactly the two blank spacer rows when the frame can afford both
        (so a roomy frame's total height does not move), and the compact rail
        replaces the option viewport and the footer when the frame cannot show
        all three labels as rows. One plan feeds the painter and the hit-test,
        which is the spec's constraint 6.
        """
        plan = super()._layout(reveal=reveal)
        width = plan.width
        budget = getattr(self, "_last_body_budget", plan.body_line_budget + 3)
        dense = self._dense(width) and budget >= 5
        base_compact = plan.body_line_budget < self.row_count
        two_row = bool(plan.space_above and plan.space_below)
        spacers = int(plan.space_above) + int(plan.space_below)
        callout_rows = 2 if two_row else 1
        compact = dense or base_compact
        if not compact and plan.body_line_budget + spacers - callout_rows < self.row_count:
            # Spending the callout's rows would push an answer LABEL out of the
            # viewport. The compact rail shows all three in one row, so it is
            # the honest rendering rather than a card that answers with a choice
            # the reader cannot see.
            compact = True
        if compact:
            rail_rows = 2 if dense else 1
            if budget < 1 + rail_rows + 1:
                # The frame cannot pay for the question, the rail and the
                # callout TOGETHER. Falling back to the base plan keeps the
                # card's own collapse rule (question alone, or nothing at all),
                # and `_shape` stays non-compact so `y`/`A` are refused exactly
                # as they were before this feature: an affirmative answer must
                # never come from a frame that shows no answer label (D9). The
                # alternative — painting three rows into a one-row budget — is
                # how the card would both overdraw and authorise from a clipped
                # frame.
                self._chrome_off = True
                self._shape = (False, False, False)
                return plan
            self._chrome_off = False
            self._shape = (dense, True, dense or two_row)
            # The rail takes the option row's place and the callout takes the
            # footer's: both are rows the base already paid for, plus (dense)
            # the two padding rows reclaimed in `_body_rows`.
            return replace(
                plan,
                show_title=False,
                space_above=False,
                space_below=False,
                show_position=False,
                show_footer=False,
                body_line_budget=0,
                description_rows={},
                show_descriptions=False,
            )
        self._chrome_off = False
        self._shape = (False, False, two_row)
        return replace(
            plan,
            space_above=False,
            space_below=False,
            body_line_budget=max(0, plan.body_line_budget + spacers - callout_rows),
        )

    # -- callout copy --------------------------------------------------------

    @staticmethod
    def _scope_word(provenance: str) -> str:
        return {"session": "session", "pin": "run"}.get(provenance, "global")

    def _compact_prefix(self) -> str:
        word = _MODE_WORD.get(self._policy.mode, "Ask")
        return f"{word}({self._scope_word(self._policy.provenance)})"

    def _state_text(self, width: int) -> str:
        """The callout's first row: what is in force, or how a save went.

        Full/short/compact by CONTENT width, not terminal columns — the callout
        lives inside the card's padding, and a form chosen from the outer width
        is two cells too optimistic. The compact form is a prefix because it has
        to share a row with the handles at the tight sizes.
        """
        policy = self._policy
        if policy.saving:
            what = "session" if policy.saving == "session" else "global"
            if width < 40:
                return "Saving…"
            return f"Saving {what} auto…"
        if policy.status == "saved-session":
            # The MODE word rather than a fixed "Auto": the same slot carries
            # the ask direction, and a sentence that called a hardened session
            # "Auto saved" would be the exact class of false statement this
            # callout exists to remove.
            word = _MODE_WORD.get(policy.mode, "Auto")
            if width >= 63:
                return (
                    f"{word} saved for this session, with this conversation. "
                    "Answer this request."
                )
            if width >= 40:
                return f"{word} saved for this session. Answer this request."
            return f"{word}(session) saved"
        if policy.status == "saved-global":
            word = _MODE_WORD.get(policy.mode, "Auto")
            if width >= 63:
                return (
                    f"{word} set as global default; this session inherits it. "
                    "Answer this request."
                )
            if width >= 40:
                return f"{word} saved as global default. Answer this request."
            return f"{word}(global) saved"
        if policy.status == "refused":
            mode = _MODE_WORD.get(policy.mode, "Ask")
            scope = self._scope_word(policy.provenance)
            if width < 40:
                return f"{mode}({scope}):locked"
            if width < 63:
                return f"This window cannot authorise auto. {mode} ({scope})."
            where = "session override" if scope == "session" else "global default"
            return f"Unchanged: {mode} ({where}). This window cannot authorise auto."
        if policy.status == "applied-unsaved":
            word = _MODE_WORD.get(policy.mode, "Auto")
            if width < 40:
                # 20-cell frame: the mode plus the failure, no room for a verb.
                return f"{word}, not saved"
            if width < 63:
                return f"{word} applied; NOT saved for resume."
            reason = policy.detail or "storage error"
            return f"{word} applied — NOT saved for resume ({reason}). " "Answer this request."
        if policy.status == "error":
            if width < 40:
                return "Not saved"
            reason = policy.detail or "storage error"
            return f"Not saved: {reason}. Approvals unchanged."
        # The plain effective state, by provenance. The AUTO/global case is the
        # one that has to say the request still needs an answer: the gate is
        # loose, the card is not a stale artefact.
        mode = _MODE_WORD.get(policy.mode, "Ask")
        if policy.provenance == "session":
            other = "Auto" if policy.mode == "ask" else "Ask"
            if width >= 63:
                return f"Approvals: {mode} — this session overrides global {other}."
            if width >= 40:
                return f"{mode} — session override; global {other}."
            return self._compact_prefix()
        if policy.provenance == "pin":
            if width >= 63:
                return f"Approvals: {mode} — fixed for this run."
            if width >= 40:
                return f"{mode} — fixed for this run."
            return self._compact_prefix()
        if policy.mode == "auto":
            if width >= 63:
                return "Approvals: Auto — global default; answer this pending request."
            if width >= 40:
                return "Auto — global default; this request needs an answer."
            return self._compact_prefix()
        if width >= 63:
            return "Approvals: Ask — global default."
        if width >= 40:
            return "Ask — global default."
        return self._compact_prefix()

    def _handles(self, width: int) -> tuple[list[tuple[str, str]], list[tuple[int, int, str]]]:
        """The two destinations as whole forms, with their clickable spans.

        Returned as SEGMENTS rather than one string: only the scope phrases are
        interactive, so only they may carry the underline and the fg ink. The
        separator is descriptive punctuation and stays muted; a row underlined
        end to end paints the punctuation as a target and makes one link out of
        two destinations.

        Spans cover the visible scope phrase INCLUDING its key hint, and nothing
        else: a click on the separator or the gap must not activate a
        destination (the spec's "no overlapping padding" rule).
        """
        if width >= self._HANDLES_FULL_MIN:
            parts = [
                ("Set auto for this session [S]", "handle"),
                ("  ·  ", "state"),
                ("Set auto for all sessions [G]", "handle"),
            ]
        elif width >= self._HANDLES_MEDIUM_MIN:
            parts = [
                ("[S] Auto for this session", "handle"),
                (" · ", "state"),
                ("[G] Auto for all sessions", "handle"),
            ]
        elif width >= self._HANDLES_SHORT_MIN:
            parts = [
                ("[S] Session auto", "handle"),
                (" · ", "state"),
                ("[G] All-session auto", "handle"),
            ]
        else:
            parts = [
                ("auto ", "state"),
                ("S:session", "handle"),
                (" ", "state"),
                ("G:all", "handle"),
            ]
        spans: list[tuple[int, int, str]] = []
        spent = 0
        for text, kind in parts:
            if kind == "handle":
                # `[S]`/`S:session` are the SESSION forms; "sessions" also
                # contains "session", so a substring test on the bare word
                # mislabels the global destination.
                scope = "session" if ("[S]" in text or text == "S:session") else "global"
                spans.append((spent, spent + cell_len(text), scope))
            spent += cell_len(text)
        return parts, spans

    def _styled(self, parts: tuple[tuple[str, str], ...], *, overflow: bool = True) -> Text:
        """One row from ``(text, kind)`` parts; ``kind`` is state or handle."""
        row = Text(no_wrap=True, overflow="ellipsis" if overflow else "fold")
        state = Style(color=theme_mod.semantic_color("muted"))
        handle = Style(color=theme_mod.semantic_color("fg"), underline=True)
        if self._policy.status == "refused":
            # A refused destination keeps its text and underline (it stays an
            # identifiable target that repeats the explanation) but loses the
            # fg ink: the frame must not offer a loosening it cannot perform.
            # `muted`, not `dim`: dim measures 2.72:1 light / 3.43:1 dark on
            # this ground, below the 4.5 floor, and the refusal is exactly the
            # sentence the reader must be able to read (design round 1, D4).
            handle = Style(color=theme_mod.semantic_color("muted"), underline=True)
        for text, kind in parts:
            row.append(text, style=state if kind == "state" else handle)
        return row

    def _callout_rows(
        self, width: int, *, two_row: bool
    ) -> tuple[list[Text], list[tuple[int, int, int, str]]]:
        """The callout as painted rows plus their hit spans."""
        state = self._state_text(width)
        if two_row:
            parts, spans = self._handles(width)
            return (
                [self._styled(((state, "state"),)), self._styled(tuple(parts))],
                [(1, start, end, scope) for start, end, scope in spans],
            )
        # The combined row uses the COMPACT state in every form: it is the only
        # one that fits beside two destinations at these sizes, and the compact
        # vocabulary is already a complete statement (`Auto(session) saved`).
        prefix = self._state_text(min(width, self._HANDLES_SHORT_MIN - 1))
        if width >= self._COMBINED_FULL_MIN:
            parts = [
                (f"{prefix} · auto: ", "state"),
                ("[S]session", "handle"),
                (" ", "state"),
                ("[G]all sessions", "handle"),
            ]
        else:
            parts = [
                (f"{prefix} auto ", "state"),
                ("S:session", "handle"),
                (" ", "state"),
                ("G:all", "handle"),
            ]
        spans = []
        spent = 0
        for text, kind in parts:
            if kind == "handle":
                scope = "session" if ("[S]" in text or text == "S:session") else "global"
                spans.append((spent, spent + cell_len(text), scope))
            spent += cell_len(text)
        if spent > width:
            # The locked-state prefix is the longest case and the one the spec
            # budgets a second row for; a destination is never clipped off to
            # keep one row.
            handle_parts, handle_spans = self._handles(width)
            return (
                [self._styled(((state, "state"),)), self._styled(tuple(handle_parts))],
                [(1, start, end, scope) for start, end, scope in handle_spans],
            )
        return [self._styled(tuple(parts))], [(0, start, end, scope) for start, end, scope in spans]

    def _row_description(self, index: int) -> str:
        """The description under one option, in the mode that is actually in force.

        A parked card can outlive the mode it was raised under: pressing `S`/`G`
        makes the NEXT gate auto while this one still asks, and leaving "ask
        again next time" under Allow states the opposite of the saved policy the
        callout above just announced (UX/design round 1, U3/D3). Only the Allow
        row changes; its replacement is shorter at every authored width, so no
        row moves and no budget changes.
        """
        text = super()._row_description(index)
        if self._policy.mode == "auto" and text == ALLOW_ASK_CONSEQUENCE:
            return ALLOW_AUTO_CONSEQUENCE
        return text

    def _rail_rows(
        self, width: int, *, two_row_rail: bool
    ) -> tuple[list[Text], list[tuple[int, int, int, int]]]:
        """The compact answer rail: every label on the choice row.

        This is a compact RENDERING of the existing list, not a second
        interaction: the order is unchanged, the selected row keeps its accent
        and caret, and a click maps to the exact label span. The alternative —
        the ordinary viewport at a 1-row budget — showed only the selected
        label, which hid the existence of Deny and Allow all at the sizes where
        the card is most constrained.
        """
        # The ACCENT stays on the caret only: as a LABEL colour it measures
        # 3.49:1 in the light theme (below 4.5), while fg+bold keeps the
        # selected choice unmistakable and legible (design round 1, D2).
        accent = Style(color=theme_mod.semantic_color("accent"))
        ordinary = Style(color=theme_mod.semantic_color("fg"))
        selected_ink = Style(color=theme_mod.semantic_color("fg"), bold=True)
        items: list[Text] = []
        for index in range(self.row_count):
            label = f"{self.row_key(index)}.{self.question.options[index].label}"
            selected = index == self.state.selected and not self.passive and not self._read_only
            item = Text(no_wrap=True, overflow="ellipsis")
            if selected:
                item.append("❯ ", style=accent)
                item.append(label, style=selected_ink)
            else:
                item.append(label, style=ordinary)
            items.append(item)

        def spans_for(item: Text, start: int, index: int) -> tuple[int, int, int, int]:
            # The caret is not part of the target: clicking the space before a
            # label should not answer the request.
            offset = 2 if item.plain.startswith("❯ ") else 0
            return (0, start + offset, start + cell_len(item.plain), index)

        # The two-row form exists for the three approval choices; guard the
        # indexing so a future question with fewer options falls back to the
        # one-row rail instead of raising IndexError on a resize.
        if not two_row_rail or len(items) < 3:
            row = Text(no_wrap=True, overflow="ellipsis")
            spans: list[tuple[int, int, int, int]] = []
            for position, item in enumerate(items):
                if position:
                    row.append("  ")
                start = cell_len(row.plain)
                row.append_text(item)
                spans.append(spans_for(item, start, position))
            return [row], spans
        first = Text(no_wrap=True, overflow="ellipsis")
        spans = []
        start = 0
        first.append_text(items[0])
        spans.append(spans_for(items[0], 0, 0))
        first.append("  ")
        start = cell_len(first.plain)
        first.append_text(items[1])
        spans.append(spans_for(items[1], start, 1))
        second = Text(no_wrap=True, overflow="ellipsis")
        second.append("  ")
        second.append_text(items[2])
        spans.append((1,) + spans_for(items[2], 2, 2)[1:])
        return [first, second], spans

    def _injected_rows(
        self, width: int
    ) -> tuple[list[Text], list[tuple[int, int, int, str]], list[tuple[int, int, int, int]]]:
        if getattr(self, "_chrome_off", False):
            # The frame could not pay for the compact card; the base plan's
            # collapse is the whole frame (see `_layout`).
            return [], [], []
        dense, compact, two_row = getattr(self, "_shape", (False, False, False))
        callout: list[Text]
        callout_spans: list[tuple[int, int, int, str]]
        if compact:
            rail, rail_spans = self._rail_rows(width, two_row_rail=dense)
            callout, callout_spans = self._callout_rows(width, two_row=dense)
            # The rail's rows precede the callout; the caller's spans are
            # shifted by that count when the whole block is spliced in.
            return rail + callout, callout_spans, rail_spans
        callout, callout_spans = self._callout_rows(width, two_row=two_row)
        return callout, callout_spans, []

    def _card_text(self) -> Text:
        """The base card, with the mode callout spliced into its own rows.

        Spliced rather than re-composed: the header, question, option viewport
        and footer are the base's, and a second full renderer here would be a
        second implementation of the clipping rules this card is most sensitive
        to. The injected rows come from ONE plan (:meth:`_layout`), and the
        line map is repaired at the same offset so a click on the callout can
        never fall through to an option row that is no longer on that line.
        """
        base = super()._card_text()
        layout = self._layout()
        width = layout.width
        injected, callout_spans, rail_spans = self._injected_rows(width)
        lines = list(base.split("\n"))
        insert_at = len(lines) - (1 if layout.show_footer else 0)
        if not injected:
            self._chrome_start = 0
            self._chrome_rows = 0
            self._callout_spans = []
            self._rail_spans = []
            return base
        out = Text()
        for position, line in enumerate(lines[:insert_at] + injected + lines[insert_at:]):
            if position:
                out.append("\n")
            out.append_text(line)
        self._chrome_start = insert_at
        self._chrome_rows = len(injected)
        # The rail is injected FIRST, so the callout's own row offsets start
        # after it. Counted from the rail's rows rather than guessed from the
        # number of spans (two destinations can share a row).
        rail_count = (
            len(self._rail_rows(width, two_row_rail=self._shape[0])[0]) if rail_spans else 0
        )
        self._rail_spans = [
            (insert_at + row, start, end, index) for row, start, end, index in rail_spans
        ]
        self._callout_spans = [
            (insert_at + rail_count + row, start, end, scope)
            for row, start, end, scope in callout_spans
        ]
        row_map = getattr(self, "_line_rows", None)
        if row_map is not None:
            row_map = list(row_map)
            self._line_rows = row_map[:insert_at] + [None] * len(injected) + row_map[insert_at:]
        return out

    # -- hit-testing ---------------------------------------------------------

    def _setting_at(self, line: int, column: int) -> str | None:
        """The setting destination painted under a body-relative point."""
        for row, start, end, scope in self._callout_spans:
            if row == line and start <= column < end:
                return scope
        return None

    def _answer_at(self, line: int, column: int) -> int | None:
        """The compact rail's option under a body-relative point."""
        for row, start, end, index in self._rail_spans:
            if row == line and start <= column < end:
                return index
        return None

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """A handle changes a SETTING; a rail label ANSWERS; anything else falls through.

        The base hit-test only knows the option lines IT painted. The injected
        rows are resolved FIRST and the event is stopped, so a handle click can
        never also select whatever option line the same offset would have named
        — the false-positive class the base's own docstring calls a defect that
        answers the agent's question on the user's behalf.
        """
        if getattr(event, "button", 1) != 1:
            return
        body = getattr(self, "_body", None)
        if body is None or not body.is_mounted:
            return
        region = body.region
        if not region.contains(event.screen_x, event.screen_y):
            return
        line = event.screen_y - region.y
        column = event.screen_x - region.x
        scope = self._setting_at(line, column)
        if scope is not None:
            event.stop()
            self.engage()
            self._request_setting(scope)
            return
        index = self._answer_at(line, column)
        if index is not None:
            event.stop()
            self.engage()
            self._move_to(index)
            self._answer_with(self.row_key(index))
            return
        super().on_click(event)

    # -- the awaited half ----------------------------------------------------
    def wait(self) -> asyncio.Future[bool]:
        """The future the engine awaits. Resolved exactly once."""
        return self._future

    def set_note(self, note: str) -> None:
        """Append a one-line note to the QUESTION of a still-parked card.

        Written into the question rather than as a new row (UX round 1, U6).
        The card's row budget is allocated in a strict priority order
        (``AskPickerScreen._allocate``) in which the question ranks above the
        option rows and below only the footer, so a note carried there can
        never be the line that sheds at a tight size — and a note that is the
        first thing to disappear on a 16-row terminal is a note that is not
        there. A new chrome row would also have to be bought from that same
        budget, which is how the footer went missing in the first place.

        The case this exists for: a ``config.yml`` loosening lands while this
        card is parked. The card is deliberately NOT auto-answered — the human
        did not type that loosening in this pane — but the frame then said
        "every tool runs without asking" two lines above "the agent needs your
        approval", with a visible ``Allow all — stop asking for this session``
        row describing a state the session was already in. The note is what
        makes the frame stop contradicting itself.

        Joined with ``·`` on ONE logical line rather than with a newline:
        ``wrap_cells`` splits on spaces alone, so an embedded newline would
        travel inside a "word" and reach the terminal as a literal break the
        row budget never counted. The separator is the one this card already
        uses between a fact and its qualifier.

        Idempotent on the same note, so a second config tick delivering the
        same transition does not stack the sentence; a settled card is left
        alone, since its question is a historical record by then.
        """
        if self._answer is not None or not note:
            return
        question = self._questions[0]
        if question.question.endswith(note):
            return
        self._questions[0] = question.model_copy(
            update={"question": f"{question.question}  ·  {strip_control_sequences(note)}"}
        )
        # The wrap cache is keyed on (row, width) and the question is not in the
        # key, so the question's own wrap has to be invalidated by repainting.
        self._invalidate_description_wraps()
        self._repaint()

    @property
    def answered(self) -> bool:
        return self._answer is not None

    @property
    def answer(self) -> str | None:
        """The key the user answered with, or ``None`` while still pending."""
        return self._answer

    def resolve(self, approved: bool, *, answer: str | None = None) -> None:
        """Settle the gate (idempotent) with the decision.

        Idempotent because several paths can end one prompt — the keystroke, the
        selected row, an abort, a transcript clear, unmount — and a second
        ``set_result`` on a settled future raises. Losing that race must not
        take the app down.
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
        # Settle the picker half too, so the base class cannot later call back
        # into `_answered` for a gate that is already decided.
        self.settle(None)

    # -- answering -----------------------------------------------------------
    def answer_keys(self) -> frozenset[str]:
        """The approval letters, routable from the composer.

        `y`/`n`/`A` rather than the picker's ordinals: these are the keys this
        prompt has always advertised and the ones people have in their fingers.

        Empty while the card is drawing no options, for the same reason
        :meth:`on_key` refuses them there — an authorisation must never be
        committable from a card that did not show what it was offering. `n` is
        excluded from that suppression inside `on_key` (denial is the safe
        direction and Escape stops the turn anyway), but it is not routed from
        the COMPOSER, where the user is typing rather than answering.
        """
        if not self.visible_rows:
            return frozenset()
        return frozenset(self._ANSWER_KEYS)

    def routed_hint(self) -> tuple[str, str] | None:
        """The letters, spelled out: they are the keys, not a range over rows."""
        keys = [key for _label, key, _why in APPROVAL_CHOICES if key in self.answer_keys()]
        if not keys:
            return None
        return ("/".join(keys), "answer")

    def answer_from_key(self, character: str) -> None:
        """Take ``character`` as an answer routed from the composer."""
        self._answer_with(character)

    def row_key(self, index: int) -> str:
        """The letter that answers this row: `y`, `n`, or `A`.

        Rendered in the gutter where the ask picker puts its ordinal, so the
        letters are discoverable again. They are the older interface here — they
        predate the selectable list and are what a returning user reaches for —
        and after the rework they were live bindings advertised nowhere at all
        (D4, design round 1). The ordinals keep working; this only changes what
        the gutter SAYS.

        Read from the same table the key handler uses, so the letter printed
        beside a row cannot disagree with the letter that answers it.
        """
        if 0 <= index < len(APPROVAL_CHOICES):
            return APPROVAL_CHOICES[index][1]
        return ""

    def _answer_with(self, key: str) -> None:
        """Settle from one of the answer KEYS, whichever route produced it.

        The single place the key -> (approved, recorded answer) mapping lives,
        so a click on a row and a keystroke cannot disagree about what the user
        authorised. It is also where ``A`` becomes ``a``: see
        :data:`ALLOW_ALL_ANSWER`.
        """
        if key == ALLOW_ALL_KEY:
            self.resolve(True, answer=ALLOW_ALL_ANSWER)
            return
        self.resolve(key == "y", answer=key)

    def _answered(self, answers: dict[str, list[str]] | None) -> None:
        """Bridge the picker's label answer onto the gate's boolean.

        ``None`` (Escape, or the app tearing the card down) is a DENIAL, not a
        skip: the engine is parked on this future, and the honest reading of "I
        am not answering this" for a permission gate is that permission was not
        given.
        """
        if self._answer is not None:
            return
        chosen = (answers or {}).get("approval") or []
        label = chosen[0] if chosen else ""
        key = APPROVAL_KEY_BY_LABEL.get(label)
        if key is None:
            self.resolve(False, answer="n")
            return
        self._answer_with(key)

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        """The answer letters, kept working beside the selectable list.

        These are why the prompt is focusable at all, and they predate the list:
        anyone who has used this app has pressed `y`. They are handled here
        rather than as BINDINGS because the base class dispatches its own
        ``on_key`` first for the free-text row, and a binding would fire after
        it — so the letters have to be taken at the same level to be certain of
        arriving.

        Case matters on ``A`` and only on ``A``. It disarms the gate for the
        whole session, and this card takes focus: a user who did not notice and
        kept typing would otherwise hit the most common letter in English and
        permanently disable the gate with no receipt.
        """
        character = event.character
        if character is None:
            return
        # The two SETTING destinations are taken before the answer letters and
        # never through `_answer_with`: they change the rule for the NEXT
        # request, so consuming them as an answer would authorise THIS one on a
        # keystroke the user pressed to stop being asked. Consumed here so they
        # cannot reach the composer either — the card holds focus.
        scope = SETTING_KEYS.get(character)
        if scope is not None and self._answer is None and not self._read_only:
            event.stop()
            event.prevent_default()
            self._request_setting(scope)
            return
        if character not in self._ANSWER_KEYS:
            return
        event.stop()
        event.prevent_default()
        # Refused while the card is drawing no options.
        #
        # On a terminal too short for the list, the card shrinks to the question
        # alone — and the letters went on working, so `y` approved
        # `rm -rf /Users/x/project/data` from a frame that showed no answers at
        # all, and at one point from a frame whose entire content was the word
        # for refusing (D9, design round 2). An affirmative authorisation must
        # never be committable from a card that did not show what it was
        # offering; this is the same rule the footer already follows when it
        # advertises only the exit on a collapsed card.
        #
        # DENIAL is still allowed, because it is the safe direction and because
        # Escape means stop everywhere in this app regardless of what any card
        # shows. Only the permissive answers are withheld.
        # The compact rail is the same exception in the other direction: it IS
        # the options (all three labels, one row), so it is what `visible_rows`
        # would have said had the base viewport been used. Without this, every
        # answer but `n` would be refused on the one frame that shows all three.
        if not self.visible_rows and not self._rail_painted() and character != "n":
            return
        self._answer_with(character)

    def _rail_painted(self) -> bool:
        """Whether this frame draws the compact answer rail.

        Read from the last plan (:meth:`_layout`), the same value the painter
        used, rather than from the spans: a click-only span table can be empty
        on a frame whose rail is drawn but not yet clicked.
        """
        return bool(getattr(self, "_shape", (False, False, False))[1])

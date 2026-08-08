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

from rich.console import Group, RenderableType
from rich.style import Style
from rich.text import Text
from textual.events import Key

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
CHOICES: tuple[tuple[str, str], ...] = (
    ("y", "allow"),
    ("n", "deny"),
    ("A", "allow all"),
    ("esc", "stop"),
)

#: Marker prefix the builtin tools put on a description whose target sits
#: outside the workspace. Surfaced as its own tinted clause because "outside
#: the workspace" is the single most decision-relevant fact in the prompt.
OUTSIDE_MARKER = "[outside workspace]"

#: Glyph opening the question row. Deliberately not the tool ledger's marker:
#: this row is a QUESTION, and it must not read as one more completed action.
PROMPT_GLYPH = "?"


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
        warning = Style(color=theme_mod.semantic_color("warning"))
        fg = Style(color=theme_mod.semantic_color("fg"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        danger = Style(color=theme_mod.semantic_color("danger"))
        # The kit's one-cell inner padding pair, subtracted the way the tool row
        # does it. Taking the box raw let the answered receipt wrap onto column
        # zero — the gutter the user's own ❯ lives in.
        width = max((self.size.width or 80) - 2, 10)

        question = Text(" " * SPINE_INDENT, no_wrap=True, overflow="ellipsis")
        if self._answer is None:
            # The question glyph is the ONE place the prompt spends warning ink
            # while it is live, so the hazard clause below can outrank it.
            question.append(f"{PROMPT_GLYPH} ", style=warning)
            question.append("allow ", style=muted)
        else:
            allowed = self._answer in ("y", "a")
            question.append("✓ " if allowed else "✗ ", style=fg if allowed else danger)
            question.append("allowed " if allowed else "denied ", style=muted)
        question.append(self.tool_name, style=fg)

        detail, outside = self._detail()
        if detail:
            question.append("  ", style=dim)
            if outside:
                # The one clause that can change the answer, so it is the only
                # thing on the row allowed to be BOLD warning — and it is said in
                # words rather than re-emitting the parser's own bracket token.
                # On a settled receipt it drops to plain dim: the decision is
                # made, and a permanent alarm in the transcript trains the eye to
                # ignore the live one.
                if self._answer is None:
                    question.append("outside the workspace — ", style=warning + Style(bold=True))
                else:
                    question.append("outside the workspace — ", style=dim)
            # No floor: once the prefix has filled the row there is no room to
            # spend, and flooring at 8 appended eight cells past the edge.
            budget = width - question.cell_len
            if budget > 0:
                question.append(truncate_cells(detail, budget), style=dim)

        if self._answer is not None:
            return question
        hint = Text(" " * (SPINE_INDENT + 2), no_wrap=True, overflow="ellipsis")
        for index, (key, label) in enumerate(CHOICES):
            if index:
                hint.append(" · ", style=dim)
            # Keys in `fg`, not warning: a tint worn by the hazard AND by the key
            # that REFUSES distinguishes nothing. The keys are affordances, so
            # they read as the brightest plain ink and leave warning to the alarm.
            hint.append(key, style=fg)
            hint.append(f" {label}", style=dim)
        return Group(question, hint)

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

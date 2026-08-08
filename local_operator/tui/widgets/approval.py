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

from rich.console import Group, RenderableType
from rich.style import Style
from rich.text import Text
from textual.message import Message

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.tool_card import truncate_cells
from local_operator.tui.widgets.transcript import SPINE_INDENT, TranscriptBlock

#: What each key does, in the order the row prints them. Kept as data so the
#: hint row and the key handler cannot disagree about what is on offer.
CHOICES: tuple[tuple[str, str], ...] = (
    ("y", "allow"),
    ("a", "allow all"),
    ("n", "deny"),
)

#: Marker prefix the builtin tools put on a description whose target sits
#: outside the workspace. Surfaced as its own tinted clause because "outside
#: the workspace" is the single most decision-relevant fact in the prompt.
OUTSIDE_MARKER = "[outside workspace]"

#: Glyph opening the question row. Deliberately not the tool ledger's marker:
#: this row is a QUESTION, and it must not read as one more completed action.
PROMPT_GLYPH = "?"


class ApprovalAnswered(Message):
    """Posted when a prompt is answered, carrying the raw key (``y``/``a``/``n``).

    The key rather than a bool: ``a`` and ``y`` both allow this call, but only
    ``a`` changes the session's policy, and that distinction belongs to the app.
    """

    def __init__(self, answer: str) -> None:
        super().__init__()
        self.answer = answer


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

    BINDINGS = [
        ("y", "answer('y')", "Allow"),
        ("a", "answer('a')", "Allow all"),
        ("n", "answer('n')", "Deny"),
        ("escape", "answer('n')", "Deny"),
    ]

    def __init__(self, tool_name: str, description: str) -> None:
        super().__init__()
        self.add_class("approval-card")
        self.tool_name = tool_name
        self.description = description
        self._answer: str | None = None
        self._future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
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
        if not self._future.done():
            self._future.set_result(approved)
        self.remove_class("approval-pending")
        self._refresh_row()
        self.finalize()
        # The app owns what an answer MEANS beyond this one call (latching
        # "allow all", handing focus back), so the widget reports rather than
        # reaches. Guarded: resolve also runs during teardown, where the widget
        # may already be detached and posting would raise.
        if self.is_attached:
            self.post_message(ApprovalAnswered(self._answer))

    @property
    def answered(self) -> bool:
        return self._answer is not None

    # -- keys ---------------------------------------------------------------
    def action_answer(self, key: str) -> None:
        """Answer from a keystroke: ``y`` once, ``a`` for the session, ``n`` deny."""
        self.resolve(key in ("y", "a"), answer=key)

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
        amber = Style(color=theme_mod.semantic_color("warning"))
        fg = Style(color=theme_mod.semantic_color("fg"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        danger = Style(color=theme_mod.semantic_color("danger"))
        width = self.size.width or 80

        question = Text(" " * SPINE_INDENT, no_wrap=True, overflow="ellipsis")
        if self._answer is None:
            question.append(f"{PROMPT_GLYPH} ", style=amber)
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
                # The one clause that changes the answer gets the warning tint.
                question.append(f"{OUTSIDE_MARKER} ", style=amber)
            budget = max(8, width - question.cell_len - SPINE_INDENT)
            question.append(truncate_cells(detail, budget), style=dim)

        if self._answer is not None:
            return question
        hint = Text(" " * (SPINE_INDENT + 2), no_wrap=True, overflow="ellipsis")
        for index, (key, label) in enumerate(CHOICES):
            if index:
                hint.append(" · ", style=dim)
            hint.append(key, style=amber)
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

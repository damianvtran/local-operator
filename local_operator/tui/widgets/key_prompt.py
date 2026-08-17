"""The in-TUI "paste your API key" prompt.

Why this exists: eleven providers offer an interactive login and eight of them
have no OAuth at all — their entire login is "open the dashboard, copy the key,
paste it here" (`create_api_key_login` in the provider registry), plus the
QwenCloud Token Plan, which reads a key *before* starting its device flow. The
TUI deliberately shipped no ``on_manual_code_input`` hook, on the reasoning that
the loopback callback server is the real path and a paste prompt is a CLI
fallback. That reasoning holds for a loopback provider and is exactly wrong for
these: there is no callback server, nothing to fall back FROM, so `/login
alibaba` opened the browser, told the user to copy a key, and then failed with
"Alibaba Cloud login requires an interactive code prompt" — the login could
only ever fail, on every one of those providers.

So the TUI answers the paste itself, the same way it already answers tool
approvals rather than letting the harness call ``input()`` on a terminal Textual
owns in raw mode.

Design constraints, all inherited from :class:`ApprovalBlock` because this is
the same kind of surface (a focused transcript block resolving a future the
flow is parked on):

- The block takes FOCUS, so typed characters reach it rather than the composer.
- Focus is RESTORED to whatever held it, so pasting a key does not silently
  move the user out of the composer.
- Every exit path resolves the future EXACTLY once (:meth:`resolve`). A dropped
  future is a login coroutine parked forever, so the app also resolves it on
  abort, on transcript clear, and on unmount.

What it does that the approval prompt does not:

- **The typed value is never echoed.** An API key is a secret that stays on
  screen in the scrollback of a shared terminal, and screen shares and recorded
  demos are exactly where `/login` gets run. The row shows the key's LENGTH as
  a mask, which is the feedback a paste actually needs (did it arrive, did it
  arrive whole) without the value.
- **Bracketed paste is handled**, because pasting is the primary input here and
  a terminal delivers it as one ``Paste`` event rather than as keystrokes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from rich.cells import cell_len
from rich.console import Group, RenderableType
from rich.style import Style
from rich.text import Text
from textual import events

from local_operator.ansi import strip_control_sequences
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.tool_card import truncate_cells
from local_operator.tui.widgets.transcript import SPINE_INDENT, TranscriptBlock

#: The mask character standing in for one typed/pasted character. A block
#: rather than an asterisk so a long key reads as a bar whose length is
#: comparable at a glance, which is the one property of the value the user
#: needs to check before pressing enter.
MASK_CHAR = "•"

#: Cells of mask the row will draw before it stops growing and switches to a
#: count. A 200-character key would otherwise wrap the transcript in dots and
#: push the hint row off the bottom, and past a certain length the bar has
#: already answered "did the whole thing arrive".
MASK_MAX_CELLS = 32

#: What each key does, in the order the hint row prints them. Data rather than
#: prose so the hint row and the key handler cannot disagree, matching
#: ``ApprovalBlock.CHOICES``. Ordered enter -> esc: the affirmative is what the
#: user came here to do, and the hint row sheds whole choices from the right,
#: so the key that submits survives to the narrowest terminals.
CHOICES: tuple[tuple[str, str], ...] = (
    ("enter", "submit"),
    ("esc", "cancel"),
)

#: Below this width the two-cell spine indent is given up, matching
#: ``approval.SPINE_FLOOR_WIDTH``: an alignment edge that only some blocks
#: honour is not an edge, so the whole transcript sheds it together.
SPINE_FLOOR_WIDTH = 24


class KeyPromptBlock(TranscriptBlock):
    """One pending API-key paste: a focused prompt row that never echoes.

    Lifecycle: constructed with the provider's label and the instruction the
    registry entry carries, mounted, focused. :meth:`resolve` settles the future
    and repaints the row as a receipt — what was asked and whether a key was
    given — so the transcript keeps the outcome instead of the question simply
    vanishing.

    The future carries ``str | None``: the pasted text, or ``None`` for a
    cancel, which is precisely the ``on_manual_code_input`` contract the
    provider flows expect.
    """

    #: Always give the prompt a blank row above it: it interrupts the login's
    #: own progress notices, and flush against them it reads as more output
    #: rather than as a question waiting on the user.
    SPACING_LEAD = True
    SPACING_KIND = "approval"

    #: Focusable so typed characters reach it rather than the composer's buffer.
    can_focus = True

    #: Escape is bound HERE, unlike ``ApprovalBlock`` which deliberately lets it
    #: bubble to the app's global stop. The difference is what Escape would
    #: otherwise do: there it aborts a running turn, which is a coherent answer
    #: to "may this tool run". Here no turn is running — the app is parked on a
    #: login — so a bubbling Escape would stop nothing and leave the prompt on
    #: screen with no way out but typing. Cancelling the login IS the local
    #: meaning of "stop" while this block owns the keyboard.
    BINDINGS = [
        ("enter", "submit", "Submit"),
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        provider_label: str,
        instructions: str | None = None,
        on_settled: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.add_class("key-prompt-card")
        # Both strings come from the provider registry rather than from a model,
        # but they are stripped on the same principle the approval prompt
        # applies to tool arguments: anything reaching a real terminal is
        # stripped at exactly one place, so a later registry entry (or an
        # embedder's own provider definition) cannot introduce CSI that erases
        # the rows above this prompt and repaints a forged one over them.
        self.provider_label = strip_control_sequences(provider_label)
        self.instructions = strip_control_sequences(instructions or "")
        self._typed: list[str] = []
        self._settled = False
        self._submitted: str | None = None
        # `get_running_loop`, not `get_event_loop`: the future must belong to the
        # loop that awaits it, and stating that precondition turns a construction
        # from a sync context into an immediate error rather than a future nobody
        # ever resolves. (3.14 removed the implicit-loop fallback.)
        self._future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self._on_settled = on_settled
        self._restore_focus: object | None = None
        self._refresh_row()

    # -- the awaited half ----------------------------------------------------
    def wait(self) -> asyncio.Future[str | None]:
        """The future the login flow awaits. Resolved exactly once."""
        return self._future

    @property
    def answered(self) -> bool:
        return self._settled

    @property
    def typed_length(self) -> int:
        """How many characters are held. The value itself is never exposed:
        tests and the renderer both need the length and neither needs the key."""
        return len(self._typed)

    def resolve(self, value: str | None) -> None:
        """Settle the prompt (idempotent) and repaint it as a receipt.

        Idempotent because several paths end one prompt — enter, escape, an
        abort, a transcript clear, unmount — and a second ``set_result`` on a
        settled future raises. Losing that race must not take the app down.
        """
        if self._settled:
            return
        self._settled = True
        self._submitted = value
        # The buffer is dropped as soon as the value has been handed over, so a
        # settled block sitting in the transcript is not still holding the
        # user's key in memory for the rest of the session.
        self._typed = []
        if not self._future.done():
            self._future.set_result(value)
        self.remove_class("key-prompt-pending")
        self._refresh_row()
        self.finalize()
        if self._on_settled is not None:
            self._on_settled()

    # -- keys ---------------------------------------------------------------
    def action_submit(self) -> None:
        """Enter: hand over what was typed, or cancel when nothing was.

        An empty submit is a CANCEL rather than an empty key, matching the CLI
        prompt's "empty to cancel" and the registry's own guard. Storing an
        empty string would write a blank credential row that shadows a working
        environment key and turns every later request into an auth failure with
        no visible cause.
        """
        typed = "".join(self._typed).strip()
        self.resolve(typed or None)

    def action_cancel(self) -> None:
        self.resolve(None)

    async def _on_paste(self, event: events.Paste) -> None:
        """Take a bracketed paste as a whole.

        Pasting is the PRIMARY way a key arrives here, and a terminal delivers
        it as one ``Paste`` event, not as a stream of key events — without this
        handler a pasted key would be silently dropped and the user would be
        looking at an empty prompt after a paste that appeared to work.

        Newlines are stripped rather than treated as a submit: a key copied out
        of a dashboard frequently carries a trailing newline, and letting that
        submit would make the paste and the confirmation the same gesture, with
        no chance to see the mask first. Control characters are stripped for the
        same reason the constructor strips its labels.
        """
        if self._settled:
            return
        text = strip_control_sequences(event.text).replace("\n", "").replace("\r", "")
        if text:
            self._typed.extend(text)
            self._refresh_row()
        event.stop()
        event.prevent_default()

    async def _on_key(self, event: events.Key) -> None:
        """Collect printable characters and handle backspace.

        Runs BEFORE ``BINDINGS`` (Textual dispatches the focused widget's own
        handlers first), so enter and escape are excluded by hand and left to
        their actions.

        Unlike ``ApprovalBlock``, a printable key is NOT passed through to the
        composer. There the answer keys are a tiny set and anything else is the
        user typing their next instruction; here every printable character is
        the answer, and forwarding them would scatter a secret across the
        composer one character at a time.
        """
        if self._settled:
            return
        if event.key in ("enter", "escape"):
            return
        if event.key == "backspace":
            if self._typed:
                self._typed.pop()
                self._refresh_row()
            event.stop()
            event.prevent_default()
            return
        if event.is_printable and event.character:
            self._typed.append(event.character)
            self._refresh_row()
            event.stop()
            event.prevent_default()

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """Clicking an unsettled prompt takes focus back so typing works again.

        The way back if focus moved elsewhere; without it the question would sit
        on screen with no discoverable way to answer it.
        """
        if self._settled:
            return
        self.focus()
        event.stop()

    def on_mount(self) -> None:
        """Take focus, remembering what had it so it can be handed back."""
        self.add_class("key-prompt-pending")
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
        """Rebuild the card at its own width.

        Bypasses the finalize guard on purpose: a settled prompt still has to
        re-fit on resize, and the content is a pure function of state — the same
        discipline ``ApprovalBlock._refresh_row`` applies.
        """
        was_finalized = self._finalized
        self._finalized = False
        try:
            self.set_content(self._build())
        finally:
            self._finalized = was_finalized

    def on_resize(self, event: object) -> None:
        self._refresh_row()

    def _mask(self) -> str:
        """The typed value as a mask, or a count once it outgrows the row.

        The count is the honest degradation: past ``MASK_MAX_CELLS`` the bar has
        stopped being comparable at a glance anyway, and a number still answers
        the only question the user has (did the whole key arrive).
        """
        count = len(self._typed)
        if count == 0:
            return ""
        if count <= MASK_MAX_CELLS:
            return MASK_CHAR * count
        return f"{MASK_CHAR * MASK_MAX_CELLS} {count} chars"

    def _build(self) -> RenderableType:
        width = max((self.size.width or 80) - 2, 10)
        fg = Style(color=theme_mod.semantic_color("fg"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        accent = Style(color=theme_mod.semantic_color("accent"))
        # The transcript's shared left edge, given up as a whole below the width
        # where it stops being affordable — see ``SPINE_FLOOR_WIDTH``.
        pad = " " * SPINE_INDENT if width >= SPINE_FLOOR_WIDTH else ""
        body = max(width - len(pad), 1)

        def row(*parts: tuple[str, Style]) -> Text:
            """One line, truncated to the body width with its styling intact.

            Truncation is applied PER PART against the room the earlier parts
            left, rather than to a flattened string: cutting the joined plain
            text and restyling it afterwards is how a row ends up with its
            colours one character out of step with its words.
            """
            line = Text(pad)
            remaining = body
            for text, style in parts:
                if remaining <= 0:
                    break
                fitted = truncate_cells(text, remaining)
                if not fitted:
                    break
                line.append(fitted, style=style)
                remaining -= cell_len(fitted)
            return line

        if self._settled:
            # The receipt. It states the OUTCOME rather than repeating the
            # question, because the question is answered and a transcript full
            # of unanswered-looking prompts is how a user loses track of which
            # one the app is actually waiting on. The submitted key's LENGTH is
            # the receipt, never the key.
            if self._submitted is None:
                return row(("· ", dim), (f"{self.provider_label} login cancelled", muted))
            return row(
                ("· ", dim),
                (f"{self.provider_label} key received ", muted),
                (f"({len(self._submitted)} chars)", dim),
            )

        lines = [row(("? ", accent), (f"Paste your {self.provider_label} API key", fg))]
        if self.instructions:
            lines.append(row(("  ", dim), (self.instructions, dim)))
        mask = self._mask()
        if mask:
            lines.append(row(("  ", dim), (mask, accent)))
        else:
            # The empty state says what to do rather than leaving a bare caret,
            # because this prompt appears right after the browser opened and the
            # user's attention was somewhere else entirely.
            lines.append(row(("  ", dim), ("paste or type the key, then press enter", dim)))

        hint_parts: list[tuple[str, Style]] = []
        for index, (key, label) in enumerate(CHOICES):
            if index:
                hint_parts.append(("   ", dim))
            hint_parts.append((key, accent))
            hint_parts.append((f" {label}", muted))
        lines.append(row(*hint_parts))
        return Group(*lines)

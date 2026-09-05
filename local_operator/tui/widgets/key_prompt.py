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
#: Cells between two choices on the hint row.
#:
#: Three spaces, which is NOT what ``ApprovalBlock`` uses — its hint row joins
#: choices with ``" · "``. The difference is deliberate: that row carries four
#: choices whose labels are single words (``deny``, ``allow``), where a middot
#: is what stops them reading as one phrase. This row has two, each already a
#: key plus a verb, and a separator glyph between them competes with the ``·``
#: the settled receipt opens with.
HINT_SEPARATOR = "   "

CHOICES: tuple[tuple[str, str], ...] = (
    ("enter", "submit"),
    ("esc", "cancel"),
)

#: Below this width the two-cell spine indent is given up, matching
#: ``approval.SPINE_FLOOR_WIDTH``: an alignment edge that only some blocks
#: honour is not an edge, so the whole transcript sheds it together.
SPINE_FLOOR_WIDTH = 24


def _short_label(label: str) -> str:
    """A registry name reduced to the company, for use inside a sentence.

    Registry names carry a parenthetical qualifier that exists to disambiguate
    ROWS IN A LIST (``xAI (Grok API key)``, ``Anthropic (Claude Pro/Max)``,
    ``QwenCloud Token Plan (usage OAuth)``). Dropped into this prompt's sentence
    they read as nonsense — ``Paste your xAI (Grok API key) API key`` names the
    credential twice and parenthesises the wrong half. The user has already
    chosen the provider by typing it, so the qualifier has no work left to do
    here; the sentence needs the company.

    Only the trailing parenthetical is removed, and only when something is left:
    a name that is nothing but a parenthetical keeps it rather than becoming
    empty.
    """
    head, sep, _tail = label.partition(" (")
    if sep and head.strip():
        return head.strip()
    return label


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
        *,
        secret: bool = True,
        sole_path: bool = True,
        field_label: str | None = None,
        default: str | None = None,
    ) -> None:
        """``secret`` says WHICH kind of value this prompt is reading.

        ``True`` (the default, and every paste-a-key provider): a long-lived API
        key. It is masked, and the prompt asks for a "key".

        ``False``: an OAuth authorization code, which is Anthropic's optional
        paste fallback. It is single-use and expires in minutes, and it is a
        long opaque ``code#state`` string a user needs to SEE to check their
        paste landed whole, so it is echoed. Masking it would cost real
        legibility to protect a value that is spent on redemption — and calling
        it an "API key" would send the user hunting a key Anthropic never
        issued them.

        ``sole_path`` is whether declining ENDS the login. True for a
        paste-a-key provider: there is nothing else to wait for. False for
        Anthropic, where the loopback callback is still listening and
        ``LoopbackFlow._await_code`` re-parks on a declined paste by design, so
        the login carries on in the browser. Only the receipt depends on it, and
        it must: reporting "login cancelled" for a login that is still running
        told the user something false, and their next `/login` was refused with
        "a login is already in progress".
        """
        super().__init__()
        self.add_class("key-prompt-card")
        self.secret = secret
        self.sole_path = sole_path
        self.field_label = strip_control_sequences(field_label or "")
        # An explicit default (including an empty optional token) distinguishes
        # Enter from Escape without changing OAuth's empty-means-cancel contract.
        self.default = default
        # Both strings come from the provider registry rather than from a model,
        # but they are stripped on the same principle the approval prompt
        # applies to tool arguments: anything reaching a real terminal is
        # stripped at exactly one place, so a later registry entry (or an
        # embedder's own provider definition) cannot introduce CSI that erases
        # the rows above this prompt and repaints a forged one over them.
        self.provider_label = _short_label(strip_control_sequences(provider_label))
        self.instructions = strip_control_sequences(instructions or "")
        self._typed: list[str] = []
        self._settled = False
        #: The LENGTH of the value handed over, or None when nothing was (a
        #: cancel). Deliberately not the value: the receipt only ever needs how
        #: many characters arrived, and a settled block outlives the login — it
        #: sits in the transcript, and the app also retains the last one to
        #: correct its receipt — so holding the key here would keep a
        #: credential resident for the rest of the session. Same reason
        #: ``_typed`` is dropped in :meth:`resolve`.
        self._submitted_length: int | None = None
        #: Set when the prompt was retired because the login completed another
        #: way (see :meth:`resolve`), so the receipt does not claim a cancel.
        self._superseded = False
        #: Set when the value this prompt handed over turned out to be unusable
        #: (see :meth:`mark_unusable`), so the receipt does not claim a success
        #: the flow rejected.
        self._unusable = False
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
    def submitted_length(self) -> int | None:
        """How many characters were handed over, or None when nothing was.

        The question "could this prompt still be owed a correction?" — only a
        block that produced a value can be, since :meth:`mark_unusable` is a
        no-op otherwise. Exposed so the app can ask it without reaching into
        the private field, and never exposes the value itself.
        """
        return self._submitted_length

    @property
    def typed_length(self) -> int:
        """How many characters are held. The value itself is never exposed:
        tests and the renderer both need the length and neither needs the key."""
        return len(self._typed)

    def resolve(self, value: str | None, *, superseded: bool = False) -> None:
        """Settle the prompt (idempotent) and repaint it as a receipt.

        Idempotent because several paths end one prompt — enter, escape, an
        abort, a transcript clear, unmount — and a second ``set_result`` on a
        settled future raises. Losing that race must not take the app down.

        ``superseded`` means the prompt was retired because the login finished
        some OTHER way, and it exists because the block cannot tell the two
        apart from ``value is None`` alone. For Anthropic the paste races the
        loopback callback: when the browser redirect wins, the paste task is
        cancelled and this resolves with ``None`` — a SUCCESSFUL login that
        painted "login cancelled" underneath its own success notice, telling
        the user their completed login had been cancelled.
        """
        if self._settled:
            return
        self._settled = True
        self._superseded = superseded
        self._submitted_length = None if value is None else len(value)
        # The buffer is dropped as soon as the value has been handed over, so a
        # settled block sitting in the transcript is not still holding the
        # user's key in memory for the rest of the session.
        self._typed = []
        self.default = None
        if not self._future.done():
            self._future.set_result(value)
        self.remove_class("key-prompt-pending")
        self._refresh_row()
        self.finalize()
        if self._on_settled is not None:
            self._on_settled()

    def mark_unusable(self) -> None:
        """Repaint a settled receipt to say the pasted value was NOT accepted.

        Called after the fact, because only the login flow can tell: the block
        hands over whatever was typed, and whether it parses into an
        authorization code is decided one layer out in
        ``_parse_pasted_callback``. Same after-the-fact correction as
        ``superseded``, for the same reason — the block cannot know the outcome
        from the value alone.

        Without this, a rejected paste settled as ``✓ … code received (107
        chars)``: a success glyph and the word "received" over a value the flow
        had just thrown away, directly above the notice explaining why. The
        count made it worse by looking like corroboration. The login is still
        live when this fires (the flow re-prompts and the loopback callback is
        still racing), so this is not a cancel and must not use the cancel
        wording.

        No-op on a prompt that is not settled or was cancelled: there is no
        success claim to correct in either case.
        """
        if not self._settled or self._submitted_length is None:
            return
        self._unusable = True
        self._refresh_row()

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
        self.resolve(typed or self.default if self.default is not None else typed or None)

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
        if event.key in ("tab", "shift+tab"):
            # TAB is swallowed while the prompt is live, and that is a security
            # boundary rather than a focus preference. Textual's default Tab
            # moves focus to the next focusable widget, which is the composer:
            # the prompt went on saying it was waiting (its ground differs from
            # the focused ground by ~1.04:1, which nobody perceives), the user
            # kept typing their key into the composer, and Enter SENT THE API
            # KEY TO THE MODEL as a chat message — into the transcript in plain
            # text and into the provider's logs. Verified end to end before this
            # guard existed.
            #
            # SHIFT+TAB never reaches here: the app binds it as a PRIORITY
            # binding (`cycle_effort`), and Textual matches priority bindings
            # before the focused widget's handlers. It is listed anyway because
            # this block's own default for a focus key must be "swallow" — if
            # that app binding is ever narrowed or moved, the safe behaviour is
            # already here rather than one forgotten line away. It costs nothing
            # and it is not what stops shift+tab today.
            #
            # A prompt the app is parked on owns the keyboard until it is
            # answered; both ways out (enter, escape) are on this block and are
            # advertised in its hint row.
            event.stop()
            event.prevent_default()
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
        if not self.secret:
            # An OAuth code is echoed: see the ``secret`` note on __init__. The
            # row's own truncation keeps a long one inside the card.
            return "".join(self._typed)
        if count <= MASK_MAX_CELLS:
            return MASK_CHAR * count
        # Past the bar's useful length the COUNT is the message, so it is
        # written first and the bar takes what is left. Written bar-first, the
        # row helper truncated the count away at narrow widths — at 40 columns a
        # 33-character and a 300-character paste rendered as the same row, so
        # "did the whole key arrive" stopped being answerable exactly where the
        # bar had already stopped answering it.
        return f"{count} chars {MASK_CHAR * MASK_MAX_CELLS}"

    def _build(self) -> RenderableType:
        width = max((self.size.width or 80) - 2, 10)
        fg = Style(color=theme_mod.semantic_color("fg"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        accent = Style(color=theme_mod.semantic_color("accent"))
        success = Style(color=theme_mod.semantic_color("success"))
        danger = Style(color=theme_mod.semantic_color("danger"))
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
            # The GLYPH carries the outcome, not the sentence, and that is what
            # makes the two receipts distinguishable when the row is truncated:
            # at 10-27 columns "… key received" and "… login cancelled" both cut
            # to the provider name and rendered byte-identical, so the narrowest
            # terminals could not tell a stored credential from a cancelled one.
            # Same device and the same two characters ``ApprovalBlock`` uses for
            # its own settled receipt, and the same pair the transcript's
            # success/error notices use.
            # The two informational receipts put their DISTINGUISHING word before
            # the provider label, because truncation eats the tail: with the
            # label first, "paste skipped" and "paste no longer needed" both cut
            # to "· Alibaba Cloud paste…" at 24 columns and below, and those two
            # are opposites (the login is still running / the login is over).
            # Same defect the glyphs fixed for success-vs-cancel, one row over.
            # Success and cancel keep the label first: their glyphs already
            # separate them, and the label is the more useful lead there.
            if self._superseded:
                # Says only what this BLOCK knows: it stopped being needed. The
                # login's own outcome notice, success or failure, follows it.
                return row(("· ", dim), ("no longer needed ", muted), (self.provider_label, dim))
            if self._submitted_length is None:
                if not self.sole_path:
                    # Declining here does not end the login (see ``sole_path``):
                    # the browser flow is still live, so the receipt says what
                    # was actually declined and what is still running. Not a
                    # failure glyph: nothing failed.
                    return row(
                        ("· ", dim),
                        ("paste skipped ", muted),
                        (f"{self.provider_label} — still waiting for the browser", dim),
                    )
                return row(("✗ ", danger), (f"{self.provider_label} login cancelled", muted))
            if self._unusable:
                # A value was handed over and the flow could not use it (see
                # ``mark_unusable``). Distinguishing word FIRST, as the two
                # informational receipts above do and for the same truncation
                # reason. The length is dropped: it read as evidence the paste
                # was fine, which is the opposite of what happened.
                return row(
                    ("✗ ", danger),
                    ("paste not usable ", muted),
                    (f"{self.provider_label} — still waiting for the browser", dim),
                )
            return row(
                ("✓ ", success),
                (
                    f"{self.provider_label} "
                    f"{self.field_label or ('key' if self.secret else 'code')} received ",
                    muted,
                ),
                (f"({self._submitted_length} chars)", dim),
            )

        noun = "API key" if self.secret else "authorization code"
        title = (
            f"{self.provider_label}: {self.field_label}"
            if self.field_label
            else f"Paste your {self.provider_label} {noun}"
        )
        lines = [row(("? ", accent), (title, fg))]
        if self.instructions:
            lines.append(row(("  ", dim), (self.instructions, dim)))
        mask = self._mask()
        if mask:
            lines.append(row(("  ", dim), (mask, accent)))
        else:
            # The empty state says what to do rather than leaving a bare caret,
            # because this prompt appears right after the browser opened and the
            # user's attention was somewhere else entirely.
            hint_noun = self.field_label or ("key" if self.secret else "code")
            lines.append(
                row(("  ", dim), (f"paste or type the {hint_noun}, then press enter", dim))
            )

        # The hint row sheds WHOLE choices from the right rather than letting
        # the last one be cut mid-word: `enter submit   esc ca…` teaches a key
        # that does not exist, and a half-word is not a shorter way of saying
        # the same thing. Built by measuring rather than by truncating, which is
        # what the row helper alone would do. ``CHOICES`` is ordered so the key
        # that SUBMITS survives longest: a prompt that only tells you how to
        # give up is one you cannot get past.
        hint_parts: list[tuple[str, Style]] = []
        used = 0
        for index, (key, label) in enumerate(CHOICES):
            separator = HINT_SEPARATOR if index else ""
            width_needed = cell_len(separator) + cell_len(key) + cell_len(f" {label}")
            if used + width_needed > body:
                break
            if separator:
                hint_parts.append((separator, dim))
            hint_parts.append((key, accent))
            hint_parts.append((f" {label}", muted))
            used += width_needed
        if hint_parts:
            lines.append(row(*hint_parts))
        return Group(*lines)

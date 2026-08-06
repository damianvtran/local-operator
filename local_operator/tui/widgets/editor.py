"""Input editor — TextArea inverted to chat semantics.

Textual's TextArea defaults to newline-on-Enter; this product wants
submit-on-Enter. The subclass inverts that and takes the terminal key idioms:

- ``Enter`` submits (posts :class:`EditorSubmitted`); with the command picker
  open it first completes the highlighted command, THEN submits
- ``Shift+Enter`` inserts a newline
- ``Ctrl+C`` posts :class:`InterruptRequested` (abort the turn) — never exits
- ``Ctrl+D`` on an EMPTY buffer quits; otherwise it falls through to delete
- ``Up``/``Down`` move the picker's highlight while it is open; otherwise they
  cycle prompt history when the caret sits at the top/bottom edge of the
  buffer, and inside the text they keep their cursor-move meaning
- ``Tab`` completes the highlighted command WITHOUT submitting
- ``Esc`` dismisses the picker, leaving the typed text alone

Key interception happens in :meth:`_on_key`, which runs BEFORE TextArea's
document-insert path, so a handled key never reaches the buffer. Unhandled
keys fall through to the stock editor behavior.

The editor OWNS its :class:`CommandPicker` (built in ``__init__``, mounted by
the app as a sibling below the input row, since the picker must draw outside
the chevron+editor row). One picker always exists, so every completion path —
Tab, Enter, mouse click — runs through the same code with no "no picker
attached" variant to keep in step.
"""

from __future__ import annotations

from textual import events
from textual.message import Message
from textual.widgets import TextArea
from textual.widgets.text_area import Edit, EditResult

from local_operator.tui.autocomplete import SlashCommand
from local_operator.tui.widgets.command_picker import CommandPicker, slash_context


class EditorSubmitted(Message):
    """Posted when the user submits the editor (Enter without Shift)."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class InterruptRequested(Message):
    """Posted on Ctrl+C: abort the running turn, never exit the app."""

    def __init__(self) -> None:
        super().__init__()


class EditorQuit(Message):
    """Posted on Ctrl+D with an empty buffer."""

    def __init__(self) -> None:
        super().__init__()


class Editor(TextArea):
    """Multiline prompt editor with submit-on-Enter, history, slash-completion."""

    #: Maximum remembered prompts.
    HISTORY_LIMIT = 200

    def __init__(
        self,
        placeholder: str = "Message Local Operator…",
        commands: list[SlashCommand] | None = None,
    ) -> None:
        # Built BEFORE super().__init__: TextArea's constructor loads its
        # initial document, which funnels through load_text() and therefore
        # through _sync_picker().
        self._picker = CommandPicker(self._apply_command)
        # tab_behavior="indent": Tab NEVER moves focus (TUI-013). Command
        # completion consumes the key first; otherwise it indents.
        super().__init__(placeholder=placeholder, soft_wrap=True, tab_behavior="indent")
        self._history: list[str] = []
        self._history_index: int | None = None  # None = not navigating
        self._draft: str = ""  # buffer text saved when history nav starts
        self.set_commands(commands or [])

    # -- public API ---------------------------------------------------------
    @property
    def picker(self) -> CommandPicker:
        """The slash-command picker. The app mounts it below the input row."""
        return self._picker

    def prompt_history(self) -> list[str]:
        """Recorded prompts, oldest first.

        Named ``prompt_history`` rather than ``history``: ``TextArea`` already
        owns a ``history`` attribute for its undo stack, and shadowing it with
        a method of an unrelated type is a live footgun for anything that
        reaches for the base class's own edit history.
        """
        return list(self._history)

    def set_commands(self, commands: list[SlashCommand]) -> None:
        """Slash commands offered to the picker (sync, no I/O)."""
        self._picker.set_commands(commands)

    def clear_content(self) -> None:
        """Empty the buffer and leave history navigation."""
        self.text = ""
        self._history_index = None

    # -- key interception ---------------------------------------------------
    async def _on_key(self, event: events.Key) -> None:
        """Handle chat keys before TextArea's insert path sees them."""
        key = event.key
        # Re-derive the picker state from the buffer BEFORE routing. The
        # buffer is the only authority on whether a command word is open, and
        # syncing here means routing never depends on a queued Changed
        # message having been drained first.
        self._sync_picker()
        if self._picker.is_open():
            if key == "escape":
                self._picker.dismiss()
                event.stop()
                event.prevent_default()
                return
            if key in ("up", "down"):
                # The picker owns the arrows while it is open: history nav and
                # caret movement both stay reachable one Esc away, and an open
                # list that ignored Up/Down would look broken.
                self._picker.move(-1 if key == "up" else +1)
                event.stop()
                event.prevent_default()
                return
            if key in ("tab", "enter"):
                # Both keys insert the SAME completion, so the highlighted row
                # can never mean two different commands depending on the key.
                #
                # Enter only SENDS when the choice is unambiguous: one match, or
                # a name the user typed in full. The registry's blast radius is
                # not uniform — `/loop` starts autonomous work and `/logout`
                # removes credentials — and with a fuzzy matcher `/lo` highlights
                # `loop` while `login` and `logout` also match, so "Enter runs
                # whatever the matcher picked" could start a loop for a user
                # reaching for login, rewriting their text and running it in one
                # keystroke. When it is ambiguous, Enter completes (Tab's
                # behaviour) and a second Enter sends — the extra keystroke only
                # appears where the intent genuinely is not clear (D16).
                name = self._picker.highlighted_name()
                if name is not None:
                    # Decided BEFORE the completion is applied: `_apply_command`
                    # rewrites the text and re-syncs the picker, so asking
                    # afterwards would measure the completed word (always one
                    # exact match) and submit unconditionally.
                    send = key == "enter" and self._picker_choice_is_unambiguous(name)
                    self._apply_command(name)
                    if send:
                        self._submit()
                    event.stop()
                    event.prevent_default()
                    return
        if key == "enter":
            self._submit()
            event.stop()
            event.prevent_default()
            return
        if key == "shift+enter":
            # Explicit newline; TextArea's stock path would also submit here,
            # so insert the newline ourselves and consume the key.
            self.insert("\n")
            event.stop()
            event.prevent_default()
            return
        if key == "ctrl+c":
            self.post_message(InterruptRequested())
            event.stop()
            event.prevent_default()
            return
        if key == "ctrl+d" and not self.text:
            self.post_message(EditorQuit())
            event.stop()
            event.prevent_default()
            return
        if key == "up" and self._caret_at_top_edge() and self._history:
            self._navigate_history(-1)
            event.stop()
            event.prevent_default()
            return
        if (
            key == "down"
            and self._caret_at_bottom_edge()
            and (self._history_index is not None or self._history)
        ):
            self._navigate_history(+1)
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)

    # -- submit -------------------------------------------------------------
    def _submit(self) -> None:
        text = self.text
        if text.strip():
            self._record_history(text)
        self._picker.close()
        self.post_message(EditorSubmitted(text))
        self.clear_content()

    # -- command completion -------------------------------------------------
    def edit(self, edit: Edit) -> EditResult:
        """Every buffer mutation funnels through here — resync the picker.

        Hooking the two mutation funnels (:meth:`edit` for inserts/deletes/
        undo, :meth:`load_text` for whole-buffer replacement) keeps the picker
        exact for keystrokes, pastes, and history navigation alike, and does
        it synchronously. Listening for ``TextArea.Changed`` instead would put
        the picker one message-loop tick behind the buffer, which is the tick
        in which Enter decides whether to complete.
        """
        result = super().edit(edit)
        self._sync_picker()
        return result

    def load_text(self, text: str) -> None:
        super().load_text(text)
        self._sync_picker()

    def _sync_picker(self) -> None:
        self._picker.sync(self.text)

    def _picker_choice_is_unambiguous(self, name: str) -> bool:
        """Whether Enter may SEND the highlighted command, not just insert it.

        Unambiguous means one of two things:

        * there is exactly one match, so the highlighted row is the only command
          the query could mean; or
        * the user typed the name in full, so they named it rather than letting
          the matcher choose.

        Everything else completes and waits for a second Enter. The point is the
        registry's uneven blast radius: `/usage` is harmless and `/loop` and
        `/logout` are not, so a fuzzy pick must not be *run* on one keystroke.
        """
        context = slash_context(self.text)
        if context is None:
            return False
        if len(self._picker.suggestions()) <= 1:
            return True
        return context.query.strip().lower() == name.strip().lower()

    def _apply_command(self, name: str) -> None:
        """Replace the typed command word with ``/name `` (trailing space).

        The trailing space is load-bearing, not cosmetic: most commands take
        an argument (``/model provider/id``), and it is also what closes the
        picker — the word is now whitespace-terminated, so the list drops away
        on the same keystroke that chose from it.
        """
        context = slash_context(self.text)
        if context is None:
            return
        # Everything from the slash onward IS the command word (that is what
        # makes it a command word), so the prefix is all that must survive.
        self.text = f"{self.text[: context.start]}/{name} "
        self.move_cursor(self._end_of_buffer())

    # -- history ------------------------------------------------------------
    def _caret_row(self) -> int:
        return self.selection.end[0]

    def _caret_at_top_edge(self) -> bool:
        return self._caret_row() == 0

    def _caret_at_bottom_edge(self) -> bool:
        return self._caret_row() == self.document.line_count - 1

    def _record_history(self, text: str) -> None:
        stripped = text.strip()
        if stripped and (not self._history or self._history[-1] != stripped):
            self._history.append(stripped)
            if len(self._history) > self.HISTORY_LIMIT:
                self._history.pop(0)
        self._history_index = None

    def _navigate_history(self, direction: int) -> None:
        if not self._history:
            return
        if self._history_index is None:
            if direction < 0:
                self._draft = self.text
                self._history_index = len(self._history) - 1
            else:
                return  # Down with no navigation active: nothing to restore
        else:
            self._history_index += direction
        if self._history_index >= len(self._history):
            # Past the newest entry: restore the draft and exit navigation.
            self._history_index = None
            self.text = self._draft
            self.move_cursor(self._end_of_buffer())
            return
        self._history_index = max(0, self._history_index)
        self.text = self._history[self._history_index]
        self.move_cursor(self._end_of_buffer())

    def _end_of_buffer(self) -> tuple[int, int]:
        last_row = self.document.line_count - 1
        return last_row, len(self.document.get_line(last_row))

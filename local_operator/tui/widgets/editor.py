"""Input editor — TextArea inverted to chat semantics.

Textual's TextArea defaults to newline-on-Enter; this product wants
submit-on-Enter. The subclass inverts that and takes the terminal key idioms:

- ``Enter`` submits (posts :class:`EditorSubmitted`)
- ``Shift+Enter`` inserts a newline
- ``Ctrl+C`` posts :class:`InterruptRequested` (abort the turn) — never exits
- ``Ctrl+D`` on an EMPTY buffer quits; otherwise it falls through to delete
- ``Up``/``Down`` cycle prompt history when the caret sits at the top/bottom
  edge of the buffer; inside the text they keep their cursor-move meaning
- ``Tab`` completes a slash command synchronously (no I/O) when unambiguous

Key interception happens in :meth:`_on_key`, which runs BEFORE TextArea's
document-insert path, so a handled key never reaches the buffer. Unhandled
keys fall through to the stock editor behavior.
"""

from __future__ import annotations

from textual import events
from textual.message import Message
from textual.widgets import TextArea

from local_operator.tui.autocomplete import SlashCommand, complete_command


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
        # tab_behavior="indent": Tab NEVER moves focus (TUI-013). Slash
        # completion consumes the key first; otherwise it indents.
        super().__init__(placeholder=placeholder, soft_wrap=True, tab_behavior="indent")
        self._history: list[str] = []
        self._history_index: int | None = None  # None = not navigating
        self._draft: str = ""  # buffer text saved when history nav starts
        self._commands: list[SlashCommand] = commands or []

    # -- public API ---------------------------------------------------------
    def prompt_history(self) -> list[str]:
        """Recorded prompts, oldest first.

        Named ``prompt_history`` rather than ``history``: ``TextArea`` already
        owns a ``history`` attribute for its undo stack, and shadowing it with
        a method of an unrelated type is a live footgun for anything that
        reaches for the base class's own edit history.
        """
        return list(self._history)

    def set_commands(self, commands: list[SlashCommand]) -> None:
        """Slash commands offered to Tab completion (sync, no I/O)."""
        self._commands = list(commands)

    def clear_content(self) -> None:
        """Empty the buffer and leave history navigation."""
        self.text = ""
        self._history_index = None

    # -- key interception ---------------------------------------------------
    async def _on_key(self, event: events.Key) -> None:
        """Handle chat keys before TextArea's insert path sees them."""
        key = event.key
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
        if key == "tab":
            if self._try_complete_slash():
                event.stop()
                event.prevent_default()
                return
            await super()._on_key(event)
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
        self.post_message(EditorSubmitted(text))
        self.clear_content()

    # -- slash completion ---------------------------------------------------
    def _try_complete_slash(self) -> bool:
        """Complete the leading ``/token`` if exactly one command matches."""
        completed = complete_command(self.text, self._commands)
        if completed is None or completed == self.text:
            return False
        self.text = completed
        self.move_cursor(self._end_of_buffer())
        return True

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

"""``/move … --to``: the working-set confirmation (docs/design/move-working-set.md §11.3).

WHY THIS SCREEN EXISTS (operator, 2026-11-18): a move mid-conversation must warn that
the session's working files are involved and REQUIRE the user's approval, so they are
confident the destination device should receive them. The TUI runs the CLI's
``--dry-run --json`` preflight first; a non-empty manifest pushes this card, which shows
the destination device, the same summary the CLI prints (``network/workset_text``), and
a checkbox the user must tick before Move is enabled. Dismisses ``True`` only for an
explicit, ticked Move; Esc / Cancel answer ``False`` and nothing moves.

A manifest the owner could not read (``state: unknown``) or an owner too old to list it
(``state: unsupported``) has no consent to give from here: the card shows the reason
and only Cancel. The escape there is the CLI's explicit ``--without-working-set``.

Kept OUT of ``tui/session_move.py`` on purpose: that module is importable without
Textual (its tests reach the parse and the subprocess call without an app).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Static

from local_operator.i18n.keys import tui_session_move_workset as _keys
from local_operator.i18n.text import tr
from local_operator.network import workset_text

_t = tr("tui.session_move_workset")


class MoveWorkingSetConfirmScreen(ModalScreen[bool]):
    """Confirm (``True``) or cancel (``False``) moving with these working files."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, block: Mapping[str, Any], *, session_id: str, device: str) -> None:
        super().__init__()
        self._block = dict(block)
        self._session_id = session_id
        self._device = workset_text.device_label(self._block, device)
        state = str(self._block.get("state") or "")
        # Only a real manifest can be consented to; see the module docstring.
        self._consentable = state not in ("unknown", "unsupported")
        self._sends = bool(self._block.get("carries_files"))

    # -- copy ----------------------------------------------------------------

    def title_text(self) -> str:
        if self._sends:
            return _t(_keys.title_sends(device=self._device))
        return _t(_keys.title_stays())

    def summary_text(self) -> str:
        return "\n".join(
            workset_text.summary_lines(self._block, session=self._session_id, device=self._device)
        )

    def checkbox_text(self) -> str:
        if self._sends:
            return _t(_keys.checkbox_sends(device=self._device))
        return _t(_keys.checkbox_stays(source=workset_text.source_label(self._block, "?")))

    # -- layout --------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Container(classes="move-workset"):
            yield Static(self.title_text(), id="move-workset-title")
            yield Static(self.summary_text(), id="move-workset-summary")
            if self._consentable:
                yield Checkbox(self.checkbox_text(), value=False, id="move-workset-check")
            with Horizontal(id="move-workset-buttons"):
                if self._consentable:
                    # DISABLED UNTIL TICKED: the operator's requirement is an explicit
                    # approval, so Enter on a focused button cannot skip the checkbox.
                    yield Button(
                        _t(_keys.confirm()),
                        id="move-workset-confirm",
                        variant="warning",
                        disabled=True,
                    )
                yield Button(_t(_keys.cancel()), id="move-workset-cancel")

    def on_mount(self) -> None:
        # Focus the checkbox (or Cancel), never Move: the safe default is not moving.
        target = "#move-workset-check" if self._consentable else "#move-workset-cancel"
        self.query_one(target).focus()

    # -- behaviour -----------------------------------------------------------

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "move-workset-check":
            self.query_one("#move-workset-confirm", Button).disabled = not event.value

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "move-workset-confirm":
            ticked = self.query_one("#move-workset-check", Checkbox).value
            self.dismiss(bool(ticked))
        elif event.button.id == "move-workset-cancel":
            self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)

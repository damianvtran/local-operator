"""``/move … --to``: the working-set confirmation (docs/design/move-working-set.md §11.3).

WHY THIS SCREEN EXISTS (operator, 2026-11-18): a move mid-conversation must warn that
the session's working files are involved and REQUIRE the user's approval, so they are
confident the destination device should receive them. The TUI runs the CLI's
``--dry-run --json`` preflight first; a non-empty manifest pushes this card, which shows
the destination device, the same summary the CLI prints (``network/workset_text``), and
a checkbox the user must tick before Move is enabled.

THE ANSWER (``MoveDecision``): ``"confirm"`` only for an explicit, ticked Move;
``"without"`` only for the ticked "Move without the working files" of an ``unknown`` /
``unsupported`` card; ``""`` for Esc / Cancel, and nothing moves.

``unknown`` (the owner could not read the git state) and ``unsupported`` (the owner is
too old to list it) have no manifest to consent to, so the card offers the one way
forward the owner's gate accepts there — the explicit "leave every working file where
it is" (``--without-working-set``) — behind the same required checkbox (review D2/U2,
round 1: the card used to offer only Cancel, a dead end during every rollout).

LAYOUT CONTRACT (review D1/U1, round 1): the title, the checkbox and the buttons are
OUTSIDE the scrolling summary, so they are painted at any terminal size; only the
summary scrolls. At 80x24 with one worktree the old single-container card put Move
below the screen, and the user ticked and confirmed blind.

UNTRUSTED TEXT (security S2, round 1): every string here interpolates paths, branch
and device names from the owner's tree and the mesh. Each is rendered with
``markup=False`` — a ``[@click=screen.dismiss(True)]`` in a path or a remedy used to
be a one-click way past the checkbox, and ``[/]`` crashed the app.

Kept OUT of ``tui/session_move.py`` on purpose: that module is importable without
Textual (its tests reach the parse and the subprocess call without an app).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.content import Content
from textual.events import Resize
from textual.geometry import Size
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Static

from local_operator.i18n.keys import tui_session_move_workset as _keys
from local_operator.i18n.text import tr
from local_operator.network import workset_text

_t = tr("tui.session_move_workset")

MoveDecision = Literal["confirm", "without", ""]


#: The card's geometry, mirrored from ``local_operator.tcss`` (``.move-workset``):
#: width cap, share of the screen, vertical/horizontal padding, and the rows its
#: fixed parts take besides their text (the ``.gap-above`` blank rows above the
#: summary, the checkbox and the buttons, and a 3-row button). ``_fit`` turns them
#: into the summary's row budget.
_CARD_WIDTH = 86
_CARD_SHARE = 0.9
_CARD_PAD_Y = 2
_CARD_PAD_X = 4
_FIXED_ROWS = 1 + 1 + 1 + 3
#: The checkbox's own cells beside its label: ``padding: 0 1`` and the ``▐X▌ `` glyph.
_CHECK_CHROME_X = 2 + 4


class _WrappingCheckbox(Checkbox):
    """A Checkbox whose label WRAPS (review D4, round 1).

    Textual's ``ToggleButton.get_content_height`` is a constant 1 and its CSS is
    ``text-wrap: nowrap; text-overflow: ellipsis``, so at 60 columns the label read
    "I understand these working files stay on lap…" — cutting the device name the
    consent depends on. The sheet sets ``text-wrap: wrap``; this reports the height
    the wrapped label needs, so the row grows instead of truncating.
    """

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        return max(1, len(self.render().wrap(max(1, width))))


def _indent(line: str) -> tuple[int, str]:
    """``(depth, text)`` for one summary line: its leading spaces become a hanging
    indent (review D3), so a wrapped continuation stays under its own worktree."""
    stripped = line.lstrip(" ")
    return len(line) - len(stripped), stripped


class MoveWorkingSetConfirmScreen(ModalScreen[str]):
    """Answer ``"confirm"``, ``"without"`` or ``""`` (cancel); see the module docstring."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, block: Mapping[str, Any], *, session_id: str, device: str) -> None:
        super().__init__()
        self._block = dict(block)
        self._session_id = session_id
        self._device = workset_text.device_label(self._block, device)
        state = str(self._block.get("state") or "")
        # A real manifest is consented to; an unreadable or unlisted one can only be
        # left behind on purpose. A destination too old to forward the answer
        # (``update_device``) cannot be helped from here: Cancel only.
        unreadable = state in ("unknown", "unsupported")
        outdated_destination = isinstance(self._block.get("update_device"), dict)
        self._consentable = not unreadable
        self._without = unreadable and not outdated_destination
        self._sends = bool(self._block.get("carries_files"))

    # -- copy ----------------------------------------------------------------

    def title_text(self) -> str:
        if self._sends and self._consentable:
            return _t(_keys.title_sends(device=self._device))
        if not self._consentable and not self._without:
            # NOTHING TO ANSWER HERE (review D9): the outdated-destination card offers
            # only Cancel, so a question it cannot act on ("Move without the working
            # files?") misread as a choice. A statement instead.
            return _t(_keys.title_blocked())
        return _t(_keys.title_stays())

    def summary_lines(self) -> list[str]:
        # THE CARD HAS THE BUTTON, SO IT NAMES THE BUTTON (review U12, round 2): on an
        # ``unknown`` / ``unsupported`` card the summary used to tell the user to
        # re-run with --without-working-set, while the control for exactly that sat
        # an inch below it. The CLI's own text keeps the flag — it has no button.
        return workset_text.summary_lines(
            self._block,
            session=self._session_id,
            device=self._device,
            without_button=_t(_keys.confirm_without()) if self._without else "",
        )

    def summary_text(self) -> str:
        return "\n".join(self.summary_lines())

    def checkbox_text(self) -> str:
        source = workset_text.source_label(self._block, "?")
        if not self._consentable:
            return _t(_keys.checkbox_without(source=source))
        if self._sends:
            return _t(_keys.checkbox_sends(device=self._device))
        return _t(_keys.checkbox_stays(source=source))

    # -- layout --------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Container(classes="move-workset"):
            yield Static(self.title_text(), id="move-workset-title", markup=False)
            # SPACING IS `.gap-above`, the sheet's ONE sanctioned blank row
            # (tests/unit/tui/test_minimalism.py pins it as the only vertical margin):
            # the summary, the checkbox and the button row each open one row above
            # themselves, the same rows the old per-widget `margin-bottom`s drew.
            with VerticalScroll(id="move-workset-summary", classes="gap-above"):
                for line in self.summary_lines():
                    depth, text = _indent(line)
                    row = Static(Content(text), classes="move-workset-line", markup=False)
                    row.styles.padding = (0, 0, 0, depth)
                    yield row
            if self._consentable or self._without:
                # A Content label (not a str) is never parsed as markup (S2), and the
                # checkbox wraps (``height: auto`` in the sheet) so the device name
                # the consent depends on is never truncated (review D4).
                yield _WrappingCheckbox(
                    Content(self.checkbox_text()),
                    value=False,
                    id="move-workset-check",
                    classes="gap-above",
                )
            with Horizontal(id="move-workset-buttons", classes="gap-above"):
                if self._consentable:
                    # DISABLED UNTIL TICKED: the operator's requirement is an explicit
                    # approval, so Enter on a focused button cannot skip the checkbox.
                    yield Button(
                        _t(_keys.confirm()),
                        id="move-workset-confirm",
                        variant="warning",
                        disabled=True,
                    )
                elif self._without:
                    yield Button(
                        _t(_keys.confirm_without()),
                        id="move-workset-without",
                        variant="warning",
                        disabled=True,
                    )
                yield Button(_t(_keys.cancel()), id="move-workset-cancel")

    def on_mount(self) -> None:
        self._fit(self.app.size)
        # Focus the checkbox (or Cancel), never Move: the safe default is not moving.
        has_check = self._consentable or self._without
        target = "#move-workset-check" if has_check else "#move-workset-cancel"
        self.query_one(target).focus()

    def on_resize(self, event: Resize) -> None:
        self._fit(event.size)

    def _fit(self, size: Size) -> None:
        """Give the summary exactly the rows the card has left (review D1/U1).

        Measured, not ``1fr``: under an ``height: auto`` card a ``1fr`` child does
        not shrink (the card grew past its cap and painted Move off the screen), and
        a fixed-height card leaves a short summary floating in empty space. So the
        budget is the card's cap minus its fixed parts — title and checkbox measured
        at the card's actual inner width, since both may wrap — and the summary is
        ``height: auto`` up to that, scrolling beyond it. Computed before the first
        paint and on every resize, so no frame shows the controls off screen.

        THE PERCENTAGES RESOLVE AGAINST THE SCREEN'S CONTENT BOX, NOT THE APP (review
        D1, round 2). ``size`` is the whole screen, but the card's ``95%`` / ``90%``
        caps are taken from inside the modal screen's own padding (its gutter): at
        60x24 the card resolved to 55x19, while a budget from 60x24 assumed 57x21. One
        row too many went to the summary at every 24-row size, and at 60 columns the
        checkbox label measured at 47 columns fit one row but wrapped to two at the
        real 45 — pushing Move and Cancel out of the card. So the budget starts from
        exactly the box the card is laid out in.
        """
        gutter = self.styles.gutter
        width = max(1, size.width - gutter.width)
        height = max(1, size.height - gutter.height)
        inner = max(10, min(_CARD_WIDTH, int(width * 0.95)) - _CARD_PAD_X)
        rows = _CARD_PAD_Y + _FIXED_ROWS
        rows += len(Content(self.title_text()).wrap(inner))
        if self._consentable or self._without:
            label = Content(self.checkbox_text()).wrap(max(1, inner - _CHECK_CHROME_X))
            rows += len(label) + 1  # the blank row above it (.gap-above)
        budget = max(3, int(height * _CARD_SHARE) - rows)
        self.query_one("#move-workset-summary").styles.max_height = budget

    # -- behaviour -----------------------------------------------------------

    def _action_button(self) -> Button | None:
        for selector in ("#move-workset-confirm", "#move-workset-without"):
            found = self.query(selector)
            if found:
                return found.first(Button)
        return None

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "move-workset-check":
            button = self._action_button()
            if button is not None:
                button.disabled = not event.value

    def on_button_pressed(self, event: Button.Pressed) -> None:
        ticked = (
            bool(self.query_one("#move-workset-check", Checkbox).value)
            if (self._consentable or self._without)
            else False
        )
        if event.button.id == "move-workset-confirm":
            self.dismiss("confirm" if ticked else "")
        elif event.button.id == "move-workset-without":
            self.dismiss("without" if ticked else "")
        elif event.button.id == "move-workset-cancel":
            self.dismiss("")

    def action_cancel(self) -> None:
        self.dismiss("")

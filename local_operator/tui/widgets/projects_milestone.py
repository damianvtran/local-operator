"""The detail page's ADD MILESTONE prompt (LOH-19 P2): two fields, no mode of its own.

Why a pinned prompt rather than another full page: adding a milestone is one
name and an optional date, and the reader needs the plan's existing milestones
on screen to know what they are adding. A full-page form would hide the list
the decision is about. The prompt is a MODE of the projects page (``milestone``)
so `esc` pops one level, the canvas actions disarm, and the page's chrome keeps
its identity — but its body never leaves the detail page.

What this widget owns: the two controls, their labels/hints, local validation
against the STORE's ``MilestoneEdit``, and where a refusal is painted (under
the field it is about, never a toast). What it does NOT own: the write. The
collected ``(name, target_date)`` leaves through ``on_submit`` and the host
performs the write through ``ProjectRegistry.set_milestone`` — the same core
the tool and the routes use — then recomposes from the store. The page adds no
second write path (spec §10.5).

The field grammar is the CREATE FORM's (``FormFieldBlock``): a label, the
control, a quiet hint and a refusal row, with the app's one input treatment.
A second field-block implementation beside the existing one would be the way
the two forms come to disagree about where a refusal lands.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import ValidationError
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input

from local_operator.i18n.keys import tui_projects as _p
from local_operator.i18n.keys import tui_projects_edit as _x
from local_operator.i18n.text import tr
from local_operator.projects import MilestoneEdit
from local_operator.tui.projects_render import StyleFor
from local_operator.tui.widgets.projects_form import FormFieldBlock, _sentence

_t = tr("tui.projects_edit")
_tp = tr("tui.projects")


class MilestonePrompt(Vertical):
    """The add-milestone controls, pinned under the detail page.

    ``can_focus`` so the prompt's own bindings work when neither input has the
    caret; in practice :meth:`focus_first` puts the caret in the name field, and
    ``esc``/``tab`` reach the prompt from there through the input's own key
    handling (neither is an ``Input`` key, so they bubble).
    """

    can_focus = True

    BINDINGS = [
        Binding("tab", "focus_next_field", _t(_x.binding_next_field()), show=False),
        Binding("shift+tab", "focus_prev_field", _t(_x.binding_next_field()), show=False),
        Binding("escape", "cancel_request", _t(_x.binding_cancel()), show=False),
    ]

    def __init__(
        self,
        on_submit: Callable[[str, str], None],
        on_cancel: Callable[[], None],
        *,
        on_state_change: Callable[[], None] | None = None,
        style_for: StyleFor | None = None,
    ) -> None:
        super().__init__(classes="projects-milestone-prompt")
        self._on_submit = on_submit
        self._on_cancel = on_cancel
        self._on_state_change = on_state_change
        self._style_for = style_for
        self._name_input = Input(id="projects-milestone-name")
        self._date_input = Input(
            placeholder=_tp(_p.form_date_placeholder()),
            id="projects-milestone-date",
        )
        self._name_block = FormFieldBlock(
            _t(_x.prompt_name_label()),
            self._name_input,
            hint=_t(_x.prompt_name_hint()),
            gap=False,
            style_for=style_for,
        )
        self._date_block = FormFieldBlock(
            _t(_x.prompt_date_label()),
            self._date_input,
            hint=_t(_x.prompt_date_hint()),
            style_for=style_for,
        )

    # -- composition --------------------------------------------------------
    def compose(self):  # type: ignore[override]
        yield self._name_block
        yield self._date_block

    def focus_first(self) -> None:
        try:
            self._name_input.focus()
        except Exception:  # noqa: BLE001 — focus is a nicety
            pass

    def fields(self) -> list[Any]:
        """The focusable controls in TAB order (the prompt's own, not Textual's)."""
        return [self._name_input, self._date_input]

    def action_focus_next_field(self) -> None:
        self._step(1)

    def action_focus_prev_field(self) -> None:
        self._step(-1)

    def focus_prev_field(self) -> None:
        """Public, for the app's priority `shift+tab` delegation."""
        self._step(-1)

    def _step(self, delta: int) -> None:
        fields = self.fields()
        focused = self.app.focused
        try:
            index = fields.index(focused)
        except ValueError:
            index = -1 if delta > 0 else 0
        target = fields[(index + delta) % len(fields)]
        try:
            target.focus()
        except Exception:  # noqa: BLE001 — focus is a nicety
            pass

    # -- state --------------------------------------------------------------
    def clear(self) -> None:
        """Return to the empty prompt (a fresh `a`)."""
        self._name_input.value = ""
        self._date_input.value = ""
        self.clear_errors()

    def restyle(self, style_for: StyleFor | None) -> None:
        """Re-resolve label and refusal ink against a LIVE resolver (theme moves)."""
        self._style_for = style_for
        for block in (self._name_block, self._date_block):
            block.set_ink(style_for)

    def clear_errors(self) -> None:
        self._name_block.set_error("")
        self._date_block.set_error("")

    def on_input_changed(self, event: Input.Changed) -> None:
        """A refusal clears the moment the value it named changes."""
        block = self._name_block if event.input is self._name_input else self._date_block
        if block.error:
            block.set_error("")
        if self._on_state_change is not None:
            self._on_state_change()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter moves from the name to the date; Enter on the date submits."""
        if event.input is self._name_input:
            try:
                self._date_input.focus()
            except Exception:  # noqa: BLE001 — focus is a nicety
                pass
            return
        self.action_submit()

    # -- validation and submit ---------------------------------------------
    def action_submit(self) -> None:
        """Validate through the store's own ``MilestoneEdit``, then hand it up.

        The store's model is the validator, so the prompt cannot drift into a
        second vocabulary of refusals: an over-long name and a malformed date
        are refused with the sentences the tool would have used, painted under
        the field they are about.
        """
        self.clear_errors()
        name = self._name_input.value.strip()
        if not name:
            self._fail("name", _t(_x.prompt_name_required()))
            return
        # A BLANK optional date means "leave the stored one alone". Passing
        # ``target_date=None`` would put the key in ``model_fields_set``, and
        # the store reads a SUPPLIED null as a deliberate CLEAR (MilestoneEdit's
        # tri-state): naming an existing milestone with this field empty would
        # erase a date the reader never meant to touch. Only a typed date is
        # sent.
        fields: dict[str, Any] = {"name": name}
        date = self._date_input.value.strip()
        if date:
            fields["target_date"] = date
        try:
            edit = MilestoneEdit(**fields)
        except ValidationError as exc:
            first = exc.errors()[0]
            location = first.get("loc") or ()
            field = str(location[0]) if location else "target_date"
            self._fail(
                "date" if field == "target_date" else "name",
                _sentence(ValueError(first["msg"])),
            )
            return
        self._on_submit(edit.name, edit.target_date or "")

    def action_cancel_request(self) -> None:
        self._on_cancel()

    def _fail(self, field: str, sentence: str) -> None:
        block = self._name_block if field == "name" else self._date_block
        block.set_error(sentence)
        try:
            block.control.focus()
        except Exception:  # noqa: BLE001 — focus is a nicety
            pass

    # -- readbacks (tests) --------------------------------------------------
    def readback(self) -> list[str]:
        rows: list[str] = []
        for block in (self._name_block, self._date_block):
            rows.append(f"{block.field_label}: {block.control.value}")
            if block.hint:
                rows.append(block.hint)
            if block.error:
                rows.append(block.error)
        return rows

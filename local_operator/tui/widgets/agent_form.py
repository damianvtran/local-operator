"""Full-page profile authoring, sharing the projects form's controls and ink.

The host owns registry access. This page keeps a draft through refusals and
pins its discard question outside the scrolling fields, even on a 60×24 grid.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Sequence

from rich.text import Text
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Input, Static, TextArea

from local_operator.i18n.keys import tui_authoring as _x
from local_operator.i18n.text import tr
from local_operator.tui.agent_authoring import AgentDraft, AgentPlan, build_agent_plan
from local_operator.tui.projects_render import _styles
from local_operator.tui.widgets.projects_form import FormCycleRow, FormFieldBlock

_t = tr("tui.authoring")


class ProfileCycleRow(FormCycleRow):
    """Keep backend values stable while rendering translated choice labels."""

    def _repaint(self) -> None:
        labels = {
            "role": _x.role,
            "specialist": _x.specialist,
            "inherit": _x.inherit,
            "yes": _x.yes,
            "no": _x.no,
            "reactive": _x.reactive,
            "proactive": _x.proactive,
        }
        label = _t(labels[self.value]()) if self.value in labels else self.value
        self.update(Text(f"‹ {label} ›", style=self._ink))

    def _step(self, delta: int) -> None:
        super()._step(delta)
        # Mouse and keyboard must update the dependent controls identically.
        callback = getattr(self, "on_value_change", None)
        if callback is not None:
            callback()


class AgentFormPage(Vertical):
    BINDINGS = [
        Binding("ctrl+s", "save", show=False),
        Binding("tab", "next_field", show=False),
        Binding("shift+tab", "previous_field", show=False),
        Binding("escape", "cancel_request", show=False),
    ]

    def __init__(
        self,
        initial: AgentDraft | None,
        *,
        tiers: Sequence[str],
        on_submit: Callable[[AgentPlan, AgentDraft], None],
        on_cancel: Callable[[], None],
        read_only: Callable[[], bool],
        style_for: Any = None,
    ) -> None:
        super().__init__(id="agent-authoring", classes="projects-form-page")
        self.initial = initial
        self.opening = initial or AgentDraft()
        self.tiers = tuple(tiers)
        self._on_submit = on_submit
        self._on_cancel = on_cancel
        self._read_only = read_only
        self._confirm = False
        self._restore: Any = None
        self._ink = _styles(style_for)
        d = self.opening
        self.controls: dict[str, Any] = {
            "name": Input(d.name, id="agent-name", disabled=initial is not None),
            "kind": ProfileCycleRow(
                ("role", "specialist"), d.kind, id="agent-kind", style_for=style_for
            ),
            "label": Input(d.label, id="agent-label"),
            "description": Input(d.description, id="agent-description"),
            "instructions": TextArea(d.instructions, id="agent-instructions", tab_behavior="focus"),
            "tools": Input(d.tools, id="agent-tools"),
            "effort": ProfileCycleRow(
                tuple(dict.fromkeys(("inherit", *tiers, d.effort))),
                d.effort,
                id="agent-effort",
                style_for=style_for,
            ),
            "delegate": ProfileCycleRow(
                ("no", "yes"),
                "yes" if d.delegate else "no",
                id="agent-delegate",
                style_for=style_for,
            ),
            "action_class": ProfileCycleRow(
                ("reactive", "proactive"), d.action_class, id="agent-class", style_for=style_for
            ),
        }
        self.controls["kind"].disabled = initial is not None
        self.controls["kind"].on_value_change = self.sync_kind
        labels = {
            "name": _x.name,
            "kind": _x.kind,
            "label": _x.label,
            "description": _x.description,
            "instructions": _x.instructions,
            "tools": _x.tools,
            "effort": _x.effort,
            "delegate": _x.delegate,
            "action_class": _x.action_class,
        }
        hints = {
            "tools": _t(_x.tools_hint()),
            "instructions": _t(_x.instructions_hint()),
            "action_class": _t(_x.class_hint()),
        }
        self.blocks = {
            key: FormFieldBlock(
                _t(labels[key]()),
                control,
                hint=hints.get(key, ""),
                gap=key != "name",
                style_for=style_for,
            )
            for key, control in self.controls.items()
        }
        self._fields = VerticalScroll(classes="projects-form-fields")
        self._question = Static("", classes="projects-form-confirm", markup=False)
        self._question.can_focus = True
        # Translations may need a second line; the pinned question must never
        # truncate the keys that tell the reader how to keep their draft.
        self._question.styles.height = "auto"
        self._question.display = False
        self._receipt = Static("", classes="projects-form-error", markup=False)
        self._receipt.display = False

    def compose(self):
        yield Static(
            Text(
                _t(_x.edit(name=self.opening.name)) if self.initial else _t(_x.create()),
                style=self._ink("muted"),
            ),
            classes="projects-form-label",
        )
        with self._fields:
            yield from self.blocks.values()
        yield self._receipt
        yield self._question
        yield Static(_t(_x.legend()), classes="projects-form-hint", markup=False)

    def on_mount(self) -> None:
        self.sync_kind()
        self.call_after_refresh(self._focus_opening)

    def _focus_opening(self) -> None:
        self.controls["label" if self.initial else "name"].focus(scroll_visible=False)
        self._fields.scroll_home(animate=False, immediate=True)

    def sync_kind(self) -> None:
        """Hiding does not erase role policy: switching back restores the draft."""
        role = self.controls["kind"].value == "role"
        for key in ("tools", "effort", "delegate"):
            self.blocks[key].display = role
            self.controls[key].disabled = not role

    def on_key(self, event: Any) -> None:
        if self._confirm:
            event.stop()
            event.prevent_default()
            if event.key == "y":
                self._on_cancel()
            elif event.key in ("n", "escape", "enter"):
                self._confirm = False
                self._question.display = False
                if self._restore is not None:
                    self._restore.focus()
            return
        # Cycle rows change their value in their own binding; update visibility
        # only after that handler, rather than racing it with a key listener.
        if self.app.focused is self.controls["kind"]:
            self.call_after_refresh(self.sync_kind)

    def draft(self) -> AgentDraft:
        values: dict[str, Any] = {}
        for key, control in self.controls.items():
            values[key] = control.text if isinstance(control, TextArea) else control.value
        values["delegate"] = values["delegate"] == "yes"
        draft = AgentDraft(**values)
        # Hidden role controls are deliberately omitted, not silently persisted
        # on a specialist. The raw pure model still refuses illicit callers.
        if draft.kind == "specialist":
            draft = replace(draft, tools="", effort="inherit", delegate=False)
        return draft

    def _focus(self, delta: int) -> None:
        if self._confirm:
            return
        controls = [
            c for key, c in self.controls.items() if not c.disabled and self.blocks[key].display
        ]
        focused = self.app.focused
        index = controls.index(focused) if focused in controls else -1
        control = controls[(index + delta) % len(controls)]
        control.focus(scroll_visible=False)
        self.blocks[
            next(key for key, value in self.controls.items() if value is control)
        ].scroll_visible()

    def on_input_changed(self, message: Input.Changed) -> None:
        self._clear_error(message.input)

    def on_text_area_changed(self, message: TextArea.Changed) -> None:
        self._clear_error(message.text_area)

    def _clear_error(self, control: Any) -> None:
        for key, value in self.controls.items():
            if value is control:
                self.blocks[key].set_error("")
                self._receipt.display = False
                break

    def action_next_field(self) -> None:
        self._focus(1)

    def action_previous_field(self) -> None:
        self._focus(-1)

    def action_cancel_request(self) -> None:
        if self._confirm:
            self._confirm = False
            self._question.display = False
            if self._restore is not None:
                self._restore.focus()
            return
        if self.draft() == self.opening:
            self._on_cancel()
            return
        self._restore = self.app.focused
        self._confirm = True
        self._question.update(Text(_t(_x.discard()), style=self._ink("refusal")))
        self._question.display = True
        self._question.focus()

    def refuse_read_only(self) -> bool:
        """Use the projects page's exact ownership notice, including followers."""
        if not self._read_only():
            return False
        from local_operator.i18n.keys import tui_projects as project_keys

        self.receipt(tr("tui.projects")(project_keys.notice_read_only()))
        return True

    def receipt(self, sentence: str) -> None:
        self._receipt.update(sentence)
        self._receipt.display = True

    def action_save(self) -> None:
        if self._confirm or self.refuse_read_only():
            return
        for block in self.blocks.values():
            block.set_error("")
        draft = self.draft()
        plan = build_agent_plan(self.initial, draft, self.tiers)
        if plan.error_code:
            codes = {
                "invalid_name": _x.invalid_name,
                "required_instructions": _x.required_instructions,
                "invalid_tools": _x.invalid_tools,
                "invalid_effort": _x.invalid_effort,
                "specialist_fields": _x.specialist_fields,
                "description_clear_unsupported": _x.description_clear_unsupported,
            }
            block = self.blocks[plan.error_field]
            block.set_error(_t(codes[plan.error_code]()))
            block.control.focus()
            block.scroll_visible()
            return
        if not plan.changes:
            self.receipt(_t(_x.no_change()))
            return
        self._on_submit(plan, draft)

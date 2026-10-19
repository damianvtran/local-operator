"""Full-page team authoring, sharing the projects form's controls and ink.

The host owns registry access. This page keeps a draft through refusals and
pins its discard question outside the scrolling fields, even on a 60×24 grid.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from rich.text import Text
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Input, Static, TextArea

from local_operator.i18n.keys import tui_team_authoring as _x
from local_operator.i18n.text import tr
from local_operator.teams import TeamMember
from local_operator.tui.projects_render import _styles
from local_operator.tui.team_authoring import TeamDraft, TeamPlan, build_team_plan
from local_operator.tui.widgets.projects_form import FormFieldBlock

_t = tr("tui.team_authoring")


class TeamFormPage(Vertical):
    BINDINGS = [
        Binding("ctrl+s", "save", show=False),
        Binding("tab", "next_field", show=False),
        Binding("shift+tab", "previous_field", show=False),
        Binding("escape", "cancel_request", show=False),
    ]

    def __init__(
        self,
        initial: TeamDraft | None,
        *,
        original_members: Sequence[TeamMember] = (),
        on_submit: Callable[[TeamPlan, TeamDraft], None],
        on_cancel: Callable[[], object],
        read_only: Callable[[], bool],
        style_for: Any = None,
    ) -> None:
        super().__init__(id="team-authoring", classes="projects-form-page")
        self.initial = initial
        self.opening = initial or TeamDraft()
        self.original_members = tuple(original_members)
        self._on_submit = on_submit
        self._on_cancel = on_cancel
        self._read_only = read_only
        self._confirm = False
        self._restore: Any = None
        self._ink = _styles(style_for)
        d = self.opening
        self.controls: dict[str, Any] = {
            key: (
                TextArea(getattr(d, key), id=f"team-{key}", tab_behavior="focus")
                if key in ("instructions", "project")
                else Input(
                    getattr(d, key),
                    id=f"team-{key}",
                    disabled=key == "name" and initial is not None,
                )
            )
            for key in (
                "name",
                "label",
                "description",
                "manager",
                "members",
                "aliases",
                "instructions",
                "project",
            )
        }
        labels = {
            "name": _x.name,
            "label": _x.label,
            "description": _x.description,
            "manager": _x.manager,
            "members": _x.members,
            "aliases": _x.aliases,
            "instructions": _x.instructions,
            "project": _x.project,
        }
        hints = {
            "members": _t(_x.members_hint()) + "\n" + _t(_x.overrides_hint()),
            "aliases": _t(_x.aliases_hint()),
            "instructions": _t(_x.instructions_hint()),
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
        self._heading = Static("", classes="projects-form-label", markup=False)

    def _sync_heading(self) -> None:
        # The name is an unbreakable registry key. Rich wrapping would move it
        # into a clipped second row of this deliberately one-row page heading.
        title = Text(
            _t(_x.edit(name=self.opening.name)) if self.initial else _t(_x.create()),
            style=self._ink("muted"),
            no_wrap=True,
        )
        title.truncate(max(1, self.content_size.width), overflow="ellipsis")
        self._heading.update(title)

    def on_resize(self) -> None:
        self._sync_heading()

    def compose(self):
        yield self._heading
        with self._fields:
            yield from self.blocks.values()
        yield self._receipt
        yield self._question
        yield Static(_t(_x.legend()), classes="projects-form-hint", markup=False)

    def on_mount(self) -> None:
        self._sync_heading()
        self.call_after_refresh(self._focus_opening)

    def _focus_opening(self) -> None:
        self.controls["label" if self.initial else "name"].focus(scroll_visible=False)
        self._fields.scroll_home(animate=False, immediate=True)

    def on_key(self, event: Any) -> None:
        if self._confirm:
            event.stop()
            event.prevent_default()
            key = event.key.casefold()
            if key == "y":
                self._on_cancel()
            elif key in ("n", "escape", "enter"):
                self._confirm = False
                self._question.display = False
                if self._restore is not None:
                    self._restore.focus()
            return

    def draft(self) -> TeamDraft:
        return TeamDraft(
            **{
                key: control.text if isinstance(control, TextArea) else control.value
                for key, control in self.controls.items()
            }
        )

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

    def action_cancel_request(self, *, keep_confirm: bool = False) -> None:
        if self._confirm:
            # Esc may back out of the question; a repeated interrupt must leave
            # it owning the keyboard rather than silently changing the answer.
            if keep_confirm:
                return
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
        """Use projects' ownership predicate, with a team-specific notice."""
        if not self._read_only():
            return False
        self.receipt(_t(_x.notice_read_only()))
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
        plan = build_team_plan(self.initial, draft, self.original_members)
        if plan.error_code:
            codes = {
                "invalid_name": _x.invalid_name,
                "required_instructions": _x.required_instructions,
                "required_manager": _x.required_manager,
                "invalid_members": _x.invalid_members,
                "invalid_aliases": _x.invalid_aliases,
            }
            block = self.blocks[plan.error_field]
            sentence = _t(codes[plan.error_code]())
            block.set_error(sentence)
            # A wrapped hint can push the adjacent error below a narrow
            # viewport. Pin the same refusal outside that viewport as well;
            # scroll only after layout includes both newly visible receipts.
            self.receipt(sentence)
            block.control.focus()
            self.call_after_refresh(block.scroll_visible, animate=False)
            return
        if not plan.changes:
            self.receipt(_t(_x.no_change()))
            return
        self._on_submit(plan, draft)

"""The projects page's FORM state (S6d parity P4): create a project as a page.

Why a form at all, and why it is a MODE of the projects page rather than a
screen of its own: the spec's information architecture makes the form a third
state beside the canvases and the detail — `c` opens it, `esc` pops ONE level,
and the page's invariants (one page per screen, the greyed dock, `r`
recomposition, focus restore) hold there too. A screen would have been a second
way of being a full-page surface in this mode.

What this module owns, and what it deliberately does not:

* It owns the FIELD SET, the tab order, the local validation and where a
  refusal is painted — in the FORM, under the field it is about, never a toast
  over a form that has already closed (spec §7.7).
* It does NOT write. The collected values leave as the store's own
  :class:`~local_operator.projects.ProjectEdit` and the host performs the write
  through the same core the ``project`` tool and the slash verbs use
  (``registry.create_project``). The page holds no registry and adds no second
  write path (spec §10.5).
* Validation is the STORE's: the collected values are put through
  ``ProjectEdit``, whose validators are the ones the tool would have run, so
  the form cannot drift into a second vocabulary of refusals. The two rules the
  store does not carry — "the key is required" and the date range — are stated
  here, with the spec's copy.

The description scaffold is STARTER TEXT, not a default value: an untouched
scaffold is never submitted (see :meth:`ProjectsFormPage.collect`), so a
project created by someone who never touched the editor has NO description
rather than a body made of headings nobody wrote. It is text, not a cage —
selecting it all and deleting is how a reader clears it.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Sequence

from pydantic import ValidationError
from rich.text import Text
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Input, Static, TextArea

from local_operator.projects import (
    ATTRIBUTION_MAX,
    DESCRIPTION_MAX,
    ESTIMATE_MAX,
    PROJECT_STATUSES,
    TAGS_MAX,
    TITLE_MAX,
    ProjectEdit,
)
from local_operator.tui.projects_render import StyleFor, _styles

#: The starter text the description editor opens with on CREATE. Starter text,
#: not a value — see the module docstring.
DESCRIPTION_SCAFFOLD = "## Summary\n\n## Goals\n\n## Notes\n"

#: The estimate vocabularies the store accepts, in the order the cycle row
#: offers them.
ESTIMATE_UNITS: tuple[str, ...] = ("points", "days")

#: The confirm row a dirty `esc` shows (spec §7.7): one inline line, no modal.
DISCARD_PROMPT = "discard edits? · y discard · esc keep editing"

#: The form's footer line when it has no notice to carry. It states the one
#: rule a reader can otherwise only learn by being refused — the key is what
#: everything else references, and the title is what people read.
FORM_FOOTER_HINT = "the key is the reference handle; the title is what people read"

#: Pydantic wraps a validator's ``ValueError`` as ``"Value error, <sentence>"``.
#: The form paints the STORE's sentence (the one the tool would have shown a
#: model, minus the machine prefix a person has no use for).
_VALUE_ERROR_PREFIX = "Value error, "


def slug_for_title(title: str) -> str:
    """The key a title implies (spec §7.7): lowercase, runs collapsed, trimmed.

    One function rather than a rule inlined at the change event, because the
    key line, the submit path and the tests all have to agree on it, and the
    spec's rule is stated in exactly one place.
    """
    lowered = title.strip().lower()
    collapsed = re.sub(r"[^a-z0-9._-]+", "-", lowered)
    return re.sub(r"-{2,}", "-", collapsed).strip("-._")


def _sentence(exc: Exception) -> str:
    """One store refusal as a person-facing sentence."""
    text = str(exc)
    return text[len(_VALUE_ERROR_PREFIX) :] if text.startswith(_VALUE_ERROR_PREFIX) else text


class FormFieldBlock(Vertical):
    """One field: its label, its control, its hint and its refusal line.

    The hint and the error are separate rows so a refusal never overwrites the
    guidance, and both are hidden rather than blanked: an empty Static still
    occupies a row, which would put a hole under every field.
    """

    def __init__(
        self,
        label: str,
        control: Any,
        *,
        hint: str = "",
        gap: bool = True,
        style_for: StyleFor | None = None,
    ) -> None:
        # The blank lead-in rides the sheet's ONE sanctioned vertical-spacing
        # class rather than a margin of this widget's own — the detail page's
        # section headings take it the same way, and a second source is what
        # the minimalism guard's count exists to catch. The FIRST block is
        # given ``gap=False``: the form opens on its title, not a blank row.
        super().__init__(classes="projects-form-field gap-above" if gap else "projects-form-field")
        self.field_label = label
        self.control = control
        # The label wears the mode's own quiet ink rather than ``Static``'s
        # default foreground: every other label in this mode is a resolved
        # theme token, and a browser-default colour is a second vocabulary
        # beside them (and the only one a theme switch would miss).
        self._label = label
        self._refusal_ink = _styles(style_for)("refusal")
        self._label_widget = Static(
            Text(label, style=_styles(style_for)("muted")), classes="projects-form-label"
        )
        self._hint_text = hint
        # ``markup=False`` on both: a hint can name a field a reader typed into,
        # and a refusal sentence is a STORE message — Textual's ``Static`` parses
        # markup by default, so a bracket-carrying sentence would raise inside
        # the handler that paints it (QA round 1, Q-3).
        self._hint_widget = Static(hint, classes="projects-form-hint", markup=False)
        self._error_widget = Static("", classes="projects-form-error", markup=False)
        #: The refusal text, kept here rather than read back off the widget:
        #: Textual 8's ``Static`` exposes no ``renderable`` (measured — the
        #: attribute is gone), and the readback is what the tests assert on.
        self._error_text = ""
        self._hint_widget.display = bool(hint)
        self._error_widget.display = False

    def compose(self):  # type: ignore[override]
        yield self._label_widget
        yield self.control
        yield self._hint_widget
        yield self._error_widget

    def set_hint(self, hint: str) -> None:
        self._hint_text = hint
        self._hint_widget.update(hint)
        self._hint_widget.display = bool(hint)

    def set_ink(self, style_for: StyleFor | None) -> None:
        """Re-resolve this field's label ink against a LIVE resolver.

        Called on every chrome paint, so a `/theme` switch reaches the labels
        exactly as it reaches every other surface in this mode — the resolver is
        rebuilt per repaint, and a Style captured at construction would be the
        old palette (the detail page's captured-resolver defect, UX round 1,
        U2).
        """
        self._refusal_ink = _styles(style_for)("refusal")
        self._label_widget.update(Text(self._label, style=_styles(style_for)("muted")))
        if self._error_text:
            # The refusal line re-inks with its label (a theme switch must not
            # leave it in the old palette while the guidance moves).
            self._error_widget.update(Text(self._error_text, style=self._refusal_ink))

    def set_error(self, text: str) -> None:
        # A refusal is a RECEIPT, not guidance: it wears the warning ink the
        # app's other refusal sentences wear, so a reader can never mistake the
        # line under a field for a hint about it (design review round 1, D4).
        self._error_text = text
        self._error_widget.update(Text(text, style=self._refusal_ink))
        self._error_widget.display = bool(text)

    @property
    def error(self) -> str:
        return self._error_text

    @property
    def hint(self) -> str:
        """The field's quiet second line, as painted (empty when hidden)."""
        return self._hint_text


class FormCycleRow(Static):
    """A ``‹ value ›`` field: ←/→ cycle it, a click cycles it (spec §7.7).

    Its own widget rather than a ``Select``: the spec's row is a value stepped
    in place — no dropdown, no overlay, no second way for a full-page mode to
    cover itself — and the fixed order is the store's own status vocabulary, so
    the row can never offer a status the store would refuse.
    """

    can_focus = True

    BINDINGS = [
        Binding("left", "previous", "Previous value", show=False),
        Binding("right", "next", "Next value", show=False),
    ]

    def __init__(
        self,
        options: Sequence[str],
        value: str,
        *,
        id: str | None = None,
        style_for: StyleFor | None = None,
    ) -> None:
        super().__init__(classes="projects-form-cycle", id=id)
        self._options = tuple(options)
        self._value = value if value in self._options else self._options[0]
        # The same resolver discipline as every other surface in this mode: the
        # row paints the theme's MUTED token, never a hex of its own, so a theme
        # switch reaches it (the detail page's captured-resolver defect, U2).
        self._ink = _styles(style_for)("muted")

    @property
    def value(self) -> str:
        return self._value

    def set_ink(self, style_for: StyleFor | None) -> None:
        """Re-resolve the row's ink against a LIVE resolver (see `set_ink`)."""
        self._ink = _styles(style_for)("muted")
        self._repaint()

    def set_value(self, value: str) -> None:
        self._value = value if value in self._options else self._options[0]
        self._repaint()

    def _step(self, delta: int) -> None:
        index = (self._options.index(self._value) + delta) % len(self._options)
        self._value = self._options[index]
        self._repaint()

    def action_previous(self) -> None:
        self._step(-1)

    def action_next(self) -> None:
        self._step(1)

    def on_mount(self) -> None:
        self._repaint()

    def on_click(self, event: Any) -> None:
        event.stop()
        self.focus()
        self._step(1)

    def _repaint(self) -> None:
        self.update(Text(f"‹ {self._value} ›", style=self._ink))

    def readback(self) -> str:
        return f"‹ {self._value} ›"


class ProjectsFormPage(Vertical):
    """The create form: the fields, their validation, and the refusal rows.

    Everything the form needs to be honest lives here; everything it needs to
    WRITE lives in the host, which it reaches only through ``on_submit``.

    The FIELDS scroll and the confirm does not: the discard question is pinned
    under them, outside the scroll body. It used to be the last child of a
    58-row scrolled page, so at 80x24 it painted at y=62 while the viewport sat
    at scroll 0 — the reader was asked a question they could not see (design
    review round 1, D1 / QA round 1, Q-1). A pinned row also means the question
    never moves the content it is asked about.
    """

    can_focus = True

    BINDINGS = [
        # `ctrl+s` is the app's remappable "resume" hotkey (keymap.py). This
        # binding is the CHILD's and wins while a field or the page has focus,
        # which is the whole time the form is up: a full-page form owns its
        # keys, and the hotkey returns the moment the form closes. Stated
        # because it is a real (if intentional) shadowing of an app key.
        Binding("ctrl+s", "save", "Save", show=False),
        Binding("tab", "focus_next_field", "Next field", show=False),
        # SHIFT+TAB never reaches this map from a keypress: the app binds it as
        # a PRIORITY binding (`cycle_effort`) and Textual matches those before
        # the focused widget. It is listed anyway because this widget's own
        # default for a focus key must be "previous field" — if that app
        # binding is ever narrowed or moved, the honest behaviour is already
        # here rather than one forgotten line away (the key prompt's rule).
        Binding("shift+tab", "focus_prev_field", "Previous field", show=False),
        Binding("escape", "cancel_request", "Cancel", show=False),
    ]

    def __init__(
        self,
        on_submit: Callable[[ProjectEdit], None],
        on_cancel: Callable[[], None],
        *,
        on_state_change: Callable[[], None] | None = None,
        known_teams: Iterable[str] = (),
        style_for: StyleFor | None = None,
    ) -> None:
        super().__init__(classes="projects-form-page")
        self._on_submit = on_submit
        self._on_cancel = on_cancel
        self._on_state_change = on_state_change
        self._style_for = style_for
        self._known_teams = tuple(known_teams)
        #: True while the key follows the title. Editing the key detaches it;
        #: clearing it re-attaches (spec §7.7).
        self._key_follows = True
        #: The value the PAGE last wrote into the key field. The change event
        #: that write raises arrives on the next message pump, so a transient
        #: "writing" flag would already be down by then and the write would be
        #: mistaken for the reader typing in the field — measured: the key kept
        #: only the first character of a typed title. Comparing the VALUE makes
        #: the guard immune to message timing, which is what the defect was.
        self._key_written = ""
        #: The confirm row is up and owns the keyboard.
        self._confirm = False
        #: The field focus returns to when the confirm is cleared.
        self._focus_before_confirm: Any | None = None
        #: The state a cancel would return to — armed by the host once it has
        #: set the initial values (``None`` until then, which reads as clean).
        self._initial: tuple[str, ...] | None = None

        self._title_input = Input(placeholder="a human-readable name", id="projects-form-title")
        self._key_input = Input(placeholder="the reference handle", id="projects-form-key")
        self._description = TextArea(id="projects-form-description")
        self._status_row = FormCycleRow(
            PROJECT_STATUSES, "planning", id="projects-form-status", style_for=style_for
        )
        self._tags_input = Input(placeholder="comma or space separated", id="projects-form-tags")
        self._team_input = Input(placeholder="team", id="projects-form-team")
        self._owner_input = Input(placeholder="owner", id="projects-form-owner")
        self._target_input = Input(placeholder="YYYY-MM-DD", id="projects-form-target")
        self._start_input = Input(placeholder="YYYY-MM-DD", id="projects-form-start")
        self._estimate_input = Input(placeholder="points", id="projects-form-estimate")
        self._unit_row = FormCycleRow(
            ESTIMATE_UNITS, "points", id="projects-form-unit", style_for=style_for
        )

        #: The fields live in their own scroll container (see the class
        #: docstring): the page itself does not scroll, so the confirm row can
        #: be pinned under it.
        self._fields_scroll = VerticalScroll(classes="projects-form-fields")
        self._key_block = FormFieldBlock(
            "key",
            self._key_input,
            hint="follows the title until you edit it",
            style_for=style_for,
        )
        self._title_block = FormFieldBlock(
            "title", self._title_input, gap=False, style_for=style_for
        )
        self._description_block = FormFieldBlock(
            "description",
            self._description,
            hint="starter text — left as it is, nothing is saved",
            style_for=style_for,
        )
        self._status_block = FormFieldBlock(
            "status", self._status_row, hint="← → change", style_for=style_for
        )
        self._tags_block = FormFieldBlock(
            "tags", self._tags_input, hint=self._tags_hint(), style_for=style_for
        )
        self._team_block = FormFieldBlock(
            "team", self._team_input, hint=self._teams_hint(), style_for=style_for
        )
        self._owner_block = FormFieldBlock("owner", self._owner_input, style_for=style_for)
        self._target_block = FormFieldBlock("target date", self._target_input, style_for=style_for)
        self._start_block = FormFieldBlock("start date", self._start_input, style_for=style_for)
        self._estimate_block = FormFieldBlock("estimate", self._estimate_input, style_for=style_for)
        self._unit_block = FormFieldBlock(
            "estimate unit", self._unit_row, hint="← → change", style_for=style_for
        )
        # ``markup=False``: a refusal sentence is DATA — a store message can
        # carry brackets — and Textual's ``Static`` parses markup by default,
        # so a bracketed sentence would raise inside the handler painting it
        # (QA round 1, Q-3).
        # `gap-above` is the sheet's ONE sanctioned blank row: the question
        # must not read as the value of whichever field the viewport clipped
        # (design review round 1 r2, D10).
        self._confirm_row = Static(
            DISCARD_PROMPT, classes="projects-form-confirm gap-above", markup=False
        )
        self._confirm_row.display = False

    # -- composition --------------------------------------------------------
    def compose(self):  # type: ignore[override]
        # TITLE FIRST (the operator's ask, and the spec's row order): the title
        # is what a person has in mind when they reach for `c`, and the key is
        # derived from it rather than the other way round. START precedes
        # TARGET: the pair reads in the direction a plan does (design review
        # round 1, N1).
        with self._fields_scroll:
            yield self._title_block
            yield self._key_block
            yield self._description_block
            yield self._status_block
            yield self._tags_block
            yield self._team_block
            yield self._owner_block
            yield self._start_block
            yield self._target_block
            yield self._estimate_block
            yield self._unit_block
        yield self._confirm_row

    def on_mount(self) -> None:
        self._description.text = DESCRIPTION_SCAFFOLD
        # The scaffold is placed after mount (a TextArea given text before
        # layout is a documented churn) and the caret lands on the Goals line —
        # the heading a writer fills in first (spec §7.7).
        try:
            self._description.move_cursor((2, 0))
        except Exception:  # noqa: BLE001 — caret placement is a nicety
            pass
        self.focus_first()

    # -- fields -------------------------------------------------------------
    def reset(self) -> None:
        """Return every field to its create-time state (a fresh `c`).

        Called on every entry: a form opened after an abandoned one must not
        inherit the refused name or a half-typed date. The scaffold goes back
        in and the caret with it, which is the state :meth:`on_mount` leaves —
        stated once here so a re-entry and a first entry cannot drift.
        """
        self._key_follows = True
        self._key_written = ""
        self._title_input.value = ""
        self._key_input.value = ""
        self._description.text = DESCRIPTION_SCAFFOLD
        self._status_row.set_value(PROJECT_STATUSES[0])
        self._tags_input.value = ""
        self._team_input.value = ""
        self._owner_input.value = ""
        self._target_input.value = ""
        self._start_input.value = ""
        self._estimate_input.value = ""
        self._unit_row.set_value(ESTIMATE_UNITS[0])
        self.clear_errors()
        self.disarm_confirm()
        self._key_block.set_hint("follows the title until you edit it")
        self._tags_block.set_hint(self._tags_hint())
        try:
            self._description.move_cursor((2, 0))
        except Exception:  # noqa: BLE001 — caret placement is a nicety
            pass
        # Un-armed until the host takes the baseline it will return to.
        self._initial = None

    def fields(self) -> list[Any]:
        """The focusable controls in TAB ORDER (the form's own, not Textual's)."""
        return [
            self._title_input,
            self._key_input,
            self._description,
            self._status_row,
            self._tags_input,
            self._team_input,
            self._owner_input,
            self._start_input,
            self._target_input,
            self._estimate_input,
            self._unit_row,
        ]

    def restyle(self, style_for: StyleFor | None) -> None:
        """Re-resolve every field's ink against a LIVE resolver.

        The view calls this on each chrome paint, so the form follows a
        ``/theme`` switch the way every other surface in this mode does. A
        resolver captured once at construction would be the OLD palette — the
        exact defect the detail page shipped and UX round 1 caught (U2).
        """
        for block in self._blocks().values():
            block.set_ink(style_for)
        self._status_row.set_ink(style_for)
        self._unit_row.set_ink(style_for)

    def arm_current(self) -> None:
        """Take the baseline a cancel returns to, from the state ON SCREEN.

        Called by the host once the form is up and its scaffold placed, so
        "dirty" means "the reader changed something" rather than "the scaffold
        differs from an empty form" — an untouched create closes on `esc`
        without asking about edits nobody made (spec §7.7).
        """
        self._initial = self.snapshot()

    def scroll_fields_home(self) -> None:
        """Put the fields back at their first row (the page no longer scrolls)."""
        self._fields_scroll.scroll_home(animate=False)

    def focus_first(self) -> None:
        try:
            self._title_input.focus()
        except Exception:  # noqa: BLE001 — focus is a nicety
            pass

    def action_focus_next_field(self) -> None:
        self._focus_step(1)

    def action_focus_prev_field(self) -> None:
        self._focus_step(-1)

    def focus_prev_field(self) -> None:
        """Public, for the app's priority `shift+tab` delegation."""
        self._focus_step(-1)

    def _focus_step(self, delta: int) -> None:
        if self._confirm:
            # The confirm OWNS the keyboard (its docstring, and the spec's
            # "inline confirm"): moving the focus into a field would hand the
            # advertised `y` to that field as text — measured, the reader's
            # next `y` typed itself into the title instead of discarding.
            return
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

    # -- key following the title -------------------------------------------
    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input is self._title_input and self._key_follows:
            self._write_key(slug_for_title(event.value))
        elif event.input is self._key_input:
            if event.value == self._key_written:
                # Our own slug write, not the reader — see ``_key_written``.
                pass
            elif not event.value.strip():
                # Cleared: follow the title again (spec §7.7).
                self._key_follows = True
                self._write_key(slug_for_title(self._title_input.value))
            else:
                self._key_follows = False
                self._sync_key_hint()
        if event.input is self._tags_input:
            self._tags_block.set_hint(self._tags_hint())
        if self._on_state_change is not None:
            self._on_state_change()

    def _write_key(self, value: str) -> None:
        self._key_written = value
        self._key_input.value = value
        self._sync_key_hint()

    def _sync_key_hint(self) -> None:
        """State whether the key is following the title.

        ONE place decides the wording, and both edges call it: the write path
        AND the detach edge — the hint stayed on "follows the title until you
        edit it" after the reader had edited the key, because only the write
        path set it (agent review round 1, R1-3).
        """
        self._key_block.set_hint(
            "follows the title until you edit it"
            if self._key_follows
            # How to get the behaviour BACK is the half a reader cannot guess
            # (design review round 1, N2).
            else "set by hand — clear it to follow the title again"
        )

    def on_text_area_changed(self, event: Any) -> None:
        if self._on_state_change is not None:
            self._on_state_change()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """``enter`` in a single-line field moves on; on the LAST one it saves."""
        fields = self.fields()
        try:
            index = fields.index(event.input)
        except ValueError:
            return
        if event.input is self._estimate_input:
            self.action_save()
            return
        target = fields[index + 1]
        try:
            target.focus()
        except Exception:  # noqa: BLE001 — focus is a nicety
            pass

    # -- hints --------------------------------------------------------------
    def _tags_hint(self) -> str:
        # The separator is already the field's PLACEHOLDER; repeating it here
        # said the same thing twice in one row (design review round 1 r2, D11).
        return f"at most {TAGS_MAX} tags"

    def _teams_hint(self) -> str:
        if not self._known_teams:
            return "no teams registered — free text is accepted"
        joined = ", ".join(self._known_teams[:6])
        suffix = " …" if len(self._known_teams) > 6 else ""
        return f"known: {joined}{suffix}"

    def set_known_teams(self, teams: Iterable[str]) -> None:
        self._known_teams = tuple(teams)
        self._team_block.set_hint(self._teams_hint())

    # -- values -------------------------------------------------------------
    def _tags(self) -> list[str]:
        return [part for part in re.split(r"[,\s]+", self._tags_input.value.strip()) if part]

    def _description_value(self) -> str | None:
        """The description, or ``None`` while the scaffold is untouched."""
        text = self._description.text
        return None if text == DESCRIPTION_SCAFFOLD else text

    def _estimate_value(self) -> str:
        return self._estimate_input.value.strip()

    def collect(self) -> ProjectEdit | None:
        """Validate and build the store's edit, or paint the refusals.

        The values go through :class:`ProjectEdit` — the store's own model, so
        the sentences are the store's — and the first refusal is answered where
        a person can act on it: under its field, with the caret in it.
        """
        self.clear_errors()
        raw: dict[str, Any] = {
            "name": self._key_input.value.strip(),
            "title": self._title_input.value.strip() or None,
            "owner": self._owner_input.value.strip() or None,
            "team": self._team_input.value.strip() or None,
            "status": self._status_row.value,
            "tags": self._tags(),
            "start_date": self._start_input.value.strip() or None,
            "target_date": self._target_input.value.strip() or None,
        }
        description = self._description_value()
        if description is not None:
            raw["description"] = description
        estimate_text = self._estimate_value()
        if estimate_text:
            try:
                raw["estimate"] = float(estimate_text)
            except ValueError:
                self._fail("estimate", "estimate must be a number")
                return None
            raw["estimate_unit"] = self._unit_row.value
        if not raw["name"]:
            self._fail(
                "key",
                "the key is required — it is the reference handle, e.g. `parity-spec`",
            )
            return None
        try:
            edit = ProjectEdit(**raw)
        except ValidationError as exc:
            first = exc.errors()[0]
            field = str(first["loc"][0]) if first.get("loc") else "title"
            self._fail(field, _sentence(ValueError(first["msg"])))
            return None
        except ValueError as exc:  # a bare store refusal
            self._fail("key", _sentence(exc))
            return None
        if not self._dates_ordered():
            return None
        return edit

    def _dates_ordered(self) -> bool:
        """The one date rule the store does not carry (spec §7.7's copy)."""
        start = self._start_input.value.strip()
        target = self._target_input.value.strip()
        if start and target and target < start:
            self._fail("target_date", "The target date is before the start date.")
            return False
        return True

    def _blocks(self) -> dict[str, FormFieldBlock]:
        return {
            "name": self._key_block,
            "key": self._key_block,
            "title": self._title_block,
            "description": self._description_block,
            "status": self._status_block,
            "tags": self._tags_block,
            "team": self._team_block,
            "owner": self._owner_block,
            "target_date": self._target_block,
            "start_date": self._start_block,
            "estimate": self._estimate_block,
            "estimate_unit": self._unit_block,
        }

    def _fail(self, field: str, sentence: str) -> None:
        block = self._blocks().get(field, self._title_block)
        block.set_error(sentence)
        control = block.control
        try:
            control.focus()
        except Exception:  # noqa: BLE001 — focus is a nicety
            pass

    def show_refusal(self, sentence: str) -> None:
        """A refusal from the WRITE (a name conflict, the schema guard).

        Painted under the title, where the spec puts it: the form stays up and
        keeps everything typed, so the reader can change the name and press
        `ctrl+s` again without losing the description.
        """
        self._title_block.set_error(sentence)
        # The sentence lands under the TITLE, so that is where the cursor goes:
        # a receipt that points at one field while the cursor sits in another is
        # half a receipt, and the reader's next act is to fix what it names
        # (design review round 1, D5).
        try:
            self._title_input.focus()
        except Exception:  # noqa: BLE001 — focus is a nicety
            pass

    def clear_errors(self) -> None:
        for block in self._blocks().values():
            block.set_error("")

    # -- dirty state, cancel, confirm ---------------------------------------
    def snapshot(self) -> tuple[str, ...]:
        """Everything a submit would send, as text — the dirty comparison's basis."""
        return (
            self._title_input.value,
            self._key_input.value,
            self._description.text,
            self._status_row.value,
            self._tags_input.value,
            self._team_input.value,
            self._owner_input.value,
            self._target_input.value,
            self._start_input.value,
            self._estimate_input.value,
            self._unit_row.value,
        )

    @property
    def dirty(self) -> bool:
        if self._initial is None:
            return False
        return self.snapshot() != self._initial

    def arm(self, initial: tuple[str, ...]) -> None:
        """Remember the state a cancel would return to (the form's baseline)."""
        self._initial = initial

    @property
    def confirming(self) -> bool:
        return self._confirm

    def action_cancel_request(self) -> None:
        if self._confirm:
            self.disarm_confirm()
            return
        if not self.dirty:
            self._on_cancel()
            return
        self._confirm = True
        self._focus_before_confirm = self.app.focused
        self._confirm_row.display = True
        try:
            self.focus()
        except Exception:  # noqa: BLE001 — focus is a nicety
            pass
        if self._on_state_change is not None:
            self._on_state_change()

    def action_discard(self) -> None:
        """``y`` — only while the confirm row is up (spec §7.7)."""
        if not self._confirm:
            return
        self.disarm_confirm()
        self._on_cancel()

    def disarm_confirm(self) -> None:
        self._confirm = False
        self._confirm_row.display = False
        target = self._focus_before_confirm
        self._focus_before_confirm = None
        if target is not None:
            try:
                target.focus()
            except Exception:  # noqa: BLE001 — focus is a nicety
                pass
        if self._on_state_change is not None:
            self._on_state_change()

    def on_key(self, event: Any) -> None:
        """The confirm row owns the keyboard while it is up (spec §7.7).

        ``y`` and ``esc`` are the two ways out and both are on this block: with
        the confirm up, a printable key must not reach the field underneath it
        — the reader is answering a question, not editing (the key prompt's
        rule, same reason). Everything else printable is swallowed rather than
        left to bubble, so no other letter is interpreted as a meaning either.
        """
        if not self._confirm:
            return
        if event.key == "y":
            event.stop()
            event.prevent_default()
            self.action_discard()
            return
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self.disarm_confirm()
            return
        if event.is_printable:
            event.stop()
            event.prevent_default()

    # -- save ---------------------------------------------------------------
    def action_save(self) -> None:
        if self._confirm:
            return
        edit = self.collect()
        if edit is None:
            return
        self._on_submit(edit)

    # -- readbacks (tests and geometry probes) ------------------------------
    def readback(self) -> list[str]:
        rows: list[str] = []
        # ``_blocks`` maps TWO field names onto the key block (the store calls
        # its handle ``name``; the form calls it ``key``), so the rows are built
        # from the ordered UNIQUE blocks or that field was read back twice
        # (agent review round 1, R1-4).
        for block in dict.fromkeys(self._blocks().values()):
            rows.append(f"{block.field_label}: {self._control_text(block)}")
            # The hint is part of what the reader SEES, so it is part of what
            # the readback states: a surface a test cannot assert is a surface
            # nobody can pin (P3's U6, the same reason).
            if block.hint:
                rows.append(f"{block.field_label} hint: {block.hint}")
            if block.error:
                rows.append(f"{block.field_label} error: {block.error}")
        if self._confirm:
            rows.append(DISCARD_PROMPT)
        return rows

    def _control_text(self, block: FormFieldBlock) -> str:
        control = block.control
        if isinstance(control, Input):
            return control.value
        if isinstance(control, TextArea):
            return control.text
        if isinstance(control, FormCycleRow):
            return control.readback()
        return ""

    # -- limits the caps put on the editor ---------------------------------
    def max_lengths(self) -> dict[str, int]:
        """The store's caps, for the tests and for a future live counter."""
        return {
            "title": TITLE_MAX,
            "description": DESCRIPTION_MAX,
            "owner": ATTRIBUTION_MAX,
            "team": ATTRIBUTION_MAX,
            "estimate": int(ESTIMATE_MAX),
        }

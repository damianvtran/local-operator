"""The full-page ``/schedule`` view: THIS conversation's scheduled wakes.

A MODE of the main screen, cloned from :class:`HarnessImportView` and, through
it, :class:`SettingsView`: the page takes the transcript's region, leaves the
dock in place and inert, and owns its own cursor and keys. The layout reuses the
settings page's CSS classes (title/rule/columns/body/pane/detail/hints) rather
than adding a second full-page skeleton, and the create/edit state is the
projects page's page-as-form pattern — one ``esc`` pops one level.

WHAT THIS PAGE OWNS, AND WHAT IT DOES NOT. It lists the LIVE scheduler's rows
(``session.wake_scheduler.schedules``) through the shared human-surface filter,
and every mutation goes through ``session.set_wake_schedules`` — the session's
one writer, which persists and re-arms. It never reads a transcript, never
writes an index file and never arms a timer itself. The machine-wide wake index
(``wakes.store``) is deliberately not read: this page is a conversation's own
ledger, and another session's rows belong to another runtime.

A FULL-PAGE LIST CLAMPS ITS ARROWS, the documented exception shared with
``/settings`` and ``/project show``. A wake list is a ledger a reader scans, not
a picker circling a short overlay, so Home and End are the deliberate ends.

ROWS ARE TWO LINES, ALWAYS. The head names the handle and the prompt; the meta
line names the due phrase, the cadence, the delivered count, what is left and
whether the completion notifies. Fixed height is what makes the cursor's
scroll-into-view arithmetic exact (``index * 2``) without the variable-height
machinery ``/review`` needs, and it is why the head is clipped rather than
allowed to wrap: a wrapped head would move every row below it.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Input, Static, TextArea

from local_operator.i18n import keys as i18n_keys
from local_operator.i18n.text import tr
from local_operator.paths import config_dir
from local_operator.tui import schedule as schedule_model
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.projects_form import FormCycleRow, FormFieldBlock
from local_operator.tui.widgets.subagent_view import HintButton
from local_operator.tui.widgets.tool_card import truncate_cells
from local_operator.wakes.install import supervisor_state

_t = tr("tui.schedule")
_k = i18n_keys.tui_schedule

#: Chrome, not copy: the page's cursor and its rule are glyphs no locale
#: translates, and the extraction ratchet is told the same by review_view.
_CURSOR = "›"
_RULE_GLYPH = "─"

#: The line a dirty form would need to ask about; the page deliberately does
#: not (see ``ScheduleForm.action_cancel``): a wake is a sentence or two, and
#: the list it returns to is one keystroke away.
_INDENT = "  "

#: The FLOOR of the pane's label column. The actual column is measured from the
#: row's own translated labels per paint (``_detail_column``): a fixed value is
#: an English-only assumption — Spanish ``Ejecuciones`` is 11 cells and
#: ``Siguiente`` is 9, so a hardcoded 9 with a bare ``ljust`` glued the label to
#: its value in the shipped translations. 9 stays the floor so the English pane
#: keeps its historical left edge.
_MIN_LABEL_COLUMN = 9


class _ScheduleBody(ScrollableContainer):
    """The list body, which tells the page when its WIDTH settles.

    The page truncates each row to the body's measured width. The body's width
    is not the page's: it loses the detail pane's 42 cells and the scrollbar,
    and it settles a layout pass after the page mounts — with no Resize on the
    PAGE itself. Without this hook the first paint kept the 80-column fallback
    and a long prompt WRAPPED in a 54-cell column (measured at 100x30, the
    layout defect this class exists for): a wrapped head is a third line, which
    breaks the page's fixed two-line row contract and its cursor arithmetic.
    """

    def __init__(self, *children: Any, on_settle: Callable[[], None], **kwargs: Any) -> None:
        super().__init__(*children, **kwargs)
        self._on_settle = on_settle

    def on_resize(self) -> None:
        self._on_settle()


class ScheduleViewDismissed(Message):
    """The page's esc hint/key was pressed; the app owns teardown."""


def _ink(key: str) -> Style:
    """The two inks this page uses, from the app's semantic ramp.

    The same keys the projects form's field blocks consume (``muted`` labels,
    ``refusal`` errors); an unknown key falls back to plain ink rather than
    raising inside a paint, which is what ``_styles`` already promises.
    """
    token = "warning" if key == "refusal" else key
    try:
        return Style(color=theme_mod.semantic_color(token))
    except KeyError:  # pragma: no cover - the ramp carries both
        return Style()


def _selection_ground() -> Style:
    """The app's one selection ground (sidebar, pickers, ToolCard)."""
    try:
        return Style(bgcolor=theme_mod.semantic_color("tint-select"))
    except KeyError:  # pragma: no cover - the ramp carries it
        return Style()


def _meta_ink(selected: bool) -> str:
    """The ink token a row's meta line reads with.

    SELECTED META USES ``fg``, NOT ``muted``. The ``muted``-on-``tint-select``
    pair inherits palette debt the page does not own (everforest measured
    3.58:1 in round-1 design D3, below AA for this text), and the meta line is
    the half of the selected row that carries due, cadence and fired count —
    the information the selection exists to expose. One rung brighter keeps the
    hierarchy (the head stays bold) and clears AA in every shipped palette.
    """
    return "fg" if selected else "muted"


class ScheduleForm(Vertical):
    """Create/edit as a page state: five fields, one refusal line, no second writer.

    The refusal is ONE line under the fields rather than a per-field row, and
    that is deliberate: ``harness.wake``'s builders return one sentence, not a
    field attribution, and guessing which field a sentence is "about" would be
    a second vocabulary the validator never agreed to. The line sits directly
    under the controls it is about.
    """

    can_focus = True

    BINDINGS = [
        # The same save chord the projects form uses: one form vocabulary.
        Binding("ctrl+s", "save", "Save", show=False),
        Binding("tab", "focus_next_field", "Next field", show=False),
        Binding("shift+tab", "focus_prev_field", "Previous field", show=False),
        Binding("escape", "cancel", "Back", show=False),
    ]

    def __init__(
        self,
        *,
        on_save: Callable[[schedule_model.FormValues], None],
        on_cancel: Callable[[], None],
    ) -> None:
        super().__init__(classes="projects-form-page schedule-form")
        self._on_save = on_save
        self._on_cancel = on_cancel
        self._title = Static(classes="schedule-form-title")
        self._message = TextArea(id="schedule-form-message")
        # Captured once: an EDIT swaps the placeholder for the instant the save
        # will keep (U3), and reset() has to put the instruction back.
        self._when_placeholder_default = _t(_k.form_when_placeholder())
        self._when = Input(placeholder=self._when_placeholder_default, id="schedule-form-when")
        self._repeat = Input(
            placeholder=_t(_k.form_repeat_placeholder()), id="schedule-form-repeat"
        )
        self._ends = Input(placeholder=_t(_k.form_ends_placeholder()), id="schedule-form-ends")
        # NOT a Textual `Checkbox`: its stock toggle is a lone `X` whose state
        # is carried by COLOUR alone ($panel-darken-2 vs $text-success), which
        # measured as a blank row on this sheet and is unreadable in a
        # colourless terminal. The projects form's cycle row is this app's
        # boolean vocabulary — `‹ off ›` / `‹ on ›` says the value in words.
        #
        # The words are the CATALOGUE's, captured once beside the widget: the
        # comparison below must be against the exact strings this row carries,
        # or a locale switch between construction and save would read the
        # toggle as "not the on word" and silently save notify=false.
        self._notify_off = _t(_k.value_off())
        self._notify_on = _t(_k.value_on())
        self._notify = FormCycleRow(
            (self._notify_off, self._notify_on), self._notify_off, id="schedule-form-notify"
        )
        self._blocks = [
            FormFieldBlock(
                _t(_k.form_message_label()),
                self._message,
                hint=_t(_k.form_message_placeholder()),
                gap=False,
            ),
            FormFieldBlock(
                _t(_k.form_when_label()),
                self._when,
                hint=_t(_k.form_when_hint()),
            ),
            FormFieldBlock(
                _t(_k.form_repeat_label()),
                self._repeat,
                hint=_t(_k.form_repeat_hint()),
            ),
            FormFieldBlock(
                _t(_k.form_ends_label()),
                self._ends,
                hint=_t(_k.form_ends_hint()),
            ),
            FormFieldBlock(
                _t(_k.form_notify_label()),
                self._notify,
                hint=_t(_k.form_notify_hint()),
            ),
        ]
        self._fields_scroll = VerticalScroll(classes="projects-form-fields schedule-form-fields")
        # ``markup=False``: the builder's sentence is DATA (it quotes the
        # user's own bad value), and Static parses brackets by default.
        self._refusal = Static("", classes="schedule-form-refusal", markup=False)
        self._keys_hint = Static(_t(_k.form_keys_hint()), classes="schedule-form-keys gap-above")

    # -- composition --------------------------------------------------------
    def compose(self):  # type: ignore[override]
        yield self._title
        with self._fields_scroll:
            for block in self._blocks:
                yield block
        # PINNED, not scrolled (round-1 design D2). The first frame shows the
        # message box, `when` and only the start of `repeat`; with the legend at
        # the scroll's end the only save instruction in the form sat below the
        # fold and a new user had to discover scrolling before learning how to
        # submit. The `gap-above` row separates it from the clipped field so it
        # reads as the page's chrome rather than as `repeat`'s hint.
        yield self._keys_hint
        # A refusal is pinned too: it is the receipt for the save the user just
        # pressed, and it must be visible wherever the fields are scrolled.
        yield self._refusal

    def on_mount(self) -> None:
        self._refusal.display = False

    # -- read surface (the page and its tests) ------------------------------
    @property
    def message_field(self) -> TextArea:
        return self._message

    @property
    def when_field(self) -> Input:
        return self._when

    @property
    def repeat_field(self) -> Input:
        return self._repeat

    @property
    def ends_field(self) -> Input:
        return self._ends

    @property
    def notify_field(self) -> FormCycleRow:
        return self._notify

    @property
    def refusal(self) -> str:
        return self._refusal_text

    def values(self) -> schedule_model.FormValues:
        return schedule_model.FormValues(
            message=self._message.text,
            when=self._when.value,
            repeat=self._repeat.value,
            ends=self._ends.value,
            notify=self._notify.value == self._notify_on,
        )

    def prefill(
        self,
        values: schedule_model.FormValues,
        *,
        wake_id: str = "",
        when_placeholder: str = "",
    ) -> None:
        """Show ``values``; ``wake_id`` also names the row in the title.

        ``when_placeholder`` is the EDIT case's value-bearing hint (round-1 UX
        U3): the field stays blank because the stored instant is what a blank
        save keeps, so the placeholder names that instant — "blank keeps the
        time" alone made the preservation uninspectable.
        """
        self._title.update(
            Text(
                _t(_k.form_title_edit(id=wake_id)) if wake_id else _t(_k.form_title_new()),
                style="bold",
            )
        )
        self._message.text = values.message
        self._when.placeholder = when_placeholder or self._when_placeholder_default
        self._when.value = values.when
        self._repeat.value = values.repeat
        self._ends.value = values.ends
        self._notify.set_value(self._notify_on if values.notify else self._notify_off)
        self.set_refusal("")

    def reset(self) -> None:
        self.prefill(schedule_model.FormValues())

    def set_refusal(self, text: str) -> None:
        self._refusal_text = text
        self._refusal.update(Text(text, style=_ink("refusal")))
        self._refusal.display = bool(text)

    def focus_first(self) -> None:
        try:
            self._message.focus()
        except Exception:  # noqa: BLE001 - focus is a nicety
            pass

    def fields(self) -> list[Any]:
        return [self._message, self._when, self._repeat, self._ends, self._notify]

    def action_focus_next_field(self) -> None:
        self._focus_step(1)

    def action_focus_prev_field(self) -> None:
        self.focus_prev_field()

    def focus_prev_field(self) -> None:
        """Move focus back one field. PUBLIC because the app's priority
        ``shift+tab`` handler delegates here instead of cycling effort (U1)."""
        self._focus_step(-1)

    def _focus_step(self, delta: int) -> None:
        controls = self.fields()
        focused = self.app.focused
        try:
            index = controls.index(focused)
        except ValueError:
            index = -1 if delta > 0 else 0
        target = controls[(index + delta) % len(controls)]
        try:
            target.focus()
        except Exception:  # noqa: BLE001 - focus is a nicety
            pass

    def action_save(self) -> None:
        self._on_save(self.values())

    def action_cancel(self) -> None:
        self._on_cancel()


class ScheduleView(Vertical):
    """The session's scheduled wakes, with create/edit/cancel as page states."""

    can_focus = True

    BINDINGS = [
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        Binding("pageup", "page(-1)", "Page up", show=False),
        Binding("pagedown", "page(1)", "Page down", show=False),
        Binding("home", "jump(0)", "First", show=False),
        Binding("end", "jump(1)", "Last", show=False),
        Binding("n", "new", "New", show=False),
        Binding("e", "edit", "Edit", show=False),
        Binding("enter", "edit", "Edit", show=False),
        Binding("d", "cancel_wake", "Cancel", show=False),
        Binding("r", "reload", "Re-read", show=False),
        Binding("escape", "leave", "Back", show=False),
    ]

    def __init__(
        self,
        session: Any,
        *,
        session_id: str = "",
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        super().__init__(classes="settings-view schedule-view")
        self._session = session
        self._session_id = session_id
        self._scheduler = getattr(session, "wake_scheduler", None)
        #: A FOLLOWER does not own the runtime the wakes live in: it may look,
        #: never write. The owner's session carries ``owns_runtime`` True.
        self._owns = bool(getattr(session, "owns_runtime", True))
        self._writable = self._owns and self._scheduler is not None
        self._config_dir = config_dir()
        self._now = now_ms or (lambda: int(time.time() * 1000))
        self._title = Static(classes="settings-view-title")
        self._rule = Static(classes="settings-view-rule")
        self._body_text = Static()
        self._body = _ScheduleBody(
            self._body_text, on_settle=self._paint, classes="settings-view-body"
        )
        self._body.can_focus = False
        self._pane_body = Static(classes="schedule-view-pane-text")
        self._pane = ScrollableContainer(
            self._pane_body, classes="settings-view-pane schedule-view-pane"
        )
        self._pane.can_focus = True
        self._columns = Horizontal(self._body, self._pane, classes="settings-view-columns")
        self._notice_text = Static(classes="settings-view-detail")
        hint_ink = {"ink": ("muted", "muted"), "lifted_ink": ("fg", "fg")}
        self._hint_move = HintButton("↑↓", lambda: self.action_move(1), **hint_ink)
        self._hint_new = HintButton("n", self.action_new, **hint_ink)
        self._hint_edit = HintButton("e", self.action_edit, **hint_ink)
        self._hint_cancel = HintButton("d", self.action_cancel_wake, **hint_ink)
        self._hint_refresh = HintButton("r", self.action_reload, **hint_ink)
        self._hint_escape = HintButton("esc", self.action_leave, **hint_ink)
        self._hint_order = (
            self._hint_move,
            self._hint_new,
            self._hint_edit,
            self._hint_cancel,
            self._hint_refresh,
            self._hint_escape,
        )
        self._hints = Horizontal(*self._hint_order, classes="settings-view-hints")
        self._form = ScheduleForm(on_save=self._save, on_cancel=self._close_form)
        self._form.display = False
        #: ``list`` or ``form``; the page's own level, not a screen.
        self.view_mode = "list"
        self._cursor = 0
        self._rows: list[schedule_model.ScheduleRow] = []
        self._narrow = False
        #: The read-only reason, resolved once. A follower is told WHY it cannot
        #: change these rows; a runtime with no scheduler says that instead.
        if not self._owns:
            self._read_only_notice = _t(_k.notice_read_only())
        elif self._scheduler is None:
            self._read_only_notice = _t(_k.notice_no_scheduler())
        else:
            self._read_only_notice = ""
        #: The one-line footer: a receipt, a refusal or the read-only reason.
        self._notice = ""
        self._confirm = False
        #: The row the form is editing, or None for a create. It is NOT derived
        #: from the selection: `n` on a non-empty list must create, and a
        #: selection-derived answer would edit the row under the cursor instead
        #: (the defect this field exists to make impossible).
        self._editing: str | None = None
        #: Set while a write is in flight so a second ctrl+s cannot submit the
        #: same create twice (both would allocate the same free id).
        self._saving = False
        #: Best-effort and never a gate: the probe runs off the event loop and
        #: an error is read as "assume it is running" rather than claiming a
        #: failure the page has not established.
        self.supervisor_checked = False
        self.supervisor_warning = ""

    # -- composition --------------------------------------------------------
    def compose(self):  # type: ignore[override]
        yield self._title
        yield self._rule
        with self._columns:
            yield self._body
            yield self._pane
        yield self._notice_text
        yield self._form
        yield self._hints

    def on_mount(self) -> None:
        self._apply_layout()
        self._reload()
        self._paint()
        self.call_after_refresh(self._paint)
        try:
            self.focus()
        except Exception:  # noqa: BLE001 - focus is a nicety
            pass
        self.run_worker(self._probe_supervisor(), exclusive=True)

    def on_resize(self) -> None:
        self._apply_layout()
        self._paint()

    def _apply_layout(self) -> None:
        """Hide the pane where it would squeeze the list below readability."""
        self._narrow = self.size.width < 90
        self._pane.display = not self._narrow

    # -- data -----------------------------------------------------------------
    def _owned(self) -> list[Any]:
        """The scheduler's OWNED view, hidden timers included (see the model)."""
        if self._scheduler is None:
            return []
        return list(getattr(self._scheduler, "schedules", ()) or ())

    def _reload(self, *, select_id: str = "") -> None:
        self._rows = schedule_model.rows(self._owned(), self._now())
        if select_id:
            found = next(
                (index for index, row in enumerate(self._rows) if row.id == select_id), None
            )
            if found is not None:
                self._cursor = found
        self._cursor = max(0, min(self._cursor, len(self._rows) - 1))

    # -- read surface ---------------------------------------------------------
    @property
    def writable(self) -> bool:
        return self._writable

    @property
    def confirming(self) -> bool:
        return self._confirm

    @property
    def form(self) -> ScheduleForm:
        return self._form

    @property
    def wants_field_tab(self) -> bool:
        """True while the form owns the keyboard.

        The app's ``shift+tab`` binding is ``priority=True``, so it fires before
        the focused field sees the chord. While this page is in form mode the
        chord is a FIELD-NAVIGATION key, not the effort cycle: the app delegates
        it to :meth:`form_focus_previous` (round-1 UX U1 — reverse-tab used to
        change the session's reasoning effort and billable setting).
        """
        return self.view_mode == "form"

    def form_focus_previous(self) -> None:
        self._form.focus_prev_field()

    @property
    def selected(self) -> schedule_model.ScheduleRow | None:
        if 0 <= self._cursor < len(self._rows):
            return self._rows[self._cursor]
        return None

    # -- keys ----------------------------------------------------------------
    def action_move(self, delta: int) -> None:
        if not self._rows or self.view_mode != "list":
            return
        self._cursor = max(0, min(len(self._rows) - 1, self._cursor + delta))
        self._paint()

    def action_page(self, delta: int) -> None:
        if not self._rows or self.view_mode != "list":
            return
        step = max(1, (self._body.size.height or 1) // 2)
        self._cursor = max(0, min(len(self._rows) - 1, self._cursor + delta * step))
        self._paint()

    def action_jump(self, edge: int) -> None:
        if not self._rows or self.view_mode != "list":
            return
        self._cursor = 0 if edge == 0 else len(self._rows) - 1
        self._paint()

    def action_new(self) -> None:
        if not self._writable or self.view_mode != "list":
            return
        self._editing = None
        self._form.reset()
        self._show_form()

    def action_edit(self) -> None:
        if not self._writable or self.view_mode != "list":
            return
        row = self.selected
        if row is None:
            return
        schedule = next((item for item in self._owned() if item.id == row.id), None)
        if schedule is None:
            # The row was cancelled under the page: re-read rather than edit a
            # handle that is no longer there.
            self.action_reload()
            return
        self._editing = row.id
        self._form.prefill(
            schedule_model.prefilled(schedule),
            wake_id=row.id,
            when_placeholder=schedule_model.kept_time(schedule),
        )
        self._show_form()

    def action_cancel_wake(self) -> None:
        if not self._writable or self.view_mode != "list" or self.selected is None or self._saving:
            return
        if self._confirm:
            return
        self._confirm = True
        self._paint()

    def action_reload(self) -> None:
        self._notice = ""
        self._confirm = False
        self._reload()
        self._paint()

    def action_leave(self) -> None:
        if self.view_mode == "form":
            self._close_form()
            return
        if self._confirm:
            self._disarm_confirm()
            return
        self.post_message(ScheduleViewDismissed())

    def on_key(self, event: Any) -> None:
        """The cancel confirm owns the keyboard while it is up.

        ``y`` cancels and ``esc``/``n``/``enter`` keep the row; every other
        printable key is swallowed so no letter is read as a second meaning
        (the projects form's rule, same reason).
        """
        if not self._confirm:
            return
        if event.key == "y":
            event.stop()
            event.prevent_default()
            self._cancel_selected()
            return
        if event.key in ("escape", "n", "enter"):
            event.stop()
            event.prevent_default()
            self._disarm_confirm()
            return
        if event.is_printable:
            event.stop()
            event.prevent_default()

    def _disarm_confirm(self) -> None:
        self._confirm = False
        self._paint()

    def _cancel_selected(self) -> None:
        row = self.selected
        if row is None or self._saving:
            # The confirm still owns the keyboard until it is answered; while a
            # write is in flight the honest answer is "not now", painted back
            # to the list rather than leaving the question up with no effect.
            self._disarm_confirm()
            return
        self._confirm = False
        remaining = schedule_model.without_schedule(self._owned(), row.id)
        self._mutate(remaining, sentence=schedule_model.notice_cancelled(row.id))

    def _mutate(self, schedules: list[Any], *, sentence: str) -> None:
        """One in-flight write at a time; the writer is the session's."""
        if self._saving:
            return
        self._saving = True
        self.run_worker(self._push_write(schedules, sentence), exclusive=True)

    async def _push_write(self, schedules: list[Any], sentence: str) -> None:
        try:
            await self._session.set_wake_schedules(list(schedules))
        except Exception as exc:  # noqa: BLE001 - the page states the failure
            self._saving = False
            self._notice = _t(_k.notice_write_failed(error=str(exc)))
            self._paint()
            return
        self._saving = False
        self._notice = sentence
        self._reload()
        self._paint()

    # -- form ----------------------------------------------------------------
    def _show_form(self) -> None:
        self.view_mode = "form"
        self._columns.display = False
        self._notice_text.display = False
        self._form.display = True
        self._form.focus_first()
        self._paint()

    def _close_form(self) -> None:
        if self.view_mode != "form":
            return
        self.view_mode = "list"
        self._editing = None
        self._form.display = False
        self._columns.display = True
        self._notice_text.display = True
        self._apply_layout()
        try:
            self.focus()
        except Exception:  # noqa: BLE001 - focus is a nicety
            pass
        self._paint()

    def _save(self, values: schedule_model.FormValues) -> None:
        if self._saving or self.view_mode != "form":
            return
        owned = self._owned()
        if self._editing is None:
            result = schedule_model.create(values, owned, self._now())
        else:
            result = schedule_model.edit(values, owned, self._editing, self._now())
        if result.schedule is None:
            self._form.set_refusal(result.sentence)
            return
        self._form.set_refusal("")
        self._saving = True
        self.run_worker(self._commit(result), exclusive=True)

    async def _commit(self, result: schedule_model.SaveResult) -> None:
        """Persist through the session's one writer, then repaint from it."""
        built = result.schedule
        if built is None:  # pragma: no cover - `_save` only calls on success
            return
        try:
            await self._session.set_wake_schedules(
                schedule_model.with_schedule(self._owned(), built)
            )
        except Exception as exc:  # noqa: BLE001 - the page states the failure
            self._saving = False
            self._form.set_refusal(_t(_k.notice_write_failed(error=str(exc))))
            return
        self._saving = False
        self._editing = None
        self._notice = result.sentence
        self.view_mode = "list"
        self._form.display = False
        self._columns.display = True
        self._notice_text.display = True
        self._reload(select_id=built.id)
        try:
            self.focus()
        except Exception:  # noqa: BLE001 - focus is a nicety
            pass
        self._paint()

    # -- supervisor probe ----------------------------------------------------
    async def _probe_supervisor(self) -> None:
        """One best-effort line about out-of-process delivery (never a gate)."""
        try:
            state = await asyncio.to_thread(supervisor_state, self._config_dir)
            running = bool(getattr(state, "running", False))
        except Exception:  # noqa: BLE001 - a failed probe must not accuse
            running = True
        self.supervisor_warning = "" if running else _t(_k.notice_supervisor())
        self.supervisor_checked = True
        self._paint()

    # -- painting ------------------------------------------------------------
    def _paint(self) -> None:
        if self.view_mode == "form":
            # The form IS the page while it is up: the title and hints belong
            # to the list state, and leaving them painted above a form would
            # claim the page is still browsable.
            self._title.display = False
            self._rule.display = False
            self._hints.display = False
            return
        self._title.display = True
        self._rule.display = True
        self._hints.display = True
        self._apply_layout()
        self._title.update(
            Text(
                _t(_k.title()) + schedule_model.SEPARATOR + _t(_k.subtitle(count=len(self._rows))),
                style="bold",
            )
        )
        self._rule.update(Text(_RULE_GLYPH * 200, style="dim"))
        self._notice_text.update(Text(self._footer_text(), style="dim"))
        self._body_text.update(self._body_rich())
        self._pane_body.update(self._pane_text())
        self._paint_hints()
        self._scroll_to_cursor()

    def _footer_text(self) -> str:
        if self._confirm:
            row = self.selected
            if row is not None:
                return _t(_k.cancel_confirm(id=row.id))
        if self._notice:
            return self._notice
        if self._read_only_notice:
            return self._read_only_notice
        return self.supervisor_warning

    def _body_rich(self) -> Text:
        text = Text()
        if not self._rows:
            text.append(_t(_k.empty_header()), style="bold")
            text.append("\n\n")
            text.append(_t(_k.empty_body()), style="dim")
            return text
        width = max(10, (self._body.size.width or self.size.width or 80) - 2)
        for index, row in enumerate(self._rows):
            selected = index == self._cursor
            ground = _selection_ground() if selected else Style()
            marker = _CURSOR if selected else " "
            head_style = Style(
                color=(
                    theme_mod.semantic_color("fg")
                    if selected
                    else theme_mod.semantic_color("muted")
                ),
                bgcolor=ground.bgcolor,
                bold=selected,
            )
            meta_style = Style(
                color=theme_mod.semantic_color(_meta_ink(selected)),
                bgcolor=ground.bgcolor,
            )
            text.append(f"{marker} ", style=head_style)
            text.append(truncate_cells(row.head, width - 2), style=head_style)
            text.append("\n")
            text.append("  ", style=meta_style)
            text.append(truncate_cells(row.meta, width - 2), style=meta_style)
            # ONE blank row between entries: the two lines of a row are a
            # unit, and without the gutter three wakes read as one paragraph.
            # Fixed height (3 per non-final row) is what keeps the cursor
            # arithmetic exact.
            if index != len(self._rows) - 1:
                text.append("\n\n")
        return text

    def _detail_column(self, row: schedule_model.ScheduleRow) -> int:
        """The label column for THIS row's translated labels.

        Measured per paint from the labels actually being drawn, with a
        two-cell minimum gap: ``max(9, widest + 2)``. A translation longer than
        the floor therefore moves the values right rather than having the label
        run into its value; the English labels all sit under the floor, so the
        untranslated pane is unchanged.
        """
        widest = max((cell_len(label) for label, _ in row.detail), default=0)
        return max(_MIN_LABEL_COLUMN, widest + 2)

    def _pane_text(self) -> Text:
        row = self.selected
        if row is None:
            return Text()
        text = Text()
        text.append(row.id, style=Style(color=theme_mod.semantic_color("accent"), bold=True))
        text.append("\n")
        text.append(row.message)
        text.append("\n\n")
        column = self._detail_column(row)
        for label, value in row.detail:
            text.append(
                label + " " * (column - cell_len(label)),
                style=Style(color=theme_mod.semantic_color("muted")),
            )
            wrapped = value.replace("\n", " ")
            text.append(wrapped)
            text.append("\n")
        return text

    def _paint_hints(self) -> None:
        # EXIT AND RE-READ ARE ALWAYS ADVERTISED. The empty state once showed
        # only "n new": a page with no rows still has an exit key and can still
        # pick up a wake armed from another surface, and a hint row that omits
        # them makes both discoverable only by guessing (measured on the empty
        # capture).
        labels: dict[HintButton, str] = {
            self._hint_refresh: _t(_k.hint_refresh()),
            self._hint_escape: _t(_k.hint_back()),
        }
        if self._rows:
            # The full row measures 57 cells and the 60-column page gives it
            # 56: the last hint's label was clipped to "esc bac" (measured in
            # the narrow capture). Below 70 columns the arrows keep their key
            # and lose the word — `↑↓` needs no gloss — which buys the five
            # cells the row was over; every key stays advertised.
            narrow = self.size.width < 70
            labels[self._hint_move] = "" if narrow else _t(_k.hint_move())
            if self._writable:
                labels[self._hint_edit] = _t(_k.hint_edit())
                labels[self._hint_cancel] = _t(_k.hint_cancel())
        if self._writable:
            labels[self._hint_new] = _t(_k.hint_new())
        lead = False
        for button in self._hint_order:
            entry = labels.get(button)
            button.display = entry is not None
            if entry is None:
                continue
            button.paint(entry, lead=lead)
            lead = True

    def _scroll_to_cursor(self) -> None:
        """Keep the selected TWO-LINE row in the viewport (fixed heights)."""
        if not self._rows:
            return
        start = self._cursor * 3
        viewport = max(1, self._body.size.height or 1)
        y = self._body.scroll_offset.y
        if viewport <= 2:
            # A row is TWO lines. Where the viewport is exactly one row (the
            # 60x24 capture, with the dock band under it), bottom-anchoring
            # scrolled to `start + 1` and hid the row's HEAD — the frame showed
            # only "in 1h · every 1d…" under a blank first line. Anchor such a
            # row at its top: the head is the identity, the meta can wait.
            target = start
        elif start < y:
            target = start
        elif start + 2 > y + viewport:
            target = start + 2 - viewport
        else:
            return
        try:
            self._body.scroll_to(y=max(0, target), animate=False, immediate=True)
        except Exception:  # noqa: BLE001 - scrolling is a nicety
            pass

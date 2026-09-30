"""The projects page's DETAIL state (S6d parity P2): one project as a page.

The detail shows the record a reader asked about — overview, description,
milestones (``↵`` toggles completion), todos and sessions — inside the same
mode as the canvases, so the mode's invariants (focus restore, greyed dock,
one page per screen, ``r`` recomposition) hold here too (spec §1/§10.1).

Why a WIDGET TREE and not a painted canvas like the canvases are: the
description is rendered through the transcript's own rich-``Markdown`` path,
rows are individual click targets, and Textual's own scroll container gives
the page wheel, scrollbars and ``scroll_visible`` for free (the settings
mechanics the spec names). The page stays I/O-free like the rest of the
mode: it renders the ONE composed view row it was handed and relays row
actions through the callback ``ProjectsView`` installs — it never reads the
store, the runtime, or a session directory. The ``r`` key and every message
belong to the host.

The ruler is NOT painted here: the view paints the rule row from
:meth:`ProjectDetailPage.section_anchors` — the same zero-row sticky
counterpart the canvases use (design §6, "on the detail page the same row
names the current in-page section").
"""

from __future__ import annotations

from typing import Any, Callable

from rich.markdown import Markdown
from rich.text import Text
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Static

from local_operator.tui.link_markup import autolink_bare_urls
from local_operator.tui.projects_render import (
    UPDATE_BODY_LINES,
    UPDATES_PER_PAGE,
    StyleFor,
    attachment_path_text,
    attachment_row_text,
    detail_meta_line,
    detail_milestone_row_text,
    detail_section_heading,
    detail_session_row_text,
    detail_todo_lines,
    older_updates_text,
    update_body_lines,
    update_day_label,
    update_more_lines_text,
    update_stamp_text,
)

#: A row action the page relays to its host: ``(kind, row)`` with kind
#: ``"session"`` (open the conversation), ``"milestone"`` (toggle it) or
#: ``"attachment"`` (hand the copied file to the platform opener).
RowAction = Callable[[str, dict[str, Any]], None]


class DetailRow(Static):
    """One row of the detail page.

    ``selectable`` decides whether the cursor may land here; ``action_label``
    is what the ``↵`` hint spells for the row (``open`` / ``toggle``), and
    ``activate`` is what ``↵`` — or a row's second click — does. Rows that are
    informational only (headings, prose, meta, todo lines) keep the defaults.
    """

    selectable = False
    #: ``(label, count)`` of the section this row starts, on heading rows only.
    section: tuple[str, str | None] | None = None
    #: The section a SELECTABLE row belongs to — the label of the last heading
    #: before it, read by the section jumps.
    section_label: str = ""

    def __init__(self, *, classes: str = "") -> None:
        super().__init__(classes=f"projects-detail-row {classes}".strip())

    def set_selected(self, selected: bool) -> None:
        """Repaint for the cursor landing on / leaving this row (no-op base)."""

    def action_label(self) -> str | None:
        return None

    def action_verb(self) -> str | None:
        """The bare verb — the hint's fallback when the named form cannot fit."""
        return None

    def activate(self) -> None:
        return None


class DetailHeadingRow(DetailRow):
    """``milestones (1/3)`` and its underline — the section's first row."""

    def __init__(
        self, name: str, count: str | None, style_for: StyleFor, *, gap: bool = True
    ) -> None:
        """``gap`` carries the blank row above the heading (``.gap-above``).

        That class is the sheet's ONLY vertical spacing declaration — the
        minimalism guard pins the count at one, so a rule-level margin here
        would be a second source AND could silently double with the class.
        The page withholds it from its first row: the heading that opens the
        page must not push every following row (and every section anchor)
        down one.
        """
        super().__init__(
            classes="projects-detail-heading gap-above" if gap else "projects-detail-heading"
        )
        self.section = (name, count)
        head, underline = detail_section_heading(name, count)
        text = Text(no_wrap=True)
        text.append(head, style=style_for("name"))
        text.append("\n" + underline, style=style_for("dim"))
        self.update(text)


class DetailSentenceRow(DetailRow):
    """One dim honest sentence (``no milestones set``, ``no sessions linked``)."""

    def __init__(self, sentence: str, style_for: StyleFor) -> None:
        super().__init__(classes="projects-detail-sentence")
        self.update(Text(sentence, style=style_for("dim"), no_wrap=True))


class DetailDayRow(DetailRow):
    """A day group's header inside the updates feed (spec §7.3).

    Not selectable and not a section: it groups the entries under it the way
    the timeline's axis rows group cards, so the ruler (which names SECTIONS)
    never reports it.
    """

    def __init__(self, label: str, style_for: StyleFor) -> None:
        super().__init__(classes="projects-detail-day")
        self.update(Text(label, style=style_for("dim"), no_wrap=True))


class DetailUpdateStampRow(DetailRow):
    """One feed entry's stamp line — ``↵`` toggles its clamped body (§7.3)."""

    selectable = True

    def __init__(
        self,
        entry: dict[str, Any],
        *,
        index: int,
        expanded: bool,
        on_toggle: Callable[[int], None],
        style_for: StyleFor,
    ) -> None:
        super().__init__(classes="projects-detail-update")
        self._entry = entry
        self._index = index
        self._expanded = expanded
        self._on_toggle = on_toggle
        self._style_for = style_for
        self.section_label = "updates"
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        self.update(update_stamp_text(self._entry, selected=selected, style_for=self._style_for))

    def action_label(self) -> str | None:
        """The verb NAMED at its effect, like every other row verb here."""
        return "collapse" if self._expanded else "expand"

    def action_verb(self) -> str | None:
        return "collapse" if self._expanded else "expand"

    def activate(self) -> None:
        self._on_toggle(self._index)


class DetailUpdateBodyRow(DetailRow):
    """One feed entry's markdown body, clamped to :data:`UPDATE_BODY_LINES`.

    The body goes through the transcript's rich-Markdown path, the same one
    the description uses. Unclamped, a long entry would dominate the page and
    push the sections below it off the viewport; the marker row under it names
    the key that opens the rest.
    """

    def __init__(self, text: str, *, expanded: bool, style_for: StyleFor) -> None:
        super().__init__(classes="projects-detail-update-body")
        lines = update_body_lines(text)
        self._shown = "\n".join(lines if expanded else lines[:UPDATE_BODY_LINES]).strip()
        if self._shown:
            self.update(Markdown(autolink_bare_urls(self._shown)))
        else:
            self.update(Text("(no text in this entry)", style=style_for("dim"), no_wrap=True))

    def readback(self) -> str | None:
        """The body's source, for ``painted_rows`` (the prose row's reason)."""
        return self._shown or None


class DetailUpdateMarkerRow(DetailRow):
    """The clamp marker under an overflowing entry (spec §7.3)."""

    def __init__(self, count: int, *, expanded: bool, style_for: StyleFor) -> None:
        super().__init__(classes="projects-detail-update-marker")
        self.update(update_more_lines_text(count, expanded=expanded, style_for=style_for))


class DetailAttachmentRow(DetailRow):
    """One attachment affordance — ``↵`` opens the file with the OS (§7.4).

    The path is NOT part of this widget: a two-line ``Text`` inside one ``Static``
    measured as four rows and painted the path's continuation as a bare ``→``
    (caught in the frame pass). The path gets its own row under this one
    (:class:`DetailAttachmentPathRow`), which is also why the row's own verb
    stays the affordance's.
    """

    selectable = True

    def __init__(
        self,
        attachment: dict[str, Any],
        *,
        on_action: RowAction,
        style_for: StyleFor,
    ) -> None:
        super().__init__(classes="projects-detail-attachment")
        self._attachment = attachment
        self._on_action = on_action
        self._style_for = style_for
        self.section_label = "updates"
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        self.update(
            attachment_row_text(self._attachment, selected=selected, style_for=self._style_for)
        )

    def action_label(self) -> str | None:
        return "open"

    def action_verb(self) -> str | None:
        return "open"

    def activate(self) -> None:
        self._on_action("attachment", self._attachment)


class DetailAttachmentPathRow(DetailRow):
    """``→ <path>`` under its attachment, always shown (spec §7.4).

    Not selectable: the affordance above it carries the verb, so a second
    cursor stop on the same file would make `↑↓` cost two presses per
    attachment. The ``[missing on disk]`` marker rides this line.
    """

    def __init__(self, attachment: dict[str, Any], style_for: StyleFor) -> None:
        super().__init__(classes="projects-detail-attachment-path")
        self._attachment = attachment
        self._style_for = style_for
        self.update(attachment_path_text(attachment, style_for=style_for))

    def fit_width(self, width: int) -> None:
        """Re-fit the path to the width the ROW actually has, not the page's.

        Textual wraps a ``Static``'s text regardless of ``no_wrap``, so a long
        path must be cut HERE, with an ellipsis; the two widths differ by the
        scrollbar's column (measured: the page's content box is 96 cells while
        the row's own region is 95), and fitting to the larger one left the
        last cell to wrap onto a row of its own.
        """
        target = self.size.width or width
        if target > 0:
            self.update(
                attachment_path_text(self._attachment, width=target, style_for=self._style_for)
            )

    def readback(self) -> str | None:
        """The path line as plain text, for ``painted_rows``."""
        return attachment_path_text(self._attachment).plain or None


class DetailMetaRow(DetailRow):
    """The overview sentence; rebuilt on resize because it sheds by width."""

    def __init__(self, view: dict[str, Any], style_for: StyleFor) -> None:
        super().__init__(classes="projects-detail-meta")
        self._view = view
        self._style_for = style_for
        self.fit_width(self.size.width)

    def fit_width(self, width: int) -> None:
        if width > 0:
            self.update(detail_meta_line(self._view, width=width, style_for=self._style_for))


class DetailProseRow(DetailRow):
    """The description, through the transcript's own rich-Markdown path.

    Empty reads the honest sentence — WITHOUT the spec's `— e edits one`
    tail, which advertises a key that arrives with the form slice (P4); the
    copy lands with it.
    """

    def __init__(self, description: str, style_for: StyleFor) -> None:
        super().__init__(classes="projects-detail-prose")
        self._source = description
        if description.strip():
            self.update(Markdown(autolink_bare_urls(description)))
        else:
            self.update(Text("no description yet", style=style_for("dim"), no_wrap=True))

    def readback(self) -> str | None:
        """The description's source text for ``painted_rows`` (UX round 1, U6).

        The row holds a rich ``Markdown`` renderable, which carries no
        ``.plain``; without this the page's own readback showed ``""`` for the
        description — the slice's headline feature, unassertable. ``None`` on
        the empty case, so the honest sentence still reads from the row's
        plain ``Text``.
        """
        return self._source if self._source.strip() else None


class DetailMilestoneRow(DetailRow):
    """One milestone; ``↵`` toggles completion (the store op the tool uses)."""

    selectable = True

    def __init__(
        self,
        row: dict[str, Any],
        *,
        section_label: str,
        on_action: RowAction,
        style_for: StyleFor,
    ) -> None:
        super().__init__(classes="projects-detail-milestone")
        self._row = row
        self._style_for = style_for
        self._on_action = on_action
        self.section_label = section_label
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        self.update(
            detail_milestone_row_text(self._row, selected=selected, style_for=self._style_for)
        )

    def action_label(self) -> str | None:
        """The verb NAMED at its target (UX round 1, U3).

        ``toggle step 00`` rather than a bare ``toggle``: a reader scrolled
        away from the row cursor could not predict what the press would
        change — or that it would yank the viewport back to the row. The bare
        verb is the hint ladder's low rung (``action_verb``), so a long name
        sheds before the key does.
        """
        name = str(self._row.get("name") or "")
        return f"toggle {name}" if name else "toggle"

    def action_verb(self) -> str | None:
        return "toggle"

    def activate(self) -> None:
        self._on_action("milestone", self._row)


class DetailSessionRow(DetailRow):
    """One linked session; ``↵`` runs the shipped conversation ladder for it."""

    selectable = True

    def __init__(
        self,
        row: dict[str, Any],
        *,
        own_session: str | None,
        section_label: str,
        on_action: RowAction,
        style_for: StyleFor,
    ) -> None:
        super().__init__(classes="projects-detail-session")
        self._row = row
        self._own_session = own_session
        self._style_for = style_for
        self._on_action = on_action
        self.section_label = section_label
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        self.update(
            detail_session_row_text(
                self._row,
                selected=selected,
                own_session=self._own_session,
                style_for=self._style_for,
            )
        )

    def action_label(self) -> str | None:
        return "open"

    def action_verb(self) -> str | None:
        return "open"

    def activate(self) -> None:
        self._on_action("session", self._row)


class DetailTodoRow(DetailRow):
    """One ``todos`` line (information only, like the spec says: no actions)."""

    def __init__(self, text: Text) -> None:
        super().__init__(classes="projects-detail-todo")
        self.update(text)


class ProjectDetailPage(VerticalScroll):
    """The detail state's body: a scroll of rows with a clamped row cursor.

    Cursor semantics follow the settings page the spec cites: ``↑↓`` move the
    cursor (CLAMPED) and reveal it, page keys and the wheel scroll the
    viewport and leave the cursor alone. ``esc``/``r``/section jumps are the
    view's (one binding list); this widget answers the queries those actions
    need and repaints when the selection moves.

    ``up``/``down`` are bound HERE, not only on the view: a focused scroll
    container consumes the arrows for its own default scrolling before an
    ancestor binding sees them (measured: the row cursor never moved), so the
    page routes them back through the host's nav callback — the same ``move``
    path the hint arming shares. ``pgup``/``pgdn`` are deliberately left to
    the container: on this page a page key scrolls the viewport and leaves
    the row cursor, which is the shipped split.
    """

    can_focus = True

    BINDINGS = [
        Binding("up", "cursor_up", "Move up", show=False),
        Binding("down", "cursor_down", "Move down", show=False),
    ]

    def __init__(
        self,
        on_action: RowAction,
        style_for: StyleFor,
        on_nav: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(classes="projects-detail")
        self._on_action = on_action
        self._style_for = style_for
        self._on_nav = on_nav
        self._view: dict[str, Any] | None = None
        self._own_session: str | None = None
        self._project_id: str | None = None
        self._project_name = ""
        #: Feed entries the reader has opened past the clamp, by entry ordinal.
        #: Page-local VIEW state (never written anywhere): `↵` on an entry's
        #: stamp toggles it, and a re-show keeps what the reader opened.
        self._expanded: set[int] = set()
        self._selectables: list[DetailRow] = []
        self._selected = 0

    @property
    def project_id(self) -> str | None:
        """The project every row on this page was built from (write provenance).

        The row verbs relay THIS id, not the canvas cursor's current row:
        bound to the same snapshot that built the rows, a write can never
        land on a project the row did not come from (QA round 1, Q1).
        """
        return self._project_id

    @property
    def project_name(self) -> str:
        """The built project's name — the jump sentence's subject."""
        return self._project_name

    def action_cursor_up(self) -> None:
        if self._on_nav is not None:
            self._on_nav(-1)

    def action_cursor_down(self) -> None:
        if self._on_nav is not None:
            self._on_nav(1)

    # -- composition --------------------------------------------------------
    def show(
        self,
        view: dict[str, Any],
        *,
        own_session: str | None,
        selected: int | None = None,
        style_for: StyleFor | None = None,
    ) -> None:
        """(Re)build the page for one composed project view.

        ``selected`` keeps the cursor across a refresh; a fresh entry lands on
        the first selectable row. ``style_for`` re-binds the row styles for
        every show: the resolver is built AT THIS MOMENT, so a theme switch
        under an open page repaints in the new palette on the next re-show
        (UX round 1, U2 — the org chart's rule, which the constructor-only
        capture had defeated).
        """
        if style_for is not None:
            self._style_for = style_for
        self._view = view
        self._own_session = own_session
        # The rows' provenance, captured in the SAME snapshot the rows are
        # built from: the id and name every row verb must target (QA round 1,
        # Q1 — a retarget once let `↵` write into the project under the
        # cursor while the body showed another).
        project_value = view.get("project")
        project = project_value if isinstance(project_value, dict) else {}
        self._project_id = str(project.get("id") or "") or None
        self._project_name = str(project.get("name") or "")
        children = self._build(view, own_session)
        self.remove_children()
        self.mount_all(children)
        self._selectables = [row for row in children if getattr(row, "selectable", False)]
        self._selected = max(
            0, min(selected if selected is not None else 0, max(len(self._selectables) - 1, 0))
        )
        self._restyle()
        self.call_after_refresh(self._reveal_selected)
        self.call_after_refresh(self._fit_meta)

    def _build(self, view: dict[str, Any], own_session: str | None) -> list[DetailRow]:
        project_value = view.get("project")
        project = project_value if isinstance(project_value, dict) else {}
        rows: list[DetailRow] = []

        rows.append(DetailHeadingRow("overview", None, self._style_for, gap=False))
        rows.append(DetailMetaRow(view, self._style_for))

        rows.append(DetailHeadingRow("description", None, self._style_for))
        rows.append(DetailProseRow(str(project.get("description") or ""), self._style_for))

        rows.extend(self._update_rows(project))

        milestones = [m for m in project.get("milestones") or [] if isinstance(m, dict)]
        done = sum(1 for m in milestones if m.get("completed_at"))
        rows.append(
            DetailHeadingRow(
                "milestones",
                f"{done}/{len(milestones)}" if milestones else None,
                self._style_for,
            )
        )
        if milestones:
            for milestone in milestones:
                rows.append(
                    DetailMilestoneRow(
                        milestone,
                        section_label="milestones",
                        on_action=self._on_action,
                        style_for=self._style_for,
                    )
                )
        else:
            rows.append(DetailSentenceRow("no milestones set", self._style_for))

        rows.append(DetailHeadingRow("todos", None, self._style_for))
        for line in detail_todo_lines(view, own_session=own_session, style_for=self._style_for):
            rows.append(DetailTodoRow(line))

        sessions = view.get("sessions") if isinstance(view.get("sessions"), list) else []
        rows.append(
            DetailHeadingRow("sessions", str(len(sessions)) if sessions else None, self._style_for)
        )
        if sessions:
            for session_row in sessions:
                if not isinstance(session_row, dict):
                    continue
                rows.append(
                    DetailSessionRow(
                        session_row,
                        own_session=own_session,
                        section_label="sessions",
                        on_action=self._on_action,
                        style_for=self._style_for,
                    )
                )
        else:
            rows.append(DetailSentenceRow("no sessions linked", self._style_for))
        return rows

    def _update_rows(self, project: dict[str, Any]) -> list[DetailRow]:
        """The ``updates`` section: day-grouped entries, newest first (spec §7.3).

        The store keeps the log oldest-first; the page reverses it and groups
        by LOCAL day, because the feed is read as a diary. An entry's body is
        clamped (``↵`` on its stamp opens the rest), and its attachments are
        child rows carrying the kind, the size and the path. The render cap
        keeps one trailing row naming whatever the page did not draw — the
        store still has it.
        """
        entries = [e for e in project.get("updates") or [] if isinstance(e, dict)]
        rows: list[DetailRow] = [
            DetailHeadingRow("updates", str(len(entries)) if entries else None, self._style_for)
        ]
        if not entries:
            rows.append(DetailSentenceRow("no updates recorded yet", self._style_for))
            return rows
        newest_first = list(reversed(entries))
        shown = newest_first[:UPDATES_PER_PAGE]
        ordinal_of = {id(entry): position for position, entry in enumerate(newest_first)}
        day = ""
        for entry in shown:
            label = update_day_label(str(entry.get("at") or ""))
            if label != day:
                day = label
                rows.append(DetailDayRow(f"── {label} ──", self._style_for))
            index = ordinal_of[id(entry)]
            expanded = index in self._expanded
            rows.append(
                DetailUpdateStampRow(
                    entry,
                    index=index,
                    expanded=expanded,
                    on_toggle=self._toggle_entry,
                    style_for=self._style_for,
                )
            )
            body = str(entry.get("text") or "")
            rows.append(DetailUpdateBodyRow(body, expanded=expanded, style_for=self._style_for))
            hidden = len(update_body_lines(body)) - UPDATE_BODY_LINES
            if hidden > 0:
                rows.append(
                    DetailUpdateMarkerRow(hidden, expanded=expanded, style_for=self._style_for)
                )
            attachments = [a for a in entry.get("attachments") or [] if isinstance(a, dict)]
            for attachment in attachments:
                rows.append(
                    DetailAttachmentRow(
                        attachment, on_action=self._on_action, style_for=self._style_for
                    )
                )
                rows.append(DetailAttachmentPathRow(attachment, self._style_for))
        older = len(newest_first) - len(shown)
        if older > 0:
            rows.append(DetailSentenceRow(older_updates_text(older).plain, self._style_for))
        return rows

    def _toggle_entry(self, index: int) -> None:
        """`↵` on an entry's stamp: show or hide the clamped tail (spec §7.3).

        Pure VIEW state, held here rather than in the host: nothing is written
        and no store read is needed, so the page stays the I/O-free renderer
        it is. The re-show keeps the row cursor — the selectable order does not
        change when a body's tail appears.
        """
        if index in self._expanded:
            self._expanded.discard(index)
        else:
            self._expanded.add(index)
        if self._view is not None:
            self.show(
                self._view,
                own_session=self._own_session,
                selected=self._selected,
                style_for=self._style_for,
            )

    # -- cursor -------------------------------------------------------------
    def painted_rows(self) -> list[str]:
        """The page as plain strings, one entry per child — assertable.

        The view's ``rendered_rows`` reads chrome through ``Static.content``;
        this is the same accessor over this page's children, so tests and
        evidence read the rows the page actually holds rather than a second
        rendering of them.
        """
        rows: list[str] = []
        for child in self.children:
            # Rows may carry their own plain readback (the prose row's rich
            # ``Markdown`` has no ``.plain``) — duck-typed, so narrow by hand.
            readback = getattr(child, "readback", None)
            value: str | None = None
            if callable(readback):
                result = readback()
                if result is not None:
                    value = str(result)
            if value is None:
                for candidate in (getattr(child, "content", None), child.render()):
                    plain = getattr(candidate, "plain", None)
                    if plain is not None:
                        value = str(plain)
                        break
            rows.append(value if value is not None else "")
        return rows

    @property
    def selectable_count(self) -> int:
        """Rows the cursor can land on (the arming rule for `↑↓ move`)."""
        return len(self._selectables)

    @property
    def selected_index(self) -> int:
        """Where the cursor sits among selectables — the refresh anchor."""
        return self._selected

    def selected_action_label(self) -> str | None:
        row = self._current()
        return row.action_label() if row is not None else None

    def selected_action_verb(self) -> str | None:
        """The bare verb — the ``↵`` hint's fallback label (UX round 1, U3)."""
        row = self._current()
        return row.action_verb() if row is not None else None

    def _current(self) -> DetailRow | None:
        if not self._selectables:
            return None
        return self._selectables[self._selected]

    def move(self, delta: int) -> None:
        if not self._selectables:
            return
        position = max(0, min(self._selected + delta, len(self._selectables) - 1))
        if position == self._selected:
            return
        self._selected = position
        self._restyle()
        self._reveal_selected()

    def jump_to_section(self, direction: int) -> None:
        """Cursor to the next/previous section's first selectable row (clamped)."""
        if not self._selectables:
            return
        order: list[str] = []
        for row in self._selectables:
            if row.section_label not in order:
                order.append(row.section_label)
        current = self._selectables[self._selected]
        if current.section_label not in order:
            return
        index = order.index(current.section_label)
        target = index + direction
        if not 0 <= target < len(order):
            return
        wanted = order[target]
        for position, row in enumerate(self._selectables):
            if row.section_label == wanted:
                if position != self._selected:
                    self._selected = position
                    self._restyle()
                    self._reveal_selected()
                return

    def activate(self) -> None:
        row = self._current()
        if row is not None:
            row.activate()

    def _restyle(self) -> None:
        for position, row in enumerate(self._selectables):
            row.set_selected(position == self._selected)

    def _reveal_selected(self) -> None:
        row = self._current()
        if row is None:
            return
        try:
            row.scroll_visible(animate=False)
        except Exception:  # noqa: BLE001 — a reveal is a bonus, never a failure
            return
        # Textual's ``scroll_visible`` treats a row sitting on the viewport's
        # bottom EDGE as visible, while the painted rows stop one cell earlier
        # — measured: at 60x24 the page's own entry row stayed one row below
        # the box, so the feed looked empty until a key was pressed. The clamp
        # below is the difference, and it only ever moves by the row or two
        # the boundary case is off by.
        box = self.region.height
        if box <= 0 or row.region.height <= 0:
            return
        try:
            top = int(self.scroll_offset.y)
            bottom = top + box - 1  # the LAST painted row
            if row.region.y > bottom:
                self.scroll_to(y=row.region.y - box + 1, animate=False)
            elif row.region.y < top:
                self.scroll_to(y=row.region.y, animate=False)
        except Exception:  # noqa: BLE001 — a reveal is a bonus, never a failure
            pass

    def reveal_selected(self) -> None:
        """Scroll the cursor's row into view — the view's DEFERRED re-reveal.

        ``show`` reveals once, but a page being ENTERED is still hidden while
        it builds, so its rows have no regions yet and that reveal is a no-op;
        the first layout can then leave the selected row below the fold.
        Measured at 60x24: the selected entry sat one row under the box, so
        the feed looked empty until the reader pressed a key. The canvas
        solves the same problem with a deferred second pass (``load``'s
        ``_scroll_cursor_into_view``); this is that pass for the page.
        """
        self._reveal_selected()

    # -- ruler anchors ------------------------------------------------------
    def section_anchors(self) -> list[tuple[int, str, str | None]]:
        """``(start_row, label, count)`` per section, in paint order.

        ``start_row`` is the heading's own row in the scrollable content's
        coordinate space: ``child.region.y`` is SCREEN-translated (it slides
        with the scroll), so the settled offset is added back — measured: the
        description heading's ``region.y`` moves 6 → −4 across offsets 2 → 12
        while ``content_region.y`` stays put, so region-minus-content alone is
        scroll-DEPENDENT and the ruler read a later section than the one the
        viewport actually shows (design review round 1, D1). No ``max(0, …)``:
        a settled content row is ≥ 0 by construction and clamping was what let
        the wrong reading look plausible. A ``scroll_y`` watch can fire BEFORE
        the children move — the caller repaints after the refresh, not inside
        the watcher (``ProjectsView._detail_scroll_changed``). Unlaid rows
        (region height 0) answer the empty list, so the caller paints the
        plain rule for that frame instead of a guess.
        """
        anchors: list[tuple[int, str, str | None]] = []
        base = self.content_region.y
        offset = int(self.scroll_offset.y)
        for child in self.children:
            section = getattr(child, "section", None)
            if section is None:
                continue
            if child.region.height <= 0:
                return []
            anchors.append((child.region.y - base + offset, section[0], section[1]))
        return anchors

    def _fit_meta(self) -> None:
        width = self.content_size.width
        if width <= 0:
            return
        for child in self.children:
            if isinstance(child, DetailMetaRow):
                child.fit_width(width)
            elif isinstance(child, DetailAttachmentPathRow):
                # The path is a whole-line fit like the meta sentence: both
                # must be cut at the box, because a Static wraps what it is
                # given (see `attachment_path_text`).
                child.fit_width(width)

    def on_resize(self) -> None:
        self.call_after_refresh(self._fit_meta)

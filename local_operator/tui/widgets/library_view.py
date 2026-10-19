"""The full-page ``/library`` view: the machine's global asset catalog.

A MODE of the main screen, cloned from :class:`SettingsView` and its siblings
(``/schedule``, ``/review``, ``/import``): the page takes the transcript's
region, leaves the dock in place and inert, and owns its own cursor and keys.
The layout reuses the settings page's CSS classes (title/rule/columns/body/
pane/detail/hints) rather than adding a second full-page skeleton, and its own
``.library-view*`` block only adds the third column the settings page does not
have.

WHAT THIS PAGE OWNS, AND WHAT IT DOES NOT. Rows, counts, tags and the detail
projection all come from :mod:`local_operator.library.service` — the one place
that decides what a row means. The page never scans a journal, never opens the
metadata database on the event loop and never derives an id from a path: it
sends a :class:`~local_operator.library.service.Query` from a worker thread and
paints the payload it gets back. Every write goes through the service's own
``update_asset``/``set_tags``, so the provenance rules stay in one module.

WHY A WORKER EVEN THOUGH THE PAGE IS LOCAL. A first catalog build walks every
session journal under a time budget; on a machine with hundreds of sessions
that is seconds of file I/O. Running it on the loop would freeze the frame
before the page's own first paint, which is the defect the host-resolved
payload pattern avoids everywhere else. The page therefore starts in a LOADING
state whose first frame is honest, and the worker replaces it. A ``building``
answer is not an error: the index persists its progress, so the page
re-requests after a short pause and the status line carries the scanner's own
progress counts.

THREE-NINE-NINE LAYOUT. The rail (categories, status, tags) is hidden below 90
columns, and the detail/preview column below 70: a 24-cell rail beside a
20-cell list is the bug the narrow captures on the sibling pages exist to
catch. The list is always the subject.

FULL-PAGE LISTS CLAMP. ``j/k`` and the arrows stop at the ends rather than
wrapping — the documented exception shared with ``/settings``, ``/project show``
and ``/schedule``: a ledger a reader scans has a bottom that is a destination.

PREVIEW IS AN INLINE OPPORTUNITY, NOT A PROMISE. ``space`` shows pixels through
the transcript's own :class:`ImageBlock` (kitty, then half-cells, then a
one-row receipt), renders markdown through the same Rich renderer the projects
detail uses, and shows text/code as text. Everything else says so and leaves
``enter`` as the action that works — the design's unsupported cell is a designed
state, never a blank pane. Reading those bytes is I/O, so it happens in a
worker too.

INLINE IMAGE PREVIEW AND THE COMPOSER. This page never duplicates the editor.
``a`` leaves the mode and the app seeds the ONE composer with an asset
reference, so the chat that follows is the ordinary conversation with the
ordinary draft semantics.
"""

from __future__ import annotations

import asyncio
import importlib
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rich.cells import cell_len
from rich.markdown import Markdown
from rich.style import Style
from rich.text import Text
from textual import work
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Input, Static, TextArea

from local_operator.i18n import keys as i18n_keys
from local_operator.i18n.text import tr
from local_operator.library import metadata as library_metadata
from local_operator.library import service as library_service
from local_operator.paths import config_dir
from local_operator.tui import attachments, library_render
from local_operator.tui import theme as theme_mod
from local_operator.tui.library_render import LibraryState
from local_operator.tui.widgets.image_block import ImageBlock
from local_operator.tui.widgets.subagent_view import HintButton
from local_operator.tui.widgets.tool_card import truncate_cells

_t = tr("tui.library")
_k = i18n_keys.tui_library

#: The page's own chrome, not copy: glyphs no locale translates.
_CURSOR = "›"
_ACTIVE = "•"
_RULE_GLYPH = "─"

#: The width the rail needs. Below it the three columns cannot each hold a
#: usable row, and the list is the subject (SettingsView draws the same line at
#: 90 for its two columns).
RAIL_MIN_WIDTH = 90

#: The detail column needs a readable name + a few meta rows. Under this the
#: list takes the whole page; selecting a row still works and `enter` still
#: opens it, so the page degrades without hiding its verbs.
DETAIL_MIN_WIDTH = 80

#: The detail column at a middle width. 40 cells is comfortable at a desk but
#: leaves a 24-cell list at 86 columns, where every file name ellipsizes; 32
#: keeps the detail readable and the list the subject.
DETAIL_NARROW_WIDTH = 30

#: Below this the detail column takes its full 40.
DETAIL_WIDE_WIDTH = 110

#: The scanner's per-pass budget. The index persists progress, so the page can
#: ask again rather than hold the first frame for the whole cold build.
BUILD_BUDGET_S = 1.5
#: How many budgeted passes one request chains before it hands control back to
#: the page's own re-request timer. Bounded so a pathological store cannot pin
#: one worker for minutes.
BUILD_PASSES = 6
#: Pause before the page asks a building catalog to continue.
BUILD_RETRY_S = 0.35

#: Read caps for an inline text/markdown preview. A preview is a glance, not a
#: pager; the file's real path is what `enter` hands to the OS.
PREVIEW_TEXT_BYTES = 256 * 1024

#: How long a transient notice owns the status line before the librarian
#: summary comes back. Long enough to read, short enough not to hide the
#: footer's standing facts for a whole session.
NOTICE_SECONDS = 5.0


def _selection_ground() -> Style:
    """The app's one selection ground (sidebar, pickers, ToolCard)."""
    try:
        return Style(bgcolor=theme_mod.semantic_color("tint-select"))
    except KeyError:  # pragma: no cover - every shipped ramp carries it
        return Style()


def _hint(key: str, action: Any) -> HintButton:
    """One footer hint in the page's readable ink.

    The widget's DEFAULTS are ``dim``/``faint``. Measured against this app's
    two shipped ramps: ``faint`` is 1.97:1 on dark and 2.08:1 on light — the
    hint labels would be decoration, not copy. The review page made the same
    correction (muted at rest, fg on hover), and both pairs clear AA on the
    two themes the evidence set covers.
    """
    return HintButton(key, action, ink=("muted", "muted"), lifted_ink=("fg", "fg"))


def _row_meta_ink(selected: bool) -> str:
    """`fg` on the selected ground, `muted` elsewhere (the schedule page's rule).

    ``muted`` on ``tint-select`` inherits palette debt the page does not own;
    the one rung brighter is what keeps the selected row's second half legible
    in every palette the app ships.
    """
    return "fg" if selected else "muted"


class _ChromeStatic(Static):
    """A ``Static`` the drag-to-select walk skips (``ALLOW_SELECT = False``).

    The same class the settings page defines for the same reason: the page is
    chrome, and a click that moves the cursor must not also arm a screen text
    selection whose grey band bleeds across rows.
    """

    ALLOW_SELECT = False


class LibraryViewDismissed(Message):
    """``esc`` on the list: the app restores the conversation."""


class LibraryViewAskRequested(Message):
    """``a``: seed the ONE composer with a question about ``row``."""

    def __init__(self, row: Mapping[str, Any]) -> None:
        super().__init__()
        self.row = dict(row)


class LibraryViewLibrarianRequested(Message):
    """``L``: open the Librarian's pinned conversation."""


@dataclass
class _RailItem:
    """One selectable row (or a header) in the filter rail."""

    kind: str  # "header" | "category" | "status" | "tag"
    value: str
    label: str
    count: int | None = None


@dataclass
class _FormState:
    """The metadata form's values, held by the page rather than the store.

    ``asset_id`` is the form's IDENTITY, captured when it opened. The cursor is
    a viewport position, not an identity: an in-flight reload can replace or
    reorder the rows while the form is up, and resolving the cursor again at
    save time would write these fields to whichever asset happened to land
    under it (review round 1, B1).
    """

    asset_id: str = ""
    name: str = ""
    description: str = ""
    tags: str = ""
    error: str = ""
    saving: bool = False


class LibraryView(Vertical):
    """The page: title + search, rule, three columns, status line, hints.

    Class-identified (see the module docstring). ``can_focus`` so the page's
    keys, the list cursor and the rail cursor land here rather than on the
    composer the mode made inert.
    """

    can_focus = True

    BINDINGS = [
        # Arrows CLAMP (see the module docstring): the list is a ledger, not a
        # picker circling a short overlay.
        Binding("j,down", "cursor_down", "down", show=False),
        Binding("k,up", "cursor_up", "up", show=False),
        Binding("pagedown", "page_down", "page down", show=False),
        Binding("pageup", "page_up", "page up", show=False),
        Binding("home", "cursor_home", "first", show=False),
        Binding("end", "cursor_end", "last", show=False),
        Binding("enter", "activate", "open", show=False),
        Binding("space", "toggle_preview", "preview", show=False),
        Binding("slash", "focus_search", "search", show=False),
        Binding("s", "cycle_sort", "sort", show=False),
        Binding("f", "focus_rail", "filter", show=False),
        Binding("e", "edit_metadata", "edit", show=False),
        Binding("a", "ask_asset", "ask", show=False),
        Binding("L", "open_librarian", "librarian", show=False),
        Binding("r", "reload", "reload", show=False),
        Binding("escape", "leave", "back", show=False),
    ]

    def __init__(self, config_root: str | Path | None = None) -> None:
        super().__init__(classes="library-view settings-view")
        self._config_dir = Path(config_root) if config_root else Path(config_dir())
        #: The last service answer, or ``None`` before the first load. Painting
        #: reads ONLY this and the state below; no handler here touches disk.
        self._payload: dict[str, Any] | None = None
        #: The page's controls. Query changes go through the worker.
        self.state = LibraryState()
        #: The cursor inside the RAIL; kept separately from the list cursor so
        #: `f` and `j/k` never move the wrong subject.
        self._rail_cursor = 0
        self._rail_items: list[_RailItem] = []
        self._rail_focused = False
        self._form = _FormState()
        self._form_open = False
        #: Transient sentences (open failures, save receipts). ``_notice_at``
        #: is what lets the status line fall back to the librarian summary
        #: without a second timer racing this one.
        self._notice_text = ""
        self._notice_tone = "info"
        self._notice_at = 0.0
        #: The pending search debounce and building re-request timers.
        self._search_timer: Any | None = None
        self._building_timer: Any | None = None
        #: Whether the host has already started the first load (``start``).
        #: ``on_mount`` starts it otherwise, so a directly-mounted page still
        #: loads; the app calls ``start`` before the deferred mount handler so
        #: the first frame is the loading state rather than an empty list.
        self._started = False
        #: The mounted preview's widget, so a toggle removes the ONE it made
        #: rather than sweeping the detail column (which also holds the meta).
        self._preview_widget: Any | None = None
        self._preview_for: str | None = None

        self._title = _ChromeStatic(classes="library-view-title")
        self._search = Input(
            placeholder=_t(_k.search_placeholder()),
            id="library-view-search",
            classes="library-view-search",
        )
        self._rule = _ChromeStatic(classes="library-view-rule")
        self._rail_canvas = _ChromeStatic(classes="library-view-rail-canvas")
        self._rail = ScrollableContainer(self._rail_canvas, classes="library-view-rail")
        self._list_canvas = _ChromeStatic(classes="library-view-list-canvas")
        self._list = ScrollableContainer(self._list_canvas, classes="library-view-list")
        self._detail_canvas = _ChromeStatic(classes="library-view-detail-canvas")
        self._preview_host = Vertical(classes="library-view-preview-host")
        self._detail = VerticalScroll(
            self._detail_canvas, self._preview_host, classes="library-view-detail"
        )
        self._status = _ChromeStatic(classes="library-view-status")
        # Form: name, description, tags. The labels are Static rows rather than
        # inputs' placeholders so a filled field still says what it is.
        self._name_input = Input(
            placeholder=_t(_k.form_name_placeholder()),
            id="library-view-name",
            classes="library-view-form-input",
            # A form opens to be EDITED, not replaced wholesale. Textual's
            # default selects the whole value on focus, which both repainted
            # the filename as pale-on-pale-blue in the light ramp and made the
            # first keystroke destroy a name the reader only meant to append
            # to (design round 1).
            select_on_focus=False,
        )
        self._description = TextArea(
            id="library-view-description", classes="library-view-form-description"
        )
        self._tags_input = Input(
            placeholder=_t(_k.form_tags_placeholder()),
            id="library-view-tags",
            classes="library-view-form-input",
            select_on_focus=False,
        )
        self._form_hint = Static("", classes="library-view-form-hint", markup=False)
        self._form_error = Static("", classes="library-view-form-error", markup=False)
        #: Filled by :meth:`compose`; the columns are built there so the form's
        #: fields are mounted WITH the page rather than in a second pass. A
        #: `mount` from the page's own ``on_mount`` is too early — Textual has
        #: not mounted the container yet (``MountError``), and a page whose
        #: form cannot mount is a page that cannot edit.

        # Footer hints. The key is the affordance; the label is shed first on a
        # narrow page (the schedule page's ladder), because the key alone is
        # what a reader learns once.
        self._hint_move = _hint("↑↓", self.action_cursor_down)
        self._hint_open = _hint("enter", self.action_activate)
        self._hint_preview = _hint("space", self.action_toggle_preview)
        self._hint_edit = _hint("e", self.action_edit_metadata)
        self._hint_ask = _hint("a", self.action_ask_asset)
        self._hint_search = _hint("/", self.action_focus_search)
        self._hint_sort = _hint("s", self.action_cycle_sort)
        self._hint_filter = _hint("f", self.action_focus_rail)
        self._hint_librarian = _hint("L", self.action_open_librarian)
        self._hint_reload = _hint("r", self.action_reload)
        self._hint_back = _hint("esc", self.action_leave)
        self._hint_save = _hint("enter", self.action_save_form)
        self._hint_cancel = _hint("esc", self.action_cancel_form)
        self._hint_order = [
            self._hint_move,
            self._hint_open,
            self._hint_preview,
            self._hint_edit,
            self._hint_ask,
            self._hint_search,
            self._hint_sort,
            self._hint_filter,
            self._hint_librarian,
            self._hint_reload,
            self._hint_save,
            self._hint_cancel,
            self._hint_back,
        ]
        self._hints = Horizontal(*self._hint_order, classes="library-view-hints")

    # -- composition -------------------------------------------------------

    def compose(self):  # type: ignore[override]
        self._title_row = Horizontal(self._title, self._search, classes="library-view-title-row")
        yield self._title_row
        yield self._rule
        # The right column owns both faces: detail/preview by default, the
        # metadata form while it is open. The form must not be a fourth row
        # under the columns — that shrinks the list to make room for a field,
        # which is how a page loses its subject while it is edited.
        with Horizontal(classes="library-view-columns") as columns:
            yield self._rail
            yield self._list
            with Vertical(classes="library-view-right") as right:
                yield self._detail
                with VerticalScroll(classes="library-view-form") as form:
                    yield Static(_t(_k.form_name()), classes="library-view-form-label")
                    yield self._name_input
                    yield Static(_t(_k.form_description()), classes="library-view-form-label")
                    yield self._description
                    yield Static(_t(_k.form_tags()), classes="library-view-form-label")
                    yield self._tags_input
                    yield self._form_hint
                    yield self._form_error
        yield self._status
        yield self._hints
        self._columns = columns
        self._right = right
        self._form_block = form
        form.display = False

    def on_mount(self) -> None:
        self._paint_form()
        self.focus()
        self._paint_all()
        # Same frame-late reason as `on_resize`: the first paint can precede
        # the column widths, and the settled pass is the one a reader sees.
        self.call_after_refresh(self._paint_all)
        if not self._started:
            self._reload()

    def on_resize(self, event: Any) -> None:
        # The EVENT's size, not `self.size`: a resize event is delivered before
        # the widget's own layout box is updated, and painting the hint ladder
        # against the stale width dropped four verbs at 86 columns (measured).
        self._paint_all(width=event.size.width, height=event.size.height)
        # Then ONE pass after the containers settle. The first pass sees the
        # new PAGE width but the OLD column widths (the rail it just hid still
        # owned cells), so the rows were truncated for a 23-cell list and the
        # harness gap stayed blank for the list's real 51 (measured at 86x30).
        self.call_after_refresh(self._paint_all)

    # -- data --------------------------------------------------------------

    def load(self, payload: Mapping[str, Any]) -> None:
        """Seed the page with an already-resolved payload (tests, host preload).

        The production path is :meth:`_load_worker`; this seam exists because
        the SAME painting code must be drivable without a config root, and a
        test that hand-built a frame would not prove the payload contract.
        """
        self._apply_payload(dict(payload))

    def start(self) -> None:
        """Begin the first load from the HOST, before the deferred repaint.

        The app calls this immediately after mounting the page. ``on_mount``
        loads too when nobody called it first (a directly-mounted page in a
        test), so this is an ordering seam rather than the only entry point.
        """
        if self._started:
            return
        self._started = True
        self._reload()

    def _reload(self) -> None:
        if self._building_timer is not None:
            self._building_timer.stop()
            self._building_timer = None
        self._load_worker(self.state.query())

    @work(exclusive=True, group="library-load")
    async def _load_worker(self, query: library_service.Query) -> None:
        payload = await asyncio.to_thread(self._fetch_blocking, query)
        if not self.is_attached:
            return
        self._apply_payload(payload)
        if payload.get("state") == "building":
            # Progress is persisted, so the next budgeted pass continues where
            # this one stopped. A timer rather than a loop keeps the loop free
            # and makes a later query cancel the chain through `_reload`.
            self._building_timer = self.set_timer(BUILD_RETRY_S, self._reload)

    def _fetch_blocking(self, query: library_service.Query) -> dict[str, Any]:
        """One service answer plus the two standing facts the footer shows.

        Runs on the worker thread. Every call is the service's own blocking
        entry point; nothing here re-implements a filter or a sort.
        """
        payload: dict[str, Any]
        try:
            payload = library_service.list_assets(
                self._config_dir,
                query,
                display=False,
                budget_s=BUILD_BUDGET_S,
            )
        except Exception as error:  # noqa: BLE001 — a page must render a refusal
            return {
                "state": "error",
                "error": str(error),
                "rows": [],
                "total": 0,
                "counts": {},
                "tags": [],
                "scan": {},
            }
        try:
            payload["stats"] = library_metadata.MetadataStore(self._config_dir).stats()
        except Exception:  # noqa: BLE001 — a stats failure must not hide the rows
            payload["stats"] = {}
        payload["librarian"] = self._librarian_state()
        return payload

    @staticmethod
    def _librarian_state() -> dict[str, Any] | None:
        """The Librarian's own status, or ``None`` where the module is absent.

        The Librarian work package OWNS :mod:`local_operator.librarian`; this
        page only reads the optional ``status()`` hook it publishes. Import is
        lazy so this branch (which carries the backend core only) still opens
        the Library with an honest "unavailable" footer instead of an import
        error at module load. The hook runs on the worker thread.
        """
        try:
            librarian = importlib.import_module("local_operator.librarian")
        except Exception:  # noqa: BLE001 — absence is a state, not a failure
            return None
        status = getattr(librarian, "status", None)
        if not callable(status):
            return None
        try:
            answer = status()
        except Exception as error:  # noqa: BLE001 — the footer reports, never raises
            return {"error": str(error)}
        return dict(answer) if isinstance(answer, Mapping) else None

    def _apply_payload(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        rows = list(payload.get("rows") or [])
        self.state.clamp(len(rows))
        self._rail_items = self._build_rail(payload)
        self._rail_cursor = max(0, min(self._rail_cursor, max(len(self._rail_items) - 1, 0)))
        self._paint_all()

    # -- payload helpers ---------------------------------------------------

    def _when(self, stamp: Any) -> str:
        """Relative time with the page's translated "now" label."""
        return library_render.format_when(stamp, now_label=_t(_k.time_now()))

    def rows(self) -> list[dict[str, Any]]:
        payload = self._payload or {}
        if payload.get("state") == "error":
            # A failed read paints the error in the list; leaving the last
            # successful payload's detail beside it would show a row the
            # page is no longer claiming to know.
            return []
        rows = payload.get("rows")
        return list(rows) if isinstance(rows, list) else []

    def selected(self) -> dict[str, Any] | None:
        rows = self.rows()
        if not rows:
            return None
        return rows[max(0, min(self.state.cursor, len(rows) - 1))]

    # -- painting ----------------------------------------------------------

    def _paint_all(self, *, width: int | None = None, height: int | None = None) -> None:
        # A worker can answer before ``compose`` has produced the columns when
        # the host called `start` first. The `on_mount` pass repaints, so
        # dropping this one frame costs nothing and paints nothing into a
        # tree that does not exist yet.
        if not self.is_attached or not hasattr(self, "_columns"):
            return
        page_width = width or self.size.width or 80
        page_height = height or self.size.height or self.screen.size.height or 24
        self._layout(page_width, page_height)
        self._paint_title()
        self._paint_rail()
        self._paint_list()
        self._paint_detail()
        self._paint_status()
        self._paint_hints(page_width)

    def _layout(self, width: int | None = None, height: int | None = None) -> None:
        """Resolve the three-column ladder against the page width.

        The rail goes first (90), then the detail column (80), and the detail
        narrows from 40 to 32 below 110. While the form is open the right
        column is the SUBJECT: on a narrow page the list yields to it rather
        than the form being squeezed into 40 cells beside a 20-cell list.
        """
        width = width or self.size.width or 80
        # `action_edit_metadata` enters the form before the page's own box has
        # necessarily been resolved, so `self.size.height` can still be zero
        # at that call. The SCREEN is the available height either way; without
        # this fallback a 120x34 form was misread as a short page and lost its
        # title (measured in the round-1 re-capture).
        height = height or self.size.height or self.screen.size.height or 24
        narrow_rail = width < RAIL_MIN_WIDTH
        narrow_detail = width < DETAIL_MIN_WIDTH
        self._rail.display = not narrow_rail
        self._right.display = self._form_open or not narrow_detail
        self._form_block.display = self._form_open
        self._detail.display = not self._form_open
        self._list.display = not (self._form_open and narrow_detail)
        if self._form_open and narrow_detail:
            self._right.styles.width = "1fr"
        else:
            self._right.styles.width = (
                "36" if width >= DETAIL_WIDE_WIDTH else str(DETAIL_NARROW_WIDTH)
            )
        # A SHORT page with the form open has no rows to spare: at 60x20 under
        # a multi-line composer the columns region was ONE row high, and no
        # scroll container can reveal a field through a one-row viewport (UX
        # round 1, U2). The form is the subject while it is open, so the
        # page's title, rule and standing footer yield their rows to it. The
        # hints stay (save/cancel are the actions) but lose their lead-in.
        # The page's own box (25 rows at 120x34 once the dock is subtracted)
        # is the budget, not the SCREEN height: 25 rows fit the 15-row form
        # beside the title and rule. Only genuinely short pages trade the
        # chrome for form rows.
        short_form = self._form_open and height < 22
        self._title_row.display = not short_form
        self._rule.display = not short_form
        self._status.display = not short_form
        self._hints.set_class(short_form, "library-view-hints-short")

    def _paint_title(self) -> None:
        payload = self._payload or {}
        total = int(payload.get("total") or 0)
        parts: list[str] = [_t(_k.title()), _t(_k.asset_count(count=total))]
        if self.state.category:
            parts.append(self._category_label(self.state.category))
        if self.state.status:
            parts.append(self._status_label(self.state.status))
        if self.state.tag:
            parts.append(f"#{self.state.tag_label or self.state.tag}")
        if self.state.q.strip():
            parts.append(self.state.q.strip())
        parts.append(_t(_k.sort_label(sort=self._sort_label(self.state.sort))))
        joined = " · ".join(parts)
        width = max(10, (self._title.size.width or self.size.width or 80) - 1)
        self._title.update(Text(truncate_cells(joined, width), style=Style(bold=True)))

    def _paint_rail(self) -> None:
        if self._payload is None:
            self._rail_canvas.update(Text(_t(_k.loading()), style=Style(dim=True)))
            return
        if not self._rail_items:
            self._rail_canvas.update(Text())
            return
        width = max(10, (self._rail.size.width or 24) - 2)
        header_ink = Style(color=theme_mod.semantic_color("muted"))
        ground = _selection_ground()
        text = Text()
        for index, item in enumerate(self._rail_items):
            if item.kind == "header":
                text.append(truncate_cells(item.label.upper(), width), style=header_ink)
                text.append("\n")
                continue
            active = self._rail_active(item)
            cursor = self._rail_focused and index == self._rail_cursor
            ink = Style(
                color=theme_mod.semantic_color("fg" if active else "muted"),
                bgcolor=ground.bgcolor if cursor else None,
                bold=active,
            )
            marker = _CURSOR if cursor else (_ACTIVE if active else " ")
            text.append(f"{marker} ", style=ink)
            label_width = width - 2 - (4 if item.count is not None else 0)
            label = truncate_cells(item.label, max(1, label_width))
            text.append(label, style=ink)
            if item.count is not None:
                pad = max(1, width - 2 - cell_len(label) - len(str(item.count)))
                text.append(" " * pad, style=ink)
                text.append(str(item.count), style=ink)
            text.append("\n")
        self._rail_canvas.update(text)
        self._rail_canvas.styles.height = max(1, len(self._rail_items))
        self._scroll_rail_to_cursor()

    def _paint_list(self) -> None:
        payload = self._payload
        if payload is None:
            self._list_canvas.update(
                Text(_t(_k.loading()), style=Style(color=theme_mod.semantic_color("muted")))
            )
            self._list_canvas.styles.height = 1
            return
        if payload.get("state") == "error":
            self._paint_message(
                _t(_k.error_title()),
                f"{payload.get('error') or _t(_k.unknown())}  {_t(_k.error_retry())}",
                tone="danger",
            )
            return
        rows = self.rows()
        if not rows:
            scan = payload.get("scan") or {}
            if payload.get("state") == "building":
                total = int(scan.get("sessions_total") or 0)
                scanned = int(scan.get("sessions_scanned") or 0)
                title = (
                    _t(_k.building(scanned=str(scanned), total=str(total)))
                    if total
                    else _t(_k.building_short())
                )
                self._paint_message(title, _t(_k.empty_body()))
            elif self.state.filtered():
                self._paint_message(_t(_k.no_matches_title()), _t(_k.no_matches_body()))
            else:
                self._paint_message(_t(_k.empty_title()), _t(_k.empty_body()))
            return

        width = max(20, (self._list.size.width or self.size.width or 80) - 2)
        text = Text()
        for index, row in enumerate(rows):
            selected = index == self.state.cursor
            ground = _selection_ground() if selected else Style()
            head = Style(
                color=theme_mod.semantic_color(
                    "fg" if selected else ("danger" if not row.get("exists", True) else "muted")
                ),
                bgcolor=ground.bgcolor,
                bold=selected,
            )
            meta_ink = Style(
                color=theme_mod.semantic_color(_row_meta_ink(selected)),
                bgcolor=ground.bgcolor,
            )
            marker_style = Style(
                color=theme_mod.semantic_color("accent" if selected else "muted"),
                bgcolor=ground.bgcolor if selected else None,
                bold=selected,
            )
            marker = _CURSOR if selected else " "
            name = str(row.get("name") or row.get("derived_name") or row.get("id") or "")
            if not row.get("exists", True):
                name = f"{name}  {_t(_k.missing())}"
            content = max(8, width - 2)
            name_width = max(10, min(26, int(content * 0.40)))
            rest = max(0, content - name_width - 2)
            meta = self._row_meta(row, rest)
            # Name first, meta right after it: at 60 columns the name keeps two
            # thirds of the row so a file remains identifiable, and the meta
            # sheds from the RIGHT (size/time go first) in `_row_meta`.
            # Two cells are the marker; the rest is the row's text budget.
            text.append(f"{marker} ", style=marker_style)
            text.append(truncate_cells(name, name_width), style=head)
            if rest > 2:
                text.append("  ", style=head)
                text.append(truncate_cells(meta, rest), style=meta_ink)
            if index != len(rows) - 1:
                text.append("\n")
        self._list_canvas.update(text)
        self._list_canvas.styles.height = max(1, len(rows))
        self._scroll_list_to_cursor()

    def _paint_message(self, title: str, body: str, *, tone: str = "fg") -> None:
        """A full-column state message, WRAPPED to the list's measured width.

        Rich text in a ``Static`` is not guaranteed to wrap here — the first
        empty-state frame proved it: the sentence ran under the detail column
        and was clipped mid-word. Wrapping is done against the same width the
        rows are truncated to, and the canvas is pinned to the wrapped line
        count so the scroll container has the height to show all of it.
        """
        width = max(20, (self._list.size.width or self.size.width or 80) - 2)
        text = Text()
        text.append(title, style=Style(color=theme_mod.semantic_color(tone), bold=True))
        text.append("\n\n")
        lines = textwrap.wrap(body, width=width) or [""]
        for index, line in enumerate(lines):
            text.append(line, style=Style(color=theme_mod.semantic_color("muted")))
            if index != len(lines) - 1:
                text.append("\n")
        self._list_canvas.update(text)
        self._list_canvas.styles.height = 2 + len(lines)

    def _row_meta(self, row: Mapping[str, Any], width: int | None = None) -> str:
        """Kind · origin · size · time, shedding the least useful part first.

        A clipped TOKEN (``92.0…``) reads as a broken value, while a missing
        origin still leaves the kind, the size and the age. So the row drops
        ``origin`` before it lets the cell budget cut ``size`` or ``time``;
        clipping is the last resort, for a name so long nothing else fits.
        """
        kind = str(row.get("kind") or "other")
        origin = str(row.get("origin_group") or "")
        size = library_render.format_size(row.get("size_bytes"))
        when = self._when(row.get("modified_at") or row.get("created_at"))
        parts = [kind]
        if origin:
            parts.append(origin)
        if size and size != "—":
            parts.append(size)
        if when:
            parts.append(when)
        if width:
            for droppable in (origin, size):
                if cell_len(" · ".join(parts)) <= width:
                    break
                if droppable and droppable in parts:
                    parts.remove(droppable)
        return " · ".join(parts)

    def _paint_detail(self) -> None:
        row = self.selected()
        selected_id = str(row.get("id") or "") if row is not None else ""
        # A preview is an asset's CONTENT, not the pane's decoration. Once the
        # selection is a different asset (or none), the old bytes are no longer
        # what the heading names — clear them, including a still-loading
        # placeholder. A late worker result is dropped by `_show_preview`'s
        # own id check below.
        if self._preview_for is not None and self._preview_for != selected_id:
            self._remove_preview()
        if row is None:
            self._detail_canvas.update(Text())
            return
        text = Text()
        text.append(
            str(row.get("name") or row.get("derived_name") or row.get("id") or ""),
            style=Style(bold=True),
        )
        text.append("\n")
        if not row.get("exists", True):
            text.append(
                _t(_k.missing_file()), style=Style(color=theme_mod.semantic_color("danger"))
            )
            text.append("\n")
        description = str(row.get("description") or "").strip()
        text.append(
            description or _t(_k.detail_no_description()),
            style=Style(color=theme_mod.semantic_color("muted")),
        )
        text.append("\n\n")
        tags = row.get("tags") if isinstance(row.get("tags"), list) else []
        if tags:
            for tag in tags:
                name = tag.get("name") if isinstance(tag, Mapping) else None
                if not name:
                    continue
                text.append(f" #{name}", style=Style(color=theme_mod.semantic_color("accent")))
            text.append("\n\n")
        else:
            text.append(
                _t(_k.detail_no_tags()), style=Style(color=theme_mod.semantic_color("muted"))
            )
            text.append("\n\n")
        meta = _t(
            _k.detail_meta(
                kind=str(row.get("kind") or "other"),
                origin=str(row.get("origin_group") or _t(_k.unknown())),
                size=library_render.format_size(row.get("size_bytes")),
                when=self._when(row.get("modified_at") or row.get("created_at"))
                or _t(_k.unknown()),
            )
        )
        text.append(meta, style=Style(color=theme_mod.semantic_color("muted")))
        text.append("\n")
        occurrences = [
            item for item in list(row.get("occurrences") or []) if isinstance(item, Mapping)
        ]
        text.append(
            _t(
                _k.detail_sources(
                    count=len(
                        {str(o.get("session_id")) for o in occurrences if isinstance(o, Mapping)}
                    )
                )
            ),
            style=Style(color=theme_mod.semantic_color("muted")),
        )
        text.append("\n")
        for occurrence in occurrences[:3]:
            line = _t(
                _k.detail_occurrence(
                    session=str(
                        occurrence.get("session_title") or occurrence.get("session_id") or ""
                    ),
                    origin=str(occurrence.get("origin") or ""),
                    when=self._when(occurrence.get("at")) or "",
                )
            )
            text.append(
                truncate_cells(line, max(10, (self._detail.size.width or 38) - 2)),
                style=Style(color=theme_mod.semantic_color("muted")),
            )
            text.append("\n")
        if row.get("highlighted"):
            text.append(
                _t(_k.detail_highlighted()), style=Style(color=theme_mod.semantic_color("accent"))
            )
            text.append("\n")
        organized = row.get("organized")
        if isinstance(organized, Mapping) and organized.get("at"):
            text.append(
                _t(_k.detail_organized(when=self._when(organized.get("at")) or "")),
                style=Style(color=theme_mod.semantic_color("muted")),
            )
        else:
            text.append(
                _t(_k.detail_unorganized()), style=Style(color=theme_mod.semantic_color("muted"))
            )
        text.append("\n")
        text.append(self._preview_line(row), style=Style(color=theme_mod.semantic_color("muted")))
        chat = row.get("chat")
        if isinstance(chat, Mapping) and chat.get("title"):
            text.append("\n")
            text.append(
                _t(_k.detail_chat(title=str(chat.get("title")))),
                style=Style(color=theme_mod.semantic_color("muted")),
            )
        self._detail_canvas.update(text)

    def _preview_line(self, row: Mapping[str, Any]) -> str:
        if not row.get("exists", True):
            return _t(_k.detail_preview_missing())
        raw_preview = row.get("preview")
        preview: Mapping[str, Any] = raw_preview if isinstance(raw_preview, Mapping) else {}
        kind = str(row.get("kind") or "other")
        if preview.get("state") != "ready":
            return _t(_k.detail_preview_unsupported())
        if kind == "image":
            return _t(_k.detail_preview_image())
        if kind in ("markdown", "text", "code", "spreadsheet"):
            return _t(_k.detail_preview_text())
        return _t(_k.detail_preview_unsupported())

    def _paint_status(self) -> None:
        now = time.monotonic()
        if self._notice_text and now - self._notice_at < NOTICE_SECONDS:
            ink = "warning" if self._notice_tone == "warning" else "muted"
            self._status.update(
                Text(self._notice_text, style=Style(color=theme_mod.semantic_color(ink)))
            )
            return
        payload = self._payload or {}
        parts: list[str] = []
        if payload.get("state") == "building":
            total = int((payload.get("scan") or {}).get("sessions_total") or 0)
            scanned = int((payload.get("scan") or {}).get("sessions_scanned") or 0)
            parts.append(
                _t(_k.building(scanned=str(scanned), total=str(total)))
                if total
                else _t(_k.building_short())
            )
        state = payload.get("librarian")
        if isinstance(state, Mapping):
            if state.get("error"):
                parts.append(_t(_k.librarian_error(error=str(state.get("error")))))
            else:
                label = (
                    _t(_k.footer_paused())
                    if state.get("paused")
                    else (self._when(state.get("last_run")) or _t(_k.footer_never_run()))
                )
                parts.append(_t(_k.footer_librarian(state=label)))
        elif self._payload is not None:
            parts.append(_t(_k.footer_librarian_unavailable()))
        raw_stats = payload.get("stats")
        stats: Mapping[str, Any] = raw_stats if isinstance(raw_stats, Mapping) else {}
        parts.append(_t(_k.footer_organized(count=str(int(stats.get("organized") or 0)))))
        self._status.update(
            Text(" · ".join(parts), style=Style(color=theme_mod.semantic_color("muted")))
        )

    def _paint_hints(self, width: int | None = None) -> None:
        width = width or self.size.width or 80
        compact = width < 110
        very_narrow = width < 60
        if self._form_open:
            labels: dict[HintButton, str | None] = {
                self._hint_save: "" if compact else _t(_k.hint_save()),
                self._hint_cancel: "" if compact else _t(_k.hint_cancel()),
                self._hint_back: None,
                self._hint_move: None,
                self._hint_open: None,
                self._hint_preview: None,
                self._hint_edit: None,
                self._hint_ask: None,
                self._hint_search: None,
                self._hint_sort: None,
                self._hint_filter: None,
                self._hint_librarian: None,
                self._hint_reload: None,
            }
        elif (self._payload or {}).get("state") == "error":
            # The error state's ONE action is retry. Leaving the navigation
            # verbs on the row both buries the recovery and overflowed `esc
            # back` at 120 columns when retry was added to every state
            # (measured in the round-1 re-capture).
            labels = {
                self._hint_reload: "" if compact else _t(_k.hint_retry()),
                self._hint_back: "" if compact else _t(_k.hint_back()),
                self._hint_move: None,
                self._hint_open: None,
                self._hint_preview: None,
                self._hint_edit: None,
                self._hint_ask: None,
                self._hint_search: None,
                self._hint_sort: None,
                self._hint_filter: None,
                self._hint_librarian: None,
                self._hint_save: None,
                self._hint_cancel: None,
            }
        else:
            labels = {
                self._hint_move: "" if compact else _t(_k.hint_move()),
                self._hint_open: "" if compact else _t(_k.hint_open()),
                self._hint_preview: (
                    None if very_narrow else ("" if compact else _t(_k.hint_preview()))
                ),
                self._hint_edit: None if very_narrow else ("" if compact else _t(_k.hint_edit())),
                self._hint_ask: None if very_narrow else ("" if compact else _t(_k.hint_ask())),
                self._hint_search: "" if compact else _t(_k.hint_search()),
                self._hint_sort: "" if compact else _t(_k.hint_sort()),
                self._hint_filter: (
                    None if width < RAIL_MIN_WIDTH else ("" if compact else _t(_k.hint_filter()))
                ),
                self._hint_librarian: (
                    None if very_narrow else ("" if compact else _t(_k.hint_librarian()))
                ),
                self._hint_reload: None,
                self._hint_save: None,
                self._hint_cancel: None,
                self._hint_back: "" if compact else _t(_k.hint_back()),
            }
        lead = False
        for button in self._hint_order:
            label = labels.get(button)
            button.display = label is not None
            if label is None:
                continue
            button.paint(label, lead=lead)
            lead = True

    # -- scrolling ---------------------------------------------------------

    def _scroll_list_to_cursor(self) -> None:
        if not self.rows():
            return
        viewport = max(1, self._list.size.height or 1)
        cursor = self.state.cursor
        y = self._list.scroll_offset.y
        if cursor < y:
            self._list.scroll_to(y=cursor, animate=False)
        elif cursor >= y + viewport:
            self._list.scroll_to(y=cursor - viewport + 1, animate=False)

    def _scroll_rail_to_cursor(self) -> None:
        if not self._rail_items or not self._rail_focused:
            return
        viewport = max(1, self._rail.size.height or 1)
        y = self._rail.scroll_offset.y
        if self._rail_cursor < y:
            self._rail.scroll_to(y=self._rail_cursor, animate=False)
        elif self._rail_cursor >= y + viewport:
            self._rail.scroll_to(y=self._rail_cursor - viewport + 1, animate=False)

    # -- labels ------------------------------------------------------------

    @staticmethod
    def _category_label(value: str) -> str:
        return _t(getattr(_k, f"category_{value}", _k.category_other)())

    @staticmethod
    def _status_label(value: str) -> str:
        mapping = {
            "highlighted": _k.status_highlighted,
            "unorganized": _k.status_unorganized,
            "favorites": _k.status_favorites,
        }
        return _t(mapping[value]())

    @staticmethod
    def _tag_category_label(value: str) -> str:
        return _t(getattr(_k, f"tag_category_{value}", _k.tag_category_custom)())

    @staticmethod
    def _sort_label(value: str) -> str:
        mapping = {
            "recent": _k.sort_recent,
            "name": _k.sort_name,
            "size": _k.sort_size,
            "type": _k.sort_type,
        }
        return _t(mapping[value]())

    # -- rail model --------------------------------------------------------

    def _build_rail(self, payload: Mapping[str, Any]) -> list[_RailItem]:
        items: list[_RailItem] = [_RailItem("header", "", _t(_k.sidebar_categories()))]
        for value, count in library_render.category_options(payload.get("counts")):
            label = _t(_k.category_all()) if value == "" else self._category_label(value)
            items.append(_RailItem("category", value, label, count))
        items.append(_RailItem("header", "", _t(_k.sidebar_status())))
        for value in library_render.STATUS_FILTERS:
            items.append(_RailItem("status", value, self._status_label(value)))
        tags = payload.get("tags") if isinstance(payload.get("tags"), list) else []
        groups = library_render.tag_groups(tags)  # type: ignore[arg-type]
        if groups:
            items.append(_RailItem("header", "", _t(_k.sidebar_tags())))
            for group, rows in groups:
                items.append(_RailItem("header", "", self._tag_category_label(group), None))
                for tag in rows:
                    name = str(tag.get("name") or "")
                    tag_id = str(tag.get("id") or "")
                    # The rail's value is the ID the service compares; the
                    # label is the name a reader sees. Sending the name made
                    # every tag filter return zero rows (QA round 1).
                    if not name or not tag_id:
                        continue
                    items.append(_RailItem("tag", tag_id, f"#{name}", int(tag.get("count") or 0)))
        return items

    def _rail_active(self, item: _RailItem) -> bool:
        if item.kind == "category":
            return self.state.category == item.value
        if item.kind == "status":
            return self.state.status == item.value
        if item.kind == "tag":
            return self.state.tag == item.value
        return False

    def _rail_selectable(self, index: int) -> bool:
        return 0 <= index < len(self._rail_items) and self._rail_items[index].kind != "header"

    def _move_rail(self, delta: int) -> None:
        index = self._rail_cursor
        while True:
            index += delta
            if index < 0:
                index = 0
                break
            if index >= len(self._rail_items):
                index = len(self._rail_items) - 1
                break
            if self._rail_selectable(index):
                break
        self._rail_cursor = max(0, index)
        if not self._rail_selectable(self._rail_cursor):
            # An empty rail or a header-only one: nothing to aim at, so the
            # cursor parks on the nearest selectable row if there is one.
            for offset in range(len(self._rail_items)):
                if self._rail_selectable(offset):
                    self._rail_cursor = offset
                    break
        self._paint_rail()

    def _apply_rail(self) -> None:
        if not self._rail_selectable(self._rail_cursor):
            return
        item = self._rail_items[self._rail_cursor]
        if item.kind == "category":
            self.state.select_category(item.value)
        elif item.kind == "status":
            self.state.select_status(item.value)
        elif item.kind == "tag":
            self.state.select_tag(item.value, item.label.removeprefix("#"))
        self._rail_focused = False
        self.focus()
        self._reload()

    # -- actions -----------------------------------------------------------

    def action_leave(self) -> None:
        if self._form_open:
            self.action_cancel_form()
            return
        if self.app.focused is self._search:
            self._rail_focused = False
            self._paint_rail()
            self.focus()
            return
        if self._rail_focused:
            self._rail_focused = False
            self.focus()
            self._paint_rail()
            self._paint_hints()
            return
        self.post_message(LibraryViewDismissed())

    def action_cursor_down(self) -> None:
        if self._form_open:
            return
        if self._rail_focused:
            self._move_rail(1)
            return
        self.state.move(1, len(self.rows()))
        self._paint_list()
        self._paint_detail()

    def action_cursor_up(self) -> None:
        if self._form_open:
            return
        if self._rail_focused:
            self._move_rail(-1)
            return
        self.state.move(-1, len(self.rows()))
        self._paint_list()
        self._paint_detail()

    def action_page_down(self) -> None:
        if self._form_open:
            return
        viewport = max(1, self._list.size.height or 10)
        if self._rail_focused:
            for _ in range(viewport):
                self._move_rail(1)
            return
        self.state.move(viewport, len(self.rows()))
        self._paint_list()
        self._paint_detail()

    def action_page_up(self) -> None:
        if self._form_open:
            return
        viewport = max(1, self._list.size.height or 10)
        if self._rail_focused:
            for _ in range(viewport):
                self._move_rail(-1)
            return
        self.state.move(-viewport, len(self.rows()))
        self._paint_list()
        self._paint_detail()

    def action_cursor_home(self) -> None:
        if self._form_open:
            return
        if self._rail_focused:
            for offset in range(len(self._rail_items)):
                if self._rail_selectable(offset):
                    self._rail_cursor = offset
                    self._paint_rail()
                    return
            return
        self.state.cursor = 0
        self._paint_list()
        self._paint_detail()

    def action_cursor_end(self) -> None:
        if self._form_open:
            return
        if self._rail_focused:
            for offset in range(len(self._rail_items) - 1, -1, -1):
                if self._rail_selectable(offset):
                    self._rail_cursor = offset
                    self._paint_rail()
                    return
            return
        self.state.cursor = max(0, len(self.rows()) - 1)
        self._paint_list()
        self._paint_detail()

    def action_activate(self) -> None:
        if self._form_open:
            return
        if self._rail_focused:
            self._apply_rail()
            return
        self.action_open_asset()

    def action_open_asset(self) -> None:
        row = self.selected()
        if row is None:
            return
        self.run_worker(self._open_asset(row), group="library-open", exclusive=True)

    def action_toggle_preview(self) -> None:
        if self._form_open:
            return
        row = self.selected()
        if row is None or not row.get("exists", True):
            if row is not None:
                self._notice(_t(_k.detail_preview_missing()), "warning")
            return
        if self._preview_for == str(row.get("id")):
            self._remove_preview()
            self._notice("")
            return
        raw_preview = row.get("preview")
        preview: Mapping[str, Any] = raw_preview if isinstance(raw_preview, Mapping) else {}
        kind = str(row.get("kind") or "other")
        if preview.get("state") != "ready" or kind not in (
            "image",
            "markdown",
            "text",
            "code",
            "spreadsheet",
        ):
            self._notice(_t(_k.preview_unsupported()), "info")
            return
        self._notice("")
        self._remove_preview()
        asset_id = str(row.get("id") or "")
        self._preview_for = asset_id
        self._preview_host.mount(
            Static(_t(_k.preview_loading()), classes="library-view-preview-note")
        )
        self._preview_worker(row)

    @work(exclusive=True, group="library-preview")
    async def _preview_worker(self, row: Mapping[str, Any]) -> None:
        asset_id = str(row.get("id") or "")
        result = await asyncio.to_thread(self._read_preview, row)
        # Both checks: the page may be gone, and the SELECTION may have moved
        # while the bytes were being read. The token is cleared by
        # `_paint_detail` on a selection change, so a late A result cannot
        # paint under B.
        if not self.is_attached or self._preview_for != asset_id:
            return
        self._show_preview(result, asset_id)

    def _read_preview(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve and read one preview payload off the loop. Never raises."""
        try:
            path = self._asset_path(row)
        except Exception as error:  # noqa: BLE001 — a missing copy is a state
            return {"error": str(error)}
        kind = str(row.get("kind") or "other")
        if kind == "image":
            payload = attachments.read_attachment_for_preview(str(path))
            if payload is None:
                return {"error": "unreadable"}
            data_b64, mime = payload
            return {
                "kind": "image",
                "data": data_b64,
                "mime": mime,
                "label": str(row.get("name") or ""),
            }
        try:
            # Bounded READ, not a bounded slice: ``read_bytes()`` would pull a
            # multi-gigabyte file into memory to then throw 99% of it away.
            with open(path, "rb") as handle:
                raw = handle.read(PREVIEW_TEXT_BYTES)
            text = raw.decode("utf-8", "replace")
        except Exception as error:  # noqa: BLE001 — same honest state as images
            return {"error": str(error)}
        return {
            "kind": "markdown" if kind == "markdown" else "text",
            "text": text,
            "name": str(row.get("name") or ""),
        }

    def _asset_path(self, row: Mapping[str, Any]) -> Path:
        """The file behind a row, resolving a digest-only asset through the service.

        The page never invents a path: a row with ``ref.path`` uses it, and a
        digest row asks the service to materialise/locate the store's copy.
        """
        ref = row.get("ref") if isinstance(row.get("ref"), Mapping) else {}
        path = ref.get("path") if isinstance(ref, Mapping) else None
        if path:
            return Path(str(path))
        resolved = library_service.resolve_content(self._config_dir, str(row.get("id") or ""))
        return Path(resolved.path)

    def _show_preview(self, result: Mapping[str, Any], asset_id: str) -> None:
        # The worker already checked; this is the same guard at the PAINT
        # boundary, so any other caller of this seam cannot skip it.
        if not self.is_attached or self._preview_for != asset_id:
            return
        self._remove_preview(keep_pending=True)
        if result.get("error"):
            # Clear the "which asset" token: a failed read must be RETRYABLE
            # by the next space, not read as "this preview is already up".
            self._preview_for = None
            self._preview_host.mount(
                Static(_t(_k.preview_failed()), classes="library-view-preview-note")
            )
            self._notice(_t(_k.notice_preview_failed()), "warning")
            return
        kind = result.get("kind")
        if kind == "image":
            self._preview_widget = ImageBlock(
                str(result.get("data") or ""),
                str(result.get("mime") or "image/png"),
                label=str(result.get("label") or ""),
                # The block is CONSTRUCTED detached, so it would fold at its
                # 80-column fallback and be clipped by the 32-40-cell pane.
                # The fold hint is the one width it can be built for.
                fold_width=max(12, (self._detail.size.width or 40) - 3),
            )
            self._preview_host.mount(self._preview_widget)
            return
        content: Any
        if kind == "markdown":
            content = Markdown(str(result.get("text") or ""))
        else:
            content = Text(str(result.get("text") or ""))
        self._preview_widget = Static(content, classes="library-view-preview-text")
        self._preview_host.mount(self._preview_widget)

    def _remove_preview(self, *, keep_pending: bool = False) -> None:
        self._preview_host.remove_children()
        self._preview_widget = None
        if not keep_pending:
            self._preview_for = None

    async def _open_asset(self, row: Mapping[str, Any]) -> None:
        name = str(row.get("name") or row.get("id") or "")
        if not row.get("exists", True):
            self._notice(_t(_k.notice_open_missing(name=name)), "warning")
            return
        try:
            path = str(await asyncio.to_thread(self._asset_path, row))
        except Exception:  # noqa: BLE001 — reported, never raised
            self._notice(_t(_k.notice_open_missing(name=name)), "warning")
            return
        opened = await attachments.open_path_quietly(path)
        if not opened:
            self._notice(_t(_k.notice_open_failed(name=name)), "warning")
        else:
            self._notice(_t(_k.notice_opened(name=name)), "info")

    def action_focus_search(self) -> None:
        if self._form_open:
            return
        # The rail and the search are one focus slot; leaving the rail cursor
        # armed while the caret is in the field made the NEXT Escape move the
        # rail instead of returning to the list.
        self._rail_focused = False
        self._paint_rail()
        self._search.focus()
        self._search.cursor_position = len(self._search.value)

    def action_cycle_sort(self) -> None:
        if self._form_open:
            return
        self.state.cycle_sort()
        self._reload()

    def action_focus_rail(self) -> None:
        if self._form_open:
            return
        if self._rail.display is False:
            self._notice(_t(_k.rail_hidden(width=str(RAIL_MIN_WIDTH))), "info")
            return
        self._rail_focused = True
        self.focus()
        self._rail_cursor = max(0, min(self._rail_cursor, max(len(self._rail_items) - 1, 0)))
        if not self._rail_selectable(self._rail_cursor):
            for offset in range(len(self._rail_items)):
                if self._rail_selectable(offset):
                    self._rail_cursor = offset
                    break
        self._paint_rail()
        self._paint_hints()

    def action_reload(self) -> None:
        if self._form_open:
            return
        self.state.clamp(len(self.rows()))
        self._reload()

    # -- search ------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input is not self._search:
            return
        if self._search_timer is not None:
            self._search_timer.stop()
        self._search_timer = self.set_timer(0.25, self._apply_search)

    def _apply_search(self) -> None:
        self._search_timer = None
        text = self._search.value
        if text == self.state.q:
            return
        self.state.set_query(text)
        self._reload()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input is self._search:
            if self._search_timer is not None:
                self._search_timer.stop()
                self._search_timer = None
            self._apply_search()
            self._rail_focused = False
            self._paint_rail()
            self.focus()
            return
        if event.input in (self._name_input, self._tags_input) and self._form_open:
            # The footer advertises `enter save` for the whole form; the name
            # field is the first control focus lands on, so Enter there must
            # do what the hint says rather than nothing (UX round 1, U3).
            self.action_save_form()

    # -- metadata form -----------------------------------------------------

    def action_edit_metadata(self) -> None:
        if self._form_open:
            return
        row = self.selected()
        if row is None:
            return
        tags = [tag for tag in list(row.get("tags") or []) if isinstance(tag, Mapping)]
        names = [str(tag.get("name")) for tag in tags if tag.get("name")]
        self._form = _FormState(
            asset_id=str(row.get("id") or ""),
            name=str(row.get("name") or row.get("derived_name") or ""),
            description=str(row.get("description") or ""),
            tags=", ".join(names),
        )
        self._form_open = True
        self._layout()
        # A previous edit may have left the scroll container at its bottom.
        # The next form opens on its FIRST field, not wherever the last one
        # was abandoned; focus alone did not move a container whose offset
        # survived the close (measured in the U2 draft re-capture).
        self._form_block.scroll_to(y=0, animate=False)
        self._name_input.value = self._form.name
        self._description.text = self._form.description
        self._tags_input.value = self._form.tags
        self._paint_form()
        self._paint_hints()
        self._name_input.focus()

    def _paint_form(self) -> None:
        self._form_error.update(self._form.error)
        self._form_error.display = bool(self._form.error)
        self._form_hint.update(
            _t(_k.form_saving())
            if self._form.saving
            else _t(_k.form_tags_hint(max=str(library_metadata.MAX_TAGS_PER_ASSET)))
        )

    def action_cancel_form(self) -> None:
        if not self._form_open:
            return
        self._form_open = False
        self._form = _FormState()
        self.focus()
        self._paint_all()

    def action_save_form(self) -> None:
        if not self._form_open or self._form.saving:
            return
        self._form.name = self._name_input.value
        self._form.description = self._description.text
        self._form.tags = self._tags_input.value
        names = library_render.tag_names(self._form.tags)
        if library_render.too_many_tags(names):
            self._form.error = _t(
                _k.form_too_many_tags(max=str(library_metadata.MAX_TAGS_PER_ASSET))
            )
            self._paint_form()
            return
        if not self._form.asset_id:
            self._form.error = _t(_k.form_asset_gone())
            self._paint_form()
            return
        # Resolve by the id captured at OPEN, never by the cursor. The row may
        # still be on the page (a reload kept it) or not (it vanished from the
        # filtered set); in the second case the edit is refused rather than
        # retargeted at whatever row now sits under the cursor (round 1, B1).
        target = next(
            (item for item in self.rows() if str(item.get("id") or "") == self._form.asset_id),
            None,
        )
        if target is None:
            self._form.error = _t(_k.form_asset_gone())
            self._paint_form()
            return
        self._form.error = ""
        self._form.saving = True
        self._paint_form()
        self._save_worker(self._form.asset_id, names, self._form.name, self._form.description)

    @work(exclusive=True, group="library-save")
    async def _save_worker(
        self,
        asset_id: str,
        names: Sequence[str],
        name: str,
        description: str,
    ) -> None:
        result = await asyncio.to_thread(
            self._save_blocking, asset_id, list(names), name, description
        )
        if not self.is_attached:
            return
        self._form.saving = False
        if result.get("error"):
            self._form.error = str(result.get("error"))
            self._paint_form()
            return
        saved = name.strip() or asset_id
        self._form_open = False
        self._layout()
        self._notice(_t(_k.form_saved(name=saved)), "info")
        self.focus()
        self._reload()

    def _save_blocking(
        self, asset_id: str, names: list[str], name: str, description: str
    ) -> dict[str, Any]:
        try:
            store = library_metadata.MetadataStore(self._config_dir)
            tag_ids = [str(store.create_tag(tag)["id"]) for tag in names]
            library_service.set_tags(self._config_dir, asset_id, tag_ids, actor="user")
            library_service.update_asset(
                self._config_dir,
                asset_id,
                {"name": name.strip() or None, "description": description},
                actor="user",
            )
        except Exception as error:  # noqa: BLE001 — the form shows the store's refusal
            return {"error": str(error)}
        return {"ok": True}

    # -- ask / librarian ---------------------------------------------------

    def action_ask_asset(self) -> None:
        if self._form_open:
            return
        row = self.selected()
        if row is None:
            return
        self.post_message(LibraryViewAskRequested(row))

    def action_open_librarian(self) -> None:
        if self._form_open:
            return
        self.post_message(LibraryViewLibrarianRequested())

    # -- notices -----------------------------------------------------------

    def _notice(self, text: str, tone: str = "info") -> None:
        self._notice_text = text
        self._notice_tone = tone
        self._notice_at = time.monotonic()
        self._paint_status()

    def notice_text(self) -> str:
        """What the status line currently says — the page's own readback seam."""
        return self._notice_text

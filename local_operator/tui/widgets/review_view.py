"""The full-page ``/review`` view: this session's code-request ledger.

A MODE of the main screen, cloned from :class:`HarnessImportView` and, through
it, :class:`SettingsView`: the page takes the transcript's region, leaves the
dock in place and inert, and owns its own cursor and keys. The layout reuses the
settings page's CSS classes (title/rule/columns/body/pane/detail/hints) rather
than adding a second full-page skeleton.

THE DATA PATH IS READ-THEN-FETCH, never a bare network read. The page first
renders the derived index's rows (no network, no writes), then asks the service
for one refresh pass over those rows, then re-reads. That ordering is what makes
the cached frame available on the first tick and a rate-limited host show its
stored data with the backend's own caption rather than an empty page. Both the
file reads and the scan of the index happen on a worker thread; the event loop
never blocks on the disk or a forge.

A FULL-PAGE LIST CLAMPS ITS ARROWS. The list is the whole page and several times
its viewport, so wrapping would throw the reader out of the group they were in
(``AGENTS.md``'s documented exception, shared with ``/settings``). Home and End
are the deliberate ends.

The footer is made of the app's clickable :class:`HintButton`s: the page is
otherwise pointer-reachable, and a hint that cannot be clicked is a caption.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, cast

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.widgets import Static

from local_operator.code_requests import ledger as code_request_ledger
from local_operator.code_requests import service as code_request_service
from local_operator.i18n import keys as i18n_keys
from local_operator.i18n.text import tr
from local_operator.paths import config_dir
from local_operator.tui import review as review_model
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.subagent_view import HintButton
from local_operator.tui.widgets.tool_card import truncate_cells

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.tui.app import OperatorApp

logger = logging.getLogger(__name__)

_t = tr("tui.review")
_kt = i18n_keys.tui_review

#: One cell per round in a lane's strip. The glyph differs for the "verdict not
#: stated" role because that mark is a border with no fill, and a filled block
#: would claim a verdict the wire never carried.
_SEGMENT_GLYPH = {"warning": "▮", "muted": "▮", "success": "▮", "outline": "▯"}

#: The rule is Chrome, not copy: a run of box-drawing cells that the settings
#: page also paints, kept out of a `Text(...)` literal so the extraction ratchet
#: has nothing to flag.
_RULE_GLYPH = "─"

_CURSOR = "›"


class ReviewViewDismissed(Message):
    """The page's esc hint/key was pressed; the app owns teardown."""


class ReviewIndexReadError(RuntimeError):
    """The derived index exists but cannot be read as this build's ledger.

    ``ledger.read_index`` deliberately answers ``None`` for an absent file AND
    for a torn or stale-schema one, because its other readers have no honest
    way to fail. The PAGE does: an absent file is "no code requests yet", while
    a file that exists and will not parse is an error the reader can retry.
    """


class ReviewView(Vertical):
    """The session's pull/merge requests, grouped opened then mentioned."""

    can_focus = True

    BINDINGS = [
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        Binding("pageup", "page(-1)", "Page up", show=False),
        Binding("pagedown", "page(1)", "Page down", show=False),
        Binding("home", "jump(0)", "First", show=False),
        Binding("end", "jump(1)", "Last", show=False),
        Binding("enter", "activate", "Open", show=False),
        Binding("r", "reload", "Refresh", show=False),
        Binding("escape", "leave", "Back", show=False),
    ]

    def __init__(self, session_id: str, *, cwd: str = "") -> None:
        super().__init__(classes="settings-view review-view")
        self._session_id = session_id
        #: The session's working directory when the runtime exposes one. The
        #: scan builds its host context from it (a self-hosted forge URL is
        #: confirmed by that checkout's remotes), so an empty value falls back
        #: to the recorded runtime/wake cwd rather than scanning with none.
        self._cwd = cwd
        #: Resolved once: the app's own config root. A page never invents one.
        self._config_dir = config_dir()
        self._title = Static(classes="settings-view-title")
        self._rule = Static(classes="settings-view-rule")
        self._body_text = Static()
        self._body = ScrollableContainer(self._body_text, classes="settings-view-body")
        self._body.can_focus = False
        # The detail pane is FOCUSABLE and scrolls: a selected request can carry
        # more lines than the pane is tall (four lanes, meta, tags, notice, url),
        # and a Static truncated the remainder with no in-page way to read it.
        # Tab reaches it; its own up/down/page bindings scroll it while focused;
        # a focused container's keys are handled by it and unhandled ones (`r`,
        # escape, enter) still bubble to the page.
        self._pane_body = Static(classes="review-view-pane-text")
        self._pane = ScrollableContainer(
            self._pane_body, classes="settings-view-pane review-view-pane"
        )
        self._pane.can_focus = True
        self._columns = Horizontal(self._body, self._pane, classes="settings-view-columns")
        self._detail = Static(classes="settings-view-detail")
        # One ink pair for every hint: the shipped `dim`/`faint` resting ink is
        # 1.97:1 on this page's surface, which reads as disabled chrome rather
        # than as the four keys that operate the page (design round 1, D1).
        # `muted` is the page's ordinary readable ink; hover lifts to `fg`.
        hint_ink: dict[str, tuple[str, str]] = {
            "ink": ("muted", "muted"),
            "lifted_ink": ("fg", "fg"),
        }
        self._hint_move = HintButton("↑↓", lambda: self.action_move(1), **hint_ink)
        self._hint_enter = HintButton("enter", self.action_activate, **hint_ink)
        self._hint_refresh = HintButton("r", self.action_reload, **hint_ink)
        self._hint_esc = HintButton("esc", self.action_leave, **hint_ink)
        self._hint_order = (
            self._hint_move,
            self._hint_enter,
            self._hint_refresh,
            self._hint_esc,
        )
        self._hints = Horizontal(*self._hint_order, classes="settings-view-hints")
        #: ``loading`` is the COLD read only; rows already on screen stay painted
        #: through a refresh, because a skeleton over data claims the data is gone.
        self._state = "loading"
        self._error = ""
        self._refreshing = False
        self._checked = False
        #: The transcript scan failed on the last pass. Rows already recorded
        #: still paint; the page's one-line status says the scan is the part
        #: that did not happen.
        self._scan_failed = False
        self._rows: list[Mapping[str, Any]] = []
        self._opened: tuple[Mapping[str, Any], ...] = ()
        self._mentioned: tuple[Mapping[str, Any], ...] = ()
        self._visible_rows: list[Mapping[str, Any]] = []
        self._index = 0
        self._narrow = False
        self._now_ms = 0.0
        #: ``(start_line, height)`` per visible row, in the body Text's own lines.
        self._row_offsets: list[tuple[int, int]] = []

    # -- composition --------------------------------------------------------
    def compose(self):  # type: ignore[override]
        yield self._title
        yield self._rule
        with self._columns:
            yield self._body
            yield self._pane
        yield self._detail
        yield self._hints

    def on_mount(self) -> None:
        self._apply_layout()
        self._paint()
        self.call_after_refresh(self._paint)
        try:
            self.focus()
        except Exception:
            pass
        self._start(force_scan=False)

    def on_resize(self) -> None:
        self._apply_layout()
        self._paint()
        self.call_after_refresh(self._paint)

    def _apply_layout(self) -> None:
        """Hide the pane where it would squeeze the list below readability."""
        self._narrow = 0 < self.size.width < 90
        self._pane.display = not self._narrow
        # Narrow mode reserves a second detail line: the selected title is
        # commonly wider than the page at 60 columns, and one truncated line
        # left the reader no in-page way to read the rest (design D2).
        self._detail.set_class(self._narrow, "review-detail-narrow")

    # -- data ---------------------------------------------------------------
    def _start(self, *, force_scan: bool) -> None:
        self.run_worker(
            self._load(force_scan=force_scan),
            group="review",
            exclusive=True,
            exit_on_error=False,
        )

    def _session_dir(self) -> Path:
        """Where this session's own files live — the transcript the scan reads.

        Composed from the config root with the canonical ``sessions`` name
        (``session/cleanup.py``, the same constant the app's other session-store
        readers use) rather than a second literal: an attached viewer knows its
        session id, not the path its owner stores it at.
        """
        from local_operator.session.cleanup import SESSIONS_DIRNAME

        return self._config_dir / SESSIONS_DIRNAME / self._session_id

    async def _scan(self, *, force: bool) -> None:
        """Bring the derived index up to date from the session's transcript.

        THE INDEX IS DERIVED, and nothing in a TUI-only run writes it: the
        desktop routes are today's only scan trigger. Without this pass a
        machine that never opened the desktop Code review pane shows an empty
        page while the conversation plainly holds PR/MR links. The scan itself
        is read-only over the transcript, and ``refresh_async`` is the same
        worker-thread entry the routes call.

        ``force`` rescans the whole journal (the ``r`` press); an open uses the
        signature ladder so an unchanged transcript costs one stat.
        """
        from local_operator.code_requests.hook import load_context_async
        from local_operator.code_requests.refs import EMPTY_CONTEXT

        cwd = self._cwd
        if not cwd:
            cwd = await asyncio.to_thread(
                code_request_ledger.session_cwd, self._config_dir, self._session_id
            )
        context = await load_context_async(cwd) if cwd else EMPTY_CONTEXT
        await code_request_ledger.refresh_async(
            self._config_dir,
            self._session_id,
            self._session_dir(),
            context=context,
            force=force,
        )

    def _read_rows(self) -> list[Mapping[str, Any]]:
        """The index's rows, or a raised error when the file exists but is torn.

        Runs on a worker thread: the read is small but it is still disk, and the
        page must paint its loading frame before any of it happens.
        """
        try:
            data = code_request_ledger.read_index(self._config_dir, self._session_id)
            path = code_request_ledger.index_path(self._config_dir, self._session_id)
        except ValueError:
            # A session id that cannot become a filename has no ledger.
            return []
        if data is None:
            if path.is_file():
                raise ReviewIndexReadError(f"unreadable index: {path}")
            return []
        rows = data.get("rows")
        if not isinstance(rows, list):
            raise ReviewIndexReadError(f"malformed index: {path}")
        return [row for row in rows if isinstance(row, Mapping)]

    async def _publish(self, rows: list[Mapping[str, Any]]) -> None:
        """Turn index rows into the page's rows and paint them.

        The caller decides WHEN a frame may be published; this only derives
        and paints, preserving the cursor across the repaints a single load
        performs (cached frame, post-scan frame, post-forge frame).
        """
        views = await asyncio.to_thread(code_request_service.view_rows, self._config_dir, rows)
        self._rows = list(views)
        groups = review_model.group_rows(self._rows)
        self._opened, self._mentioned = groups.opened, groups.mentioned
        self._visible_rows = [*self._opened, *self._mentioned]
        if self._visible_rows:
            self._index = max(0, min(self._index, len(self._visible_rows) - 1))
        else:
            self._index = 0
        self._state = "ready"
        self._paint()

    async def _load(self, *, force_scan: bool) -> None:
        """Read, scan, read, fetch, read — publishing each frame as it lands.

        THREE STAGES, and the order is the contract:

        1. **The cached frame first.** The index and the stored entries already
           hold displayable rows; withholding them until an unrelated network
           pass settles made a slow forge hide known requests behind a loading
           screen (agent review F3 / UX U1). An EMPTY cached frame is not
           published: "No code reviews yet" is a claim, and on a cold machine
           the scan one step later may disprove it — so an empty read keeps the
           loading frame until the scan settles.
        2. **The transcript scan**, which is also the repair path for a torn or
           stale-schema index. A first-read failure is NOT terminal: the scan's
           ``write_index`` rewrites exactly the file the read could not parse
           (QA Q1). Only a scan that also failed, with a read that still fails,
           is the error state.
        3. **The forge pass**, forced only when the reader asked. `r` is an
           explicit "check now", so its intent reaches ``service.refresh_session``
           as ``force=True`` and defeats a fresh TTL — cooling and per-key
           backoff are the service's and stay intact (agent review F2). A cold
           open leaves ``force`` False and respects the cache.

        Nothing here blocks the event loop: every disk/network call runs in a
        worker thread, and the page is interactive from the first painted frame
        (selection and Enter keep working while the pass is held).
        """
        cold = self._state != "ready"
        if cold:
            self._state = "loading"
        self._refreshing = not cold
        self._scan_failed = False
        self._paint()

        first_error: Exception | None = None
        rows: list[Mapping[str, Any]] = []
        try:
            rows = await asyncio.to_thread(self._read_rows)
        except Exception as exc:  # noqa: BLE001
            first_error = exc
            # Show the error frame, but do NOT stop: the scan below is the
            # repair path the retry hint promises.
            self._fail(exc)
            self._paint()
        if rows:
            self._refreshing = True
            try:
                await self._publish(rows)
            except Exception as exc:  # noqa: BLE001
                self._fail(exc)
                self._paint()
                return

        scan_error: Exception | None = None
        try:
            await self._scan(force=force_scan)
        except Exception as exc:  # noqa: BLE001
            logger.debug("code-request scan failed for %s", self._session_id, exc_info=True)
            scan_error = exc
            self._scan_failed = True

        try:
            rows = await asyncio.to_thread(self._read_rows)
        except Exception as exc:  # noqa: BLE001
            if first_error is not None or not self._visible_rows:
                self._fail(scan_error or exc)
                self._paint()
                return
            # Cached rows were already painted. Keep them rather than blanking
            # the page over a re-read that failed after they were shown.
            logger.debug("code-request re-read failed for %s", self._session_id, exc_info=True)
            self._refreshing = False
            self._paint()
            return

        if rows:
            try:
                await code_request_service.refresh_session(
                    self._config_dir, self._session_id, rows, force=force_scan
                )
            except Exception:  # noqa: BLE001
                # A failed fetch pass keeps the last known data standing, and
                # each row's own ``refresh_error`` is the sentence that says so.
                logger.debug("code-request refresh failed for %s", self._session_id, exc_info=True)
            try:
                rows = await asyncio.to_thread(self._read_rows)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "code-request post-refresh read failed for %s", self._session_id, exc_info=True
                )

        try:
            await self._publish(rows)
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)
            self._paint()
            return
        if scan_error is not None and not self._visible_rows:
            # Nothing was recorded and the conversation could not be read: the
            # empty state would be a claim this page cannot make.
            self._fail(scan_error)
            self._paint()
            return
        self._refreshing = False
        # "Checked just now" is a receipt for the READER's press, and `force`
        # arrives only from one: a cold open's automatic pass leaves it False,
        # so it never claims a check nobody asked for. This includes a retry
        # from the error state, which really is an explicit check.
        self._checked = force_scan
        self._paint()

    def _fail(self, exc: BaseException) -> None:
        self._error = f"{type(exc).__name__}: {exc}"
        self._state = "error"
        self._refreshing = False

    def _selected_row(self) -> Mapping[str, Any] | None:
        if 0 <= self._index < len(self._visible_rows):
            return self._visible_rows[self._index]
        return None

    # -- keys ---------------------------------------------------------------
    def action_move(self, delta: int) -> None:
        if not self._visible_rows:
            return
        self._index = max(0, min(len(self._visible_rows) - 1, self._index + delta))
        self._paint()

    def action_page(self, delta: int) -> None:
        if not self._visible_rows:
            return
        step = max(1, (self._body.size.height or 1) // 2)
        self._index = max(0, min(len(self._visible_rows) - 1, self._index + delta * step))
        self._paint()

    def action_jump(self, edge: int) -> None:
        if not self._visible_rows:
            return
        self._index = 0 if edge == 0 else len(self._visible_rows) - 1
        self._paint()

    def action_activate(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        url = str(row.get("url") or "")
        if url:
            # The app's ONE guarded opener; this page must not grow a second.
            # Typed through the app rather than duck-typed: a missing opener must
            # be a static error, not a press that silently does nothing.
            cast("OperatorApp", self.app).open_transcript_link(url)

    def action_reload(self) -> None:
        if self._refreshing:
            return
        self._start(force_scan=True)

    def action_leave(self) -> None:
        self.post_message(ReviewViewDismissed())

    # -- painting -----------------------------------------------------------
    def _detail_line(self, width: int) -> str:
        """The selected request's title, wrapped in narrow mode.

        Wide mode has a pane that repeats the clause, so one clipped line is a
        caption for it. Narrow mode hides the pane, and a title wider than 60
        cells then had nowhere to be read: two reserved lines carry the whole
        value, with an ellipsis only when it is still too long.
        """
        content = self._detail_text()
        if not self._narrow or not content:
            return truncate_cells(content, width)
        lines = _wrap(content, width)
        if len(lines) <= 2:
            return "\n".join(lines)
        head = lines[:2]
        head[-1] = truncate_cells(head[-1] + " …", width)
        return "\n".join(head)

    def _paint(self) -> None:
        self._apply_layout()
        self._now_ms = time.time() * 1000.0
        self._title.update(Text(_t(_kt.title()), style="bold"))
        self._rule.update(Text(_RULE_GLYPH * 200, style="dim"))
        self._pane_body.update(self._pane_text())
        width = max(10, self._detail.size.width or self.size.width or 80)
        self._detail.update(Text(self._detail_line(width), style="dim"))
        self._paint_hints()
        self._body_text.update(self._body_rich())
        self._scroll_to_cursor()

    def _paint_hints(self) -> None:
        visible = self._hint_labels()
        lead = False
        for button in self._hint_order:
            entry = visible.get(button)
            button.display = entry is not None
            if entry is None:
                continue
            button.paint(entry, lead=lead)
            lead = True

    def _hint_labels(self) -> dict[HintButton, str]:
        labels: dict[HintButton, str] = {}
        if self._visible_rows:
            labels[self._hint_move] = _t(_kt.hint_move())
            labels[self._hint_enter] = _t(_kt.hint_open())
        labels[self._hint_refresh] = _t(_kt.hint_refresh())
        labels[self._hint_esc] = _t(_kt.hint_back())
        return labels

    def _detail_text(self) -> str:
        if self._state == "loading":
            return _t(_kt.loading())
        if self._state == "error":
            return ""
        if self._scan_failed:
            return _t(_kt.scan_failed())
        if self._refreshing:
            return _t(_kt.refreshing())
        row = self._selected_row()
        if row is not None:
            title = review_model.row_title(row)
            # The detail line exists to show the FULL title the list truncates;
            # a link-only row has no title, only the URL its pane already prints.
            return "" if title == str(row.get("url") or "") else title
        if self._checked:
            return _t(_kt.checked_just_now())
        return ""

    def _body_rich(self) -> Text:
        text = Text()
        if self._state == "loading":
            text.append(_t(_kt.loading()) + "\n", style="dim")
            return text
        if self._state == "error":
            text.append(_t(_kt.error_header()) + "\n\n", style="bold")
            text.append(_t(_kt.error_body()) + "\n")
            # The technical detail can be a long path. It is one diagnostic
            # line, not prose: letting it wrap pushed a scrollbar into an
            # otherwise empty page at 60 columns.
            text.append(
                truncate_cells(_t(_kt.error_detail(error=self._error)), self._line_width()) + "\n",
                style="dim",
            )
            text.append(_t(_kt.error_retry()) + "\n", style="dim")
            return text
        if not self._visible_rows:
            text.append(_t(_kt.empty_header()) + "\n\n", style="bold")
            text.append(_t(_kt.empty_body()) + "\n")
            return text
        return self._rows_rich()

    def _rows_rich(self) -> Text:
        text = Text()
        width = self._line_width()
        self._row_offsets = []
        index = 0
        for label, rows in (
            (_t(_kt.group_opened()), self._opened),
            (_t(_kt.group_mentioned()), self._mentioned),
        ):
            if not rows:
                continue
            self._append_group_header(text, label, len(rows), width)
            for row in rows:
                start = text.plain.count("\n")
                self._append_row(text, row, selected=index == self._index, width=width)
                self._row_offsets.append((start, max(1, text.plain.count("\n") - start)))
                index += 1
        return text

    def _line_width(self) -> int:
        return max(20, (self._body.size.width or 60) - 2)

    def _append_group_header(self, text: Text, label: str, count: int, width: int) -> None:
        count_text = str(count)
        pad = max(1, width - cell_len(label) - cell_len(count_text))
        text.append(label, style=Style(color=theme_mod.semantic_color("muted"), bold=True))
        text.append(" " * pad + count_text, style=Style(color=theme_mod.semantic_color("dim")))
        text.append("\n")

    def _append_row(
        self,
        text: Text,
        row: Mapping[str, Any],
        *,
        selected: bool,
        width: int,
    ) -> None:
        background = theme_mod.semantic_color("tint-select") if selected else None
        base = Style(bgcolor=background)
        cursor = Style(
            color=theme_mod.semantic_color("muted") if selected else None,
            bgcolor=background,
        )
        text.append(f"{_CURSOR} " if selected else "  ", style=cursor)

        # Line A: identity, the disambiguating tags, and the state pill pushed
        # to the right edge. A link-only row has no summary, so no pill.
        tags = [tag for tag in (review_model.relation_tag(row), review_model.acted_tag(row)) if tag]
        left = review_model.identity_label(row)
        if tags:
            left = f"{left}  {' · '.join(tags)}"
        pill = review_model.state_pill(row)
        pill_label = pill.label if pill is not None else ""
        room = max(8, width - 2 - (cell_len(pill_label) + 1 if pill_label else 0))
        left = truncate_cells(left, room)
        text.append(left, style=base)
        if pill is not None:
            gap = max(1, width - 2 - cell_len(left) - cell_len(pill_label))
            text.append(" " * gap, style=base)
            text.append(
                pill_label,
                style=Style(color=_pill_color(pill.variant), bgcolor=background, bold=True),
            )
        text.append("\n")

        # Line B: the title, the row's loudest run.
        text.append_text(
            _styled_line("  " + truncate_cells(review_model.row_title(row), width - 2), base)
        )

        # Line C: the lane state. Wide mode gives each lane its own line; a
        # narrow page (four-line body at 60x20) gives the whole request one
        # dense summary instead, so its identity and meta stay in the viewport
        # (UX U2). The detail pane keeps the full clause where there is room.
        if not row.get("link_only"):
            lanes = review_model.lane_lines(row)
            if self._narrow and lanes:
                self._append_lane_summary(text, lanes, width=width, background=background)
            else:
                for line in lanes:
                    self._append_lane_line(text, line, width=width, background=background)

        # Line D: the meta figures, `·`-seamed.
        meta = self._meta_spans(row, width=width, background=background)
        if meta is not None:
            text.append_text(meta)

        # Line E: the one quiet line — loading when the first read is pending,
        # otherwise the backend's own notice.
        notice = (
            _t(_kt.row_fetching())
            if review_model.row_is_loading(row)
            else review_model.row_notice(row)
        )
        if notice:
            notice_style = Style(color=theme_mod.semantic_color("muted"), bgcolor=background)
            text.append_text(_styled_line("  " + truncate_cells(notice, width - 2), notice_style))
        else:
            text.append("\n", style=base)
        # A blank row between requests keeps two wide rows from reading as one
        # block. Narrow rows are already a compact four lines, so the gap is
        # what pushes the next identity out of a four-line viewport; drop it
        # there and let the notice slot separate them.
        if not self._narrow:
            text.append("\n", style=base)

    def _append_lane_line(
        self, text: Text, line: review_model.LaneLine, *, width: int, background: Any
    ) -> None:
        dim = Style(color=theme_mod.semantic_color("dim"), bgcolor=background)
        muted = Style(color=theme_mod.semantic_color("muted"), bgcolor=background)
        text.append("  ", style=dim)
        # The NAME is content the reader compares across rows, so it takes the
        # readable `muted` ink rather than the decorative `dim` (design D1).
        text.append(review_model.lane_name_cell(line.name) + " ", style=muted)
        if line.segments_truncated:
            # The true round stays in the clause; the strip says it is a head,
            # not the whole history (agent review F1).
            text.append("…", style=dim)
        for tone in line.segments:
            text.append(
                _SEGMENT_GLYPH[tone],
                style=Style(color=_lane_tone_color(tone), bgcolor=background),
            )
        strip_width = len(line.segments) + (1 if line.segments_truncated else 0)
        gap = 1 if strip_width else 0
        room = max(6, width - 2 - review_model.LANE_NAME_CELLS - 1 - strip_width - gap)
        text.append(" " * gap, style=dim)
        text.append(truncate_cells(line.clause, room), style=muted)
        text.append("\n")

    def _append_lane_summary(
        self,
        text: Text,
        lines: tuple[review_model.LaneLine, ...],
        *,
        width: int,
        background: Any,
    ) -> None:
        """One dense line: lane name and round, `Code 2 · Design 1 · QA 1`.

        The round is what distinguishes a clean lane from one that was reviewed
        and remediated, so it stays; the prose state and freshness are the
        detail pane's and wide mode's job.
        """
        muted = Style(color=theme_mod.semantic_color("muted"), bgcolor=background)
        dim = Style(color=theme_mod.semantic_color("dim"), bgcolor=background)
        parts = [
            f"{line.name} {line.round}" if line.round is not None else line.name for line in lines
        ]
        text.append("  ", style=dim)
        text.append(truncate_cells(" · ".join(parts), max(6, width - 2)), style=muted)
        text.append("\n")

    def _meta_spans(self, row: Mapping[str, Any], *, width: int, background: Any) -> Text | None:
        summary = row.get("summary")
        if row.get("link_only") or not isinstance(summary, Mapping):
            return None
        ci = review_model.ci_clause(summary.get("ci"))
        comments = review_model.comment_clause(summary.get("comments"))
        updated_at = summary.get("updated_at")
        updated = (
            review_model.updated_clause(updated_at, self._now_ms)
            if isinstance(updated_at, (int, float))
            else None
        )
        parts: list[tuple[str, Any]] = []
        if ci is not None:
            parts.append((ci.text, Style(color=_ci_color(ci.tone), bgcolor=background)))
        muted = Style(color=theme_mod.semantic_color("muted"), bgcolor=background)
        if comments:
            parts.append((comments, muted))
        if updated:
            parts.append((updated, muted))
        if not parts:
            return None
        text = Text("  ")
        room = max(10, width - 2)
        seam = Style(color=theme_mod.semantic_color("dim"), bgcolor=background)
        for position, (part, style) in enumerate(parts):
            if position:
                text.append(" · ", style=seam)
            remaining = room - (cell_len(text.plain) - 2)
            if remaining <= 0:
                break
            text.append(truncate_cells(part, remaining), style=style)
        if text.plain.strip() == "":
            return None
        # Every painted line ends with exactly one newline. Without this the
        # notice appended on the same line as the meta (and `_row_offsets`
        # undercounted the row's visual height).
        text.append("\n")
        return text

    def _pane_text(self) -> Text:
        text = Text()
        width = self._pane_line_width()
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        row = self._selected_row()
        if row is None:
            text.append(_t(_kt.subtitle()), style=dim)
            return text
        text.append(review_model.identity_label(row) + "\n", style="bold")
        title = review_model.row_title(row)
        url = str(row.get("url") or "")
        # A link-only row's `title` IS its URL; printing both would put the same
        # line twice on a pane that then repeats it once more at the bottom.
        if title and title != url:
            for line in _wrap(title, width):
                text.append(line + "\n")
        pill = review_model.state_pill(row)
        summary = row.get("summary")
        head: list[str] = []
        if pill is not None:
            head.append(pill.label)
        if isinstance(summary, Mapping):
            ci = review_model.ci_clause(summary.get("ci"))
            if ci is not None:
                head.append(ci.text)
            comments = review_model.comment_clause(summary.get("comments"))
            if comments:
                head.append(comments)
            updated_at = summary.get("updated_at")
            if isinstance(updated_at, (int, float)):
                head.append(review_model.updated_clause(updated_at, self._now_ms))
        if head:
            text.append("\n")
            for line in _wrap(" · ".join(head), width):
                text.append(line + "\n", style=muted)
        if not row.get("link_only"):
            lanes = review_model.lane_lines(row)
            if lanes:
                text.append("\n")
                for line in lanes:
                    cells = review_model.LANE_NAME_CELLS + 1
                    first = review_model.lane_name_cell(line.name) + " "
                    clause_lines = _wrap(line.clause, max(6, width - cells))
                    for position, clause in enumerate(clause_lines):
                        text.append(first if position == 0 else " " * cells, style=muted)
                        text.append(clause + "\n", style=muted)
        tags = [tag for tag in (review_model.relation_tag(row), review_model.acted_tag(row)) if tag]
        if tags:
            text.append("\n")
            for line in _wrap(" · ".join(tags), width):
                text.append(line + "\n", style=muted)
        notice = review_model.row_notice(row)
        if notice:
            text.append("\n")
            for line in _wrap(notice, width):
                text.append(line + "\n", style=muted)
        url = str(row.get("url") or "")
        if url:
            text.append("\n")
            for line in _wrap(url, width):
                text.append(line + "\n", style=muted)
        return text

    def _pane_line_width(self) -> int:
        # `.settings-view-pane` is 34 cells with 3 of left padding; fall back to
        # that before the first layout pass has measured the widget.
        return max(10, (self._pane.size.width or 34) - 3)

    def _scroll_to_cursor(self) -> None:
        """Keep the cursor row in the viewport using the ACTUAL row heights."""
        if not self._row_offsets:
            return
        start, height = self._row_offsets[min(self._index, len(self._row_offsets) - 1)]
        viewport = max(1, self._body.size.height or 1)
        y = self._body.scroll_offset.y
        if height >= viewport:
            # A row taller than the viewport (narrow mode: a four-line identity
            # block against a four-line body) is anchored at its TOP. Bottom
            # alignment would scroll its identity off the screen to show the
            # reserved notice line, which is the U2 report exactly.
            target = start
        elif start < y:
            target = start
        elif start + height > y + viewport:
            target = start + height - viewport
        else:
            return
        try:
            self._body.scroll_to(y=max(0, target), animate=False, immediate=True)
        except Exception:
            pass


def _ci_color(tone: str) -> str:
    """CI's ink role: ``danger`` red, ``plain`` the default muted ink."""
    return {
        "danger": theme_mod.semantic_color("danger"),
        "muted": theme_mod.semantic_color("dim"),
        "plain": theme_mod.semantic_color("muted"),
    }.get(tone, theme_mod.semantic_color("muted"))


def _lane_tone_color(tone: str) -> str:
    """The lane mark's ink role, resolved from the theme's semantic ramp.

    ``outline`` is the role for a review whose verdict was never stated: the
    desktop draws it as a border with no fill, and the TUI's whole translation
    of that is the hollow glyph in ``_SEGMENT_GLYPH`` plus the dimmest ink.
    """
    return {
        "warning": theme_mod.semantic_color("warning"),
        "muted": theme_mod.semantic_color("muted"),
        "success": theme_mod.semantic_color("success"),
        "outline": theme_mod.semantic_color("dim"),
    }.get(tone, theme_mod.semantic_color("dim"))


def _pill_color(variant: str) -> str:
    """The state pill's ink role: one badge vocabulary, no hex in the widget."""
    return {
        "info": theme_mod.semantic_color("signal"),
        "outline": theme_mod.semantic_color("muted"),
        "success": theme_mod.semantic_color("success"),
        "neutral": theme_mod.semantic_color("muted"),
    }.get(variant, theme_mod.semantic_color("muted"))


def _styled_line(content: str, style: Style) -> Text:
    text = Text(content, style=style)
    text.append("\n")
    return text


def _wrap(value: str, width: int) -> list[str]:
    """Wrap a sentence to ``width`` cells, never dropping a word."""
    if width <= 0:
        return [value]
    lines: list[str] = []
    current = ""
    for word in value.split():
        candidate = f"{current} {word}".strip()
        if current and cell_len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


__all__ = ["ReviewIndexReadError", "ReviewView", "ReviewViewDismissed"]

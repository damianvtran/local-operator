"""The full-page ``/import`` view.

A MODE of the main screen, cloned from :class:`SettingsView`: the page takes
the transcript's region, leaves the dock in place and inert, and owns its own
cursor and keys. The service does the reading and writing; this widget only
paints and asks. It never parses a foreign store, never creates a target
itself, and never imports on a bare open: an import requires a selected source,
at least one selected session, a completed preview and a second Enter.

The layout reuses the settings page's CSS classes (title/rule/columns/body/
pane/detail/hints) rather than adding a second full-page skeleton. Rows are
fixed-height and clipped to the body width exactly like the settings list:
a long title must never push the date onto another line or make the cursor
swallow a wrapped block. Below 90 page columns the right pane is hidden and the
body takes the page width, because a 34-cell pane beside a 20-cell list is the
bug the narrow capture caught, not a layout.

The footer is made of the app's clickable :class:`HintButton`s, not inert text:
the page is otherwise pointer-reachable, and a hint that cannot be clicked is
a caption.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.widgets import Static

from local_operator.harness_import.jobs import JOBS, ImportJobBusy
from local_operator.harness_import.service import HarnessImportService, normalise_title
from local_operator.harness_import.types import ImportRequest
from local_operator.i18n import keys as i18n_keys
from local_operator.i18n.text import tr
from local_operator.paths import config_dir
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.subagent_view import HintButton
from local_operator.tui.widgets.tool_card import truncate_cells

_t = tr("tui.import")
_kt = i18n_keys.tui_import

ImportAction = Literal["list", "preview", "run", "status", "cancel"]


@dataclass(frozen=True, slots=True)
class ImportLaunch:
    """What ``/import …`` asked the page to open on.

    Parsed by the app from the typed command. ``session_id`` is a source id;
    ``all`` is the deliberate ``run <source> all`` form. Nothing here imports:
    ``run`` stops at the preview, and the page's second Enter is the confirm.
    """

    action: ImportAction = "list"
    source: str = ""
    session_id: str = ""
    all: bool = False


class HarnessImportViewDismissed(Message):
    """The page's esc hint/key was pressed; the app owns teardown."""


class HarnessImportView(Vertical):
    """Sources → sessions → preview → job progress → result."""

    can_focus = True

    BINDINGS = [
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        Binding("pageup", "page(-1)", "Page up", show=False),
        Binding("pagedown", "page(1)", "Page down", show=False),
        Binding("home", "jump(0)", "First", show=False),
        Binding("end", "jump(1)", "Last", show=False),
        Binding("space", "select", "Select", show=False),
        Binding("a", "all", "All", show=False),
        Binding("enter", "activate", "Open", show=False),
        Binding("r", "reload", "Reload", show=False),
        Binding("c", "cancel", "Cancel", show=False),
        Binding("escape", "leave", "Back", show=False),
    ]

    def __init__(self, launch: ImportLaunch | None = None) -> None:
        super().__init__(classes="settings-view harness-import-view")
        self.launch = launch or ImportLaunch()
        self._service = HarnessImportService(config_dir())
        self._title = Static(classes="settings-view-title")
        self._rule = Static(classes="settings-view-rule")
        self._body_text = Static()
        self._body = ScrollableContainer(self._body_text, classes="settings-view-body")
        self._body.can_focus = False
        self._pane = Static(classes="settings-view-pane")
        self._columns = Horizontal(self._body, self._pane, classes="settings-view-columns")
        self._detail = Static(classes="settings-view-detail")
        #: Every footer hint is a real HintButton; the visible subset and its
        #: labels change with the state and the page width.
        self._hint_move = HintButton("↑↓", lambda: self.action_move(1))
        self._hint_space = HintButton("space", self.action_select)
        self._hint_all = HintButton("a", self.action_all)
        self._hint_enter = HintButton("enter", self.action_activate)
        self._hint_cancel = HintButton("c", self.action_cancel)
        self._hint_reload = HintButton("r", self.action_reload)
        self._hint_esc = HintButton("esc", self.action_leave)
        self._hint_order = (
            self._hint_move,
            self._hint_space,
            self._hint_all,
            self._hint_enter,
            self._hint_cancel,
            self._hint_reload,
            self._hint_esc,
        )
        self._hints = Horizontal(*self._hint_order, classes="settings-view-hints")
        self._sources: list[Any] = []
        self._sessions: list[Any] = []
        self._source_index = 0
        self._session_index = 0
        self._selected: set[str] = set()
        self._state = "loading"
        self._source = ""
        self._empty = ""
        self._empty_header = ""
        self._loading = _t(_kt.loading_sources())
        self._preview: Any = None
        self._request: ImportRequest | None = None
        self._job: Any = None
        self._error = ""
        self._poll = None
        self._narrow = False

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
        self._begin()

    def on_resize(self) -> None:
        self._apply_layout()
        self._paint()
        self.call_after_refresh(self._paint)

    def on_unmount(self) -> None:
        self._stop_poll()

    def _apply_layout(self) -> None:
        """Hide the pane where it would squeeze the list below readability."""
        self._narrow = self.size.width < 90
        self._pane.display = not self._narrow

    # -- launch -------------------------------------------------------------
    def _begin(self) -> None:
        action = self.launch.action
        if action == "status":
            self._show_status()
            return
        if action == "cancel":
            self._cancel()
            return
        if action in ("preview", "run") and self.launch.source:
            self.run_worker(
                self._open_source(self.launch.source, auto=action),
                group="harness-import",
                exclusive=True,
                exit_on_error=False,
            )
            return
        self.run_worker(
            self._load_sources(), group="harness-import", exclusive=True, exit_on_error=False
        )

    async def _load_sources(self) -> None:
        self._state = "loading"
        self._loading = _t(_kt.loading_sources())
        self._paint()
        try:
            self._sources = await self._service.sources()
            self._state = "sources"
            self._source_index = min(self._source_index, max(0, len(self._sources) - 1))
        except Exception as exc:  # noqa: BLE001 — the page reports, never crashes
            self._fail(exc)
        self._paint()

    async def _open_source(self, source: str, *, auto: str = "") -> None:
        if source not in self._service.adapters:
            self._error = _t(_kt.error_source(source=source))
            self._state = "error"
            self._paint()
            return
        self._source = source
        self._state = "loading"
        self._loading = _t(_kt.loading_sessions(source=self._source_label(source)))
        self._paint()
        try:
            self._sessions = await self._service.list_sessions(source)
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)
            self._paint()
            return
        self._selected = set()
        self._session_index = 0
        if not self._sessions:
            self._empty_header = ""
            self._empty = _t(_kt.empty_sessions(source=self._source_label(source)))
            self._state = "empty"
            self._paint()
            return
        if self.launch.all:
            self._selected = {row.source_id for row in self._sessions}
        if self.launch.session_id:
            wanted = self.launch.session_id
            for index, row in enumerate(self._sessions):
                if row.source_id == wanted:
                    self._session_index = index
                    self._selected = {wanted}
                    break
        self._state = "sessions"
        self._paint()
        if auto in ("preview", "run") and (self.launch.session_id or self.launch.all):
            await self._preview_current()

    async def _preview_current(self) -> None:
        if not self._selected:
            self._empty_header = _t(_kt.select_header())
            self._empty = _t(_kt.error_select())
            self._state = "empty"
            self._paint()
            return
        request = ImportRequest(
            sources=[self._source],
            session_ids={self._source: sorted(self._selected)},
        )
        self._state = "loading"
        self._loading = _t(_kt.loading_preview())
        self._paint()
        try:
            self._preview = await self._service.preview(request)
            self._request = request
            self._state = "preview"
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)
        self._paint()

    # -- job ----------------------------------------------------------------
    def _arm_poll(self) -> None:
        if self._poll is None:
            self._poll = self.set_interval(0.1, self._tick_job)

    def _start(self) -> None:
        if self._request is None:
            return
        try:
            self._job = JOBS.start(self._service, self._request, session_key="tui")
        except ImportJobBusy:
            self._error = _t(_kt.error_busy())
            self._state = "error"
            self._paint()
            return
        self._state = "progress"
        self._paint()
        self._arm_poll()

    def _tick_job(self) -> None:
        if self._job is None:
            self._stop_poll()
            return
        if self._job.result is not None:
            self._stop_poll()
            self._state = "result"
            self._paint()
            return
        self._paint()

    def _stop_poll(self) -> None:
        if self._poll is not None:
            self._poll.stop()
            self._poll = None

    def _cancel(self) -> None:
        """Cancel the shared job; a page opened by ``/import cancel`` reattaches.

        The key binding is restricted to the progress state, but this method is
        also the ``/import cancel`` launch path: a page opened directly by that
        command starts in ``loading`` and must attach to the running job rather
        than returning before it looks. A genuinely jobless direct cancel still
        reports the truthful no-job error.
        """
        try:
            payload = JOBS.cancel(session_key="tui")
        except KeyError:
            # No running task. A completed job still answers this command with
            # its result; only a genuinely empty registry is an error.
            settled = JOBS.latest(session_key="tui")
            if settled is None or settled.result is None:
                self._error = _t(_kt.error_cancel())
                self._state = "error"
                self._paint()
                return
            payload = {"job_id": settled.job_id}
        job = JOBS.get(str(payload.get("job_id") or ""), session_key="tui")
        if job is None:
            self._error = _t(_kt.error_no_job())
            self._state = "error"
            self._paint()
            return
        self._job = job
        self._request = job.request or self._request
        self._selected = set(_request_ids(self._request)) or self._selected
        if job.result is not None:
            self._stop_poll()
            self._state = "result"
        else:
            self._state = "progress"
            self._arm_poll()
        self._paint()

    def _show_status(self) -> None:
        self._job = JOBS.latest(session_key="tui")
        if self._job is None:
            self._error = _t(_kt.error_no_job())
            self._state = "error"
        else:
            self._request = self._job.request or self._request
            self._selected = set(_request_ids(self._request)) or self._selected
            if self._job.result is not None:
                self._state = "result"
            else:
                self._state = "progress"
                self._arm_poll()
        self._paint()

    # -- keys ---------------------------------------------------------------
    def action_move(self, delta: int) -> None:
        if self._state == "sources" and self._sources:
            self._source_index = max(0, min(len(self._sources) - 1, self._source_index + delta))
        elif self._state == "sessions" and self._sessions:
            self._session_index = max(0, min(len(self._sessions) - 1, self._session_index + delta))
        else:
            return
        self._paint()

    def action_page(self, delta: int) -> None:
        step = max(1, (self._body.size.height or 1) // 2)
        if self._state == "sources" and self._sources:
            self._source_index = max(
                0, min(len(self._sources) - 1, self._source_index + delta * step)
            )
        elif self._state == "sessions" and self._sessions:
            self._session_index = max(
                0, min(len(self._sessions) - 1, self._session_index + delta * step)
            )
        else:
            return
        self._paint()

    def action_jump(self, edge: int) -> None:
        if self._state == "sources" and self._sources:
            self._source_index = 0 if edge == 0 else len(self._sources) - 1
        elif self._state == "sessions" and self._sessions:
            self._session_index = 0 if edge == 0 else len(self._sessions) - 1
        else:
            return
        self._paint()

    def action_select(self) -> None:
        if self._state != "sessions" or not self._sessions:
            return
        source_id = self._sessions[self._session_index].source_id
        if source_id in self._selected:
            self._selected.discard(source_id)
        else:
            self._selected.add(source_id)
        self._paint()

    def action_all(self) -> None:
        if self._state != "sessions" or not self._sessions:
            return
        every = {row.source_id for row in self._sessions}
        self._selected = set() if self._selected >= every else every
        self._paint()

    def action_activate(self) -> None:
        if self._state == "sources" and self._sources:
            self.run_worker(
                self._open_source(self._sources[self._source_index].source),
                group="harness-import",
                exclusive=True,
                exit_on_error=False,
            )
        elif self._state == "sessions":
            self.run_worker(
                self._preview_current(),
                group="harness-import",
                exclusive=True,
                exit_on_error=False,
            )
        elif self._state == "preview":
            self._start()
        elif self._state == "result":
            self._state = "sources"
            self.run_worker(
                self._load_sources(), group="harness-import", exclusive=True, exit_on_error=False
            )

    def action_reload(self) -> None:
        if self._state in ("sources", "empty", "error"):
            self._empty = ""
            self._empty_header = ""
            self._error = ""
            self.run_worker(
                self._load_sources(), group="harness-import", exclusive=True, exit_on_error=False
            )

    def action_cancel(self) -> None:
        if self._state == "progress":
            self._cancel()

    def action_leave(self) -> None:
        if self._state == "sessions":
            self._state = "sources"
            self._paint()
            return
        if self._state in ("preview", "empty", "error"):
            self._state = "sessions" if self._sessions else "sources"
            self._paint()
            return
        if self._state in ("sources", "progress", "result"):
            self.post_message(HarnessImportViewDismissed())

    # -- painting -----------------------------------------------------------
    def _fail(self, exc: BaseException) -> None:
        self._error = _t(_kt.error_generic(error=f"{type(exc).__name__}: {exc}"))
        self._state = "error"

    def _source_label(self, source: str) -> str:
        for info in self._sources:
            if info.source == source:
                return info.label
        return source

    def _detail_text(self) -> str:
        if self._state == "sources" and self._sources:
            info = self._sources[self._source_index]
            return f"{info.label} · {_availability_label(info)}"
        if self._state == "sessions":
            return _t(_kt.selected_count(count=str(len(self._selected))))
        if self._state == "preview":
            if self._narrow:
                if self._only_metadata_only():
                    return _t(_kt.preview_narrow_metadata(count=len(self._selected)))
                return _t(_kt.preview_narrow(count=len(self._selected)))
            return _t(_kt.selected_count(count=str(len(self._selected))))
        if self._state == "progress" and self._job is not None and self._narrow:
            return _t(
                _kt.job_status(job=self._job.job_id, state=_state_label(self._job.progress.state))
            )
        return ""

    def _paint(self) -> None:
        self._apply_layout()
        self._title.update(Text(_t(_kt.title()), style="bold"))
        self._rule.update(Text("─" * 200, style="dim"))
        self._pane.update(Text(self._pane_text()))
        width = max(10, self._detail.size.width or self.size.width or 80)
        self._detail.update(Text(truncate_cells(self._detail_text(), width), style="dim"))
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
            label = entry
            button.paint(label, lead=lead)
            lead = True

    def _hint_labels(self) -> dict[HintButton, str]:
        state = self._state
        labels: dict[HintButton, str] = {}
        # The hint row still fits at 80x24 (76 cells); only the 60x20 row needs
        # to shed it. Gate on the measured page width, not the pane breakpoint.
        show_move = state in ("sources", "sessions") and (not self._narrow or self.size.width >= 74)
        if show_move:
            labels[self._hint_move] = _t(_kt.hint_move())
        if state == "sources":
            labels[self._hint_enter] = _t(_kt.hint_open())
            labels[self._hint_reload] = _t(_kt.hint_reload())
        elif state == "sessions":
            labels[self._hint_space] = _t(_kt.hint_select())
            labels[self._hint_all] = _t(_kt.all_none())
            labels[self._hint_enter] = _t(_kt.hint_preview())
        elif state == "preview":
            labels[self._hint_enter] = (
                _t(_kt.hint_confirm()) if self._only_metadata_only() else _t(_kt.hint_import())
            )
        elif state == "progress":
            labels[self._hint_cancel] = _t(_kt.hint_cancel())
        elif state == "result":
            labels[self._hint_enter] = _t(_kt.hint_list())
        labels[self._hint_esc] = _t(_kt.hint_back())
        return labels

    def _body_rich(self) -> Text:
        text = Text()
        if self._state == "loading":
            text.append(self._loading + "\n", style="dim")
            return text
        if self._state == "error":
            text.append(_t(_kt.error_header()) + "\n\n", style="bold")
            text.append(self._error + "\n")
            return text
        if self._state == "empty":
            text.append((self._empty_header or _t(_kt.empty_header())) + "\n\n", style="bold")
            text.append(self._empty + "\n")
            return text
        if self._state == "sources":
            return self._sources_rich()
        if self._state == "sessions":
            return self._sessions_rich()
        if self._state == "preview":
            return self._preview_rich()
        if self._state in ("progress", "result"):
            return self._job_rich()
        return text

    def _line_width(self) -> int:
        return max(20, (self._body.size.width or 60) - 2)

    def _sources_rich(self) -> Text:
        text = Text()
        width = self._line_width()
        text.append(_t(_kt.source_header()) + "\n\n", style="bold")
        if not self._sources or not any(info.detected for info in self._sources):
            text.append(_t(_kt.no_sources()) + "\n\n", style="dim")
        for index, info in enumerate(self._sources):
            selected = index == self._source_index
            row = _t(
                _kt.source_row(
                    label=info.label,
                    count=info.session_count,
                    availability=_availability_label(info),
                )
            )
            self._append_row(text, row, selected=selected, width=width)
            detail = info.detail or _home_relative(info.root) or _t(_kt.source_no_detail())
            text.append(
                truncate_cells("  " + _shorten_home(detail), max(10, width - 2)),
                style=Style(color=theme_mod.semantic_color("dim")),
            )
            text.append("\n")
        return text

    def _sessions_rich(self) -> Text:
        text = Text()
        width = self._line_width()
        text.append(
            _t(_kt.session_header(source=self._source_label(self._source))) + "\n\n", style="bold"
        )
        for index, row in enumerate(self._sessions):
            selected = index == self._session_index
            checked = row.source_id in self._selected
            title = normalise_title(row.title) or row.source_id
            date = _date(row.updated_ts or row.created_ts)
            # 2 cursor cells, 4 marker cells, 2 gap, 16 date, 1 safety cell.
            room = max(6, width - 2 - 4 - 2 - 16)
            title = truncate_cells(title, room)
            line = f"{title}  {date}"
            meta = _t(
                _kt.session_meta(
                    messages=int(row.message_count or 0),
                    compactions=int(row.compaction_count or 0),
                    availability=_availability_label(row),
                )
            )
            self._append_row(
                text,
                line,
                selected=selected,
                width=width,
                marker=_t(_kt.selected()) if checked else _t(_kt.unselected()),
                checked=checked,
            )
            text.append(
                truncate_cells("    " + meta, max(10, width - 4)),
                style=self._row_style(selected),
            )
            text.append("\n")
            if row.cwd:
                text.append(
                    truncate_cells("    " + _shorten_home(row.cwd), max(10, width - 4)),
                    style=self._row_style(selected, dim=True),
                )
                text.append("\n")
        return text

    def _append_row(
        self,
        text: Text,
        line: str,
        *,
        selected: bool,
        width: int,
        marker: str = "",
        checked: bool = False,
    ) -> None:
        background = theme_mod.semantic_color("tint-select") if selected else None
        # The row GROUND already says "selected"; the caret only points, so it
        # uses muted ink. The accent remains on the checked box. This is the
        # /resume rule and clears the light ramp's 4.29:1 accent-on-tint floor
        # (D7).
        cursor_style = Style(
            color=theme_mod.semantic_color("muted") if selected else None, bgcolor=background
        )
        text.append("› " if selected else "  ", style=cursor_style)
        if marker:
            text.append(
                marker + " ",
                style=Style(
                    color=(
                        theme_mod.semantic_color("accent")
                        if checked
                        else theme_mod.semantic_color("dim")
                    ),
                    bgcolor=background,
                    bold=checked,
                ),
            )
        marker_width = cell_len(marker) + 1 if marker else 0
        text.append(
            truncate_cells(line, max(8, width - 2 - marker_width)),
            style=self._row_style(selected),
        )
        text.append("\n")

    def _row_style(self, selected: bool, *, dim: bool = False) -> Style | None:
        if not selected:
            return Style(color=theme_mod.semantic_color("dim")) if dim else None
        return Style(bgcolor=theme_mod.semantic_color("tint-select"))

    def _body_leading_lines(self) -> int:
        """Rendered lines above the first clickable row.

        Both list renderers write a one-line heading followed by a blank line.
        Sources additionally write the no-sources block (two more lines) when
        no harness was detected. Click and scroll arithmetic must subtract
        this prefix or every hit lands one row late.
        """
        if self._state == "sources":
            no_sources = not self._sources or not any(info.detected for info in self._sources)
            return 2 + (2 if no_sources else 0)
        if self._state == "sessions":
            return 2
        return 0

    def _only_metadata_only(self) -> bool:
        selected = [row for row in self._sessions if row.source_id in self._selected]
        return bool(selected) and all(row.availability != "full" for row in selected)

    def _append_warning(self, text: Text, message: Any, width: int, *, prefix: str = "· ") -> None:
        body = " ".join(str(message).split())
        lines = _wrap_words(prefix + body, max(10, width)).splitlines() or [prefix]
        for line in lines:
            text.append(line + "\n", style="yellow")

    def _append_result_row(self, text: Text, row: dict[str, Any], width: int) -> None:
        for line in self._result_row(row).splitlines():
            text.append(truncate_cells(line, width) + "\n")

    def _preview_rich(self) -> Text:
        text = Text()
        width = self._line_width()
        text.append(_t(_kt.preview_header()) + "\n\n", style="bold")
        if self._preview is None:
            return text
        text.append(
            truncate_cells(
                _t(
                    _kt.preview_summary(
                        s=self._preview.total_sessions,
                        m=self._preview.total_messages,
                        c=self._preview.total_compactions,
                    )
                ),
                width,
            )
            + "\n"
        )
        if self._preview.warnings:
            for warning in self._preview.warnings[:4]:
                self._append_warning(text, warning.message, width)
            if len(self._preview.warnings) > 4:
                text.append(
                    _t(_kt.preview_warnings(count=len(self._preview.warnings))) + "\n",
                    style="dim",
                )
        if self._request is not None and self._request.include_memory:
            text.append(
                truncate_cells(
                    _t(
                        _kt.preview_memory(
                            records=self._preview.total_memory_records,
                            tokens=f"{self._preview.projected_memory_tokens:,}",
                            global_tokens=f"{self._preview.projected_global_tokens:,}",
                        )
                    ),
                    width,
                )
                + "\n",
                style="dim",
            )
        if self._has_metadata_only():
            text.append(
                "\n" + _wrap_words(_t(_kt.metadata_only_explain()), width) + "\n", style="dim"
            )
        only_metadata = self._only_metadata_only()
        if self._narrow:
            detail = (
                _t(_kt.preview_narrow_metadata(count=len(self._selected)))
                if only_metadata
                else _t(_kt.preview_narrow(count=len(self._selected)))
            )
            text.append("\n" + detail + "\n", style="dim")
        else:
            confirm = (
                _t(_kt.preview_confirm_metadata()) if only_metadata else _t(_kt.preview_confirm())
            )
            text.append("\n" + _wrap_words(confirm, width) + "\n", style="dim")
        return text

    def _has_metadata_only(self) -> bool:
        return any(
            row.source_id in self._selected and row.availability != "full" for row in self._sessions
        )

    def _job_rich(self) -> Text:
        text = Text()
        width = self._line_width()
        if self._job is None:
            return text
        if self._state == "progress":
            text.append(_t(_kt.progress_header()) + "\n\n", style="bold")
            progress = self._job.progress
            text.append(
                truncate_cells(
                    _t(
                        _kt.progress_row(
                            completed=str(progress.completed),
                            total=str(progress.total),
                            phase=_phase_label(progress.phase),
                        )
                    ),
                    width,
                )
                + "\n"
            )
            done = {str(row.get("source_id")): row for row in progress.results}
            wanted = _request_ids(self._job.request)
            if wanted:
                for session_id in wanted:
                    row = done.get(session_id)
                    if row is not None:
                        self._append_result_row(text, row, width)
                    elif progress.session_id == session_id:
                        text.append(
                            truncate_cells(_t(_kt.session_importing(session=session_id)), width)
                            + "\n"
                        )
                    else:
                        text.append(
                            truncate_cells(_t(_kt.session_pending(session=session_id)), width)
                            + "\n"
                        )
            else:
                for row in progress.results:
                    self._append_result_row(text, row, width)
                if progress.session_id and progress.session_id not in done:
                    text.append(
                        truncate_cells(
                            _t(_kt.session_importing(session=progress.session_id)), width
                        )
                        + "\n"
                    )
                pending = max(
                    0,
                    progress.total
                    - progress.completed
                    - (1 if progress.session_id and progress.session_id not in done else 0),
                )
                if pending:
                    text.append(_t(_kt.pending_count(count=pending)) + "\n", style="dim")
            return text
        text.append(_t(_kt.result_header()) + "\n\n", style="bold")
        result = self._job.result
        if result is None:
            return text
        text.append(
            truncate_cells(
                _t(
                    _kt.result_summary(
                        state=_state_label(result.state),
                        sessions=len(result.sessions),
                        messages=result.messages,
                    )
                ),
                width,
            )
            + "\n",
        )
        for row in result.sessions:
            self._append_result_row(text, row.to_dict(), width)
        if (
            result.memory_imported
            or result.memory_skipped
            or result.redactions
            or (self._request is not None and self._request.include_memory)
        ):
            text.append(
                truncate_cells(
                    _t(
                        _kt.result_memory(
                            imported=str(result.memory_imported),
                            skipped=str(result.memory_skipped),
                            redactions=str(result.redactions),
                        )
                    ),
                    width,
                )
                + "\n",
                style="dim",
            )
        warning_entries = _result_warning_entries(result)
        for prefix, message in warning_entries[:4]:
            self._append_warning(text, message, width, prefix=prefix)
        if any(row.outcome == "metadata_only" for row in result.sessions):
            text.append(
                "\n" + _wrap_words(_t(_kt.metadata_only_explain()), width) + "\n", style="dim"
            )
        for path in (result.instruction_review_path, result.memory_review_path):
            if path:
                key = (
                    _kt.review_instructions
                    if path == result.instruction_review_path
                    else _kt.review_memory
                )
                text.append(_wrap_words(_t(key(path=path)), width) + "\n", style="dim")
        if warning_entries:
            text.append(
                _t(_kt.result_warnings(count=len(warning_entries))) + "\n",
                style="yellow",
            )
        text.append("\n" + _wrap_words(_t(_kt.rerun_hint()), width) + "\n", style="dim")
        if result.error:
            text.append("\n" + result.error + "\n", style="red")
        return text

    def _result_row(self, row: dict[str, Any]) -> str:
        outcome = str(row.get("outcome", ""))
        warnings = row.get("warnings") or []
        if (
            outcome == "failed"
            and self._job is not None
            and self._job.progress.state == "cancelled"
            and "import cancelled" in warnings
        ):
            outcome = "cancelled"
        source_id = str(row.get("source_id", ""))
        target = str(row.get("target_session_id", "") or "")
        head = (
            _t(_kt.result_head(source_id=source_id, target=target))
            if target
            else _t(_kt.result_head_no_target(source_id=source_id))
        )
        detail = _t(
            _kt.result_detail(
                outcome=_outcome_label(outcome),
                messages=int(row.get("messages", 0)),
            )
        )
        # At 60x20 the body fits only the summary and one row line, so the
        # part a user must read (outcome + message count) leads; the target is
        # the second line and stays reachable by scrolling. Wide layouts keep
        # the natural source → target reading order.
        return detail + "\n" + head if self._narrow else head + "\n" + detail

    def _pane_text(self) -> str:
        if self._state == "preview" and self._preview is not None:
            # Warnings, memory counts and the confirmation live in the body;
            # repeating them in the pane made the same sentence appear twice on
            # one frame. The pane keeps only the page's standing promise.
            return _t(_kt.subtitle())
        if self._state == "progress" and self._job is not None:
            return _t(
                _kt.job_status(job=self._job.job_id, state=_state_label(self._job.progress.state))
            )
        if self._state in ("result", "error"):
            return ""
        return _t(_kt.subtitle())

    def _scroll_to_cursor(self) -> None:
        """Keep the cursor row in the viewport using the ACTUAL row heights."""
        if self._state == "sources":
            offsets = [2] * len(self._sources)
            index = self._source_index
        elif self._state == "sessions":
            offsets = [2 + (1 if row.cwd else 0) for row in self._sessions]
            index = self._session_index
        else:
            return
        if not offsets:
            return
        top = self._body_leading_lines() + sum(offsets[:index])
        height = max(1, self._body.size.height or 1)
        y = self._body.scroll_offset.y
        if top < y:
            target = top
        elif top + offsets[index] > y + height:
            target = top + offsets[index] - height
        else:
            return
        try:
            self._body.scroll_to(y=max(0, target), animate=False, immediate=True)
        except Exception:
            pass

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        if getattr(event, "button", 1) != 1:
            return
        if not self._body.region.contains(event.screen_x, event.screen_y):
            return
        line = (
            event.screen_y
            - self._body.region.y
            + self._body.scroll_offset.y
            - self._body_leading_lines()
        )
        if line < 0:
            return
        if self._state == "sources":
            index = line // 2
            if index < len(self._sources):
                self._source_index = index
                event.stop()
                self.action_activate()
            return
        if self._state == "sessions":
            offset = 0
            for index, row in enumerate(self._sessions):
                height = 2 + (1 if row.cwd else 0)
                if offset <= line < offset + height:
                    self._session_index = index
                    self.action_select()
                    event.stop()
                    return
                offset += height


def _result_warning_entries(result: Any) -> list[tuple[str, str]]:
    """Deduplicate the same failure recorded at result and session level.

    The core's failure path appends one ``ImportWarning`` to the result and the
    same text to ``SessionResult.warnings``. Rendering both made one failure
    read as two and doubled the count (R2-4). Dedupe on the source/session and
    normalized message, keeping the first (result-level) copy.
    """
    entries: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for warning in result.warnings:
        message = " ".join(str(warning.message).split())
        key = (str(warning.session_id), message)
        if key in seen:
            continue
        seen.add(key)
        entries.append(("· ", message))
    for session in result.sessions:
        for warning in session.warnings:
            message = " ".join(str(warning).split())
            key = (str(session.source_id), message)
            if key in seen:
                continue
            seen.add(key)
            entries.append((f"{session.source_id}: ", message))
    return entries


def _availability_label(info: Any) -> str:
    if getattr(info, "detected", True) is False and getattr(info, "session_count", 0) == 0:
        return _t(_kt.source_not_found())
    value = str(info.availability)
    mapping = {
        "full": _kt.availability_full,
        "metadata_only": _kt.availability_metadata_only,
        "unsupported": _kt.availability_unsupported,
    }
    function = mapping.get(value)
    return _t(function()) if function is not None else _outcome_label(value)


def _outcome_label(value: str) -> str:
    mapping = {
        "imported": _kt.outcome_imported,
        "resumed": _kt.outcome_resumed,
        "skipped_already_imported": _kt.outcome_skipped_already_imported,
        "metadata_only": _kt.outcome_metadata_only,
        "failed": _kt.outcome_failed,
        "cancelled": _kt.outcome_cancelled,
    }
    function = mapping.get(value)
    return _t(function()) if function is not None else value


def _state_label(value: str) -> str:
    mapping = {
        "queued": _kt.state_queued,
        "running": _kt.state_running,
        "done": _kt.state_done,
        "failed": _kt.state_failed,
        "cancelled": _kt.state_cancelled,
    }
    function = mapping.get(value)
    return _t(function()) if function is not None else value


def _phase_label(value: str) -> str:
    mapping = {
        "selecting": _kt.phase_selecting,
        "importing": _kt.phase_importing,
        "memory": _kt.phase_memory,
        "done": _kt.phase_done,
        "cancelled": _kt.phase_cancelled,
        "session": _kt.phase_session,
    }
    function = mapping.get(value)
    return _t(function()) if function is not None else value


def _request_ids(request: ImportRequest | None) -> list[str]:
    if request is None:
        return []
    ordered: list[str] = []
    for ids in request.session_ids.values():
        for session_id in ids:
            if str(session_id) not in ordered:
                ordered.append(str(session_id))
    return ordered


def _wrap_words(text: str, width: int) -> str:
    """Hard-wrap a catalogue sentence for the narrow body.

    The body Static wraps too, but a wrapped line can be taken for a new row;
    this keeps the paragraph block shaped the same at 60 and 150 columns.
    """
    if width <= 0:
        return text
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and cell_len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _shorten_home(value: Any) -> str:
    text = str(value)
    home = str(Path.home())
    if not home or home == "/":
        return text
    if text == home:
        return "~"
    return text.replace(home + "/", "~/").replace(home, "~")


def _home_relative(path: Any) -> str:
    return _shorten_home(path)


def _date(timestamp: float) -> str:
    if not timestamp:
        return "-"
    try:
        return dt.datetime.fromtimestamp(timestamp).astimezone().strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return "-"

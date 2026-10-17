"""The full-page ``/import`` view.

A MODE of the main screen, cloned from :class:`SettingsView`: the page takes
the transcript's region, leaves the dock in place and inert, and owns its own
cursor and keys. The service does the reading and writing; this widget only
paints and asks. It never parses a foreign store, never creates a target
itself, and never imports on a bare open: an import requires a selected source,
at least one selected session, a completed preview and a second Enter.

The layout reuses the settings page's CSS classes (title/rule/columns/body/
pane/detail/hints) rather than adding a second full-page skeleton. The body is
the source or session list, the pane is the preview/job/result summary, and the
detail line carries the live footer.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
        self._hints_text = Static()
        self._hints = Horizontal(self._hints_text, classes="settings-view-hints")
        self._sources: list[Any] = []
        self._sessions: list[Any] = []
        self._source_index = 0
        self._session_index = 0
        self._selected: set[str] = set()
        self._state = "loading"
        self._source = ""
        self._preview: Any = None
        self._request: ImportRequest | None = None
        self._job: Any = None
        self._error = ""
        self._poll = None

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
        self._paint()
        try:
            self.focus()
        except Exception:
            pass
        self._begin()

    def on_unmount(self) -> None:
        self._stop_poll()

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
            self._error = _t(_kt.no_sessions(source=source))
            self._state = "error"
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
            self._error = _t(_kt.error_select())
            self._state = "error"
            self._paint()
            return
        request = ImportRequest(
            sources=[self._source],
            session_ids={self._source: sorted(self._selected)},
        )
        self._state = "loading"
        self._paint()
        try:
            self._preview = await self._service.preview(request)
            self._request = request
            self._state = "preview"
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)
        self._paint()

    # -- job ----------------------------------------------------------------
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
        self._poll = self.set_interval(0.1, self._tick_job)

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
        try:
            payload = JOBS.cancel(session_key="tui")
        except KeyError:
            self._error = _t(_kt.error_cancel())
            self._state = "error"
            self._paint()
            return
        self._job = JOBS.get(str(payload.get("job_id") or ""), session_key="tui")
        self._state = "progress" if self._job is not None else "result"
        self._paint()

    def _show_status(self) -> None:
        self._job = JOBS.latest(session_key="tui")
        if self._job is None:
            self._error = _t(_kt.error_no_job())
            self._state = "error"
        elif self._job.result is not None:
            self._state = "result"
        else:
            self._state = "progress"
            self._poll = self.set_interval(0.1, self._tick_job)
        self._paint()

    # -- keys ---------------------------------------------------------------
    def action_move(self, delta: int) -> None:
        if self._state == "sources" and self._sources:
            self._source_index = (self._source_index + delta) % len(self._sources)
        elif self._state == "sessions" and self._sessions:
            self._session_index = (self._session_index + delta) % len(self._sessions)
        else:
            return
        self._paint()

    def action_page(self, delta: int) -> None:
        step = 8 * delta
        if self._state == "sources" and self._sources:
            self._source_index = max(0, min(len(self._sources) - 1, self._source_index + step))
        elif self._state == "sessions" and self._sessions:
            self._session_index = max(0, min(len(self._sessions) - 1, self._session_index + step))
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
        if self._state != "sessions":
            return
        self._selected = {row.source_id for row in self._sessions}
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
        if self._state in ("sources", "error"):
            self._error = ""
            self.run_worker(
                self._load_sources(), group="harness-import", exclusive=True, exit_on_error=False
            )

    def action_cancel(self) -> None:
        if self._state in ("progress", "loading"):
            self._cancel()

    def action_leave(self) -> None:
        if self._state in ("sessions", "preview", "error", "result"):
            if self._state == "result":
                self.post_message(HarnessImportViewDismissed())
            elif self._state == "sessions":
                self._state = "sources"
                self._paint()
            else:
                self._state = "sessions" if self._sessions else "sources"
                self._paint()
            return
        self.post_message(HarnessImportViewDismissed())

    # -- painting -----------------------------------------------------------
    def _fail(self, exc: BaseException) -> None:
        self._error = _t(_kt.error_generic(error=f"{type(exc).__name__}: {exc}"))
        self._state = "error"

    def _detail_text(self) -> str:
        """The selected row's own facts; empty keeps the footer from duplicating."""
        if self._state == "sources" and self._sources:
            info = self._sources[self._source_index]
            return f"{info.label} · {info.availability}"
        if self._state == "sessions" and self._sessions:
            row = self._sessions[self._session_index]
            return normalise_title(row.title) or row.source_id
        if self._state == "preview" and self._preview is not None:
            return _t(_kt.selected_count(count=str(len(self._selected))))
        if self._state in ("progress", "result") and self._job is not None:
            return f"{self._job.progress.state} · {self._job.job_id}"
        return ""

    def _hint(self) -> str:
        if self._state == "sources":
            return _t(_kt.hints_sources())
        if self._state == "sessions":
            return _t(_kt.hints_sessions())
        if self._state == "preview":
            return _t(_kt.hints_preview())
        if self._state == "progress":
            return _t(_kt.hints_progress())
        if self._state == "result":
            return _t(_kt.hints_result())
        return _t(_kt.hints_error())

    def _paint(self) -> None:
        self._title.update(Text(_t(_kt.title()), style="bold"))
        self._rule.update(Text("─" * 200, style="dim"))
        self._pane.update(Text(self._pane_text()))
        self._detail.update(Text(self._detail_text(), style="dim"))
        self._hints_text.update(Text(self._hint(), style="dim"))
        self._body_text.update(self._body_rich())
        self._scroll_to_cursor()

    def _body_rich(self) -> Text:
        text = Text()
        if self._state == "loading":
            text.append(_t(_kt.loading()) + "\n", style="dim")
            return text
        if self._state == "error":
            text.append(_t(_kt.error_header()) + "\n\n", style="bold")
            text.append(self._error + "\n")
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

    def _sources_rich(self) -> Text:
        text = Text()
        text.append(_t(_kt.source_header()) + "\n\n", style="bold")
        if not self._sources or not any(info.detected for info in self._sources):
            text.append(_t(_kt.no_sources()) + "\n\n", style="dim")
        for index, info in enumerate(self._sources):
            selected = index == self._source_index
            row = _t(
                _kt.source_row(
                    label=info.label,
                    count=info.session_count,
                    availability=info.availability,
                )
            )
            text.append(
                ("› " if selected else "  ") + row + "\n", style="reverse" if selected else ""
            )
            text.append(_t(_kt.source_path(path=_home_relative(info.root))) + "\n", style="dim")
        return text

    def _sessions_rich(self) -> Text:
        text = Text()
        text.append(_t(_kt.session_header(source=self._source)) + "\n\n", style="bold")
        for index, row in enumerate(self._sessions):
            selected = index == self._session_index
            checked = row.source_id in self._selected
            title = normalise_title(row.title) or row.source_id
            date = _date(row.updated_ts or row.created_ts)
            marker = _t(_kt.selected()) if checked else _t(_kt.unselected())
            line = f"{marker} {title}  {date}"
            meta = _t(
                _kt.session_meta(
                    messages=str(row.message_count or 0),
                    compactions=str(row.compaction_count or 0),
                    availability=row.availability,
                )
            )
            text.append(
                ("› " if selected else "  ") + line + "\n", style="reverse" if selected else ""
            )
            text.append("  " + meta + "\n", style="dim")
            if row.cwd:
                text.append(_t(_kt.session_cwd(cwd=row.cwd)) + "\n", style="dim")
        selected = _t(_kt.selected_count(count=str(len(self._selected))))
        text.append("\n" + selected + "\n", style="dim")
        return text

    def _preview_rich(self) -> Text:
        text = Text()
        text.append(_t(_kt.preview_header()) + "\n\n", style="bold")
        if self._preview is None:
            return text
        text.append(
            _t(
                _kt.preview_summary(
                    s=self._preview.total_sessions,
                    m=self._preview.total_messages,
                    c=self._preview.total_compactions,
                )
            )
            + "\n",
        )
        if self._preview.warnings:
            text.append(
                _t(_kt.preview_warnings(count=len(self._preview.warnings))) + "\n",
                style="yellow",
            )
        text.append("\n" + _t(_kt.preview_confirm()) + "\n", style="dim")
        return text

    def _job_rich(self) -> Text:
        text = Text()
        if self._job is None:
            return text
        state = "progress" if self._state == "progress" else "result"
        if state == "progress":
            text.append(_t(_kt.progress_header()) + "\n\n", style="bold")
            progress = self._job.progress
            text.append(
                _t(
                    _kt.progress_row(
                        completed=str(progress.completed),
                        total=str(progress.total),
                        phase=progress.phase,
                        session=progress.session_id or "-",
                    )
                )
                + "\n"
            )
            done = {str(row.get("source_id")): row for row in progress.results}
            for session_id in sorted(self._selected):
                row = done.get(session_id)
                if row is not None:
                    text.append(self._result_row(row) + "\n")
                elif progress.session_id == session_id:
                    text.append(_t(_kt.session_importing(session=session_id)) + "\n")
                else:
                    text.append(_t(_kt.session_pending(session=session_id)) + "\n")
            return text
        text.append(_t(_kt.result_header()) + "\n\n", style="bold")
        result = self._job.result
        if result is None:
            return text
        text.append(
            _t(
                _kt.result_summary(
                    state=result.state,
                    sessions=len(result.sessions),
                    messages=result.messages,
                )
            )
            + "\n\n",
        )
        for row in result.sessions:
            text.append(self._result_row(row.to_dict()) + "\n")
        for path in (result.instruction_review_path, result.memory_review_path):
            if path:
                key = (
                    _kt.review_instructions
                    if path == result.instruction_review_path
                    else _kt.review_memory
                )
                text.append(_t(key(path=path)) + "\n", style="dim")
        if result.warnings:
            text.append(
                _t(_kt.result_warnings(count=len(result.warnings))) + "\n",
                style="yellow",
            )
        text.append("\n" + _t(_kt.rerun_hint()) + "\n", style="dim")
        if result.error:
            text.append("\n" + result.error + "\n", style="red")
        return text

    def _result_row(self, row: dict[str, Any]) -> str:
        outcome = str(row.get("outcome", ""))
        # The core records a session interrupted by a cancel as ``failed`` with
        # the ``import cancelled`` warning. That is the job's control state, not
        # a defect in the source, so the page says what happened rather than
        # accusing an unchanged source of failing.
        warnings = row.get("warnings") or []
        if (
            outcome == "failed"
            and self._job is not None
            and self._job.progress.state == "cancelled"
            and "import cancelled" in warnings
        ):
            outcome = "cancelled"
        return _t(
            _kt.result_row(
                source_id=str(row.get("source_id", "")),
                target=str(row.get("target_session_id", "") or "-"),
                outcome=outcome,
                messages=int(row.get("messages", 0)),
            )
        )

    def _pane_text(self) -> str:
        if self._state == "preview" and self._preview is not None:
            warnings = self._preview.warnings
            if warnings:
                return "\n".join(
                    _t(_kt.preview_warning_row(message=warning.message)) for warning in warnings[:4]
                )
            return _t(_kt.preview_confirm())
        if self._state == "progress" and self._job is not None:
            return _t(
                _kt.job_status(
                    job=self._job.job_id,
                    state=self._job.progress.state,
                )
            )
        if self._state in ("result", "error"):
            return ""
        return _t(_kt.subtitle())

    def _scroll_to_cursor(self) -> None:
        if self._state == "sources":
            top = self._source_index * 2
        elif self._state == "sessions":
            top = self._session_index * 2
        else:
            return
        try:
            self._body.scroll_to(y=max(0, top - 2), animate=False, immediate=True)
        except Exception:
            pass


def _home_relative(path: Any) -> str:
    """A source path the way a person reads it; the tool result keeps paths out.

    This surface is the local UI, not a model result, so naming the detected
    root is useful; printing the scratch HOME's absolute temporary prefix is
    not. ``~`` is the same shortening the settings page uses for config.yml.
    """
    text = str(path)
    home = str(Path.home())
    if text == home:
        return "~"
    if text.startswith(home + "/"):
        return "~" + text[len(home) :]
    return text


def _date(timestamp: float) -> str:
    if not timestamp:
        return "-"
    try:
        return dt.datetime.fromtimestamp(timestamp).astimezone().strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return "-"

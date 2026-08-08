"""The dock-band subagent panel (item 6-TUI).

The status band already carries the ◍/⊞ counters; this panel is the DETAIL
surface — one row per task job in the session's job manager: label, state
glyph (a moving spinner while running, ✓/✗ once settled), elapsed time, and
the latest progress the engine relayed. A row is the click/Enter target for
the trajectory viewer (``widgets/trajectory.py``), which replays the child
session's retained events.

Rows are live by construction: the app repaints the panel on every Subagent*
event AND on the 1 Hz job poll, and the panel advances its own spinner while
anything is running — motion, not colour, says "alive": the accent green is
spent at exactly five sites (see the tcss preamble) and a sixth spinner is
not one of them. Settled rows follow the tool ledger's ink law: ✓ dim,
✗ danger, nothing else.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from rich.console import RenderableType
from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Static

from local_operator.ansi import strip_control_sequences
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.tool_card import format_duration

#: Spinner cadence shared with the status band: 12.5 fps is the app's one
#: notion of "this is moving", and two different speeds on one screen would
#: read as two different states.
_SPINNER_FRAMES = ("⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷")
_SPINNER_INTERVAL_S = 0.08

#: Status glyphs reuse the tool ledger's vocabulary: the outcome is readable
#: in a colourless frame (✓/✗), and failure is the only state that gets colour.
_GLYPH_DONE = "✓"
_GLYPH_FAILED = "✗"


def job_elapsed(job: Any) -> str:
    """How long the job has been on the ledger, in the ledger's own grammar.

    A running job is measured against now; a settled one against its settle
    time — so a row read five minutes after completion still reports the job's
    own duration, not how long ago the user glanced at it. Missing clocks
    degrade to ``0s`` rather than a negative or a crash: an observability row
    must not be able to take the app down.
    """
    try:
        start = float(getattr(job, "start_time", 0.0) or 0.0)
        settled = getattr(job, "settled_at", None)
        end = float(settled) if settled else time.time()
        return format_duration(max(end - start, 0.0))
    except Exception:
        return "0s"


class SubagentRow(Static):
    """One task job: bullet, label, state glyph, elapsed, latest progress.

    The whole row is the click target for the trajectory viewer, and Enter on
    a focused row does the same — the keyboard and the mouse must agree, or
    one of them is a guess.
    """

    can_focus = True
    BINDINGS = [
        Binding("enter", "open_trajectory", "Open trajectory", show=False),
    ]

    def __init__(self, job_id: str, on_open: Callable[[str], None]) -> None:
        super().__init__(classes="subagent-row")
        self._job_id = job_id
        self._on_open = on_open
        #: Painted fingerprint, so a repaint happens only when the row's
        #: inputs move — the spinner tick would otherwise redraw every row
        #: eight times a second.
        self._fingerprint: tuple[Any, ...] | None = None
        self._running = False

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def running(self) -> bool:
        return self._running

    def paint(self, job: Any, *, spinner_glyph: str) -> bool:
        """Rebuild the row from the job's live state; returns its running-ness."""
        status = str(getattr(job, "status", "running"))
        label = strip_control_sequences(str(getattr(job, "label", "") or self._job_id))
        running = status == "running" and not getattr(job, "queued", False)
        self._running = running

        details = getattr(job, "latest_details", None) or {}
        progress = ""
        if running:
            progress = str(details.get("progress") or "")
        else:
            # Settled rows carry the outcome's first line: the band row is
            # the summary, the trajectory is the detail, and the row between
            # them says which side of that split it is on.
            if status == "failed":
                progress = str(getattr(job, "error_text", "") or "")
            else:
                progress = str(getattr(job, "result_text", "") or "")
        progress = " ".join(strip_control_sequences(progress).split())
        elapsed = job_elapsed(job)

        fingerprint = (status, label, progress, elapsed, spinner_glyph if running else "")
        if fingerprint == self._fingerprint:
            return running
        self._fingerprint = fingerprint
        self.update(self._build_row(status, label, progress, elapsed, spinner_glyph, running))
        return running

    def _build_row(
        self,
        status: str,
        label: str,
        progress: str,
        elapsed: str,
        spinner_glyph: str,
        running: bool,
    ) -> RenderableType:
        fg = Style(color=theme_mod.semantic_color("fg"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        danger = Style(color=theme_mod.semantic_color("danger"))

        row = Text(no_wrap=True, overflow="ellipsis")
        row.append("• ", style=dim)
        row.append(label, style=fg)
        row.append("  ", style=dim)
        if running:
            # Motion says "alive"; the ink stays neutral. The glyph advances
            # on the panel's timer (``paint`` is re-called with the next
            # frame), so a stopped timer means a frozen row — visible at once.
            row.append(spinner_glyph, style=muted)
        elif status == "failed":
            row.append(_GLYPH_FAILED, style=danger)
        elif status == "cancelled":
            row.append("⊘", style=dim)
        else:
            row.append(_GLYPH_DONE, style=dim)
        row.append(f" {elapsed}", style=dim)
        if progress:
            row.append("  ", style=dim)
            # Dim for settled noise, muted for live progress: the running
            # row's one changing fact deserves the brighter of the quiet inks.
            row.append(progress, style=muted if running else dim)
        return row

    def action_open_trajectory(self) -> None:
        self._on_open(self._job_id)

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self._on_open(self._job_id)


class SubagentPanel(Container):
    """The task-job list in the dock band.

    Visibility follows the ledger: shown while the manager has ANY task job
    (running or settled within retention — the manager's own sweep evicts the
    rest), hidden when it has none. Rows are keyed by job id and reused
    across repaints, so focus survives the 1 Hz refresh.
    """

    def __init__(self, on_open: Callable[[str], None]) -> None:
        super().__init__(id="subagent-panel", classes="band-slot")
        self._on_open = on_open
        self._header = Static(id="subagent-header", classes="band-body")
        self._list = Vertical(id="subagent-rows")
        self._rows: dict[str, SubagentRow] = {}
        #: Last ledger read, keyed by job id. Refresh repopulates it; the
        #: spinner tick repaints from it between refreshes rather than
        #: re-querying the manager eight times a second.
        self._jobs_by_id: dict[str, Any] = {}
        self._spinner_index = 0
        self._spinner_timer = None
        #: Fingerprint of the header row's inputs.
        self._header_shown: tuple[int, int] | None = None
        self.display = False

    def compose(self):  # type: ignore[override]
        yield self._header
        yield self._list

    def on_unmount(self) -> None:
        self._stop_spinner()

    # -- refresh -------------------------------------------------------------
    def refresh(self, session: Any) -> None:
        """Re-read ``session.jobs`` and repaint the rows.

        Called on every Subagent* event (immediate) and on the 1 Hz poll (the
        belt to the events' suspenders — elapsed time moves with no event at
        all). Never raises: this is a status surface.
        """
        try:
            manager = getattr(session, "jobs", None)
            jobs = manager.list() if manager is not None else []
        except Exception:
            jobs = []
        task_jobs = [job for job in jobs if getattr(job, "type", "") == "task"]
        if not task_jobs:
            self._jobs_by_id = {}
            self._sync_rows([])
            self.display = False
            self._stop_spinner()
            return
        self.display = True
        self._jobs_by_id = {str(getattr(job, "id", "") or ""): job for job in task_jobs}
        self._sync_rows(task_jobs)
        self._paint_all()
        if any(row.running for row in self._rows.values()):
            self._start_spinner()
        else:
            self._stop_spinner()

    def _sync_rows(self, jobs: list[Any]) -> None:
        """Bring the row set into agreement with the ledger (add/drop/order)."""
        seen: set[str] = set()
        order: list[str] = []
        for job in jobs:
            job_id = str(getattr(job, "id", "") or "")
            if not job_id:
                continue
            seen.add(job_id)
            order.append(job_id)
            if job_id not in self._rows:
                self._rows[job_id] = SubagentRow(job_id, self._on_open)
                self._list.mount(self._rows[job_id])
        for job_id in list(self._rows):
            if job_id not in seen:
                self._rows.pop(job_id).remove()
        # The manager hands jobs back in start order; keep the DOM in the
        # same order so a stable ledger paints a stable list.
        children = [self._rows[job_id] for job_id in order]
        if list(self._list.children) != children:
            for index, row in enumerate(children):
                self._list.move_child(row, before=index)

    def _paint_all(self) -> None:
        glyph = _SPINNER_FRAMES[self._spinner_index]
        for job_id, row in self._rows.items():
            job = self._jobs_by_id.get(job_id)
            if job is not None:
                row.paint(job, spinner_glyph=glyph)
        self._paint_header()

    def _paint_header(self) -> None:
        running = sum(1 for row in self._rows.values() if row.running)
        total = len(self._rows)
        fingerprint = (running, total)
        if fingerprint == self._header_shown:
            return
        self._header_shown = fingerprint
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        header = Text(no_wrap=True, overflow="ellipsis")
        header.append("Subagents", style=muted)
        header.append(" · ", style=dim)
        header.append(f"{running}/{total}", style=dim)
        self._header.update(header)

    def _job_for(self, job_id: str) -> Any:
        """The ledger's current record for ``job_id``, read fresh at spin time.

        The tick advances between refreshes, and status/progress can move in
        that gap, so the repaint reads the manager rather than the snapshot —
        one cheap dict get per moving row, never a crash path.
        """
        session = getattr(self.app, "_session", None)
        manager = getattr(session, "jobs", None) if session is not None else None
        if manager is None:
            return None
        try:
            return manager.get(job_id)
        except Exception:
            return None

    # -- spinner -------------------------------------------------------------
    def _start_spinner(self) -> None:
        if self._spinner_timer is None:
            self._spinner_timer = self.set_interval(_SPINNER_INTERVAL_S, self._tick)

    def _stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    def _tick(self) -> None:
        self._spinner_index = (self._spinner_index + 1) % len(_SPINNER_FRAMES)
        glyph = _SPINNER_FRAMES[self._spinner_index]
        # Repaint ONLY the rows that move: settled rows are skipped by the
        # running check, so the common case — one live row among several
        # settled ones — repaints one widget eight times a second, not all.
        for row in self._rows.values():
            if row.running:
                job = self._job_for(row.job_id)
                if job is not None:
                    row.paint(job, spinner_glyph=glyph)

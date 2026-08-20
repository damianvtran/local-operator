"""The dock-band wake panel: the session's scheduled wakes, above the input.

Mirrors :class:`~local_operator.tui.widgets.todo_panel.TodoPanel` — a
transparent band SLOT that hides itself when the session has no wakes, an
equality guard so the 1 Hz poll repaints only on change, and a row budget read
from the screen so the band stays inside short terminals. The point of the
panel is that a session's autonomy is otherwise invisible: a wake fires with
no keystroke, and without a standing list the only way to know the session
will wake at 09:00 is to catch the delivery line as it scrolls past.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.style import Style
from rich.text import Text
from textual.containers import Container
from textual.widgets import Static

from local_operator.harness.wake import format_duration
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.tool_card import truncate_cells

#: The most wake rows the band will spend. ``MAX_WAKE_SCHEDULES`` is 16, far
#: more than the band can afford; the cap plus an overflow marker keeps a
#: full scheduler from eating the transcript.
MAX_WAKE_ROWS = 3
#: Rows the panel never shrinks below while displayed (header + one wake).
_MIN_BODY_ROWS = 2
#: The dock's fixed rows around the band — the same figure ``TodoPanel``
#: budgets against (``_DOCK_ROWS``), kept in step by the panels sharing one
#: band. Ceiling/floor arithmetic lives in :meth:`WakePanel._body_rows`.
_DOCK_ROWS = 8


class WakePanel(Container):
    """The session's scheduled wakes, rendered in the dock band above todos.

    One row per SCHEDULE, not per occurrence: a wake that fires every hour
    for a week is still one schedule (``w1``), so it gets one line naming its
    next fire and a snippet of its prompt — the recurrence is stated once on
    that line, never re-listed per trigger. Visibility follows the scheduler:
    ``display: none`` while it holds no schedules, so the band collapses.
    """

    def __init__(self) -> None:
        super().__init__(id="wake-panel", classes="band-slot")
        self._body = Static(classes="band-body", id="wake-body")
        #: What is painted: the per-schedule fingerprint AND the row budget it
        #: was rendered against, so the 1 Hz poll repaints only when either
        #: moved (``TodoPanel``'s discipline — same contents, different space
        #: is a different paint).
        self._shown: tuple[tuple[tuple[str, ...], ...], int] | None = None
        # Hidden until the first schedule exists: an empty panel is not content.
        self.display = False

    def compose(self):  # type: ignore[override]
        yield self._body

    # -- sync -----------------------------------------------------------------
    def sync(self, session: Any) -> None:
        """Re-read the scheduler and repaint only on change.

        Called on the app's 1 Hz band poll. A wake's next-fire time moves with
        the wall clock between events, so the due label rides IN the
        fingerprint and the once-a-minute rollover is caught on the next poll
        rather than a tick late. Never raises: a status surface must not be
        able to take the app down.
        """
        try:
            scheduler = getattr(session, "wake_scheduler", None)
            schedules = list(scheduler.schedules) if scheduler is not None else []
            fingerprint = tuple(self._fingerprint(schedule) for schedule in schedules)
            budget = self._body_rows()
            state = (fingerprint, budget)
            if state == self._shown:
                return  # equality guard — identical list and budget = no work
            self._shown = state
            if not fingerprint:
                self.display = False
                return
            self.display = True
            self._body.update(self._build(fingerprint))
        except Exception:
            self.display = False

    @staticmethod
    def _fingerprint(schedule: Any) -> tuple[str, ...]:
        """One schedule as a paint-relevant tuple.

        The due label is rounded to the MINUTE: a sub-minute drift in the wall
        clock must not count as a change, or the once-a-second poll would
        repaint a panel whose visible text did not move.
        """
        due = datetime.fromtimestamp(schedule.next_due_at / 1000).astimezone()
        due_label = due.strftime("%H:%M" if due.date() == datetime.now().date() else "%b %d %H:%M")
        every = f"every {format_duration(schedule.every_ms)}" if schedule.every_ms else "once"
        message = " ".join(str(schedule.message).split())
        return (str(schedule.id), due_label, every, message)

    # -- rendering ------------------------------------------------------------
    def _build(self, rows: tuple[tuple[str, ...], ...]) -> Text:
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))

        header = Text(no_wrap=True, overflow="ellipsis")
        header.append("Wakes", style=muted)
        header.append(" · ", style=dim)
        header.append(f"{len(rows)} scheduled" if len(rows) != 1 else "1 scheduled", style=muted)

        room = max(1, self._body_rows() - 1)
        cap = min(room, MAX_WAKE_ROWS)
        marker = len(rows) > cap
        if marker:
            # Reserve a row for the "… N more" marker. At the floor budget
            # (room == 1) this drops the one visible wake in favour of the
            # count — a single "w1 …" line beside a silent "+5 hidden" is the
            # bigger lie, since the header's total already implies the misses.
            cap = min(max(room - 1, 0), MAX_WAKE_ROWS)
            if len(rows) == cap + 1:
                # "… 1 more wake" costs exactly the row the wake itself costs.
                cap += 1
                marker = False
        visible = rows[:cap]

        cells = self._row_cells()
        lines = [header]
        for wake_id, due_label, every, message in visible:
            row = Text(no_wrap=True, overflow="ellipsis")
            row.append("- ", style=dim)
            row.append(wake_id, style=muted)
            row.append(f" {due_label} · {every}", style=dim)
            if message:
                snippet = truncate_cells(message, max(cells - cell_len_of(row) - 3, 0))
                if snippet:
                    row.append(f" — {snippet}", style=dim)
            lines.append(row)
        if marker:
            overflow = Text(no_wrap=True, overflow="ellipsis")
            overflow.append(f"… {len(rows) - len(visible)} more wakes", style=dim)
            lines.append(overflow)
        return Text("\n").join(lines)

    # -- geometry (the TodoPanel budget discipline) ----------------------------
    def predicted_rows(self) -> int:
        """Content rows this panel will paint, for a caller that cannot measure."""
        try:
            content = str(self._body.content)
        except Exception:
            content = ""
        if content:
            return max(1, len(content.split("\n")))
        return max(1, self._body_rows())

    def _body_rows(self) -> int:
        """Rows this paint may fill — header, wakes and any overflow marker.

        Falls back to the ceiling whenever the screen cannot be consulted —
        a panel synced before mount, in a test, or on a reduced host must
        still paint rather than hide itself (an exception here used to land
        in ``sync``'s guard and flip the panel invisible off-app).
        """
        ceiling = MAX_WAKE_ROWS + 2
        try:
            screen_height = self.screen.size.height
        except Exception:  # no screen yet (tests, reduced hosts, pre-mount)
            return ceiling
        if screen_height <= 0:
            return ceiling
        try:
            spare = screen_height - _DOCK_ROWS - self._band_inset_rows() - self._band_sibling_rows()
        except Exception:
            return ceiling
        return max(_MIN_BODY_ROWS, min(ceiling, spare))

    def _band_inset_rows(self) -> int:
        try:
            parent = self.parent
        except Exception:  # not mounted
            return 0
        if parent is None:
            return 0
        try:
            return 1 if parent.has_class("has-slot") else 0
        except Exception:
            return 0

    def _band_sibling_rows(self) -> int:
        try:
            parent = self.parent
        except Exception:  # not mounted (tests, reduced hosts)
            return 0
        if parent is None:
            return 0
        from local_operator.tui.app import slot_rows

        return sum(slot_rows(slot) for slot in parent.children if slot is not self and slot.display)

    def _row_cells(self) -> int:
        """Cells one row may occupy, or a safe default before layout."""
        try:
            width = self.size.width
        except Exception:
            width = 0
        return max(width - 2, 24) if width else 80


def cell_len_of(text: Text) -> int:
    """Display cells a ``Text`` occupies (no style), for snippet budgeting."""
    from local_operator.tui.widgets.transcript import cell_len

    return cell_len(text.plain)

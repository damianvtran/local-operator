"""The dock-band wake panel: the session's scheduled wakes, above the input.

Mirrors :class:`~local_operator.tui.widgets.todo_panel.TodoPanel` — a
transparent band SLOT that hides itself when the session has no wakes, an
equality guard so the 1 Hz poll repaints only on change, and a row budget read
from the screen so the band stays inside short terminals. The point of the
panel is that a session's autonomy is otherwise invisible: a wake fires with
no keystroke, and without a standing list the only way to know the session
will wake at 09:00 is to catch the delivery line as it scrolls past.

**It also reports a wake that is NOT being delivered.** A schedule whose
supervisor engagement keeps failing used to paint exactly like a healthy one,
so the surface an operator actually watches could not say the thing the wake
subsystem exists to prevent (scheduled work silently not running). The due
slot now shows the supervisor's delivery state and how long the fire has been
owed, in the warning ink, read from the supervisor's own ledger
(:mod:`local_operator.wakes.deliveries`) — one vocabulary with ``lop wake
status``, and one fact (the age) the row otherwise lacks.
"""

from __future__ import annotations

import time
from typing import Any

from rich.style import Style
from rich.text import Text
from textual.containers import Container
from textual.widgets import Static

from local_operator.harness.wake import format_duration
from local_operator.tui import theme as theme_mod
from local_operator.wakes.display import format_wake_time

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

#: Transcript rows this panel may never take, the same floor
#: ``todo_panel._COLLAPSED_TRANSCRIPT_FLOOR`` and
#: ``subagent_panel._TRANSCRIPT_FLOOR_ROWS`` honour. The three panels share one
#: column, so a floor observed by two of them is simply a floor the third
#: spends — which is what left a resumed session on a short terminal with the
#: conversation off screen (UX round 2, U7).
_COLLAPSED_TRANSCRIPT_FLOOR = 2


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
        #: What is painted: the per-schedule fingerprint AND the row/width budgets it
        #: was rendered against, so the 1 Hz poll repaints only when either
        #: moved (``TodoPanel``'s discipline — same contents, different space
        #: is a different paint).
        self._shown: tuple[tuple[tuple[str, ...], ...], int, int] | None = None
        #: ``(config root, deliveries dir mtime_ns, ledger)`` — see :meth:`_owed`.
        self._owed_cache: tuple[str, int, dict[str, dict[str, Any]]] | None = None
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
        rather than a tick late. The supervisor's delivery state rides in it too
        (:meth:`_owed`), so a wake whose engagement starts failing repaints on
        the next tick without either surface polling harder. Never raises: a
        status surface must not be able to take the app down.
        """
        try:
            scheduler = getattr(session, "wake_scheduler", None)
            schedules = list(scheduler.schedules) if scheduler is not None else []
            owed = self._owed(session)
            fingerprint = tuple(self._fingerprint(schedule, owed) for schedule in schedules)
            budget = self._body_rows()
            state = (fingerprint, budget, self._row_cells())
            if state == self._shown:
                return  # equality guard — identical list and budgets = no work
            self._shown = state
            if not fingerprint:
                self.display = False
                return
            self.display = True
            self._body.update(self._build(fingerprint))
        except Exception:
            self.display = False

    # -- the supervisor's delivery ledger --------------------------------------
    def _owed(self, session: Any) -> dict[str, Any] | None:
        """The session's owed fire from the supervisor's ledger, if it has one.

        MEMOISED ON THE LEDGER DIRECTORY'S MTIME, because this runs on the 1 Hz
        band poll: every mutation of a record goes through ``write_delivery``'s
        ``os.replace`` or ``remove_delivery``'s ``unlink``, both of which land
        inside that directory and move its ``st_mtime_ns``, so the common case
        costs one ``stat`` per tick rather than one read per owed wake. A
        filesystem with coarser timestamp resolution can serve a record up to
        its own granularity late — bounded by the tick that would have read it
        again anyway.

        Reads the same root the CLI and the supervisor use, so both surfaces
        make one statement about one file. Returns ``None`` for a session with
        no id, no ledger, or an unreadable one: a delivery mark is an addition
        to the row, never a precondition for painting it.
        """
        try:
            session_id = str(getattr(session, "session_id", "") or "")
            if not session_id:
                return None
            from local_operator.paths import config_dir
            from local_operator.wakes import deliveries

            root = config_dir()
            directory = deliveries.deliveries_dir(root)
            try:
                stamp = directory.stat().st_mtime_ns
            except OSError:
                self._owed_cache = None
                return None
            if self._owed_cache is None or self._owed_cache[:2] != (str(directory), stamp):
                self._owed_cache = (str(directory), stamp, deliveries.read_deliveries(root))
            return self._owed_cache[2].get(session_id)
        except Exception:
            return None

    @staticmethod
    def _owed_label(owed: dict[str, Any], now_ms: int) -> str:
        """``retrying · owed 9d`` for the due slot, or ``""`` when not owed.

        The AGE is the fact the row lacked, and it is the one that separates
        "about to fire on a busy host" from "stuck since last week" without
        the reader having to know what the state words mean. A record with no
        usable first-attempt stamp still gets its state word.
        """
        state = str(owed.get("state") or "retrying")
        first = owed.get("first_attempt_ms")
        if isinstance(first, int) and not isinstance(first, bool):
            # WHOLE SECONDS: `format_duration` is a compound renderer that falls
            # back to raw milliseconds for a sub-second remainder, which turned a
            # nine-day age into `777600440ms`. The remainder carries no
            # information at this granularity.
            age_ms = max(now_ms - first, 0) // 1000 * 1000
            return f"{state} · owed {format_duration(age_ms)}"
        return state

    @classmethod
    def _fingerprint(cls, schedule: Any, owed: dict[str, Any] | None = None) -> tuple[str, ...]:
        """One schedule as a paint-relevant tuple.

        The due label is rounded to the MINUTE: a sub-minute drift in the wall
        clock must not count as a change, or the once-a-second poll would
        repaint a panel whose visible text did not move.

        ``owed`` is the session's ledger record. It replaces the due label —
        which is the state the reader needs — when the record names THIS
        occurrence, and its presence is also what selects the warning ink, so a
        delivery that starts or stops failing is a repaint at the next tick.
        """
        due_label = format_wake_time(schedule.next_due_at)
        ink = "dim"
        record = owed
        if record is not None and record.get("occurrence_ms") == schedule.next_due_at:
            label = cls._owed_label(record, int(time.time() * 1000))
            if label:
                due_label = label
                ink = "warning"
        every = f"every {format_duration(schedule.every_ms)}" if schedule.every_ms else "once"
        message = " ".join(str(schedule.message).split())
        return (str(schedule.id), due_label, every, message, ink)

    # -- rendering ------------------------------------------------------------
    def _build(self, rows: tuple[tuple[str, ...], ...]) -> Text:
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        warning = Style(color=theme_mod.semantic_color("warning"))

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
        for wake_id, due_label, every, message, ink in visible:
            row = Text(no_wrap=True, overflow="ellipsis")
            row.append("- ", style=dim)
            row.append(wake_id, style=muted)
            row.append(" ", style=dim)
            # THE SLOT CARRIES THE INK: a healthy wake's due label is dim like
            # the rest of the row, an owed fire's state is the warning colour —
            # the same two-fact split the CLI's DUE column makes, in a band that
            # cannot afford a column of its own.
            row.append(due_label, style=warning if ink == "warning" else dim)
            row.append(f" · {every}", style=dim)
            if message:
                row.append(f" — {message}", style=dim)
            lines.append(row)
        if marker:
            overflow = Text(no_wrap=True, overflow="ellipsis")
            overflow.append(f"… {len(rows) - len(visible)} more wakes", style=dim)
            lines.append(overflow)
        # The band is content-sized: Rich overflow alone cannot constrain its
        # natural width. Clamp every row, including a long id/date/recurrence,
        # against the screen just as TodoPanel does, before it is measured.
        for line in lines:
            line.truncate(cells, overflow="ellipsis")
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
            spare = (
                screen_height
                - _DOCK_ROWS
                - _COLLAPSED_TRANSCRIPT_FLOOR
                - self._band_inset_rows()
                - self._band_sibling_rows()
            )
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
        """Screen cells minus the body rail, never the content-sized band width.

        Longer local clock labels can otherwise grow the band beyond a narrow
        terminal. Include this width in the paint fingerprint so a resize also
        recomputes truncation without waiting for the schedule to change.
        """
        try:
            width = self.screen.size.width
        except Exception:
            width = 0
        return max(width - 2, 1) if width else 80

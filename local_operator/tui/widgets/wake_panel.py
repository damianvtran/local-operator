"""The dock-band wake panel: the session's scheduled wakes, above the input.

Mirrors :class:`~local_operator.tui.widgets.todo_panel.TodoPanel` — a
transparent band SLOT that hides itself when the session has no wakes, an
equality guard so the 1 Hz poll repaints only on change, and a row budget read
from the screen so the band stays inside short terminals. The point of the
panel is that a session's autonomy is otherwise invisible: a wake fires with
no keystroke, and without a standing list the only way to know the session
will wake at 08:30 is to catch the delivery line as it scrolls past.

Monitor rows join the SAME band (design §12): wake rows first, then a monitor
section — one line per armed monitor with its due/state slot and its health —
under the same shared row budget. One band, not a fourth panel: the dock's
rows are a shared column, and a new sibling would spend everyone's floor (U7).

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
from local_operator.wakes.display import format_age, format_wake_time

#: The most wake rows the band will spend. ``MAX_WAKE_SCHEDULES`` is 16, far
#: more than the band can afford; the cap plus an overflow marker keeps a
#: full scheduler from eating the transcript.
MAX_WAKE_ROWS = 3
#: The most monitor rows the band will spend (design §12), and the floor below
#: which the monitor section is dropped whole — its header plus one row. A
#: half-section (a header with no row, or rows with no header) reads as a
#: different defect than "there is more than fits", so the band shows one or
#: nothing; the wake cap's own marker arithmetic (``_row_window``) is what
#: "fits" means on either side.
MAX_MONITOR_ROWS = 2
_MIN_MONITOR_SECTION_ROWS = 2
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


def _overflow_line(label: str, dim: Style) -> Text:
    """The "… N more" marker line a section appends when rows are hidden."""
    line = Text(no_wrap=True, overflow="ellipsis")
    line.append(label, style=dim)
    return line


#: Where the band's copy of a shared health SENTENCE stops. The sentence is
#: written once (``monitors.store.health_hint``) and the CLI, the agent tool and
#: the desktop route all print it whole — but the band is a two-row glance
#: surface, and the explanation half of "tool unavailable since 12:56 — retrying"
#: is what pushed the monitor's NAME out of the row at 100 columns. Splitting on
#: the sentence's own clause separator keeps ONE spelling: the band shows the
#: state, the surfaces with room show the state and its consequence.
_BAND_HEALTH_SEPARATOR = " — "


def _band_health_clause(sentence: str) -> str:
    """The state clause of a shared health sentence (see above)."""
    return sentence.split(_BAND_HEALTH_SEPARATOR, 1)[0].strip()


class WakePanel(Container):
    """The session's scheduled wakes, rendered in the dock band above todos.

    One row per SCHEDULE, not per occurrence: a wake that fires every hour
    for a week is still one schedule (``w1``), so it gets one line naming its
    next fire and a snippet of its prompt — the recurrence is stated once on
    that line, never re-listed per trigger. Monitor rows ride the same band
    under their own section (design §12), wake rows first. Visibility follows
    the schedulers: ``display: none`` while the session holds neither wakes
    nor monitors, so the band collapses.
    """

    def __init__(self) -> None:
        super().__init__(id="wake-panel", classes="band-slot")
        self._body = Static(classes="band-body", id="wake-body")
        #: What is painted: the per-schedule and per-monitor fingerprints AND
        #: the row/width budgets they were rendered against, so the 1 Hz poll
        #: repaints only when either moved (``TodoPanel``'s discipline — same
        #: contents, different space is a different paint).
        self._shown: (
            tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...], int, int] | None
        ) = None
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
            # HIDDEN rows (patience waits) never paint: the panel is a human
            # surface, and the requirement is "no wake listing, no badge" for
            # a timer the user was never told about (design §8.2.2 item 5).
            # The one shared filter (``wakes.store.scheduled_rows``) is used
            # here too, so the panel cannot disagree with the CLI or the
            # sidebar about which rows exist. Filtering before the
            # fingerprint also means a patience-only session collapses the
            # panel entirely, which is exactly what its empty state means.
            from local_operator.wakes.store import scheduled_rows

            schedules = scheduled_rows(scheduler.schedules) if scheduler is not None else []
            monitors = self._monitor_rows(session)
            owed = self._owed(session)
            fingerprint = tuple(self._fingerprint(schedule, owed) for schedule in schedules)
            budget = self._body_rows()
            state = (fingerprint, monitors, budget, self._row_cells())
            if state == self._shown:
                return  # equality guard — identical list and budgets = no work
            self._shown = state
            if not fingerprint and not monitors:
                self.display = False
                return
            self.display = True
            self._body.update(self._build(fingerprint, monitors))
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
            # This is an approximate age, not round-trippable schedule syntax:
            # even whole-second ages need more than two terms after a poll tick.
            return f"{state} · owed {format_age(max(now_ms - first, 0) / 1000)}"
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

    def _monitor_rows(self, session: Any) -> tuple[tuple[str, ...], ...]:
        """Monitor rows as paint-relevant tuples — the wake fingerprint's twin.

        Read from the LIVE scheduler (the runtime the session owns), not the
        derived index: a monitor's health lives in its counters (next due, the
        disabled reason, the failure count the auto-disable ladder walks), and
        the index is only rewritten on change events, so a cold reader would
        paint stale health on the surface that exists to make it visible.
        Defensive like the rest of ``sync``: any failure yields no monitor
        rows rather than taking the band down.
        """
        try:
            scheduler = getattr(session, "monitor_scheduler", None)
            specs = list(scheduler.monitors) if scheduler is not None else []
        except Exception:
            return ()
        rows: list[tuple[str, ...]] = []
        for spec in specs:
            counters: dict[str, Any] = {}
            if scheduler is not None:
                try:
                    runtime = scheduler.runtime(spec.id)
                    if runtime is not None:
                        counters = dict(runtime.counters or {})
                except Exception:
                    counters = {}
            rows.append(self._monitor_fingerprint(spec, counters))
        return tuple(rows)

    @classmethod
    def _monitor_fingerprint(cls, spec: Any, counters: dict[str, Any]) -> tuple[str, ...]:
        """One monitor as a paint-relevant tuple; the wake fingerprint's shape.

        ``(id, due_or_state, every, name, ink, health, rank)``. Two rules here
        are the whole reason the band can be read at a glance:

        - **The state word takes the DUE slot** for every state a reader must
          not miss (``disabled`` / ``stalled`` / ``idle``), the way the wake row
          already does for an owed fire. The band has no health column, and the
          clock it replaces is the least informative thing in the row (design
          review round 1, D1/D2 — at 60 columns the sentence after the name was
          the first thing truncation took, and `idle` was invisible entirely).
        - **The health sentence sits BEFORE the name and the interval**, so a
          narrow row loses `every 1m` and the label rather than the hint.

        ``rank`` orders the section worst-first (D5): the cap is two rows, and a
        hidden row must never be the one carrying the news.
        """
        from local_operator.monitors import store as monitor_store

        now_ms = int(time.time() * 1000)
        row = {
            **counters,
            "id": getattr(spec, "id", ""),
            "tool": getattr(spec, "tool", ""),
            "created_at": getattr(spec, "created_at", 0) or 0,
            "every_ms": getattr(spec, "every_ms", None),
        }
        disabled = bool(counters.get("disabled"))
        since = counters.get("unavailable_since")
        unavailable = isinstance(since, int) and not isinstance(since, bool) and since > 0
        failures = counters.get("consecutive_failures")
        fail_count = failures if isinstance(failures, int) and not isinstance(failures, bool) else 0
        failing = fail_count > 0
        idle = (not disabled) and (not unavailable) and monitor_store.is_idle(row, now_ms)
        hint = monitor_store.health_hint(row, now_ms)

        due = counters.get("next_due_at")
        if disabled:
            label = "disabled"
        elif unavailable:
            label = "stalled"
        elif idle:
            label = "idle"
        elif isinstance(due, int) and not isinstance(due, bool):
            label = format_wake_time(due)
        else:
            label = "waiting"

        ink = "dim"
        rank = 6
        health = ""
        if disabled:
            health = " ".join(str(counters.get("disabled_reason") or "").split())
            ink, rank = "warning", 0
        elif unavailable:
            # AN UNAVAILABLE EPISODE OUTRANKS A STALE FAILURE COUNT (D3): the
            # count froze when the tool went out of reach, because unavailable
            # ticks charge no strike — and "3 failed" where the CLI says
            # "retrying" is the cross-surface disagreement §D6 exists to stop.
            health = _band_health_clause(hint or "tool unavailable — retrying")
            ink, rank = "warning", 1
        elif idle:
            # The state word is the whole message here; the CLI carries the
            # "overdue by 3h" tail for readers who go looking.
            health = ""
            ink, rank = "warning", 2
        elif hint is not None and hint.startswith("never checked"):
            health = _band_health_clause(hint)
            ink, rank = "warning", 3
        elif failing:
            health = f"{fail_count} failed"
            ink, rank = "warning", 5
        elif hint:
            # NEUTRAL by design — a quiet watch may be watching something quiet
            # — but painted in MUTED rather than DIM: the dim token measures
            # 4.18:1 on the band panel, below the AA floor, which made the new
            # hint the least legible text in the frame (D8).
            health = _band_health_clause(hint)
            ink, rank = "muted", 4

        every_ms = getattr(spec, "every_ms", None)
        every = f"every {format_duration(every_ms)}" if every_ms else "once"
        name = " ".join(str(getattr(spec, "name", "")).split())
        return (str(spec.id), label, every, name, ink, health, str(rank))

    # -- rendering ------------------------------------------------------------
    def _build(
        self,
        rows: tuple[tuple[str, ...], ...],
        monitors: tuple[tuple[str, ...], ...] = (),
    ) -> Text:
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        warning = Style(color=theme_mod.semantic_color("warning"))

        budget = self._body_rows()
        lines: list[Text] = []
        if rows:
            # Wake rows first (design §12). The wake section keeps the band's
            # original contract and the first share of the budget; the monitor
            # section reserves the floor it needs (its header + one row), and
            # when the budget cannot afford even that the wakes win whole —
            # two half-sections each claiming rows they cannot show is the
            # worse frame, and the wake rows are the band's original contract.
            room = (
                budget if not monitors else max(_MIN_BODY_ROWS, budget - _MIN_MONITOR_SECTION_ROWS)
            )
            lines.extend(self._wake_section(rows, room=room, dim=dim, muted=muted, warning=warning))
        if monitors:
            room = (budget - len(lines)) if rows else budget
            if room >= _MIN_MONITOR_SECTION_ROWS:
                lines.extend(
                    self._monitor_section(
                        monitors, room=room, dim=dim, muted=muted, warning=warning
                    )
                )
        # The band is content-sized: Rich overflow alone cannot constrain its
        # natural width. Clamp every row, including a long id/date/recurrence,
        # against the screen just as TodoPanel does, before it is measured.
        cells = self._row_cells()
        for line in lines:
            line.truncate(cells, overflow="ellipsis")
        return Text("\n").join(lines)

    @staticmethod
    def _row_window(count: int, room: int, limit: int) -> tuple[int, bool]:
        """How many rows fit under a section header, and whether to mark the rest.

        The wake cap's arithmetic, extracted so the monitor section cannot
        drift from it. When a row would be hidden, reserve a row for the
        "… N more" marker; at the floor (``room == 0``) the visible rows are
        dropped in favour of the count — one "w1 …" line beside a silent
        "+5 hidden" is the bigger lie, since the header's total already
        implies the misses. When EXACTLY one row would be hidden, showing it
        costs the same row the marker costs, so it is shown instead.
        """
        room = max(0, room)
        cap = min(room, limit)
        marker = count > cap
        if marker:
            cap = min(max(room - 1, 0), limit)
            if count == cap + 1:
                cap += 1
                marker = False
        return cap, marker

    def _wake_section(
        self,
        rows: tuple[tuple[str, ...], ...],
        *,
        room: int,
        dim: Style,
        muted: Style,
        warning: Style,
    ) -> list[Text]:
        """The wake half of the band. ``room`` counts its header too."""
        header = Text(no_wrap=True, overflow="ellipsis")
        header.append("Wakes", style=muted)
        header.append(" · ", style=dim)
        header.append(f"{len(rows)} scheduled" if len(rows) != 1 else "1 scheduled", style=muted)

        cap, marker = self._row_window(len(rows), max(1, room) - 1, MAX_WAKE_ROWS)
        visible = rows[:cap]
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
            lines.append(_overflow_line(f"… {len(rows) - len(visible)} more wakes", dim))
        return lines

    def _monitor_section(
        self,
        rows: tuple[tuple[str, ...], ...],
        *,
        room: int,
        dim: Style,
        muted: Style,
        warning: Style,
    ) -> list[Text]:
        """The monitor half of the band, the wake section's twin.

        One row per armed monitor: id, the due-or-state slot, its interval,
        its name, and — when there is one — the health tail (the disabled
        reason, or the failure count the auto-disable ladder is walking,
        §11.3).
        """
        header = Text(no_wrap=True, overflow="ellipsis")
        header.append("Monitors", style=muted)
        header.append(" · ", style=dim)
        # The count names what is ARMED, and calls out the broken rows: a header
        # reading "1 watching" directly above a row reading "disabled" says
        # something the row itself contradicts (design review round 1, D7).
        broken = sum(1 for row in rows if row[4] == "warning")
        count = f"{len(rows)} monitor" if len(rows) == 1 else f"{len(rows)} monitors"
        if broken:
            attention = "1 needs attention" if broken == 1 else f"{broken} need attention"
            header.append(f"{count} · {attention}", style=muted)
        else:
            header.append(f"{count} armed", style=muted)

        # WORST FIRST, then the cap: with two visible rows the ones behind the
        # marker may not be the ones carrying the news (D5).
        ordered = sorted(rows, key=lambda row: int(row[6]))
        cap, marker = self._row_window(len(rows), max(1, room) - 1, MAX_MONITOR_ROWS)
        visible = ordered[:cap]
        lines = [header]
        for monitor_id, label, every, name, ink, health, _rank in visible:
            row = Text(no_wrap=True, overflow="ellipsis")
            row.append("- ", style=dim)
            row.append(monitor_id, style=muted)
            row.append(" ", style=dim)
            tone = warning if ink == "warning" else muted if ink == "muted" else dim
            row.append(label, style=tone)
            # HEALTH BEFORE THE BOILERPLATE (D1): a narrow row then loses the
            # interval and the name, never the sentence that says what is wrong.
            if health:
                row.append(f" · {health}", style=tone)
            if name:
                row.append(f" — {name}", style=dim)
            row.append(f" · {every}", style=dim)
            lines.append(row)
        if marker:
            lines.append(_overflow_line(f"… {len(rows) - len(visible)} more monitors", dim))
        return lines

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
        """Rows this paint may fill — both headers, both sections and markers.

        Falls back to the ceiling whenever the screen cannot be consulted —
        a panel synced before mount, in a test, or on a reduced host must
        still paint rather than hide itself (an exception here used to land
        in ``sync``'s guard and flip the panel invisible off-app).
        """
        # Two headers + both section caps + one overflow marker each.
        ceiling = MAX_WAKE_ROWS + MAX_MONITOR_ROWS + 4
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

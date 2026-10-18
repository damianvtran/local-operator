"""Pure derivations for the ``/schedule`` page: what a wake IS, as text.

WHY A MODEL MODULE. The page's job is to list, create, edit and cancel THIS
conversation's wakes, and every one of those answers has to agree with the
sentences the rest of the product already uses — the wake tool's refusals, the
CLI's ``format_duration``, the band panel's ``format_wake_time``. A widget that
parsed or formatted any of it itself would be the second vocabulary this repo
has paid for repeatedly. So the parsing, the validation call and the rendered
fields live here, importable without Textual, and the widget only paints.

THE WRITE IS NOT HERE. ``create``/``edit`` validate; the caller applies the
result to the scheduler's full list and hands it to
``Session.set_wake_schedules`` — the session's ONE writer. This module never
persists, never arms, and never reads a file.

THE FULL LIST, NOT THE VISIBLE ONE. ``scheduled_rows`` is the human filter:
patience waits and ask deadlines are internal timers that share the wake store
and must never appear on this page. They are still passed to the builders and
returned in the write list, because a write that dropped them would cancel
timers the user never asked about — the reason every function here takes and
returns the scheduler's OWNED view.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from local_operator.harness.wake import (
    build_wake_edit,
    build_wake_schedule,
    format_duration,
    parse_wake_duration,
)
from local_operator.harness.wake_types import WakeSchedule
from local_operator.i18n import keys as i18n_keys
from local_operator.i18n.text import tr
from local_operator.wakes.display import format_age, format_wake_time
from local_operator.wakes.store import scheduled_rows

_t = tr("tui.schedule")
_k = i18n_keys.tui_schedule

#: A wake due within this window reads "due now" rather than "overdue": the
#: scheduler pumps on its own tick, and a row a few seconds past its instant is
#: ordinary queueing, not a missed fire. A minute is also the first rung of the
#: age ladder below it, so it is the smallest age the page can state at all —
#: choosing anything smaller would print "overdue 5s" beside a wake that is
#: about to fire.
DUE_GRACE_MS = 60_000

#: The one separator this page joins row fragments with. Chrome, not copy: no
#: locale translates a middle dot, and keeping it in one place is what stops the
#: list row and the detail pane from drifting apart.
SEPARATOR = " · "


@dataclass(frozen=True, slots=True)
class WhenField:
    """A parsed When input: which request key it fills, and its value."""

    key: str  # "in" | "at"
    value: str


@dataclass(frozen=True, slots=True)
class FormValues:
    """Exactly what the form's five controls hold, as text plus a flag."""

    message: str = ""
    when: str = ""
    repeat: str = ""
    ends: str = ""
    notify: bool = False


@dataclass(frozen=True, slots=True)
class SaveResult:
    """One save attempt: the built row, or the builder's own sentence.

    ``sentence`` is what the page paints on success too — the receipt names the
    row and its new instant — so the form has ONE line to render whatever the
    outcome was.
    """

    schedule: WakeSchedule | None = None
    error: str = ""
    sentence: str = ""

    @property
    def ok(self) -> bool:
        return self.schedule is not None


def when_field(text: str) -> WhenField | None:
    """Classify the When input into the request key it fills (``None`` = blank).

    The form has ONE field because a user thinks "when", not "in or at". The
    rule that splits it is the same one the wake tool's schema teaches: a bare
    DURATION is the relative spelling (``30m`` = ``in 30m``), while a bare clock,
    ISO timestamp or ``+duration`` is the absolute one (``09:00`` = ``at
    09:00``). Anything unrecognised is handed to the builder as an ``at`` so its
    sentence — which names all three accepted forms — is the refusal a user
    reads, rather than a page-local guess.
    """
    stripped = text.strip()
    if not stripped:
        return None
    lowered = stripped.casefold()
    if lowered == "in" or lowered.startswith("in "):
        return WhenField("in", stripped[2:].strip())
    if lowered == "at" or lowered.startswith("at "):
        return WhenField("at", stripped[2:].strip())
    if parse_wake_duration(stripped) is not None:
        return WhenField("in", stripped)
    return WhenField("at", stripped)


def when_request(text: str) -> dict[str, str]:
    """The request fragment the When field contributes (``{}`` when blank)."""
    field = when_field(text)
    return {} if field is None else {field.key: field.value}


def repeat_request(text: str) -> dict[str, Any]:
    """``{"every": interval-or-None}`` — blank is a one-shot (or a clear).

    ``every`` is always PRESENT: on a create ``None`` means no interval, and on
    an edit it means the user blanked a recurring wake into a one-shot. An
    absent key would mean "keep the row's cadence", which is the opposite of
    what an empty Repeat field says.
    """
    stripped = text.strip()
    if not stripped:
        return {"every": None}
    lowered = stripped.casefold()
    if lowered == "every" or lowered.startswith("every "):
        stripped = stripped[5:].strip()
    return {"every": stripped}


def ends_request(text: str) -> dict[str, Any]:
    """``{"limit": n-or-None, "until": text-or-None}`` — blank is no bound.

    Both keys are PRESENT for the same reason ``every`` is in
    :func:`repeat_request`: a blank Ends field CLEARS a bound, and the edit
    builder reads a present ``None`` as exactly that. Non-numeric, non-positive
    and unparseable values ride through untouched so the builder's own
    sentences are the refusals.
    """
    stripped = text.strip()
    if not stripped:
        return {"limit": None, "until": None}
    lowered = stripped.casefold()
    if lowered == "limit" or lowered.startswith("limit "):
        value = stripped[5:].strip()
        try:
            return {"limit": int(value)}
        except ValueError:
            return {"limit": value}
    if lowered == "until" or lowered.startswith("until "):
        return {"until": stripped[5:].strip()}
    try:
        return {"limit": int(stripped)}
    except ValueError:
        return {"until": stripped}


def _request(values: FormValues) -> dict[str, Any]:
    return {
        "message": values.message,
        "notify": values.notify,
        **when_request(values.when),
        **repeat_request(values.repeat),
        **ends_request(values.ends),
    }


def create(values: FormValues, existing: Sequence[WakeSchedule], now_ms: int) -> SaveResult:
    """Validate a create through ``build_wake_schedule`` and name the result."""
    outcome = build_wake_schedule(_request(values), list(existing), now_ms)
    if "error" in outcome:
        return SaveResult(error=outcome["error"], sentence=outcome["error"])
    schedule = outcome["schedule"]
    return SaveResult(
        schedule=schedule,
        sentence=notice_created(schedule.id, schedule.next_due_at),
    )


def edit(
    values: FormValues, existing: Sequence[WakeSchedule], wake_id: str, now_ms: int
) -> SaveResult:
    """Validate an edit through ``build_wake_edit`` and name the result.

    The builder is the merge: it keeps the row's id, fires and creation stamp,
    and keeps ``next_due_at`` unless the When field contributes ``in``/``at``.
    A blank When is therefore a real instruction — "leave the time alone" — and
    cannot be expressed by omitting the key here, because a create would then
    have no time at all and the two forms share one request shape.
    """
    outcome = build_wake_edit(_request(values), list(existing), wake_id, now_ms)
    if "error" in outcome:
        return SaveResult(error=outcome["error"], sentence=outcome["error"])
    schedule = outcome["schedule"]
    return SaveResult(
        schedule=schedule,
        sentence=notice_edited(schedule.id, schedule.next_due_at),
    )


def prefilled(schedule: WakeSchedule) -> FormValues:
    """The form's starting values for an EDIT of ``schedule``.

    When is deliberately BLANK: the row's instant is already stored exactly,
    and re-typing it into a text field is how a DST fold moves a value by an
    hour. The bound is prefilled as an offset-bearing ISO timestamp for the
    same reason — a naive local string would be re-read under whatever offset
    applies on the day, which is not necessarily the one the row was armed in.
    """
    ends = ""
    if schedule.limit is not None:
        ends = f"limit {schedule.limit}"
    elif schedule.until_at is not None:
        until = datetime.fromtimestamp(schedule.until_at / 1000.0).astimezone().isoformat()
        ends = f"until {until}"
    return FormValues(
        message=schedule.message,
        when="",
        repeat=format_duration(schedule.every_ms) if schedule.every_ms is not None else "",
        ends=ends,
        notify=schedule.notify,
    )


def with_schedule(existing: Sequence[WakeSchedule], schedule: WakeSchedule) -> list[WakeSchedule]:
    """``existing`` with ``schedule`` replacing its id, or appended.

    Returns a NEW list: the caller's is the scheduler's own view, and a write
    that mutated it in place would corrupt the page's next read.
    """
    rows = list(existing)
    for index, row in enumerate(rows):
        if row.id == schedule.id:
            rows[index] = schedule
            return rows
    rows.append(schedule)
    return rows


def without_schedule(existing: Sequence[WakeSchedule], wake_id: str) -> list[WakeSchedule]:
    """``existing`` minus ``wake_id`` — a cancel, as the writer's full list."""
    return [row for row in existing if row.id != wake_id]


@dataclass(frozen=True, slots=True)
class ScheduleRow:
    """One user-facing wake, already rendered for both the list and the pane."""

    id: str
    message: str
    next_due_at: int
    #: The list's first line: handle plus a one-line prompt excerpt.
    head: str
    #: The list's second line: the fragments below, joined.
    meta: str
    #: ``(label, value)`` pairs for the detail pane, in reading order.
    detail: tuple[tuple[str, str], ...]
    relative_due: str
    cadence: str
    ends: str
    fired: str
    remaining: str
    notify: str
    status: str


def _until_text(delta_ms: int) -> str:
    """A live clock delta, on ``format_age``'s single-unit ladder.

    NOT ``format_duration``, measured twice on rendered frames: that formatter
    is built for the round values a schedule STORES (30m, 1d) and decomposes at
    most two terms, so the real delta to an armed instant — 7,199,251 ms for
    "two hours from now", and a second later 7,198,251 — fell through its
    documented raw-milliseconds fallback (``in 7199251ms``). ``format_age`` is
    the tree's answer for a moving clock: the CLI's own ``next due in …`` uses
    it, it floors to one unit (45s/12m/3h/2d), and it never prints a unitless
    number; the monitor listing's own ``next due in …`` already reads that way.
    The precise instant is the pane's ``Next`` row.
    """
    return format_age(max(0, delta_ms) / 1000)


def _relative_due(next_due_at: int, now_ms: int) -> str:
    remaining = next_due_at - now_ms
    if remaining > 0:
        return _t(_k.due_in(duration=_until_text(remaining)))
    if -remaining <= DUE_GRACE_MS:
        return _t(_k.due_now())
    return _t(_k.due_overdue(age=format_age((now_ms - next_due_at) / 1000)))


def _status(next_due_at: int, now_ms: int) -> str:
    remaining = next_due_at - now_ms
    if remaining > 0:
        return _t(_k.status_scheduled())
    if -remaining <= DUE_GRACE_MS:
        return _t(_k.status_due())
    return _t(_k.status_overdue(age=format_age((now_ms - next_due_at) / 1000)))


def _ends(schedule: WakeSchedule) -> str:
    if schedule.limit is not None:
        return _t(_k.ends_limit(limit=str(schedule.limit)))
    if schedule.until_at is not None:
        return _t(_k.ends_until(time=format_wake_time(schedule.until_at)))
    return ""


def _fired(schedule: WakeSchedule) -> str:
    if schedule.limit is not None:
        return _t(_k.fired_of(count=schedule.fired_count, limit=str(schedule.limit)))
    return _t(_k.fired(count=schedule.fired_count))


def _fired_detail(schedule: WakeSchedule) -> str:
    if schedule.limit is not None:
        return _t(_k.fired_detail_of(count=str(schedule.fired_count), limit=str(schedule.limit)))
    return _t(_k.fired_detail(count=str(schedule.fired_count)))


def _remaining(schedule: WakeSchedule, now_ms: int) -> str:
    if schedule.limit is not None:
        return _t(_k.ends_left(count=max(0, schedule.limit - schedule.fired_count)))
    if schedule.until_at is not None and schedule.until_at > now_ms:
        return _t(_k.ends_until_left(duration=_until_text(schedule.until_at - now_ms)))
    return ""


def _cadence(schedule: WakeSchedule) -> str:
    if schedule.every_ms is None:
        return _t(_k.cadence_once())
    return _t(_k.cadence_every(interval=format_duration(schedule.every_ms)))


def _notify(schedule: WakeSchedule) -> str:
    return _t(_k.notify_on()) if schedule.notify else ""


def _single_line(message: str) -> str:
    """The prompt's list excerpt is ONE line: a newline would break row height."""
    return " ".join(message.split())


def rows(schedules: Sequence[WakeSchedule], now_ms: int) -> list[ScheduleRow]:
    """The user-facing rows, in the scheduler's own order.

    Order is the scheduler's ``created_at`` sort, not the due time: the page is
    a ledger of what this conversation has armed, and re-sorting it by "next"
    would move a row under the cursor every time a repeat fired.
    """
    built: list[ScheduleRow] = []
    for schedule in scheduled_rows(schedules):
        cadence = _cadence(schedule)
        status = _status(schedule.next_due_at, now_ms)
        fired = _fired(schedule)
        ends = _ends(schedule)
        remaining = _remaining(schedule, now_ms)
        notify = _notify(schedule)
        relative_due = _relative_due(schedule.next_due_at, now_ms)
        meta = SEPARATOR.join(
            fragment for fragment in (relative_due, cadence, fired, remaining, notify) if fragment
        )
        # The prompt is NOT a labelled detail row: the pane already paints the
        # full message as its headline, and a "Prompt" row under it restated the
        # same sentence on every frame (design inspection, first crop).
        detail: list[tuple[str, str]] = [
            (_t(_k.detail_next()), format_wake_time(schedule.next_due_at)),
            (_t(_k.detail_cadence()), cadence),
        ]
        if ends:
            detail.append((_t(_k.detail_ends()), ends))
        detail.extend(
            [
                (_t(_k.detail_fired()), _fired_detail(schedule)),
                (
                    _t(_k.detail_notify()),
                    _t(_k.value_on()) if schedule.notify else _t(_k.value_off()),
                ),
                (_t(_k.detail_status()), status),
            ]
        )
        built.append(
            ScheduleRow(
                id=schedule.id,
                message=schedule.message,
                next_due_at=schedule.next_due_at,
                head=f"{schedule.id}  {_single_line(schedule.message)}",
                meta=meta,
                detail=tuple(detail),
                relative_due=relative_due,
                cadence=cadence,
                ends=ends,
                fired=fired,
                remaining=remaining,
                notify=notify,
                status=status,
            )
        )
    return built


def notice_created(wake_id: str, next_due_at: int) -> str:
    """The receipt a create paints — also what the model's own tests pin.

    Takes the INSTANT, not a preformatted string: the page and the builder's
    success path would otherwise each choose a format, and the receipt is the
    one place the row's new time is stated to the user.
    """
    return _t(_k.notice_created(id=wake_id, due=format_wake_time(next_due_at)))


def notice_edited(wake_id: str, next_due_at: int) -> str:
    return _t(_k.notice_edited(id=wake_id, due=format_wake_time(next_due_at)))


def notice_cancelled(wake_id: str) -> str:
    return _t(_k.notice_cancelled(id=wake_id))

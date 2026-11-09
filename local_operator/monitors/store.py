"""The monitor index: one small JSON file per monitor-carrying session.

``<config_dir>/monitors/<session_id>.json`` answers, for a process that has no
session open, *which sessions have monitors, and which are armed?* Absent file
⇒ no monitors.

**Derived, never authoritative** — the exact ``wakes/store.py`` contract. The
transcript's ``monitor_schedules`` entry (written by
``Session._persist_monitor_schedules``) is the source of truth; this file is a
projection the session rewrites immediately after every transcript append AND
on every open. The open-time rewrite is the self-healing property: a deleted,
corrupt or stale entry is repaired the next time the session is built, and an
entry for a session whose list was emptied is removed.

**Why it lives outside the session directory**: the junk-session reap deletes
whole session directories, and a cold scan (cleanup guards, the park marker,
``lop monitor status``) must not open 4,000 session directories to find the
twelve with monitors. Flat, one file per armed session; the state files live
under ``monitors/state/…`` and the index scan skips anything not ending
``.json``.

**Why this module must stay import-light**: its readers include the session
cleanup guards and the pristine probe, which run before any session is built.
Importing asyncio, pydantic, or anything under ``session``/``harness`` here
would put those on their path. So: stdlib only, pins in
``tests/unit/test_import_graph.py`` beside the wake siblings.

**Divergence from ``wakes/store.py``, deliberate (§10.2)**: this index is NOT
rewritten on every quiet tick — a 30 s monitor would rewrite it 2,880×/day for
no reader's benefit; the counters file alone moves, and a cold reader's
``next_due_at`` is best-effort between change events.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

#: Subdirectory of the config dir holding one ``<session_id>.json`` per
#: monitor-carrying session. Flat: see the module docstring.
MONITORS_DIRNAME = "monitors"

#: Bumped only on an incompatible change to the entry shape. Readers skip
#: entries whose schema they do not understand rather than guess; the owning
#: session rewrites the entry on its next open, so a bump heals like a
#: deleted file.
INDEX_SCHEMA = 1

#: How long an entry with no transcript must sit before the ghost sweep
#: removes it. The floor exists for one race: a session is being CREATED (its
#: directory is not written yet, or its transcript not yet appended) while its
#: index entry lands, and deleting that entry would lose a live arm. An hour
#: is far longer than any creation path takes and far shorter than the time an
#: operator waits to notice a ghost row.
GHOST_MIN_AGE_MS = 3_600_000


def monitors_dir(config_dir: Path) -> Path:
    return Path(config_dir) / MONITORS_DIRNAME


def entry_path(config_dir: Path, session_id: str) -> Path:
    return monitors_dir(config_dir) / f"{session_id}.json"


def _row_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    dump = getattr(row, "model_dump", None)
    if callable(dump):
        dumped = dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    raise TypeError(f"not a monitor row: {row!r}")


def is_held(entry: Mapping[str, Any] | None) -> bool:
    """Whether an entry is parked: ``stopped_at`` (the stop lever's marker).

    Monitor entries carry no ``held_at`` — there is no Aida engine for
    monitors — so this is the one key. Kept beside the entry shape so every
    reader does not re-spell it.
    """
    if not isinstance(entry, Mapping):
        return False
    return bool(entry.get("stopped_at"))


def has_armed(entry: Mapping[str, Any] | None, now_ms: int) -> bool:
    """Whether any row in the entry can still FIRE (§11.5's cleanup rule).

    A disabled monitor is failed state; an expired one (`until_at` passed) is
    finished. Neither arms the delete refusal — "a marker that cannot fire is
    not pending live work" (``_has_armed_wake``'s rationale) — while a durable
    or future-bounded one does. Tolerates malformed rows: a scan must not die
    on one bad file.
    """
    if not isinstance(entry, Mapping):
        return False
    for raw in entry.get("monitors") or ():
        if not isinstance(raw, Mapping):
            continue
        if raw.get("disabled"):
            continue
        until = raw.get("until_at")
        if isinstance(until, int) and not isinstance(until, bool) and until <= now_ms:
            continue
        return True
    return False


def next_due_at(entry: Mapping[str, Any]) -> int | None:
    """Earliest ``next_due_at`` across an entry's rows, or ``None``.

    Best-effort by contract: the index is only rewritten on change events, so
    a cold reader's value is the last-written one, not the live schedule.
    """
    earliest: int | None = None
    for raw in entry.get("monitors") or ():
        if not isinstance(raw, Mapping) or raw.get("disabled"):
            continue
        due = raw.get("next_due_at")
        if isinstance(due, bool) or not isinstance(due, int):
            continue
        if earliest is None or due < earliest:
            earliest = due
    return earliest


def read_entry(config_dir: Path, session_id: str) -> dict[str, Any] | None:
    """One session's entry, or ``None`` when absent or unreadable.

    Unreadable is treated exactly like absent — the transcript is the truth
    and the next open rewrites the file.
    """
    path = entry_path(config_dir, session_id)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("monitor index: unreadable entry %s; treating as absent", path)
        return None
    if not isinstance(data, dict) or data.get("schema") != INDEX_SCHEMA:
        logger.warning("monitor index: skipping entry %s with unknown schema", path)
        return None
    return data


def read_index(config_dir: Path) -> dict[str, dict[str, Any]]:
    """Every readable entry keyed by session id; a missing directory is empty."""
    return read_index_report(config_dir)[0]


def read_index_report(config_dir: Path) -> tuple[dict[str, dict[str, Any]], bool]:
    """``(entries, read_error)`` — plus whether the DIRECTORY could not be listed.

    The two empty answers are different claims ("no monitors" vs "cannot
    know"), the ``wakes/store.py`` distinction kept verbatim.
    """
    directory = monitors_dir(config_dir)
    try:
        names = sorted(os.listdir(directory))
    except FileNotFoundError:
        return {}, False
    except OSError:
        logger.warning("monitor index: cannot list %s", directory)
        return {}, True
    index: dict[str, dict[str, Any]] = {}
    for name in names:
        if not name.endswith(".json") or name.startswith("."):
            continue  # staged temp files, stray dotfiles and state/ are never entries
        session_id = name[: -len(".json")]
        entry = read_entry(config_dir, session_id)
        if entry is None:
            continue
        # The filename is the lookup key; a body whose session_id disagrees is
        # a copied file, and the filename wins because that is what the owning
        # session will overwrite.
        entry["session_id"] = session_id
        index[session_id] = entry
    return index, False


def write_entry(
    config_dir: Path,
    session_id: str,
    *,
    cwd: str,
    monitors: Sequence[Any],
    preserve: Mapping[str, Any] | None = None,
    clear: tuple[str, ...] = (),
) -> Path | None:
    """Write (replace) one session's entry, or remove it when empty.

    ``preserve`` keeps unknown keys across the rewrite (today ``stopped_at``,
    which the park marker stamps and the open-time rewrite clears); keys named
    in ``clear`` are dropped even when present in ``preserve``.
    """
    rows = [_row_dict(row) for row in monitors]
    if not rows:
        remove_entry(config_dir, session_id)
        return None
    entry: dict[str, Any] = {}
    if preserve:
        entry.update({k: v for k, v in preserve.items() if k not in clear})
    entry.update(
        {
            "schema": INDEX_SCHEMA,
            "session_id": session_id,
            "cwd": cwd,
            "updated_at": int(time.time() * 1000),
            "monitors": rows,
        }
    )
    path = entry_path(config_dir, session_id)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{session_id}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(entry, handle, separators=(",", ":"), sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


#: The floor under every staleness threshold below: an interval-derived bound
#: can be seconds long, and "overdue" on a 5 s monitor is not a hint.
HEALTH_MIN_STALE_MS = 300_000


def _stale_threshold_ms(row: Mapping[str, Any]) -> int:
    """``max(2 × every, 5 min)`` — when a monitor counts as not running."""
    every = row.get("every_ms")
    if isinstance(every, int) and not isinstance(every, bool) and every > 0:
        return max(2 * every, HEALTH_MIN_STALE_MS)
    return HEALTH_MIN_STALE_MS


def _clock(ms: int) -> str:
    return time.strftime("%H:%M", time.localtime(ms / 1000))


#: How many consecutive empty results a published row must carry before the
#: listing says so. Deliberately its own number rather than an import of the
#: scheduler's ``EMPTY_CHECKS_BEFORE_NOTICE``: this module is stdlib-only (see
#: ``health_hint``) and the reader's threshold is a rendering decision, not the
#: scheduler's policy — three is the same number because both answer "has it
#: EVER returned anything", not because one owns the other.
EMPTY_HINT_CHECKS = 3

#: How many consecutive ORDINARY failures the row calls "keeps failing". Mirrors
#: ``EMPTY_HINT_CHECKS``'s reasoning: this module is stdlib-only and the
#: reader's threshold is a rendering decision, not the scheduler's policy — two
#: is the same number because both answer "is this one blip or a run", not
#: because one owns the other.
STRIKING_HINT_CHECKS = 2

#: How much of a row's ``last_error`` the shared hint repeats (review round 1,
#: MINOR-2). The counter file keeps 500 chars — the NOTICE quotes the failure in
#: full, where the reader asked for it — but a list row is one line, and an
#: unclipped detail measured a single 639-char row next to a neighbour that clips
#: to 80.
HINT_DETAIL_CHARS = 120


def _clip_detail(text: str) -> str:
    """One line of whitespace-collapsed detail, bounded for a list row."""
    flat = " ".join(str(text or "").split())
    if len(flat) <= HINT_DETAIL_CHARS:
        return flat
    return flat[: HINT_DETAIL_CHARS - 1].rstrip() + "…"


def _count(value: Any) -> int:
    """A row field as a count, or 0 — ``bool`` is not an int here.

    The counters round-trip through JSON and the index is written by more than one
    composer, so a defensive read keeps a malformed row out of the sentences below
    instead of raising inside a listing.
    """
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def health_hint(row: Mapping[str, Any], now_ms: int | None) -> str | None:
    """One line explaining a monitor that is not doing what the reader assumes.

    THE shared helper for every listing surface (the agent tool's rows, the
    CLI, the desktop route, the TUI band), because the failure this closes was
    a DISCOVERABILITY one: the live store held monitors that had never checked
    (their session was never open), monitors with 0 deliveries after many
    checks, and one disabled by a flapping MCP tool — and every surface
    rendered all four as healthy rows.

    ``now_ms is None`` means THE CALLER HOLDS A PUBLISHED ROW, NOT A CLOCK: the
    owner writes a monitor row to the frontend state / index only on a change
    event (§10.2), never on a quiet tick, so a viewer's copy of ``checks`` and
    ``created_at`` is frozen at the last such event while its own clock keeps
    moving. Judging that copy against ``now`` invents a failure the owner never
    reported — the TUI band called a healthy, ticking watch "never checked" one
    threshold after arming (agent review round 1, MAJOR-1). Every hint derived
    from the row's own PUBLISHED fields (blind, unavailable, suppressed, the
    counts) still stands; only the comparison to ``now`` is declined.

    Returns ``None`` when there is nothing to say (a normal, working monitor),
    which is the common case and must stay silent. ``format_age`` is
    deliberately NOT used here: this module is stdlib-only and the reading
    surfaces own their own durations.
    """
    if isinstance(row, Mapping) and row.get("disabled"):
        # A disabled row already renders its own reason and count; the shared
        # hint would only restate it.
        return None
    if (
        isinstance(row, Mapping)
        and row.get("blind")
        # With no clock the expiry test is skipped rather than assumed: ``until_at``
        # is an absolute instant, so a caller that HAS a clock still lets it
        # outrank blind, and the caller without one simply keeps the sentence.
        and (now_ms is None or not _is_expired(row, now_ms))
    ):
        # Blind outranks every other hint: the watch is armed, its call still
        # runs, and nothing it sees reaches the operator — the exact state the
        # 2026-10-20 Slack monitor rendered as healthy for seven missed replies.
        # EXPIRED outranks blind (review round 1, F5): a watch past ``until_at``
        # will not tick again, so "cannot compare" would describe a failure that
        # can no longer produce a missed change.
        reason = " ".join(str(row.get("blind_reason") or "").split())
        detail = reason or "its full result is not available"
        return f"BLIND: cannot compare its result — {detail}; changes are not being reported"
    since = row.get("unavailable_since") if isinstance(row, Mapping) else None
    if isinstance(since, int) and not isinstance(since, bool) and since > 0:
        return f"tool unavailable since {_clock(since)} — retrying"
    if not isinstance(row, Mapping):
        return None
    # A FAILING WATCH MUST NOT READ AS A QUIET ONE (D8). Before this, a row whose
    # call had been erroring every tick fell through to "nothing has changed
    # (confirm the call observes what you expect)" — the sentence the incident's
    # reporter was staring at for 90 minutes while the call could not run at all.
    # The news leads, like the suppression clause below, because the TUI band
    # renders the first em-dash clause.
    # THE NEWS LEADS, and both failing sentences keep it in the first clause so a
    # band that renders clause one still shows it (D8).
    if row.get("failing") is True:
        detail = _clip_detail(str(row.get("last_error") or "")) or "its check returned an error"
        checks_n = _count(row.get("failing_checks"))
        threshold_n = _count(row.get("failing_notice_checks"))
        if checks_n and threshold_n and checks_n < threshold_n:
            # ONE failed check of a watch that has worked is a blip the notice policy
            # deliberately does not page on (``FAILING_CHECKS_AFTER_BASELINE``), so the
            # row must not claim the watch IS failing — it says the count and the
            # threshold instead (QA round 1, Q3). The threshold rides the row rather
            # than being re-derived here: this module is stdlib-only and the scheduler
            # owns the policy, including the no-baseline exception.
            plural = "" if checks_n == 1 else "s"
            return (
                f"{checks_n} failed check{plural} so far, below the {threshold_n}-check "
                f"threshold — {detail}"
            )
        # THE WHY LEADS, BESIDE THE NEWS: the TUI band renders a hint's first
        # em-dash clause, so a sentence that put the count first and the reason
        # after the dash showed "WATCH IS FAILING: 3 checks in a row" and dropped
        # the only part that tells the reader what to fix (review round 1, MINOR-4).
        tail = (
            f"{checks_n} checks in a row, nothing is being reported"
            if checks_n > 1
            else "nothing is being reported"
        )
        return f"WATCH IS FAILING: {detail} — {tail}"
    # The STRIKE class, which is a DIFFERENT fault from the two above (round-2
    # manager decision): the CHECK keeps failing — a timeout, a tool error — so
    # this watch really is heading for the disable and the row says so. One
    # failure is below the reporting threshold, and the row states that instead of
    # claiming the watch is failing (QA round 1, Q3).
    strikes = _count(row.get("consecutive_failures"))
    if strikes:
        detail = _clip_detail(str(row.get("last_error") or "")) or "its check returned an error"
        if strikes < STRIKING_HINT_CHECKS:
            plural = "" if strikes == 1 else "s"
            return (
                f"{strikes} failed check{plural} so far, below the "
                f"{STRIKING_HINT_CHECKS}-check threshold — {detail}"
            )
        limit = _count(row.get("failure_limit"))
        at = f" at {limit}" if limit else ""
        return f"the check keeps failing: {detail} — repeated failures disable it{at}"
    # The "returns nothing" episode is the other silent-but-not-quiet state. It says
    # only what is known (review round 1, MAJOR-2): the call HAS returned nothing so
    # far. It must NOT claim the watch can never report anything — an empty result is
    # a real digest, so the first non-empty one IS compared and delivered.
    empty_checks = row.get("empty_checks")
    if _count(empty_checks) >= EMPTY_HINT_CHECKS:
        return (
            f"the call has returned nothing for {_count(empty_checks)} checks so far — "
            "the watch is armed and reports the first result it gets"
        )
    threshold = _stale_threshold_ms(row)
    checks = row.get("checks")
    checks_n = int(checks) if isinstance(checks, int) and not isinstance(checks, bool) else 0
    if checks_n == 0:
        created = row.get("created_at")
        if now_ms is None or not isinstance(created, int) or isinstance(created, bool):
            # No clock (a published row) or no arming instant: no such claim can
            # be made. ``checks == 0`` on a snapshot means "no check had been
            # published when this row was written", which is not the same fact.
            return None
        if now_ms - created > threshold:
            return "never checked — its session was not open since arming"
        return None
    deliveries = row.get("deliveries")
    deliveries_n = (
        int(deliveries) if isinstance(deliveries, int) and not isinstance(deliveries, bool) else 0
    )
    if checks_n >= 3 and deliveries_n == 0:
        suppressed = _suppressed_total(row)
        if suppressed:
            # The neutral sentence below claims nothing CHANGED. A monitor that
            # suppressed its changes knows otherwise: the call did produce
            # deltas, the gate filed them as bookkeeping, and the reader is
            # owed that fact instead of "nothing has changed" — the live store
            # held ten such monitors, one with 93 suppressions. The NEWS LEADS
            # (design round 1, D1): the TUI band renders the first em-dash
            # clause, and the old ordering there deleted the suppression fact.
            plural = "" if suppressed == 1 else "s"
            return (
                f"{suppressed} change{plural} suppressed as non-material "
                f"({checks_n} checks, {deliveries_n} deliveries) — the call may have "
                "changed without telling you"
            )
        # Neutral, not a warning: a watch that has seen nothing may simply be
        # watching something quiet — or watching the wrong thing.
        return (
            f"{checks_n} checks, 0 deliveries — nothing has changed "
            "(confirm the call observes what you expect)"
        )
    return None


def _is_expired(row: Mapping[str, Any], now_ms: int) -> bool:
    """Whether ``until_at`` has passed — the one rule all surfaces share."""
    until = row.get("until_at")
    return isinstance(until, int) and not isinstance(until, bool) and until <= now_ms


def _suppressed_total(row: Mapping[str, Any]) -> int:
    """Every counted suppression on a row, whatever the gate called it.

    The counters file names the classes (``non_material_metadata``,
    ``ignorable``, ``rate_cap``) and an older file may carry others; the reader
    only needs the total, and it must tolerate a missing/malformed mapping
    because the index is best-effort derived state.
    """
    suppressed = row.get("suppressed")
    if not isinstance(suppressed, Mapping):
        return 0
    total = 0
    for value in suppressed.values():
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            total += value
    return total


def unavailable_since_of(row: Mapping[str, Any]) -> int:
    """The epoch-ms an unavailable episode began, or 0 (the wire's own reader)."""
    value = row.get("unavailable_since") if isinstance(row, Mapping) else None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 0


def is_idle(row: Mapping[str, Any], now_ms: int | None) -> bool:
    """Whether a monitor is overdue because nothing is hosting it (§D6).

    Overdue beyond ``max(2 × every, 5 min)`` and NOT held: a monitor ticks
    only while its session is open, so a dormant row is normal — but a reader
    looking at "next due 4 hours ago" needs to be told which of the two it is.

    ``now_ms is None`` (a caller holding a published row, see ``health_hint``)
    answers False, and the answer is not "healthy" but "this reader cannot
    judge the deadline". ``next_due_at`` is advanced by every tick, so a copy
    frozen at the last change event falls behind the wall clock and would turn
    a live watch into an "idle" warning on the band built to show it.
    """
    if now_ms is None:
        return False
    if not isinstance(row, Mapping) or row.get("disabled") or is_held(row):
        return False
    due = row.get("next_due_at")
    if not isinstance(due, int) or isinstance(due, bool):
        return False
    return now_ms - due > _stale_threshold_ms(row)


def striking_clause(row: Mapping[str, Any]) -> str:
    """The band's compressed form of the strike sentence (QA round 2, Q-R2-4).

    ``health_hint`` writes the sentence every surface with room prints, and the
    band shows only its FIRST clause — which for the strike class is the error,
    so the countdown that makes the row an alarm (this watch stops at the fifth
    failure) fell outside it and the glance surface showed neither reason nor
    countdown, just "2 failed". Moving the countdown into that clause is not a
    fix on its own: measured at 100 columns the full sentence runs to 106
    characters on a row that also has to name the monitor, and truncation then
    ate the countdown AND the name. So the band gets a shorter spelling — state,
    then the deadline, then the reason, which is the order width can afford to
    lose — while the CLI, the desktop rows and the lifecycle notice keep the
    full sentence. ``disabled_clause`` is the same arrangement for a disabled
    row: the store owns the words, the band owns the budget.
    """
    detail = _clip_detail(str(row.get("last_error") or "")) or "its check returned an error"
    limit = _count(row.get("failure_limit"))
    if limit:
        return f"keeps failing, stops at {limit}: {detail}"
    return f"keeps failing: {detail}"


def disabled_clause(row: Mapping[str, Any]) -> str:
    """The one-line cause a listing row leads with for a disabled monitor.

    The stored reason is whatever disabled the watch, and for the ordinary
    ladder that is the tool's own last error — one live class is a multi-line
    argument banner (``invalid arguments:\\n- path: Extra inputs are not
    permitted``), which is developer text on the surface a person reads to find
    out what happened (UX review round 1, U4). A reason that already reads as
    one sentence is KEPT, because that is the informative case ("monitor can't
    watch …: it is not in this session's tool set."); only a raw multi-line
    banner is replaced, by the failure the counters describe, in the words the
    lifecycle notice already uses.

    Whitespace is folded on the way out for the same round's U3: the CLI prints
    this inside a table cell, and a reason that still carried its newline broke
    the row onto column 0.
    """
    raw = str(row.get("disabled_reason") or "")
    folded = " ".join(raw.split())
    if folded and "\n" not in raw:
        return folded
    kind = str(row.get("disabled_kind") or "")
    if kind == "unreachable":
        return "its tool stayed unreachable for 24 hours"
    if kind == "fatal":
        return "a check that cannot succeed"
    failures = row.get("consecutive_failures")
    count = failures if isinstance(failures, int) and not isinstance(failures, bool) else 0
    return f"{count} consecutive failed checks" if count else "repeated failed checks"


def idle_detail(row: Mapping[str, Any], now_ms: int) -> str:
    """``overdue by 3h — session not open`` for an idle row (see :func:`is_idle`)."""
    due = row.get("next_due_at")
    overdue_ms = now_ms - int(due) if isinstance(due, int) and not isinstance(due, bool) else 0
    return f"overdue by {format_age_ms(overdue_ms)} — session not open"


def format_age_ms(ms: int) -> str:
    """A compact age: ``45s``, ``12m``, ``3h``, ``2d``.

    Local rather than imported: the reading surfaces that call the helpers
    above have their own formatters (``wakes.display.format_age`` lives behind
    a heavier import), and this module's whole contract is that nothing outside
    the stdlib is on its path.
    """
    seconds = max(0, int(ms // 1000))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    return f"{hours // 24}d"


def prune_ghost_entries(
    config_dir: Path,
    now_ms: int,
    *,
    session_exists: Callable[[str], bool],
) -> list[str]:
    """Remove index entries whose session no longer exists; returns the ids.

    A GHOST is an entry that can never be engaged: the session directory is
    gone (a reap, a hand-deleted directory, a QA scratch home), so the row
    still lists monitors that cannot fire and each engage burns a whole
    deadline proving it. The live store carried one (``9a7c31e40b22.json``,
    whose ``cwd`` pointed into a scratch home).

    THREE conditions, all required:

    - the session has no transcript. The predicate is INJECTED rather than
      imported so this module stays stdlib-only (``tests/unit/test_import_graph.py``
      pins that, and the supervisor's own guard is the caller's); a caller that
      cannot prove absence must pass a predicate that answers ``True``;
    - ``updated_at`` is older than :data:`GHOST_MIN_AGE_MS` (the creation race
      above);
    - the entry is not HELD (``stopped_at``) — a parked entry belongs to a
      session someone stopped on purpose, and archive is a hide flag rather
      than destruction (design monitor-tool.md §D5).

    Plus its ``state/<session_id>`` directory, through
    :func:`local_operator.monitors.state.remove_session_state`, which unlinks
    files and rmdirs. Callers treat this as maintenance: an unreadable index
    directory prunes nothing rather than raising.
    """
    from local_operator.monitors import state as monitor_state

    index, read_error = read_index_report(config_dir)
    if read_error:
        return []
    removed: list[str] = []
    for session_id, entry in index.items():
        if is_held(entry):
            continue
        updated_at = entry.get("updated_at")
        if not isinstance(updated_at, int) or isinstance(updated_at, bool):
            continue
        if now_ms - updated_at < GHOST_MIN_AGE_MS:
            continue
        if session_exists(session_id):
            continue
        if not remove_entry(config_dir, session_id):
            continue
        monitor_state.remove_session_state(config_dir, session_id)
        removed.append(session_id)
        logger.info("monitor index: pruned the ghost entry for %s", session_id)
    return removed


def remove_entry(config_dir: Path, session_id: str) -> bool:
    """Delete one session's entry. Idempotent: absent is success."""
    path = entry_path(config_dir, session_id)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True

"""Auto-re-engage a session runtime that died INVOLUNTARILY (the rescue pass).

**The gap this closes.** The wake supervisor re-engages only what is DUE — a
scheduled occurrence, a spooled turn, a pending trigger (``supervisor._due_sessions``).
A runtime that dies with none of those has nothing that will ever ask for it
again. That is not a corner: on 2026-09-30 an external multi-target SIGTERM
sweep ended **21 session runtimes** inside a minute, none was re-engaged by the
product, and a human resumed all 21 by hand (forensics bundle
``massfailure/REPORT.md``; the same gap is documented for 2026-09-15 and
2026-09-28). The residency pass next door is not a recovery mechanism — it
exists to END record-less orphans and deliberately refuses these.

**What a rescue pass does, and what it deliberately does not.** It finds
sessions whose runtime died involuntarily and re-engages them through the
supervisor's OWN start path (``engage_runtime`` with a ``WakeErrand``, which
delivers nothing), so every property of that path — record reuse, the transcript
lease, the deadline, throttled failures — is inherited rather than reinvented.
It never signals, stops, kills or drains anything: it only asks for a runtime to
exist, and the transcript lease admits at most one SERVING runtime, so a rescue
attempt against a live runtime is a no-op even if every other guard raced. The
interrupted WORK is not resumed here; the successor's boot narration already
tells the session a turn was lost (``journal.open_row_after_death`` →
``restored_interruption``), and the nudge message is the dispatch layer's.

**Why it is a SIBLING of the residency pass and not an extension of it.**
Reclaim ENDS processes and is deliberately fail-closed toward action; rescue
SPAWNS. Putting "may_end" and "may_spawn" decisions on one census would give
them one fail direction, and they need opposite ones. So this is its own module
with its own seat in the wake supervisor loop, mirroring ``reclaim.py``.

**Import-light, like ``registry`` and ``reclaim``.** The wake supervisor imports
this module lazily from its sweep seat, and the supervisor's whole justification
is staying cheap: stdlib plus ``registry``/``types``/``retention``, nothing that
pulls the harness in. ``tests/unit/test_import_graph.py`` pins that.

**The predicate is conservative and its inputs are ordered.** Liveness is read
BEFORE the completion row and independently of it: the survivors of the
2026-09-30 sweep each carried a STALE ``error/disposed`` completion record from
an earlier wave while running fine, so a predicate that keyed on the completion
row alone would have mis-fired on every one of them. See
``docs/design/reengage-rescue.md`` (session ``f8c756c83bef``) for the full design
and the calibration fixture in ``tests/unit/session/runtime/test_rescue.py``.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_operator.session.runtime import registry
from local_operator.session.runtime.types import HOST_RUN_DIRNAME, RUN_DIRNAME

logger = logging.getLogger(__name__)

# --- calibration constants --------------------------------------------------
#
# These are constants, not config keys, in v1: they are calibrated to the
# 2026-09-30 sweep and reviewed as a set. Each carries its rationale inline so a
# future reader can re-derive it rather than guess.

#: How long a death must be STABLE before it is eligible. Covers the wave's own
#: retry passes (+1.5 s / +2.1 s) and any engage machinery still constructing the
#: session's FIRST runtime — a death younger than this may be a live handover
#: mid-flight, and rescuing it would race the thing that is already fixing it.
RESCUE_SETTLE_S = 30.0

#: How old a death may be and still be rescued. Past this the session's own
#: catch-up owns it (the same reasoning as ``supervisor.STALE_AFTER_S`` for
#: wakes): a week-old death is not tonight's incident, and a fleet-wide rescue of
#: ancient corpses would be a stampede with no reader.
RESCUE_RECENT_S = 3600.0

#: A dead run younger than this is refused: a write-then-die storm (a boot that
#: failed at load) belongs to the engage machinery's own retries, not a rescue.
RESCUE_YOUNG_RUN_S = 120.0

#: A session CREATED less than this ago with no completed turn is refused: its
#: first engage is likely still constructing, and a rescue would be a second
#: spawn racing the first.
RESCUE_YOUNG_SESSION_S = 600.0

#: Attempts per RUN KEY before the episode is abandoned. Five attempts at the
#: backoff below span ~25 min, which is past the point where re-engaging is
#: productive — if five spawns failed, the sixth will too.
RESCUE_MAX_ATTEMPTS = 5

#: Backoff between attempts, in seconds: 30 s / 1 m / 2 m / 5 m / 15 m. The floor
#: is the settle window (the first 30 s are already covered before the first
#: attempt), and the ceiling matches ``wakes.deliveries.RETRY_CAP_S``.
RESCUE_BACKOFF_S: tuple[float, ...] = (30.0, 60.0, 120.0, 300.0, 900.0)

#: After a run key RESOLVES (engaged/verified/abandoned), the next death of the
#: same session starts a NEW episode but only after this cooldown. Bounds a
#: session that dies repeatedly (e.g. a build that crashes on load) to at most
#: one rescue episode per ten minutes.
RESCUE_EPISODE_COOLDOWN_S = 600.0

#: More than this many episodes inside RESCUE_BREAKER_WINDOW_S trips the session
#: breaker: a session engaging in a loop under a repeating sweeper is the failure
#: this exists to prevent, and it needs a human or a kill switch to clear.
RESCUE_BREAKER_EPISODES = 3
RESCUE_BREAKER_WINDOW_S = 3600.0

#: How many ENGAGES a single pass may start. Twenty-one candidates then take
#: several passes rather than one fan-out: after the 2026-09-30 wave the manual
#: resume storm drove load to 273, and rescue must not add a burst on top.
RESCUE_MAX_STARTS_PER_PASS = 4

#: The ledger directory, a sibling of ``wakes/`` — deliberately NOT inside it, so
#: the wake index's glob can never pick a rescue record up as a wake entry.
RESCUE_LEDGER_DIRNAME = "rescue"

#: How many resolved episodes the ledger keeps for the breaker's window. Bounded
#: because the breaker only ever asks about the last hour.
RESCUE_EPISODE_HISTORY = 16

#: Tolerance when comparing two readings of the same run key, in seconds, mirroring
#: ``attention._RUN_KEY_TOLERANCE_S``: the key is written by two processes from the
#: same clock, so this only absorbs rounding.
RUN_KEY_TOLERANCE_S = 1.0

#: Verdicts. ``FIRE`` means re-engage; ``SKIP`` means the death is explained and
#: deliberate or out of scope; ``UNCLASSIFIED`` means the death is real but the
#: evidence does not name it well enough to act (logged once, never rescued).
FIRE = "fire"
SKIP = "skip"
UNCLASSIFIED = "unclassified"


# --- the durable ledger -----------------------------------------------------


def rescue_dir(config_dir: Path) -> Path:
    """``<config>/rescue`` — the rescue ledger's directory (creates nothing)."""
    return config_dir / RESCUE_LEDGER_DIRNAME


def ledger_path(config_dir: Path, session_id: str) -> Path:
    """One flat JSON file per session, keyed by session id (not pid).

    Keyed by SESSION because the ledger must outlive the run it describes: the
    run key (``pid``/``started_at``) lives INSIDE the record, so a new run of the
    same session updates the same file and its episode history accumulates in
    one place — which is what the breaker reads.
    """
    return rescue_dir(config_dir) / f"{session_id}.json"


def read_ledger(config_dir: Path, session_id: str) -> dict[str, Any] | None:
    """The session's ledger record, or ``None``.

    Tolerant by contract: an unreadable or malformed record means "no rescue
    state", never an exception on a pass that must not fail the supervisor loop.
    """
    try:
        data = json.loads(ledger_path(config_dir, session_id).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_ledger(config_dir: Path, session_id: str, entry: dict[str, Any]) -> None:
    """Write the ledger record staged (temp + ``os.replace``), best-effort.

    The same write shape ``registry._staged_write`` uses and for the same reason
    (a reader sees the old record or the new one, never a torn file). It is a
    local copy rather than a call to that helper because the helper is private to
    ``registry`` and this module's import contract is stdlib + a few leaf modules
    (see the module docstring); the two writes are three lines of identical code
    and neither owns a rule the other could disagree about.

    Best-effort by contract with the caller: a failed write must never make a
    rescue pass raise. The consequence of losing a write is one extra attempt
    inside the episode, never a wrong decision.
    """
    directory = rescue_dir(config_dir)
    target = directory / f"{session_id}.json"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{session_id}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(entry, handle)
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError as exc:
        logger.debug("could not write rescue ledger for %s: %s", session_id, exc)


def _new_entry(
    session_id: str, run_key: dict[str, Any], death_at: float, cls: str, now: float
) -> dict[str, Any]:
    """A fresh ledger record for a new episode."""
    return {
        "schema": 1,
        "session_id": session_id,
        "run_key": run_key,
        "death_at": death_at,
        "first_seen_at": now,
        "class": cls,
        "attempts": [],
        "count": 0,
        # ARMED ON CREATION, not left null: the first attempt waits
        # ``RESCUE_BACKOFF_S[0]`` (30 s), which is the design's own reading — the
        # first backoff is not spent on the settle window, because the settle
        # window is time the runtime should not even be considered in yet.
        "next_at_ms": int((now + RESCUE_BACKOFF_S[0]) * 1000),
        "state": "pending",
        "engaged": None,
        "errand": "wake",
        # Episode HISTORY is what the breaker reads; it is carried across run
        # keys in this same file (bounded to RESCUE_EPISODE_HISTORY) because a
        # breaker that only saw the current episode could never count to >3.
        "episodes": [],
    }


def _resolve_episode(entry: dict[str, Any], outcome: str, now: float) -> None:
    """Record a resolved episode in the history the breaker reads."""
    entry["state"] = outcome
    history = entry.setdefault("episodes", [])
    history.append(
        {
            "run_key": entry.get("run_key"),
            "at": entry.get("death_at"),
            "outcome": outcome,
            "resolved_at": now,
        }
    )
    del history[:-RESCUE_EPISODE_HISTORY]


def note_rescue_attempt(
    config_dir: Path,
    session_id: str,
    *,
    outcome: str,
    detail: str = "",
    engaged_pid: int | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Record one engage ATTEMPT against the session's current run key.

    Called by the supervisor seat after an engage resolves (``verified``,
    ``started``, ``failed``) — the pass itself only ever READS the ledger and
    writes the record when it opens a new episode. ``outcome`` is one of
    ``verified`` | ``started`` | ``failed`` | ``raised`` | ``wedged``.

    ``verified`` and ``raised``/``failed`` past the attempt cap resolve the
    episode; anything else advances the attempt count and arms ``next_at_ms``
    from ``RESCUE_BACKOFF_S``. Returns the (possibly updated) entry, or ``None``
    when there was nothing to update.
    """
    moment = time.time() if now is None else now
    entry = read_ledger(config_dir, session_id)
    if entry is None:
        return None
    attempt = {"at": moment, "outcome": outcome}
    if detail:
        attempt["detail"] = detail[:400]
    attempts = entry.setdefault("attempts", [])
    attempts.append(attempt)
    entry["count"] = len(attempts)
    if outcome == "verified":
        entry["state"] = "verified"
        entry["engaged"] = {"pid": engaged_pid, "at": moment}
        _resolve_episode(entry, "verified", moment)
    elif outcome in {"failed", "raised", "wedged"}:
        if entry["count"] >= RESCUE_MAX_ATTEMPTS:
            _resolve_episode(entry, "abandoned", moment)
            entry["next_at_ms"] = None
        else:
            entry["state"] = "engaged" if engaged_pid else "pending"
            # THE WAIT AFTER ATTEMPT n IS ``RESCUE_BACKOFF_S[n]`` (0-based), not
            # ``n - 1``: the FIRST backoff (30 s) is the one armed when the
            # episode opens, so the ladder is 30 (before attempt 1), then 60,
            # 120, 300 and 900 after attempts 1-4, and the 5th attempt abandons.
            # Indexing at ``count - 1`` here would spend 30 s twice and never
            # reach the 15 min rung.
            entry["next_at_ms"] = int(
                (moment + RESCUE_BACKOFF_S[min(entry["count"], len(RESCUE_BACKOFF_S) - 1)]) * 1000
            )
    else:  # "started": a runtime was asked for but no successor record was seen yet
        entry["state"] = "engaged"
        if engaged_pid is not None:
            # ONLY WITH A PID. An ``engaged`` record naming no process is not
            # evidence of anything, and it would read as a rescued runtime to a
            # later pass or an operator reading the ledger.
            entry["engaged"] = {"pid": engaged_pid, "at": moment}
        if entry["count"] >= RESCUE_MAX_ATTEMPTS:
            _resolve_episode(entry, "abandoned", moment)
            entry["next_at_ms"] = None
        else:
            entry["next_at_ms"] = int(
                (moment + RESCUE_BACKOFF_S[min(entry["count"], len(RESCUE_BACKOFF_S) - 1)]) * 1000
            )
    write_ledger(config_dir, session_id, entry)
    return entry


# --- evidence readers (stdlib only) ----------------------------------------


def _latest_completion(config_dir: Path, session_id: str) -> dict[str, Any] | None:
    """The session's most recent ``attention.db`` completion row, or ``None``.

    Read with plain ``sqlite3`` against ``mode=ro`` rather than through
    :mod:`local_operator.session.attention`: the supervisor's rescue module must
    stay import-light (see the module docstring), and the row is a handful of
    columns. Tolerates an absent or unreadable store by returning ``None`` — the
    caller then degrades to the journal rungs and never raises.

    THE READ IS ``ORDER BY sequence DESC LIMIT 1``, i.e. the LATEST row, and that
    is load-bearing rather than incidental: two victims of the 2026-09-30 sweep
    died AGAIN after completing post-resume turns, so their newest completions
    are innocent. A pass that scanned the window for "any suspicious row" would
    rescue them forever.
    """
    path = config_dir / "attention.db"
    if not path.is_file():
        return None
    conversation = f"session/{session_id}"
    try:
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT sequence, kind, cause FROM completions WHERE conversation=? "
                "ORDER BY sequence DESC LIMIT 1",
                (conversation,),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return dict(row) if row is not None else None


def _has_completed_turn(config_dir: Path, session_id: str) -> bool:
    """Whether the session has ever completed a real turn.

    The young-session rule's second half. Read from the same store, same
    tolerance; ``kind='complete'`` is the taxonomy's planned-end token.
    """
    path = config_dir / "attention.db"
    if not path.is_file():
        return False
    try:
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=2.0)
        try:
            row = conn.execute(
                "SELECT 1 FROM completions WHERE conversation=? AND kind='complete' LIMIT 1",
                (f"session/{session_id}",),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    return row is not None


def session_dir(config_dir: Path, session_id: str) -> Path:
    """``<config>/sessions/<id>`` — the conversation directory."""
    from local_operator.session.retention import SESSIONS_DIRNAME

    return config_dir / SESSIONS_DIRNAME / session_id


def _journal_open(config_dir: Path, session_id: str) -> bool:
    """Whether the dead run left an OPEN journal row (a turn was in flight).

    ``journal.open_row_after_death`` owns the rule (a row exists, is open, and
    the pid that wrote it is gone) — this is a lazy import of it, which is fine
    here because it is off the CLI startup path and this module is only loaded
    by the supervisor's rescue seat.
    """
    from local_operator.session.runtime.journal import open_row_after_death

    try:
        return open_row_after_death(session_dir(config_dir, session_id)) is not None
    except Exception:  # noqa: BLE001 — journal evidence must never fail the pass
        return False


def read_signal_receipt(config_dir: Path, session_id: str) -> dict[str, Any] | None:
    """The stop-attribution receipt, when the lane that writes it has landed.

    FEATURE-DETECTED BY FILE PRESENCE, deliberately: the receipt
    (``sessions/<id>/runtime-signal.json``) is written by the stop-attribution
    lane, which had not merged at the time this module was written. A pass must
    therefore work both with and without it, and the tightening it enables must
    turn on by itself the moment the file starts appearing — no flag, no version
    check. Absence is the v1 fallback's permanent case for SIGKILL/crash too,
    because those write no receipt at all.
    """
    path = session_dir(config_dir, session_id) / "runtime-signal.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _receipt_covers_run(
    receipt: dict[str, Any], session_id: str, pid: int, started_at: float
) -> bool:
    """Whether a receipt attests to THIS run (session + pid + start).

    The same run-key rule ``attention._stop_marker_covers_run`` applies to a
    stop marker, and the lane's own ``signal_receipt.covers_run`` applies to the
    receipt: a receipt is refused unless it can be verified, unlike a marker
    (which is permissive when no run key survives).
    """
    if str(receipt.get("session_id") or "") != session_id:
        return False
    try:
        if int(receipt.get("pid") or -1) != int(pid):
            return False
    except (TypeError, ValueError):
        return False
    stamp = receipt.get("started_at")
    if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) and stamp:
        return abs(float(stamp) - float(started_at)) < RUN_KEY_TOLERANCE_S
    return True


def _marker_covers_run(config_dir: Path, session_id: str, pid: int, started_at: float) -> bool:
    """Whether a durable stop marker attests to THIS run.

    ``registry.read_stop_marker`` is the public reader of the file; the coverage
    RULE lives in ``attention._stop_marker_covers_run`` and is imported lazily
    rather than restated, because it is a wrong-verdict hole in the direction
    that HIDES a crash if it drifts (QA round 1's Q-1 on that function). The
    import stays inside the function to match this module's lazy-import
    convention. NAMED ``_marker_covers_run`` here rather than reusing the
    attention name, because a local function of the same name that imports its
    own namesake is a trap for the next reader.
    """
    marker = registry.read_stop_marker(session_dir(config_dir, session_id))
    if marker is None:
        return False
    from local_operator.session.attention import _stop_marker_covers_run

    return _stop_marker_covers_run(
        marker, session_dir(config_dir, session_id), _RunKey(pid, started_at)
    )


@dataclass(frozen=True)
class _RunKey:
    """The ``(pid, started_at)`` a run-key comparison reads.

    A tiny carrier so ``attention._stop_marker_covers_run`` can take an object
    with the two attributes it reads (``pid``, ``started_at``) without this
    module importing a record type it does not otherwise need.
    """

    pid: int
    started_at: float


# --- the predicate ----------------------------------------------------------


def _death_class_fallback(
    config_dir: Path, session_id: str, *, stopped_marker: bool
) -> tuple[str, str]:
    """The v1 death-class table: ``(verdict, tag)`` from the durable records.

    This is the CONSERVATIVE FALLBACK — the receipt is not assumed present (see
    :func:`read_signal_receipt`). ``stopped_marker`` is the wake ``stopped_at``
    corroboration, computed by the caller so this stays a pure decision.

    THE ``interrupted``/``user-stop`` RUNG IS THE INTERESTING ONE. ``user-stop``
    is written by exactly two sanctioned paths, both of which stage evidence
    before signalling (a ``stopped_at`` wake stamp, or a covering
    ``runtime-stop.json``). The 2026-09-30 sweep's ten ``user-stop`` victims had
    NEITHER — no marker file, no wake entry — so a stop with no staged evidence
    is treated as involuntary here. That is a deliberate err toward FIRE: the
    worst case of a false positive is one wake and one boot notice bounded by
    five attempts and the episode cooldown, while the worst case of a false
    negative is a silent, unrecovered death.
    """
    row = _latest_completion(config_dir, session_id)
    if row is None:
        if _journal_open(config_dir, session_id):
            return FIRE, "no-completion-journal-open"
        return UNCLASSIFIED, "no-completion-no-journal"
    kind = str(row.get("kind") or "")
    cause = str(row.get("cause") or "")
    if kind == "error":
        if cause == "disposed":
            return FIRE, "disposed"
        if cause == "runtime-shutdown":
            return FIRE, "runtime-shutdown"
        if cause == "runtime-killed":
            return FIRE, "runtime-killed"
        if _journal_open(config_dir, session_id):
            return FIRE, "error-journal-open"
        return UNCLASSIFIED, f"error-unattributed{(':' + cause) if cause else ''}"
    if kind == "interrupted":
        if cause == "user-stop":
            if stopped_marker:
                return SKIP, "user-stop-stopped-at"
            return FIRE, "user-stop-no-corroboration"
        # Any other interrupted cause is a stop sweep that reached its target,
        # which the taxonomy already names; it is not a rescue.
        return SKIP, f"interrupted{(':' + cause) if cause else ''}"
    if kind == "complete":
        return SKIP, "complete"
    return UNCLASSIFIED, f"unknown-kind:{kind}"


def _death_class_receipt(receipt: dict[str, Any]) -> tuple[str, str]:
    """The tightened (§1.4) verdict from a covering receipt: ``(verdict, tag)``.

    Applied only when the receipt covers this exact run. The last signal is the
    one that ended it.
    """
    signals = receipt.get("signals")
    if not isinstance(signals, list) or not signals:
        # A receipt with no readable signal is NOT evidence of an unsanctioned
        # stop, so it does not fire. This differs from the stop-attribution
        # lane's ``covers_run`` on purpose: that predicate answers "does this
        # receipt describe THIS run" (an identity question, which an empty
        # signal list does not affect), while this one answers "what did the
        # signals say", and an empty list says nothing. The fallback rungs then
        # decide, exactly as they would for a receipt-less death.
        return UNCLASSIFIED, "receipt-no-signals"
    last = signals[-1] if isinstance(signals[-1], dict) else {}
    sanction = str(last.get("sanction") or "")
    if sanction == "none":
        # The affirmative statement the fallback has to guess at: a termination
        # signal reached this runtime with nothing sanctioning it.
        return FIRE, "receipt-unsanctioned"
    if sanction == "marker":
        # Bound ONCE to a definite dict: pyright's narrowing of
        # ``last.get("stop_marker")`` does not survive a second call to ``get``,
        # so the conditional expression that used to inline both reads reported
        # ``"get" is not a known attribute of "None"`` on the second one.
        raw_marker = last.get("stop_marker")
        stop_marker: dict[str, Any] = raw_marker if isinstance(raw_marker, dict) else {}
        deliberate = stop_marker.get("deliberate")
        if deliberate is True:
            return SKIP, "receipt-deliberate"
        if deliberate is False:
            mechanism = str(stop_marker.get("mechanism") or "")
            in_flight = bool(last.get("in_flight"))
            if mechanism == "reclaim" and not in_flight:
                # An idle reclaim of an orphan is the reclaim pass's own job.
                return SKIP, "receipt-reclaim-idle"
            if mechanism == "install" and in_flight:
                # An install tore a turn mid-flight: rescue on the new build.
                return FIRE, "receipt-install-in-flight"
            return SKIP, f"receipt-marker-other:{mechanism or 'unknown'}"
        return SKIP, "receipt-marker-unspecified"
    return UNCLASSIFIED, f"receipt-sanction:{sanction or 'unknown'}"


# --- the pass ---------------------------------------------------------------


@dataclass
class RescueDecision:
    """One session's evaluation in a pass."""

    session_id: str
    verdict: str
    tag: str
    pid: int = 0
    started_at: float = 0.0
    cwd: str = ""
    death_at_ms: int = 0


@dataclass
class RescueReport:
    """What one rescue pass decided, in the shape the supervisor logs.

    Mirrors ``reclaim.ReclaimReport``: the pass summary prints the same way, and
    the refusals are a ``Counter`` so a steady-state "skipped: live 30" collapses
    to one throttled line rather than thirty.
    """

    decisions: list[RescueDecision] = field(default_factory=list)
    to_engage: list[RescueDecision] = field(default_factory=list)
    refusals: Counter[str] = field(default_factory=Counter)
    verified: list[str] = field(default_factory=list)
    abandoned: list[str] = field(default_factory=list)

    def fire(self) -> list[RescueDecision]:
        return [d for d in self.decisions if d.verdict == FIRE]

    def summary(self) -> str:
        """One line for ``wake-supervisor.log``, in the residency line's style.

        The refusals are appended only when there are any: an empty ``Counter``
        would otherwise render a dangling ``refused `` and read as a truncated
        line rather than a clean pass. (The ``or`` that used to sit here was
        dead code — the concatenation is never the empty string.)
        """
        refusals = ", ".join(f"{reason} {count}" for reason, count in sorted(self.refusals.items()))
        line = (
            f"rescue: {len(self.decisions)} dead session(s) considered, "
            f"{len(self.to_engage)} to engage, {len(self.verified)} verified, "
            f"{len(self.abandoned)} abandoned"
        )
        return f"{line}; refused {refusals}" if refusals else line


def _young_session(
    config_dir: Path, session_id: str, now: float, *, created_at: float | None
) -> bool:
    """Whether the session is too new to rescue (see ``RESCUE_YOUNG_SESSION_S``).

    ``created_at`` is the EARLIEST runtime start the session's own discovery
    records carry — deliberately NOT the conversation directory's
    ``st_birthtime``. Two reasons: the records are the product's own statement
    of when this session's runtimes began, while a directory's birth time says
    only when the directory was made; and a birth time cannot be reconstructed
    from the store OR set by a fixture, so a rule keyed on it is untestable and
    would mis-fire the moment a store is restored, moved or re-created (every
    session would read as newborn). The OLDEST record wins because the question
    is "could a spawn for this session still be constructing", which is about the
    session's whole life, not its latest run.
    """
    if created_at is None:
        return False
    if now - created_at >= RESCUE_YOUNG_SESSION_S:
        return False
    return not _has_completed_turn(config_dir, session_id)


def _viewer_attached(config_dir: Path, session_id: str) -> bool:
    """Whether a live viewer currently has this session on screen.

    A viewer-attached session recovers viewer-side (``tui/app.py``: "the runtime
    retired ITSELF … Re-engage at once"), so v1 skips it to avoid two spawns
    racing on every attach. ``scan_viewers`` returns heartbeat-fresh records
    only, and ``has_window`` is required because a closed window reports the last
    session it showed without displaying anything.
    """
    from local_operator.session.runtime.viewers import scan_viewers

    try:
        for viewer in scan_viewers(config_dir, reap=False):
            if viewer.current_session == session_id and viewer.has_window:
                return True
    except Exception:  # noqa: BLE001 — an unreadable viewer table is not an attach
        return False
    return False


def _bound_state(
    config_dir: Path,
    session_id: str,
    run_key: dict[str, Any],
    now: float,
) -> tuple[dict[str, Any] | None, str]:
    """Apply the ledger bounds: ``(entry_to_use, refusal_tag)``.

    Returns ``(entry, "")`` when the run may be attempted (the entry is the
    session's record, opened fresh when this is a new run key), or
    ``(entry, tag)`` when a bound refuses it. The tags are the reasons the pass
    summarises.
    """
    entry = read_ledger(config_dir, session_id)
    if entry is not None and entry.get("run_key") == run_key:
        state = entry.get("state")
        if state == "verified":
            return entry, "already-verified"
        if state == "abandoned":
            return entry, "abandoned"
        if entry.get("count", 0) >= RESCUE_MAX_ATTEMPTS:
            return entry, "attempts-exhausted"
        next_at = entry.get("next_at_ms")
        if isinstance(next_at, int) and now * 1000.0 < next_at:
            return entry, "backing-off"
        return entry, ""
    # A NEW run key: this is a new episode of the same session. The cooldown and
    # the breaker are what stop a repeating sweeper from engaging in a loop.
    if entry is not None:
        episodes = [e for e in entry.get("episodes", []) if isinstance(e, dict)]
        recent = [
            e
            for e in episodes
            if now - float(e.get("resolved_at") or 0.0) <= RESCUE_BREAKER_WINDOW_S
        ]
        if len(recent) > RESCUE_BREAKER_EPISODES:
            return entry, "breaker"
        if recent:
            last = max(float(e.get("resolved_at") or 0.0) for e in recent)
            if now - last < RESCUE_EPISODE_COOLDOWN_S:
                return entry, "cooldown"
    return entry, ""


def _death_records(records: list[tuple[Any, str]]) -> list[Any]:
    return [record for record, state in records if state == "stale"]


def _parse_session_record(data: Any) -> Any:
    """The ``run/mobile`` namespace's record, which IS a ``SessionRecord``."""
    from local_operator.session.runtime.types import SessionRecord

    return SessionRecord.from_json(data)


def _parse_boot_record(data: Any) -> Any:
    """The ``run/host`` namespace's record (see the census's own comment).

    Imported lazily because this module stays stdlib + registry-level, and RAISES
    rather than returning ``None`` when the payload is not a boot record: a
    ``None`` would reach ``registry.classify`` as a record and crash the pass,
    while a raise is exactly what ``registry.scan``'s per-entry rescue expects.
    """
    from local_operator.session.runtime.journal import BootRecord

    record = BootRecord.from_json(data)
    if record is None:
        raise ValueError("not a boot record")
    return record


def rescue_scan(config_dir: Path, *, now: float | None = None, apply: bool = True) -> RescueReport:
    """One rescue pass over the store. Blocking; callers hand it to a worker thread.

    ``apply=False`` makes the pass a pure DECISION — no ledger writes, no
    engagement — which is what the calibration test drives to assert that the
    candidate set is exactly the 2026-09-30 victims and none of the survivors.

    The one coupling that is not a file read is the ``attention.db`` read; it
    degrades to the journal rungs when the store is absent or unreadable and
    never raises (see the module docstring).
    """
    moment = time.time() if now is None else now
    report = RescueReport()

    # The census: one scan per namespace, READ-ONLY (`reap=False`) because a
    # sweep that reaped evidence while deciding would destroy the answer to "why
    # did this die" — the same rule reclaim.py states for its own census.
    by_session: dict[str, dict[str, Any]] = {}
    for dirname in (RUN_DIRNAME, HOST_RUN_DIRNAME):
        # THE PARSE IS PER-NAMESPACE. ``run/host`` holds ``journal.BootRecord``
        # (pid/session_id/cwd/started_at/heartbeat_at and nothing else), so
        # parsing it with ``SessionRecord.from_json`` raises for the four fields
        # that type requires — and ``registry.scan`` drops an entry that will not
        # parse, per entry, by design. Measured consequence: the ENTIRE
        # ``run/host`` namespace was invisible to the census, so a death whose
        # only record was its boot record yielded no decision at all, while the
        # same session under ``run/mobile`` fired. A host-kind death is exactly
        # the shape a spawn that died at load leaves behind.
        parse = _parse_session_record if dirname == RUN_DIRNAME else _parse_boot_record
        for record, state in registry.scan(config_dir, dirname, parse=parse, reap=False):
            sid = getattr(record, "session_id", "") or ""
            if not sid:
                continue
            slot = by_session.setdefault(
                sid, {"live": False, "wedged": False, "dead": [], "earliest": None}
            )
            started = float(getattr(record, "started_at", 0.0) or 0.0)
            if started and (slot["earliest"] is None or started < slot["earliest"]):
                # The session's own record of when its first runtime began — the
                # young-session rule's ``created_at`` (see ``_young_session``).
                slot["earliest"] = started
            if dirname == RUN_DIRNAME:
                # Liveness is run/mobile's question: a boot record in run/host is
                # deliberately NOT a liveness record (see journal.BootRecord).
                if state == "live":
                    slot["live"] = True
                elif state == "wedged":
                    slot["wedged"] = True
            if state == "stale":
                # Stored WITH its state so ``_death_records`` is the one place
                # that decides what counts as a death (and a second namespace or
                # state can be added without a second filter).
                slot["dead"].append((record, state))

    for sid in sorted(by_session):
        slot = by_session[sid]
        entry = read_ledger(config_dir, sid)
        if slot["live"]:
            # Resolve an open episode to verified: the session is live again, and
            # whether rescue or the user's own resume put it there, the episode
            # is over. (Done before the skip so the ledger cannot stay "engaged"
            # forever against a session that recovered.)
            if apply and entry is not None and entry.get("state") in {"pending", "engaged"}:
                note_rescue_attempt(config_dir, sid, outcome="verified", now=moment)
                report.verified.append(sid)
            report.refusals["live"] += 1
            report.decisions.append(RescueDecision(sid, SKIP, "live"))
            continue
        if slot["wedged"]:
            # A wedged owner is not answering, which is not a verdict that it is
            # dead; never a rescue (see registry.classify).
            report.refusals["wedged"] += 1
            report.decisions.append(RescueDecision(sid, SKIP, "wedged"))
            continue
        dead = _death_records(slot["dead"])
        if not dead:
            continue
        record = max(dead, key=lambda r: float(getattr(r, "started_at", 0.0) or 0.0))
        pid = int(getattr(record, "pid", 0) or 0)
        started_at = float(getattr(record, "started_at", 0.0) or 0.0)
        cwd = str(getattr(record, "cwd", "") or "")
        heartbeat_at = float(getattr(record, "heartbeat_at", 0.0) or 0.0)
        death_at_ms = int(heartbeat_at * 1000)

        def refuse(tag: str) -> None:
            report.refusals[tag] += 1
            report.decisions.append(
                RescueDecision(
                    sid, SKIP, tag, pid=pid, started_at=started_at, cwd=cwd, death_at_ms=death_at_ms
                )
            )

        if not session_dir(config_dir, sid).is_dir():
            refuse("ghost")
            continue
        # The stop levers, checked every pass: a covering marker or a wake
        # `stopped_at`/`held_at` is the user's own act and outranks everything.
        if _marker_covers_run(config_dir, sid, pid, started_at):
            refuse("stop-marker")
            continue
        try:
            from local_operator.wakes.store import is_held, read_entry

            if is_held(read_entry(config_dir, sid)):
                refuse("held")
                continue
        except Exception:  # noqa: BLE001 — an unreadable index is not a hold
            pass
        age = moment - heartbeat_at
        if age < RESCUE_SETTLE_S:
            refuse("young-run-settle")
            continue
        if age > RESCUE_RECENT_S:
            refuse("stale")
            continue
        if moment - started_at < RESCUE_YOUNG_RUN_S:
            refuse("young-run")
            continue
        if _young_session(config_dir, sid, moment, created_at=slot["earliest"]):
            refuse("young-session")
            continue
        if _viewer_attached(config_dir, sid):
            refuse("attached")
            continue

        receipt = read_signal_receipt(config_dir, sid)
        if receipt is not None and _receipt_covers_run(receipt, sid, pid, started_at):
            verdict, tag = _death_class_receipt(receipt)
        else:
            stopped_marker = False
            try:
                from local_operator.wakes.store import read_entry

                wake_entry = read_entry(config_dir, sid)
                stopped_marker = bool(isinstance(wake_entry, dict) and wake_entry.get("stopped_at"))
            except Exception:  # noqa: BLE001
                stopped_marker = False
            verdict, tag = _death_class_fallback(config_dir, sid, stopped_marker=stopped_marker)

        decision = RescueDecision(
            sid, verdict, tag, pid=pid, started_at=started_at, cwd=cwd, death_at_ms=death_at_ms
        )
        report.decisions.append(decision)
        if verdict != FIRE:
            report.refusals[tag] += 1
            continue

        run_key = {"pid": pid, "started_at": started_at}
        entry, refusal = _bound_state(config_dir, sid, run_key, moment)
        if refusal:
            report.refusals[refusal] += 1
            continue
        if len(report.to_engage) >= RESCUE_MAX_STARTS_PER_PASS:
            report.refusals["pass-budget"] += 1
            continue
        if apply and (entry is None or entry.get("run_key") != run_key):
            # Open the episode. The attempt itself is recorded by the seat once
            # the engage resolves (it knows the outcome); here we only establish
            # which run key the episode is about.
            fresh = _new_entry(sid, run_key, heartbeat_at, tag, moment)
            # THE EPISODE HISTORY IS CARRIED ACROSS RUN KEYS. A new run key means
            # a new episode, but the session breaker counts episodes, so wiping
            # the list here would reset the only input it reads and the breaker
            # could never trip (measured: the seat test caught exactly this).
            if entry is not None:
                fresh["episodes"] = list(entry.get("episodes") or [])
            write_ledger(config_dir, sid, fresh)
        report.to_engage.append(decision)

    if not report.decisions:
        logger.debug("rescue: no dead session runtimes to consider")
    return report


__all__ = [
    "RESCUE_BACKOFF_S",
    "RESCUE_BREAKER_EPISODES",
    "RESCUE_BREAKER_WINDOW_S",
    "RESCUE_EPISODE_COOLDOWN_S",
    "RESCUE_MAX_ATTEMPTS",
    "RESCUE_MAX_STARTS_PER_PASS",
    "RESCUE_RECENT_S",
    "RESCUE_SETTLE_S",
    "RESCUE_YOUNG_RUN_S",
    "RESCUE_YOUNG_SESSION_S",
    "RescueDecision",
    "RescueReport",
    "ledger_path",
    "note_rescue_attempt",
    "read_ledger",
    "read_signal_receipt",
    "rescue_dir",
    "rescue_scan",
    "session_dir",
    "write_ledger",
]

"""Machine-wide desktop delivery presence: is a desktop app reachable at all?

A session record (:mod:`local_operator.session.runtime.registry`) says where a
transcript lives; a viewer record (:mod:`local_operator.session.runtime.viewers`)
says which process can put a session ON SCREEN. This module answers a third
question, deliberately asked at the MACHINE rather than at a session:

**"Can a desktop app raise a banner for this backend, right now — and if so,
which conversation is its window really showing?"**

WHY IT IS NOT A PER-SESSION LEASE. Every signal that existed before this module
was scoped to one session: ``DesktopSubscription.visible``,
``RuntimeServer._desktop_lease_live``, ``notification_surfaces()``. A background
session holds none of them — the desktop has no way to lease a session it is not
displaying — so a completion in session B raised a **backend** OS toast even
while the app was up and focused on session A. That is the operator's report:
"the app is right there and nothing reaches it."

WHY A FILE, AND NOT A CALL TO THE SESSION RUNTIME. The app may be paired to a
backend on another host (an ssh forward, a tunnel). It cannot write to that
host's filesystem, so the *client* cannot own this file — the HTTP server does,
aggregating what its live feed subscriptions report
(``server/utils/desktop_presence.py``), and every sibling process on the host
reads it here. A purely local file written by the app would make a remote app
invisible to the runtime, and a per-session lease would make a *background*
completion invisible to it. The server-side route is what makes local and remote
apps behave identically.

THE TWO REAPING RULES ARE ``scan_viewers``' RULES, deliberately. A record is
believed only while its ``pid`` is alive and its ``heartbeat_at`` is inside
:data:`PRESENCE_TTL_S` — the same two tests and the same 45 s timeout a viewer
record is held to, so an operator debugging one has already learned the other.

STDLIB-ONLY AND IMPORT-LIGHT, by the same contract as ``viewers`` and
``registry``. The runtime reads this on the gate/announce path and the HTTP
server reads it when it decides whether the client may claim delivery presence,
so neither may pull a server stack into a session process to answer one stat.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_operator.paths import config_dir
from local_operator.session.runtime.registry import pid_alive

#: Deliberately NOT ``run/viewers``. A viewer record is PROCESS-scoped and
#: answers routing; this one is BACKEND-scoped and answers reachability. They
#: are written by different processes (Electron main vs the HTTP server) and
#: consumed by different readers, and folding them together would make the
#: lifetime of a routing answer depend on a delivery lease.
DESKTOP_RUN_DIRNAME = "run/desktop"

#: The aggregate one backend publishes for every process on the host to read.
#:
#: LEGACY, AND READ-ONLY SINCE REVIEW ROUND 1 (R6). It is the pre-R6 single-file
#: layout: ONE path that every serve process on the machine wrote and unlinked,
#: so two live backends clobbered each other in both directions — the last
#: WRITER decided the machine's answer (a second server advertising
#: ``can_notify=False`` revoked a live first server's lease on every beat) and
#: the first one to EXIT deleted the file out from under the other. Publishers
#: now own a record each (see :func:`delivery_record_path`) and readers aggregate
#: them. This name survives because a sibling started before the change still
#: writes it, and a reader that stopped looking at it would go blind to that
#: sibling for the life of its process.
DELIVERY_FILENAME = "delivery.json"

#: The directory of PER-INSTANCE records, beside the legacy file. 0700 like its
#: parent, and the reason the split is safe: one file per publishing process, so
#: a publisher can only ever withdraw its own.
DELIVERY_DIRNAME = "delivery"

#: How long a lease lives with no beat behind it. Matches ``WATCH_TTL``,
#: ``DESKTOP_WATCH_LEASE_S`` and ``VIEWER_HEARTBEAT_TIMEOUT_S``: "is this
#: alive" reads the same way in all four places, and one timeout is one thing
#: to remember. The app beats every :data:`PRESENCE_BEAT_S`, so three beats
#: must be missed before anything is believed dead.
PRESENCE_TTL_S = 45.0

#: The app's beat against the TTL above.
PRESENCE_BEAT_S = 15.0

#: How long a reader may reuse one answer.
#:
#: The reader runs on the runtime's ANNOUNCE path — synchronously, on the
#: session process's event loop (``_announce_pending``) — and a directory read
#: there is the shape that produced the sidebar stalls
#: (``TUI_BACKGROUND_RESPONSIVENESS.md``). One stat and one small read is
#: comparable to ``note_session``'s measured 0.808 ms, but the point is that it
#: must not be paid once per eligible completion on a loop that also serves
#: turns. Two seconds bounds how stale a *revocation* can be, which is the only
#: direction that matters: a lease that has just gone away may still suppress
#: one banner, and the durable unseen mark means nothing is lost by it.
PRESENCE_CACHE_TTL_S = 2.0

#: ADR 0008 (desk presence) §3.5: the operator's "send to phone when I'm away
#: from my computer for N" choices, in seconds. An ALLOW-LIST rather than a
#: range because every surface that edits it (desktop Settings, the phone's
#: per-computer row, ``/settings``) renders it as four fixed options, and a
#: receiver must never compute a deadline from a value no surface can show.
#: "Immediately, even at the computer" (0) is left out on purpose: the operator
#: did not ask for it, and adding it later is one more entry here.
AWAY_AFTER_CHOICES_S: tuple[int, ...] = (60, 120, 300, 600)

#: The default threshold ("2 min", ADR 0008 §3.5). Also the value a desk
#: report that omits ``away_after_s`` is evaluated with, so a partial report
#: never invents a longer hold than the operator could have chosen by default.
DEFAULT_AWAY_AFTER_S = 120

#: How long after a LOCK the desk still counts as active (ADR 0008 §3.2).
#:
#: Zero on purpose: locking is a deliberate "I'm leaving", and the operator's
#: own example is walking away with the phone. The cost is a possible duplicate
#: on a quick lock/unlock (risk R5); moving to Slack's 60 s is this one line.
LOCK_GRACE_S = 0.0


@dataclass(frozen=True)
class DeskReport:
    """One desktop app's "is a person at this computer" report, materialised.

    ADR 0008 §3.2. The app sends RAW, RELATIVE facts (``idle_s``, ``locked``,
    ``idle_trusted``, ``away_after_s``) and the RECEIVER turns them into absolute
    deadlines on its own clock at receipt. That is what removes clock skew: the
    sender's wall clock never crosses the wire, so a Mac whose clock is ten
    minutes off produces exactly the same ``away_at`` as one that is right
    (risk R6).

    WALL CLOCK, NOT MONOTONIC. Every field here is ``time.time()`` epoch
    seconds. The HTTP server stamps it and the push DAEMON — another process —
    compares it with its own ``time.time``. ``time.monotonic`` means nothing in
    another process, and on macOS it stops during sleep, which is exactly when
    ``away_at`` must keep running. ``PresenceClaim.seen_at`` is monotonic because
    it never leaves the server process; copying that pattern here would be a bug.

    PRIVACY. These values are a person's minute-by-minute activity, so nothing
    in core writes them to a log line at any level (ADR 0008 §3.1). A decision
    may log ``desk_active=<bool>``, never the reading behind it.
    """

    received_at: float
    last_input_at: float
    away_at: float
    #: False when the app could not vouch for its idle reading (Wayland, R2).
    #: An untrusted report is "unknown", which leans towards notifying, and it
    #: never contributes to the any-aggregation: a Wayland ``idle_s = 0`` must
    #: not keep the desk "active" beside a trusted report that has gone idle.
    idle_trusted: bool = False
    locked: bool = False

    def fresh(self, now: float) -> bool:
        """Received within one presence TTL of ``now`` (three missed beats)."""
        return now - self.received_at <= PRESENCE_TTL_S

    def known(self, now: float) -> bool:
        """Fresh AND trusted: the only reports a decision may read."""
        return self.fresh(now) and self.idle_trusted

    def active(self, now: float) -> bool:
        """ADR 0008 §3.2's predicate for ONE report. Strict ``<`` on ``away_at``."""
        return self.known(now) and not self.locked and now < self.away_at

    def as_record(self) -> dict[str, Any]:
        """The JSON shape the publisher writes and :meth:`from_record` reads."""
        return {
            "received_at": self.received_at,
            "last_input_at": self.last_input_at,
            "away_at": self.away_at,
            "idle_trusted": self.idle_trusted,
            "locked": self.locked,
        }

    @classmethod
    def from_record(cls, raw: Any) -> "DeskReport | None":
        """Parse one published entry, or ``None`` when it is not one.

        Strict about types for the reason the reader is strict everywhere: a
        torn or foreign value must read as "no report" (which notifies), never
        as a deadline that holds a push.
        """
        if not isinstance(raw, dict):
            return None
        try:
            received_at = float(raw["received_at"])
            last_input_at = float(raw["last_input_at"])
            away_at = float(raw["away_at"])
        except (KeyError, TypeError, ValueError):
            return None
        return cls(
            received_at=received_at,
            last_input_at=last_input_at,
            away_at=away_at,
            idle_trusted=raw.get("idle_trusted") is True,
            locked=raw.get("locked") is True,
        )


def configured_away_after_s(values: Any) -> int:
    """``notifications.phone_away_after_s`` from a config mapping, validated.

    The consumer behind the ``/settings`` row. Anything that is not one of
    :data:`AWAY_AFTER_CHOICES_S` (absent, a hand-edited 90, a string, a bool)
    answers :data:`DEFAULT_AWAY_AFTER_S`: a deadline must only ever come from a
    value a settings surface can show.
    """
    notifications = values.get("notifications") if isinstance(values, dict) else None
    raw = notifications.get("phone_away_after_s") if isinstance(notifications, dict) else None
    if isinstance(raw, int) and not isinstance(raw, bool) and raw in AWAY_AFTER_CHOICES_S:
        return raw
    return DEFAULT_AWAY_AFTER_S


def materialise_desk_report(
    *,
    received_at: float,
    idle_s: int | None,
    locked: bool | None,
    idle_trusted: bool | None,
    away_after_s: int | None,
) -> DeskReport:
    """Turn one beat's relative desk fields into receiver-clock deadlines.

    ADR 0008 §3.2, verbatim::

        last_input_at = t_r - idle_s
        away_at       = t_r if locked else last_input_at + away_after_s

    MISSING FIELDS LEAN TOWARDS NOTIFYING. A beat that carries some desk fields
    but not ``idle_s`` or ``idle_trusted`` is untrusted (unknown → push now);
    a missing ``away_after_s`` is the default rather than the longest choice; a
    missing ``locked`` is unlocked, which is safe because the deadline still
    runs from the (trusted) idle reading.
    """
    trusted = bool(idle_trusted) and idle_s is not None
    idle = max(0, int(idle_s or 0))
    threshold = away_after_s if away_after_s in AWAY_AFTER_CHOICES_S else DEFAULT_AWAY_AFTER_S
    last_input_at = received_at - idle
    away_at = received_at + LOCK_GRACE_S if locked else last_input_at + threshold
    return DeskReport(
        received_at=received_at,
        last_input_at=last_input_at,
        away_at=away_at,
        idle_trusted=trusted,
        locked=bool(locked),
    )


@dataclass(frozen=True)
class DesktopPresence:
    """What ``run/desktop/delivery.json`` means, once reaped.

    ``deliverable`` is the only field rung 2 reads, and it is deliberately the
    narrow question: "can a banner actually be raised for this backend right
    now?" It is false for an app that is connected but cannot notify (headless,
    ``Notification.isSupported()`` false, or no live feed subscription), which is
    exactly the standard the per-session watch lease is already held to.

    ``attended`` is the WINDOW STATE, and it is what rung 1 needs.
    ``can_notify`` is reachability, not attention: an app whose window is
    behind another app can raise a banner perfectly well and is not, in any
    sense a routing decision cares about, *watching* the session it has open.
    Deriving "a human is reading X" from reachability is the bug this field
    exists to prevent — it suppresses every OS surface for a conversation
    nobody is looking at.

    ``session_id`` names the conversation that window is really showing, or
    ``""`` when it is showing none (a catalogue view, a booting window). It is
    what makes the difference between "the completion's card is already on
    screen" and "the user is looking at A while B finished".
    """

    deliverable: bool = False
    #: True when a LIVE, FRESH lease was read at all — whatever it asserts.
    #:
    #: Distinguished from :attr:`deliverable` deliberately, because the two ask
    #: different questions and rung 1 needs the weaker one. "There is a desktop
    #: app here and here is its window state" is an answer a reader may act on
    #: even when the app cannot notify (a headless run, a denied permission);
    #: "no lease at all" is the old UI's case, where the per-session lease flag
    #: written by the renderer remains the only signal and must keep working
    #: verbatim.
    present: bool = False
    #: The notification KINDS this app claims it can deliver, from
    #: ``can_notify``'s kinds counterpart.
    #:
    #: THE PRESENCE IS NARROWED BY KIND ON PURPOSE. The feed carries completions
    #: only (``BRIDGE_NOTIFIABLE_KINDS``); gate cards ride the per-session bridge
    #: and keep the per-session lease. A machine-wide "an app can notify"
    #: predicate applied to a gate would silence a background session's parked
    #: question with nothing to replace it, because no machine-wide channel
    #: carries gate frames — a regression against today. So a reader asks for the
    #: KIND it is about to route, and the gate path never asks this at all.
    kinds: frozenset[str] = frozenset()
    session_id: str = ""
    attended: bool = False
    has_window: bool = False
    pid: int = 0
    heartbeat_at: float = 0.0
    #: ADR 0008: every live app's desk report, one per subscription that sent
    #: the desk fields. EMPTY means no reporter speaks the desk protocol (an
    #: app older than ``features.desk_presence``), which is what selects the
    #: one-release legacy focus fallback in the push worker.
    desk: tuple[DeskReport, ...] = ()
    #: Kept for diagnostics and for the tests that assert a reaped record is
    #: reported absent rather than reported empty. Never read by a decision.
    detail: str = field(default="")

    def delivers(self, kind: str | None = None) -> bool:
        """Whether this app claims it can deliver ``kind`` (any kind if None).

        ``can_notify`` here means "can ATTEMPT delivery", not "the user will
        see it": ``Notification.isSupported()`` knows nothing about macOS
        Focus/DND, Windows Focus Assist, or a permission the user denied. A
        claim never advances the read watermark, so a suppressed banner costs a
        banner — never the durable ``unseen`` mark, which is what the sidebar
        and the next open surface read.
        """
        if not self.deliverable:
            return False
        return bool(self.kinds) if kind is None else kind in self.kinds

    def attending(self, session_id: str) -> bool:
        """Whether an attended window is showing ``session_id`` (rung 1).

        THE ONE PREDICATE for "the desktop is displaying this conversation to
        someone looking at it". Two consumers must answer it identically, or
        the phone can push what the desktop deliberately kept quiet: the
        desktop feed's banner gate (``desktop_feed._focus_policy_for``, which
        raises nothing for exactly this case) and the mobile push worker's
        presence gate (``push_worker._settle_deferred``). Operator, 2026-11-15:
        "I feel like I'm still seeing more notifications on mobile than I see
        pop up on desktop" — phone <= desktop is only provable if both read
        this one method rather than two copies of its expression.
        """
        return self.attended and bool(session_id) and self.session_id == session_id

    @property
    def desk_reported(self) -> bool:
        """Whether any live app sent desk fields at all (new protocol in use).

        False selects the legacy focus rule for ONE release (ADR 0008 §3.3,
        version skew row "old app, new core"). After that release a report with
        no desk fields is to be read as unknown, i.e. notify.
        """
        return bool(self.desk)

    def desk_active(self, now: float) -> bool:
        """ADR 0008 §3.2: is a person at ANY desk this backend serves?

        ANY-aggregation, each report judged on its own: an untrusted or stale
        report never contributes, so it cannot win a max and keep the desk
        "active" beside a trusted report that has gone idle (review m3).
        """
        return any(report.active(now) for report in self.desk)

    def desk_known(self, now: float) -> bool:
        """Whether at least one fresh, trusted report exists (else: unknown)."""
        return any(report.known(now) for report in self.desk)

    def last_input_at(self, now: float) -> float | None:
        """Newest input time over fresh TRUSTED reports only, or ``None``.

        Rule 4's "input after the banner" reads this. Untrusted reports are
        excluded for the same reason as in :meth:`desk_active`: a Wayland
        ``idle_s = 0`` is not evidence anybody touched anything.
        """
        known = [report.last_input_at for report in self.desk if report.known(now)]
        return max(known) if known else None


#: The absent answer, shared so callers can compare against one object.
NO_DESKTOP_PRESENCE = DesktopPresence()

#: Per-process cache: ``{root: (expires_at_monotonic, DesktopPresence)}``.
#:
#: KEYED BY ROOT, EXPIRED ONLY BY TIME. There is no invalidation key on
#: purpose — an invalidation input would have to be read to be checked, which
#: is the read this cache exists to avoid. The root is identity rather than an
#: invalidation: one process can serve more than one config root (the e2e
#: suite does exactly that), and a TTL-only single slot would hand a test's
#: freshly written lease back to the wrong root for up to two seconds.
_CACHE: dict[str, tuple[float, DesktopPresence]] = {}


def desktop_run_dir(root: Path | None = None) -> Path:
    """The delivery-lease directory AS A PATH — creates NOTHING; see :func:`ensure_desktop_run_dir`.

    The directory permissions ARE the authorization story, and they are the reason
    the whole mechanism is safe to make machine-wide: the file carries a routing
    answer, never a credential, and only the owning account can read the directory
    it lives in.

    Creating is the WRITER's business, and it is a separate function now because
    "a reader reaches it too and a missing directory is an ordinary answer" — the
    reasoning this docstring used to carry — is true about the ANSWER and wrong
    about the WRITE. A reader that materialises the namespace it reads is the
    shape ruled on in ``server/utils/desktop_feed.py`` (review round 1, MAJOR 1 /
    QA Q2) and again for the network and run planes on #1666 (review round 1,
    R1-1), so this module follows the same split rather than repeating the
    reasoning a third time.
    """
    return (root or config_dir()) / DESKTOP_RUN_DIRNAME


def ensure_desktop_run_dir(root: Path | None = None) -> Path:
    """``desktop_run_dir``, created 0700 on first use — the WRITERS' spelling."""
    path = desktop_run_dir(root)
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def delivery_path(root: Path | None = None) -> Path:
    """The LEGACY aggregate lease path. Read-only; see :data:`DELIVERY_FILENAME`."""
    return desktop_run_dir(root) / DELIVERY_FILENAME


def delivery_dir(root: Path | None = None) -> Path:
    """The directory holding one record per publishing backend (R6), as a PATH.

    ``glob`` over an absent directory answers "no publisher has ever run here",
    which is the truthful answer and the reason nothing needs creating here.
    """
    return desktop_run_dir(root) / DELIVERY_DIRNAME


def ensure_delivery_dir(root: Path | None = None) -> Path:
    """``delivery_dir``, created 0700 down the chain — the WRITERS' spelling."""
    ensure_desktop_run_dir(root)
    path = delivery_dir(root)
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def delivery_record_path(instance_id: str, root: Path | None = None) -> Path:
    """The record ONE publishing backend owns, and may withdraw (R6).

    The instance id is generated per :class:`DesktopDeliveryPublisher` and is
    never derived from anything a caller passes, so this path can only ever name
    a file in ``run/desktop/delivery/`` — the same containment the tests in
    ``tests/unit/session/test_no_session_deletion.py`` assert for every other
    mutation in this project.
    """
    return delivery_dir(root) / f"{instance_id}.json"


def _window_of(record: dict[str, Any]) -> dict[str, Any]:
    """The ``window`` object of one record, typed. Absent/malformed is empty."""
    raw = record.get("window")
    return raw if isinstance(raw, dict) else {}


def _load_record(path: Path) -> dict[str, Any] | None:
    """One record, or ``None`` when it is absent, torn, or no longer live.

    The two reaping rules are unchanged from the single-file reader and are
    ``scan_viewers``' rules: the ``pid`` must be alive and the ``heartbeat_at``
    must be inside :data:`PRESENCE_TTL_S`. Reaped here rather than unlinked — the
    file belongs to a process that may be starting up again under the same pid,
    and unlinking another process's lease is not a reader's business.
    """
    try:
        data: Any = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        pid = int(data.get("pid") or 0)
        heartbeat = float(data.get("heartbeat_at") or 0.0)
    except (TypeError, ValueError):
        return None
    if pid <= 0 or not pid_alive(pid):
        return None
    if time.time() - heartbeat > PRESENCE_TTL_S:
        return None
    return data


def _record_paths(root: Path | None) -> list[Path]:
    """Every record a reader must consider: the legacy file, then the per-instance ones.

    Built from the directory rather than through :func:`delivery_path` so that a
    failure to list it degrades to "no records" instead of raising — this reader
    runs on the announce path, where an exception would break a turn's
    completion to answer a banner question. Nothing is created on the way now
    either: an absent directory IS "no records".
    """
    try:
        directory = delivery_dir(root)
        records = sorted(directory.glob("*.json"))
    except OSError:
        return []
    return [directory.parent / DELIVERY_FILENAME, *records]


def read_delivery(root: Path | None = None) -> DesktopPresence:
    """Read and reap every publisher's record, and aggregate them, uncached.

    AN AGGREGATE OVER EVERY LIVE PUBLISHER, NOT ONE FILE (review round 1, R6).
    Presence is a PER-PROCESS assertion — one serve process owns one set of live
    feed subscriptions — but it used to be stored in a machine-wide file, so two
    backends overwrote and revoked each other. Storing it per process and
    unioning on read is what makes "can this host raise a banner" mean the same
    thing as "can this host raise a banner for this backend", which is the
    contract the runtime is actually held to.

    Every failure is the absent answer rather than an exception. This runs on
    the runtime's announce path, where a missing or half-written file is
    ordinary (no app paired) and a raise would break a turn's completion for
    the sake of a banner decision.
    """
    live: list[dict[str, Any]] = []
    for path in _record_paths(root):
        record = _load_record(path)
        if record is not None:
            live.append(record)
    if not live:
        return NO_DESKTOP_PRESENCE

    def heartbeat_of(record: dict[str, Any]) -> float:
        return float(record.get("heartbeat_at") or 0.0)

    # WINDOW STATE IS SINGULAR, SO IT IS TAKEN RATHER THAN UNIONED: "which
    # conversation is on screen" has exactly one answer. The freshest beat wins,
    # which is the same rule the single-writer version already applied among one
    # process's claims, extended across processes. Records WITH a window are
    # preferred over windowless ones even when a windowless sibling beats more
    # recently — taking the freshest unconditionally would blank the session id
    # of a window that is genuinely on screen, which is a rung-1 signal.
    windowed = [record for record in live if _window_of(record).get("exists")]
    newest = max(windowed or live, key=heartbeat_of)
    # ANNOTATED ON PURPOSE. `newest` is whatever `json.loads` produced, so the
    # window object arrives untyped and the type checker cannot see that the
    # `isinstance` below is what makes every `.get` on it safe. The annotation
    # is the reader's promise that the branch really did run.
    window: dict[str, Any] = _window_of(newest)
    session_id = str(newest.get("session_id") or "")
    has_window = bool(window.get("exists"))

    # REACHABILITY AND KINDS *ARE* UNIONED. Two live backends are two live
    # servers, and a banner either of them can raise does reach this machine.
    # Intersecting would let the weaker sibling veto the stronger one, which is
    # precisely the clobbering this change removes.
    kinds: set[str] = set()
    deliverable = False
    # DESK REPORTS ARE UNIONED TOO, and evaluated per report by the predicate
    # (ADR 0008 §3.2 "any"): a person at any desk this backend serves is at a
    # desk. A malformed entry is skipped, i.e. it contributes nothing.
    desk: list[DeskReport] = []
    for record in live:
        raw_desk = record.get("desk")
        if isinstance(raw_desk, list):
            desk.extend(
                report for report in map(DeskReport.from_record, raw_desk) if report is not None
            )
    for record in live:
        if not bool(record.get("can_notify")):
            continue
        if int(record.get("subscribers") or 0) > 0:
            deliverable = True
        raw_kinds = record.get("can_notify_kinds")
        if isinstance(raw_kinds, list):
            kinds.update(str(entry) for entry in raw_kinds if isinstance(entry, str))

    return DesktopPresence(
        present=True,
        deliverable=deliverable,
        kinds=frozenset(kinds),
        # A windowless app cannot be displaying anything, so its ``session_id``
        # is not evidence of a card on screen. This is the backend half of
        # "clear ``current_session`` when the last window closes": the field
        # may be stale for a beat, and a routing decision must not read it.
        session_id=session_id if has_window else "",
        # FOCUSED, VISIBLE AND NOT MINIMISED, and the publisher is the window's
        # OWN process (Electron main). The renderer's
        # ``document.visibilityState``/``hasFocus()`` cannot answer this:
        # ``local-operator-ui`` documents that test as unsound, because a
        # window the user is looking at reports ``hidden`` while Electron is
        # throttling it and a window behind another app reports ``visible``.
        attended=has_window
        and bool(window.get("focused") and window.get("visible") and not window.get("minimized")),
        has_window=has_window,
        pid=int(newest.get("pid") or 0),
        heartbeat_at=max(heartbeat_of(record) for record in live),
        desk=tuple(desk),
    )


def desktop_presence(root: Path | None = None, *, cached: bool = True) -> DesktopPresence:
    """The aggregate lease. See :data:`PRESENCE_CACHE_TTL_S`.

    ``cached`` is the default because the ANNOUNCE path asks this question on
    every turn settle and the answer is decorative there: a 2 s-stale
    "a desktop is reachable" only costs a deferral the retry ladder re-checks.

    A caller whose decision is TERMINAL passes ``cached=False``, because for it
    a stale answer is not a delay but a wrong answer nothing revisits. The feed's
    banner gate is the one such caller today: it reads focus state to decide
    whether to raise a banner at all, and nothing re-decides that afterwards —
    the frame is simply not published. Reading it through the cache meant a
    completion landing within ``PRESENCE_CACHE_TTL_S`` of the user leaving the
    window was suppressed machine-wide, on the strength of focus the user had
    already given up. Costs a ``readdir`` plus one small read per candidate
    banner, which is why only this caller pays it.
    """
    if not cached:
        return read_delivery(root)
    key = str((root or config_dir()))
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit is not None and hit[0] > now:
        return hit[1]
    value = read_delivery(root)
    _CACHE[key] = (now + PRESENCE_CACHE_TTL_S, value)
    return value


def desktop_delivery_present(root: Path | None = None, kind: str | None = None) -> bool:
    """True when a desktop app on THIS HOST can attempt a banner for ``kind``.

    The one predicate rung 2 is built from, in both of its consumers: the
    runtime's completion arm and the TUI's background announcer. They must read
    the same answer, or a machine running both raises the banner from whichever
    polls faster.

    ``kind`` is REQUIRED to be supplied by a caller routing a specific event,
    and the gate path does not call this at all: the presence is narrowed by
    kind because the machine-wide feed carries completions only, so an `ask` or
    an `approval` must keep the per-session lease and the per-session claim that
    reaches it today.
    """
    return desktop_presence(root).delivers(kind)


def desktop_viewing_session(root: Path | None = None) -> str:
    """The session a desktop window is really showing, or ``""``.

    Empty unless the window is genuinely attended (see
    :attr:`DesktopPresence.attended`), because "which conversation is on
    screen" is only a meaningful answer when somebody is looking at the screen.
    """
    presence = desktop_presence(root)
    return presence.session_id if presence.attended else ""


def desktop_attending_session(
    session_id: str, root: Path | None = None, *, cached: bool = True
) -> bool:
    """Whether a desktop window is attended AND showing ``session_id``.

    Rung 1's desktop contribution. A desktop lease is a *machine-wide*
    reachability signal; without this test it would read as "a human is reading
    every session", which is the failure mode the visibility rung exists to
    avoid.

    Kinds are deliberately NOT consulted: a card already painted in-band on the
    session's own stream is a card somebody is looking at, whatever the app
    would be able to do with a banner.

    ``cached`` follows :func:`desktop_presence`: a caller whose decision is
    terminal (the push worker DROPS a completion on a True here) passes
    ``cached=False`` so it never acts on focus the user has already left.
    """
    return desktop_presence(root, cached=cached).attending(session_id)


def reset_cache() -> None:
    """Drop the cache. For tests and for a process that changes config root."""
    _CACHE.clear()

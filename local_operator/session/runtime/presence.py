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
DELIVERY_FILENAME = "delivery.json"

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
    """The delivery-lease directory, created 0700 on first use.

    Creating is the WRITER's business, but a reader reaches it too and a
    missing directory is an ordinary answer ("no desktop app has ever been
    paired here"), so this stays mkdir-on-read like ``viewer_run_dir``.

    The directory permissions ARE the authorization story, and they are the
    reason the whole mechanism is safe to make machine-wide: the file carries a
    routing answer, never a credential, and only the owning account can read
    the directory it lives in.
    """
    path = (root or config_dir()) / DESKTOP_RUN_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def delivery_path(root: Path | None = None) -> Path:
    """Where the backend's aggregate lease lives."""
    return desktop_run_dir(root) / DELIVERY_FILENAME


def read_delivery(root: Path | None = None) -> DesktopPresence:
    """Read and reap the aggregate lease, uncached.

    Every failure is the absent answer rather than an exception. This runs on
    the runtime's announce path, where a missing or half-written file is
    ordinary (no app paired) and a raise would break a turn's completion for
    the sake of a banner decision.
    """
    try:
        data: Any = json.loads(delivery_path(root).read_text())
    except (OSError, ValueError):
        return NO_DESKTOP_PRESENCE
    if not isinstance(data, dict):
        return NO_DESKTOP_PRESENCE
    try:
        pid = int(data.get("pid") or 0)
        heartbeat = float(data.get("heartbeat_at") or 0.0)
    except (TypeError, ValueError):
        return NO_DESKTOP_PRESENCE
    if pid <= 0 or not pid_alive(pid):
        # A dead writer left its last assertion behind. Reaped here rather than
        # unlinked: the file belongs to a process that may be starting up
        # again under the same pid, and unlinking another process's lease is
        # not a reader's business.
        return NO_DESKTOP_PRESENCE
    if time.time() - heartbeat > PRESENCE_TTL_S:
        return NO_DESKTOP_PRESENCE
    subscribers = int(data.get("subscribers") or 0)
    window = data.get("window") if isinstance(data.get("window"), dict) else {}
    session_id = str(data.get("session_id") or "")
    has_window = bool(window.get("exists"))
    raw_kinds = data.get("can_notify_kinds")
    kinds = (
        frozenset(str(entry) for entry in raw_kinds if isinstance(entry, str))
        if isinstance(raw_kinds, list)
        else frozenset()
    )
    return DesktopPresence(
        present=True,
        deliverable=bool(data.get("can_notify")) and subscribers > 0,
        kinds=kinds,
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
        pid=pid,
        heartbeat_at=heartbeat,
    )


def desktop_presence(root: Path | None = None) -> DesktopPresence:
    """The cached aggregate lease. See :data:`PRESENCE_CACHE_TTL_S`."""
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


def desktop_attending_session(session_id: str, root: Path | None = None) -> bool:
    """Whether a desktop window is attended AND showing ``session_id``.

    Rung 1's desktop contribution. A desktop lease is a *machine-wide*
    reachability signal; without this test it would read as "a human is reading
    every session", which is the failure mode the visibility rung exists to
    avoid.

    Kinds are deliberately NOT consulted: a card already painted in-band on the
    session's own stream is a card somebody is looking at, whatever the app
    would be able to do with a banner.
    """
    presence = desktop_presence(root)
    return presence.attended and bool(session_id) and presence.session_id == session_id


def reset_cache() -> None:
    """Drop the cache. For tests and for a process that changes config root."""
    _CACHE.clear()

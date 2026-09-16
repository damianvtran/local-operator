"""Publish one backend's desktop delivery presence for every sibling process.

The writer half of :mod:`local_operator.session.runtime.presence`. The HTTP
server owns it because it is the only process that can see the desktop app's
live feed subscriptions — and, unlike the app itself, it is on the machine
whose runtimes need the answer. A paired app on another host therefore reports
through ``POST /v1/desktop/presence`` and the answer materialises here, where
``local_operator serve``'s session runtimes read it.

WHY IT AGGREGATES RATHER THAN FORWARDS. ``can_notify`` must mean "whoever holds
this can actually deliver", which is the same standard the per-session watch
lease is held to (``docs/DESKTOP_API.md``). So a claim is believed only while
its SSE socket is live — a dropped socket revokes it — and three consecutive
missed beats expire it. The aggregate is therefore never "an app once said
yes"; it is "an app said yes and is still answering".

THE DIRECTORY PERMISSIONS ARE THE AUTHORIZATION STORY, copied from
``viewers.publish_viewer`` rather than reinvented: 0700 directory, 0600 file,
staged write so a reader sees either the old file or the new one and never a
half-written one. A torn read here would be read as "no desktop app", which is
a silently wrong answer to a routing question rather than an error.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_operator.session.runtime.presence import (
    PRESENCE_BEAT_S,
    PRESENCE_TTL_S,
    delivery_dir,
    delivery_record_path,
)
from local_operator.session.runtime.registry import pid_alive

logger = logging.getLogger(__name__)

#: How long a SIBLING's record must have been silent before this publisher may
#: unlink it (review round 2, R15).
#:
#: TWO full TTLs, and the multiplier is the safety argument: the reader has
#: already stopped believing the record one TTL earlier, and the owner has missed
#: six beats — so nothing is being taken away from a publisher that is merely
#: slow, and a process that resumes beating simply rewrites its record through
#: the staged ``os.replace`` below. One TTL would be the shortest thing that
#: works and would race a sibling whose beat was delayed by a busy loop.
DEAD_RECORD_AGE_S = 2 * PRESENCE_TTL_S


@dataclass
class PresenceClaim:
    """One live feed subscription's assertion about its app.

    Refreshed wholesale by every beat: a client that stops sending the window
    state is not "still focused", it is unknown, and the last known value is
    deliberately NOT retained — an app that crashed with its window focused
    would otherwise keep suppressing a runtime banner from beyond its own
    heartbeat.
    """

    can_notify: bool = False
    #: The notification kinds this app claims it can deliver. Empty means "no
    #: claim", which the reader treats as "reaches nothing" rather than "any
    #: kind": an app that forgot to advertise must not win rung 2 for a kind
    #: nobody checked. See ``session/runtime/presence``.
    kinds: frozenset[str] = frozenset()
    #: Whether the app currently has a window at all. A windowless app (macOS
    #: dock, no window) can raise a banner — Electron's Notification is
    #: window-independent — but cannot be DISPLAYING anything, so the reader
    #: discards its ``session_id``.
    has_window: bool = False
    session_id: str = ""
    focused: bool = False
    visible: bool = False
    minimized: bool = False
    seen_at: float = field(default_factory=time.monotonic)


class DesktopDeliveryPublisher:
    """The lease ONE HTTP server owns, published in its own record (R6).

    Created with the feed and torn down with it. Every method is safe to call
    from the event loop: the only filesystem work is the small staged write
    below, which is the same cost ``viewer_server.note_session`` already pays
    on the TUI's focus path.

    WHAT THIS CLASS MAY WITHDRAW. Only its own record, named by
    :attr:`instance_id`. Several serve processes can be live on one machine at
    once — that is ordinary here, since each config root gets its own — and
    while they shared a single ``run/desktop/delivery.json`` the last writer
    decided the whole machine's answer and the first process to exit deleted a
    live sibling's lease. Ownership is the fix, and it is why ``close`` unlinks a
    path derived from this instance rather than the machine-wide one.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        #: The serve instance this lease belongs to. Not a credential and not a
        #: routing token — diagnostics, so an operator looking at a stale file
        #: can tell which server wrote it.
        self.instance_id = uuid.uuid4().hex
        self.claims: dict[str, PresenceClaim] = {}
        self._beat_task: asyncio.Task[None] | None = None

    # -- the route's two entry points --------------------------------------

    def update(
        self,
        subscription_id: str,
        *,
        can_notify: bool,
        can_notify_kinds: list[str] | None = None,
        session_id: str = "",
        window: dict[str, Any] | None = None,
    ) -> None:
        """Record one subscription's beat and republish the aggregate.

        Called from ``POST /v1/desktop/presence``, which is why it must not
        raise: a malformed window object degrades to "not attended" rather than
        failing the app's heartbeat. The alternative — a 422 for a state field
        — would let a client's stale field list revoke the very presence the
        route exists to establish.
        """
        state = window if isinstance(window, dict) else {}
        raw_kinds = can_notify_kinds if isinstance(can_notify_kinds, list) else []
        self.claims[subscription_id] = PresenceClaim(
            can_notify=bool(can_notify),
            kinds=frozenset(str(kind) for kind in raw_kinds if isinstance(kind, str)),
            has_window=bool(state.get("exists")),
            session_id=str(session_id or ""),
            focused=bool(state.get("focused")),
            visible=bool(state.get("visible")),
            minimized=bool(state.get("minimized")),
        )
        self._write()
        self._ensure_beat()

    def drop(self, subscription_id: str) -> None:
        """Revoke one subscription's claim, on its socket's disconnect.

        THE LOAD-BEARING REVOCATION. Route 11 of the design's risk list is "rung
        2 staying eligible while the desktop's feed is actually dead": a lease
        withdrawn only on a missed heartbeat leaves a 45 s window in which the
        runtime stays silent for a banner nobody can raise. The SSE socket is
        the liveness signal, so its teardown takes the claim with it — the same
        rule ``DesktopSubscription.overflow`` applies to a session stream.
        """
        if self.claims.pop(subscription_id, None) is None:
            return
        self._write()

    def close(self) -> None:
        """Stop beating and withdraw THIS instance's record. Idempotent, never raises.

        Withdrawing only its own record is the point (R6): this is an exit path,
        and it used to delete the machine-wide file — so a stopping server took a
        live sibling's lease with it. The sibling's own reader now keeps seeing
        the sibling.
        """
        self.claims.clear()
        if self._beat_task is not None:
            self._beat_task.cancel()
            self._beat_task = None
        try:
            self._record_path().unlink()
        except OSError:
            # Best-effort by contract, like ``unpublish_viewer``: an exit path
            # must not raise over a missing file.
            pass

    def _record_path(self) -> Path:
        """This instance's own record. The ONLY delivery path this class may remove."""
        return delivery_record_path(self.instance_id, self.root)

    def present(self) -> bool:
        """Whether the aggregate currently asserts reachability."""
        return any(claim.can_notify for claim in self.claims.values())

    def kinds(self) -> frozenset[str]:
        """Every kind any live claim says it can deliver.

        A union rather than an intersection: two subscribers are two windows,
        and a kind either of them can deliver IS deliverable by this app.
        """
        return frozenset(
            kind for claim in self.claims.values() if claim.can_notify for kind in claim.kinds
        )

    # -- the beat ----------------------------------------------------------

    def _ensure_beat(self) -> None:
        if self._beat_task is not None and not self._beat_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop (a synchronous caller, or a test driving the methods
            # directly). The write above already happened; only the renewals
            # need a loop, and there is nothing to renew them for.
            return
        self._beat_task = loop.create_task(self._beat_loop())

    async def _beat_loop(self) -> None:
        """Renew the lease every beat, and expire what nobody is renewing.

        The writer's own ``heartbeat_at`` must never be freshened on behalf of a
        claim that has stopped arriving: that would be the server asserting "the
        app is still there" on evidence it does not have, and would keep rung 2
        eligible for a dead app indefinitely. So the beat first drops anything
        older than the TTL, then writes only if something survives.
        """
        try:
            while True:
                await asyncio.sleep(PRESENCE_BEAT_S)
                self._reap()
                if not self.claims:
                    # Nothing left to assert. Stop ticking; the next claim
                    # restarts the loop.
                    self._write()
                    return
                self._write()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a lease is chrome; never break the server
            logger.debug("desktop delivery presence beat failed", exc_info=True)

    def _reap(self) -> None:
        cutoff = time.monotonic() - PRESENCE_TTL_S
        for subscription_id, claim in list(self.claims.items()):
            if claim.seen_at < cutoff:
                del self.claims[subscription_id]

    # -- the file ----------------------------------------------------------

    def _prune_dead_records(self) -> None:
        """Unlink the sibling records this publisher can PROVE dead (R15).

        WHY A WRITER, AND WHY NOT THE READER. A reader reaps a dead record in
        the ANSWER (``presence._load_record``), which is right — it must not
        delete a lease that belongs to a process which may be starting up again
        under the same pid — but it means nothing ever removes the file. The
        repo restarts serve daemons on build drift by design, so every process
        that dies without reaching ``close()`` leaves one behind for good, and
        every reader pays for all of them: a readdir plus one small read each,
        on the announce path and on every banner decision. A PUBLISHER may
        prune, because it is the only kind of process that owns a record here
        and it can prove death with the reader's own two rules plus an age.

        TWO GUARDS, because this is the one operation in this class that can
        take a SIBLING's lease away:

        * never this publisher's own record (that is ``close``'s business), and
        * the entry is re-identified between the decision and the unlink, so a
          record a live sibling REPLACED in that window is left alone. The
          staged write ends in ``os.replace``, which changes the inode, so an
          inode/mtime change is exactly "somebody rewrote this while I was
          reading it" — and without that check, a sweep would race the very
          revocation the per-instance layout exists to prevent.

        Nothing here touches the DIRECTORY, only individual entries, and every
        failure is suppressed: an unlink is atomic, so a crash between two of
        them leaves a directory the next publisher can still read and write in
        full.

        THE BOUNDARY IS THE READER'S LIVE-SET, NOT "CAN I PARSE IT" (review
        round 3, N6 — this used to claim no malformed entry is ever deleted,
        which the `pid > 0` conjunct below already contradicted). An entry that
        does not parse into a mapping, or whose ``pid``/``heartbeat_at`` are not
        numeric, is LEFT ALONE: no writer in this repo produces that shape, so
        this loop cannot attribute it to a process and cannot call it dead,
        and a reader answers it as absent in the meantime. A record that DOES
        parse and positively declares no live pid — ``pid`` missing, zero or
        negative, resolved as ``int(data.get("pid") or 0)``, the same spelling
        `presence._load_record` uses — or silence past
        :data:`DEAD_RECORD_AGE_S` is unlinked on the first beat, and that is
        safe for exactly the reason the reader's own rule is safe: `pid <= 0`
        can never satisfy the reader's liveness conjunct, so no reader can be
        holding that record as live.

        The keep-set is deliberately WIDER than the reader's: two TTLs of
        silence here against one in `_load_record`, so a record a reader could
        still answer as live is never even a candidate for deletion. Anything
        that changes how either side resolves `pid` — a schema that stops
        writing it, a reader that stops requiring it — has to change BOTH, or
        this sweep becomes the revocation path the inode check below exists to
        prevent.
        """
        try:
            entries = list(delivery_dir(self.root).glob("*.json"))
        except OSError:
            return
        own = self._record_path()
        cutoff = time.time() - DEAD_RECORD_AGE_S
        for path in entries:
            if path == own:
                continue
            try:
                before = path.stat()
                data: Any = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            try:
                pid = int(data.get("pid") or 0)
                heartbeat = float(data.get("heartbeat_at") or 0.0)
            except (TypeError, ValueError):
                continue
            # The reader's two rules, with the age standing in for them: a
            # publisher that is alive and beating every PRESENCE_BEAT_S never
            # reaches this cutoff, and a pid that is gone prunes at once.
            if pid > 0 and pid_alive(pid) and heartbeat >= cutoff:
                continue
            try:
                after = path.stat()
                if (after.st_ino, after.st_mtime_ns) != (before.st_ino, before.st_mtime_ns):
                    continue
                path.unlink()
            except OSError:
                continue

    def _write(self) -> None:
        """Materialise this instance's record, or withdraw it when nothing is left.

        Written on every change AND on the beat, so a reader's 45 s TTL is
        always three missed beats away rather than one slow moment away.

        ONE RECORD PER PROCESS (R6). The payload is this server's own assertion
        about its own live subscriptions; it is not a machine-wide answer, and
        the reader is what unions several of them. So there is no cross-process
        state to merge here, and no writer can clobber another's.
        """
        # BEFORE the early return below: a publisher whose own claims have all
        # gone is still the right process to sweep a sibling that died without
        # cleaning up, so the sweep must not depend on this instance having
        # anything to say. Best-effort throughout — a lease is chrome, and this
        # runs inside the route's heartbeat.
        self._prune_dead_records()
        if not self.claims:
            try:
                self._record_path().unlink()
            except OSError:
                pass
            return

        # Staged in the record's OWN directory so the replace stays on one
        # filesystem, and created 0700 with the parent.
        directory = delivery_dir(self.root)
        claims = list(self.claims.values())
        # The newest claim's window is the one reported: one app, one window
        # (a second instance is refused by the app itself), so "which window is
        # attended" has exactly one answer within this process, and picking the
        # freshest beat makes a replacement window's state take effect
        # immediately.
        newest = max(claims, key=lambda claim: claim.seen_at)
        payload = {
            "pid": os.getpid(),
            "instance_id": self.instance_id,
            "can_notify": any(claim.can_notify for claim in claims),
            "can_notify_kinds": sorted(self.kinds()),
            "subscribers": len(claims),
            "window": {
                "exists": newest.has_window,
                "focused": newest.focused,
                "visible": newest.visible,
                "minimized": newest.minimized,
            },
            # NO SESSION WITHOUT A WINDOW. A windowless app is not displaying
            # anything, and a reader that honoured a stale id would treat a
            # closed window's last conversation as "on screen" — suppressing
            # the banner for the one session the user cannot see (round 1, m2).
            "session_id": newest.session_id if newest.has_window and newest.can_notify else "",
            "written_at": time.time(),
            "heartbeat_at": time.time(),
        }
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".delivery.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self._record_path())
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

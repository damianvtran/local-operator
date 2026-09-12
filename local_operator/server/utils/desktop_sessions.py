"""HTTP viewers of canonical runtimes, never a second execution host.

One bridge is shared by concurrent HTTP operations and event subscribers. Its
receipt sequence is deliberately independent of the runtime's frontend revision:
a snapshot covers paint state, not semantic receipts such as steering delivery.
The last reader detaches; neither socket disposal nor HTTP shutdown stops work.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import re
import sqlite3
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anyio import CancelScope

from local_operator.resume import (
    is_user_session,
    read_session_attachment,
    session_preview,
    write_session_attachment,
)
from local_operator.session.attached import AttachedSession
from local_operator.session.attachments import ATTACHMENTS_DIRNAME, AttachmentStore
from local_operator.session.attention import AttentionStore
from local_operator.session.catalog import load_catalog
from local_operator.session.frontend_state import (
    FrontendSync,
    FrontendUpdate,
    sync_wire_payload,
)
from local_operator.session.retention import DESKTOP_MARKER_NAME
from local_operator.session.transcript import read_transcript_page

logger = logging.getLogger(__name__)

SESSION_ID = re.compile(r"^[a-f0-9]{12}$")
REPLAY_COUNT = 256
REPLAY_BYTES = 8 * 1024 * 1024
SUBSCRIBER_COUNT = 32
BRIDGE_COUNT = 64
WATCH_TTL = 45.0

#: The completion kinds the DESKTOP BRIDGE may put on the wire as a
#: ``notification`` frame. Narrower than ``NotificationKind`` on purpose.
#:
#: ``ask``/``approval`` are absent because they already reach the desktop as
#: ``pending_gate`` in the snapshot and update frames, and a second channel for
#: the same card is the duplicate this whole contract exists to prevent.
#:
#: ``interrupted`` is absent because the user pressed Ctrl+C or Esc a moment
#: ago and already knows — telling them their own stop worked is the definition
#: of a notification nobody wants, which is the same call the TUI already
#: makes. The counter-argument (on the desktop an interruption can come from
#: another surface) is real but undecidable here: ``AgentEndEvent`` carries
#: ``aborted`` with no actor. One frozenset entry away if that ever changes.
BRIDGE_NOTIFIABLE_KINDS = frozenset({"complete", "error"})


async def _no_takeover() -> None:
    raise RuntimeError("Desktop viewers cannot own a runtime")


@dataclass(eq=False)
class DesktopSubscription:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    queue: asyncio.Queue[tuple[dict[str, Any], int] | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=REPLAY_COUNT)
    )
    queued_bytes: int = 0
    visible: bool = False
    can_notify: bool = False
    expires: float = 0.0
    overflow: bool = False


class DesktopSessionBridge:
    def __init__(self, root: Path, session_id: str, cwd: str) -> None:
        self.root, self.session_id, self.cwd = root, session_id, cwd
        self.remote: AttachedSession | None = None
        self.epoch = uuid.uuid4().hex
        self.sequence = 0
        self.replay: deque[tuple[dict[str, Any], int]] = deque()
        self.replay_bytes = 0
        self.subscribers: dict[str, DesktopSubscription] = {}
        self.users = 0
        self.touched = time.monotonic()
        self.lock = asyncio.Lock()
        self.watch_lock = asyncio.Lock()
        self.unsubscribers: list[Any] = []
        self.watch_task: asyncio.Task[None] | None = None
        self.attention_task: asyncio.Task[None] | None = None
        self.attention: dict[str, Any] = {}
        self.attention_poll_key: tuple[tuple[int, int, int], bool] | None = None

    async def acquire(self) -> AttachedSession:
        async with self.lock:
            self.users += 1
            self.touched = time.monotonic()
            try:
                if self.remote is None:
                    remote = await AttachedSession.cold(
                        self.session_id,
                        config_dir=self.root,
                        cwd=self.cwd,
                        takeover_factory=_no_takeover,
                        surface="desktop",
                    )
                    self.remote = remote
                    # A detached interval has no receipt feed. A new epoch makes
                    # that gap explicit even when the runtime itself never died.
                    self.epoch = uuid.uuid4().hex
                    self.sequence = 0
                    self.replay.clear()
                    self.replay_bytes = 0
                    self.unsubscribers = [
                        remote.subscribe(self._event),
                        remote.subscribe_frontend(self._frontend).unsubscribe,
                    ]
                await self.remote.attach_existing()
                if self.attention_task is None:
                    self.attention_task = asyncio.create_task(self._poll_attention())
                return self.remote
            except BaseException:
                self.users -= 1
                if self.users == 0:
                    await self._detach()
                raise

    async def release(self) -> None:
        async with self.lock:
            self.users -= 1
            self.touched = time.monotonic()
            if self.users == 0:
                await self._detach()

    async def _detach(self) -> None:
        if self.watch_task is not None:
            self.watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.watch_task
            self.watch_task = None
        for unsubscribe in self.unsubscribers:
            unsubscribe()
        self.unsubscribers.clear()
        remote, self.remote = self.remote, None
        if remote is not None:
            await remote.dispose()
        # LAST, and suppressing the task's OWN failure rather than only
        # CancelledError: awaiting a task that already died re-raises its
        # exception, and with this block ahead of `dispose()` a store error
        # aborted teardown midway -- leaking the runtime's session and its
        # subscriptions while `users` had already reached 0, so a later
        # `acquire()` reused a half-torn bridge. A read receipt must never
        # strand a session runtime.
        if self.attention_task is not None:
            self.attention_task.cancel()
            with contextlib.suppress(BaseException):
                await self.attention_task
            self.attention_task = None

    async def close(self) -> None:
        for sub in self.subscribers.values():
            self._disconnect(sub)
        async with self.lock:
            await self._detach()

    def _disconnect(self, sub: DesktopSubscription) -> None:
        sub.overflow = True
        sub.visible = sub.can_notify = False
        while not sub.queue.empty():
            sub.queue.get_nowait()
        sub.queued_bytes = 0
        sub.queue.put_nowait(None)

    def publish(self, kind: str, payload: dict[str, Any], *, replay: bool = True) -> None:
        """Put one frame on every live subscriber, and normally into replay.

        ``replay=False`` publishes LIVE ONLY. It exists for the ``notification``
        frame, whose whole value is timeliness: replaying it on reconnect toasts
        the user about a turn that finished while their laptop lid was shut,
        possibly hours later. The durable signal for "you missed something" is
        not lost — the ``attention`` frame and the sidebar's unseen mark both
        survive a reconnect and are the right surface for it.

        THE SEQUENCE STILL ADVANCES for a non-replayed frame, and that is
        load-bearing rather than incidental. :meth:`events` computes ``gap``
        from ``after_seq < first - 1`` where ``first`` is the oldest RETAINED
        frame's seq, so a skipped seq simply never becomes ``first`` and a
        client reconnecting at the notification's own cursor still satisfies
        the test against the next retained frame. Not incrementing would
        instead make ``seq`` non-monotonic across the two paths and break the
        receipt cursor every renderer keeps.
        """
        self.sequence += 1
        frame = {
            "session_id": self.session_id,
            "epoch": self.epoch,
            "seq": self.sequence,
            "type": kind,
            "payload": payload,
        }
        size = len(json.dumps(frame, separators=(",", ":")).encode())
        if replay:
            self.replay.append((frame, size))
            self.replay_bytes += size
            while self.replay and (
                len(self.replay) > REPLAY_COUNT or self.replay_bytes > REPLAY_BYTES
            ):
                _, removed = self.replay.popleft()
                self.replay_bytes -= removed
        for sub in self.subscribers.values():
            if sub.overflow:
                continue
            if sub.queue.full() or sub.queued_bytes + size > REPLAY_BYTES:
                # Never silently discard a semantic event. Closing forces an
                # authoritative gap snapshot on reconnect, and revokes presence.
                self._disconnect(sub)
            else:
                sub.queue.put_nowait((frame, size))
                sub.queued_bytes += size

    def _event(self, event: Any) -> None:
        self.publish("event", event.model_dump(mode="json"))

    def _frontend(self, update: FrontendUpdate) -> None:
        # Keep the runtime's field deltas, not a full snapshot per streamed token.
        # Trajectories are intentionally opt-in on the runtime and absent here;
        # large roster/usage fields still pass through the shared wire budget.
        payload = update.model_dump(mode="json")
        # Receipt revisions outlive a runtime epoch. Only the independent durable
        # projection below may update them; a delayed runtime delta must not undo
        # a read made through another process while this stream stays mounted.
        payload["changes"].pop("attention", None)
        payload["job_trajectory_appends"] = {}
        payload["job_trajectory_replacements"] = []
        if {"jobs", "usage_components"} & update.changes.keys():
            bounded = self.state()["snapshot"]
            for key in ("jobs", "usage_components"):
                if key in payload["changes"]:
                    payload["changes"][key] = bounded[key]
        self.publish("frontend.update", payload)

    async def refresh_attention(self) -> dict[str, Any]:
        state = await asyncio.to_thread(
            AttentionStore(self.root / "attention.db").state, f"session/{self.session_id}"
        )
        remote = self.remote
        state["supported"] = bool(
            remote is not None
            and (remote.is_cold or getattr(remote, "supports_completion_ack", False))
        )
        if state != self.attention:
            previous = self.attention
            self.attention = state
            # The initial snapshot owns the baseline; later changes have their
            # own receipt clock rather than borrowing a runtime sequence.
            if previous:
                self.publish("attention", state)
                # THE NOTIFICATION EDGE, published AFTER the attention frame so
                # a reader that toasts already holds the receipt state that
                # explains the toast. The same `previous` baseline rule governs
                # both: a bridge's FIRST read is the session's history, not
                # news, and opening a conversation must not announce the
                # completion it ended on last week.
                await self._maybe_publish_notification(previous, state)
        return state

    async def _maybe_publish_notification(
        self, previous: dict[str, Any], state: dict[str, Any]
    ) -> None:
        """Turn a newly published, unseen completion into one notification frame.

        THE AUTHORITY IS THE ATTENTION PUBLICATION, not any engine event. A
        `completions` row exists only because ``Session._publish_attention_
        outcome`` decided the turn produced a notifiable outcome — in the
        process that owns the job manager, using the same delegated-children
        check the TUI uses (``job.type == "task" and job.status == "running"``).
        A delegating parent's premature ``agent_end`` writes an ``eligible:
        False`` marker and publishes nothing, so there is simply no row for the
        bridge to see, and each settled child re-enters as a fresh turn whose
        own completion publishes normally.

        That is why this method asks no questions about jobs, ``agent_end`` or
        ``turn_end``: reconstructing the decision here would mean making it
        again in a process with less information, which is how a frontend ends
        up disagreeing with the TUI about whether a turn finished. The bridge
        OBSERVES the decision; it does not judge it.

        NO CLAIM IS TAKEN HERE. ``claim_delivery`` is claim-then-deliver, and
        the claimant must be the deliverer — between this frame and an OS
        banner lie an SSE socket, the Electron main process, a support check
        and a focus gate. A claim taken here that the renderer then suppresses
        would mark the completion delivered while nobody was told, for good. A
        frame is an OFFER; the renderer claims through ``POST /notified``
        immediately before it shows the banner.

        Guarded end to end: a notification is chrome, and this runs inside the
        1 s attention poll whose loop already treats a store error as costing
        one tick rather than the feature.
        """
        token = state.get("completion_token")
        if (
            not token
            or token == previous.get("completion_token")
            or not state.get("unseen")
            or state.get("kind") not in BRIDGE_NOTIFIABLE_KINDS
        ):
            return
        try:
            from local_operator.notifications import (
                NOTIFICATION_CONTRACT_VERSION,
                compose,
            )

            # The LIVE name wins over the sidecar: a rename reaches frontend
            # state before it reaches `title.json`, and `compose` falls back to
            # the stored title on its own when this is empty (a cold bridge has
            # no runtime to ask).
            remote = self.remote
            session_name = ""
            if remote is not None:
                session_name = getattr(remote.frontend_state, "conversation_title", "") or ""
            # `compose` reads up to 128 KiB for the title and 64 KB for the
            # preview, and `refresh_attention` runs on the event loop. Off-loop
            # for the same reason the store read above is.
            composed = await asyncio.to_thread(
                compose,
                state["kind"],
                session_dir=self.root / "sessions" / self.session_id,
                session_name=session_name,
            )
            self.publish(
                "notification",
                {
                    "contract": NOTIFICATION_CONTRACT_VERSION,
                    "kind": composed.kind,
                    "title": composed.title,
                    "status": composed.status,
                    "body": composed.body,
                    "body_is_snippet": composed.body_is_snippet,
                    # Additive since the first draft of this frame; a renderer
                    # that does not know the field simply shows the body, which
                    # is already the right thing to do with it.
                    "body_is_failure": composed.body_is_failure,
                    "title_is_session_name": composed.title_is_session_name,
                    # Keyed on the DURABLE completion token rather than on this
                    # bridge's sequence: `acquire()` mints a new epoch and
                    # resets `sequence` to 0 after a detached interval, so a
                    # seq-keyed dedupe re-toasts the same completion on every
                    # reconnect.
                    "dedupe_key": f"complete:{self.session_id}:{token}",
                    "completion_token": token,
                    "session_name": composed.title if composed.title_is_session_name else None,
                    "focus_policy": "when_unfocused",
                },
                replay=False,
            )
        except Exception:  # noqa: BLE001 — chrome must not cost the attention poll
            logger.debug("notification compose failed for %s", self.session_id, exc_info=True)

    async def _poll_attention(self) -> None:
        # Read-only polling is shared by every subscriber of this bridge and
        # independent of watch leases. It also works while no runtime is running.
        #
        # The body is guarded because this store has other writers: a `database
        # is locked` that outlives its 2 s timeout is routine contention, and
        # letting it end the loop stopped cross-process read sync for the life
        # of the bridge -- the phone and the TUI would clear an unread
        # completion while the desktop kept showing it, silently and forever.
        # A transient store error must cost one poll, not the feature. Matches
        # the suppression `_expire_watches` already uses for the same reason.
        store = AttentionStore(self.root / "attention.db")
        failing = 0
        while True:
            try:
                # `revision()` exists for exactly this loop and is far cheaper
                # than the full per-conversation read; the steady state is a
                # store nothing has written since the last tick.
                #
                # The runtime's own state is part of the key because `supported`
                # is derived from it, not from the store: a runtime starting or
                # going cold changes that answer while the store is untouched,
                # so gating on the revision alone would pin `supported` to
                # whatever happened to be true when the bridge attached.
                remote = self.remote
                key = (
                    await asyncio.to_thread(store.revision),
                    remote is not None
                    and (remote.is_cold or getattr(remote, "supports_completion_ack", False)),
                )
                if key != self.attention_poll_key:
                    await self.refresh_attention()
                    self.attention_poll_key = key
                if failing:
                    logger.info(
                        "attention poll recovered for %s after %d failure(s)",
                        self.session_id,
                        failing,
                    )
                    failing = 0
            except Exception as error:
                # Log the TRANSITION, not the tick. A transient error costs one
                # line, but a persistent one (corrupt schema, permissions, full
                # disk) would otherwise write ~3,600 identical warnings an hour
                # per bridge, across up to BRIDGE_COUNT bridges, burying
                # whatever else the operator needs to read. Recovery is logged
                # too, so the pair brackets the outage rather than leaving a
                # single warning of unknown duration.
                failing += 1
                if failing == 1:
                    logger.warning(
                        "attention poll failed for %s (further failures quiet "
                        "until it recovers): %s",
                        self.session_id,
                        error,
                    )
            await asyncio.sleep(1)

    def state(self) -> dict[str, Any]:
        assert self.remote is not None
        state = self.remote.frontend_state.model_copy(update={"attention": self.attention})
        return sync_wire_payload(
            FrontendSync(
                epoch=state.epoch,
                sequence=state.sequence,
                snapshot=state,
                live_cursor=state.history_cursor,
            )
        )

    async def snapshot(self) -> dict[str, Any]:
        # Decorative, so a busy or damaged receipt sidecar cannot stop a
        # conversation from OPENING. Before this field existed the snapshot
        # never touched `attention.db`; letting it raise here turned routine
        # write contention into a failure of the primary read path. The last
        # known state is kept rather than blanked -- it is what the previous
        # successful poll actually saw.
        with contextlib.suppress(sqlite3.Error, OSError):
            await self.refresh_attention()
        state = self.state()
        seq, epoch = self.sequence, self.epoch
        cursor = state["snapshot"].get("history_cursor")
        history: dict[str, Any] = {"entries": [], "has_more": False, "cursor_missing": False}
        if cursor:
            history = await self.history(through_id=cursor)
        return {
            "session_id": self.session_id,
            "epoch": epoch,
            "seq": seq,
            "type": "snapshot",
            "payload": {
                "frontend": state,
                "history": history,
                "cold": self.remote is None or self.remote.is_cold,
            },
        }

    async def history(
        self, *, before_id: str | None = None, through_id: str | None = None, limit: int = 100
    ) -> dict[str, Any]:
        try:
            page = await asyncio.to_thread(
                read_transcript_page,
                self.root / "sessions" / self.session_id,
                before_id=before_id,
                through_id=through_id,
                limit=limit,
            )
        except FileNotFoundError:
            return {
                "entries": [],
                "has_more": False,
                "cursor_missing": bool(before_id or through_id),
            }
        return {
            "entries": [json.loads(row.to_json()) for row in page.entries],
            "has_more": page.has_more,
            "cursor_missing": page.reconciled,
        }

    async def watch(self, subscription_id: str, *, visible: bool, can_notify: bool) -> None:
        sub = self.subscribers.get(subscription_id)
        if sub is None or sub.overflow:
            raise KeyError("This event subscription is no longer connected")
        sub.visible, sub.can_notify = visible, can_notify
        sub.expires = time.monotonic() + WATCH_TTL
        await self.refresh_watch()
        if self.watch_task is None or self.watch_task.done():
            self.watch_task = asyncio.create_task(self._expire_watches())

    async def refresh_watch(self) -> None:
        async with self.watch_lock:
            live = [
                s
                for s in self.subscribers.values()
                if not s.overflow and s.expires > time.monotonic()
            ]
            remote = self.remote
            if remote is not None and not remote.is_cold:
                await remote.update_desktop_watch(
                    visible=any(s.visible for s in live),
                    can_notify=any(s.can_notify for s in live),
                )

    async def _expire_watches(self) -> None:
        while True:
            remaining = [
                s.expires
                for s in self.subscribers.values()
                if not s.overflow and s.expires > time.monotonic()
            ]
            if not remaining:
                # LAST lease has expired. Returning here without a final refresh
                # left the runtime holding whatever presence the previous pass
                # asserted -- visible, notifiable -- for the rest of the
                # session, because nothing else recomputes it once the loop is
                # gone. The expiry that ends the loop is exactly the one the
                # runtime still needs to be told about.
                with contextlib.suppress(ConnectionError, RuntimeError):
                    await self.refresh_watch()
                return
            await asyncio.sleep(max(0, min(remaining) - time.monotonic()))
            with contextlib.suppress(ConnectionError, RuntimeError):
                await self.refresh_watch()

    def subscribe(self) -> DesktopSubscription:
        if len(self.subscribers) >= SUBSCRIBER_COUNT:
            raise ValueError("Too many event subscribers")
        sub = DesktopSubscription()
        self.subscribers[sub.id] = sub
        return sub

    async def events(
        self, sub: DesktopSubscription, *, epoch: str | None, after_seq: int
    ) -> AsyncGenerator[dict[str, Any], None]:
        try:
            cutoff = self.sequence
            first = self.replay[0][0]["seq"] if self.replay else cutoff + 1
            gap = epoch != self.epoch or after_seq < first - 1 or after_seq > cutoff
            replay = (
                [f for f, _ in self.replay if after_seq < f["seq"] <= cutoff] if not gap else []
            )
            snapshot = await self.snapshot()
            yield {
                "session_id": self.session_id,
                "epoch": self.epoch,
                "seq": cutoff,
                "type": "open",
                "payload": {
                    "subscription_id": sub.id,
                    "gap": gap,
                    "watch_ttl_seconds": WATCH_TTL,
                },
            }
            # Replay receipts BEFORE the authoritative snapshot so cumulative
            # record updates cannot repaint newer snapshot text with old deltas.
            # The open frame is metadata, NOT permission to skip this replay.
            for frame in replay:
                yield frame
            yield snapshot
            while True:
                try:
                    item = await asyncio.wait_for(sub.queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield {"type": "heartbeat", "session_id": self.session_id}
                    continue
                if item is None:
                    yield {"type": "gap", "session_id": self.session_id}
                    return
                frame, size = item
                sub.queued_bytes -= size
                if frame["seq"] > cutoff:
                    yield frame
        finally:
            self.subscribers.pop(sub.id, None)
            # ASGI disconnect runs inside a cancelled anyio scope. Cleanup must
            # still reach the runtime; otherwise a dead renderer leaves presence
            # asserted until TTL expiry and the bridge never releases its socket.
            with CancelScope(shield=True), contextlib.suppress(ConnectionError, RuntimeError):
                await self.refresh_watch()


class DesktopSessions:
    """Bounded adapter cache; canonical identity lives in the session directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.bridges: dict[str, DesktopSessionBridge] = {}
        self.lock = asyncio.Lock()

    async def acknowledge_attention(self, session_id: str, token: str) -> dict[str, Any]:
        """A read receipt never admits work, binds a viewer, or starts a runtime.

        Validate the same durable user-session namespace as the bridge, but do
        not enter its acquire path: a completed cold conversation is readable
        even when its runtime and the mobile daemon are both stopped.
        """

        def acknowledge() -> dict[str, Any]:
            if not SESSION_ID.fullmatch(session_id):
                raise KeyError("Unknown session")
            path = self.root / "sessions" / session_id
            if not path.is_dir() or not is_user_session(path):
                raise KeyError("Unknown session")
            return AttentionStore(self.root / "attention.db").acknowledge(
                f"session/{session_id}", token
            )

        return await asyncio.to_thread(acknowledge)

    async def claim_notification(self, session_id: str, token: str) -> bool:
        """Claim the right to TOAST ``token``; exactly one surface ever wins.

        NOTIFYING IS NOT READING, and this is the boundary that keeps the two
        watermarks apart. ``claim_delivery`` writes ``deliveries`` only: the
        sidebar's unseen mark and ``receipts.acknowledged`` are untouched, so a
        session the user was merely *told about* stays unread until they
        actually open it. Routing this through :meth:`acknowledge_attention`
        instead would clear the mark for a conversation nobody looked at, which
        is the one thing ``docs/ATTENTION.md`` forbids outright.

        Cold path, exactly like :meth:`acknowledge_attention`: same session-id
        validation, no bridge acquire, no runtime spawn. A completion worth
        announcing is usually one whose owner has already exited, and a banner
        is never a reason to start a process.

        ``backend="desktop"`` names the claimant. The column is diagnostics
        only — no decision may read it, because a claim that consulted anything
        beyond the monotonic sequence would stop being clock-free — but naming
        it correctly is what makes a store dump readable when two surfaces
        disagree about who toasted.

        Returns ``False`` for an unknown or foreign token rather than raising:
        the caller's next step is "show or do not show a banner", and a
        surface that cannot claim simply stays quiet.
        """

        def claim() -> bool:
            if not SESSION_ID.fullmatch(session_id):
                raise KeyError("Unknown session")
            path = self.root / "sessions" / session_id
            if not path.is_dir() or not is_user_session(path):
                raise KeyError("Unknown session")
            return AttentionStore(self.root / "attention.db").claim_delivery(
                f"session/{session_id}", token, "desktop"
            )

        return await asyncio.to_thread(claim)

    async def attachment(self, session_id: str, digest: str) -> tuple[bytes, str]:
        """Decoded bytes and mime type for one content-addressed attachment.

        Durable transcript rows reference images by digest, not by payload:
        ``transcript._externalize_attachments`` strips ``data`` from any block
        over 1 KiB of base64 and leaves ``{"attachment": <digest>,
        "mime_type": ...}`` behind. ``/history`` serves those rows verbatim,
        so a reading surface can see that an image WAS there and has no way to
        fetch it. This is that way.

        Deliberately outside :meth:`session`, exactly like
        :meth:`acknowledge_attention` and for the same reason: reading a
        screenshot out of a finished conversation must not start a runtime
        process. The session id is still validated against the same durable
        user-session namespace, so the route cannot be used to probe arbitrary
        directories, and the store is shared rather than per-session because
        the digest IS the content key.

        ``KeyError`` for an unknown session or an unresolvable digest — the
        store's own contract is that a miss is ordinary (an interrupted write,
        a hand-pruned store) and callers degrade to a placeholder rather than
        treating it as a fault.

        The session id is an EXISTENCE check, not a binding: it proves *a* user
        conversation by that name is on this machine, never that this digest
        belongs to it. The store is content-addressed and shared across
        conversations by design, so any valid user session id resolves any
        digest in it. The bearer already authorises the whole desktop surface,
        so this is not an escalation — but it is not per-session scoping
        either, and the URL shape reads as though it were.
        """

        def read() -> tuple[bytes, str]:
            # Both halves of this gate carry weight and neither is redundant.
            # The shape check keeps a crafted id from escaping the sessions
            # namespace through ``..`` before a path is ever built; the origin
            # check keeps this route out of SUBAGENT conversations, which are a
            # machine's delegated runs the user never opened and which the
            # desktop surface does not list. Dropping either is a one-token
            # edit, so each has a named test standing on it.
            if not SESSION_ID.fullmatch(session_id):
                raise KeyError("Unknown session")
            path = self.root / "sessions" / session_id
            if not path.is_dir() or not is_user_session(path):
                raise KeyError("Unknown session")
            resolved = AttachmentStore(self.root / ATTACHMENTS_DIRNAME).get(digest)
            if resolved is None:
                raise KeyError("Unknown attachment")
            data_b64, mime_type = resolved
            return base64.b64decode(data_b64), mime_type

        return await asyncio.to_thread(read)

    async def create(self, cwd: str, *, target: dict[str, str] | None = None) -> str:
        directory = Path(cwd).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError("Choose an existing working directory")
        binding = {"agent": "", "team": ""}
        if target:
            from local_operator.agents import AgentRegistry
            from local_operator.server.utils.desktop_profiles import validate_target
            from local_operator.teams import TeamRegistry

            binding[target["kind"]] = await asyncio.to_thread(
                validate_target,
                AgentRegistry(self.root),
                TeamRegistry(self.root),
                target["kind"],
                target["name"],
            )
        session_id = uuid.uuid4().hex[:12]
        path = self.root / "sessions" / session_id

        def persist() -> None:
            path.mkdir(parents=True, mode=0o700)
            if target:
                write_session_attachment(path, **binding, goal="")
                stored = read_session_attachment(path)
                if (
                    stored is None
                    or stored.agent != binding["agent"]
                    or stored.team != binding["team"]
                ):
                    # Never publish desktop.json after a best-effort writer lost
                    # the attachment. No possibly admitted work is deleted.
                    raise ValueError(
                        "The selected profile could not be saved. Retry after checking storage."
                    )
            # An explicitly created desktop draft needs an identity after an
            # HTTP restart, unlike the TUI's uncommitted welcome-screen draft.
            marker = path / DESKTOP_MARKER_NAME
            marker.write_text(json.dumps({"version": 1, "cwd": str(directory)}))
            marker.chmod(0o600)

        await asyncio.to_thread(persist)
        return session_id

    async def binding(self, session_id: str) -> dict[str, str | None]:
        def read() -> dict[str, str | None]:
            stored = read_session_attachment(self.root / "sessions" / session_id)
            return {
                "agent": stored.agent or None if stored else None,
                "team": stored.team or None if stored else None,
            }

        return await asyncio.to_thread(read)

    async def list(self, limit: int) -> list[dict[str, Any]]:
        def rows() -> list[dict[str, Any]]:
            entries = load_catalog(self.root, limit=limit)[:limit]
            attention: dict[str, dict[str, Any]] = {}
            with contextlib.suppress(sqlite3.Error, OSError):
                attention = AttentionStore(self.root / "attention.db").state_many(
                    f"session/{entry.id}" for entry in entries
                )
            result = []
            for entry in entries:
                row = entry.row._asdict()
                stored = read_session_attachment(self.root / "sessions" / entry.id)
                row.update(
                    {
                        "active": entry.active,
                        "status": {"code": entry.status_code, "label": entry.status},
                        "binding": {
                            "agent": stored.agent or None if stored else None,
                            "team": stored.team or None if stored else None,
                        },
                        "preview": session_preview(self.root / "sessions" / entry.id),
                    }
                )
                if f"session/{entry.id}" in attention:
                    row["attention"] = attention[f"session/{entry.id}"]
                result.append(row)
            return result

        return await asyncio.to_thread(rows)

    @contextlib.asynccontextmanager
    async def session(self, session_id: str) -> AsyncIterator[DesktopSessionBridge]:
        if not SESSION_ID.fullmatch(session_id):
            raise KeyError("Unknown session")
        async with self.lock:
            bridge = self.bridges.get(session_id)
            if bridge is None:
                path = self.root / "sessions" / session_id

                def locate() -> str:
                    if not path.is_dir() or not is_user_session(path):
                        raise KeyError("Unknown session")
                    marker = path / DESKTOP_MARKER_NAME
                    if marker.exists():
                        return str(json.loads(marker.read_text())["cwd"])
                    # The cold facade restores cwd from the durable canonical
                    # checkpoint. This fallback is only used by pre-checkpoint
                    # transcripts, whose historical launch directory is unknown.
                    from local_operator.session.frontend_state import (
                        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
                    )
                    from local_operator.session.transcript import Transcript

                    checkpoint = Transcript(path).latest_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE)
                    return str((checkpoint or {}).get("state", {}).get("cwd") or self.root.parent)

                cwd = await asyncio.to_thread(locate)
                if len(self.bridges) >= BRIDGE_COUNT:
                    idle = [b for b in self.bridges.values() if b.users == 0]
                    if not idle:
                        raise ValueError("Too many active desktop sessions")
                    oldest = min(idle, key=lambda b: b.touched)
                    del self.bridges[oldest.session_id]
                bridge = DesktopSessionBridge(self.root, session_id, cwd)
                self.bridges[session_id] = bridge
            # Reserve under the pool lock; eviction must not remove a bridge
            # between lookup and its first acquire.
            await bridge.acquire()
        try:
            yield bridge
        finally:
            with CancelScope(shield=True):
                await bridge.release()

    async def close(self) -> None:
        await asyncio.gather(*(bridge.close() for bridge in self.bridges.values()))
        self.bridges.clear()

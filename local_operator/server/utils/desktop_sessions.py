"""HTTP viewers of canonical runtimes, never a second execution host.

One bridge is shared by concurrent HTTP operations and event subscribers. Its
receipt sequence is deliberately independent of the owner's frontend revision:
a snapshot covers paint state, not semantic receipts such as steering delivery.
The last reader detaches; neither socket disposal nor HTTP shutdown stops work.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anyio import CancelScope

from local_operator.resume import is_user_session, recent_session_rows, session_preview
from local_operator.session.frontend_state import (
    FrontendSync,
    FrontendUpdate,
    sync_wire_payload,
)
from local_operator.session.remote import RemoteSession
from local_operator.session.transcript import read_transcript_page

SESSION_ID = re.compile(r"^[a-f0-9]{12}$")
REPLAY_COUNT = 256
REPLAY_BYTES = 8 * 1024 * 1024
SUBSCRIBER_COUNT = 32
BRIDGE_COUNT = 64
WATCH_TTL = 45.0


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
        self.remote: RemoteSession | None = None
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

    async def acquire(self) -> RemoteSession:
        async with self.lock:
            self.users += 1
            self.touched = time.monotonic()
            try:
                if self.remote is None:
                    remote = await RemoteSession.cold(
                        self.session_id,
                        config_dir=self.root,
                        cwd=self.cwd,
                        takeover_factory=_no_takeover,
                        surface="desktop",
                    )
                    self.remote = remote
                    # A detached interval has no receipt feed. A new epoch makes
                    # that gap explicit even when the owner itself never died.
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
        if self.attention_task is not None:
            self.attention_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.attention_task
            self.attention_task = None
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

    def publish(self, kind: str, payload: dict[str, Any]) -> None:
        self.sequence += 1
        frame = {
            "session_id": self.session_id,
            "epoch": self.epoch,
            "seq": self.sequence,
            "type": kind,
            "payload": payload,
        }
        size = len(json.dumps(frame, separators=(",", ":")).encode())
        self.replay.append((frame, size))
        self.replay_bytes += size
        while self.replay and (len(self.replay) > REPLAY_COUNT or self.replay_bytes > REPLAY_BYTES):
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
        # Keep the owner's field deltas, not a full snapshot per streamed token.
        # Trajectories are intentionally opt-in on the runtime and absent here;
        # large roster/usage fields still pass through the shared wire budget.
        payload = update.model_dump(mode="json")
        # Receipt revisions outlive an owner epoch. Only the independent durable
        # projection below may update them; a delayed owner delta must not undo
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
        from local_operator.session.attention import AttentionStore

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
            # own receipt clock rather than borrowing a runtime owner sequence.
            if previous:
                self.publish("attention", state)
        return state

    async def _poll_attention(self) -> None:
        # Read-only polling is shared by every subscriber of this bridge and
        # independent of watch leases. It also works while no owner is running.
        while True:
            await self.refresh_attention()
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
                # left the owner holding whatever presence the previous pass
                # asserted -- visible, notifiable -- for the rest of the
                # session, because nothing else recomputes it once the loop is
                # gone. The expiry that ends the loop is exactly the one the
                # owner still needs to be told about.
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
            # still reach the owner; otherwise a dead renderer leaves presence
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
        """A read receipt never admits work, binds a viewer, or starts an owner.

        Validate the same durable user-session namespace as the bridge, but do
        not enter its acquire path: a completed cold conversation is readable
        even when its owner and the mobile daemon are both stopped.
        """
        from local_operator.session.attention import AttentionStore

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

    async def create(self, cwd: str) -> str:
        directory = Path(cwd).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError("Choose an existing working directory")
        session_id = uuid.uuid4().hex[:12]
        path = self.root / "sessions" / session_id

        def persist() -> None:
            path.mkdir(parents=True, mode=0o700)
            # An explicitly created desktop draft needs an identity after an
            # HTTP restart, unlike the TUI's uncommitted welcome-screen draft.
            marker = path / "desktop.json"
            marker.write_text(json.dumps({"version": 1, "cwd": str(directory)}))
            marker.chmod(0o600)

        await asyncio.to_thread(persist)
        return session_id

    async def list(self, limit: int) -> list[dict[str, Any]]:
        def rows() -> list[dict[str, Any]]:
            existing = [row._asdict() for row in recent_session_rows(self.root, limit=limit)]
            ids = {row["id"] for row in existing}
            for marker in (self.root / "sessions").glob("*/desktop.json"):
                if marker.parent.name not in ids and SESSION_ID.fullmatch(marker.parent.name):
                    existing.append(
                        {"id": marker.parent.name, "name": "", "mtime": marker.stat().st_mtime}
                    )
            visible = sorted(existing, key=lambda row: row["mtime"], reverse=True)[:limit]
            # Previews are read AFTER the sort and the limit, so the tail scan
            # runs once per row the caller will actually see rather than once
            # per session on disk. Measured 0.10 ms per row on a 38-session
            # store; the whole call already runs on a worker thread.
            from local_operator.session.attention import AttentionStore

            # One read connection per list, never one per row or an owner bind.
            attention = AttentionStore(self.root / "attention.db").state_many(
                f"session/{row['id']}" for row in visible
            )
            for row in visible:
                row["preview"] = session_preview(self.root / "sessions" / row["id"])
                row["attention"] = attention[f"session/{row['id']}"]
            return visible

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
                    marker = path / "desktop.json"
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

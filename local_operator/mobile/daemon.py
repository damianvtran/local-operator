"""The mobile daemon: one process, the phone-facing control plane.

``lop mobile serve`` runs this. It is deliberately small because the design
moved the hard parts elsewhere: sessions fold themselves into projections
(:mod:`.projection`), discovery is a directory scan (:mod:`.registry`), auth
is a signed cookie (:mod:`.auth`). What remains here is plumbing with three
moving parts:

- **Adoption** — scan the record directory, dial each live registrant's
  control socket with the record's key, and relay its projection pushes into
  an in-memory table the web layer reads. Owned sessions (started from the
  phone) register through the SAME socket path — every phone-visible session
  has one shape, so the web layer never branches on ownership.
- **The web app** — a Starlette application serving the built SPA, a small
  REST surface, and the SSE stream. SSE, never WebSocket, on the phone leg:
  an identity proxy's login redirect cannot be followed by a WebSocket
  handshake, so every realtime push here is an EventSource repaint.
- **Repaint, not deltas** — every push is the whole projection for one
  session. No delta protocol means no drift; caps in the fold keep repaints
  cheap.

Threading: one asyncio loop. Registrants run their own loops in their own
processes; this loop only dials them. Blocking work (session construction,
which reads provider catalogues) goes through ``asyncio.to_thread`` so a
phone starting a session never stalls the SSE streams of the others.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from local_operator.mobile import registry
from local_operator.mobile.auth import (
    COOKIE_NAME,
    check_password,
    sign_cookie,
    verify_cookie,
)
from local_operator.mobile.types import (
    PROTOCOL_VERSION,
    SessionProjection,
    SessionRecord,
)

logger = logging.getLogger(__name__)

#: How often the record directory is scanned. Records change rarely (a
#: session starts or dies); the scan is cheap, and 2 s makes a new terminal
#: session appear on the phone before the user reaches for it.
SCAN_INTERVAL_S = 2.0

#: Backoff before re-dialing a refused control socket. A registrant whose
#: record is fresh but whose socket refuses is mid-restart; hammering it
#: buys nothing.
REDIAL_BACKOFF_S = 5.0

#: SSE keepalive cadence — under the 60 s idle cutoff of common proxies.
SSE_KEEPALIVE_S = 25.0

#: Default daemon port. Loopback only; remote access is a tunnel's job.
DEFAULT_PORT = 4098

_WEB_DIR = Path(__file__).parent / "web"
_DIST_DIR = _WEB_DIR / "dist"
#: The cropped LO mark (figure with a raised hand) served to the login page
#: and, via the same path, to the SPA. Lives next to this module so the
#: login HTML never depends on a Vite rebuild.
_STATIC_DIR = Path(__file__).parent / "static"


def _mark_data_uri() -> str:
    """The mark as a data URI — the login page and the SPA header inline it
    rather than fetch ``/mark.png``, because over an identity-proxied tunnel
    (Cloudflare Access) that fetch is itself gated: the pre-auth login page's
    <img> got a 302-to-IdP HTML body and rendered the broken-image glyph.
    An inline URI needs no request, so it renders behind Access and on the
    unauthenticated login screen alike. 7 KB; one copy in each surface.
    """
    import base64

    data = base64.b64encode((_STATIC_DIR / "mark.png").read_bytes()).decode()
    return "data:image/png;base64," + data


# ---------------------------------------------------------------------------
# Session table
# ---------------------------------------------------------------------------


class SessionEntry:
    """One phone-visible session: its latest projection plus the dial state.

    The projection arrives whole on every push; ``subscribers`` are the open
    SSE queues waiting for repaints of this session.
    """

    def __init__(self, record: SessionRecord) -> None:
        self.record = record
        self.projection: SessionProjection | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.next_dial_at: float = 0.0
        self.degraded = False
        self.ended = False
        # SSE ownership is session-level in SessionTable. A process entry is a
        # replaceable routing generation and must never own a conversation view.
        # Kept as an alias only for compatibility with focused diagnostics.
        self.subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        # Monotonic request id for control frames we originate.
        self._req_seq = 0

    def next_req(self) -> int:
        self._req_seq += 1
        return self._req_seq


class SessionTable:
    """The daemon's whole runtime state. One instance, owned by the loop."""

    def __init__(self) -> None:
        self.entries: dict[int, SessionEntry] = {}  # by pid
        self.list_subscribers: set[asyncio.Queue[None]] = set()
        # Durable conversation identity owns viewers across zero or many host
        # generations. Entries route watch commands but never own these queues.
        self.session_subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self.provisional_active: set[str] = set()

    def summaries(self) -> list[dict[str, Any]]:
        """Reconcile live generations with durable conversations by session id."""
        from local_operator.paths import config_dir
        from local_operator.resume import recent_session_rows

        durable = {row.id: row for row in recent_session_rows(config_dir(), limit=100)}
        active: dict[str, SessionEntry] = {}
        for entry in self.entries.values():
            if entry.ended:
                continue
            prior = active.get(entry.record.session_id)
            if prior is None or entry.record.heartbeat_at > prior.record.heartbeat_at:
                active[entry.record.session_id] = entry
        out: list[dict[str, Any]] = []
        for session_id in set(durable) | set(active):
            entry = active.get(session_id)
            p = entry.projection if entry else None
            row = durable.get(session_id)
            out.append(
                {
                    "session_id": session_id,
                    "section": (
                        "active" if entry or session_id in self.provisional_active else "previous"
                    ),
                    "conversation_name": (p.conversation_name if p else "")
                    or (entry.record.conversation_name if entry else "")
                    or (row.name if row else ""),
                    "cwd": p.cwd if p else (entry.record.cwd if entry else ""),
                    "model_label": (
                        p.model_label if p else (entry.record.model_label if entry else "")
                    ),
                    "streaming": bool(p and p.streaming),
                    "needs_attention": bool(p and p.pending),
                    "pending_kind": p.pending.kind if p and p.pending else "",
                    "subagents_running": sum(
                        1 for subagent in (p.subagents if p else []) if subagent.status == "running"
                    ),
                    "todos_open": sum(
                        1
                        for phase in (p.todos if p else [])
                        for todo in phase.items
                        if todo.status in ("pending", "blocked")
                    ),
                    "mtime": row.mtime if row else entry.record.started_at if entry else 0,
                }
            )
        out.sort(
            key=lambda summary: (
                summary["section"] != "active",
                not summary["needs_attention"],
                not summary["streaming"],
                -summary["mtime"],
                summary["session_id"],
            )
        )
        return out

    def notify_list_changed(self) -> None:
        for queue in self.list_subscribers:
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass


def _entry_for_session(daemon: "MobileDaemon", session_id: str) -> SessionEntry | None:
    """Select the newest live generation without exposing its pid publicly."""
    candidates = [
        entry
        for entry in daemon.table.entries.values()
        if entry.record.session_id == session_id and not entry.ended
    ]
    return max(candidates, key=lambda entry: entry.record.heartbeat_at, default=None)


def _durable_user_session_dir(session_id: str) -> Path | None:
    """Return a strictly addressed durable user conversation, if it exists.

    Mobile routes are public identifiers, not filesystem paths. Checking the
    name before joining prevents traversal and checking the origin marker keeps
    subagent/scheduled transcripts out of the human conversation surface.
    """
    from local_operator.paths import config_dir
    from local_operator.resume import is_user_session

    if session_id in ("", ".", "..") or Path(session_id).name != session_id:
        return None
    directory = config_dir() / "sessions" / session_id
    if not (directory / "transcript.jsonl").is_file() or not is_user_session(directory):
        return None
    return directory


def _durable_projection(session_id: str) -> SessionProjection | None:
    """Fold a user conversation from disk when no execution host exists."""
    from local_operator.mobile.projection import ProjectionFold
    from local_operator.resume import stored_session_title
    from local_operator.session.transcript import Transcript

    directory = _durable_user_session_dir(session_id)
    if directory is None:
        return None
    projection = SessionProjection(
        session_id=session_id,
        pid=0,
        kind="daemon",
        conversation_name=stored_session_title(directory),
        cwd="",
        model_label="",
    )
    fold = ProjectionFold(projection)
    fold.fold_history(Transcript(directory).build_llm_history())
    projection.ended = False
    projection.degraded = False
    return projection


# ---------------------------------------------------------------------------
# Registrant connections
# ---------------------------------------------------------------------------


async def _dial(daemon: "MobileDaemon", entry: SessionEntry) -> None:
    """Open (or re-open) the control socket to one registrant and pump its
    frames until the connection dies. One task per session."""
    record = entry.record
    try:
        # Match the registrant's 1 MB line limit. The default 64 KB
        # StreamReader cap is what made a transcript push raise
        # ValueError and leave the session stuck on "connecting…".
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
    except OSError:
        entry.degraded = True
        entry.next_dial_at = time.monotonic() + REDIAL_BACKOFF_S
        return
    entry.writer = writer
    entry.degraded = False
    try:
        writer.write(json.dumps({"key": record.control_key}).encode() + b"\n")
        await writer.drain()
        while True:
            try:
                line = await reader.readline()
            except ValueError:
                # A frame longer than the stream limit is not a reason to
                # drop the session — a transcript push can outgrow 64 KB.
                # Skip the oversized line and keep the connection.
                logger.warning("mobile daemon: oversized control frame from pid %s", record.pid)
                continue
            if not line:
                break
            try:
                frame = json.loads(line.decode("utf-8", "replace"))
            except ValueError:
                continue
            op = frame.get("op")
            if op in ("projection", "welcome"):
                try:
                    data = frame.get("data") or {}
                    entry.projection = _projection_from_json(data, record)
                except (TypeError, ValueError, KeyError):
                    # A malformed push (mid-upgrade registrant, renamed field)
                    # must not tear the dial loop down to the reconnect path —
                    # the NEXT push is a full repaint that repairs the view.
                    logger.debug("mobile daemon: dropping malformed projection", exc_info=True)
                    continue
                entry.projection.degraded = False
                entry.projection.ended = False
                daemon.capture_subagent_details(entry.projection)
                daemon.session_projections[entry.record.session_id] = entry.projection
                daemon.table.provisional_active.discard(entry.record.session_id)
                daemon.table.notify_list_changed()
                _fan_out(entry, daemon)
            # acks/errors are matched by req id in _request's future map.
            pending = daemon._pending_reqs.pop((record.pid, frame.get("req")), None)
            if pending is not None and not pending.done():
                pending.set_result(frame)
    except (ConnectionResetError, BrokenPipeError, OSError):
        pass
    finally:
        if entry.writer is writer:
            entry.writer = None
        entry.next_dial_at = time.monotonic() + REDIAL_BACKOFF_S
        entry.degraded = not entry.ended
        _fan_out(entry, daemon)


def _projection_from_json(data: dict[str, Any], record: SessionRecord) -> SessionProjection:
    """The daemon's rebuild seam: the shared wire-types rebuild, plus the
    opening-user-message pin (a DAEMON-side repair for older sessions — see
    :func:`_pin_opening_user_message`), which no other consumer wants."""
    from local_operator.mobile.types import _projection_from_json as _rebuild

    projection = _rebuild(data, record)
    _pin_opening_user_message(projection, record)
    return projection


def _pin_opening_user_message(projection: SessionProjection, record: SessionRecord) -> None:
    """Guarantee the transcript opens with the conversation's first user
    message, even when the SESSION that folded it is running older code.

    Two independent gaps hid it: the harness never emits MessageStartEvent
    for user messages (fixed in the handle), and the 80-entry tail cap drops
    the opening prompt on any long session (fixed in the fold). Both fixes
    live in the session's own process — so a session on an older binary
    still pushes a wire projection with no user rows. The daemon can't fix
    the session's fold, but it CAN repair the view: read the opening user
    turn from the on-disk transcript (the same store /resume reads) and pin
    it at the head. Idempotent — a projection that already opens with a user
    row is left alone.
    """
    transcript = projection.transcript
    if any(e.kind == "user" for e in transcript):
        return
    try:
        from local_operator.paths import config_dir

        path = config_dir() / "sessions" / record.session_id / "transcript.jsonl"
        if not path.exists():
            return
        # Read only the head: the opening user turn is within the first few
        # entries, and a 10 MB transcript should not be replayed per repaint.
        import json as _json

        with path.open() as fh:
            # Scan a bounded head, not the whole file: the opening user turn
            # is normally within the first few entries, but if it was pruned
            # or compacted away the first surviving user message can sit
            # arbitrarily deep, and this runs on every projection reload.
            # Give up after MAX_SCAN lines — a session whose opening prompt
            # no longer exists simply has nothing to pin.
            MAX_SCAN = 400
            for i, line in enumerate(fh):
                if i >= MAX_SCAN:
                    return
                try:
                    entry = _json.loads(line)
                except ValueError:
                    continue
                if entry.get("type") != "message":
                    continue
                payload = entry.get("payload") or {}
                if payload.get("role") != "user":
                    continue
                # Transcript text blocks are stored as {"text": ...} WITHOUT a
                # "type" discriminator (the in-memory TextContent adds it), so
                # match on the text key itself rather than a type field.
                text = "".join(
                    block["text"]
                    for block in payload.get("content", [])
                    if isinstance(block, dict) and isinstance(block.get("text"), str)
                )
                if not text.strip():
                    continue
                from local_operator.mobile.types import TranscriptEntry

                projection.transcript = [
                    TranscriptEntry(
                        # The transcript persists message.id as the entry id,
                        # so it is always present — use it, no pid fallback.
                        id=entry["id"],
                        kind="user",
                        text=text,
                        final=True,
                    ),
                    *transcript,
                ]
                return
    except Exception:  # noqa: BLE001 — a missing/odd transcript must never break a repaint
        return


def _transcript_entry_json(entry: Any) -> dict[str, Any]:
    """Serialize one mobile TranscriptEntry for the history payload."""
    return entry.to_json()


def _history_page(
    session_id: str, before: str | None, limit: int, *, durable_only: bool = True
) -> tuple[list[Any], bool]:
    """Fold the session's full on-disk transcript and return the page of
    entries immediately OLDER than ``before`` (chronological within the page)
    plus whether more history exists beyond it.

    Runs off the event loop (``asyncio.to_thread`` at the call site): folding
    a long transcript rehydrates every message and is not loop-safe work.
    """
    from local_operator.mobile.projection import fold_messages_to_entries
    from local_operator.session.transcript import Transcript

    if durable_only:
        directory = _durable_user_session_dir(session_id)
    else:
        # A live SessionEntry already established the route's identity. Keep
        # the pre-existing live behavior (including non-user hosts) while still
        # requiring one safe path component before touching disk.
        from local_operator.paths import config_dir

        directory = (
            config_dir() / "sessions" / session_id
            if session_id not in ("", ".", "..") and Path(session_id).name == session_id
            else None
        )
    if directory is None or not (directory / "transcript.jsonl").is_file():
        return [], False
    try:
        transcript = Transcript(directory)
        history = transcript.build_llm_history()
        entries = fold_messages_to_entries(history)
    except Exception:  # noqa: BLE001 — an odd transcript yields no history, not a 500
        logger.exception("history fold failed for session %s", session_id)
        return [], False

    if before:
        # A ``before`` that resolves to nothing means the client's anchor was
        # pruned (a compaction between scrolls). Serving the newest page then
        # would duplicate the client's live window — return empty and let the
        # client treat it as end-of-history rather than loop on the same rows.
        anchor = next((i for i, e in enumerate(entries) if e.id == before), None)
        if anchor is None:
            return [], False
        cut = anchor
    else:
        cut = len(entries)
    older = entries[:cut]
    page = older[-limit:] if len(older) > limit else older
    has_more = len(older) > len(page)
    return page, has_more


def _image_bytes(record: SessionRecord, entry_id: str, index: int) -> tuple[bytes, str] | None:
    """Decode the ``index``-th image block of message ``entry_id`` from the
    session's on-disk transcript into raw bytes plus mime type.

    Reads from disk (not the live fold) so it serves attachments from history
    the projection tail dropped as well as recent ones, and reuses the
    transcript's own attachment resolution — the same base64 the model saw.
    Returns ``None`` for any miss (unknown message, out-of-range index, a
    reference that no longer resolves) so the caller answers a clean 404.

    Runs off the event loop (``asyncio.to_thread`` at the call site): building
    the history rehydrates every message and is not loop-safe work.
    """
    import base64
    import binascii

    from local_operator.harness.types import ImageContent, Message
    from local_operator.paths import config_dir
    from local_operator.session.transcript import Transcript

    directory = config_dir() / "sessions" / record.session_id
    if not (directory / "transcript.jsonl").exists():
        return None
    try:
        transcript = Transcript(directory)
        history = transcript.build_llm_history()
    except Exception:  # noqa: BLE001 — an odd transcript serves no image, not a 500
        logger.exception("image fetch: history fold failed for %s", record.session_id)
        return None
    message = next((m for m in history if isinstance(m, Message) and m.id == entry_id), None)
    if message is None or not isinstance(message.content, list):
        return None
    images = [b for b in message.content if isinstance(b, ImageContent)]
    # ``index`` is the position among IMAGE blocks (what _image_refs emits),
    # not among all content blocks — text blocks do not count.
    if index < 0 or index >= len(images):
        return None
    data = images[index].data
    if not data:
        return None
    try:
        raw = base64.b64decode(data)
    except (binascii.Error, ValueError):
        logger.warning("image fetch: undecodable base64 for %s[%d]", entry_id, index)
        return None
    return raw, images[index].mime_type or "image/png"


def _fan_out(entry: SessionEntry, daemon: "MobileDaemon | None" = None) -> None:
    """Push a repaint to durable session viewers, never one pid generation."""
    if entry.projection is None:
        return
    frame = entry.projection.to_json()
    queues = (
        daemon.table.session_subscribers.get(entry.record.session_id, set())
        if daemon is not None
        else entry.subscribers
    )
    for queue in queues:
        while True:
            try:
                queue.put_nowait(frame)
                break
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break  # racing consumer drained it; retry the put


# ---------------------------------------------------------------------------
# The daemon
# ---------------------------------------------------------------------------


class MobileDaemon:
    def __init__(self, *, port: int = DEFAULT_PORT, password: str | None = None) -> None:
        self.port = port
        self.password = password
        self.table = SessionTable()
        # The session repaint carries only roster summaries. Full child state is
        # retained separately and fetched for the active route, otherwise one
        # busy descendant makes every root token repaint resend every transcript.
        # Ordered dict insertion order supplies a simple bounded oldest-session
        # eviction without putting another cache dependency on the daemon path.
        self.subagent_details: dict[tuple[str, str], dict[str, Any]] = {}
        self._subagent_detail_sessions: list[str] = []
        self._pending_reqs: dict[tuple[int, Any], asyncio.Future[dict[str, Any]]] = {}
        self._dial_tasks: dict[int, asyncio.Task[None]] = {}
        self._slash_commands: list[dict[str, Any]] | None = None
        # Session id -> pid of a resume spawn already in flight, so a retried
        # resume POST returns the same child instead of forking a second.
        self.resumes_in_flight: dict[str, int] = {}
        # A session route outlives every process generation. Retain its latest
        # repaint so an open phone remains a normal conversation while idle.
        self.session_projections: dict[str, SessionProjection] = {}
        self._wake_settle_tasks: dict[str, asyncio.Task[None]] = {}

    def capture_subagent_details(self, projection: SessionProjection) -> None:
        """Move descendant-only state out of the aggregate repaint.

        Keeping the legacy fields on ``SubagentRow`` makes mixed-version
        registrants additive: an old host can still send the aggregate shape,
        while every browser receives lightweight summaries and asks for only
        the selected child. The cache is bounded by root sessions because a
        phone can navigate only within one root lineage at a time.
        """
        session_id = projection.session_id
        if session_id not in self._subagent_detail_sessions:
            self._subagent_detail_sessions.append(session_id)
        while len(self._subagent_detail_sessions) > 16:
            expired = self._subagent_detail_sessions.pop(0)
            for key in [key for key in self.subagent_details if key[0] == expired]:
                self.subagent_details.pop(key, None)
        for row in projection.subagents[:256]:
            detail = row.to_json()
            detail["version"] = projection.version
            self.subagent_details[(session_id, row.job_id)] = detail
            # The aggregate needs enough to paint and route rows, not the launch
            # prompt or terminal payload. Those can each be many kilobytes and
            # belong to selected detail exactly like transcript and todos.
            row.prompt = ""
            row.result_text = ""
            row.error_text = ""
            row.transcript = []
            row.todos = []

    # -- scanning --------------------------------------------------------------

    async def scan_loop(self) -> None:
        while True:
            try:
                await self._scan_once()
            except Exception:  # noqa: BLE001 — the scan must never die
                logger.warning("mobile daemon scan failed", exc_info=True)
            await asyncio.sleep(SCAN_INTERVAL_S)

    async def _scan_once(self) -> None:
        seen: set[int] = set()
        changed = False
        for record, state in await asyncio.to_thread(registry.scan):
            seen.add(record.pid)
            entry = self.table.entries.get(record.pid)
            if entry is None:
                entry = SessionEntry(record)
                self.table.entries[record.pid] = entry
                changed = True
                # Durable viewers predate this process generation after a
                # Previous wake. Rebind only the reaper signal to the new host;
                # the SSE queues remain owned by session_id in the daemon.
                if self.table.session_subscribers.get(record.session_id):
                    self.notify_watch_transition(record.pid, watching=True)
            else:
                # Re-adopt record updates (model label, name, /resume's new
                # session id) — the socket survives them by design.
                entry.record = record
            if state == "stale":
                entry.ended = True
                changed = True
                # SIGKILL cannot run owner cleanup. Discovery already proved the
                # record pid dead; the lease helper revalidates generation and
                # process identity under the recovery lock before removing only
                # that claim and its pid mirror. Transcript data is untouched.
                from local_operator.paths import config_dir
                from local_operator.session_lease import reap_proven_dead_session_claim

                await asyncio.to_thread(
                    reap_proven_dead_session_claim,
                    config_dir() / "sessions" / record.session_id,
                    record.pid,
                )
                self.table.provisional_active.discard(record.session_id)
                projection = await asyncio.to_thread(_durable_projection, record.session_id)
                if projection is not None:
                    self.session_projections[record.session_id] = projection
                    for queue in self.table.session_subscribers.get(record.session_id, set()):
                        try:
                            queue.put_nowait(projection.to_json())
                        except asyncio.QueueFull:
                            try:
                                queue.get_nowait()
                                queue.put_nowait(projection.to_json())
                            except asyncio.QueueEmpty:
                                pass
            elif state == "wedged":
                entry.degraded = True
            # Degraded is precisely "we owe this session a redial" — the only
            # gates are ended, an open socket, and the backoff clock. Excluding
            # degraded entries here was the starvation bug: one refused dial
            # meant never trying again.
            if not entry.ended and entry.writer is None:
                if time.monotonic() >= entry.next_dial_at and (
                    record.pid not in self._dial_tasks or self._dial_tasks[record.pid].done()
                ):
                    self._dial_tasks[record.pid] = asyncio.ensure_future(_dial(self, entry))
        # Reap entries whose record vanished entirely.
        for pid in list(self.table.entries):
            if pid not in seen:
                entry = self.table.entries[pid]
                if not entry.ended:
                    entry.ended = True
                    changed = True
                    session_id = entry.record.session_id
                    self.table.provisional_active.discard(session_id)
                    projection = await asyncio.to_thread(_durable_projection, session_id)
                    if projection is not None:
                        self.session_projections[session_id] = projection
                        for queue in self.table.session_subscribers.get(session_id, set()):
                            try:
                                queue.put_nowait(projection.to_json())
                            except asyncio.QueueFull:
                                try:
                                    queue.get_nowait()
                                    queue.put_nowait(projection.to_json())
                                except asyncio.QueueEmpty:
                                    pass
        if changed:
            self.table.notify_list_changed()

    def retain_provisional_active(self, session_id: str) -> None:
        """Hold a wake transition until discovery or one scan interval wins."""
        previous = self._wake_settle_tasks.pop(session_id, None)
        if previous is not None:
            previous.cancel()

        async def settle() -> None:
            try:
                await asyncio.sleep(SCAN_INTERVAL_S)
                await self._scan_once()
                if _entry_for_session(self, session_id) is None:
                    self.table.provisional_active.discard(session_id)
                    projection = await asyncio.to_thread(_durable_projection, session_id)
                    if projection is not None:
                        self.session_projections[session_id] = projection
                        for queue in self.table.session_subscribers.get(session_id, set()):
                            try:
                                queue.put_nowait(projection.to_json())
                            except asyncio.QueueFull:
                                pass
                    self.table.notify_list_changed()
            finally:
                self._wake_settle_tasks.pop(session_id, None)

        self._wake_settle_tasks[session_id] = asyncio.create_task(settle())

    # -- control requests ---------------------------------------------------------

    def notify_watch_transition(self, pid: int, *, watching: bool) -> None:
        """Push watch/unwatch to a session when its phone SSE subscriber
        count crosses 0 <-> N.

        Scheduled, not awaited: the SSE handshake must not block on a slow
        (or old, op-rejecting) registrant. The fire-and-forget task rides the
        daemon's loop; errors are swallowed at the task boundary — an OLD
        registrant's `error: unknown op` reply arrives as a RuntimeError from
        ``request`` and is expected during rolling upgrades."""

        async def send() -> None:
            try:
                await self.request(pid, "watch" if watching else "unwatch")
            except (RuntimeError, TimeoutError, KeyError, asyncio.CancelledError):
                # KeyError: no dial yet (the SSE stream can open before the
                # control connection is established). RuntimeError: old
                # registrant or op rejected. Both are fine — the session's
                # watch_supported latch stays unlatched and its reaper (if
                # any) stays inert, which is the safe direction.
                logger.debug("watch push to pid %s skipped (%s)", pid, watching)

        try:
            asyncio.get_running_loop().create_task(send())
        except RuntimeError:  # no loop (tests constructing the daemon directly)
            pass

    async def request(self, pid: int, op: str, **fields: Any) -> dict[str, Any]:
        """Send one control frame to a session and await its ack/error."""
        entry = self.table.entries.get(pid)
        if entry is None or entry.writer is None:
            raise KeyError(f"session {pid} is not connected")
        req = entry.next_req()
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_reqs[(pid, req)] = future
        frame = {"op": op, "req": req, **fields}
        try:
            entry.writer.write(json.dumps(frame).encode() + b"\n")
            await entry.writer.drain()
            reply = await asyncio.wait_for(future, timeout=15.0)
        finally:
            self._pending_reqs.pop((pid, req), None)
        if reply.get("op") == "error":
            raise RuntimeError(str(reply.get("message", "request failed")))
        return reply

    # -- owned sessions ---------------------------------------------------------

    async def spawn_session(
        self,
        cwd: str,
        provider: str | None = None,
        model_id: str | None = None,
        resume: str | None = None,
    ) -> int:
        """Spawn a daemon-owned session in a supervised CHILD process and let
        discovery adopt it.

        A child process, not an in-process session: the daemon is supervisable
        state (launchd restarts it), and a session living inside it would die
        with every restart — taking an in-flight turn with it. A child with
        its own pid gets the same lifetime as a terminal session: the daemon
        going away costs the phone its view, never the session its work. The
        child runs the registrant standalone (``python -m
        local_operator.mobile.child``), so the record + control socket path is
        literally the same code the TUI uses.
        """
        import sys

        env = dict(os.environ)
        env["LOP_MOBILE_CHILD_CWD"] = cwd
        if provider:
            env["LOP_MOBILE_CHILD_PROVIDER"] = provider
        if model_id:
            env["LOP_MOBILE_CHILD_MODEL"] = model_id
        if resume:
            env["LOP_MOBILE_CHILD_RESUME"] = resume
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "local_operator.mobile.child",
            env=env,
            # Detached stdio: the child speaks through its record and socket;
            # a pipe back to the daemon would die with the daemon and take
            # the child's stdout with it.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return process.pid

    # -- slash command catalogue ----------------------------------------------------

    def slash_commands(self) -> list[dict[str, Any]]:
        """The phone's slash sheet. Imported lazily (the TUI registry pulls
        the app's command table) and cached — the registry is static."""
        if self._slash_commands is None:
            from local_operator.tui.app import SLASH_COMMANDS
            from local_operator.tui.autocomplete import ArgumentMode

            excluded = {"exit", "quit", "clear"}  # TUI chrome, meaningless on a phone
            self._slash_commands = [
                {
                    "name": cmd.name,
                    "description": cmd.description,
                    "aliases": list(cmd.aliases),
                    "arguments": (
                        cmd.arguments.name.lower()
                        if isinstance(cmd.arguments, ArgumentMode)
                        else "none"
                    ),
                }
                for cmd in SLASH_COMMANDS
                if cmd.name not in excluded
            ]
        return self._slash_commands


# ---------------------------------------------------------------------------
# Web application
# ---------------------------------------------------------------------------


def build_app(daemon: MobileDaemon):
    """The Starlette app. Imported inside the function so ``lop`` without the
    server extra never pays for starlette at startup."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import (
        FileResponse,
        HTMLResponse,
        JSONResponse,
        PlainTextResponse,
        RedirectResponse,
        Response,
        StreamingResponse,
    )
    from starlette.routing import BaseRoute, Mount, Route

    # -- auth helpers -----------------------------------------------------------

    def authed(request: Request) -> bool:
        if not daemon.password:
            return False
        return verify_cookie(request.cookies.get(COOKIE_NAME), daemon.password)

    def gate(request: Request) -> Response | None:
        """None = allowed. Browsers get the login redirect, API calls a 401 —
        the split contract the health check asserts."""
        if authed(request):
            return None
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "authentication required"}, status_code=401)
        return RedirectResponse("/login", status_code=303)

    def secure_cookie(response: Response, request: Request) -> None:
        # Secure only when the request arrived over TLS (the tunnel case);
        # plain loopback HTTP must still set the cookie or first-run dev is
        # impossible. HttpOnly + SameSite=Lax always: the cookie is never
        # read from JS and never rides a cross-site POST.
        response.headers["Cache-Control"] = "no-store"
        return None

    # -- routes -------------------------------------------------------------------

    async def healthz(request: Request) -> Response:
        return JSONResponse(
            {
                "ok": True,
                "version": PROTOCOL_VERSION,
                "sessions": len(daemon.table.entries),
                "dist": _DIST_DIR.exists(),
            }
        )

    async def login_page(request: Request) -> Response:
        if authed(request):
            return RedirectResponse("/", status_code=303)
        return HTMLResponse(_LOGIN_HTML.replace("__MARK_DATA_URI__", _mark_data_uri()))

    async def login_submit(request: Request) -> Response:
        form = await request.form()
        candidate = str(form.get("password", ""))
        if not daemon.password or not check_password(candidate, daemon.password):
            return HTMLResponse(
                _LOGIN_HTML.replace("__MARK_DATA_URI__", _mark_data_uri()).replace(
                    "<!--ERROR-->", _LOGIN_ERROR
                ),
                status_code=401,
            )
        response = RedirectResponse("/", status_code=303)
        secure = request.url.scheme == "https"
        response.set_cookie(
            COOKIE_NAME,
            sign_cookie(daemon.password),
            max_age=30 * 24 * 3600,
            httponly=True,
            samesite="lax",
            secure=secure,
        )
        return response

    async def logout(request: Request) -> Response:
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE_NAME)
        return response

    async def mark_png(request: Request) -> Response:
        """The LO mark — unauthenticated because the login page needs it
        before a cookie exists. It is a public brand asset, not a secret."""
        path = _STATIC_DIR / "mark.png"
        if not path.exists():
            return PlainTextResponse("mark missing", status_code=404)
        # no-store: a phone that loaded this while the wheel lacked the file
        # cached the 404 and kept showing a broken image after the fix. The
        # asset is tiny; the freshness guarantee is worth more than the cache.
        response = FileResponse(path, media_type="image/png")
        response.headers["Cache-Control"] = "no-store"
        return response

    async def index(request: Request) -> Response:
        denied = gate(request)
        if denied is not None:
            return denied
        if not _DIST_DIR.exists():
            return PlainTextResponse(
                "mobile web bundle not built — run: "
                "cd local_operator/mobile/web && pnpm install && pnpm build",
                status_code=503,
            )
        response = FileResponse(_DIST_DIR / "index.html")
        response.headers["Cache-Control"] = "no-store"  # the SPA shell; assets are hashed
        return response

    async def api_sessions(request: Request) -> Response:
        denied = gate(request)
        if denied is not None:
            return denied
        return JSONResponse({"sessions": daemon.table.summaries()})

    async def api_session_events(request: Request) -> Response:
        """SSE repaint stream for one session — the phone's only realtime
        channel. Opens with the current projection so a reconnecting phone
        renders immediately."""
        denied = gate(request)
        if denied is not None:
            return denied
        session_id = str(request.path_params["session_id"])
        entry = _entry_for_session(daemon, session_id)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=8)
        subscribers = daemon.table.session_subscribers.setdefault(session_id, set())
        first_watcher = not subscribers
        subscribers.add(queue)
        if entry is not None and first_watcher:
            daemon.notify_watch_transition(entry.record.pid, watching=True)
        # First subscriber = a phone just started watching: tell the session so
        # its self-reaper (if any) counts this front end and holds the session
        # in ACTIVE. Fire-and-forget on the already-open dial writer; an OLD
        # registrant answers `error: unknown op` and that must not 500 the SSE
        # handshake — RuntimeError/TimeoutError are swallowed by design.

        async def stream():
            try:
                live = _entry_for_session(daemon, session_id)
                projection = live.projection if live is not None else None
                if projection is None:
                    projection = await asyncio.to_thread(_durable_projection, session_id)
                    if projection is not None:
                        daemon.session_projections[session_id] = projection
                if projection is not None:
                    yield _sse("projection", projection.to_json())
                while True:
                    try:
                        frame = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_S)
                        yield _sse("projection", frame)
                    except TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                current = daemon.table.session_subscribers.get(session_id)
                if current is not None:
                    current.discard(queue)
                    if not current:
                        daemon.table.session_subscribers.pop(session_id, None)
                        live = _entry_for_session(daemon, session_id)
                        if live is not None:
                            daemon.notify_watch_transition(live.record.pid, watching=False)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                # Identity proxies buffer SSE by default; this is the header
                # that turns buffering off at nginx-family proxies.
                "X-Accel-Buffering": "no",
            },
        )

    async def api_list_events(request: Request) -> Response:
        """SSE for the session LIST, so the home screen needs no polling."""
        denied = gate(request)
        if denied is not None:
            return denied
        queue: asyncio.Queue[None] = asyncio.Queue(maxsize=4)
        daemon.table.list_subscribers.add(queue)

        async def stream():
            try:
                yield _sse("sessions", {"sessions": daemon.table.summaries()})
                while True:
                    try:
                        await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_S)
                        yield _sse("sessions", {"sessions": daemon.table.summaries()})
                    except TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                daemon.table.list_subscribers.discard(queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    async def api_subagent_detail(request: Request) -> Response:
        """Full state for the one descendant named by the active phone route."""
        denied = gate(request)
        if denied is not None:
            return denied
        session_id = str(request.path_params["session_id"])
        job_id = str(request.path_params["job_id"])
        detail = daemon.subagent_details.get((session_id, job_id))
        if detail is None:
            return JSONResponse({"error": "unknown subagent"}, status_code=404)
        return JSONResponse(detail)

    async def api_subagent_history(request: Request) -> Response:
        """Page one child's transcript, never the root conversation.

        The selected detail proves the child session id belongs to this root
        lineage before the route reaches disk. That isolation matters because
        child transcripts intentionally do not qualify as public root routes.
        """
        denied = gate(request)
        if denied is not None:
            return denied
        session_id = str(request.path_params["session_id"])
        job_id = str(request.path_params["job_id"])
        detail = daemon.subagent_details.get((session_id, job_id))
        child_session_id = detail.get("session_id") if detail else None
        if not isinstance(child_session_id, str) or not child_session_id:
            return JSONResponse({"error": "subagent history unavailable"}, status_code=404)
        before = request.query_params.get("before")
        try:
            limit = max(1, min(int(request.query_params.get("limit", "80")), 200))
        except ValueError:
            limit = 80
        page, has_more = await asyncio.to_thread(
            _history_page, child_session_id, before, limit, durable_only=False
        )
        return JSONResponse(
            {"entries": [_transcript_entry_json(entry) for entry in page], "has_more": has_more}
        )

    async def api_session_history(request: Request) -> Response:
        """Older transcript entries for lazy loading.

        The live projection (SSE) is a tail WINDOW — the fold caps it, so a
        long session's older messages never reach the phone. This endpoint
        folds the session's FULL on-disk transcript with the same render
        semantics and serves the pages the cap dropped, so scrolling up
        back-fills history. ``before`` is the id of the oldest entry the
        phone already has; the response is the page of entries immediately
        OLDER than it (chronological within the page).
        """
        denied = gate(request)
        if denied is not None:
            return denied
        session_id = str(request.path_params["session_id"])
        entry = _entry_for_session(daemon, session_id)
        # A host generation is optional for reads: Previous conversations keep
        # the same public route and page directly from their durable transcript.
        # Live sessions retain the existing eligibility path; durable-only
        # routes must prove they are user sessions before any filesystem read.
        if entry is None and _durable_user_session_dir(session_id) is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        before = request.query_params.get("before")
        try:
            limit = max(1, min(int(request.query_params.get("limit", "80")), 200))
        except ValueError:
            limit = 80
        page, has_more = await asyncio.to_thread(
            _history_page, session_id, before, limit, durable_only=entry is None
        )
        return JSONResponse(
            {
                "entries": [_transcript_entry_json(e) for e in page],
                "has_more": has_more,
            }
        )

    async def api_session_image(request: Request) -> Response:
        """One image attachment's bytes, fetched lazily by the transcript.

        The projection carries only lightweight image REFERENCES (entry id +
        block index + mime) so a per-token repaint stays small; the pixels are
        served here on demand. The bytes come from the on-disk transcript
        (which resolves the attachment store back to inline base64), so this
        works for history the live fold long dropped as well as the tail.

        Cacheable and immutable: the true content key is the ``entry`` id — a
        globally-unique message uuid — plus the image-only ``i``. The ``pid``
        in the path only routes to a live session; pids recycle, but a
        recycled pid maps to a DIFFERENT session whose transcript does not
        contain this message uuid, so it 404s rather than serving another
        session's cached bytes. The uuid content key is what makes ``immutable``
        safe despite the mutable pid in the URL.
        """
        denied = gate(request)
        if denied is not None:
            return denied
        session_id = str(request.path_params["session_id"])
        entry = _entry_for_session(daemon, session_id)
        if entry is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        entry_id = request.query_params.get("entry", "")
        try:
            index = int(request.query_params.get("i", "0"))
        except ValueError:
            return JSONResponse({"error": "bad image index"}, status_code=400)
        if not entry_id:
            return JSONResponse({"error": "entry id is required"}, status_code=400)
        found = await asyncio.to_thread(_image_bytes, entry.record, entry_id, index)
        if found is None:
            return JSONResponse({"error": "no such image"}, status_code=404)
        data, mime_type = found
        return Response(
            content=data,
            media_type=mime_type,
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    async def api_command(request: Request) -> Response:
        """The one mutation endpoint: {op, ...} → control frame. Keeping
        mutations on one route mirrors the registrant's dispatch and keeps
        the auth gate in one place."""
        denied = gate(request)
        if denied is not None:
            return denied
        session_id = str(request.path_params["session_id"])
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "request body must be an object"}, status_code=400)
        body = dict(body)
        op = body.pop("op", None)
        if not isinstance(op, str) or not op:
            return JSONResponse({"error": "op must be a non-empty string"}, status_code=422)
        try:
            from local_operator.mobile.types import (
                ContinuationCommand,
                validate_control_frame,
            )

            entry = _entry_for_session(daemon, session_id)
            if op == "prompt" and entry is None and _durable_user_session_dir(session_id) is None:
                raise KeyError(session_id)
            validate_control_frame({"op": op, "session_id": session_id, **body})
            # HTTP is a reconnectable producer boundary, so prompt/steer identity
            # is mandatory even though protocol-v2 loopback clients remain valid.
            if op in ("prompt", "steer"):
                ContinuationCommand.from_json(
                    {**body, "session_id": session_id, "images": body.get("images", [])}
                )
            if op == "prompt" and entry is None:
                # Only an existing durable user conversation may wake a host.
                # Besides authorization, this prevents a malformed/unknown id
                # from spawning a child that can never own a transcript.
                if _durable_user_session_dir(session_id) is None:
                    raise KeyError(session_id)
                from local_operator.mobile.attach_client import continue_command

                command = ContinuationCommand.from_json(
                    {**body, "session_id": session_id, "images": body.get("images", [])}
                )
                from local_operator.paths import config_dir

                # Publish the accepted wake intent before process discovery. It
                # remains authoritative until a live projection arrives or the
                # attempt fails, so even a 50 ms worker is observable in list SSE.
                daemon.table.provisional_active.add(session_id)
                daemon.table.notify_list_changed()
                projection = daemon.session_projections.get(session_id) or _durable_projection(
                    session_id
                )
                if projection is not None:
                    projection.ended = False
                    projection.degraded = False
                    daemon.session_projections[session_id] = projection
                    for target in daemon.table.session_subscribers.get(session_id, set()):
                        target.put_nowait(projection.to_json())
                try:
                    client, detail = await continue_command(config_dir(), command)
                except BaseException:
                    daemon.table.provisional_active.discard(session_id)
                    daemon.table.notify_list_changed()
                    raise
                client.close()
                daemon.retain_provisional_active(session_id)
                return JSONResponse({"ok": True, "detail": detail})
            if entry is None:
                raise KeyError(session_id)
            reply = await daemon.request(entry.record.pid, op, **body)
        except KeyError:
            return JSONResponse({"error": "session not connected"}, status_code=409)
        except TimeoutError:
            return JSONResponse({"error": "session did not answer"}, status_code=504)
        except (ConnectionError, OSError) as exc:
            # Child construction and daemon/socket failures are transport
            # failures. The web composer maps every non-2xx continuation reply
            # to its stable retry message while retaining the original command.
            return JSONResponse({"error": str(exc)[:200]}, status_code=502)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        return JSONResponse({"ok": True, "detail": reply.get("detail", "")})

    async def api_commands(request: Request) -> Response:
        denied = gate(request)
        if denied is not None:
            return denied
        return JSONResponse({"commands": daemon.slash_commands()})

    async def api_start_session(request: Request) -> Response:
        """Start a daemon-owned session in ``cwd`` and register it through
        the normal record+socket path, so it is indistinguishable from a
        terminal session to the web layer."""
        denied = gate(request)
        if denied is not None:
            return denied
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        cwd_raw = str(body.get("cwd") or Path.home())
        # Resolve to a real directory the picker is allowed to open: anywhere
        # under the owner's home, OR the system temp dir. The spawn runs with
        # the daemon's own environment (it is the owner's account either way),
        # so the check guards against fat-fingered/traversed input, not trust
        # — /tmp is a deliberate, common scratch root the phone offers as a
        # starting directory, so it is on the allowlist beside home.
        cwd_path = Path(cwd_raw).expanduser().resolve()
        if not cwd_path.is_dir() or not _spawn_dir_allowed(cwd_path):
            return JSONResponse(
                {"error": f"not an allowed start directory: {cwd_raw}"}, status_code=400
            )
        cwd = str(cwd_path)
        provider = body.get("provider")
        model_id = body.get("model_id")
        try:
            pid = await daemon.spawn_session(
                cwd,
                provider=str(provider) if provider else None,
                model_id=str(model_id) if model_id else None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("mobile session spawn failed", exc_info=True)
            return JSONResponse({"error": str(exc)[:300]}, status_code=500)
        return JSONResponse({"ok": True, "pid": pid})

    async def api_resume_session(request: Request) -> Response:
        """Reopen a past session as a NEW live session the phone can attach to.

        The old past-sessions flow made the user copy an id and run
        ``/resume <id>`` by hand. This is the button: spawn a daemon-owned
        child whose session resumes that transcript (the same ``--resume``
        mechanism the CLI uses), so the conversation comes back live, open,
        and able to take a command. The new session registers through
        discovery like any other; the phone navigates to it by pid.
        """
        denied = gate(request)
        if denied is not None:
            return denied
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        session_id = str(body.get("session_id") or "").strip()
        if not session_id:
            return JSONResponse({"error": "session_id is required"}, status_code=400)
        # Resolve to a real resumable directory first — spawning a child on a
        # bad id would exit with an unhelpful construction failure.
        from local_operator.paths import config_dir
        from local_operator.resume import ResumeNotFound, resume_dir

        try:
            resume_dir(config_dir(), session_id)
        except ResumeNotFound:
            return JSONResponse({"error": f"no such past session: {session_id}"}, status_code=404)
        # Server-side idempotency: a flapping phone on a slow tunnel can retry
        # the POST, and only the client guarded the double-tap. One in-flight
        # resume per session id — a retry returns the SAME spawn's pid instead
        # of forking a second child resuming the same conversation.
        existing = daemon.resumes_in_flight.get(session_id)
        if existing is not None:
            return JSONResponse({"ok": True, "pid": existing, "session_id": session_id})
        # The transcript dir does not reliably record a cwd, so resume in the
        # owner's home: always a valid directory under the spawn gate. The
        # user can steer the reopened session to a directory from there.
        try:
            pid = await daemon.spawn_session(str(Path.home()), resume=session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mobile resume spawn failed", exc_info=True)
            return JSONResponse({"error": str(exc)[:300]}, status_code=500)
        daemon.resumes_in_flight[session_id] = pid
        return JSONResponse({"ok": True, "pid": pid, "session_id": session_id})

    async def api_search_sessions(request: Request) -> Response:
        """Search past sessions by name, id, OR what was said in them.

        The same mechanism the TUI's /resume picker uses: a cached digest per
        session (search_index.build_index, re-digested only when a transcript
        changes) plus a substring match over name/id (filter_rows semantics).
        A row that matched only on its conversation body is marked so the
        phone can say why it surfaced.
        """
        denied = gate(request)
        if denied is not None:
            return denied
        query = request.query_params.get("q", "")
        try:
            limit = max(1, min(int(request.query_params.get("limit", "40")), 200))
        except ValueError:
            limit = 40
        rows = await asyncio.to_thread(_search_sessions, query, limit)
        return JSONResponse({"sessions": rows, "query": query})

    async def api_directories(request: Request) -> Response:
        """The new-session form's cwd picker: home plus the directories of
        recent sessions (where the user has been working lately)."""
        denied = gate(request)
        if denied is not None:
            return denied
        recent = await asyncio.to_thread(_recent_directories)
        # ``tmp`` is offered as an explicit scratch start dir beside home and
        # the recents — the spawn gate admits it (see _spawn_dir_allowed).
        return JSONResponse({"home": str(Path.home()), "recent": recent, "tmp": _tmp_dir()})

    async def api_past_sessions(request: Request) -> Response:
        """Resumable past sessions — the phone's "go back to a conversation"
        list, from the same store the TUI's /resume picker reads."""
        denied = gate(request)
        if denied is not None:
            return denied
        rows = await asyncio.to_thread(_past_sessions)
        return JSONResponse({"sessions": rows})

    async def api_models(request: Request) -> Response:
        """The model sheet's catalogue: providers with stored credentials and
        their models, reusing the server's model listing so the phone and the
        desktop see the same inventory."""
        denied = gate(request)
        if denied is not None:
            return denied
        try:
            models = await asyncio.to_thread(_list_models)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": str(exc)[:200]}, status_code=502)
        return JSONResponse({"models": models})

    routes: list[BaseRoute] = [
        Route("/healthz", healthz),
        Route("/login", login_page, methods=["GET"]),
        Route("/login", login_submit, methods=["POST"]),
        Route("/logout", logout),
        Route("/api/sessions", api_sessions),
        Route("/api/sessions/start", api_start_session, methods=["POST"]),
        Route("/api/sessions/events", api_list_events),
        Route("/api/directories", api_directories),
        Route("/api/sessions/past", api_past_sessions),
        Route("/api/sessions/resume", api_resume_session, methods=["POST"]),
        Route("/api/sessions/search", api_search_sessions),
        Route("/api/sessions/{session_id:str}/events", api_session_events),
        Route("/api/sessions/{session_id:str}/agents/{job_id:str}", api_subagent_detail),
        Route(
            "/api/sessions/{session_id:str}/agents/{job_id:str}/history",
            api_subagent_history,
        ),
        Route("/api/sessions/{session_id:str}/history", api_session_history),
        Route("/api/sessions/{session_id:str}/image", api_session_image),
        Route("/api/sessions/{session_id:str}/command", api_command, methods=["POST"]),
        Route("/api/commands", api_commands),
        Route("/api/models", api_models),
        Route("/mark.png", mark_png),
        Route("/", index),
    ]
    if _DIST_DIR.exists():
        # The mount is resolved at app build time: a rebuilt bundle needs
        # `lop mobile restart` to appear, which is the documented upgrade
        # path — per-request checks would slow every asset hit to catch a
        # once-per-upgrade event.
        routes.append(
            Mount(
                "/assets",
                app=__import__("starlette.staticfiles", fromlist=["StaticFiles"]).StaticFiles(
                    directory=_DIST_DIR / "assets"
                ),
                name="assets",
            )
        )
    return Starlette(routes=routes)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _past_sessions(limit: int = 20) -> list[dict[str, Any]]:
    """Resumable past sessions for the phone's history list."""
    try:
        from local_operator.paths import config_dir
        from local_operator.resume import recent_session_rows

        return [
            {"id": row.id, "name": row.name, "mtime": row.mtime}
            for row in recent_session_rows(config_dir(), limit=limit)
        ]
    except Exception:  # noqa: BLE001
        return []


def _search_sessions(query: str, limit: int = 40) -> list[dict[str, Any]]:
    """Past sessions matching ``query`` by name, id, or conversation body.

    Mirrors the TUI picker's two channels: a name/id substring match, and a
    body match through the cached search index (re-digested only for
    transcripts that changed). A row that matched ONLY on its body is marked
    ``body_match`` so the UI can say why it surfaced — otherwise it reads as a
    result the filter had no reason to return.
    """
    from local_operator.paths import config_dir
    from local_operator.resume import recent_session_rows
    from local_operator.session.search_index import build_index, search_digests

    cfg = config_dir()
    rows = recent_session_rows(cfg, limit=200)
    needle = query.strip().lower()
    if not needle:
        return [
            {"id": r.id, "name": r.name, "mtime": r.mtime, "body_match": False}
            for r in rows[:limit]
        ]
    try:
        digests = build_index(cfg, [r.id for r in rows])
        body_hits = search_digests(digests, needle)
    except Exception:  # noqa: BLE001 — a broken index degrades to name/id only
        body_hits = set()
    out = []
    for r in rows:
        name_hit = needle in r.name.lower() or needle in r.id.lower()
        body_hit = r.id in body_hits
        if not (name_hit or body_hit):
            continue
        out.append(
            {
                "id": r.id,
                "name": r.name,
                "mtime": r.mtime,
                # Marked only when the name/id did NOT explain the match.
                "body_match": body_hit and not name_hit,
            }
        )
        if len(out) >= limit:
            break
    return out


def _tmp_dir() -> str:
    """The system temp directory, resolved. Offered as a scratch start dir on
    the phone's new-session form and admitted by the spawn gate. Resolved (not
    the raw ``/tmp``) so it matches the gate's resolved comparison on hosts
    where ``/tmp`` is a symlink (macOS: ``/private/tmp``)."""
    import tempfile

    return str(Path(tempfile.gettempdir()).resolve())


def _spawn_dir_allowed(cwd_path: Path) -> bool:
    """Whether a resolved directory may host a phone-started session: anywhere
    under the owner's home, or the system temp dir (a common scratch root).
    Both bounds are on RESOLVED paths so a symlinked ``/tmp`` still matches."""
    home = Path.home().resolve()
    if cwd_path == home or home in cwd_path.parents:
        return True
    tmp = Path(_tmp_dir())
    return cwd_path == tmp or tmp in cwd_path.parents


def _recent_directories(limit: int = 8) -> list[str]:
    """The new-session form's cwd suggestions: the working directories of
    recently active agents from the on-disk registry (the durable store both
    the CLI and server write), deduped, live directories only."""
    try:
        from local_operator.agents import AgentRegistry
        from local_operator.paths import config_dir

        registry = AgentRegistry(config_dir=config_dir())
        agents = sorted(
            registry.list_agents(),
            key=lambda agent: agent.last_message_datetime or "",
            reverse=True,
        )
        seen: list[str] = []
        for agent in agents:
            cwd = agent.current_working_directory
            if cwd and cwd not in seen and Path(cwd).is_dir():
                seen.append(cwd)
            if len(seen) >= limit:
                break
        return seen
    except Exception:  # noqa: BLE001
        return []


def _list_models() -> list[dict[str, Any]]:
    """The model sheet's rows: every model of every provider the owner can
    actually call — a provider with no stored credential is clutter in a
    picker. Credential detection consults BOTH stores, because the two
    sanctioned flows write different ones: ``lop credential update`` writes
    the legacy CredentialManager file, and ``/login`` writes the providers
    AuthStore (auth.db) — a picker reading only the first would hide every
    OAuth-logged-in provider, which on a current install is most of them.
    Runs in a thread: catalogue reads are file I/O."""
    from local_operator.credentials import CredentialManager
    from local_operator.model.registry import SupportedHostingProviders, static_models
    from local_operator.paths import config_dir
    from local_operator.providers.auth_store import AuthStore

    credential_manager = CredentialManager(config_dir=config_dir())
    store = AuthStore()
    try:
        authed_providers = {c.provider for c in store.list_credentials()}
    finally:
        store.close()

    rows: list[dict[str, Any]] = []
    for provider in SupportedHostingProviders:
        required = provider.requiredCredentials
        has_key = bool(required) and any(
            credential_manager.get_credential(key).get_secret_value() for key in required
        )
        # AuthStore login aliases: the oauth flavour of a provider logs in
        # under its own id (e.g. ``alibaba-token-plan-oauth``); the catalogue
        # key is the base id, so prefix matching covers both spellings.
        has_login = provider.id in authed_providers or any(
            p.startswith(f"{provider.id}-") for p in authed_providers
        )
        if required and not has_key and not has_login:
            continue
        if not required and not has_login:
            continue
        for model_id, info in static_models(provider.id).items():
            rows.append(
                {
                    "selector": f"{provider.id}/{model_id}",
                    "provider": provider.id,
                    "model_id": model_id,
                    "name": getattr(info, "name", "") or model_id,
                }
            )
    return rows


#: The login page is server-rendered (not part of the SPA) so the auth gate
#: has zero client-side surface: no bundle, no router state, no way for a
#: stale cached SPA to sit in front of a password form.
_LOGIN_ERROR = '<p class="error">Wrong password.</p>'

_LOGIN_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#14110c">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>local operator — sign in</title>
<style>
  /* Values from local_operator/tui/theme.py BRAND_TOKENS.dark — the login
     page is server-rendered so the auth gate has zero client-side surface,
     so it tracks the TUI's own tokens by hand. Keep the two in sync: the
     TUI palette is the brand. Layout follows the TUI welcome lockup
     (welcome.py): the mark, then the letterspaced wordmark, then the
     form — no box, no accent spent on the identity. */
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    min-height: 100dvh;
    display: grid;
    place-items: center;
    padding: max(env(safe-area-inset-top), 32px) 24px max(env(safe-area-inset-bottom), 32px);
    background: #14110c;                           /* bg */
    color: #e9e5db;                                /* fg */
    font: 16px/1.5 -apple-system, "SF Pro Text", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  form {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    width: min(320px, 100%);
  }
  .lockup {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 40px;
  }
  .mark {
    width: 72px;
    height: 72px;
    display: block;
    /* The PNG is already tinted to TUI dim (#837c6d); do not fade it
       further — opacity on a white glyph is what made it read cooler
       and brighter than the TUI rest colour. */
  }
  h1 {
    margin: 20px 0 0;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: lowercase;
    color: #e9e5db;                                /* fg — the brightest row */
    text-align: center;
  }
  .field { display: flex; flex-direction: column; gap: 8px; }
  label {
    font-size: 12px;
    letter-spacing: 0.04em;
    color: #837c6d;                                /* dim */
  }
  input {
    font-size: 16px;                               /* iOS no-zoom floor */
    line-height: 1.4;
    padding: 14px 16px;
    border-radius: 10px;                           /* radius-md */
    border: 1px solid #3b3527;                     /* edge */
    background: #1e1a14;                           /* surface */
    color: #e9e5db;
    width: 100%;
    -webkit-appearance: none;
    appearance: none;
  }
  input::placeholder { color: #837c6d; }           /* dim */
  input:focus {
    outline: 2px solid #38c96a;                    /* accent — the one green */
    outline-offset: 1px;
    border-color: transparent;
  }
  button {
    margin-top: 20px;
    font-size: 15px;
    font-weight: 500;
    letter-spacing: 0.01em;
    padding: 14px 16px;
    min-height: 48px;
    border: 1px solid #3b3527;                     /* edge — not a filled bar */
    border-radius: 10px;
    background: #1e1a14;                           /* surface */
    color: #e9e5db;                                /* fg */
    cursor: pointer;
    -webkit-appearance: none;
    appearance: none;
  }
  button:active { background: #272219; }           /* raised */
  .error {
    color: #ef8078;                                /* danger */
    text-align: center;
    margin: 0 0 16px;
    font-size: 13px;
  }
</style>
</head>
<body>
<form method="post" action="/login">
  <div class="lockup">
    <img class="mark" src="__MARK_DATA_URI__" width="72" height="72" alt="">
    <h1>local operator</h1>
  </div>
  <!--ERROR-->
  <div class="field">
    <label for="password">password</label>
    <input id="password" type="password" name="password"
           autocomplete="current-password" autofocus required>
  </div>
  <button type="submit">sign in</button>
</form>
</body>
</html>
"""

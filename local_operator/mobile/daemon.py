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

Threading: one asyncio loop. Session runtimes run their own loops in their own
processes; this loop only dials them. Blocking work (session construction,
which reads provider catalogues) goes through ``asyncio.to_thread`` so a
phone starting a session never stalls the SSE streams of the others.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    SubagentRow,
)
from local_operator.session.runtime import registry

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

#: Projection summaries and their route detail are one cache unit. The daemon
#: may see many historical sessions over its lifetime, so bound both together;
#: evicting detail alone leaves a retained projection advertising dead routes.
MAX_RETAINED_SESSION_PROJECTIONS = 64

#: TTL for the summaries cache. The durable half of a listing is a 100-directory
#: scan plus bounded head reads (300 ms to several seconds under loop
#: contention, measured on the operator's 3,925-session store), and a live
#: session repaints the list ~30x/s — without a TTL every repaint re-ran that
#: scan and starved every other request on the single daemon loop. The TTL is
#: the staleness bound for the DURABLE half only (a new terminal session
#: appears within it); live-projection fields are merged fresh on every call,
#: so streaming/pending state never ages. Structural changes (registration,
#: heartbeat, wake, session death) invalidate the cache outright via
#: ``notify_list_changed``, so the TTL is only what a quiet machine pays.
SUMMARIES_CACHE_TTL_S = 1.0


def _durable_fold_cache():
    """The daemon-wide cache of incremental durable folds (see
    :mod:`.durable`). Created lazily so importing the daemon never pays for
    the fold machinery, and so tests that patch ``config_dir`` before first
    use get a cache keyed by THEIR directories."""
    global _DURABLE_FOLD_CACHE
    if _DURABLE_FOLD_CACHE is None:
        from local_operator.mobile.durable import DurableFoldCache

        _DURABLE_FOLD_CACHE = DurableFoldCache()
    return _DURABLE_FOLD_CACHE


_DURABLE_FOLD_CACHE: Any = None


def _custom_snapshot_cache():
    """The daemon-wide newest-wins custom-snapshot cache (see
    :class:`.durable.CustomSnapshotCache`). Deliberately separate from the
    fold cache: a deep roster asks every child transcript for its todo
    snapshot, and routing those reads through the bounded fold cache would
    evict the ROOT's fold — re-folding a 50 MB transcript on the next open."""
    global _CUSTOM_SNAPSHOT_CACHE
    if _CUSTOM_SNAPSHOT_CACHE is None:
        from local_operator.mobile.durable import CustomSnapshotCache

        _CUSTOM_SNAPSHOT_CACHE = CustomSnapshotCache()
    return _CUSTOM_SNAPSHOT_CACHE


_CUSTOM_SNAPSHOT_CACHE: Any = None


class _StaleProjection(Exception):
    """A fenced owner frame with no retained payload to republish."""


@dataclass
class _ProjectionGeneration:
    """Daemon-local ordering for one registrant generation's projection epochs."""

    identity: tuple[int, float, str] | None
    started_at: float | None
    retired: tuple[tuple[int, float, str], ...]
    local_version: int
    offset: int
    epoch: int
    terminal: bool = False


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
        # Per-session seen state the "unseen" verdict reads. Owned by the table
        # (the merge reads it per row) and created LAZILY on first verdict (see
        # the property), so a test that patches config_dir AFTER construction
        # still gets ITS directory, and a bare SessionTable never touches disk
        # until a verdict actually runs. The daemon's own seen_store property
        # delegates here so the /seen endpoint and the verdict share ONE
        # instance.
        self._seen_store: Any = None
        # The durable half of summaries() is a directory scan; cache it behind a
        # short TTL (SUMMARIES_CACHE_TTL_S) with single-flight refresh so N
        # concurrent list consumers pay one scan, not N. The merged summary list
        # is cached separately because the merge itself walks every live entry.
        self._durable_rows_cache: dict[str, Any] | None = None
        self._durable_rows_at = 0.0
        self._durable_rows_task: asyncio.Task[dict[str, Any]] | None = None
        self._summaries_cache: list[dict[str, Any]] | None = None
        self._summaries_at = 0.0
        self._summaries_task: asyncio.Task[list[dict[str, Any]]] | None = None

    def invalidate_summaries_cache(self) -> None:
        """Drop both summaries caches so the next read rescans.

        Called on every structural change (registration, heartbeat, wake,
        session death — all funnel through ``notify_list_changed``) and when
        the phone marks a session seen. The TTL alone would heal the same
        facts within a second; outright invalidation makes the next repaint
        correct instead of merely eventually correct.
        """
        self._durable_rows_cache = None
        self._durable_rows_at = 0.0
        self._summaries_cache = None
        self._summaries_at = 0.0

    async def _refresh_durable_rows(self) -> dict[str, Any]:
        """Single-flight TTL refresh of the durable listing rows.

        Runs ``recent_session_rows`` OFF the event loop: it stats and reads a
        hundred session directories, which measured 300 ms to several seconds
        under contention — blocking work that froze every SSE stream on this
        loop while it ran. A concurrent caller joins the in-flight task
        instead of starting a second scan.
        """
        from local_operator.paths import config_dir
        from local_operator.resume import recent_session_rows

        task = self._durable_rows_task
        if task is not None and not task.done():
            return await task

        async def _load() -> dict[str, Any]:
            rows = await asyncio.to_thread(recent_session_rows, config_dir(), 100)
            return {row.id: row for row in rows}

        task = asyncio.ensure_future(_load())
        self._durable_rows_task = task
        try:
            rows = await task
        except BaseException:
            # A failed scan must not poison the shared task: the next caller
            # retries instead of awaiting a raised future forever.
            if self._durable_rows_task is task:
                self._durable_rows_task = None
            raise
        self._durable_rows_cache = rows
        self._durable_rows_at = time.monotonic()
        return rows

    async def summaries(self) -> list[dict[str, Any]]:
        """Reconcile live generations with durable conversations by session id.

        Async because its durable half is blocking disk work (see
        ``_refresh_durable_rows``); every call site awaits it off the loop.
        The result is cached for ``SUMMARIES_CACHE_TTL_S`` — live-projection
        fields are merged fresh on every build, so only the durable rows can
        age, and structural changes invalidate the cache outright.
        """
        now = time.monotonic()
        cached = self._summaries_cache
        if cached is not None and now - self._summaries_at < SUMMARIES_CACHE_TTL_S:
            return cached
        task = self._summaries_task
        if task is not None and not task.done():
            return await task

        async def _build() -> list[dict[str, Any]]:
            rows = self._durable_rows_cache
            if rows is None or time.monotonic() - self._durable_rows_at >= SUMMARIES_CACHE_TTL_S:
                rows = await self._refresh_durable_rows()
            return self._merge_summaries(rows)

        task = asyncio.ensure_future(_build())
        self._summaries_task = task
        try:
            out = await task
        except BaseException:
            if self._summaries_task is task:
                self._summaries_task = None
            raise
        self._summaries_cache = out
        self._summaries_at = time.monotonic()
        return out

    def _merge_summaries(self, durable: dict[str, Any]) -> list[dict[str, Any]]:
        """Merge cached durable rows with fresh live state into summary rows.

        Pure in-memory work (safe on the loop); split out of ``summaries`` so
        the cache layer and the row shape are separately testable.
        """
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
                    # Unread verdict (see :mod:`.seen`): activity newer than the
                    # phone's last view. The activity clock is the transcript
                    # mtime for durable rows — already statted by the listing
                    # scan, no extra syscall — and the record heartbeat for
                    # live rows. First observation records a baseline so an
                    # upgrade never lights up the whole store.
                    "unseen": self._is_unseen(session_id, row, entry),
                }
            )
        out.sort(
            key=lambda summary: (
                summary["section"] != "active",
                not summary["needs_attention"],
                # ONE ladder serves the sort and the render: NEEDS DECISION >
                # WORKING > UNREAD > IDLE (see the client's SessionCard, which
                # states the same order). Unread outranks recency — burying the
                # mark under plain mtime order is what made it decorative — but
                # sits BELOW streaming, because the client renders "new" only
                # for COMPLETED unviewed activity and suppresses it on an
                # in-flight row. Ranking unseen above streaming here hoisted a
                # streaming+unseen row over newer rows while it rendered no
                # mark to explain the position: a sort the surface contradicts.
                # And below needs_attention, because a pending ask blocks a
                # turn outright while unread only means unlooked-at.
                not summary["streaming"],
                not summary["unseen"],
                -summary["mtime"],
                summary["session_id"],
            )
        )
        return out

    def notify_list_changed(self) -> None:
        """Wake the list SSE subscribers for a repaint.

        Deliberately does NOT invalidate the summaries cache. Every projection
        push from a live registrant calls this (~30x/s while streaming), and
        invalidating here re-ran the full durable directory scan on each one —
        which defeated the TTL cache in exactly the busy case it was built for
        (measured: 30 repaints produced 30 scans at 42-92 ms each). A push
        changes only LIVE fields, and ``_merge_summaries`` recomputes those
        from ``self.entries`` on every build, so the cached durable rows stay
        correct across it. Callers that genuinely change the DURABLE set
        (registration, death, wake, seen) call
        :meth:`invalidate_summaries_cache` themselves.
        """
        for queue in self.list_subscribers:
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    @property
    def seen_store(self):
        """The persisted seen-state store, created on first use.

        Lazy so the store resolves ``config_dir()`` at verdict time — tests
        patch it after building the table, and a bare SessionTable pays
        nothing until a verdict runs.
        """
        if self._seen_store is None:
            from local_operator.mobile.seen import SEEN_STORE_NAME, SeenStore
            from local_operator.paths import config_dir

            self._seen_store = SeenStore(config_dir() / SEEN_STORE_NAME)
        return self._seen_store

    def _is_unseen(self, session_id: str, row: Any, entry: SessionEntry | None) -> bool:
        """The seen-store verdict for one summary row.

        Cheap: dict lookups only, no disk access — the store is fully in
        memory and persists itself on ``mark_seen``.

        Two rules the naive version got wrong:

        - **Holding the projection SSE stream IS viewing** (spec §3). A user
          watching a turn finish must never come back to that session marked
          new, so an open subscriber re-stamps ``last_seen`` here rather than
          relying on the client to POST /seen again after every repaint.
        - **The activity clock is never ``heartbeat_at``.** That is rewritten
          unconditionally every HEARTBEAT_INTERVAL_S whether or not anything
          happened, so using it re-lit a just-cleared session every 15 s
          forever. Only the transcript mtime dates real activity; without a
          durable row there is nothing to date against, and the session's own
          ``started_at`` is a fixed instant that cannot creep.
        """
        store = self.seen_store
        if self.session_subscribers.get(session_id):
            # Viewing now: keep the stamp at the current instant so activity
            # arriving during the watch is already covered when it lands. The
            # disk write is debounced inside the store — this runs on every
            # repaint of a watched session.
            store.touch_watched(session_id)
            return False
        if row is not None:
            activity = row.mtime
        elif entry is not None:
            # A live session with no durable row yet (younger than its first
            # transcript write, or outside the 100-row window). started_at is
            # a birth instant, not a liveness clock.
            activity = entry.record.started_at
        else:
            activity = 0.0
        if not activity:
            return False
        return store.is_unseen(session_id, activity)


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
    """Fold a user conversation and its routable child lineage from disk.

    Reads through the daemon's incremental fold cache (:mod:`.durable`): the
    first open of a session pays one full fold, every later open reads only
    the bytes appended since. The projection object itself is rebuilt on
    every call (callers mutate and fence it), so what is cached is the fold,
    not the projection.
    """
    from local_operator.mobile.projection import (
        SUBAGENT_ERROR_CHARS,
        SUBAGENT_OUTCOME_CHARS,
        SUBAGENT_PROMPT_PREVIEW_CHARS,
        ProjectionFold,
        _compact,
        _compact_multiline,
    )
    from local_operator.resume import stored_session_title
    from local_operator.tools.builtin import todo_snapshot

    directory = _durable_user_session_dir(session_id)
    if directory is None:
        return None
    try:
        state = _durable_fold_cache().load(directory)
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 — an odd transcript yields no projection, not a 500
        logger.exception("durable fold failed for session %s", session_id)
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
    # fold_history reads messages without mutating them, so the cached list
    # can be shared; the fold builds its own TranscriptEntry rows.
    fold.fold_history(state.history)

    # The persisted roster is the restart-safe ownership record for child
    # routes. Rebuilding from it keeps old session projections useful without
    # retaining every child's unbounded transcript in daemon memory forever.
    snapshot = state.latest_customs.get("subagent_roster") or {}
    jobs = {str(row.get("id") or ""): row for row in snapshot.get("jobs") or []}
    records = [row for row in snapshot.get("records") or [] if row.get("job_id")]
    by_parent: dict[str | None, list[str]] = {}
    for record in records:
        parent = str(record["parent_job_id"]) if record.get("parent_job_id") else None
        by_parent.setdefault(parent, []).append(str(record["job_id"]))
    for record in records:
        job_id = str(record["job_id"])
        job = jobs.get(job_id, {})
        raw_dir = record.get("session_dir")
        child_dir = Path(str(raw_dir)) if raw_dir else None
        status = str(record.get("outcome") or job.get("status") or "cancelled")
        if status in ("queued", "starting", "running"):
            status = "cancelled"
        elif status in ("paused", "pausing") or record.get("paused"):
            status = "parked"
        elif status in ("interrupted", "gone"):
            status = "cancelled"
        parent_id = str(record["parent_job_id"]) if record.get("parent_job_id") else None
        peers = [item for item in by_parent.get(parent_id, []) if item != job_id]
        ancestors: list[str] = []
        ancestor_ids: list[str] = []
        cursor = parent_id
        record_by_id = {str(item["job_id"]): item for item in records}
        while cursor and cursor in record_by_id:
            ancestor = record_by_id[cursor]
            ancestor_ids.insert(0, cursor)
            ancestors.insert(0, str(ancestor.get("label") or cursor))
            cursor = str(ancestor["parent_job_id"]) if ancestor.get("parent_job_id") else None
        raw_todos = todo_snapshot(child_dir.name) if child_dir else []
        if not raw_todos and child_dir is not None and child_dir.is_dir():
            # The child's todo snapshot through the dedicated snapshot cache:
            # a deep roster used to full-parse every child transcript on every
            # durable projection, once per child. Routed around the fold cache
            # so an 80-child roster cannot evict the root's fold.
            raw_todos = (_custom_snapshot_cache().load(child_dir, "todo_snapshot") or {}).get(
                "items"
            ) or []
        row = SubagentRow(
            job_id=job_id,
            label=str(record.get("label") or job_id),
            agent=str(record.get("agent_role") or job.get("agent_role") or "task"),
            status=status,  # type: ignore[arg-type] -- normalized persisted literals
            model_label=str(job.get("model_label") or ""),
            # Bound the settled text on the wire to match the live fold, and for
            # the same reasons the two fields differ there (see
            # SUBAGENT_OUTCOME_CHARS / SUBAGENT_ERROR_CHARS): ``result_text`` is a
            # preview recoverable from the child transcript the phone fetches
            # lazily, while ``error_text`` is the parent runner's ``str(exc)``,
            # never in that transcript, so the wire value is the only copy the
            # Outcome panel can render and it is carried generously. Newlines
            # preserved so a multi-line handoff or stack trace stays legible.
            result_text=_compact_multiline(
                str(record.get("result_text") or ""), SUBAGENT_OUTCOME_CHARS
            ),
            error_text=_compact_multiline(
                str(record.get("error_text") or job.get("error_text") or ""),
                SUBAGENT_ERROR_CHARS,
            ),
            parent_job_id=parent_id,
            session_id=child_dir.name if child_dir else None,
            # Compacted preview only, same bound as the live fold — see
            # SUBAGENT_PROMPT_PREVIEW_CHARS. Uncapped prompts across a deep
            # durable roster reintroduce the oversized-frame wedge.
            prompt=_compact(str(record.get("prompt") or ""), SUBAGENT_PROMPT_PREVIEW_CHARS),
            launch_message_id=str(record.get("launch_message_id") or ""),
            effort=str(record.get("effort") or job.get("effort") or ""),
            ancestors=ancestors,
            ancestor_ids=ancestor_ids,
            child_ids=list(by_parent.get(job_id, [])),
            peer_ids=peers,
            # A child transcript is NEVER carried on the wire — see
            # ``ProjectionFold.set_subagent_hydrated_details`` for the full
            # rationale. The daemon serves it lazily from disk over
            # ``/api/sessions/{sid}/agents/{job_id}/history`` (this durable row's
            # ``session_id`` is the child dir the endpoint reads), so even a
            # reconstructed roster stays small. ``child_transcript`` is still
            # read above for the durable todo snapshot fallback.
            transcript=[],
            todos=fold._todo_phases(raw_todos),
        )
        projection.subagents.append(row)
    projection.ended = False
    projection.degraded = False
    return projection


# ---------------------------------------------------------------------------
# Session runtime connections
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
        # ``locality: remote`` is the truthful description of this connection.
        # The daemon dials over loopback, but it is a RELAY: the human driving
        # it is holding a phone, reaching this machine through the mobile
        # portal's tunnel. Operations that act on the user's physical
        # surroundings must not be run for it — an OAuth grant would open a
        # browser tab on this desktop, in front of nobody, and write a
        # credential the phone's owner cannot see or use.
        #
        # Loopback proves the CALLER is on this machine; it does not prove the
        # PERSON is (`ClientLocality`). The daemon is the one client today for
        # which those differ, and `mobile/types.py` already admits
        # ``slash_result``, so the phone can reach `/mcp reauth` through it.
        writer.write(json.dumps({"key": record.control_key, "locality": "remote"}).encode() + b"\n")
        await writer.drain()
        while True:
            try:
                line = await reader.readline()
            except ValueError:
                # A frame longer than the 1 MB stream limit is skipped, not a
                # reason to drop the session. ``StreamReader.readline`` already
                # DRAINS the oversized line on LimitOverrunError (it removes the
                # complete line through the separator, or clears the buffer when
                # no separator is in range) BEFORE raising — unlike
                # ``readuntil``, which would leave the bytes in place and make
                # this loop re-raise on the same data forever. So ``continue``
                # here degrades to "drop this one frame, keep the connection,
                # deliver the next" without any manual read-and-discard. The
                # real guard against ever reaching this path is keeping frames
                # small: subagent transcripts are no longer embedded in the
                # projection (see ``ProjectionFold.set_subagent_hydrated_details``
                # and ``_durable_projection``) and are fetched lazily instead. A
                # sustained flood of this warning means a producer regressed and
                # is pushing oversized frames again — every skipped frame leaves
                # ``entry.projection`` stale, so the phone falls back to the
                # durable disk fold and stops updating live.
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
                    incoming = _projection_from_json(data, record)
                except (TypeError, ValueError, KeyError):
                    # A malformed push (mid-upgrade registrant, renamed field)
                    # must not tear the dial loop down to the reconnect path —
                    # the NEXT push is a full repaint that repairs the view.
                    logger.debug("mobile daemon: dropping malformed projection", exc_info=True)
                    continue
                incoming.degraded = False
                incoming.ended = False
                try:
                    captured = daemon.capture_subagent_details(incoming, record=record)
                except _StaleProjection:
                    # A predecessor frame can arrive after its payload cache entry
                    # was evicted. Its identity remains fenced by the epoch ledger.
                    continue
                entry.projection = captured
                # Only the FIRST push after a wake changes the durable picture:
                # it retires the provisional-active marker, which moves the row
                # between sections. Invalidating on EVERY push is what defeated
                # the summaries cache — a streaming session pushes ~30x/s and
                # each one re-ran the full directory scan.
                session_id = entry.record.session_id
                if session_id in daemon.table.provisional_active:
                    daemon.table.provisional_active.discard(session_id)
                    daemon.table.invalidate_summaries_cache()
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


def _projection_frame(projection: SessionProjection) -> dict[str, Any]:
    """The daemon's serialization boundary: one capped frame dict.

    Both wire paths must agree on size. The registrant caps before broadcast
    (see ``RuntimeServer._projection_payload``); the daemon serves the SAME
    projection shape over SSE and republishes durable rebuilds, so it caps at
    every serialization site too — a durable fold of a long session can embed
    80 rows x 8 KB tool outputs, which no socket or phone renderer wants
    whole. Degradation is tiered and lossless for the collapsed view (see
    ``cap_projection_frame``); the retained projection object is untouched.
    """
    from local_operator.mobile.projection import cap_projection_frame

    data, degraded = cap_projection_frame(projection)
    if degraded:
        logger.debug(
            "mobile daemon: capped oversized projection frame for session %s",
            projection.session_id,
        )
    return data


def _history_page(
    session_id: str, before: str | None, limit: int, *, durable_only: bool = True
) -> tuple[list[Any], bool]:
    """Return the page of folded entries immediately OLDER than ``before``
    (chronological within the page) plus whether more history exists beyond it.

    Reads through the daemon's incremental fold cache (:mod:`.durable`), so a
    page costs one fold per session per daemon lifetime plus the appended
    tail since — not the whole-file re-parse every page used to pay. Runs off
    the event loop (``asyncio.to_thread`` at the call site): even the cached
    path touches disk and the fold is not loop-safe work.
    """
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
        state = _durable_fold_cache().load(directory)
        entries = state.render
    except FileNotFoundError:
        return [], False
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
    frame = _projection_frame(entry.projection)
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
    def __init__(
        self,
        *,
        port: int = DEFAULT_PORT,
        password: str | None = None,
        dial_registrants: bool = True,
    ) -> None:
        self.port = port
        self.password = password
        self.table = SessionTable()
        # False makes this daemon a READ-ONLY observer of the record directory:
        # it lists sessions and serves durable folds, but never dials a
        # registrant's control socket and never reaps a stale claim. A second
        # daemon on the same machine MUST run this way: a registrant admits at
        # most ONE daemon connection, so a secondary dial would evict the
        # production daemon's live bridge mid-session. Set via
        # ``LO_MOBILE_NO_DIAL=1`` (see ``service.amain``).
        self.dial_registrants = dial_registrants
        # Per-session "last seen by phone" state lives on the table (the merge
        # reads it per row); the daemon's ``seen_store`` property delegates to
        # it so the /seen endpoint and the verdict share one lazily-created
        # store.
        # The session repaint carries only roster summaries. Full child state is
        # retained separately and fetched for the active route, otherwise one
        # busy descendant makes every root token repaint resend every transcript.
        # Dict insertion order is the LRU clock for projection/detail cache
        # units. A detail route must live exactly as long as the projection that
        # advertises it, never under an independent bound.
        self.subagent_details: dict[tuple[str, str], dict[str, Any]] = {}
        self._pending_reqs: dict[tuple[int, Any], asyncio.Future[dict[str, Any]]] = {}
        self._dial_tasks: dict[int, asyncio.Task[None]] = {}
        self._slash_commands: list[dict[str, Any]] | None = None
        # Session id -> pid of a resume spawn already in flight, so a retried
        # resume POST returns the same child instead of forking a second.
        self.resumes_in_flight: dict[str, int] = {}
        # A session route outlives every process generation. Retain its latest
        # repaint so an open phone remains a normal conversation while idle.
        self.session_projections: dict[str, SessionProjection] = {}
        # ProjectionFold versions restart at zero with every owner. Ordering is
        # deliberately NOT part of the bounded payload cache: an open route or
        # live registrant may outlive cache pressure, and its browser has already
        # observed this epoch. The ledger is retired only after payload, process,
        # and subscriber ownership are all gone.
        self._projection_generations: dict[str, _ProjectionGeneration] = {}
        self._wake_settle_tasks: dict[str, asyncio.Task[None]] = {}

    def _projection_route_owned(self, session_id: str) -> bool:
        """Whether an epoch can still be observed by a process or browser."""
        return bool(self.table.session_subscribers.get(session_id)) or any(
            entry.record.session_id == session_id and not entry.ended
            for entry in self.table.entries.values()
        )

    @property
    def seen_store(self):
        """The persisted seen-state store — delegates to the table so the
        /seen endpoint and the summaries verdict share one instance."""
        return self.table.seen_store

    def _prune_projection_generation(self, session_id: str) -> None:
        """Retire ordering only when no durable in-memory route can emit again."""
        if session_id in self.session_projections or self._projection_route_owned(session_id):
            return
        self._projection_generations.pop(session_id, None)

    def capture_subagent_details(
        self,
        projection: SessionProjection,
        *,
        record: SessionRecord | None = None,
        terminal: bool = False,
    ) -> SessionProjection:
        """Cache full descendant state and return a lightweight aggregate copy.

        ``SessionRecord.started_at`` plus its per-registration control key forms
        the process birth identity: PID alone can be reused, while the key is
        regenerated for every registrant. The birth timestamp orders replacements;
        a bounded retired set fences a late predecessor even on a clock collision.
        """
        session_id = projection.session_id
        previous_projection = self.session_projections.get(session_id)
        state = self._projection_generations.get(session_id)
        retained_recapture = projection is previous_projection
        identity = (
            (record.pid, record.started_at, record.control_key) if record is not None else None
        )
        started_at = record.started_at if record is not None else None
        generation_changed = False

        if previous_projection is not None and retained_recapture:
            # Wake and reconnect deliberately republish this stripped object. It
            # is already stamped with the daemon epoch and cannot be a new frame.
            self.session_projections.pop(session_id)
            self.session_projections[session_id] = previous_projection
            return previous_projection

        if state is not None and identity is not None and identity != state.identity:
            if identity in state.retired or (
                state.started_at is not None
                and started_at is not None
                and started_at < state.started_at
            ):
                # A predecessor socket may still have a decoded frame queued when
                # its replacement registers. Payload eviction must not make that
                # stale identity publishable; its caller drops the fenced frame.
                if previous_projection is None:
                    raise _StaleProjection
                self.session_projections.pop(session_id)
                self.session_projections[session_id] = previous_projection
                return previous_projection
            generation_changed = True
        elif state is not None and identity is not None and state.identity is None:
            # A disk-only repaint has no process identity. The first registrant is
            # authoritative even when its ProjectionFold counter starts lower.
            generation_changed = True

        # A durable fold carries no process identity (``record is None``). It is a
        # reconstruction of the session from disk, never a live registrant frame,
        # so it can never reopen or advance a generation — it only republishes the
        # retained epoch the browser already observed. This is the seam the whole
        # lifecycle contract turns on: an evicted payload on a still-owned route
        # must rematerialize here, while a genuine late predecessor frame (which
        # DOES carry its identity) stays fenced below.
        durable_rematerialize = False
        if state is not None and not generation_changed:
            stale = (
                (state.terminal and not terminal)
                or (terminal and state.terminal and projection.version <= state.local_version)
                or (not terminal and projection.version < state.local_version)
            )
            if stale:
                if previous_projection is not None:
                    self.session_projections.pop(session_id)
                    self.session_projections[session_id] = previous_projection
                    return previous_projection
                if identity is None:
                    # Payload cache pressure evicted this route's only payload
                    # while its generation ledger survived (a live/subscribed or
                    # durably reconstructable route keeps it). Rebuild the payload
                    # at the retained epoch instead of fencing detail/history/SSE
                    # reconstruction to an HTTP 500. Only a truly-gone session —
                    # whose ledger was already pruned — reaches ``state is None``.
                    durable_rematerialize = True
                else:
                    # A predecessor socket frame whose payload is gone: it must
                    # not become publishable, so its caller drops the fenced frame.
                    raise _StaleProjection

        if state is None:
            epoch = projection.version
            offset = 0
        elif durable_rematerialize:
            # Republish at exactly the observed epoch; the generation ledger is
            # left intact so a subsequent live owner frame still advances it.
            epoch = state.epoch
            offset = state.offset
        elif generation_changed or terminal:
            epoch = max(state.epoch + 1, projection.version)
            offset = epoch - projection.version
        else:
            offset = state.offset
            epoch = offset + projection.version

        retired = state.retired if state is not None else ()
        if generation_changed and state is not None and state.identity is not None:
            # A socket can only race a small number of replacements. Bounding the
            # fence avoids turning a frequently resumed route into an append-only
            # process history while still covering every plausible late frame.
            retired = (*retired, state.identity)[-8:]
        # A durable rematerialization only rebuilds the evicted payload; it must
        # leave the ledger's fencing fields exactly as the last live owner set
        # them. Writing the low durable version into ``local_version`` (or
        # clearing ``terminal``) would open a window for a genuine late old frame.
        self._projection_generations[session_id] = _ProjectionGeneration(
            identity=identity if identity is not None else (state.identity if state else None),
            started_at=(
                started_at if started_at is not None else (state.started_at if state else None)
            ),
            retired=retired,
            local_version=(
                state.local_version if durable_rematerialize and state else projection.version
            ),
            offset=offset,
            epoch=epoch,
            terminal=(state.terminal if durable_rematerialize and state else terminal),
        )
        # Root transcript and todo state are immutable during this call; copying
        # only descendant rows avoids duplicating the whole repaint per token.
        summary = copy.copy(projection)
        summary.subagents = copy.deepcopy(projection.subagents)
        summary.version = epoch
        # Reinsert on every repaint so active/reconnected routes are most recent.
        self.session_projections.pop(session_id, None)
        self.session_projections[session_id] = summary
        while len(self.session_projections) > MAX_RETAINED_SESSION_PROJECTIONS:
            expired = next(iter(self.session_projections))
            self.session_projections.pop(expired, None)
            for key in [key for key in self.subagent_details if key[0] == expired]:
                self.subagent_details.pop(key, None)
            self._prune_projection_generation(expired)
        # Every summary published in the roster must resolve through the detail
        # route. The process already bounds concurrent jobs; settled lineage is
        # intentionally durable, so a second arbitrary 256-row cache bound made
        # older rendered rows deterministic 404s in long-lived sessions.
        published_ids = {row.job_id for row in projection.subagents}
        for key in [
            key
            for key in self.subagent_details
            if key[0] == session_id and key[1] not in published_ids
        ]:
            self.subagent_details.pop(key, None)
        for row, summary_row in zip(projection.subagents, summary.subagents, strict=True):
            key = (session_id, row.job_id)
            incoming = row.to_json()
            existing = None if generation_changed else self.subagent_details.get(key)
            if existing is not None:
                # Summary fields are real lifecycle updates even when empty.
                # Detail-only empties are ambiguous after projection stripping,
                # so only nonempty values replace the richer cached payload.
                for field in ("prompt", "launch_message_id", "transcript", "todos"):
                    if not incoming[field]:
                        incoming[field] = existing.get(field, incoming[field])
                if retained_recapture:
                    # Only the retained aggregate is known to have had these
                    # lifecycle payloads stripped. A fresh host repaint owns
                    # empty values too, which clears stale terminal outcomes
                    # when a child is resumed or settles without result text.
                    for field in ("result_text", "error_text"):
                        if not incoming[field]:
                            incoming[field] = existing.get(field, incoming[field])
                if row.status == "completed":
                    incoming["error_text"] = ""
                elif row.status == "failed":
                    incoming["result_text"] = ""
                elif not retained_recapture:
                    incoming["result_text"] = ""
                    incoming["error_text"] = ""
            incoming["version"] = epoch
            self.subagent_details[key] = incoming
            # The aggregate needs enough to paint and route rows, not the launch
            # prompt or terminal payload. Those can each be many kilobytes and
            # belong to selected detail exactly like transcript and todos.
            summary_row.prompt = ""
            summary_row.launch_message_id = ""
            summary_row.result_text = ""
            summary_row.error_text = ""
            summary_row.transcript = []
            summary_row.todos = []
        return summary

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
            registration_changed = entry is not None and (
                entry.record.started_at,
                entry.record.control_key,
            ) != (record.started_at, record.control_key)
            if entry is None or registration_changed:
                if entry is not None:
                    # PID reuse must replace every process-scoped flag and request
                    # sequence. Reusing the ended/degraded entry would make the
                    # new registrant permanently ineligible for adoption.
                    if entry.writer is not None:
                        entry.writer.close()
                    old_dial = self._dial_tasks.pop(record.pid, None)
                    if old_dial is not None and not old_dial.done():
                        old_dial.cancel()
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
                # A no-dial daemon is an observer: lease reaping belongs to the
                # production daemon that owns the session, and two reapers on
                # one store is a claim race.
                if self.dial_registrants:
                    from local_operator.paths import config_dir
                    from local_operator.session_lease import (
                        reap_proven_dead_session_claim,
                    )

                    await asyncio.to_thread(
                        reap_proven_dead_session_claim,
                        config_dir() / "sessions" / record.session_id,
                        record.pid,
                    )
                self.table.provisional_active.discard(record.session_id)
                projection = await asyncio.to_thread(_durable_projection, record.session_id)
                if projection is not None:
                    projection = self.capture_subagent_details(
                        projection, record=record, terminal=True
                    )
                    for queue in self.table.session_subscribers.get(record.session_id, set()):
                        # Serialize ONCE per repaint: the QueueFull retry below
                        # re-puts the same frame, and capping is a json.dumps.
                        frame = _projection_frame(projection)
                        try:
                            queue.put_nowait(frame)
                        except asyncio.QueueFull:
                            try:
                                queue.get_nowait()
                                queue.put_nowait(frame)
                            except asyncio.QueueEmpty:
                                pass
                self._prune_projection_generation(record.session_id)
            elif state == "wedged":
                entry.degraded = True
            # Degraded is precisely "we owe this session a redial" — the only
            # gates are ended, an open socket, and the backoff clock. Excluding
            # degraded entries here was the starvation bug: one refused dial
            # meant never trying again. A no-dial daemon owes no dial at all:
            # its entries exist so the list and durable routes work, and the
            # production daemon owns every control socket.
            if self.dial_registrants and not entry.ended and entry.writer is None:
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
                        projection = self.capture_subagent_details(
                            projection, record=entry.record, terminal=True
                        )
                        for queue in self.table.session_subscribers.get(session_id, set()):
                            frame = _projection_frame(projection)
                            try:
                                queue.put_nowait(frame)
                            except asyncio.QueueFull:
                                try:
                                    queue.get_nowait()
                                    queue.put_nowait(frame)
                                except asyncio.QueueEmpty:
                                    pass
                    self._prune_projection_generation(session_id)
        if changed:
            # Structural: a session registered, was replaced, or died, so the
            # durable listing itself may have moved.
            self.table.invalidate_summaries_cache()
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
                        projection = self.capture_subagent_details(projection, terminal=True)
                        for queue in self.table.session_subscribers.get(session_id, set()):
                            try:
                                queue.put_nowait(_projection_frame(projection))
                            except asyncio.QueueFull:
                                pass
                    # Structural: the wake settled into a durable (or dead)
                    # session, which changes what the listing scan returns.
                    self.table.invalidate_summaries_cache()
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
        local_operator.session.runtime.process``), so the record + control socket path is
        literally the same code the TUI uses.
        """
        if not self.dial_registrants:
            # An observer daemon cannot adopt what it spawns (it never dials),
            # so a spawned child would be orphaned from its own control plane.
            raise RuntimeError("observer daemon cannot start sessions")
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
            "local_operator.session.runtime.process",
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

    def cross_origin_mutation(request: Request) -> Response | None:
        """SameSite cookies do not separate sibling personal-tunnel subdomains.

        A page on another owner's hostname can send a simple text/plain POST
        carrying this host's cookies. Compare the exact authority before any
        mutation; the authenticated tunnel gateway independently verifies the
        public HTTPS Origin before forwarding to this loopback HTTP server.
        Non-browser local API callers legitimately have no Origin header.
        """
        if request.method in {"GET", "HEAD", "OPTIONS"}:
            return None
        origin = request.headers.get("origin")
        host = request.headers.get("host", "")
        if (
            origin is not None and origin not in {f"http://{host}", f"https://{host}"}
        ) or request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"error": "same-origin request required"}, status_code=403)
        return None

    def gate(request: Request) -> Response | None:
        """None = allowed. Browsers get the login redirect, API calls a 401 —
        the split contract the health check asserts."""
        origin_denied = cross_origin_mutation(request)
        if origin_denied is not None:
            return origin_denied
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
        denied = cross_origin_mutation(request)
        if denied is not None:
            return denied
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
        # Drafts and uncertain command bodies are private authenticated state.
        # The browser, not JavaScript lifecycle guesses, owns complete cleanup
        # when this cookie's user signs out.
        response.headers["Clear-Site-Data"] = '"storage"'
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
        return JSONResponse({"sessions": await daemon.table.summaries()})

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
                        try:
                            projection = daemon.capture_subagent_details(projection)
                        except _StaleProjection:
                            # A subscribed but payload-less route whose owner is
                            # mid-replacement: skip the seed frame rather than
                            # tearing down the handshake. The next live repaint or
                            # keepalive carries the view.
                            projection = None
                if projection is not None:
                    yield _sse("projection", _projection_frame(projection))
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
                        daemon._prune_projection_generation(session_id)

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
                yield _sse("sessions", {"sessions": await daemon.table.summaries()})
                while True:
                    try:
                        await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_S)
                        yield _sse("sessions", {"sessions": await daemon.table.summaries()})
                    except TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                daemon.table.list_subscribers.discard(queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    async def api_session_seen(request: Request) -> Response:
        """The phone marks a session seen; the unread verdict clears.

        Auth-gated like every /api route. The verdict is durable (see
        :mod:`.seen`), so it survives a daemon restart. Unknown ids 404 the
        same way the history route does — a live generation OR a durable user
        session — so the endpoint cannot be used to probe arbitrary paths.
        """
        denied = gate(request)
        if denied is not None:
            return denied
        session_id = str(request.path_params["session_id"])
        entry = _entry_for_session(daemon, session_id)
        if entry is None and _durable_user_session_dir(session_id) is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        daemon.seen_store.mark_seen(session_id)
        # The next list paint must already show the cleared verdict.
        daemon.table.invalidate_summaries_cache()
        daemon.table.notify_list_changed()
        return JSONResponse({"ok": True})

    async def api_subagent_detail(request: Request) -> Response:
        """Full state for the one descendant named by the active phone route."""
        denied = gate(request)
        if denied is not None:
            return denied
        session_id = str(request.path_params["session_id"])
        job_id = str(request.path_params["job_id"])
        detail = daemon.subagent_details.get((session_id, job_id))
        if detail is None:
            projection = await asyncio.to_thread(_durable_projection, session_id)
            if projection is not None:
                try:
                    daemon.capture_subagent_details(projection)
                except _StaleProjection:
                    # Reconstruction races a live owner replacement: the fence is
                    # correct, but a still-durable route must not 500. The next
                    # request rebuilds once the ledger settles.
                    pass
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
        if detail is None:
            projection = await asyncio.to_thread(_durable_projection, session_id)
            if projection is not None:
                try:
                    daemon.capture_subagent_details(projection)
                except _StaleProjection:
                    # See api_subagent_detail: a fenced reconstruction is not an
                    # error; fall through to the durable 404 rather than a 500.
                    pass
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
                if not daemon.dial_registrants:
                    # Waking starts a host process the observer could never
                    # dial; the production daemon owns wake transitions.
                    raise RuntimeError("observer daemon cannot wake sessions")
                from local_operator.mobile.attach_client import continue_command

                command = ContinuationCommand.from_json(
                    {**body, "session_id": session_id, "images": body.get("images", [])}
                )
                from local_operator.paths import config_dir

                # Publish the accepted wake intent before process discovery. It
                # remains authoritative until a live projection arrives or the
                # attempt fails, so even a 50 ms worker is observable in list SSE.
                daemon.table.provisional_active.add(session_id)
                # Structural: the session moves to the active section.
                daemon.table.invalidate_summaries_cache()
                daemon.table.notify_list_changed()
                projection = daemon.session_projections.get(session_id) or _durable_projection(
                    session_id
                )
                if projection is not None:
                    projection.ended = False
                    projection.degraded = False
                    try:
                        projection = daemon.capture_subagent_details(projection)
                    except _StaleProjection:
                        # The optimistic wake repaint is a courtesy; a fenced
                        # reconstruction just means the live owner's frame wins.
                        # The wake itself proceeds regardless.
                        projection = None
                    if projection is not None:
                        for target in daemon.table.session_subscribers.get(session_id, set()):
                            target.put_nowait(_projection_frame(projection))
                try:
                    client, detail = await continue_command(config_dir(), command)
                except BaseException:
                    daemon.table.provisional_active.discard(session_id)
                    # Structural: the failed wake moves it back to previous.
                    daemon.table.invalidate_summaries_cache()
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
        if not isinstance(body, dict):
            return JSONResponse({"error": "request body must be an object"}, status_code=400)
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
        if not isinstance(body, dict):
            return JSONResponse({"error": "request body must be an object"}, status_code=400)
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
        Route("/api/sessions/{session_id:str}/seen", api_session_seen, methods=["POST"]),
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
    """Resumable past sessions for the phone's history list.

    ``forked`` rides along for the same reason the TUI picker draws it: a fork
    that has not named itself yet displays the title it inherited, so it and its
    parent are byte-identical rows — same name, same age — separable only by a
    12-hex id. The row builder already knows the fact (it is derived from the
    ``origin.json`` the scan parsed), and dropping it here is what would make
    the phone the one surface still showing the twin rows.
    """
    try:
        from local_operator.paths import config_dir
        from local_operator.resume import recent_session_rows

        return [
            {"id": row.id, "name": row.name, "mtime": row.mtime, "forked": row.forked}
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
    from local_operator.resume import fork_haystack, recent_session_rows
    from local_operator.session.search_index import build_index, search_digests

    cfg = config_dir()
    rows = recent_session_rows(cfg, limit=200)
    needle = query.strip().lower()
    if not needle:
        return [
            {
                "id": r.id,
                "name": r.name,
                "mtime": r.mtime,
                "body_match": False,
                "forked": r.forked,
            }
            for r in rows[:limit]
        ]
    try:
        digests = build_index(cfg, [r.id for r in rows])
        body_hits = search_digests(digests, needle)
    except Exception:  # noqa: BLE001 — a broken index degrades to name/id only
        body_hits = set()
    out = []
    for r in rows:
        # Through the picker's own composition, so typing `fork` on the phone
        # finds the rows the phone visibly tags — the same what-is-shown-is-
        # searchable invariant `resume.fork_haystack` documents.
        name_hit = needle in fork_haystack(r).lower() or needle in r.id.lower()
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
                "forked": r.forked,
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
<script>
  /* U2: clear private authenticated state (uncertain command envelopes and
     drafts) whenever the unauthenticated login screen is shown. This is the
     one reachable, WebKit-safe cleanup path: logout, an expired cookie, and a
     401-driven reload ALL land here, and this runs in the page's own engine
     rather than depending on the `Clear-Site-Data` response header, which
     WebKit (the iOS phone target) may ignore. It deliberately does NOT touch
     theme or other non-private preferences — only the two private prefixes,
     kept in sync with web/src/private-storage.ts. */
  (function () {
    try {
      var prefixes = ["lo-mobile-command:", "lo-mobile-draft:"];
      for (var i = localStorage.length - 1; i >= 0; i--) {
        var key = localStorage.key(i);
        if (!key) continue;
        for (var p = 0; p < prefixes.length; p++) {
          if (key.indexOf(prefixes[p]) === 0) {
            localStorage.removeItem(key);
            break;
          }
        }
      }
    } catch (e) {
      /* Private mode or a storage-disabled engine has nothing to clear. */
    }
  })();
</script>
</body>
</html>
"""

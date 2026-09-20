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
import contextlib
import copy
import gzip
import json
import logging
import os
import secrets
import sqlite3
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from local_operator.mobile.attach_client import AttachClient

from local_operator.harness.approval import (
    frame_authority,
    handshake_proof,
    handshake_proof_ok,
    is_wire_hex,
    operator_cap_for,
    operator_nonce,
    request_proof,
)
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
from local_operator.procstate import detached_popen_kwargs
from local_operator.session.creation import session_category, session_created_at
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

# Startup is acknowledged only after the authenticated welcome and control
# connection. The separate handoff lease bounds an abandoned HTTP response;
# it is NOT a session deadline. A real SSE subscriber holds its own viewer.
SESSION_START_TIMEOUT_S = 30.0
PHONE_HANDOFF_S = 60.0

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

#: The wire name for "the durable half of this listing could not be re-read".
#:
#: The phone's conversation list is MEMBERSHIP: the client replaces everything
#: it is showing with this frame, so a read that failed may not be published as
#: an empty list any more than it may be on the desktop sidebar. When the scan
#: cannot be walked the daemon now serves the last listing it actually read and
#: names the failure here, so a client can say "couldn't refresh" instead of
#: rendering a confident negative.
#:
#: DELIBERATELY ITS OWN WORD, not one of ``session.catalog.DECORATION_SOURCES``:
#: those name decorations whose failure costs a MARK while the rows stand, and
#: this names a failure of the read the rows THEMSELVES came from. A renderer
#: that treated the two as one vocabulary would under-report this one.
DEGRADED_DURABLE_LISTING = "sessions"


def _durable_fold_cache():
    """The daemon-wide cache of incremental durable folds (see
    :mod:`.durable`). Created lazily so importing the daemon never pays for
    the fold machinery, and so tests that patch ``config_dir`` before first
    use get a cache keyed by THEIR directories."""
    global _DURABLE_FOLD_CACHE
    if _DURABLE_FOLD_CACHE is None:
        try:
            from local_operator.mobile.durable import DurableFoldCache
        except ImportError as exc:
            # The daemon's own lazy import is a seam that sees a torn install
            # directly (``durable`` pulls ``_journal_injection_ids`` at module
            # scope). Log it NAMED, then re-raise: the fold cannot run without
            # the module, and swallowing the import would turn a diagnosable
            # install race into "no projection" with no cause anywhere.
            _log_import_failure(exc, "local_operator.mobile.durable", where="durable fold cache")
            raise
        _DURABLE_FOLD_CACHE = DurableFoldCache()
    return _DURABLE_FOLD_CACHE


_DURABLE_FOLD_CACHE: Any = None

#: The build this DAEMON PROCESS loaded, stamped once at construction. Compared
#: against the install on disk by ``update.classify_import_failure``, which is
#: the only way to tell a lazy import that lost a name to a HALF-REPLACED
#: install from a genuine packaging bug (design §5.2). ``None`` means "not
#: stamped yet", which classifies nothing rather than guessing.
_BOOT_BUILD: Any = None
_BOOT_BUILD_STAMPED = False


def _boot_build() -> Any:
    """This daemon's boot build, memoised on first read.

    Called from ``MobileDaemon.__init__`` so the normal case stamps at process
    start, before any install can replace the tree. A seam that somehow runs
    first stamps there and then, which can only ever make the comparison MORE
    conservative (a stamp taken after the swap equals the install, so the
    failure stays an ordinary traceback).
    """
    global _BOOT_BUILD, _BOOT_BUILD_STAMPED
    if not _BOOT_BUILD_STAMPED:
        _BOOT_BUILD_STAMPED = True
        try:
            from local_operator.update import installed_build

            _BOOT_BUILD = installed_build()
        except Exception:  # noqa: BLE001 — an unreadable stamp classifies nothing
            _BOOT_BUILD = None
    return _BOOT_BUILD


def _log_import_failure(exc: BaseException, module: str, *, where: str) -> None:
    """Log a lazy-import failure, naming it when the install moved under us.

    One logger call for the daemon's two lazy-import seams, so the 605-occurrence
    ``durable fold failed for session X`` traceback becomes a sentence that says
    WHAT happened and WHY, and the cause can be fed to the next turn's cut-off
    vocabulary.
    """
    from local_operator.update import classify_import_failure

    reason = classify_import_failure(exc, module, boot=_boot_build())
    if reason is None:
        logger.error("%s failed while importing %s", where, module, exc_info=exc)
        return
    logger.error("%s failed: %s (%s)", where, reason, module, exc_info=exc)


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
        #: The operator handshake's state for THIS connection (issue #1310). The
        #: relay is a legitimate console exactly when it spawned the runtime —
        #: the phone's "new session" path does that, a relay watching a
        #: terminal's session does not — so the capability is resolved per dial
        #: and the proof is only ever attached on an authority-increasing frame.
        #: Set by ``_dial`` (auth + welcome) and read by ``request``.
        self.operator_cap: bytes | None = None
        self.operator_nonce = ""
        self.operator_salt = ""
        self.authority_bearing = False
        self.ready = asyncio.Event()
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
        # The last durable listing that was actually READ, kept SEPARATELY from
        # the cache above because the two stop being the same question the
        # moment a read fails: ``_durable_rows_cache`` is cleared on every
        # structural change (that is what invalidation means), while this is
        # the fallback a failed re-read is answered from -- a failed read must
        # never be published as an empty conversation list, which is the defect
        # this pair exists to stop. Only ever replaced by a read that succeeded.
        self._durable_rows_last_good: dict[str, Any] | None = None
        #: Whether the most recent durable re-read FAILED. Published beside the
        #: rows (see ``listing_degraded``) so the marker and the rows cannot
        #: disagree: it is set exactly on the failed path and cleared on the
        #: next successful one.
        self._durable_listing_degraded = False
        self._summaries_cache: list[dict[str, Any]] | None = None
        self._summaries_at = 0.0
        self._summaries_task: asyncio.Task[list[dict[str, Any]]] | None = None
        self._attention_states: dict[str, dict[str, Any]] = {}
        self._creation_dates: dict[str, float] = {}

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
        # ``_durable_rows_last_good`` is deliberately NOT cleared: this method
        # is about the cache being STALE, and a re-read that then fails has to
        # be answerable from the last true listing rather than from nothing.
        self._summaries_cache = None
        self._summaries_at = 0.0

    async def _refresh_durable_rows(self) -> dict[str, Any] | None:
        """Single-flight TTL refresh of the durable listing rows.

        Runs ``recent_session_rows`` OFF the event loop: it stats and reads a
        hundred session directories, which measured 300 ms to several seconds
        under contention — blocking work that froze every SSE stream on this
        loop while it ran. A concurrent caller joins the in-flight task
        instead of starting a second scan.

        ``strict=True``, because this listing is MEMBERSHIP on the phone: the
        client replaces its conversation list with this frame, so a store that
        cannot be walked must not be published as "you have no conversations".
        The scan raises instead, and the failure is answered with the last
        listing that WAS read (``_durable_rows_last_good``) plus the marker
        ``listing_degraded`` reports — never with an empty list. That is the
        same contract the desktop route states for its own refusal: keep the
        rows you have and retry; the difference is only that this side holds
        the rows, so it can honour it itself.

        Returns ``None`` when there is nothing to serve: the read failed and no
        listing has ever been read on this daemon (a cold start against a store
        that cannot be opened). The caller falls back to an empty listing, which
        is the only answer left -- and the marker says so, so a client can tell
        it apart from a store with no conversations in it.
        """
        from local_operator.paths import config_dir
        from local_operator.resume import recent_session_rows
        from local_operator.session.errors import SessionStoreUnavailable

        task = self._durable_rows_task
        if task is not None and not task.done():
            return await task

        # Active conversations may fall outside the bounded history listing.
        # Resolve their immutable dates in the same worker, never from the
        # heartbeat or by adding filesystem work to the in-memory merge path.
        live_ids = {entry.record.session_id for entry in self.entries.values() if not entry.ended}

        def load() -> tuple[dict[str, Any], dict[str, float]]:
            directory = config_dir()
            # ``recent_session_rows`` itself pays no creation-metadata reads —
            # it is on the CLI startup path. Each surface that needs birth
            # dates pays for them itself, and the two surfaces differ in HOW
            # because their row counts differ. These stable-order listings are
            # bounded and need a date for EVERY row, so they enrich eagerly
            # here. The /resume picker needs one only for the row under the
            # cursor, over an unbounded list, so it resolves them lazily and
            # caches per session id instead (``session/preview.py``,
            # ``SessionPreviews.created_at``) — measured, an eager loop there
            # cost 24.6% of the picker's open on a 151-session store.
            rows = {
                row.id: row._replace(created_at=session_created_at(directory / "sessions" / row.id))
                for row in recent_session_rows(directory, 100, strict=True)
            }
            dates = {}
            for session_id in live_ids - rows.keys():
                if session_id not in ("", ".", "..") and Path(session_id).name == session_id:
                    dates[session_id] = session_created_at(directory / "sessions" / session_id)
            return rows, dates

        async def _load() -> dict[str, Any]:
            rows, self._creation_dates = await asyncio.to_thread(load)
            return rows

        task = asyncio.ensure_future(_load())
        self._durable_rows_task = task
        try:
            rows = await task
        except SessionStoreUnavailable:
            # A store that exists but could not be walked. Logged (the operator
            # could not otherwise reconstruct why a phone list went quiet) and
            # answered from the last listing that was read -- never from an
            # empty one.
            logger.warning("phone listing could not read the session store", exc_info=True)
            if self._durable_rows_task is task:
                self._durable_rows_task = None
            self._durable_listing_degraded = True
            # Stamp the attempt: the TTL is a backoff here as much as a
            # freshness bound, or a store that stays unreadable would be
            # rescanned on every repaint of every phone screen.
            self._durable_rows_at = time.monotonic()
            return self._durable_rows_last_good
        except BaseException:
            # A failed scan must not poison the shared task: the next caller
            # retries instead of awaiting a raised future forever.
            if self._durable_rows_task is task:
                self._durable_rows_task = None
            raise
        self._durable_listing_degraded = False
        self._durable_rows_cache = rows
        self._durable_rows_at = time.monotonic()
        self._durable_rows_last_good = rows
        return rows

    def listing_degraded(self) -> list[str]:
        """What could not be read for the listing being published.

        Always a list, so a client reads it without a presence check, and empty
        when the durable half was read on the most recent attempt. The values
        are the phone's own (see ``DEGRADED_DURABLE_LISTING``) rather than the
        desktop's decoration names, because this reports the read the ROWS came
        from while those report decorations on rows that were read.
        """
        return [DEGRADED_DURABLE_LISTING] if self._durable_listing_degraded else []

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
            # The timestamp alone decides freshness, and it is stamped by a
            # FAILED attempt too: a store that cannot be walked must not be
            # rescanned on every repaint of every phone screen, and the fallback
            # below is what answers in the meantime.
            if time.monotonic() - self._durable_rows_at >= SUMMARIES_CACHE_TTL_S:
                rows = await self._refresh_durable_rows()
            if rows is None:
                # Nothing was ever read and the read is failing now, so there is
                # no listing to serve — empty, with ``listing_degraded`` naming
                # the read that failed so a client need not render that as "you
                # have no conversations".
                rows = self._durable_rows_last_good or {}
            from local_operator.session.attention import AttentionStore

            identities = {f"session/{session_id}" for session_id in rows}
            identities.update(
                f"session/{entry.record.session_id}" for entry in self.entries.values()
            )
            self._attention_states = await asyncio.to_thread(
                AttentionStore().state_many, identities
            )
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
                    # A SIGNALLED RUNTIME STREAMS TOO, so this list's own spinner
                    # said "working" about a session somebody had already asked
                    # to leave. The record's phrase is carried here so the
                    # phone's row ladder CAN say what it is instead of inferring
                    # it from ``streaming``; the field is additive, so a client
                    # that does not know it renders exactly as before (UX
                    # round 2, U8).
                    "leaving": (str(getattr(entry.record, "leaving", "") or "") if entry else ""),
                    # THE UPDATE WINDOW, carried the same way and for the same reason as
                    # the phrase one line up: an idle runtime moving to the build on disk
                    # is alive, accepting messages and about to run them, and the phone's
                    # row would otherwise describe it exactly as it describes an idle
                    # session — the one state the operator most needs to be told about,
                    # because it is the one where their message is queued rather than
                    # refused (``types.UPDATING``). The value is the build pair; a client
                    # that does not know the field renders exactly as before.
                    "updating": (str(getattr(entry.record, "updating", "") or "") if entry else ""),
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
                    "created_at": (
                        row.created_at if row else self._creation_dates.get(session_id, 0.0)
                    ),
                    "completion_kind": self._attention_states.get(f"session/{session_id}", {}).get(
                        "kind"
                    )
                    or "",
                    # Shared completion receipts, not transcript activity or
                    # heartbeat freshness, decide whether an outcome is unread.
                    "unseen": self._is_unseen(session_id, row, entry),
                }
            )
        out.sort(
            key=lambda summary: (
                summary["section"] != "active",
                session_category(
                    pending=summary["needs_attention"],
                    busy=summary["streaming"],
                    unseen=summary["unseen"],
                    kind=summary["completion_kind"],
                    live=summary["section"] == "active",
                ),
                # Token/heartbeat refreshes must not move a finger's target.
                -summary["created_at"],
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
        """Only completed outcomes count; subscribers and mtimes prove no read."""
        return bool(self._attention_states.get(f"session/{session_id}", {}).get("unseen", False))


def _classify_discovered_death(session_id: str, *, reaped_owner: Any | None = None) -> None:
    """Publish the durable outcome for a runtime the scan just found dead.

    WHY THE DAEMON HAS TO DO THIS. Every other writer of a session's durable
    outcome needs either a process that still exists or an open that happens
    after the fact: the dying runtime writes its OWN marker (so a SIGTERM is
    covered and a SIGKILL is not), a watching viewer journals what it witnessed,
    and ``Session.__init__`` classifies on the next open. A daemon-OWNED session
    killed while nobody watched therefore had NO record at all — no notice on the
    phone, no ``completion_kind`` for the list, and no outcome for the frame's
    ``stop_reason`` to be filled from (UX round 1, U1/U5).

    The same import, on the same worker thread, as ``_bootstrap_mobile_attention``
    already runs for up to 100 directories at boot — this is that sweep moved to
    the moment the death is DISCOVERED, which is the only new thing about it.
    The caller bounds it to one call per discovered death and to a daemon that
    dials (an observer daemon's contract is to write nothing), and
    ``bootstrap_transcript`` itself publishes nothing while a live owner holds
    the record — which is what makes the successor race harmless: a runtime that
    retired has already republished, so this classifies nothing and the
    successor's own outcome stands.

    ``reaped_owner`` is the dead record ``registry.scan`` reported, handed on to
    the classification so a caller that classified after its own sweep still
    carries the record — ``scan`` MOVES a dead record into the run namespace's
    ``reaped/`` sidecar rather than deleting it (and the classifier reads that
    sidecar too), so this is now belt-and-braces rather than the only road to
    an answer. Without it a caller whose sweep was what proved the pid dead
    has nothing left to hand over, and a future ``scan`` that moved a record
    somewhere this reader does not look would erase the evidence for the very
    death it just discovered (review round 2, MINOR-1).
    """
    from local_operator.session.attention import bootstrap_transcript
    from local_operator.session.transcript import Transcript

    directory = _durable_user_session_dir(session_id)
    if directory is None:
        return
    try:
        bootstrap_transcript(
            Transcript(directory, defer_materialise=True), reaped_owner=reaped_owner
        )
    except Exception:  # noqa: BLE001 — a listing must survive an unparsable transcript
        logger.debug("classifying a discovered death failed", exc_info=True)


def _bootstrap_mobile_attention() -> None:
    """Migrate the retained list once on daemon startup, not on passive reads."""
    from local_operator.paths import config_dir
    from local_operator.session.attention import bootstrap_transcript
    from local_operator.session.transcript import Transcript

    root = config_dir() / "sessions"
    if not root.is_dir():
        return
    # Match the relay's bounded recent-history surface. Older conversations are
    # imported when a real Session loads them, avoiding an unbounded boot scan.
    live = {record.session_id for record, state in registry.scan() if state == "live"}
    directories = sorted(root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
    for directory in directories[:100]:
        if directory.name in live:
            continue
        if _durable_user_session_dir(directory.name) is not None:
            bootstrap_transcript(Transcript(directory, defer_materialise=True))


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
    from local_operator.session.attention import AttentionStore
    from local_operator.tools.builtin import todo_snapshot

    directory = _durable_user_session_dir(session_id)
    if directory is None:
        return None
    try:
        state = _durable_fold_cache().load(directory)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 — an odd transcript yields no projection, not a 500
        _log_import_failure(
            exc, "local_operator.mobile.durable", where=f"durable fold for {session_id}"
        )
        return None
    projection = SessionProjection(
        session_id=session_id,
        pid=0,
        kind="daemon",
        conversation_name=stored_session_title(directory),
        attention=AttentionStore().state(f"session/{session_id}"),
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


#: How long the daemon accumulates unreadable-control-frame reports before it
#: emits one aggregated line.
OVERSIZED_CONTROL_WINDOW_S = 30.0


class _OversizedControlFrames:
    """Report the RATE of unreadable control frames, not each frame.

    Every one of these costs a session its live projection — a skipped frame
    leaves ``entry.projection`` stale, so the phone falls back to the durable
    disk fold and stops updating live — which makes the rate the signal worth an
    operator's attention. One line per frame is not a stronger signal, it is a
    buried one: measured on the operator's machine, 6,104,351 of these records
    were 78% of a 420 MB log file, and the other records in that file (the
    schedule, the MCP client, a stalled runtime) could not be read past them.

    The first sighting is always reported on its own, so a one-off is never lost;
    after that a run collapses into one line per window carrying the window count
    and a monotonic total.

    A flood that stops mid-window states neither figure by itself: the counts sit
    in memory until that pid queues another frame. That is the trade this makes,
    and it is worth naming rather than papering over — the stopped flood has
    already been seen once (its first sighting is unconditional) and its
    unreported residue is bounded by a single window, while the behaviour it
    replaces buried the file 6,104,351 times over. Flushing a stopped window would
    need a timer task inside the daemon, which is more machinery than the
    diagnostic is worth.
    """

    #: Ceiling on tracked pids. A long-lived daemon sees one entry per session it
    #: has ever dialled — ~32 MB at 100k pids — so the oldest is forgotten rather
    #: than kept forever. It is dropped by insertion order: the pid that has been
    #: known longest is the least likely to still be speaking, and a pid that
    #: speaks again is re-counted from its next first sighting.
    MAX_TRACKED_PIDS = 1024

    def __init__(self) -> None:
        self._windows: dict[int | None, tuple[float, int]] = {}
        self._totals: dict[int | None, int] = {}

    def note(self, pid: int | None) -> None:
        now = time.monotonic()
        started, count = self._windows.get(pid, (now, 0))
        total = self._totals.get(pid, 0) + 1
        self._totals[pid] = total
        if total == 1:
            self._windows[pid] = (now, 0)
            logger.warning(
                "mobile daemon: first oversized control frame from pid %s; each skipped "
                "frame leaves that session's projection stale",
                pid,
            )
            self._forget_oldest()
            return
        if now - started < OVERSIZED_CONTROL_WINDOW_S:
            self._windows[pid] = (started, count + 1)
            return
        self._windows[pid] = (now, 0)
        logger.warning(
            "mobile daemon: %d oversized control frame%s from pid %s over %.0fs "
            "(%d this process); each skipped frame leaves that session's projection stale",
            count + 1,
            "" if count + 1 == 1 else "s",
            pid,
            now - started,
            total,
        )

    def _forget_oldest(self) -> None:
        """Keep the tracked-pid map bounded when a daemon has seen many sessions."""
        while len(self._totals) > self.MAX_TRACKED_PIDS:
            oldest = next(iter(self._totals))
            self._totals.pop(oldest, None)
            self._windows.pop(oldest, None)


#: One instance per daemon process; the reader loop is single-threaded on one
#: event loop, so the counters need no lock.
_OVERSIZED_CONTROL_FRAMES = _OversizedControlFrames()


def _adopt_operator_handshake(entry: SessionEntry, frame: dict[str, Any]) -> None:
    """Verify the runtime's proof that it holds the capability this relay holds.

    The welcome is the only frame that carries it (see
    ``RuntimeServer._welcome_operator_proof``), and it is checked BEFORE any
    authority-increasing request is written for this connection. A record
    rewritten to point this dial at an impostor gets no proof, so nothing is
    presented to it — the harness/approval ``_proof`` rationale, applied to the
    relay's own socket rather than only to the attach client's.
    """
    # A REPAINT CARRIES NEITHER KEY; a handshake attempt carries at least one.
    # That, and not the salt's SHAPE, is what separates them — and the difference
    # is security-relevant rather than cosmetic: a frame with ``operator_salt``
    # present but unusable is exactly the impostor's answer to our nonce, and
    # treating it as a repaint would leave a previous handshake standing on a
    # connection that just failed one (agent review round 3, R3-3: the comment
    # claimed this while the guard checked ``is_wire_hex``).
    if "operator_salt" not in frame and "operator_proof" not in frame:
        # AN ORDINARY REPAINT, and it must leave this connection's authority
        # exactly as it is: only the WELCOME carries the handshake material
        # (``RuntimeServer._push_to``), while every projection push goes through
        # this loop. Clearing on each one destroyed the handshake 49-275 ms after
        # it was established, so the phone's next command was refused with the
        # authority copy — verified against a real socket, one ``_push()``
        # between the welcome and the request (agent review round 2 R2-2 = UX U6).
        return
    salt = frame.get("operator_salt")
    entry.operator_salt = ""
    entry.authority_bearing = False
    if entry.operator_cap is None or not entry.operator_nonce:
        return
    if is_wire_hex(salt) and handshake_proof_ok(
        supplied=frame.get("operator_proof"),
        held=entry.operator_cap,
        client_nonce=entry.operator_nonce,
        server_salt=str(salt),
    ):
        entry.operator_salt = str(salt)
        entry.authority_bearing = True


def _operator_request_proof(entry: SessionEntry, op: str, fields: dict[str, Any]) -> str | None:
    """This connection's proof for an authority-increasing frame, else ``None``.

    ``None`` covers every ordinary request — leaving the frame byte-identical to
    what an older runtime served — and every connection whose runtime never
    proved it holds the same capability.
    """
    if not entry.authority_bearing or entry.operator_cap is None:
        return None
    if frame_authority({"op": op, **fields}) != "authority-increasing":
        return None
    return request_proof(
        entry.operator_cap, client_nonce=entry.operator_nonce, server_salt=entry.operator_salt
    )


def _operator_handshake(entry: SessionEntry, op: str) -> str | None:
    """The handshake proof that lets a REPORT pick the true sentence.

    ``None`` on every ordinary op — leaving the frame byte-identical to what an
    older runtime served — and on every connection whose runtime never proved it
    holds the same capability. Only ``slash_result`` asks for it, because that is
    the op that builds a report whose wording depends on whether the READER may
    loosen (agent review round 3, R3-1 = UX U10).
    """
    if op != "slash_result" or not entry.authority_bearing or entry.operator_cap is None:
        return None
    return handshake_proof(
        entry.operator_cap, client_nonce=entry.operator_nonce, server_salt=entry.operator_salt
    )


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
    entry.ready.clear()
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
        # THE RELAY'S HALF OF THE OPERATOR HANDSHAKE (issue #1310, UX round 1
        # U3). The relay is a console for the sessions IT started — the phone's
        # "new session" path spawns the runtime from this process — and a
        # follower for everyone else's. Resolving that here, per dial, is what
        # makes the surface table's "works iff the relay spawned that runtime"
        # true rather than aspirational: this frame carried no nonce before, so
        # NO phone command could ever loosen a gate its own relay owned.
        entry.operator_cap = operator_cap_for(record.pid)
        # STAGE D also resolves the DEVICE tier here: whether this relay has an
        # operator-signed certificate for the phone answering right now, and which
        # challenge the runtime minted for it. The capability above stays as it is
        # — it is still the no-prompt path for a relay that spawned the runtime —
        # and it is exactly the path that was always ``None`` on this surface
        # (`mobile/daemon.py` never passes `--operator-fd`), which is why the
        # phone could not loosen before revision 2.
        entry.operator_nonce = operator_nonce() if entry.operator_cap is not None else ""
        entry.operator_salt = ""
        entry.authority_bearing = False
        auth: dict[str, Any] = {"key": record.control_key, "locality": "remote"}
        if entry.operator_nonce:
            # A nonce, never the capability: see ``harness/approval._proof``.
            auth["operator_nonce"] = entry.operator_nonce
        # AND THE PAIRED-DEVICE DECLARATION (stage D). A certificate in this
        # machine's store says a phone is authorised to sign for the operator, so
        # the runtime can answer "this connection may loosen" before any frame
        # arrives — which is what decides whether `/approvals` offers the phone a
        # command it can carry out or sends it looking for a window it does not
        # have. A certificate is PUBLIC data; the private half never leaves the
        # phone, so declaring one here grants nothing and the runtime verifies it
        # under the anchored operator key regardless.
        #
        # Absent when nothing is paired, and an older runtime ignores the field
        # entirely, which is why no PROTOCOL_VERSION moves.
        device_certificate = daemon.operator_device_certificate()
        if device_certificate:
            auth["operator_device"] = device_certificate
        writer.write(json.dumps(auth).encode() + b"\n")
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
                _OVERSIZED_CONTROL_FRAMES.note(record.pid)
                continue
            if not line:
                break
            try:
                frame = json.loads(line.decode("utf-8", "replace"))
            except ValueError:
                continue
            op = frame.get("op")
            if op in ("projection", "welcome"):
                _adopt_operator_handshake(entry, frame)
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
                entry.ready.set()
                if op == "welcome" and daemon.table.session_subscribers.get(record.session_id):
                    # A route can subscribe before registration. Replay only
                    # after the writer can actually accept the control op.
                    daemon.notify_watch_transition(record.pid, watching=True)
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
            entry.ready.clear()
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
    from local_operator.harness.rows import completion_notice
    from local_operator.mobile.projection import cap_projection_frame
    from local_operator.mobile.types import TranscriptEntry, advance_received_age

    # RE-DATE BEFORE SERIALIZING. The runtime re-dates its own frames, but this
    # copy is only as fresh as the frame it arrived on: during a long phase —
    # prose, a running call — no band event fires, so the runtime sends nothing
    # new and the age it named has been running down on this process's clock
    # ever since. Serving the stored number is what made a phone attaching
    # mid-phase paint `0s` counting from its own mount (review round 3, MAJOR
    # 1); the reference stamped at ingest says how long ago the reading was
    # taken, which is all the arithmetic needs. A projection with no reference
    # (a durable rebuild) is left as it is.
    advance_received_age(projection)
    data, degraded = cap_projection_frame(projection)
    attention = projection.attention
    data["attention"] = attention
    # A CUT-OFF'S END REACHES THE PHONE HERE OR NOWHERE. ``ProjectionFold``
    # learns ``stop_reason``/``cut_off`` from a folded ``AgentEndEvent``, and a
    # runtime that dies mid-turn never emits one — the follower's socket simply
    # closes, so the projection is left with the empty field that means "no turn
    # has ended yet". Measured on three real phone paths (attached with and
    # without another follower, and a daemon-OWNED session) × two signals
    # (SIGKILL/SIGTERM): ``stop_reason='' cut_off=False`` at every sample to
    # t+40 s, while a turn that COMPLETES does fold ``'completed'``. And
    # ``composer.tsx`` gates its whole resume affordance on
    # ``stop_reason === "aborted"``, so the notice and the list mark arrived and
    # the button that D7 exists to word never did — for a cut-off AND for a
    # deliberate stop issued from the phone (UX round 1, U1).
    #
    # The durable outcome this frame's notice is already built from carries the
    # end for exactly those arms, so the frame fills the MISSING end from it:
    # one record decides both the sentence and the button, which is what keeps
    # the word and the affordance from naming one act two ways (D7).
    # FILL, never override: the fold's own ABORT outranks a durable record. A
    # record may describe an EARLIER turn than the one the fold last saw, and
    # the fold is the only party that saw an end event for the current one — so
    # when it says `aborted`, its word and its `cut_off` flag stand, including
    # the deliberate stop it classified (`aborted` + `cut_off=False`).
    #
    # `"completed"` is NOT such an end for this purpose: a completion cannot be
    # the end being filled for (a completed turn publishes `kind='complete'`, so
    # the store would not be carrying an error), and a session that finished a
    # turn and then had the NEXT one stopped from the phone leaves exactly this
    # pair — `stop_reason='completed'` from the earlier fold plus an
    # `interrupted` outcome from the turn the fold never saw end. Requiring an
    # EMPTY field there silently withheld the button from a deliberate stop.
    if not projection.streaming and projection.stop_reason != "aborted":
        kind = str(attention.get("kind") or "")
        if kind in {"error", "interrupted"}:
            from local_operator.incidents import is_deliberate_cause

            data["stop_reason"] = "aborted"
            # WHICH word the button says. ``aborted`` covers both acts; only an
            # ``error`` kind that is not a recorded deliberate act is the
            # involuntary one. ``error`` with no cause at all is still a cut-off
            # — that is the "cause could not be determined" row, whose notice
            # above already says so.
            data["cut_off"] = kind == "error" and not is_deliberate_cause(
                str(attention.get("cause") or "")
            )
    if not projection.streaming and attention.get("kind") in {"error", "interrupted"}:
        # The sentence AND its severity come from `harness/rows.py`, which owns
        # row decisions for both surfaces: the phone's `NoticeRow` picks its
        # glyph and ink from `details.severity`, so a frame that carried an empty
        # `details` painted a cut-off as a routine `·` receipt — the same
        # flattening the TUI's poller had, on the surface the operator reads from
        # a phone (design review round 1, D4). The suppression above is unchanged
        # and still correct: a LIVE mid-turn session banners nothing, regardless
        # of the last outcome.
        text, severity = completion_notice(
            str(attention["kind"]), str(attention.get("reason") or "")
        )
        data["transcript"] = [
            *data["transcript"],
            TranscriptEntry(
                id=attention["anchor_id"],
                kind="notice",
                text=text,
                details={"severity": severity},
            ).to_json(),
        ]
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
        # Stamp the build THIS process loaded, before any lazy import can meet a
        # replaced tree: every later comparison in ``_log_import_failure`` is
        # against this value.
        _boot_build()
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
        self._phone_attaches: dict[str, AttachClient] = {}
        self._phone_generations: dict[str, tuple[int, str]] = {}
        self._phone_attach_tasks: dict[str, asyncio.Task[None]] = {}
        self._phone_handoffs: dict[str, asyncio.TimerHandle] = {}
        self._session_starts: dict[str, asyncio.Task[int]] = {}

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
        if not getattr(self, "_attention_bootstrapped", False):
            await asyncio.to_thread(_bootstrap_mobile_attention)
            self._attention_bootstrapped = True
        from local_operator.session.attention import AttentionStore

        revision = await asyncio.to_thread(AttentionStore().revision)
        if revision != getattr(self, "_attention_revision", None):
            self._attention_revision = revision
            # Receipts change the merged badge, not durable transcript metadata.
            self.table._summaries_cache = None
            self.table._summaries_at = 0.0
            self.table.notify_list_changed()
            watched = list(self.table.session_subscribers.items())
            attention_states = await asyncio.to_thread(
                AttentionStore().state_many,
                [f"session/{session_id}" for session_id, _ in watched],
            )
            for session_id, queues in watched:
                entry = _entry_for_session(self, session_id)
                projection = (
                    entry.projection
                    if entry is not None
                    else await asyncio.to_thread(_durable_projection, session_id)
                )
                if projection is not None:
                    if entry is None:
                        # A receipt changes no transcript generation. Reuse the
                        # relay's existing epoch/fencing before a cold repaint.
                        projection = self.capture_subagent_details(projection)
                    projection.attention = attention_states.get(
                        f"session/{session_id}", projection.attention
                    )
                    frame = _projection_frame(projection)
                    for queue in queues:
                        if queue.full():
                            queue.get_nowait()
                        queue.put_nowait(frame)
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
                # ONE classification per discovered death, gated on the TRANSITION
                # (this branch re-runs every scan for as long as the stale record
                # stays on disk, and a transcript import per 2 s pass is not a
                # price a list may pay). Only a daemon that owns sockets writes:
                # an observer daemon's whole contract is that it lists and serves
                # and touches nothing durable.
                first_sighting = not entry.ended
                entry.ended = True
                changed = True
                if first_sighting and self.dial_registrants:
                    # THE RECORD RIDES ALONG, because `registry.scan` has already
                    # unlinked it: this branch runs on the tuple scan RETURNED,
                    # and by the time we classify, the dead record the
                    # classification depends on is gone from the run directory.
                    # Re-reading (which is what `_run_record_evidence` does) then
                    # finds nothing and every discovered death lands the
                    # no-evidence arm — "the cause could not be determined" for
                    # the one shape where the daemon just PROVED the pid dead
                    # (review round 2, MINOR-1; the same sentence was measured on
                    # the phone by the design round, D6). Passing the record in
                    # is the evidence, and it is exactly as trustworthy as the
                    # scan that produced it.
                    await asyncio.to_thread(
                        _classify_discovered_death, record.session_id, reaped_owner=record
                    )
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
            if not entry.ended and self.table.session_subscribers.get(record.session_id):
                self._schedule_phone_attach(record)
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

    def operator_device_certificate(self) -> str:
        """A paired device certificate to DECLARE on this relay's auth frames, or "".

        Reads the same store the pairing flow writes (``operator/devices``), so a
        phone paired while the daemon is running is picked up on the next dial
        rather than needing a restart: the dial is the moment the value is read,
        and nothing caches it.

        It is a DECLARATION, not a credential. The relay forwards the phone's
        signature and can no more produce one having read this file than it could
        before — which is the property the whole stage turns on, and the reason
        this method is allowed to exist on a process that holds the portal
        password.
        """
        from local_operator.operator import devices
        from local_operator.paths import config_dir

        try:
            return devices.paired_certificate(config_dir()) or ""
        except (OSError, ValueError):  # pragma: no cover — an unreadable store
            return ""

    async def request(self, pid: int, op: str, **fields: Any) -> dict[str, Any]:
        """Send one control frame to a session and await its ack/error."""
        entry = self.table.entries.get(pid)
        if entry is None or entry.writer is None:
            raise KeyError(f"session {pid} is not connected")
        req = entry.next_req()
        # THE PHONE'S ROUTE TO THE SAME SOCKET, so it takes the same guard the
        # attach client does: a prompt relayed from the web composer carries
        # base64 images in one line, and an over-limit line makes the
        # registrant's reader discard the frame — the message would simply never
        # arrive. Shared helper rather than a second implementation, so the two
        # producers cannot disagree about what fits.
        #
        # Fitted BEFORE the future is registered: `OversizedRequest` is a
        # `ValueError`, which this module's HTTP layer already renders as a 422
        # the composer shows while RETAINING the user's command — so a refusal
        # must not leave a future parked in `_pending_reqs` for a frame that was
        # never written.
        from local_operator.mobile.attach_client import fit_request_frame

        frame: dict[str, Any] = {"op": op, "req": req, **fields}
        proof = _operator_request_proof(entry, op, fields)
        if proof is not None:
            frame["operator_cap"] = proof
        # STAGE D'S SEAM, marked rather than left to be found (revision 2, §5).
        # This is the relay's hand-written writer, and it is the ONE place a phone
        # signature has to be attached: the phone signs a challenge the runtime
        # minted for it, the relay forwards `operator_sig`/`operator_key_id`/
        # `operator_cert` here, and it must do so WITHOUT being able to mint any of
        # them — the certificate is operator-signed and the challenge is spent on
        # use. Nothing in this block may become a place where the relay computes a
        # signature of its own: the relay holds the portal password, and the whole
        # point of the device tier is that the password stops being authority.
        handshake = _operator_handshake(entry, op)
        if handshake is not None:
            frame["operator_handshake"] = handshake
        frame = await fit_request_frame(frame)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_reqs[(pid, req)] = future
        try:
            entry.writer.write(json.dumps(frame).encode() + b"\n")
            await entry.writer.drain()
            reply = await asyncio.wait_for(future, timeout=15.0)
        finally:
            self._pending_reqs.pop((pid, req), None)
        if reply.get("op") == "error":
            # A TYPED refusal when the frame names one, so the phone's HTTP layer
            # can carry the copy verbatim (its ``ValueError`` arm answers 422) and
            # a client can key on the category rather than on the sentence; any
            # other error keeps the bare message it always had.
            from local_operator.session.errors import admission_error

            # THE TRIGGER TRAVELS TOO (UX round 3, U11): it is what picks WHICH
            # refusal sentence is rebuilt, and without it the phone — the surface
            # the card copy was written for — was answered with the COMMAND's
            # sentence, about a command its user never typed, on a card that
            # survived. ``attach_client`` has always forwarded it; this writer is
            # the relay's own and simply did not.
            known = admission_error(
                str(reply.get("error_code", "")),
                reply.get("error_count"),
                reply.get("error_trigger"),
            )
            if known is not None:
                raise known
            raise RuntimeError(str(reply.get("message", "request failed")))
        return reply

    # -- owned sessions ---------------------------------------------------------

    def _phone_wanted(self, session_id: str) -> bool:
        return bool(
            self.table.session_subscribers.get(session_id)
            or session_id in self._phone_handoffs
            or session_id in self._session_starts
        )

    def _schedule_phone_attach(self, record: SessionRecord) -> asyncio.Task[None] | None:
        """One real viewer per OPEN phone route, never per adopted runtime.

        Modern runtimes count authenticated attach clients for idle residency;
        the daemon's permanent adoption dial and notification-only ``watch``
        ops do not. This bridge exists only while a human is opening/viewing
        the route, so an idle phone can think before sending its first prompt.
        """
        session_id = record.session_id
        if not self.dial_registrants or not self._phone_wanted(session_id):
            return None
        client = self._phone_attaches.get(session_id)
        generation = (record.pid, record.control_key)
        same_generation = self._phone_generations.get(session_id) == generation
        if client is not None and client.connected and same_generation:
            return self._phone_attach_tasks.get(session_id)
        task = self._phone_attach_tasks.get(session_id)
        if task is not None and not task.done():
            if same_generation:
                return task
            task.cancel()
        if client is not None:
            client.close()
        self._phone_generations[session_id] = generation
        task = asyncio.create_task(self._connect_phone(record))
        self._phone_attach_tasks[session_id] = task

        def finished(done: asyncio.Task[None]) -> None:
            if self._phone_attach_tasks.get(session_id) is done:
                self._phone_attach_tasks.pop(session_id, None)
            if not done.cancelled() and done.exception() is not None:
                logger.debug("phone viewer attach failed for %s", session_id)

        task.add_done_callback(finished)
        return task

    async def _connect_phone(self, record: SessionRecord) -> None:
        from local_operator.mobile.attach_client import AttachClient

        session_id = record.session_id

        def repaint(projection: SessionProjection) -> None:
            if self._phone_attaches.get(session_id) is not client:
                return
            if projection.session_id != session_id:
                # A terminal may rebind an owner after the welcome. This
                # phone is still viewing the old conversation; neither its
                # paint nor its readiness lease may follow the new one.
                client.close()
                self._phone_attaches.pop(session_id, None)
                self._phone_generations.pop(session_id, None)
                return
            entry = self.table.entries.get(record.pid)
            if entry is not None and entry.record.control_key != record.control_key:
                # This callback belongs to the current verified client (fenced
                # above), so it can replace a cached predecessor at a reused
                # PID. Old callbacks cannot overwrite the replacement.
                if entry.writer is not None:
                    entry.writer.close()
                old_dial = self._dial_tasks.pop(record.pid, None)
                if old_dial is not None:
                    old_dial.cancel()
                entry = None
            if entry is None:
                entry = SessionEntry(record)
                self.table.entries[record.pid] = entry
            # AttachClient verifies the welcome's durable identity before this
            # callback. Retain that first paint before acknowledging creation.
            entry.projection = self.capture_subagent_details(projection, record=record)
            self.table.notify_list_changed()
            _fan_out(entry, self)

        client = AttachClient(
            repaint,
            lambda _reason: None,
            locality="remote",
            # A PROMPT RAISED FOR A RELAY'S OWN SIGNATURE NAMES THE SESSION IN THE
            # LOG (UX round 6, U3 = design round 6, D3). There is no human at this
            # process's terminal — the person who asked is on the phone, and the
            # gesture is on the machine — so `warning` rather than the reader-facing
            # pane notice the TUI paints, and it is the only channel this surface
            # has. Deliberately passed rather than left to `AttachClient`'s
            # debug-level fallback: a machine-side presence prompt with nobody
            # looking is the event an operator most needs to find afterwards.
            on_operator_prompt=lambda copy: logger.warning("relay: %s", copy),
        )
        self._phone_attaches[session_id] = client
        try:
            await client.connect(record, session_id)
            if not self._phone_wanted(session_id):
                self.release_phone_view(session_id)
        except BaseException:
            if self._phone_attaches.get(session_id) is client:
                self._phone_attaches.pop(session_id, None)
            client.close()
            raise

    def claim_phone_view(self, session_id: str) -> None:
        """Transfer the startup lease to the already-registered SSE subscriber."""
        timer = self._phone_handoffs.pop(session_id, None)
        if timer is not None:
            timer.cancel()
        entry = _entry_for_session(self, session_id)
        if entry is not None:
            self._schedule_phone_attach(entry.record)

    def release_phone_view(self, session_id: str) -> None:
        """Drop readiness ownership after the LAST phone leaves; never abort work."""
        if self.table.session_subscribers.get(session_id):
            return
        timer = self._phone_handoffs.pop(session_id, None)
        if timer is not None:
            timer.cancel()
        task = self._phone_attach_tasks.pop(session_id, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        client = self._phone_attaches.pop(session_id, None)
        self._phone_generations.pop(session_id, None)
        if client is not None:
            client.close()

    async def close_phone_views(self) -> None:
        """Retire only this relay's owned startup/viewer resources on shutdown."""
        tasks: list[asyncio.Task[Any]] = [
            *self._session_starts.values(),
            *self._phone_attach_tasks.values(),
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for timer in self._phone_handoffs.values():
            timer.cancel()
        self._phone_handoffs.clear()
        for client in self._phone_attaches.values():
            client.close()
        self._phone_attaches.clear()
        self._phone_generations.clear()

    async def _prepare_phone_record(self, record: SessionRecord) -> None:
        task = self._schedule_phone_attach(record)
        if task is not None:
            await task
        entry = self.table.entries[record.pid]
        # Commands use the relay's remote-locality adoption dial. Its actual
        # welcome is the readiness publication; the viewer holds residency
        # while that handshake is in flight.
        dial = self._dial_tasks.get(record.pid)
        if dial is None or dial.done():
            entry.ready.clear()
            self._dial_tasks[record.pid] = asyncio.create_task(_dial(self, entry))
        await entry.ready.wait()

    def _arm_phone_handoff(self, session_id: str) -> None:
        prior = self._phone_handoffs.pop(session_id, None)
        if prior is not None:
            prior.cancel()
        if not self.table.session_subscribers.get(session_id):
            self._phone_handoffs[session_id] = asyncio.get_running_loop().call_later(
                PHONE_HANDOFF_S, self.release_phone_view, session_id
            )

    async def spawn_session(
        self,
        cwd: str,
        provider: str | None = None,
        model_id: str | None = None,
        resume: str | None = None,
    ) -> int:
        session_id = resume or uuid.uuid4().hex[:12]
        task = self._session_starts.get(session_id)
        if task is None:
            task = asyncio.create_task(self._spawn_session(cwd, provider, model_id, session_id))
            self._session_starts[session_id] = task

            def finished(done: asyncio.Task[int]) -> None:
                if self._session_starts.get(session_id) is done:
                    self._session_starts.pop(session_id, None)
                if not done.cancelled():
                    done.exception()  # retrieve failure even if the HTTP client left

            task.add_done_callback(finished)
        # Two resume requests share one constructor. A disconnected caller must
        # not cancel the other caller's operation; the startup deadline and
        # handoff lease still bound an abandoned request.
        return await asyncio.shield(task)

    async def _spawn_session(
        self, cwd: str, provider: str | None, model_id: str | None, session_id: str
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
        from local_operator.mobile.attach_client import find_runtime_record
        from local_operator.paths import config_dir

        # A durable resume can already have an owner (including one started
        # by another surface). Acknowledge that verified owner, never the PID
        # of a losing speculative constructor.
        existing, _ = await asyncio.to_thread(find_runtime_record, config_dir(), session_id)
        if existing is not None:
            try:
                async with asyncio.timeout(SESSION_START_TIMEOUT_S):
                    await self._prepare_phone_record(existing)
                self._arm_phone_handoff(session_id)
                return existing.pid
            except BaseException:
                self.release_phone_view(session_id)
                raise

        env = dict(os.environ)
        env["LOP_MOBILE_CHILD_CWD"] = cwd
        if provider:
            env["LOP_MOBILE_CHILD_PROVIDER"] = provider
        else:
            env.pop("LOP_MOBILE_CHILD_PROVIDER", None)
        if model_id:
            env["LOP_MOBILE_CHILD_MODEL"] = model_id
        else:
            env.pop("LOP_MOBILE_CHILD_MODEL", None)
        env["LOP_MOBILE_CHILD_RESUME"] = session_id
        # Unlike a viewer's synthesized birth seed these optional fields came
        # from the Start request itself, so they are deliberate overrides.
        if provider or model_id:
            env["LOP_MODEL_SELECTION_OVERRIDE"] = "1"
        else:
            env.pop("LOP_MODEL_SELECTION_OVERRIDE", None)
        # A deliberate Start is not speculative prewarming inherited from an
        # enclosing process. The child's existing adopt path mints this exact ID.
        env.pop("LOP_RUNTIME_DEFER_MATERIALISE", None)
        # BOTH name axes, exactly as the viewer's own spawn in
        # ``session/runtime/launch.py`` does — this is the PHONE-started runtime,
        # and until now it was the one session start with no branding at all, so
        # every session begun from the phone showed up as a bare ``python3.x``
        # row in Activity Monitor for its whole life. The session id is truncated
        # to 8 to match ``launch.py``, so `ps` correlates the same handle from
        # either surface; `spawn_identity` applies the label only alongside a
        # planted image, because a labelled `argv[0]` costs the child its
        # `sys.executable` on Linux (see `procname.spawn_identity`).
        from local_operator import procname
        from local_operator.interpreter import SAFE_PATH_FLAG
        from local_operator.session.runtime.types import RUNTIME_MODULE

        argv0, executable = procname.spawn_identity(
            procname.LABEL_SESSION_ANON, id=str(session_id)[:8]
        )
        process = await asyncio.create_subprocess_exec(
            # ``python_argv``'s shape with the label in element 0: the label
            # replaces only argv[0], and ``SAFE_PATH_FLAG`` stays at index 1
            # because interpreter options are recognised only before ``-m``.
            # The flag is what this spawn gets from ``python_argv``: it passes
            # no ``cwd=`` either, so without it a daemon started from a checkout
            # of this project would run that checkout instead of the install —
            # the same defect as the viewer's spawn in
            # ``session/runtime/launch.py``, and harder to notice here because
            # nobody is watching a phone daemon's version. See
            # :mod:`local_operator.interpreter`.
            argv0,
            SAFE_PATH_FLAG,
            "-m",
            # THE SPAWN CONTRACT, from its one home: the residency sweep's census
            # matches this module by this exact argv word, so a literal here could
            # drift from ``session/runtime/launch.py``'s without anything failing.
            RUNTIME_MODULE,
            executable=executable,
            env=env,
            # Detached stdio: the child speaks through its record and socket;
            # a pipe back to the daemon would die with the daemon and take
            # the child's stdout with it.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # REAL detachment, per platform. ``start_new_session=True`` is
            # documented "(POSIX only)" and the Windows ``_execute_child``
            # parameter is literally ``unused_start_new_session``: a child
            # spawned for a phone session would silently stay on this daemon's
            # console, where a console close takes it down — the one thing this
            # spawn exists to survive. The helper also keeps POSIX on
            # ``setsid`` (byte-identical behaviour there).
            **detached_popen_kwargs(),
        )

        async def ready() -> None:
            while True:
                record, _ = await asyncio.to_thread(find_runtime_record, config_dir(), session_id)
                if record is not None:
                    if record.pid != process.pid:
                        raise RuntimeError("another runtime acquired this session; retry resume")
                    await self._prepare_phone_record(record)
                    return
                # Records are filesystem publication, with no in-process
                # notifier. Polling observes that fact; timeout only bounds a
                # wedged constructor and is never treated as proof of readiness.
                await asyncio.sleep(0.05)

        readiness = asyncio.create_task(ready())
        exited = asyncio.create_task(process.wait())
        try:
            async with asyncio.timeout(SESSION_START_TIMEOUT_S):
                done, _ = await asyncio.wait(
                    {readiness, exited}, return_when=asyncio.FIRST_COMPLETED
                )
                if exited in done:
                    raise RuntimeError("session exited before becoming ready; check mobile logs")
                await readiness
                self._arm_phone_handoff(session_id)
                return process.pid
        except BaseException:
            self.release_phone_view(session_id)
            if process.returncode is None:
                # A known resume route can receive work from another surface
                # before this HTTP request finishes. Even a missing welcome is
                # not proof of an idle constructor: inbox/wake work starts
                # before publication. Only the owner's pristine-retire op may
                # decide to stop it. An unverifiable owner is left to its own
                # residency predicate, never blindly signalled by the relay.
                async def retire_if_pristine() -> None:
                    from local_operator.mobile.attach_client import AttachClient

                    record, _ = await asyncio.to_thread(
                        find_runtime_record, config_dir(), session_id
                    )
                    if record is None or record.pid != process.pid:
                        return
                    client = AttachClient(
                        lambda _projection: None,
                        lambda _reason: None,
                        locality="remote",
                        # Same reason as the phone attach above: the machine's key is
                        # what signs here and the log is this surface's only channel.
                        on_operator_prompt=lambda copy: logger.warning("relay: %s", copy),
                    )
                    try:
                        await client.connect(record, session_id)
                        await client.retire_if_pristine()
                    finally:
                        client.close()

                try:
                    await asyncio.wait_for(retire_if_pristine(), timeout=3)
                except (ConnectionError, RuntimeError, TimeoutError, OSError):
                    logger.debug("failed mobile startup left its owner to retire safely")
            raise
        finally:
            readiness.cancel()
            exited.cancel()
            await asyncio.gather(readiness, exited, return_exceptions=True)

    # -- slash command catalogue ----------------------------------------------------

    def slash_commands(self) -> list[dict[str, Any]]:
        """The phone's slash sheet. Imported lazily (the TUI registry pulls
        the app's command table) and cached — the registry is static."""
        if self._slash_commands is None:
            from local_operator.slash_commands import SLASH_COMMANDS
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

    class SessionEventResponse(StreamingResponse):
        async def stream_response(self, send) -> None:
            try:
                await super().stream_response(send)
            finally:
                # async-for does not close its generator when sending a chunk
                # raises. Explicit closure also covers cancelled proxy streams;
                # otherwise their unseen subscriber can pin a viewer forever.
                close = getattr(self.body_iterator, "aclose", None)
                if close is not None:
                    await close()

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

    async def _list_frame() -> dict[str, Any]:
        """The session-list payload, in ONE place because it goes out two ways.

        ``/api/sessions`` and the ``sessions`` event frame are the same answer
        on two transports, and the phone's home screen reads the SECOND — so a
        marker added to only one of them would be a marker the screen never
        sees. The name is the thing a client keys on, so it is spelled once.

        ``degraded`` is present on every frame, empty when the durable half was
        read: the same additive shape the desktop listing uses, so a client can
        tell "nothing to report" from "this server is too old to know" (see
        ``DEGRADED_DURABLE_LISTING``).
        """
        return {
            "sessions": await daemon.table.summaries(),
            "degraded": daemon.table.listing_degraded(),
        }

    async def api_sessions(request: Request) -> Response:
        denied = gate(request)
        if denied is not None:
            return denied
        return JSONResponse(await _list_frame())

    async def api_session_events(request: Request) -> Response:
        """SSE repaint stream for one session — the phone's only realtime
        channel. Opens with the current projection so a reconnecting phone
        renders immediately."""
        denied = gate(request)
        if denied is not None:
            return denied
        session_id = str(request.path_params["session_id"])

        async def stream():
            # A response whose headers fail to send never began viewing. Own
            # the queue only inside the iterator, paired with its finally.
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=8)
            subscribers = daemon.table.session_subscribers.setdefault(session_id, set())
            first_watcher = not subscribers
            subscribers.add(queue)
            try:
                daemon.claim_phone_view(session_id)
                live = _entry_for_session(daemon, session_id)
                if live is not None and first_watcher:
                    # Notification routing is distinct from idle readiness;
                    # mixed-version registrants may reject this optional op.
                    daemon.notify_watch_transition(live.record.pid, watching=True)
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
                        daemon.release_phone_view(session_id)
                        live = _entry_for_session(daemon, session_id)
                        if live is not None:
                            daemon.notify_watch_transition(live.record.pid, watching=False)
                        daemon._prune_projection_generation(session_id)

        return SessionEventResponse(
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
                yield _sse("sessions", await _list_frame())
                while True:
                    try:
                        await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_S)
                        yield _sse("sessions", await _list_frame())
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
        from local_operator.session.attention import (
            SUPERSEDED_TOKEN_CODE,
            AttentionStore,
            SupersededCompletionToken,
        )

        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            return JSONResponse(
                {"error": "completion_token is required; update the client"}, status_code=422
            )
        token = body.get("completion_token") if isinstance(body, dict) else None
        if not isinstance(token, str):
            return JSONResponse({"error": "completion_token is required"}, status_code=422)
        try:
            state = await asyncio.to_thread(
                AttentionStore().acknowledge, f"session/{session_id}", token
            )
        except SupersededCompletionToken:
            # A REAL token that a newer completion has replaced. Its own 409
            # rather than the unknown-token sentence, because the phone's remedy
            # differs: the completion it is looking at is no longer the one the
            # conversation is asking about, so re-reading the projection and
            # acknowledging the token it now names is what clears the mark. The
            # body carries the machine code for exactly that branch (§ Read APIs
            # and transports in docs/ATTENTION.md).
            return JSONResponse(
                {
                    "error": "completion token superseded by a newer completion",
                    "code": SUPERSEDED_TOKEN_CODE,
                },
                status_code=409,
            )
        except ValueError:
            return JSONResponse({"error": "unknown completion token"}, status_code=409)
        # The next list paint must already show the authoritative verdict.
        daemon.table.invalidate_summaries_cache()
        daemon.table.notify_list_changed()
        return JSONResponse({"ok": True, "attention": state})

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
        # THE OPERATOR CAPABILITY IS NOT PART OF THE REMOTE CONTRACT (issue
        # #1310). It is a LOCAL process fact: the relay attaches it itself
        # (``AttachClient._present_authority``) when this process is the one
        # that started the runtime, so a value arriving in an HTTP body can only
        # be a forgery attempt. Dropped rather than refused so a client that
        # sends one learns nothing about the field's shape — and so the
        # endpoint's error surface is unchanged for every ordinary request.
        body.pop("operator_cap", None)
        # ...AND THE SIGNATURE FIELDS ARE ADMITTED (stage D, revision 2 §2.4,
        # §4.2). This is the NARROWING the earlier revision's comment marked as
        # its own reversal point, and the direction is the whole design: the
        # phone cannot be a signer while the relay refuses to carry its
        # signature. `operator_cap` above stays dropped forever — it is
        # MACHINE-HELD proof material, the relay mints its own when it is the
        # spawner, and a value arriving in an HTTP body can only be a forgery.
        #
        # The three admitted fields are a different class:
        #
        # * `operator_sig` is an ES256 signature over a challenge THIS runtime
        #   minted for THIS connection, action and request id; a local attacker
        #   who replays one gains nothing, because the challenge is single-used
        #   and popped before verification (`server._operator_signature_verdict`)
        #   — the second presentation finds no challenge at all;
        # * `operator_cert` is a PUBLIC statement the operator signed; presenting
        #   it proves nothing without the device's private half, which never
        #   leaves the phone;
        # * `operator_key_id` only routes which key to try.
        #
        # None of the three can be MINTED here, which is the property that keeps
        # the relay a courier: it holds the portal password and nothing else that
        # can produce a signature. `operator_handshake` is deliberately NOT in
        # this list and never arrives from a caller — the relay computes its own
        # in `request()` and overwrites whatever a body carried, so a forged one
        # is not refused, it is replaced.
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
            # THE TYPED CODE TRAVELS WITH THE COPY (UX round 6, U6 = design round 6,
            # D5's mechanism). The phone is a REMOTE surface whose next step depends
            # on WHICH refusal this is — "operator authority is not installed on that
            # machine" is a different instruction from "that machine refused your
            # signature" — and it was left matching substrings of English that this
            # branch had already rewritten twice. `error` keeps the copy verbatim for
            # every client that exists today; `code` is additive and carries the
            # category alone, which is the same rule the module boundary states for
            # frames: an enumerated token, never prose this side composed.
            body: dict[str, Any] = {"error": str(exc)}
            code = getattr(exc, "code", "")
            if isinstance(code, str) and code:
                body["code"] = code
            return JSONResponse(body, status_code=422)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        return JSONResponse({"ok": True, "detail": reply.get("detail", "")})

    async def api_operator_challenge(request: Request) -> Response:
        """Mint a per-action operator challenge for THIS session's live connection.

        THE PHONE'S HALF OF THE SIGNING FLOW (stage D, revision 2 §2.3). The phone
        cannot ask the runtime directly — the runtime's control socket is loopback
        and speaks the record key, which lives on this machine — so it asks us and
        we relay one ordinary frame. Ordinary is the load-bearing word: the op
        grants nothing by itself, so it rides the record key like any other control
        request and needs no authority of its own. The SIGNATURE the phone then
        produces is what carries authority, and only the runtime can judge it.

        THE CHALLENGE MUST BE MINTED ON THE CONNECTION THE FRAME WILL ARRIVE ON,
        and that is why this goes through ``daemon.request`` rather than opening
        anything of its own: the runtime binds a challenge to
        ``(connection, session_id, action, request_id)``, and ``request()`` writes
        on the relay's single persistent connection per session. A challenge minted
        anywhere else would be refused — correctly.

        Only the CHALLENGE travels back. It is not a credential (it is worth
        exactly one signature, which only the paired phone can make), and holding
        one lets nobody sign: the private half never leaves the device.
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

        # THE AUTHORITY FIELDS ARE STRIPPED FROM THIS BODY, and only here. This
        # endpoint's whole output is a challenge, so a body carrying proof or
        # signature material is a caller confusing two endpoints — and relaying
        # it would send a half-finished frame through the ordinary path, where an
        # accidental `operator_cap` could be replayed where it means something.
        # The real command arrives on the command endpoint, which is the one that
        # was deliberately narrowed to admit signatures.
        for field in ("operator_cap", "operator_sig", "operator_key_id", "operator_cert"):
            body.pop(field, None)
        action = body.pop("action", None)
        if action not in ("loosen", "approve"):
            return JSONResponse({"error": "action must be 'loosen' or 'approve'"}, status_code=422)
        request_id = body.pop("request_id", "")
        if not isinstance(request_id, str):
            return JSONResponse({"error": "request_id must be a string"}, status_code=422)
        session_id = str(body.pop("session_id", "") or request.path_params.get("session_id", ""))
        entry = _entry_for_session(daemon, session_id)
        if entry is None:
            return JSONResponse({"error": "session not connected"}, status_code=409)
        try:
            reply = await daemon.request(
                entry.record.pid, "operator_challenge", action=action, request_id=request_id
            )
        except KeyError:
            return JSONResponse({"error": "session not connected"}, status_code=409)
        except TimeoutError:
            return JSONResponse({"error": "session did not answer"}, status_code=504)
        except (ConnectionError, OSError) as exc:
            return JSONResponse({"error": str(exc)[:200]}, status_code=502)
        except (ValueError, RuntimeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        challenge = reply.get("challenge")
        if not isinstance(challenge, str) or not challenge:
            return JSONResponse({"error": "the session sent no challenge"}, status_code=502)
        return JSONResponse(
            {
                "challenge": challenge,
                "expires_s": int(reply.get("expires_s") or 0),
                "session_id": entry.record.session_id,
                "action": action,
                "request_id": request_id,
            }
        )

    async def api_pair(request: Request) -> Response:
        """Claim a pairing code with a device's public key (stage D).

        A COURIER'S ENDPOINT, and the security argument is what it CANNOT do. It
        checks the code ``lop pair`` minted, records the device's public point,
        and answers — it holds no operator key, cannot obtain a signature, and
        therefore cannot make a device a signer. The certificate that does that
        is produced on the machine by the operator's own gesture and is written by
        ``lop pair``, not by anything reachable from here. A local attacker driving
        this endpoint with the portal password (matrix cell N2) can at most leave a
        pending request the operator must still refuse.

        The PRIVATE HALF IS NEVER SENT HERE, and the shape of the body is what
        enforces it: there is no field for it. The phone generates its key in
        WebCrypto with ``extractable: false``, so it could not export one even if
        this endpoint asked.
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

        from local_operator.operator import devices
        from local_operator.paths import config_dir

        root = config_dir()
        code = str(body.get("code") or "").strip()
        live_code = devices.read_pairing(root)
        if live_code is None:
            # NO LIVE CODE and A WRONG CODE are answered identically on purpose:
            # the distinction would tell a guesser whether a pairing window is
            # open on this machine, which is the one fact they need to time an
            # attempt.
            return JSONResponse({"error": "that pairing code is not valid"}, status_code=403)
        if not secrets.compare_digest(code, live_code):
            return JSONResponse({"error": "that pairing code is not valid"}, status_code=403)

        spki = devices.decode_spki(body.get("spki"))
        if spki is None:
            return JSONResponse(
                {"error": "spki must be an uncompressed P-256 public point, base64url"},
                status_code=422,
            )
        device_id = devices.new_device_id(spki)
        if devices.is_revoked(root, device_id):
            # A revoked device must not be able to re-pair on a fresh code and
            # quietly become a signer again. The renaming attack is closed by the
            # id being DERIVED from the key rather than chosen.
            #
            # `is_revoked`, not `is_revoked_here` (UX round 6, U5): the anchor is
            # the authoritative list, and an operator who revokes by editing it —
            # which is the documented way to revoke a device whose certificate
            # this machine still holds — left the local record silent. Measured
            # before the fix: that device re-paired on a fresh code with HTTP 200.
            return JSONResponse({"error": "this device has been revoked"}, status_code=403)
        name = str(body.get("name") or "")[:64]
        devices.write_pending(
            root,
            device_id=device_id,
            name=name,
            spki=devices.encode_spki(spki),
            code=live_code,
        )
        return JSONResponse({"ok": True, "device_id": device_id})

    async def api_pair_status(request: Request) -> Response:
        """Whether the operator has approved a claimed code yet.

        The phone polls this. It never returns private material — a certificate is
        a public statement — and it is the only way the device learns the string
        it has to present, since the certificate is minted on the machine rather
        than by anything the phone can reach.
        """
        denied = gate(request)
        if denied is not None:
            return denied
        from local_operator.operator import devices
        from local_operator.paths import config_dir

        device_id = str(request.path_params["device_id"])
        try:
            stored = devices.read_device(config_dir(), device_id)
        except ValueError:
            return JSONResponse({"error": "bad device id"}, status_code=422)
        if stored is None:
            return JSONResponse({"paired": False, "device_id": device_id})
        # WHETHER THE MACHINE CAN HONOUR A SIGNATURE AT ALL (UX round 6, U2).
        # The phone's success box promised authority it may not have: between
        # `lop operator init` (which stages the anchor) and `lop operator install`
        # (the privileged step that lands it) a correctly paired device signs and
        # the runtime refuses every one of them, because there is no key to verify
        # against. The portal cannot read the machine's filesystem, so the machine
        # has to say — and it is the same predicate the runtime and the pairing
        # receipt use, so the three surfaces cannot disagree.
        from local_operator.operator import operator_authority_unusable

        return JSONResponse(
            {
                "paired": True,
                "device_id": stored.device_id,
                "certificate": stored.certificate,
                "operator_key_id": stored.operator_key_id,
                "scope": list(stored.scope),
                "exp": stored.not_after,
                "name": stored.name,
                "authority_ready": not operator_authority_unusable(),
            }
        )

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
        # The route is a durable conversation identity, never a PID. Mint it
        # before spawn so the child, response and first SSE all agree.
        session_id = uuid.uuid4().hex[:12]
        try:
            pid = await daemon.spawn_session(
                cwd,
                provider=str(provider) if provider else None,
                model_id=str(model_id) if model_id else None,
                resume=session_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("mobile session spawn failed", exc_info=True)
            return JSONResponse({"error": str(exc)[:300]}, status_code=500)
        return JSONResponse({"ok": True, "pid": pid, "session_id": session_id})

    async def api_resume_session(request: Request) -> Response:
        """Reopen a past session as a NEW live session the phone can attach to.

        The old past-sessions flow made the user copy an id and run
        ``/resume <id>`` by hand. This is the button: spawn a daemon-owned
        child whose session resumes that transcript (the same ``--resume``
        mechanism the CLI uses), so the conversation comes back live, open,
        and able to take a command. The new session registers through
        discovery like any other; the phone keeps the durable session route.
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
        # spawn_session coalesces concurrent resume requests until the verified
        # owner is ready, and later retries reattach that same live owner.
        # The transcript dir does not reliably record a cwd, so resume in the
        # owner's home: always a valid directory under the spawn gate. The
        # user can steer the reopened session to a directory from there.
        try:
            pid = await daemon.spawn_session(str(Path.home()), resume=session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mobile resume spawn failed", exc_info=True)
            return JSONResponse({"error": str(exc)[:300]}, status_code=500)
        return JSONResponse({"ok": True, "pid": pid, "session_id": session_id})

    async def api_search_sessions(request: Request) -> Response:
        """Search past sessions by name, id, OR what was said in them.

        The same mechanism the TUI's /resume picker uses: a cached digest per
        session (search_index.build_index, re-digested only when a transcript
        changes) plus a substring match over name/id (filter_rows semantics).
        A row that matched only on its conversation body is marked so the
        phone can say why it surfaced.

        The same ``degraded`` marker the two listing routes carry, for the same
        reason: this is the route the shipped ``#/past`` screen renders, and its
        empty branch is what a failed read must not reach (see
        ``_search_sessions``).
        """
        denied = gate(request)
        if denied is not None:
            return denied
        query = request.query_params.get("q", "")
        try:
            limit = max(1, min(int(request.query_params.get("limit", "40")), 200))
        except ValueError:
            limit = 40
        rows, degraded = await asyncio.to_thread(_search_sessions, query, limit)
        return JSONResponse({"sessions": rows, "query": query, "degraded": degraded})

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
        rows, degraded = await asyncio.to_thread(_past_sessions)
        # The same marker the home listing carries, for the same reason: this is
        # another list of the operator's conversations, and a read that failed
        # must not reach it as "there are none".
        return JSONResponse({"sessions": rows, "degraded": degraded})

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
        return _maybe_gzip(request, JSONResponse({"models": models}))

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
        Route(
            "/api/sessions/{session_id:str}/operator/challenge",
            api_operator_challenge,
            methods=["POST"],
        ),
        Route("/api/pair", api_pair, methods=["POST"]),
        Route("/api/pair/{device_id:str}", api_pair_status),
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

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            await daemon.close_phone_views()

    return Starlette(routes=routes, lifespan=lifespan)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _past_sessions(limit: int = 20) -> tuple[list[dict[str, Any]], list[str]]:
    """Resumable past sessions for the phone's history list.

    ``forked`` rides along for the same reason the TUI picker draws it: a fork
    that has not named itself yet displays the title it inherited, so it and its
    parent are byte-identical rows — same name, same age — separable only by a
    12-hex id. The row builder already knows the fact (it is derived from the
    ``origin.json`` the scan parsed), and dropping it here is what would make
    the phone the one surface still showing the twin rows.

    STRICT, and the marker comes back with the rows for the reason the home
    listing is strict: this is a list of the operator's conversations, so a
    store that cannot be walked may not be answered as "there are none" — and
    the caller has to be able to tell the two apart, which is what the returned
    marker is for.

    The broad ``except Exception: return []`` this replaces is gone rather than
    narrowed. Swallowing everything made every failure look like an empty
    history, including the ones that are bugs — the same laundering the search
    path next door refuses in its own docstring. The one failure this function
    can actually answer for is the store read, and that is the one it catches;
    anything else is a defect and must reach the log as one.
    """
    from local_operator.paths import config_dir
    from local_operator.resume import recent_session_rows
    from local_operator.session.errors import SessionStoreUnavailable

    try:
        rows = recent_session_rows(config_dir(), limit=limit, strict=True)
    except SessionStoreUnavailable:
        logger.warning("phone history listing could not read the session store", exc_info=True)
        return [], [DEGRADED_DURABLE_LISTING]
    return [
        {"id": row.id, "name": row.name, "mtime": row.mtime, "forked": row.forked} for row in rows
    ], []


def _search_sessions(query: str, limit: int = 40) -> tuple[list[dict[str, Any]], list[str]]:
    """Past sessions matching ``query`` by name, id, or conversation body.

    One call into ``session_search.search_store``, which is the SAME admission,
    soft-matching and ranking the TUI's ``/resume`` picker and the desktop chat
    search use. Before this it was a private second implementation: name/id and
    an EXACT body substring only, no typo/prefix tier and no ranking, so a query
    the picker resolved confidently found nothing on the phone. The phone's own
    composition is only what it RENDERS from the answer — the dict shape below
    is the wire format, not a second filter.

    ``body_match`` marks a row the conversation surfaced (exact body, a past
    name, or a soft match) rather than its visible name, so the phone can say
    why it is on screen instead of showing a row with no visible reason.

    STRICT, and the marker comes back with the rows for the same reason the
    history route's does: this is the query the shipped ``#/past`` screen runs
    on mount (``mobile/web/src/screens/past-sessions.tsx``, with an empty
    ``q``), and it renders the answer as the WHOLE list. An unreadable store
    delivered here as zero matches is the membership lie this change exists to
    stop — "no past sessions yet" about a history that was never read.
    ``search_store`` tolerates that ``OSError`` by design for the display-only
    callers; the phone's history screen is not one of them any more, so it asks
    for ``strict=True`` and answers with the marker instead.

    What is NOT caught here is anything else — a bug in the search must not be
    laundered into a confident "nothing matched".
    """
    from local_operator.paths import config_dir
    from local_operator.session.errors import SessionStoreUnavailable
    from local_operator.session.session_search import search_store

    try:
        matches = search_store(config_dir(), query, limit=limit, strict=True)
    except SessionStoreUnavailable:
        logger.warning("phone search could not read the session store", exc_info=True)
        return [], [DEGRADED_DURABLE_LISTING]
    return [
        {
            "id": match.row.id,
            "name": match.row.name,
            "mtime": match.row.mtime,
            "body_match": match.body_match,
            "forked": match.row.forked,
        }
        for match in matches
    ], []


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


def _provider_display_name(provider_id: str) -> str:
    """The registry's human name for ``provider_id`` (the id when it has none).

    The unavailable-catalogue message names providers the way the owner met
    them in ``/login``, which is the registry's ``name`` — the id is an
    implementation spelling and reads as a typo in an error the phone shows.
    """
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition(provider_id)
    return definition.name if definition is not None else provider_id


#: Below this, compression costs more than it saves: the gzip header and the CPU
#: on both ends are not repaid by a few hundred bytes, and every small JSON
#: response on this daemon is well under it.
_GZIP_MIN_BYTES = 1024


def _accepts_gzip(accept_encoding: str) -> bool:
    """Whether ``Accept-Encoding`` asks for gzip, per RFC 9110 §12.5.3.

    A substring test cannot tell ASKING FOR gzip from REFUSING it: ``gzip;q=0``
    is the spec's way of saying "not acceptable", and reading it as consent
    served a compressed body to a client that had explicitly declined one. That
    is not only a spec violation — a client which does not auto-decode (``urllib``
    does not) gets a ``UnicodeDecodeError`` on the gzip magic bytes rather than
    JSON.

    Deliberately requires gzip to be named EXPLICITLY: a lone ``*`` is left
    un-compressed exactly as before. RFC 9110 would permit treating the wildcard
    as consent, but that would newly compress for clients this daemon has always
    answered in the clear, which is a behaviour change this fix has no reason to
    make. ``identity``, ``deflate``, ``br`` and an absent header keep answering
    uncompressed for the same reason.
    """
    for part in accept_encoding.split(","):
        token, _, params = part.strip().partition(";")
        token = token.strip().lower()
        # ``x-gzip`` is the historical spelling of the same coding.
        if token not in {"gzip", "x-gzip"}:
            continue
        quality = 1.0
        for param in params.split(";"):
            key, _, value = param.partition("=")
            if key.strip().lower() != "q":
                continue
            try:
                quality = float(value.strip())
            except ValueError:
                # An unparseable qvalue is not consent to ignore it; the entry
                # is malformed, so fall back to "not acceptable" rather than
                # compressing on a guess.
                quality = 0.0
        if quality > 0:
            return True
    return False


def _maybe_gzip(request: Any, response: Any) -> Any:
    """Gzip ``response`` in place when the client accepts it and it is worth it.

    PER-ROUTE, not a middleware, and that is the whole design constraint.
    ``GZipMiddleware`` wraps every response including
    ``/api/sessions/{id}/events``, which is a Server-Sent Events stream: gzip
    buffers, so the stream the phone relies on for live turn output would stop
    arriving event-by-event and arrive in compressed blocks instead — trading a
    transfer saving on one endpoint for a broken realtime surface on another.
    Applying it at the one route whose body is large and one-shot keeps the
    streaming routes byte-for-byte untouched.

    ``/api/models`` is that route: the sheet's catalogue is ~234 KB of JSON that
    compresses to ~20 KB, and the phone is typically on a mobile link through a
    tunnel, where that difference is seconds of an empty sheet.
    """
    # BEFORE any early return: a cache keys on the headers of the representation
    # it stored, so announcing this only on the compressed leg leaves the
    # identity response — the variant an intermediary is most likely to keep —
    # looking like the single valid answer for this URL, to be replayed to
    # clients that did ask for gzip and to clients that did not alike.
    response.headers["vary"] = "Accept-Encoding"
    if not _accepts_gzip(request.headers.get("accept-encoding", "")):
        return response
    # Starlette strips a HEAD response's body after the handler returns, so
    # compressing here would advertise the compressed LENGTH for a body the
    # client never receives.
    if getattr(request, "method", "GET").upper() == "HEAD":
        return response
    body = getattr(response, "body", b"")
    if not body or len(body) < _GZIP_MIN_BYTES:
        return response
    packed = gzip.compress(body, compresslevel=6)
    if len(packed) >= len(body):
        # Already-compressed or incompressible payloads grow by the gzip header.
        # JSON never reaches this, but the guard keeps the helper honest for any
        # future route: spending CPU to make a response BIGGER is never right.
        return response
    response.body = packed
    response.headers["content-encoding"] = "gzip"
    response.headers["content-length"] = str(len(packed))
    return response


class _UnreadableAuthStore:
    """A store that answers "I hold no rows" to the one question asked of it.

    Used ONLY when ``AuthStore()`` itself could not open the database, to build
    the cached catalogue that degradation serves. It exists so that path reaches
    ``usable_providers``'s own documented degradation instead of restating the
    rule: with no rows, no provider is claimed connected, and
    ``picker_rows(usable=None)`` lists everything rather than asserting the owner
    owns nothing.

    Every member but the listing raises, which is deliberate rather than lazy.
    This stand-in must never be mistaken for a working store: the only
    legitimate use is the read-only catalogue build below, and a caller that
    tries to log in or persist a credential through it has a bug that should be
    loud rather than silently written to nowhere. The members are spelled out
    (not a ``__getattr__`` catch-all) so it structurally satisfies
    ``ControllerAuthStore`` and a future addition to that protocol fails the
    type check here instead of at runtime on a degraded phone.
    """

    def _unavailable(self, operation: str) -> RuntimeError:
        return RuntimeError(
            f"the credential store is unreadable; {operation} is not available on "
            "the catalogue-only fallback"
        )

    def list_credentials(self, provider: str | None = None) -> list[Any]:
        return []

    def upsert_credential(self, provider: str, credential: dict[str, Any]) -> Any:
        raise self._unavailable("upsert_credential")

    def delete_credentials_for_provider(self, provider: str, disabled_cause: str = "") -> int:
        raise self._unavailable("delete_credentials_for_provider")

    def disable_credential(self, credential_id: int, cause: str) -> None:
        raise self._unavailable("disable_credential")

    def active_local_credential(self, provider: str, endpoint: str) -> Any:
        raise self._unavailable("active_local_credential")

    async def get_oauth_access(self, provider: str) -> Any:
        raise self._unavailable("get_oauth_access")

    async def list_oauth_accesses(self, provider: str) -> list[Any]:
        raise self._unavailable("list_oauth_accesses")

    def list_oauth_identities(self, provider: str) -> list[Any]:
        raise self._unavailable("list_oauth_identities")

    async def get_api_key(self, provider: str) -> str | None:
        raise self._unavailable("get_api_key")


def _model_rows(rows: "list[Any]") -> list[dict[str, Any]]:
    """Serialize ranked picker rows into the sheet's wire objects.

    The field set is what the phone RENDERS, deliberately. Shipping the whole
    row was 301 KB over a tunnel, 159 KB of which was
    ``context_window``/``input_price``/``output_price``/``routed`` — fields no
    ``.tsx`` in the bundle reads. A forward-looking payload is not free when the
    consumer is a phone on a mobile link; re-add a field here when a surface
    actually renders it.
    """
    return [
        {
            "selector": row.selector,
            "provider": row.provider,
            "model_id": row.model_id,
            # ``name`` is pre-existing and keeps its meaning: a DISPLAY name for
            # this model. It is sourced from ``listing_name`` — the listing's OWN
            # human name — rather than from ``label``, because ``label`` is the
            # picker's resolved form and ``naming._unambiguous_name`` refuses a
            # RESELLER's name there (the two shipped aggregators share ~398 of
            # ~400 names, so a name alone cannot say which route answers, and the
            # route is what differs in price and quota). That refusal is right on
            # the TUI, whose row paints a separate selector column; here the row
            # has two slots and the provider slot ALREADY carries the route, so
            # the same rule left 916 of 996 rows rendering ``anthropic/claude-
            # opus-5`` where the desktop renders ``Claude Opus 5``. Falling back
            # through ``label`` and then the id keeps a row that named nothing
            # rendering exactly as it did.
            "name": (
                row.listing_name
                or (row.label if row.label and row.label != row.selector else "")
                or row.model_id
            ),
            # ``label`` stays EXACTLY as the TUI spells it, unresolved names and
            # all — it is the parity contract, not a display fallback, and a
            # surface comparing the two must see the same string the desktop got.
            "label": row.label,
            "connected": row.connected,
            "aggregated": row.aggregated,
        }
        for row in rows
    ]


def _list_models() -> list[dict[str, Any]]:
    """The model sheet's rows: what the owner can run, ranked exactly as ``/model``.

    THE SAME CATALOGUE AND THE SAME ORDER AS THE DESKTOP, by construction rather
    than by convention. This used to walk ``model/registry.SupportedHostingProviders``
    and emit rows in registry order, which broke in three measurable ways:

    * ORDER. 962 rows went out grouped radient(445) > openai(12) > anthropic(18)
      > openrouter(445) > …, so ~445 aggregated Radient rows rendered before the
      first direct provider — roughly 45 phone screens of scrolling to reach
      ``anthropic/``. The sheet looked like it only knew Radient and OpenRouter.
      :func:`picker_rows` is the TUI's own ranking (direct-connected first,
      newest version first, aggregators last), so the two surfaces cannot drift.
    * COVERAGE. ``SupportedHostingProviders`` is the stale enumeration; the live
      one is ``providers.registry.PROVIDER_REGISTRY``. A phone therefore could
      not see ``alibaba-token-plan``, ``openai-device``, ``radient-key``,
      ``xai-oauth`` or ``zai-oauth`` at all, even fully logged in to them.
    * FRESHNESS. Non-aggregators were served from the SHIPPED registry, which
      offers ids the provider has since withdrawn (11 dead OpenAI ids the TUI's
      live catalogue does not list) and misses anything released after the last
      release of this package.

    Only PERSISTED credentials authorize a listing here — see
    :meth:`ProviderController.persisted_providers`. A service manager's ambient
    environment must never add an account to a picker reachable over a tunnel,
    which is exactly the rung that separates that method from ``usable_providers``.

    ONE catalogue per request, chosen by whether the credential question could
    be answered — NOT the picker's stale-then-update. The TUI paints
    ``initial_catalogue()`` and repaints on the live result because it has two
    frames to spend; this endpoint answers a single synchronous HTTP request, so
    a first catalogue would never be rendered and building one on the normal path
    is pure cost (measured: 1455 entries, 18 ms, discarded). The cached catalogue
    is therefore built only where it is the ANSWER — when the store could not be
    read and a live fetch would be unauthorized guessing.

    The live pass asks for :data:`PICKER_TTL_S` rather than discovery's 24 h
    default for the same reason the TUI does — opening the sheet is the one
    moment a fresh list is worth a request — and the fetch runs off the relay's
    event loop, which must never block on a provider round trip.
    """
    from contextlib import closing

    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.model.configure import _openai_use_max_context_window
    from local_operator.paths import config_dir
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.catalogue import picker_rows
    from local_operator.providers.controller import PICKER_TTL_S, ProviderController

    directory = config_dir()
    try:
        settings = dict(ConfigManager(directory).get_config().values)
    except Exception:  # noqa: BLE001 — an unreadable config must not empty the sheet
        settings = {}
    use_max_context = _openai_use_max_context_window(settings)
    try:
        store = AuthStore()
    except (sqlite3.Error, OSError) as exc:
        # The store could not be OPENED. This is the same rung
        # ``persisted_providers`` documents as ``None`` — "cannot tell" — and it
        # has to be caught HERE because that is where the read happens:
        # ``AuthStore.__init__`` connects eagerly, so an unreadable ``auth.db``
        # raised out of the constructor before the method with the degradation
        # was ever called, and the phone got a 502 carrying a raw SQLite string
        # instead of the cached list. 502-ing claims less than the app knows —
        # the disk cache still describes the catalogue — so the degradation
        # runs the same way the documented one does: show the cached models,
        # fetch nothing, because "which accounts may I speak for" is exactly the
        # question that just failed.
        logger.warning("credential store unreadable; serving the cached catalogue: %s", exc)
        # ``_UnreadableAuthStore`` rather than a bespoke branch: it makes
        # ``usable_providers()`` take its OWN documented degradation, so this
        # path produces exactly the catalogue an unreadable store already
        # produces one layer down — every model listed, none claimed
        # unconnected — instead of a second, drifting statement of that rule.
        controller = ProviderController(_UnreadableAuthStore())
        cached_rows, _cached_hidden = picker_rows(
            controller.initial_catalogue(),
            usable=None,
            use_max_context=use_max_context,
        )
        return _model_rows(cached_rows)
    with closing(store):
        controller = ProviderController(store, CredentialManager(config_dir=directory))
        admitted = controller.persisted_providers()
        statuses: dict[str, str] = {}
        if admitted is None:
            # The credential store could not be READ. Serving the cached
            # catalogue is right — an empty sheet would claim the owner owns no
            # models — but a live fetch is not, because "which accounts may I
            # speak for" is the question that just failed to resolve. This is
            # the ONE path that wants the cached catalogue, which is why it is
            # built here rather than unconditionally above: on every normal
            # request the live pass replaces it wholesale, so building it there
            # cost 1455 entries and ~18 ms per request for a value nothing read.
            entries = controller.initial_catalogue()
        else:
            # ``asyncio.run`` is safe here: ``api_models`` offloads this whole
            # synchronous helper to a worker thread, so there is no running loop
            # on it to clash with.
            entries, statuses = asyncio.run(
                controller.live_catalogue(ttl_s=PICKER_TTL_S, providers=admitted)
            )
        rows, _hidden = picker_rows(
            entries,
            usable=admitted,
            use_max_context=use_max_context,
        )
    if not rows:
        # A failed cold fetch is not an authoritative empty inventory — the same
        # rule this endpoint has always had, restated against the controller's
        # per-provider statuses. A provider counts as unavailable when it
        # contributed NO rows and did not say ``empty``: ``empty`` is the
        # provider itself answering "I list no models", which is a real answer,
        # while ``static``/``stale``/``unauthenticated`` on an aggregator (which
        # bundles nothing) means the listing never landed. Keep the message
        # credential-free while making a retry/re-login actionable.
        listed_providers = {entry.provider for entry in entries}
        unavailable = sorted(
            _provider_display_name(provider)
            for provider, status in statuses.items()
            if status != "empty" and provider not in listed_providers
        )
        if unavailable:
            raise RuntimeError(
                f"Model catalogue unavailable for {', '.join(unavailable)}; "
                "retry or log in again"
            )
    return _model_rows(rows)


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

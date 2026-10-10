"""The per-session code-request surface: what did this conversation do with PRs?

Two routes, one question each:

- ``GET  /v1/desktop/sessions/{session_id}/code-requests`` — this session's code
  requests, from the derived index MERGED with the fetch cache (state, lanes,
  CI). ``?include=mentions_tool`` expands the refs the scan collapsed because
  they appeared only in tool output.
- ``POST /v1/desktop/sessions/{session_id}/code-requests/refresh`` — the UI's
  refresh affordance, answered 202: it rescans the journal here and queues the
  FETCH half as a background pass (``force`` bypasses TTLs but not a cooling
  host), whose completion moves the index revision the feed frame carries.

**Why these routes do NOT take a session bridge.** Every other per-session desktop read
goes through ``host(request).session(...)`` because it needs the session's live state.
This one does not: the whole point of the derived index is that a cold reader can answer
without opening a session, and the index lives OUTSIDE the session directory for exactly
that reason. So the door is not involved, no bridge is built, and a session whose
runtime is asleep answers as fast as a live one. The refresh path is the same shape: it
runs the scanner against the journal on a worker thread, which needs the session's
directory and nothing else.

**GET never blocks on the network**, and in this slice it does not block on the DISK
either: a journal that has moved since the last scan gets a background refresh started
(``asyncio.create_task`` over the scanner's own thread hop) while the response carries
the last known rows and ``scan_state: "refreshing"``. On top of that, eligible rows —
dirty, past their TTL, or never fetched — schedule ONE single-flight FETCH pass whose
result reaches the client through the feed frame and the next GET. That is what makes a
poll cheap: one stat, one small JSON read, and tasks that are single-flight per session.

**What the caller may trust.** ``rows`` is derived, never authoritative: the event rows
in the transcript are the record, this file is a projection of them, and the fetch cache
is a revalidation of that projection against the forge. A deleted index heals on the
next scan, which is why the route reports ``scan_state`` rather than pretending the
answer is live.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Annotated, Any, Mapping

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as PathParam
from fastapi import Query, Request
from pydantic import BaseModel, ConfigDict, Field

from local_operator.code_requests import cache as code_requests_cache
from local_operator.code_requests import ledger, service
from local_operator.code_requests.hook import load_context_async
from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_code_requests import (
    CodeRequestListing,
    CodeRequestRefreshReceipt,
    CodeRequestRow,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import reply

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Desktop code requests"], dependencies=[Depends(require_desktop)])

#: The session-id shape every desktop session route declares, so a malformed handle is
#: refused by FastAPI before any handler runs.
SessionID = Annotated[str, PathParam(pattern=r"^[a-f0-9]{12}$")]

#: In-flight background scans, keyed by ``(config_dir, session_id)``. Single-flight per
#: session: a poll that arrives while a scan runs joins it rather than queueing another,
#: which is what keeps a 1 s client poll from becoming a 1 s disk scan. The strong
#: reference is held here on purpose — a bare ``create_task`` has only a weak referent and
#: can be collected mid-flight (the pattern ``transcript_index.start_refresh`` uses).
_IN_FLIGHT: dict[tuple[str, str], asyncio.Task[Any]] = {}


class RefreshBody(BaseModel):
    """``POST …/refresh`` body.

    ``extra="forbid"``: the fields are the contract, and a caller that misspells ``force``
    should learn that from a 422 rather than from a refresh that silently ignored it.
    """

    model_config = ConfigDict(extra="forbid")

    keys: list[str] = Field(default_factory=list)
    force: bool = False


def _config_dir(request: Request) -> Path:
    return Path(request.app.state.config_manager.config_dir)


def _session_dir(config_dir: Path, session_id: str) -> Path:
    return config_dir / "sessions" / session_id


def _session_exists(config_dir: Path, session_id: str) -> bool:
    return _session_dir(config_dir, session_id).is_dir()


def _revision(entry: Mapping[str, Any] | None) -> int:
    """A monotone per-session token for the index as it stands.

    Milliseconds of the index's own ``updated_at``, which the writer stamps from the
    wall clock: every rewrite moves it forward, and a client comparing two reads can tell
    "the list changed" from "the stream stuttered". Not a counter in memory, because a
    reader that never opened the session has to be able to derive it — and not the file's
    mtime, because a restore-from-backup sets that backwards.
    """
    if entry is None:
        return 0
    try:
        return int(float(entry.get("updated_at") or 0) * 1000)
    except (TypeError, ValueError):
        return 0


def _row_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    """One merged row (index facts + fetch overlay) as the wire model takes it.

    ``raw`` is a :func:`service.view_row` output: the index row's flat ref
    fields and relations, the folded ``mentions`` list, and — once the fetch
    cache has data for the ref — ``summary``/``lanes``/``fetched_at``/
    ``stale``/``refresh_error``. This function folds the mentions into the
    single ``mention`` object the client draws and otherwise passes fields
    through; unknown keys survive (``extra="allow"``) so a later slice can
    add one without touching this function.
    """
    mentions: list[Any] = raw["mentions"] if isinstance(raw.get("mentions"), list) else []
    sources = [str(item.get("source")) for item in mentions if isinstance(item, Mapping)]
    last_at = None
    count = 0
    first_at = None
    for item in mentions:
        if not isinstance(item, Mapping):
            continue
        try:
            count += int(item.get("count") or 0)
        except (TypeError, ValueError):
            continue
        for key, current in (("first_at", first_at), ("last_at", last_at)):
            value = item.get(key)
            if not isinstance(value, (int, float)):
                continue
            if key == "first_at":
                first_at = value if current is None else min(current, value)
            else:
                last_at = value if current is None else max(current, value)
    row: dict[str, Any] = {
        "key": str(raw.get("key") or ""),
        "url": str(raw.get("url") or ""),
        "forge": str(raw.get("forge") or ""),
        "host": str(raw.get("host") or ""),
        "project": str(raw.get("project") or ""),
        "number": int(raw.get("number") or 0),
        "relation": str(raw.get("relation") or "mentioned"),
        "relations": list(raw.get("relations") or []),
        "acted": list(raw.get("acts") or []),
        "mention": {"sources": sources, "count": count, "first_at": first_at, "last_at": last_at},
        "link_only": bool(raw.get("link_only", True)),
        "first_at": raw.get("first_at"),
        "last_at": raw.get("last_at"),
    }
    for key in (
        "via",
        "inherited_from",
        "unknown_reason",
        "evidence",
        "reason",
        "link_only_hint",
        "cooling_until",
        "summary",
        "lanes",
        "fetched_at",
        "stale",
        "refresh_error",
    ):
        if raw.get(key) is not None:
            row[key] = raw[key]
    return row


def _listing(
    config_dir: Path,
    session_id: str,
    *,
    include_tool_mentions: bool,
    scan_state: str,
) -> CodeRequestListing:
    # ``read_index`` returns ``None`` for an absent, unreadable or stale-schema file, and
    # the absent case is the common one (a session that never touched a pull request).
    # Normalising it to an empty mapping ONCE is what keeps every read below a plain
    # lookup: an ``isinstance`` guard per field is how one of them eventually forgets.
    entry: Mapping[str, Any] = ledger.read_index(config_dir, session_id) or {}
    rows_raw = entry.get("rows")
    # The merge (index facts + fetch overlay) lives in the service so the route,
    # the tool and a test all read ONE shape: ``view_rows`` adds the flat ref
    # fields, the folded fetch fields and the ``link_only`` verdict per row.
    merged = service.view_rows(
        config_dir,
        [
            item
            for item in (rows_raw if isinstance(rows_raw, list) else [])
            if isinstance(item, Mapping)
        ],
    )
    rows = [CodeRequestRow(**(_row_payload(item))) for item in merged]
    collapsed = 0
    truncated = False
    hints: list[dict[str, Any]] = []
    try:
        collapsed = int(entry.get("tool_output_only") or 0)
    except (TypeError, ValueError):
        collapsed = 0
    truncated = bool(entry.get("tool_only_truncated"))
    raw_hints = entry.get("hints")
    hints = [dict(item) for item in raw_hints] if isinstance(raw_hints, list) else []
    if include_tool_mentions:
        cached = ledger.read_cache(config_dir, session_id)
        scan = cached.get("scan") if isinstance(cached, Mapping) else None
        tool_rows = scan.get("tool_only_rows") if isinstance(scan, Mapping) else None
        if isinstance(tool_rows, list):
            merged_tool = service.view_rows(
                config_dir, [item for item in tool_rows if isinstance(item, Mapping)]
            )
            rows.extend(CodeRequestRow(**(_row_payload(item))) for item in merged_tool)
    return CodeRequestListing(
        session_id=session_id,
        revision=_revision(entry or None),
        rows=rows,
        tool_output_only_count=collapsed,
        tool_output_truncated=truncated,
        hints=hints,
        # The per-host cooling state (design §D.4): a host that answered
        # 403/429 is shown with the instant its retry lifts. In-process state,
        # like the rest of the cache's throttles.
        cooling=code_requests_cache.cooling_map(),
        scan_state=scan_state,
        updated_at=float(entry.get("updated_at") or 0) if entry else None,
    )


def _scan_state(config_dir: Path, session_id: str) -> tuple[str, bool]:
    """``("ready"|"stale"|"refreshing"|"missing", needs_refresh)`` for one session.

    "Stale" is a comparison of the journal's stat against the cache's recorded signature:
    equal means the index describes the file as it is, anything else means a scan is owed.
    The two callers differ only in what they DO about it — the route starts a background
    scan, the refresh route runs one in the foreground.
    """
    session_dir = _session_dir(config_dir, session_id)
    journal = ledger.transcript_path(session_dir)
    try:
        stat = journal.stat()
    except OSError:
        return ("missing", False)
    cached = ledger.read_cache(config_dir, session_id)
    if isinstance(cached, Mapping) and ledger.sig_matches(cached.get("sig"), stat):
        return ("ready", False)
    key = (str(config_dir), session_id)
    task = _IN_FLIGHT.get(key)
    if task is not None and not task.done():
        return ("refreshing", False)
    return ("refreshing", True)


def _context_cwd(config_dir: Path, session_id: str, session_dir: Path) -> str:
    """The cwd a scan's host context is built from: the session's own, else its directory.

    The session's recorded cwd is where the operator was actually working, which is what
    makes a self-hosted forge URL resolvable through the checkout's remotes. A session
    that records none (deleted from the runtime registry and never woke) keeps the older
    behaviour deliberately rather than borrowing THIS PROCESS's cwd — a daemon's own
    directory is the one directory the session certainly was not working in.
    """
    return ledger.session_cwd(config_dir, session_id) or str(session_dir)


def _start_scan(config_dir: Path, session_dir: Path, session_id: str, *, force: bool) -> None:
    """Start (or join) the single-flight background scan for one session. Never raises."""
    key = (str(config_dir), session_id)
    existing = _IN_FLIGHT.get(key)
    if existing is not None and not existing.done():
        return

    async def _run() -> None:
        try:
            context = await load_context_async(_context_cwd(config_dir, session_id, session_dir))
            await ledger.refresh_async(
                config_dir, session_id, session_dir, context=context, force=force
            )
        except Exception:  # noqa: BLE001 - a failed scan leaves the last answer standing
            logger.warning("code-request scan failed for %s", session_id, exc_info=True)
        finally:
            _IN_FLIGHT.pop(key, None)

    try:
        _IN_FLIGHT[key] = asyncio.get_running_loop().create_task(
            _run(), name=f"code-requests:{session_id}"
        )
    except RuntimeError:  # no running loop: nothing to schedule on, and that is fine
        logger.debug("no event loop to schedule a code-request scan on")


@router.get(
    "/v1/desktop/sessions/{session_id}/code-requests",
    response_model=CRUDResponse[CodeRequestListing],
)
async def code_requests(
    session_id: SessionID,
    request: Request,
    include: str | None = Query(default=None),
):
    """This conversation's code requests, from the derived index.

    A READ, and a cold one: no bridge, no runtime, no network. A journal that has moved
    since the last scan gets one background scan started and is answered from the last
    known rows with ``scan_state: "refreshing"`` — the client polls, and the next read is
    current. A session directory that does not exist is a 404: unlike a draft, there is
    no journal whose emptiness could make an empty list the honest answer.

    ``include=mentions_tool`` expands the refs the scan collapsed (those seen only in tool
    output). It is a query parameter rather than a second route because it is the same
    listing with one group unhidden, and a client that never expands it never pays for it.
    """
    config_dir = _config_dir(request)
    if not _session_exists(config_dir, session_id):
        raise HTTPException(
            404,
            {"code": "session_not_found", "message": f"no conversation with id {session_id}"},
        )
    scan_state, needs_scan = _scan_state(config_dir, session_id)
    if needs_scan:
        _start_scan(config_dir, _session_dir(config_dir, session_id), session_id, force=False)
    listing = await asyncio.to_thread(
        _listing,
        config_dir,
        session_id,
        include_tool_mentions=(include or "") == "mentions_tool",
        scan_state=scan_state,
    )
    # THE FETCH HALF (design §D.6), never awaited: when a row is eligible —
    # dirty, past its TTL, or never fetched — kick ONE background pass. The
    # response carries the cached state as it stands; the pass's completion
    # moves the index's ``updated_at``, which is the feed frame's revision, and
    # that frame is what makes the client refetch and draw the new data.
    await _maybe_kick_fetch_refresh(config_dir, session_id)
    return reply(listing.model_dump())


async def _maybe_kick_fetch_refresh(config_dir: Path, session_id: str) -> None:
    """Plan (cheap, off-loop) and schedule one background fetch pass. Never raises.

    The plan reads the dirty marks and one cache entry per row, so it runs on a
    worker thread; the schedule itself needs the loop that the route is on.
    """
    try:
        entry = await asyncio.to_thread(ledger.read_index, config_dir, session_id)
        rows = entry.get("rows") if isinstance(entry, Mapping) else None
        if not isinstance(rows, list) or not rows:
            return
        planned = await asyncio.to_thread(service.plan, config_dir, session_id, rows)
        if not planned.refs:
            return
        service.schedule_session_refresh(config_dir, session_id, rows)
    except Exception:  # noqa: BLE001 - a kick is an optimisation, never a fault
        logger.debug("could not schedule a code-request fetch refresh", exc_info=True)


@router.post(
    "/v1/desktop/sessions/{session_id}/code-requests/refresh",
    response_model=CRUDResponse[CodeRequestRefreshReceipt],
    status_code=202,
)
async def code_requests_refresh(session_id: SessionID, body: RefreshBody, request: Request):
    """Rescan the transcript, queue a fetch of the rows' remote state, answer 202.

    TWO HALVES, and the receipt is honest about both: the transcript scan runs
    here (its result is what the next list shows), and the FETCH half — the
    operator's press is exactly the design's ``POST … refresh`` with
    ``force`` bypassing TTLs but never cooling — is scheduled as one background
    pass because this route's contract is that reads never block on the
    network. Rows without a fetchable host or without a working login stay
    link-only, and the list says so per row.
    """
    config_dir = _config_dir(request)
    if not _session_exists(config_dir, session_id):
        raise HTTPException(
            404,
            {"code": "session_not_found", "message": f"no conversation with id {session_id}"},
        )
    session_dir = _session_dir(config_dir, session_id)
    started_at = time.time()
    try:
        context = await load_context_async(_context_cwd(config_dir, session_id, session_dir))
        await ledger.refresh_async(
            config_dir, session_id, session_dir, context=context, force=body.force
        )
        scanned = True
    except Exception:  # noqa: BLE001 - the receipt reports the attempt, never a traceback
        logger.warning("code-request refresh failed for %s", session_id, exc_info=True)
        scanned = False
    queued = False
    if scanned:
        entry = await asyncio.to_thread(ledger.read_index, config_dir, session_id)
        rows = entry.get("rows") if isinstance(entry, Mapping) else None
        if isinstance(rows, list) and rows:
            queued = service.schedule_session_refresh(
                config_dir,
                session_id,
                rows,
                keys=[key for key in body.keys if key] or None,
                force=body.force,
            )
    receipt = CodeRequestRefreshReceipt(
        session_id=session_id,
        accepted=scanned,
        keys=list(body.keys),
        force=body.force,
        note=(
            "Rescanned the session's transcript and queued a refresh of its code "
            "requests' state, comments and CI. Rows without a fetchable host or a "
            "working login stay link-only."
            if queued
            else "Rescanned the session's transcript. Nothing was queued for fetching "
            "(no tracked rows, or a fetch pass is already running)."
        ),
    )
    listed = f"{time.time() - started_at:.3f}s"
    logger.debug("code-request refresh for %s took %s", session_id, listed)
    return reply(receipt.model_dump())

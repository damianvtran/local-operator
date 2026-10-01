"""The machine-wide monitor surface: list, arm and cancel standing watches.

Three routes, one question each, over the monitor STATE the harness already
keeps:

- ``GET    /v1/desktop/monitors``                          — which conversations have monitors
- ``POST   /v1/desktop/monitors``                          — arm one on an existing conversation
- ``DELETE /v1/desktop/monitors/{session_id}/{monitor_id}``— cancel one

**Who writes.** ``Session._persist_monitor_schedules`` is the one writer of
monitor state, and this module never becomes a second one. It resolves the
OWNER first, exactly as the wake surface does, and reaches it the same way —
the wake twin's routed command ladder now has a monitor twin
(``serving._monitor_slash``, reached by ``route_shared_slash("monitor", …)``):

- a live, answering owner ⇒ the ladder: the mutation runs INSIDE the process
  that owns the schedules, so the in-memory list, the transcript entry and the
  derived index move together through the session's own machinery;
- no live owner ⇒ ``monitors/arm.py`` (transcript first, index second);
- an owner that cannot be used at all ⇒ a retryable 503 refusal, in the
  writer's own sentences, and NEVER a file write. That covers a wedged runtime
  (its process holds the transcript lease), a dialable owner whose ladder call
  came back with no answer, and a live process with no discovery record (the
  WRITER's own guard). An append from here would be deleted by that session's
  next persist (which republishes its whole in-memory list) while this route
  answered 200, and until then nothing would tick the watch either — the
  silently dead reminder both families' writers exist to remove.

**Why the listing is not a per-session field.** ``GET /v1/desktop/sessions`` is
ranked by recency and capped at 500, and a session armed once and never opened
has its mtime stamped at arm time — so a monitor armed months ago would
silently be absent from a listing built on that page. The monitor index is one
small file per monitor-carrying session, so this route is O(sessions with
monitors) and complete whatever the store's size.

**No request journal.** The wake surface receipts its creates because a
create+arm makes two durable things and a retry must answer the first attempt.
A monitor arm makes one, and its dedupe identity (``sha256(tool +
canonical-JSON(arguments))``, §11.4) is the at-most-once mechanism: a retried
arm finds the spec already in force and reports ``already_armed`` instead of
appending a second row; a retried cancel about a gone watch is an honest
``no monitor with id`` refusal. So these routes render refusals directly and
carry no ``request_id``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as PathParam
from fastapi import Query, Request
from pydantic import Field, StrictBool

from local_operator.monitors.arm import (
    WEDGED_MESSAGE,
    MonitorWriteError,
    MonitorWriteOutcome,
    arm_monitor,
    cancel_monitor,
)
from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_monitors import (
    MonitorEntry,
    MonitorListing,
    MonitorRow,
    MonitorWriteReceipt,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import Input, errors, host, reply

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Desktop monitors"], dependencies=[Depends(require_desktop)])

#: The listing's default page. NOT a UI page size — the realistic cardinality
#: is the number of monitor-carrying sessions on the machine — it exists so a
#: pathological store degrades visibly (``truncated``) instead of shipping an
#: unbounded document to a poller.
MONITOR_LIST_LIMIT_DEFAULT = 200
MONITOR_LIST_LIMIT_MAX = 500

#: The monitor-id shape the harness allocates (``m1``…``m16``). Declared as a
#: path pattern so a malformed handle is refused by FastAPI before any handler
#: runs, rather than being interpolated into a message or a lookup.
MonitorId = Annotated[str, PathParam(pattern=r"^m\d{1,4}$")]

#: What a refusal means, as an HTTP status. ONE table for every refusal this
#: module can surface, so the same mistake can never mean two different
#: statuses depending on which writer raised it.
_STATUS_FOR_CODE = {
    # 422 for input that could never have been valid (a bad duration, a bad
    # regex); 409 for a well-formed request that conflicts with what can be
    # watched (the interval floor, a past ``until``, a target that is not
    # read-only, the cap, the storm guard) — the writer's own split.
    "monitor_invalid": 422,
    "monitor_refused": 409,
    "monitor_not_found": 404,
    "session_not_found": 404,
    "monitor_unavailable": 409,
    "monitor_write_conflict": 409,
    # 409, NOT 503: the lock FILE could not be created (a read-only session
    # directory), so nothing was contended and a retry changes nothing until
    # the directory's mode does — the "conflict with the state itself" class.
    "monitor_write_unavailable": 409,
    # 503 for every "nothing was written; retrying is the fix": a contended
    # lock, and the owner states — an owner that cannot be used at all
    # (present-but-not-dialable, wedged), and a dialable owner whose ladder
    # call came back with no answer at all.
    "monitor_write_busy": 503,
    "monitor_owner_present": 503,
    "monitor_owner_wedged": 503,
    "monitor_owner_unavailable": 503,
}


class MonitorCreate(Input):
    """``POST /v1/desktop/monitors``.

    ONE shape, deliberately: the wake create's ``cwd``+``target`` variant
    exists because a wake must have a conversation to fire in. A monitor never
    engages a cold session (§10.4), so arming one against a conversation that
    does not exist yet would mint a phantom row that cannot tick until a person
    opens it — a session the user did not ask for. Arm-an-existing only; a
    create shape can grow later if a UI needs one, under its own review.

    The create fields are deliberately not bounded to the harness's own caps
    (``build_monitor_spec`` produces the sentences that own those numbers — a
    second bound here would answer the same mistake with pydantic's wording).
    The generous ceilings are only there so a body cannot be arbitrarily large.
    """

    session_id: str = Field(min_length=1, max_length=64)
    name: str | None = Field(default=None, max_length=200_000)
    tool: str | None = Field(default=None, max_length=200)
    arguments: dict[str, Any] | None = None
    every: str | None = Field(default=None, max_length=200)
    until: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=200_000)
    # ``StrictBool``, matching every other boolean on this plane: a generous
    # client's ``"yes"``/``1`` must get a 422, not silently arm a watch with a
    # notification setting nobody chose.
    notify: StrictBool | None = None
    sort_lines: StrictBool | None = None
    ignore: list[str] | None = None


@router.get("/v1/desktop/monitors", response_model=CRUDResponse[MonitorListing])
async def list_monitors(
    request: Request,
    limit: int = Query(default=MONITOR_LIST_LIMIT_DEFAULT, ge=1, le=MONITOR_LIST_LIMIT_MAX),
    include_dormant: bool = Query(default=True),
):
    """Every conversation on this machine that has monitors.

    Every disk read for the whole listing happens in ONE worker thread — an
    index read, a bounded name scan and a transcript existence check per entry,
    none of which may run on the event loop (a sidebar poll here must not be
    able to stall the loop that serves every other request).
    """
    root = request.app.state.config_manager.config_dir
    async with errors(request):
        return reply(await asyncio.to_thread(_collect_listing, root, limit, include_dormant))


@router.post("/v1/desktop/monitors", response_model=CRUDResponse[MonitorWriteReceipt])
async def create_monitor(body: MonitorCreate, request: Request):
    """Arm a monitor on an existing conversation (see ``MonitorCreate``).

    Not receipted: the dedupe identity makes a retry idempotent (see the module
    docstring), so the refusal IS the response.
    """
    async with errors(request):
        try:
            return reply(
                await _mutate(
                    request,
                    body.session_id,
                    op="create",
                    monitor_request=_monitor_request(body),
                )
            )
        except MonitorWriteError as error:
            _raise_refusal(error, session_id=body.session_id)


@router.delete(
    "/v1/desktop/monitors/{session_id}/{monitor_id}",
    response_model=CRUDResponse[MonitorWriteReceipt],
)
async def delete_monitor_route(session_id: str, monitor_id: MonitorId, request: Request):
    """Cancel one monitor.

    Cancelling the LAST monitor removes the index entry rather than writing an
    empty one, which is also what releases the cleanup reap guard the session
    held while it had something scheduled (§11.5).
    """
    async with errors(request):
        try:
            return reply(await _mutate(request, session_id, op="cancel", monitor_id=monitor_id))
        except MonitorWriteError as error:
            _raise_refusal(error, session_id=session_id)


# ---------------------------------------------------------------------------
# The listing (pure filesystem work; its caller moves it off the loop)
# ---------------------------------------------------------------------------


def _collect_listing(config_dir: Path, limit: int, include_dormant: bool) -> dict[str, Any]:
    from local_operator.monitors.store import is_held, read_index_report
    from local_operator.resume import session_name, session_origin
    from local_operator.wakes.supervisor import _session_exists

    index, read_error = read_index_report(Path(config_dir))
    now_ms = int(time.time() * 1000)
    entries: list[MonitorEntry] = []
    for session_id, raw in index.items():
        if not isinstance(raw, Mapping):
            continue
        # ``stopped_at`` is the one park marker monitors carry — there is no
        # Aida engine for them — and ``is_held`` is the store's own spelling
        # for it, so this listing and the cleanup guards cannot disagree.
        dormant = is_held(raw)
        if dormant and not include_dormant:
            continue
        session_dir = Path(config_dir) / "sessions" / session_id
        cwd = str(raw.get("cwd") or "")
        monitors = _monitor_rows(raw, now_ms, dormant)
        entries.append(
            MonitorEntry(
                session_id=session_id,
                name=_display_name(session_dir, session_id, cwd, session_name),
                cwd=cwd,
                origin=_origin(session_dir, session_origin),
                updated_at=_as_int(raw.get("updated_at")) or 0,
                dormant=dormant,
                # GHOST, asked with the supervisor's own predicate: the index
                # outlives the session directory it names. A dormant entry is
                # not a ghost — its session is deliberately parked, not gone.
                ghost=not dormant and not _session_exists(Path(config_dir), session_id),
                next_due_at=_next_due(monitors),
                monitors=monitors,
            )
        )
    # Soonest first, undateable last, ties by id: the same rule the wake
    # surface applies to the same values, so the two pages agree when both are
    # on screen.
    entries.sort(
        key=lambda entry: (entry.next_due_at is None, entry.next_due_at or 0, entry.session_id)
    )
    total = len(entries)
    return {
        "entries": [entry.model_dump() for entry in entries[:limit]],
        "generated_at": now_ms,
        "total": total,
        "truncated": total > limit,
        "read_error": read_error,
    }


def _monitor_rows(entry: Mapping[str, Any], now_ms: int, dormant: bool) -> list[MonitorRow]:
    from local_operator.monitors import store as monitor_store

    rows: list[MonitorRow] = []
    for raw in entry.get("monitors") or ():
        if not isinstance(raw, Mapping):
            continue
        due = _as_int(raw.get("next_due_at"))
        last = _as_int(raw.get("last_check_at")) or 0
        disabled = bool(raw.get("disabled"))
        until = _as_int(raw.get("until_at"))
        expired = until is not None and until <= now_ms
        # The CLI's precedence (``cli._monitor_state_word``): dormancy wins
        # over disabled, so a failure word cannot point a reader at the wrong
        # remedy; disabled wins over the clock, so a watch that does not tick
        # never reads as merely late. "expired" is the tool's word for the
        # third terminal state; the rendered clock is left to the client.
        if dormant:
            state = "dormant"
        elif disabled:
            state = "disabled"
        elif expired:
            state = "expired"
        else:
            state = "armed"
        arguments = raw.get("arguments")
        ignore = raw.get("ignore")
        rows.append(
            MonitorRow(
                id=str(raw.get("id") or ""),
                name=str(raw.get("name") or ""),
                tool=str(raw.get("tool") or ""),
                arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
                description=str(raw.get("description") or ""),
                every_ms=_as_int(raw.get("every_ms")),
                until_at=until,
                notify=bool(raw.get("notify")),
                sort_lines=bool(raw.get("sort_lines")),
                ignore=[str(item) for item in ignore] if isinstance(ignore, (list, tuple)) else [],
                cwd=str(raw.get("cwd") or ""),
                created_at=_as_int(raw.get("created_at")) or 0,
                next_due_at=due,
                last_check_at=last,
                checks=_as_int(raw.get("checks")) or 0,
                deliveries=_as_int(raw.get("deliveries")) or 0,
                consecutive_failures=_as_int(raw.get("consecutive_failures")) or 0,
                disabled=disabled,
                disabled_reason=str(raw.get("disabled_reason") or ""),
                due_in_s=None if due is None else (due - now_ms) / 1000.0,
                last_check_age_s=None if not last else max((now_ms - last) / 1000.0, 0.0),
                state=state,
                unavailable_since=monitor_store.unavailable_since_of(raw),
                health=monitor_store.health_hint({**raw, "next_due_at": due}, now_ms),
            )
        )
    rows.sort(key=lambda row: (row.next_due_at is None, row.next_due_at or 0, row.id))
    return rows


def _next_due(rows: list[MonitorRow]) -> int | None:
    return min((row.next_due_at for row in rows if row.next_due_at is not None), default=None)


def _display_name(session_dir: Path, session_id: str, cwd: str, read_name) -> str:
    """The conversation's name, or a floor built from its id and directory.

    The wake route's helper, kept verbatim: the shared half is
    ``resume.session_name`` (so one conversation is named one way on both
    pages), and this floor is the local fallback — duplicated rather than
    imported, because route modules do not reach into each other's privates.
    Best-effort by contract: a name is decoration, and a listing that answered
    500 because one transcript could not be read would hide every other monitor
    on the machine. The floor is not optional either — a nameless row is a row
    the user cannot identify or open with confidence.
    """
    try:
        name = read_name(session_dir)
    except Exception:  # noqa: BLE001
        logger.warning("could not read the name of %s", session_id, exc_info=True)
        name = ""
    if name:
        return name
    basename = Path(cwd).name if cwd else ""
    return f"{session_id[:8]} ({basename})" if basename else session_id[:8]


def _origin(session_dir: Path, read_origin) -> str:
    try:
        return read_origin(session_dir)
    except Exception:  # noqa: BLE001 — grouping metadata, never a reason to fail a listing
        return ""


def _as_int(value: Any) -> int | None:
    """A tolerant int read. Every field here comes from a file another process
    writes, and one malformed value must degrade a field, not the whole listing
    (a cast that raised would 500 the page)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


# ---------------------------------------------------------------------------
# Mutations: owner first, files only when there is no owner at all
# ---------------------------------------------------------------------------


def _monitor_request(body: MonitorCreate) -> dict[str, Any]:
    """The create half of a body, projected onto the fields
    ``build_monitor_spec`` reads.

    The same projection discipline as the wake route: the body is a SUBCLASS
    carrying ``session_id``, and a raw dump would hand the validator keys that
    belong to this route. ``build_monitor_spec`` reads only the keys it knows
    today — the projection is what keeps that true when one gains fields.
    """
    fields = (
        "name",
        "tool",
        "arguments",
        "every",
        "until",
        "description",
        "notify",
        "sort_lines",
        "ignore",
    )
    return {
        key: value for key, value in body.model_dump(exclude_unset=True).items() if key in fields
    }


async def _mutate(
    request: Request,
    session_id: str,
    *,
    op: str,
    monitor_request: dict[str, Any] | None = None,
    monitor_id: str = "",
) -> dict[str, Any]:
    root = request.app.state.config_manager.config_dir
    async with host(request).session(session_id) as bridge:
        assert bridge.remote is not None
        if bridge.remote.owner_reachable:
            return await _via_owner(
                root,
                bridge,
                session_id,
                op=op,
                monitor_id=monitor_id,
                monitor_request=monitor_request,
            )
        wedged = await asyncio.to_thread(_wedged, root, session_id)
        if wedged is not None:
            # NEVER WRITE AROUND A WEDGED OWNER: its process holds the transcript
            # lease, so a monitor written here could not tick — and the live
            # session would overwrite the append from its own in-memory list on
            # its next persist anyway. 503, with the writer's retryable sentence.
            raise MonitorWriteError(
                WEDGED_MESSAGE,
                status=_refusal_status("monitor_owner_wedged"),
                code="monitor_owner_wedged",
            )
        if op == "create":
            outcome = await arm_monitor(root, session_id, monitor_request or {}, cwd=bridge.cwd)
        else:
            outcome = await cancel_monitor(root, session_id, monitor_id)
    return _receipt(outcome)


async def _via_owner(
    config_dir: Path,
    bridge: Any,
    session_id: str,
    *,
    op: str,
    monitor_id: str,
    monitor_request: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run the mutation inside the process that owns the schedules.

    The runtime's ``monitor`` word is the ladder twin of the wake route's
    (``serving._monitor_slash``), and it ends in ``scheduler.create``/
    ``scheduler.cancel`` -> ``Session._persist_monitor_schedules``. Nothing
    here writes a transcript or an index entry: doing so is what would produce
    a watch the session's next persist deletes and nothing ticks.
    """
    payload = json.dumps({"op": op, "monitor_id": monitor_id, "request": monitor_request or {}})
    outcome = await bridge.remote.route_shared_slash("monitor", payload)
    if not isinstance(outcome, Mapping):
        # NOT a monitor refusal the runtime decided: the bridge answered
        # nothing at all (no ack shape from the owner). Typed for the client
        # all the same — it is raised before any write and its sentence asks
        # the caller to come back.
        raise MonitorWriteError(
            "The conversation's runtime answered nothing for this monitor.",
            status=_refusal_status("monitor_owner_unavailable"),
            code="monitor_owner_unavailable",
        )
    if outcome.get("kind") == "error":
        data = outcome.get("data") or {}
        code = str(data.get("code") or "monitor_refused")
        raise MonitorWriteError(
            str(outcome.get("text") or "The monitor was refused."),
            status=_refusal_status(code),
            code=code,
        )
    data = outcome.get("data") or {}
    root = Path(config_dir)
    # VERIFIED, not assumed (the wake route's rule): the owner's index write is
    # best-effort and swallows its failure (the transcript is what matters),
    # so this asks the file rather than reporting the intent.
    index_written = await asyncio.to_thread(
        _index_reflects, root, session_id, op, str(data.get("monitor_id") or monitor_id)
    )
    return _receipt(
        MonitorWriteOutcome(
            session_id=session_id,
            monitor_id=str(data.get("monitor_id") or ""),
            name=str(data.get("name") or ""),
            remaining=_as_int(data.get("remaining")) or 0,
            index_written=index_written,
            next_due_at=_as_int(data.get("next_due_at")) if op != "cancel" else None,
            already_armed=bool(data.get("already_armed")),
            reactivated=bool(data.get("reactivated")),
        )
    )


def _index_reflects(config_dir: Path, session_id: str, op: str, monitor_id: str) -> bool:
    from local_operator.monitors.store import read_entry

    try:
        entry = read_entry(Path(config_dir), session_id)
    except Exception:  # noqa: BLE001 — an unreadable index is reported, not raised
        return False
    ids = {
        str(raw.get("id") or "")
        for raw in (entry or {}).get("monitors") or ()
        if isinstance(raw, Mapping)
    }
    if op == "cancel":
        # An empty remainder REMOVES the entry (``store.write_entry``), so a
        # missing entry is exactly what a reflected cancel looks like.
        return monitor_id not in ids
    return monitor_id in ids


def _wedged(config_dir: Path, session_id: str) -> Any:
    """The supervisor's own wedge predicate, imported rather than re-derived so
    this route and the process that fires a wake cannot disagree about which
    sessions are unreachable — the monitor writer's guard asks the same one."""
    from local_operator.wakes.supervisor import wedged_runtime

    return wedged_runtime(Path(config_dir), session_id)


def _receipt(outcome: MonitorWriteOutcome) -> dict[str, Any]:
    return {
        "session_id": outcome.session_id,
        "monitor_id": outcome.monitor_id,
        "name": outcome.name,
        "next_due_at": outcome.next_due_at,
        "remaining": outcome.remaining,
        "already_armed": outcome.already_armed,
        "reactivated": outcome.reactivated,
        "receipt": "applied",
        "index_written": outcome.index_written,
    }


def _refusal_status(code: str, default: int = 409) -> int:
    """The status a refusal travels as (one table; see ``_STATUS_FOR_CODE``)."""
    return _STATUS_FOR_CODE.get(code, default)


def _refusal_detail(code: str, message: str, *, session_id: str = "") -> dict[str, Any]:
    """The refusal body, in the shape the wake surface's refusals use, so the
    desktop client unwraps either family with one reader."""
    detail: dict[str, Any] = {"code": code, "message": message}
    if session_id:
        detail["session_id"] = session_id
    return detail


def _raise_refusal(error: MonitorWriteError, *, session_id: str = "") -> NoReturn:
    """Render the writer's refusal on a route that has no receipt journal.

    ``error.status`` is the writer's own, and the table only overrides the
    codes it knows — the same rule the wake route applies to its refusals.
    """
    raise HTTPException(
        _refusal_status(error.code, error.status),
        _refusal_detail(error.code, str(error), session_id=session_id),
    )

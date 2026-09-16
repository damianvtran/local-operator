"""The machine-wide wake surface: list, arm, edit and cancel scheduled wakes.

Four routes, one question each, over the wake STATE the harness already keeps:

- ``GET    /v1/desktop/wakes``                          — which conversations have wakes
- ``POST   /v1/desktop/wakes``                          — arm one (creating the session on demand)
- ``PATCH  /v1/desktop/wakes/{session_id}/{wake_id}``    — reword or re-bound one
- ``DELETE /v1/desktop/wakes/{session_id}/{wake_id}``    — cancel one

**Who writes.** ``Session._persist_wake_schedules`` is the one writer of
schedule state, and this module never becomes a second one. It resolves the
OWNER first: a session with a live runtime is mutated through that runtime's
own command ladder (``serving._wake_slash``, reached by
``route_shared_slash("wake", …)``), so the in-memory list, the transcript and
the derived index move together under the scheduler's lock. A session with no
runtime — nothing to overwrite the change, nobody to ask — goes through
``wakes/arm.py``, the same helper the CLI now uses.

**Why not simply write files in both cases.** A live session republishes its
WHOLE in-memory list on its next persist (a fire, a retirement, the agent's own
``wake`` tool), which would delete an externally appended row from the
transcript AND the index; in the meantime the supervisor skips any session
with a live record, so the appended wake never fired either. That is a silently
dead reminder answered with a 200, which is the failure this whole subsystem
keeps removing. So the owner is asked, or the request is refused:

- no live owner  ⇒ ``arm.py`` (transcript first, index second, install hook);
- an owner that answers ⇒ the owner command;
- an owner that is WEDGED ⇒ 503 with a retryable sentence, never a file write
  (its lease blocks every engage, so a written wake could not fire either).

**Why a page-sized listing is not a per-session field.** ``GET
/v1/desktop/sessions`` is ranked by recency and capped at 500, and a session
armed once and never opened has its mtime stamped at arm time — so a schedule
armed months ago would silently be absent from a listing built on that page.
The wake index is one small file per wake-carrying session, so this route is
O(sessions with wakes) and complete whatever the store's size.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as PathParam
from fastapi import Query, Request
from pydantic import Field, model_validator

from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_wakes import (
    SupervisorInfo,
    WakeEntry,
    WakeListing,
    WakeScheduleRow,
    WakeWriteReceipt,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import (
    Input,
    RequestID,
    SessionTarget,
    errors,
    host,
    receipts,
    reply,
)
from local_operator.server.utils.desktop_sessions import read_desktop_marker
from local_operator.wakes.arm import WakeWriteError, arm_wake, cancel_wake, edit_wake

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Desktop wakes"], dependencies=[Depends(require_desktop)])

#: The listing's default page. NOT a UI page size — the realistic cardinality
#: is the number of wake-carrying sessions on the machine (9 on the reference
#: host) — it exists so a pathological store degrades visibly (``truncated``)
#: instead of shipping an unbounded document to a poller.
WAKE_LIST_LIMIT_DEFAULT = 200
WAKE_LIST_LIMIT_MAX = 500

#: The wake-id shape the harness allocates (``w1``…``w16``). Declared as a path
#: pattern so a malformed handle is refused by FastAPI before any handler runs,
#: rather than being interpolated into a message or a lookup.
WakeId = Annotated[str, PathParam(pattern=r"^w\d{1,4}$")]

#: What a refused owner command means, as an HTTP status. The codes are set by
#: ``serving._wake_slash`` from the SAME validator that the route-less arm path
#: uses (``build_wake_schedule`` via ``build_wake_edit``), so "malformed input"
#: and "refused against existing state" mean the same thing on both paths.
_STATUS_FOR_CODE = {
    "wake_invalid": 422,
    "wake_refused": 409,
    "wake_not_found": 404,
    "wake_unavailable": 409,
    "wake_write_conflict": 409,
    "wake_store_corrupt": 409,
}


class WakeRequest(Input):
    """The scheduling fields, shared by the create and edit bodies.

    Deliberately NOT bounded on ``message`` to the harness's own
    ``MAX_WAKE_MESSAGE_CHARS``: the cap has one sentence, produced by
    ``build_wake_schedule``, and a second bound here would answer the same
    mistake with pydantic's wording instead. The generous ceiling is only
    there so a body cannot be arbitrarily large.
    """

    message: str | None = Field(default=None, max_length=200_000)
    # ``in`` is a Python keyword, so the field is named and aliased — the same
    # spelling the agent's wake tool and the CLI advertise. Only the alias is
    # accepted (``populate_by_name`` is off), which is what keeps one key on
    # the wire.
    field_in: str | None = Field(default=None, alias="in", max_length=200)
    at: str | None = Field(default=None, max_length=200)
    every: str | None = Field(default=None, max_length=200)
    until: str | None = Field(default=None, max_length=200)
    limit: int | None = Field(default=None, ge=1)


class WakeCreate(WakeRequest):
    """``POST /v1/desktop/wakes``.

    Exactly one of two shapes: ``session_id`` arms an existing conversation, or
    ``cwd`` (+ optional ``target``) creates one and arms it in the same
    request. The XOR is enforced here rather than in the handler so a body that
    mixes them is refused before the receipt journal is claimed — a claimed
    request that then refused would answer its own retry as indeterminate.
    """

    request_id: RequestID
    session_id: str | None = Field(default=None, min_length=1, max_length=64)
    cwd: str | None = Field(default=None, min_length=1, max_length=4096)
    target: SessionTarget | None = None
    #: The name the row shows. On the create shape it is optional and defaults
    #: to the schedule's own prompt; on the arm-an-existing shape it is the ONLY
    #: way to name the conversation (see ``_birth_title``), because a prompt is
    #: not a reason to rename something the user already has.
    title: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def one_subject(self):
        if self.session_id is not None and (self.cwd is not None or self.target is not None):
            raise ValueError("give session_id, or cwd and target — not both")
        if self.session_id is None and self.cwd is None:
            raise ValueError("give session_id, or cwd to create a conversation")
        return self


class WakeEdit(WakeRequest):
    """``PATCH``'s body: only the fields that are changing.

    ``None`` means two different things and the distinction is load-bearing:
    an ABSENT field keeps the row's current value, whereas one sent as ``null``
    is a deliberate CLEAR (``limit: null`` removes the bound). That is what
    ``model_dump(exclude_unset=True)`` preserves for the handler, and why the
    fields are declared optional rather than defaulted to a value.
    """


@router.get("/v1/desktop/wakes", response_model=CRUDResponse[WakeListing])
async def list_wakes(
    request: Request,
    limit: int = Query(default=WAKE_LIST_LIMIT_DEFAULT, ge=1, le=WAKE_LIST_LIMIT_MAX),
    include_dormant: bool = Query(default=True),
):
    """Every conversation on this machine that has wakes.

    The supervisor is elected rather than re-derived, and every disk read for
    the whole listing happens in ONE worker thread — an index read, a bounded
    name scan and a transcript existence check per entry, none of which may run
    on the event loop (the supervisor re-reads this same index every 10 s, and
    a poll here must not be able to stall the loop that serves every other
    request).
    """
    root = request.app.state.config_manager.config_dir
    async with errors():
        return reply(await asyncio.to_thread(_collect_listing, root, limit, include_dormant))


@router.post("/v1/desktop/wakes", response_model=CRUDResponse[WakeWriteReceipt])
async def create_wake(body: WakeCreate, request: Request):
    """Arm a wake, creating the conversation first when none is named.

    Receipted on the SAME mechanism the other creates use, because the
    create+arm shape makes two durable things: a retry that was already applied
    must answer the first attempt's result rather than make a second session.
    """
    host(request).assert_admitting()
    root = request.app.state.config_manager.config_dir

    async def create() -> dict[str, Any]:
        if body.session_id is not None:
            receipt = await _mutate(
                request,
                body.session_id,
                op="create",
                wake_request=_wake_request(body),
                created_session=False,
            )
            if body.title:
                # An EXPLICIT name is the only reason to touch an existing
                # conversation's title here: a schedule's prompt is not a
                # rename request, and silently re-titling a session the user
                # already named would be this route editing their history.
                await asyncio.to_thread(_birth_title, root, body.session_id, body.title, True)
            return receipt
        session_id, cwd = await _create_session(request, body)
        try:
            outcome = await arm_wake(root, session_id, _wake_request(body), cwd=cwd)
        except BaseException:
            # Nothing may be left behind that the user cannot see or reach:
            # a session directory with no wake in it is a phantom conversation
            # in the sidebar. Narrow on purpose — see `_rollback_created`.
            await asyncio.to_thread(_rollback_created, root, session_id)
            raise
        await asyncio.to_thread(
            _birth_title, root, session_id, body.title or body.message, bool(body.title)
        )
        return _receipt(
            root, session_id, outcome.wake_id, outcome.next_due_at, outcome.index_written, True
        )

    async with errors():
        result = await receipts(request).run(
            "create-wake:" + body.request_id,
            body.model_dump(by_alias=True, exclude_unset=True),
            create,
        )
        if result.get("replayed"):
            result = {**result, "receipt": "replayed"}
        return reply(result)


@router.patch(
    "/v1/desktop/wakes/{session_id}/{wake_id}", response_model=CRUDResponse[WakeWriteReceipt]
)
async def edit_wake_route(session_id: str, wake_id: WakeId, body: WakeEdit, request: Request):
    """Reword or re-bound one wake, keeping its id.

    Not receipted: the body carries no ``request_id`` and the operation is
    idempotent — applying the same PATCH twice ends in the same row — so there
    is no duplicate side effect for a journal to prevent.
    """
    async with errors():
        return reply(
            await _mutate(
                request,
                session_id,
                op="edit",
                wake_id=wake_id,
                # Through the same projection as a create: an edit body carries
                # no route-level fields today, and routing both through one
                # helper is what keeps that true when one gains them.
                wake_request=_wake_request(body),
            )
        )


@router.delete(
    "/v1/desktop/wakes/{session_id}/{wake_id}", response_model=CRUDResponse[WakeWriteReceipt]
)
async def delete_wake_route(session_id: str, wake_id: WakeId, request: Request):
    """Cancel one wake.

    Cancelling the LAST wake removes the index entry rather than writing an
    empty one, which is also what releases the cleanup reap guard the session
    held while it had something scheduled.
    """
    async with errors():
        return reply(
            await _mutate(request, session_id, op="cancel", wake_id=wake_id, wake_request={})
        )


# ---------------------------------------------------------------------------
# The listing (pure filesystem work; its caller moves it off the loop)
# ---------------------------------------------------------------------------


def _collect_listing(config_dir: Path, limit: int, include_dormant: bool) -> dict[str, Any]:
    from local_operator.resume import session_name, session_origin
    from local_operator.wakes.store import read_index_report
    from local_operator.wakes.supervisor import _is_stale_ms, _session_exists

    index, read_error = read_index_report(Path(config_dir))
    now_ms = int(time.time() * 1000)
    entries: list[WakeEntry] = []
    for session_id, raw in index.items():
        if not isinstance(raw, Mapping):
            continue
        dormant = bool(raw.get("stopped_at"))
        if dormant and not include_dormant:
            continue
        session_dir = Path(config_dir) / "sessions" / session_id
        cwd = str(raw.get("cwd") or "")
        schedules = _schedule_rows(raw, now_ms, _is_stale_ms)
        entries.append(
            WakeEntry(
                session_id=session_id,
                name=_display_name(session_dir, session_id, cwd, session_name),
                cwd=cwd,
                origin=_origin(session_dir, session_origin),
                updated_at=_as_int(raw.get("updated_at")) or 0,
                dormant=dormant,
                # GHOST, asked with the supervisor's own predicate. The CLI had
                # to be fixed for exactly this: its listing derived freshness
                # itself and said "1 armed, 10m overdue" about a wake the
                # supervisor had already retired over. A dormant entry is not a
                # ghost — its session is deliberately parked, not gone.
                ghost=not dormant and not _session_exists(Path(config_dir), session_id),
                next_due_at=_next_due_at(schedules),
                schedules=schedules,
            )
        )
    # Soonest first, undateable last, ties by id: the same rule the run pane
    # applies to the same values, so the two surfaces agree when both are on
    # screen.
    entries.sort(
        key=lambda entry: (entry.next_due_at is None, entry.next_due_at or 0, entry.session_id)
    )
    total = len(entries)
    return {
        "entries": [entry.model_dump() for entry in entries[:limit]],
        "generated_at": now_ms,
        "total": total,
        "truncated": total > limit,
        "supervisor": _supervisor_info(Path(config_dir)).model_dump(),
        "read_error": read_error,
    }


def _schedule_rows(entry: Mapping[str, Any], now_ms: int, is_stale) -> list[WakeScheduleRow]:
    rows: list[WakeScheduleRow] = []
    for raw in entry.get("schedules") or ():
        if not isinstance(raw, Mapping):
            continue
        due = _as_int(raw.get("next_due_at"))
        if due is None:
            continue
        rows.append(
            WakeScheduleRow(
                id=str(raw.get("id") or ""),
                message=str(raw.get("message") or ""),
                next_due_at=due,
                every_ms=_as_int(raw.get("every_ms")),
                until_at=_as_int(raw.get("until_at")),
                limit=_as_int(raw.get("limit")),
                fired_count=_as_int(raw.get("fired_count")) or 0,
                overdue_s=max((now_ms - due) / 1000.0, 0.0),
                stale=is_stale(due, now_ms),
                last_fired_at=_as_int(raw.get("last_fired_at")),
                last_attempt_at=_as_int(raw.get("last_attempt_at")),
            )
        )
    rows.sort(key=lambda row: (row.next_due_at, row.id))
    return rows


def _next_due_at(rows: list[WakeScheduleRow]) -> int | None:
    return min((row.next_due_at for row in rows), default=None)


def _display_name(session_dir: Path, session_id: str, cwd: str, read_name) -> str:
    """The conversation's name, or a floor built from its id and directory.

    Best-effort by contract: a name is decoration, and a listing that answered
    500 because one transcript could not be read would hide every other
    schedule on the machine. The floor is not optional either — a nameless row
    is a row the user cannot identify or open with confidence.
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


def _supervisor_info(config_dir: Path) -> SupervisorInfo:
    """Whether anything is actually watching this store's wakes.

    ``verifiable`` rides along although the contract names three fields: on a
    store outside the real home (every sandboxed run) launchd cannot speak
    about it at all, and reporting ``running: false`` there would be the same
    lie in the other direction — a claim about someone else's domain.
    """
    from local_operator.wakes.install import is_supported, supervisor_state

    try:
        state = supervisor_state(config_dir)
    except Exception:  # noqa: BLE001 — a probe failure must not fail the listing
        logger.warning("could not read the wake supervisor state", exc_info=True)
        return SupervisorInfo(supported=is_supported(), running=False, detail="unavailable")
    return SupervisorInfo(
        supported=is_supported(),
        running=state.running,
        detail=state.detail,
        verifiable=state.verifiable,
    )


def _as_int(value: Any) -> int | None:
    """A tolerant int read. Every field here comes from a file another process
    writes, and a hand-edited or half-written row must cost one field rather
    than the whole listing (a cast that raised would 500 the page)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


# ---------------------------------------------------------------------------
# Mutations: owner first, files only when there is no owner at all
# ---------------------------------------------------------------------------


def _wake_request(body: WakeRequest) -> dict[str, Any]:
    """The scheduling half of a body, keyed by the wire's OWN names (``in``),
    because that is what the shared validator is given on every other path.

    PROJECTED onto :class:`WakeRequest`'s own fields rather than dumped from
    whatever object arrived: the create body is a SUBCLASS carrying
    ``request_id``/``session_id``/``cwd``/``target``/``title``, and
    ``exclude_unset=True`` keeps every one of those that the request actually
    set. Those belong to this route — the runtime's validator is closed
    (``extra="forbid"``) and refuses a body carrying them, which is how a live
    session's arm answered 422 for a perfectly good schedule (found by driving
    the route against a real runtime; the cold path hid it, because
    ``build_wake_schedule`` reads only the keys it knows and ignores the rest).
    """
    aliases = {
        name if field.alias is None else field.alias
        for name, field in WakeRequest.model_fields.items()
    }
    return {
        key: value
        for key, value in body.model_dump(by_alias=True, exclude_unset=True).items()
        if key in aliases
    }


async def _mutate(
    request: Request,
    session_id: str,
    *,
    op: str,
    wake_request: dict[str, Any],
    wake_id: str = "",
    created_session: bool = False,
) -> dict[str, Any]:
    root = request.app.state.config_manager.config_dir
    async with host(request).session(session_id) as bridge:
        assert bridge.remote is not None
        if bridge.remote.owner_reachable:
            return await _via_owner(root, bridge, session_id, op, wake_id, wake_request)
        wedged = await asyncio.to_thread(_wedged, root, session_id)
        if wedged is not None:
            # NEVER WRITE AROUND A WEDGED OWNER. Its process holds the
            # transcript lease, which blocks every engage, so a wake written
            # here could not fire — and the live session would overwrite the
            # append from its own in-memory list on its next persist anyway.
            # 503, with a sentence naming the next step, because the client's
            # move is to retry or stop that conversation.
            raise HTTPException(
                503,
                {
                    "code": "wake_owner_wedged",
                    "message": (
                        "This conversation's runtime is not responding. "
                        "Retry in a moment, or stop that conversation."
                    ),
                },
            )
        return await _via_files(
            root,
            session_id,
            op=op,
            wake_id=wake_id,
            wake_request=wake_request,
            cwd=bridge.cwd,
            created_session=created_session,
        )


async def _via_owner(
    config_dir: Path,
    bridge: Any,
    session_id: str,
    op: str,
    wake_id: str,
    wake_request: dict[str, Any],
) -> dict[str, Any]:
    """Run the mutation inside the process that owns the schedules.

    The runtime's ``wake`` word is in the same command ladder ``/rename`` and
    ``/goal`` use, and it ends in ``scheduler.update`` ->
    ``Session._persist_wake_schedules``. Nothing here writes a transcript or an
    index entry: doing so is what would produce a wake the session's next
    persist deletes and the supervisor never fires.
    """
    payload = json.dumps({"op": op, "wake_id": wake_id, "request": wake_request})
    outcome = await bridge.remote.route_shared_slash("wake", payload)
    if not isinstance(outcome, Mapping):
        raise HTTPException(503, "The conversation's runtime answered nothing for this wake.")
    if outcome.get("kind") == "error":
        data = outcome.get("data") or {}
        code = str(data.get("code") or "wake_refused")
        raise HTTPException(
            _STATUS_FOR_CODE.get(code, 409),
            {"code": code, "message": str(outcome.get("text") or "The wake was refused.")},
        )
    data = outcome.get("data") or {}
    root = Path(config_dir)
    return _receipt(
        root,
        session_id,
        str(data.get("wake_id") or wake_id),
        _as_int(data.get("next_due_at")) if op != "cancel" else None,
        # VERIFIED, not assumed. The owner's persist writes the index
        # best-effort and swallows its failure (the transcript is what
        # matters), so this asks the file rather than reporting the intent.
        await asyncio.to_thread(
            _index_reflects, root, session_id, op, str(data.get("wake_id") or wake_id)
        ),
        False,
    )


async def _via_files(
    root: Path,
    session_id: str,
    *,
    op: str,
    wake_id: str,
    wake_request: dict[str, Any],
    cwd: str | None,
    created_session: bool,
) -> dict[str, Any]:
    """The cold path: nobody owns the session, so ``arm.py`` is the writer."""

    async def run():
        if op == "create":
            return await arm_wake(root, session_id, wake_request, cwd=cwd)
        if op == "edit":
            return await edit_wake(root, session_id, wake_id, wake_request, cwd=cwd)
        if op == "cancel":
            return await cancel_wake(root, session_id, wake_id, cwd=cwd)
        raise WakeWriteError(f"unknown wake operation {op!r}", status=422, code="wake_invalid")

    try:
        outcome = await run()
    except WakeWriteError as error:
        raise HTTPException(error.status, {"code": error.code, "message": str(error)}) from None
    return _receipt(
        root,
        session_id,
        outcome.wake_id,
        outcome.next_due_at,
        outcome.index_written,
        created_session,
    )


def _receipt(
    config_dir: Path,
    session_id: str,
    wake_id: str,
    next_due_at: int | None,
    index_written: bool,
    created_session: bool,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "wake_id": wake_id,
        "next_due_at": next_due_at,
        "created_session": created_session,
        "supervisor": _supervisor_info(Path(config_dir)).model_dump(),
        "receipt": "applied",
        "index_written": index_written,
    }


def _wedged(config_dir: Path, session_id: str) -> Any:
    """The supervisor's own wedge predicate, imported rather than re-derived so
    this route and the process that fires the wake cannot disagree about which
    sessions are unreachable."""
    from local_operator.wakes.supervisor import wedged_runtime

    return wedged_runtime(Path(config_dir), session_id)


def _index_reflects(config_dir: Path, session_id: str, op: str, wake_id: str) -> bool:
    from local_operator.wakes.store import read_entry

    try:
        entry = read_entry(Path(config_dir), session_id)
    except Exception:  # noqa: BLE001 — an unreadable index is reported, not raised
        return False
    ids = {
        str(raw.get("id") or "")
        for raw in (entry or {}).get("schedules") or ()
        if isinstance(raw, Mapping)
    }
    if op == "cancel":
        return wake_id not in ids
    return wake_id in ids


# ---------------------------------------------------------------------------
# Create+arm
# ---------------------------------------------------------------------------


async def _create_session(request: Request, body: WakeCreate) -> tuple[str, str]:
    """The conversation, and the directory its runtime should start in.

    ``assert_admitting`` has already run at the route, and the pool re-asks it
    as its own first statement (its contract). ``resolve_working_directory`` is
    the same admission ``sessions.create`` applies, and it is what turns the
    caller's string into the absolute path the wake's session is born in.
    """
    from local_operator.server.utils.desktop_sessions import resolve_working_directory

    pool = host(request)
    session_id = await pool.create(
        body.cwd or "", target=body.target.model_dump() if body.target else None
    )
    cwd = await asyncio.to_thread(lambda: str(resolve_working_directory(body.cwd or "")))
    return session_id, cwd


def _rollback_created(config_dir: Path, session_id: str) -> None:
    """Remove the session directory this request just made — and NOTHING else.

    The arm failed after the create succeeded, so the alternative is a phantom
    conversation: a directory with a marker and no transcript, listed in the
    sidebar, that can never fire the wake the user asked for. The removal is
    guarded on every fact that would make the directory NOT provably ours and
    empty, because deleting a session the user can see holds real work:

    - a transcript (something wrote into it — it is a real conversation now);
    - a wake index entry (a wake exists, so the session must not vanish);
    - a runtime record (a process owns it);
    - a marker we did not write (the create wrote one, so its absence means
      this is not the directory ``create`` returned).

    Anything else — including a failure during the removal itself — leaves the
    directory alone and lets the caller's error body carry the ``session_id``,
    which is the state the API documents: retry the arm against that id.

    **Why this is not ``cleanup.remove_session_dir``**, which is otherwise the
    one remover of a session directory in this tree, and why the call is
    allow-listed by name in ``tests/unit/session/test_no_session_deletion.py``
    (reviewers: the row records the same argument): that remover refuses any
    target in a store without the cleanup STORE MARKER, and ``mark_store``'s
    own contract forbids cleanup from marking its own target. A store where no
    session has ever been BUILT carries no marker — and that is precisely the
    store a freshly created desktop draft lives in, so routing this through
    the remover would turn the rollback into a silent no-op in the one case it
    exists for. The safety here is the identity proof above rather than the
    store marker, and it is strictly narrower: it removes one directory, this
    request's own, and only while it is still empty of everything.
    """
    from local_operator.mobile.attach_client import find_runtime_record
    from local_operator.resume import TRANSCRIPT_NAME
    from local_operator.wakes.store import read_entry

    path = Path(config_dir) / "sessions" / session_id
    try:
        if not path.is_dir():
            return
        if read_desktop_marker(path) is None:
            return
        if (path / TRANSCRIPT_NAME).exists():
            return
        if read_entry(Path(config_dir), session_id) is not None:
            return
        if find_runtime_record(Path(config_dir), session_id)[1] is not None:
            return
        shutil.rmtree(path)
    except Exception:  # noqa: BLE001 — a failed cleanup must not mask the arm error
        logger.warning("could not roll back the session created for a wake", exc_info=True)


def _birth_title(config_dir: Path, session_id: str, text: str | None, user_set: bool) -> None:
    """Name a conversation, best-effort.

    The designer asked for the prompt to seed the conversation's opening user
    turn; the harness deliberately does NOT synthesise a user message (a fake
    turn would be replayed into the model's context as something the user
    said). What it can do honestly is name the conversation after the prompt,
    through the same sidecar ``resume.session_name`` reads FIRST — so the row
    that appears in the sidebar says what the schedule is for, instead of
    "a1b2c3d4e5f6".

    Never fails the request: a title is decoration, and ``write_session_title``
    is documented as best-effort for the same reason.
    """
    from local_operator.resume import write_session_title

    text = (text or "").strip()
    if not text:
        return
    try:
        write_session_title(
            Path(config_dir) / "sessions" / session_id,
            text,
            user_set=user_set,
            past_names=[],
        )
    except Exception:  # noqa: BLE001
        logger.warning("could not name the session created for a wake", exc_info=True)

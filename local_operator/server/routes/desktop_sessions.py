"""Additive authenticated session API; legacy per-turn chat stays unchanged."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import pathlib
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Callable, Literal, NamedTuple

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)
from starlette.background import BackgroundTask

from local_operator.harness.types import ModelSpec
from local_operator.media import SUPPORTED_IMAGE_MIME_TYPES
from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_mesh import MESH_ID_PATTERN
from local_operator.server.models.desktop_sessions import (
    AdmissionStatus,
    AnswerReceipt,
    ArchiveState,
    AttentionState,
    ChildTranscriptPage,
    CommandReceipt,
    CreatedSession,
    DeletedSession,
    DraftPreviewPayload,
    HistoryPage,
    InterruptReceipt,
    MessageAdmission,
    MoveReceipt,
    NotificationClaim,
    PinState,
    PresenceReceipt,
    SessionList,
    SessionSearch,
    SessionSnapshot,
    WarmReceipt,
    WatchReceipt,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.retire import RETIRING_STATE_ATTR, DaemonRetiring
from local_operator.server.utils.desktop_commands import OWNER_COMMANDS, native_action
from local_operator.server.utils.desktop_feed import DesktopFeed
from local_operator.server.utils.desktop_receipts import (
    DesktopReceipts,
    ReceiptConflict,
)
from local_operator.server.utils.desktop_sessions import (
    ADMISSION_FAILED_FRAME,
    CHILD_PAGE_LIMIT,
    FAILED_ADMISSION_STATUS,
    SUBSCRIBER_COUNT,
    DesktopSessionBridge,
    DesktopSessions,
    LegacySubscriberDuringMove,
    SessionDeletionRefused,
    SubagentChildUnavailable,
    move_session,
    resolve_working_directory,
)
from local_operator.server.utils.store_failures import (
    STORE_BUSY,
    STORE_OUT_OF_SPACE,
    StoreFailure,
    display_root,
    sqlite_store_failure,
    store_failure,
)
from local_operator.session.attached import RuntimeUnresponsiveError
from local_operator.session.attention import SupersededCompletionToken
from local_operator.session.cold_model import synthesise_cold_state
from local_operator.session.errors import (
    MoveIndeterminate,
    OperatorAuthorityRequired,
    SessionStoreUnavailable,
)
from local_operator.session.frontend_state import (
    FrontendSync,
    SlashResult,
    sync_wire_payload,
)
from local_operator.session.runtime.presence import PRESENCE_TTL_S
from local_operator.slash_commands import (
    command_argument_refusal,
    slash_command_for,
    whole_draft_command,
)

logger = logging.getLogger(__name__)

#: How long a dispatched admission waits for the OWNER'S ACKNOWLEDGEMENT before
#: the receipt answers PENDING rather than waiting any longer. See
#: :func:`admit_receipt_request` for the measurement behind it.
#:
#: SECONDS RATHER THAN LOOP TURNS, and that is the whole of review round 2's F1.
#: The sibling's budget (``serving.py::_ADMISSION_PRELUDE_TURNS``) is correct for
#: its subject — an IN-PROCESS ``prompt``, whose reportable refusals are raised
#: before its first suspension point — and was the wrong instrument here:
#: ``bridge.remote`` is an ``AttachedSession`` whose ``admit_prompt`` is an
#: ``AttachClient`` SOCKET round trip, whose answer arrives on a selector cycle.
#: Three ``sleep(0)`` turns therefore never saw it, the ``task.done()`` arm below
#: was dead in practice, and a REFUSED admission was answered ``admitted`` while
#: the user's text was dropped with a log line as its only trace.
_ADMISSION_ACK_BOUND_S = 2.0

#: The machine code a CONTROL path answers when the session's runtime could not be
#: reached at all — a refused dial, a socket that died, a bind that ran out of its
#: envelope. A session-scoped fact, which the status alone cannot express: 503 is
#: also what a server that is not answering, and a daemon that is retiring, both
#: produce. Part of the two-repo contract in ``docs/DESKTOP_API.md``: the renderer
#: keys on this code, and the sentence below is carried for the clients that
#: predate it.
RUNTIME_UNREACHABLE = "runtime_unreachable"

#: The vetted sentence that accompanies :data:`RUNTIME_UNREACHABLE`.
#:
#: DELIBERATELY THE UNCHANGED TEXT, while the design (D8) says a read should stop
#: talking about an "owner" at all. Two reasons, and the first is a hard
#: constraint rather than caution: the shipped desktop app recognises this exact
#: prefix to give its MCP row the "this conversation's session is not running"
#: sentence, so rewording it here would change that copy on every machine whose
#: app has not been updated yet — a UI regression produced by a backend fix. The
#: code above is what removes the coupling; the wording goes when the renderer
#: keys on the code (the UI half of this change). Second, every READ that used to
#: reach this ladder now answers cold instead, so the remaining callers are
#: control paths, where the sentence is about a request that genuinely was not
#: served.
RUNTIME_UNREACHABLE_MESSAGE = (
    "Session owner is unavailable. Reconnect and reconcile before retrying."
)

#: The machine code for a control call whose runtime IS alive and reachable but
#: did not answer inside the desktop control envelope
#: (``session/attached.py::DESKTOP_CONTROL_ATTACH_S``) — a loop busy mid-turn, a
#: long synchronous step. Split from :data:`RUNTIME_UNREACHABLE` because the two
#: call for different client behaviour: unreachable means reconcile, busy means
#: the same request will very likely succeed shortly and is safe to resend (the
#: receipt journal makes an admission at-most-once per request id).
#:
#: The MESSAGE is deliberately the unchanged :data:`RUNTIME_UNREACHABLE_MESSAGE`:
#: a shipped app that predates this code matches that prefix for its copy, and a
#: backend fix must not move user-visible text on machines whose app has not
#: updated. ``retryable``/``retry_after_ms`` and a ``Retry-After`` header are
#: additive fields a newer renderer keys on.
RUNTIME_BUSY = "runtime_busy"

#: How soon a client may usefully resend a ``runtime_busy`` request. Short
#: because the refusal is produced in ``DESKTOP_CONTROL_ATTACH_S`` rather than
#: 15 s, so two retries still fit well inside the renderer's 20 s deadline.
#:
#: DELIBERATELY SHORTER THAN THE 3 s ENVELOPE (review round 1, N2). It is the
#: PAUSE before the next attempt, not a forecast of when the owner answers: the
#: retry spends its own ``DESKTOP_CONTROL_ATTACH_S`` waiting for the owner, so an
#: owner that recovers within ~5 s of the refusal is admitted by the first
#: retry, and refuse + pause + retry cycles (3 + 2 + 3 + 2 + 3 = 13 s) keep three
#: attempts inside the renderer's 20 s deadline. Aligning it to 3 s buys no extra
#: chance of admission and costs the third attempt's headroom.
RUNTIME_BUSY_RETRY_AFTER_MS = 2000

#: The receipt's three dispositions (``AdmissionDetail.status``). ``status`` is
#: the ONE-WORD answer to "did the owner take this text", which is why a false
#: ``admitted`` is not a wording problem: it is the field a renderer branches on.
#:
#: * ``admitted`` — the owner acknowledged the admission. ``detail`` is the
#:   owner's own sentence, passed through verbatim (``prompt admitted``,
#:   ``steering queued``).
#: * ``pending`` — the acknowledgement had not arrived inside the bound. The
#:   request HAS been written to the owner's connection; whether the owner took
#:   it is unknown at the moment of answering, and this receipt says so. A later
#:   FAILURE of that admission is published as :data:`ADMISSION_FAILED_FRAME`.
#: * ``failed`` — the owner answered with an error, or the transport did. The
#:   request was NOT admitted and the caller may issue a new one.
#:
#: Annotated with ``AdmissionStatus`` rather than left to widen to ``str``: the
#: model's field is that ``Literal``, so a typo here is a pyright failure at the
#: constant instead of a pydantic rejection at response time (a 500 at the
#: caller, review round 3, NIT-2).
ADMITTED_ADMISSION_STATUS: AdmissionStatus = "admitted"
PENDING_ADMISSION_STATUS: AdmissionStatus = "pending"
#: The failure status and the two admission frames it bookends are IMPORTED
#: from ``server/utils/desktop_sessions.py``, where the frames are composed: the
#: names are re-exported here because this module is where the contract is
#: documented and where callers (and the UI's own mirror of it) look for them.
#: One definition, so the pool's purpose-named announcements cannot publish a
#: name this module has drifted away from.

#: The phrases this host ADDS to ``admission.detail``, one per pending shape.
#: Two rather than one because the caller must still be able to tell a text
#: handed to the running turn's steer path from one handed to a session with no
#: turn running (review round 1, NIT-1) — and neither claims the owner accepted
#: it, which is what the phrase they replace (``admitted; the owner's
#: acknowledgement was still in flight``) did.
PENDING_ADMISSION_DETAIL = "pending; the owner has not acknowledged it"
PENDING_STEER_ADMISSION_DETAIL = (
    "pending; the steer into the turn already running is not acknowledged yet"
)


#: The session-stream frame that carries a DETACHED admission's failure to the
#: UI. Session-scoped because the failure concerns one session's text, and the
#: stream is where a viewer is already listening (``DESKTOP_API.md``, "Stream
#: ordering and lifecycle": a renderer that does not know the type ignores it).
#: RETAINED ONLY FOR THE LIFE OF THE ATTACHMENT, and that bound is load-bearing
#: rather than incidental: publishing is the settle path's last act before it
#: releases, so a failure on the session's last held reference detaches the
#: bridge, and the next attach rebuilds the facade with a new epoch and an empty
#: replay — the reconnect that comes to read the frame is what discards it
#: (review round 3, F2). A client that must know reconciles against the durable
#: transcript and the receipt instead.


class AdmissionOutcome(NamedTuple):
    """What :func:`admit_receipt_request` reports: the receipt's own fields.

    ``status`` carries the MODEL's own ``Literal`` rather than a bare ``str``:
    the route copies it straight into ``AdmissionDetail.status``, so a mistyped
    status is caught here by the type checker instead of by pydantic at response
    time, on the caller's side of the wire (review round 3, NIT-2).
    """

    status: AdmissionStatus
    detail: str
    duplicate: bool


def _admission_failure_detail(error: BaseException) -> str:
    """A VETTED sentence for an admission that failed, never the transport's.

    The discipline is ``errors()``'s below and for the same reason: an
    ``AttachClient`` raises bare ``ConnectionError``s carrying socket errors and
    control ports, and an owner's ``RuntimeError`` carries the owner's own
    prose. Neither may be echoed at a renderer, so only two shapes are quoted —
    the enumerated admission refusals, whose wording is rebuilt LOCALLY from a
    category (``session/errors.py::admission_error``), and the transport
    distinction a caller acts on. Everything else gets the generic sentence.
    """
    from local_operator.session.errors import (
        AttachmentUnavailable,
        ProfileRegistryUnavailable,
        RuntimeRetiring,
    )

    if isinstance(error, (AttachmentUnavailable, ProfileRegistryUnavailable, RuntimeRetiring)):
        # Safe by construction: these carry no owner-supplied text.
        return f"failed; {error}"
    if isinstance(error, TimeoutError):
        # BEFORE the ConnectionError arm: ``OwnerAckTimeout`` subclasses both,
        # and "the owner is alive but did not answer" is the actionable half.
        return "failed; the owner did not answer in time"
    if isinstance(error, ConnectionError):
        return "failed; the session owner could not be reached"
    return "failed; the owner did not admit the request"


def _admission_failure_payload(
    request_id: str,
    detail: str,
    *,
    mode: str | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """The outcome frame's body — ONE spelling for both submission surfaces.

    The inline ``/messages`` route and the detached receipt path both resolve an
    acknowledgement, and a viewer must be able to read either frame the same way:
    ``request_id`` is the correlation, ``status`` the named condition and
    ``detail`` the vetted sentence. Built here rather than written twice so the
    two call sites cannot drift into two shapes for one contract.

    Each surface adds the name IT knows the request by — the inline route the mode
    it submitted with, the receipt path the slash command it dispatched. Those are
    additive; a renderer that reads only the frame's contract needs neither.
    """
    payload: dict[str, Any] = {
        "request_id": request_id,
        "status": FAILED_ADMISSION_STATUS,
        "detail": detail,
    }
    if mode is not None:
        payload["mode"] = mode
    if command is not None:
        payload["command"] = command
    return payload


def _refusal_resolves_the_acknowledgement(error: BaseException) -> bool:
    """Whether THIS attempt's failure is the outcome its viewers are waiting for.

    Two failures are deliberately NOT published, and both would be wrong DATA
    rather than a missing frame:

    * ``ReceiptConflict`` — another attempt at the same request id is already
      running. Its outcome is that attempt's to report; publishing a failure here
      would resolve an acknowledgement whose work is still in flight.
    * Cancellation — the caller went away. There is nobody left to tell, and
      awaiting the publish inside a cancelled task is precisely where it would not
      happen.

    Everything else — an owner that cannot be reached or that leaves mid-admission,
    an acknowledgement that lapses, a daemon that latches, an attachment that is
    unavailable — IS this request's outcome, and the viewer holding its
    acknowledgement must be told rather than left with a promise that never
    resolves.
    """
    return not isinstance(error, (asyncio.CancelledError, ReceiptConflict))


async def _give_the_bridge_back(bridge: DesktopSessionBridge) -> None:
    """Release one reference taken by ``acquire()``, protected from cancellation.

    The RELEASE has cancellation windows of its own, and both are real rather
    than theoretical (review round 3, F1): ``release`` takes the bridge's
    lock, which every other route on this session contends — a concurrent
    ``acquire`` holds it across a cold ``attach_existing`` — and once ``users``
    reaches 0 its ``_detach`` awaits the owner connection's tear-down. A
    cancellation delivered inside either window propagated INTO the awaited
    release (a task's cancellation reaches the future it is waiting on), leaving
    the count undecremented — the same permanent pin, on a second path — and a
    retry from there would take ``users`` below zero instead.

    ``shield`` is what makes the reference's fate independent of the request's:
    the AWAIT is cancelled and the request unwinds promptly, while the release
    keeps running to completion, exactly once. When the request is already
    unwinding there is no caller left to raise the release's own failure to, so
    shield's own callback retrieves it rather than leaving the loop to log an
    unretrieved task exception.
    """
    await asyncio.shield(bridge.release())


async def _settle_detached_admission(
    bridge: DesktopSessionBridge,
    task: "asyncio.Task[tuple[str, bool]]",
    *,
    command: str,
    command_id: str,
) -> None:
    """Follow a dispatched admission to its end, and free the bridge it holds.

    THE RECEIPT IS ALREADY SENT when this runs, so a failure here has no caller
    left to reach — which is exactly why it must not be a log line alone (review
    round 2, F1): nobody else would ever tell the user their text was dropped,
    and nothing on the desktop feed would show it either. The failure is
    therefore published on the session's own stream as
    :data:`ADMISSION_FAILED_FRAME`, where the mounted viewer reads it, and the
    log line stays for an operator looking at the process.

    It also OWNS THE BRIDGE REFERENCE the dispatch took (see
    :func:`admit_receipt_request`), which is why the release is here rather than
    in the request's ``finally``: the lease must outlive the admission, not the
    reply.
    """
    try:
        await task
    except BaseException as error:  # noqa: BLE001 — a failed admission is data here
        detail = _admission_failure_detail(error)
        logger.warning("a receipt's request failed after admission: %s", error)
        bridge.publish(
            ADMISSION_FAILED_FRAME,
            _admission_failure_payload(command_id, detail, command=command),
        )
    finally:
        await _give_the_bridge_back(bridge)


async def admit_receipt_request(
    bridge: DesktopSessionBridge,
    text: str,
    *,
    command: str,
    command_id: str,
    images: list[dict[str, str]],
) -> AdmissionOutcome:
    """Admit one receipt's request on the owner, and report what the OWNER said.

    WHY IT IS BOUNDED RATHER THAN AWAITED. ``admit_prompt`` is an ack on the
    owner's ``prompt``/``steer`` op, and the ``prompt`` op resolves that ack on
    the DURABLE TRANSCRIPT APPEND, never on queue insertion
    (``serving.py::prompt``: "ACK is the durable transcript append, never
    insertion into this queue"). Awaiting it while a turn is running parks this
    reply for the whole of that turn; past the client's ``ACK_TIMEOUT_S`` (15 s)
    the caller is told the owner is unavailable and, on retry under the same
    request id, that the outcome is indeterminate — while the goal IS set and the
    turn IS queued. That is the failure the sibling host answers the same way
    (``serving.py::_admit_without_waiting_for_the_turn``).

    WHY THE BOUND IS ``_ADMISSION_ACK_BOUND_S`` AND WHY THAT CANNOT PARK THE
    CALLER. ``asyncio.wait`` returns the moment the ack lands, so the bound is a
    CEILING on the wedged case, not a cost every receipt pays. What it has to
    cover is one socket round trip plus the leg's own work, and the two legs are
    both short: a STEER (a turn is running) acks on QUEUE INSERTION, and a PROMPT
    on an idle session acks on the TURN'S OWN START — the drain reaches it
    immediately because nothing is ahead of it. Measured on the assembled stack
    (``tests/e2e/test_desktop_goal_mid_turn.py`` prints both legs) at single-digit
    milliseconds against a 2 s bound, which also covers what the loop-turn budget
    could not: an owner whose steer bounds images before its first refusal is a
    thread hop, and a thread hop resolves in milliseconds but never inside three
    ``sleep(0)`` turns (``serving.py:3427``, measured "not done at 2 sleep(0)s,
    done by 20"). 2 s is deliberately a seventh of the caller's own ack deadline,
    so a receipt that waits the whole bound is still answered long before the
    caller gives up — the park, and the 503-then-409 ladder on retry, stay gone.

    WHAT IT DOES WITH EACH ANSWER. The owner's own acknowledgement is reported
    VERBATIM under ``admitted``. An error that reaches inside the bound is
    reported as ``failed`` (:func:`_admission_failure_detail`) — a truthful
    disposition rather than a raise, because a raise leaves this request's
    receipt UNFINISHED (``DesktopReceipts._claim``) and the caller's retry under
    the same id then reads "outcome is indeterminate", which is the ladder this
    route exists to remove. Silence past the bound is reported as ``pending``:
    the text is with the owner, unacknowledged, and saying anything stronger
    would be a claim about a state nobody has observed.

    THE BRIDGE IS HELD ACROSS THE DISPATCH (review round 2, F2).
    ``bridge.remote`` is a facade the pool DISPOSES when its last user releases
    it (``DesktopSessionBridge.release`` → ``_detach`` → ``remote.dispose()`` →
    ``client.close()``), and the reader pump then fails every pending request
    future. Returning while the call is still in flight therefore handed the
    pool a promise it could not keep: a POST that was its session's only user
    closed the connection its own admission was still using, manufacturing this
    helper's failure — and a spurious warning — out of nothing. One extra
    reference makes the lease outlive the admission instead, released exactly
    once, by whichever path settles it; a teardown from anywhere else
    (``close()``, process shutdown) fails the admission deliberately, and
    :func:`_settle_detached_admission` reports that failure rather than hiding it.

    A turn already running takes the text the way both other hosts take it — the
    TUI's ``_submit_prompt`` and ``serving.py::_complete_unconsumed_action``
    steer when the session is streaming — so ``/goal <text>`` mid-turn behaves
    here as it does on one Enter in the terminal. That choice also keeps the ack
    off the running turn: ``steer`` answers on queue insertion ("steering
    queued"), where ``prompt`` would wait for the append.

    AN OWNER FROM BEFORE THE ``steer`` OP: the call fails, and the failure is one
    the caller can see — the owner never acknowledges an op it does not have, so
    the bound expires and the receipt answers ``pending``, with the refusal
    itself reported by the detached continuation (log plus
    :data:`ADMISSION_FAILED_FRAME`) rather than turned into an error for a goal
    that IS set. The same skew already governs the ``steer`` mode of
    ``/messages``, so this adds no new compatibility surface.

    EVERY EXIT FROM THE DISPATCH GIVES THE REFERENCE BACK EXACTLY ONCE (review
    round 3, F1). ``await asyncio.wait`` below is the one suspension point
    between ``acquire()`` and the hand-off, and it had no ``try``/``finally``: a
    cancellation landing in it (uvicorn cancels in-flight connection tasks once
    its graceful-shutdown timeout is exceeded; a supersession does the same)
    made BOTH release sites — the settled path and the detached continuation's
    ``finally`` — unreachable. The leak is permanent, not slow: ``release`` is
    the only thing that drops ``users`` and the only trigger for ``_detach``,
    and the pool's cap-eviction path only ever considers bridges at
    ``users == 0``, so the facade, the owner connection and its runtime stayed
    resident for the life of the process. The cancellation path therefore hands
    the task AND the reference to the SAME continuation the pending path uses.

    NOT AN IMMEDIATE RELEASE, deliberately: releasing while the admission is
    still in flight is the defect round 2's F2 is about — the pool would dispose
    the facade this call is using, the request would fail of our own teardown,
    and the ``admission.failed`` frame that followed would blame the owner for
    it. Handing the reference to the settling continuation keeps the hold for
    exactly as long as the admission lives, gives it back in one release, and
    keeps the report: a refusal that lands after the cancellation is published
    on the session's stream instead of becoming an unretrieved task exception.
    Every release then goes through :func:`_give_the_bridge_back`, because the
    release itself is the second place a cancellation can take the reference
    with it.
    """
    remote = bridge.remote
    assert remote is not None, "the route binds the runtime before admitting"
    queued = bool(getattr(remote, "is_streaming", False))
    await bridge.acquire()
    # Non-``None`` only once the dispatch exists: before that the reference is
    # ours alone, with nothing to hand anywhere.
    task: asyncio.Task[tuple[str, bool]] | None = None
    try:
        task = asyncio.ensure_future(
            remote.admit_prompt(text, command_id=command_id, images=images, steer=queued)
        )
        done, _ = await asyncio.wait({task}, timeout=_ADMISSION_ACK_BOUND_S)
    except BaseException:
        if task is None:
            # Cancelled before the admission was dispatched: no continuation is
            # owed one, so this is the only releaser there is.
            await _give_the_bridge_back(bridge)
        else:
            # The continuation owns the task AND the bridge reference from here,
            # exactly as on the pending path below — it awaits the admission,
            # reports a refusal on the session's stream, and releases in its
            # ``finally``.
            asyncio.ensure_future(
                _settle_detached_admission(bridge, task, command=command, command_id=command_id)
            )
        raise
    if not done:
        # The continuation owns the task AND the bridge reference from here.
        # ``ensure_future`` rather than a done-callback so the release it owes is
        # awaited: a callback is sync, and ``release`` is where the pool tears a
        # facade down.
        asyncio.ensure_future(
            _settle_detached_admission(bridge, task, command=command, command_id=command_id)
        )
        return AdmissionOutcome(
            PENDING_ADMISSION_STATUS,
            PENDING_STEER_ADMISSION_DETAIL if queued else PENDING_ADMISSION_DETAIL,
            False,
        )
    await _give_the_bridge_back(bridge)
    if task.cancelled():
        # Not a refusal to report: the session is going away and the receipt is
        # the least of it. Reported as ``failed`` all the same, because the text
        # did not reach the owner and a caller must not read that as accepted.
        return AdmissionOutcome(
            FAILED_ADMISSION_STATUS, "failed; the admission was cancelled", False
        )
    error = task.exception()
    if error is not None:
        return AdmissionOutcome(FAILED_ADMISSION_STATUS, _admission_failure_detail(error), False)
    detail, duplicate = task.result()
    return AdmissionOutcome(ADMITTED_ADMISSION_STATUS, detail, duplicate)


router = APIRouter(tags=["Desktop sessions"], dependencies=[Depends(require_desktop)])
RequestID = Annotated[
    str, Field(pattern=r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")
]
#: The attachment store names files ``<digest>.bin``, so the digest reaches a
#: filesystem path directly. Constraining the shape HERE rather than in the
#: handler is what makes traversal unreachable by construction: FastAPI rejects
#: a non-matching path before the handler runs, and a later edit inside the
#: handler cannot route around a declaration. 32 hex characters is
#: ``attachments._DIGEST_CHARS``.
AttachmentDigest = Annotated[str, Path(pattern=r"^[a-f0-9]{32}$")]
#: What a stored attachment may claim to be on the wire. The store records the
#: mime its CALLER supplied — ``transcript._externalize_attachments`` copies
#: ``block["mime_type"]`` verbatim with no allowlist — and the sidecar carrying
#: it is not digest-verified, so neither the value's shape nor its meaning is
#: guaranteed at this boundary. Two failures follow from trusting it, and both
#: were measured on this route: a mime containing CRLF makes h11 reject the
#: header and the client gets NO response at all, contradicting the docstring's
#: "404, never 500"; and ``text/html``/``image/svg+xml`` round-trip out of here
#: as active content served from an authenticated local port. Today no live
#: ingress stores a non-image mime, but that is a property of callers upstream
#: that nothing HERE enforces, and this is the boundary that pays for it
#: changing. ``routes/static.py`` already establishes the allowlist convention
#: for image bytes over HTTP; the narrower ``media`` set is used because it is
#: exactly what the ingress can produce (``sniff_image`` returns these four)
#: and it excludes ``image/svg+xml``, which is script-bearing.
ATTACHMENT_FALLBACK_MIME = "application/octet-stream"


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Seen(Input):
    # A durable completion UUID is the only admission: timestamps or a caller's
    # runtime epoch could accidentally acknowledge a later, unseen outcome.
    completion_token: RequestID


class Notified(Input):
    # Same admission as `Seen`, and for the same reason: a durable completion
    # UUID is the only identity that survives a runtime epoch, so nothing a
    # caller can invent (a timestamp, its own epoch) may enter the watermark.
    # The two routes are otherwise unrelated — this one claims the right to
    # notify and NEVER acknowledges a read.
    completion_token: RequestID


class Pin(Input):
    """The pin STATE the caller wants this session to be in.

    A desired state rather than a toggle verb, and that is deliberate: see the
    route's docstring. ``extra="forbid"`` (inherited from ``Input``) is what
    makes an omitted ``pinned`` a 422 rather than a silent false — the field is
    the whole request, so a body that does not carry it is not a request this
    route can honour.

    ``StrictBool`` rather than a bare ``bool``, matching every other boolean on
    this plane (``PresenceWindow``, ``PresenceBeat``, ``Watch``, ``Answer``). A
    bare ``bool`` coerces the strings and integers a generous JSON client sends
    — ``{"pinned": "yes"}`` and ``{"pinned": 1}`` both pin the session — so a
    client whose serialiser is producing the wrong type gets a 200 and no
    signal, and the bug surfaces later as "the pin came from nowhere". The
    neighbours refuse that shape, and a pin is durable state rather than a
    display hint: being wrong about it silently is what this route exists to
    prevent.
    """

    pinned: StrictBool


class Archive(Input):
    """The ARCHIVE state the caller wants this session to be in.

    ``Pin``'s shape and ``Pin``'s reasons: a desired state rather than a toggle
    verb (a retried toggle flips the archive back, which the user reports as
    "the archive keeps un-archiving itself"), ``extra="forbid"`` so an omitted
    ``archived`` is a 422 rather than a silent false, and ``StrictBool`` so a
    client whose serialiser produces ``"yes"`` or ``1`` gets a signal instead of
    a 200 over state it did not mean to set.
    """

    archived: StrictBool


class ConfirmDeletion(Input):
    """The explicit confirmation a PERMANENT deletion requires.

    THE FIELD IS THE WHOLE REQUEST, which is why it is required and why
    ``extra="forbid"`` is inherited: a delete that dispatch sends without a
    confirmation must not be a delete. ``confirmed: true`` is the only accepted
    value — see :meth:`_require_confirmation` for why ``false`` is a 422 rather
    than a quieter refusal.
    """

    confirmed: StrictBool

    @model_validator(mode="after")
    def _require_confirmation(self) -> "ConfirmDeletion":
        """Refuse ``confirmed: false`` as a malformed request, not as a decision.

        ``StrictBool`` alone is not enough: it accepts ``False``, and a body that
        explicitly says it is NOT confirming would then reach the handler, where
        the only honest things to do are refuse it (a 409 that says the user
        withholds consent for something they never asked to withhold) or delete
        (silence, and the worst possible reading). Recognising it HERE makes it a
        422 — the status this ladder already uses for a body it cannot honour,
        the same one an omitted field gets — so a client bug reads as a client
        bug. The UI only ever sends ``true``; this is the guard for the day it
        does not.
        """
        if not self.confirmed:
            raise ValueError("confirmed must be true")
        return self


class PresenceWindow(Input):
    """The desktop window's REAL state, as reported by its main process.

    NOT the renderer's ``document.visibilityState``/``hasFocus()``. The UI
    already documents that pair as unsound (a throttled window the user is
    looking at reports ``hidden``; a window behind another app reports
    ``visible``), and the backend cannot re-derive the truth from anything it
    holds. Electron main owns the window, so main reports it here and the
    backend believes it — which is the only judgement this field carries.

    ``exists`` is separate from ``visible`` on purpose: a macOS app with every
    window closed is alive in the dock and CAN raise a banner, but it is
    displaying nothing, so a `session_id` it reports must not be read as "that
    conversation is on screen".
    """

    exists: StrictBool = False
    focused: StrictBool = False
    visible: StrictBool = False
    minimized: StrictBool = False


class PresenceBeat(Input):
    """One desktop delivery-presence heartbeat.

    ``subscription_id`` is the id ``GET /v1/desktop/events`` handed the client
    in its ``open`` frame. The lease is held AGAINST that live SSE socket, which
    is what makes ``can_notify`` mean "whoever holds this can actually deliver"
    rather than "an app once said yes" — a dropped socket revokes the claim.

    ``can_notify_kinds`` is REQUIRED to be honest about what the app can
    deliver. The feed carries completions only, so the gate path keeps its
    existing per-session lease and its existing toast; an app that claimed every
    kind here would silence a background session's parked question with nothing
    to replace it.
    """

    subscription_id: str = Field(min_length=1, max_length=64)
    can_notify: StrictBool
    can_notify_kinds: list[str] = Field(default_factory=list, max_length=8)
    #: The conversation the app is showing, or "" for none. Constrained to the
    #: 12-hex session-id shape when present so a malformed id is a 422 rather
    #: than a value that silently matches no session and therefore reads as
    #: "nothing is on screen".
    session_id: str = Field(default="", pattern=r"^$|^[a-f0-9]{12}$")
    window: PresenceWindow = Field(default_factory=PresenceWindow)


class SessionTarget(Input):
    kind: Literal["agent", "team"]
    name: str = Field(min_length=1, max_length=128)


class DraftModel(Input):
    """The model a NEW conversation should be born on, plus its reasoning level.

    Deliberately the SAME shape the canonical frontend state publishes for a
    conversation's model (``selected_model``: ``provider``, ``model_id``,
    ``reasoning_effort``), so the client hands back the row its own picker is
    already showing rather than translating one vocabulary into another — a
    second shape is how the chip and the model that answers come to disagree.

    ``reasoning_effort=None`` means "no level chosen". It is NOT the same as "this
    model has no ladder": the ladder check is what refuses an unexpressible level
    below, and a model with an empty ladder refuses every level but ``None``. It is
    also NOT "the model's own default": the conversation is born on the machine's
    configured ``model_effort`` (clamped into this model's ladder), and only a
    config that expresses no opinion falls back to the model's seeded rung. The
    MARKER records the ``None`` — the choice — while the pane and the launch report
    that resolved level; see ``session.cold_model.resolve_birth_effort``.
    """

    provider: str = Field(min_length=1, max_length=64)
    model_id: str = Field(min_length=1, max_length=256)
    reasoning_effort: str | None = Field(default=None, max_length=32)


class CreateSession(Input):
    request_id: RequestID
    cwd: str = Field(min_length=1, max_length=4096)
    target: SessionTarget | None = None
    #: Omitted or null ⇒ today's behaviour, byte for byte: the session is born on
    #: the configured default. This is the ONLY optional admission of the two
    #: routes, and it is what lets an older client keep posting the old body.
    model: DraftModel | None = None
    #: The DEVICE to create the conversation on (``features.peers``). Omitted means
    #: this device, and the body is then byte-identical to the pre-mesh one — the
    #: ``model`` field's own rule (Addendum 1, item 5). The id is shape-checked here
    #: rather than looked up: the PEER is the only party that can say whether it is a
    #: member of the network it is being addressed through, and its sentence is the
    #: one the user should see.
    peer: str | None = Field(default=None, pattern=MESH_ID_PATTERN)


class MoveSession(Input):
    """A live session's new working directory.

    Same field shape and bound as :class:`CreateSession.cwd` and
    :class:`DraftPreview.cwd`: one bound for "a path a user typed", so the two
    routes cannot disagree about what fits in one.

    The path is validated against the SESSION's directory, not this process's:
    `cwd` may therefore be relative (``../sibling``) or carry ``~``, exactly as
    the TUI's ``/move`` accepts, because the same machinery resolves it
    (`local_operator.tui.move_targets.expand_path`). `create` is the route that
    must resolve against the server's own cwd, and it keeps its own resolver.
    """

    request_id: RequestID
    cwd: str = Field(min_length=1, max_length=4096)


class DraftPreview(Input):
    """A new-conversation pane's readings, for a session that does not exist.

    Same shape as :class:`CreateSession` on purpose: the pane is asking the
    question it will ask for real on the first send, and a body that differed
    would let the two answers diverge.

    ``request_id`` is carried for envelope parity and deliberately NOT journalled
    as a durable receipt (see the route). A preview has no side effect to make
    at-most-once, receipts are never pruned, and a renderer may re-issue this on
    every re-render — writing a row per pane would make simply opening one cost
    disk, which is exactly what this op exists to avoid.
    """

    request_id: RequestID
    cwd: str = Field(min_length=1, max_length=4096)
    target: SessionTarget | None = None
    #: The same optional birth selection ``create`` accepts, resolved by the same
    #: authority — the pane is asking the question it will ask for real on the
    #: first send, so a body that may differ here would be a body that diverges.
    model: DraftModel | None = None


def _draft_model_spec(model: DraftModel) -> ModelSpec:
    """The spec the first turn will actually run on, or a 422 saying why not.

    REFUSED HERE, AT THE BOUNDARY, and never by silently degrading: the client
    rendered a choice the user made, and answering with a different model than
    the one it named is the failure this whole feature exists to remove. The
    three refusals are the three ways a pick can fail to be servable:

    * an unknown provider (``get_provider_definition``, which resolves the
      registry's legacy aliases, so an alias is accepted exactly where the
      engine accepts it and nowhere else);
    * a model id the provider's catalogue does not serve — the same authority
      the model picker paints from (:func:`discovery.offered_model_ids`), and no
      refusal at all for a catalogue that cannot be enumerated offline (an
      aggregator on a cold cache, a local endpoint): "we have not looked" is
      not "it does not exist", and refusing on it would kill working picks;
    * a ``reasoning_effort`` this model's ladder does not offer, including any
      level at all on a model with no ladder — the ladder is read off the SAME
      resolver the owner constructs its spec with (``build_model_spec``), so the
      level accepted here is the level the first turn will send.

    **A pick that named no level comes back carrying NONE**, never the model's
    seeded default rung. ``build_model_spec`` seeds that rung ("the level this
    model would use if nobody said"), and a seed is not a choice: letting it
    through would store a level the user never picked, pin it in the marker and
    have the first turn take it over the machine's configured ``model_effort``
    (review round 1, R1). ``null`` therefore stays distinguishable from "the
    level happened to equal the seed" all the way to the launch, where the
    owner's own resolution supplies the configured level, clamped by the ladder.

    Runs OFF the event loop by its callers: it reads the catalogue cache and, for
    a model the registry does not describe, resolves metadata (memoised, and
    disk-cached in the common case the picker just filled it).
    """
    from local_operator.model.configure import build_model_spec
    from local_operator.model.discovery import offered_model_ids
    from local_operator.providers.registry import (
        get_provider_definition,
        is_decision_only,
    )

    provider = model.provider.strip()
    model_id = model.model_id.strip()
    if get_provider_definition(provider) is None:
        raise HTTPException(
            422,
            {
                "code": "provider_unknown",
                "message": f"'{provider}' is not a known provider.",
            },
        )
    if is_decision_only(provider):
        # A decision-only provider (TypeSafe's Jev) rejects ``chat/completions`` on
        # every host we reach it through, so accepting this pick would store a
        # session model that 400s on its first turn — the failure the model
        # catalogue and the ranking already refuse to offer. It must be refused HERE,
        # before the enumeration below, because that check CANNOT catch it:
        # ``offered_model_ids`` answers ``None`` for a provider whose catalogue is not
        # enumerable offline, and ``None`` means "we have not looked, accept the
        # pair" — which is the right reading for an aggregator or a local endpoint
        # and the wrong one for a provider whose catalogue is empty by construction.
        raise HTTPException(
            422,
            {
                "code": "provider_decision_only",
                "message": (
                    f"'{provider}' serves decision-model calls, not chat completions, "
                    "so no session can run on it."
                ),
            },
        )
    served = offered_model_ids(provider)
    if served is not None and model_id not in served:
        raise HTTPException(
            422,
            {
                "code": "model_unknown",
                "message": f"'{model_id}' is not a model {provider} serves.",
            },
        )
    effort = (model.reasoning_effort or "").strip().lower()
    try:
        spec = build_model_spec(provider, model_id)
    except Exception as error:  # noqa: BLE001 — a spec we cannot build is a 422, not a 500
        raise HTTPException(
            422,
            {"code": "model_unavailable", "message": f"'{model_id}' could not be resolved."},
        ) from error
    if effort:
        if not spec.reasoning_efforts:
            raise HTTPException(
                422,
                {
                    "code": "effort_unsupported",
                    "message": f"{model_id} has no reasoning-effort levels",
                },
            )
        if effort not in spec.reasoning_efforts:
            raise HTTPException(
                422,
                {
                    "code": "effort_unsupported",
                    "message": (
                        f"{model_id} accepts {', '.join(spec.reasoning_efforts)} "
                        f"\u2014 not '{effort}'"
                    ),
                },
            )
        spec = spec.model_copy(update={"reasoning_effort": effort})
    elif spec.reasoning_effort is not None:
        # The seed, not a choice — and the two must not become indistinguishable
        # in the marker, which is the ONLY consumer of this cleared value
        # (``create`` persists it). This is NOT what the plane resolves against:
        # ``session.cold_model.resolve_birth_effort`` reads the SEED, and its third
        # case is defined on it, so a caller that hands it this cleared spec
        # answers "no level" where the launch answers the seed. Round 2's R6 is
        # exactly that mistake, made by the preview; the honest shape of the two
        # halves is ASYMMETRIC — the marker stores the CHOICE, the reading resolves
        # the RUNNING level — and the earlier comment here claimed they were the
        # same thing.
        spec = spec.model_copy(update={"reasoning_effort": None})
    return spec


def _preview_birth_model(root: pathlib.Path, model: DraftModel) -> ModelSpec:
    """The picked pair as the model describes it, at the level the turn will RUN at.

    Two values come out of the body's pick and they are NOT the same value:

    * what was CHOSEN, which is what ``create`` stores in the marker (``null`` for
      "this model, no level") — ``_draft_model_spec``'s answer, and it is used here
      only for its refusals;
    * what the first turn will RUN at, which is the reading this pane may publish.

    So the spec built here is ``build_model_spec``'s own result — it still carries
    the model's SEEDED rung, which is what ``resolve_birth_effort``'s third case is
    defined on. Handing it the seed-CLEARED spec instead made the preview report
    "no level" whenever the machine configured no ``model_effort``, while the cold
    frame and the first turn reported the seed (review round 2, R6).
    """
    from local_operator.model.configure import build_model_spec
    from local_operator.session.cold_model import resolve_birth_effort

    chosen = _draft_model_spec(model)
    spec = build_model_spec(chosen.provider, chosen.model_id)
    return spec.model_copy(
        update={"reasoning_effort": resolve_birth_effort(spec, chosen.reasoning_effort, root)}
    )


class Image(Input):
    data_b64: str = Field(max_length=1_000_000)
    mime_type: Literal["image/png", "image/jpeg", "image/gif", "image/webp"]

    @field_validator("data_b64")
    @classmethod
    def validate_data(cls, value: str) -> str:
        if not value or not base64.b64decode(value, validate=True):
            raise ValueError("An image must contain base64 data")
        return value


class Prompt(Input):
    request_id: RequestID
    text: str = Field(max_length=200_000)
    images: list[Image] = Field(default_factory=list, max_length=8)
    mode: Literal["prompt", "steer"] = "prompt"

    @model_validator(mode="after")
    def nonempty(self):
        if not self.text.strip() and not self.images:
            raise ValueError("Enter a message or attach an image")
        # A slash CONTROL must never become paid model chat, and this is the one
        # test that decides it: a draft that, as a whole, IS a command was meant
        # for the command endpoint. Anything else — every multi-line draft, every
        # leading command word followed by PROSE, every path and every sentence —
        # is a message and is accepted (the operator's own report: a three-line
        # draft opening with "/mcp logout …" was refused forever by the blanket
        # `lstrip().startswith("/")` this replaces).
        #
        # "IS a command" is not "starts with a command word": it is the word PLUS
        # an argument the desktop actually consumes — a prompt, a value from a
        # list, or a shape the command route validates or forwards
        # (`slash_commands.command_argument_is_used`). That is what keeps
        # `/compact hello` and `/usage more prose` messages while `/mcp logout`,
        # `/login openai` and `/move ~/x` stay refused: those three are controls
        # the composer RUNS, and a control that became prose here would spend a
        # paid turn on it.
        #
        # The predicate is `slash_commands.whole_draft_command`, not a second
        # inline test, because the composer plans the same draft and a second
        # derivation is how a prose draft gets planned as `send` and then refused
        # here with no way to resend it.
        if len(self.model_dump_json().encode()) > 900_000:
            raise ValueError("Message exceeds the canonical control-frame limit")
        command = whole_draft_command(self.text)
        # The sentence is GENERIC on purpose, and one row is why it must stay
        # that way: `/credential` is a whole-draft command whose text belongs to
        # the masked form, so "move it below your text" would name exactly the
        # prose form that still reaches the model (`MESSAGE_DRAFTS` pins those two
        # forms as messages). Latent rather than live, because the app's shaper
        # publishes "The request has invalid fields." for every body-validation 422
        # (`server/app.py`), so this text reaches no wire — an in-process caller
        # only, while the route keeps the masked-form instruction. Giving this row
        # its own sentence here is a behaviour change and does not belong in a
        # comment-only pass; if that shaper ever starts publishing validator
        # detail, this row needs one first.
        if command is not None:
            raise ValueError(
                f"/{command[0].name} is a command, not a message. "
                "Send it on its own, or move it below your text."
            )
        return self


class Command(Input):
    request_id: RequestID
    command: str = Field(pattern=r"^/?[A-Za-z]+$", max_length=64)
    args: str = Field(default="", max_length=200_000)
    images: list[Image] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def wire_budget(self):
        if len(self.model_dump_json().encode()) > 900_000:
            raise ValueError("Command exceeds the canonical control-frame limit")
        return self


class Answer(Input):
    epoch: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    value: str | None = Field(default=None, max_length=32768)
    approved: StrictBool | None = None
    question_index: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def one_answer(self):
        if (self.value is None) == (self.approved is None):
            raise ValueError("Supply either value or approved")
        if self.value is not None and self.question_index is None:
            raise ValueError("An ask answer requires question_index")
        if self.approved is not None and self.question_index is not None:
            raise ValueError("An approval cannot carry question_index")
        return self


class Watch(Input):
    subscription_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    visible: StrictBool
    can_notify: StrictBool


class Warm(Input):
    """No fields, and closed to any that arrive.

    The session is already named by the path and a warm carries no intent
    beyond "start one" — there is nothing a caller could usefully say here.
    Declared as a model rather than omitted so ``extra="forbid"`` still
    applies: a client that invents an option gets a 422 naming it, instead of
    having it silently ignored and believing it took effect.
    """


class Interrupt(Input):
    """One field, and deliberately no ``confirmed``.

    An interrupt destroys nothing: it stops the turn that is running and
    leaves the session, its runtime, its child sessions and its process alone,
    so it needs no confirmation to be safe. Requiring one would also make Esc
    unusable, which is the whole point of the keystroke — the stop this route
    replaces (``POST /v1/desktop/stop``) is confirmed precisely because it
    ends the session.

    ``request_id`` is the at-most-once key the receipt journal claims on, so a
    retry after a lost response cannot fire a second interrupt at a turn that
    has moved on. Declared as ``RequestID`` (the canonical UUID shape) rather
    than ``str`` because the journal keys are session-scoped and a caller-
    chosen free string would let two different presses collide on one row.
    """

    request_id: RequestID


def host(request: Request) -> DesktopSessions:
    pool = getattr(request.app.state, "desktop_sessions", None)
    if pool is None:
        pool = DesktopSessions(
            request.app.state.config_manager.config_dir,
            # The DAEMON's retirement, asked of app state rather than of the pool:
            # the pool is built lazily by the first request that needs it, and a
            # pool built after the announcement must refuse for the same reason
            # (and with the same 503) as one built before it.
            retiring=lambda: bool(getattr(request.app.state, RETIRING_STATE_ATTR, False)),
        )
        request.app.state.desktop_sessions = pool
    return pool


def receipts(request: Request) -> DesktopReceipts:
    value = getattr(request.app.state, "desktop_receipts", None)
    if value is None:
        value = DesktopReceipts(request.app.state.config_manager.config_dir)
        request.app.state.desktop_receipts = value
    return value


#: Refusals that did NOT leave a session behind on the peer, so a request id stays
#: usable: a retry after freeing a slot up, or after joining the network, must be able
#: to run rather than replay a refusal forever. ``relay_unavailable`` and
#: ``peer_unreachable`` are deliberately NOT here — in both, this device could not
#: prove the frame never landed, and the one failure that matters most is a retry that
#: mints a SECOND conversation on the peer. These are recorded instead, and the route
#: answers 503 (unconfirmed) rather than 409 (nothing changed).
_CREATE_UNCONFIRMED_CODES = frozenset({"relay_unavailable", "peer_unreachable"})


async def _remote_lifecycle(
    request: Request,
    session_id: str,
    *,
    action: Literal["archive", "unarchive", "delete"],
    confirm: bool = False,
) -> dict[str, Any] | None:
    """Run a lifecycle verb on the OWNER when the row is a peer's, else ``None``.

    ``None`` means "this device's session", and the caller keeps its existing path,
    which is what makes the branch additive: on a machine in no network the owner
    lookup answers ``None`` having read no relay, and the local route behaves exactly
    as it did.

    A REMOTE REFUSAL IS A 409 WITH THE OWNER'S OWN SENTENCE, or 404 for the one code
    that means the conversation is not there at all. Not 500: nothing failed, and the
    conversation the user can see exists — the sentence names the condition and its
    remedy, which is the same argument the local ``session_delete_refused`` arm makes
    one level down.

    THE DAEMON FORGETS A DELETED ID HERE TOO, for the reason the local delete does:
    a session that MOVED home and is then deleted on the peer would otherwise stay
    resident in this process, and this daemon would answer 200 where a fresh one
    answers 404 (the failure PR #390 fixed on the local path).
    """
    from local_operator.server.utils.desktop_mesh import (
        lifecycle_on_owner,
        remote_owner,
    )

    host_root = host(request).root
    owner = await asyncio.to_thread(remote_owner, host_root, session_id)
    if owner is None:
        return None
    device_id, device_name = owner
    result = await asyncio.to_thread(
        lifecycle_on_owner,
        host_root,
        session_id,
        action=action,
        peer=device_id,
        confirmed=confirm,
    )
    if result.get("ok"):
        if action == "delete":
            try:
                await host(request).forget(session_id)
            except Exception:  # noqa: BLE001 - forgetting is best effort, never the answer
                logger.exception("desktop pool could not drop the removed session %s", session_id)
            return {"session_id": session_id, "deleted": bool(result.get("deleted"))}
        return {"session_id": session_id, "archived": action == "archive"}
    code = str(result.get("code") or "session_lifecycle_refused")
    message = str(result.get("message") or "")
    if not message:
        message = f"{device_name or device_id} refused that and said nothing further"
    raise HTTPException(
        404 if code == "session_not_found" else 409, {"code": code, "message": message}
    )


def reply(result: Any) -> CRUDResponse[Any]:
    return CRUDResponse(status=200, message="Desktop session result.", result=result)


async def _join_owned(operation: "asyncio.Task[dict[str, Any]]") -> dict[str, Any]:
    """Await an owned operation, joining it even when THIS waiter is cancelled.

    The repeated shield-and-check loop is the point, and a bare
    ``await asyncio.shield(task)`` is NOT equivalent: shielding protects the TASK
    from cancellation, but the ``await`` itself is still interruptible, so a
    cancelled waiter returns while the transaction keeps running — and a second
    cancellation (a shutdown arriving twice, a client that goes away and the
    route task being torn down afterwards) would leave it unobserved with its
    filesystem work in flight. Here the waiter drains the task first and only
    then propagates the cancellation, and the operation's own exception is
    re-raised (``task.result()``) rather than swallowed — a failed move must be
    reported, not hidden behind a cancellation.
    """
    cancelled = False
    result: dict[str, Any]
    while True:
        try:
            result = await asyncio.shield(operation)
            break
        except asyncio.CancelledError:
            cancelled = True
            if operation.done():
                result = operation.result()
                break
    if cancelled:
        raise asyncio.CancelledError
    return result


def store_root(request: Request) -> pathlib.Path:
    """The config root whose volume a store failure is about.

    Read from the APP's config manager rather than from the process's env
    default so the answer is about the volume the stores actually live on: a
    backend started against a relocated root, or a test that mounted the app on
    a ``tmp_path``, must not have its free space measured somewhere else.
    """
    manager = getattr(request.app.state, "config_manager", None)
    directory = getattr(manager, "config_dir", None)
    if directory is not None:
        # ``pathlib`` rather than ``Path``: this module imports FastAPI's ``Path``
        # for path parameters, and the name is taken.
        return pathlib.Path(directory)
    from local_operator.paths import config_dir

    return config_dir()


#: Composes a route's refusal sentence from a classified store failure and the
#: volume the store lives on. ``None`` means the classifier's own sentence, which
#: is what every route that carries a MESSAGE gets -- see :func:`_store_refusal`.
StoreRefusalCopy = Callable[[StoreFailure, pathlib.Path | None], str]


def receipts_refusal(failure: StoreFailure, root: pathlib.Path | None) -> str:
    """The receipt routes' own refusal sentence.

    Both of them: ``POST /v1/desktop/sessions/{session_id}/seen`` clears one
    conversation's receipt and ``POST /v1/desktop/attention/seen`` clears a batch,
    and neither sends a message. A composer passed to one and not the other is how
    this defect shipped twice (review round 4, M1), so the pins below are
    handler-level on each route rather than on this function alone.

    WHY THESE ROUTES COMPOSE THEIR OWN COPY (QA round 2, Q1). The classifier's
    sentences are the SEND path's and were written for a request carrying a
    message: on a full volume both routes answered "the message could not be
    written ... and send it again" about a receipt clear, which has no message in
    it and sends nothing, and both answered "it will catch up on its own" to a
    write the user had just asked for. That is the defect this PR already fixed on
    the TUI (agent review round 1 F1 / UX round 1 U8), left standing on this
    surface; the fix has the same shape: the CLASSIFICATION stays shared, the
    SENTENCE says what the route was doing.

    Three conditions, three answers, because they need three different actions:
    contention is retryable and the remedy is to ask again -- the desktop client
    paints this sentence and reads the retry case from the CODE, so the sentence
    still has to carry the instruction; a full volume needs space freed on the
    volume this store lives on, then the request again; anything else needs the
    machine looked at and will not clear by retrying.

    ``root`` is the config root :func:`store_root` resolved, and it is named for
    the same reason the classifier names it: a machine has several volumes, and
    "check the disk" with no destination is not an instruction. No exception text
    is composed in here -- a store error names file paths, the rule
    :func:`_store_refusal` states at length.

    The shape follows the arm above it: the ``SessionStoreUnavailable`` arm
    already composes its own sentence rather than taking the exception's, and
    carries its own code. This is the same move for the same reason, one arm
    down.
    """
    where = display_root(root)
    if failure.code == STORE_BUSY:
        return "Read state is busy right now, so nothing was written. Try again in a moment."
    if failure.code == STORE_OUT_OF_SPACE:
        return (
            "This computer is out of disk space, so nothing was written. "
            f"Free some space on the volume holding {where}, then try again."
        )
    return (
        "The read state could not be written. Retrying will not help; "
        f"check {where} and the disk it is on."
    )


def _store_refusal(
    request: Request,
    failure: StoreFailure,
    error: BaseException,
    copy: StoreRefusalCopy | None = None,
) -> HTTPException:
    """Log what really happened, and build the client's vetted refusal.

    THE LOG RECORD IS THE DELIVERABLE, not a courtesy. This ladder used to raise
    ``from None`` with no record at all, so a store that could not be written
    left the operator a sentence about a busy read state and an empty log to
    check: attributing the 2026-09-17 disk-full incident took an hour of log
    archaeology through a runtime log that had recorded the same condition three
    other times. The exception is logged where it is still live, with the route
    and the session, because the client's copy may never carry it (a store error
    names file paths -- the rule the ConnectionError arm below states at length).

    ``copy`` is the route's own sentence composer, and ``None`` is every route that
    carries a message and can say so honestly: those keep the shared classifier's
    sentence, which a client paints verbatim rather than keeping a second copy of.
    The arms that need their own nouns say why at their own composer --
    :func:`receipts_refusal` today, because a receipt clear is not a message send.
    """
    session_id = request.path_params.get("session_id")
    logger.log(
        failure.level,
        "desktop store failure %s at %s %s%s",
        failure.code,
        request.method,
        request.url.path,
        f" (session {session_id})" if session_id else "",
        # The traceback rides only the two conditions an operator has to act on,
        # where the stack IS the finding; contention is routine and clears on its
        # own, so a traceback per retry is noise that buries the records worth
        # reading (review round 1, R5). The line itself is emitted either way.
        exc_info=error if failure.traceback else None,
    )
    return HTTPException(
        failure.status,
        {
            "code": failure.code,
            "message": failure.message if copy is None else copy(failure, store_root(request)),
        },
    )


@asynccontextmanager
async def errors(request: Request, copy: StoreRefusalCopy | None = None) -> AsyncIterator[None]:
    """The control plane's shared failure ladder.

    ``request`` is taken rather than reached for, the way ``host(request)`` and
    ``receipts(request)`` beside it are: two arms below must name the route they
    failed on and the volume the store lives on, and a ladder shared by six
    route modules cannot invent either.

    ``copy`` is the calling ROUTE's sentence composer for a classified store
    failure, and it is optional because almost every route here carries a message
    and can let the classifier speak for it. The routes that cannot pass their
    own: the two receipt routes (``POST /v1/desktop/sessions/{session_id}/seen``
    and ``POST /v1/desktop/attention/seen``) clear read receipts, so the send
    path's nouns are false about them (QA round 2, Q1; review round 4, M1, for
    passing it to one and not the other -- see :func:`receipts_refusal`).
    """
    try:
        yield
    except DaemonRetiring as error:
        # The daemon has announced its retirement and refuses to start work it
        # would not finish. 503, not 500: the process is alive and deliberately
        # not admitting, so the client's move is to rediscover the successor
        # through the record (design §7) — a message and a code it can key on,
        # never a traceback.
        raise HTTPException(503, {"code": error.code, "message": str(error)}) from None
    except OperatorAuthorityRequired as error:
        # THE REFUSAL IS THE ANSWER (agent review round 1 R1-2 = design D1 = UX
        # U4 = QA Q1). This one arrived as a bare ``RuntimeError`` and fell
        # through to the 503 below, so the desktop client was told the runtime
        # was UNREACHABLE and asked to reconcile — a different problem, with a
        # remedy that cannot work — while every other route (the phone relay's
        # 422) carried the copy that names the real remedies. 422, not 503: the
        # runtime answered, promptly and deliberately, and the request is the
        # thing that has to change.
        #
        # ``code`` rides along so a client can key on the category rather than
        # on the sentence, exactly as the ladder's other arms do.
        raise HTTPException(422, {"code": error.code, "message": str(error)}) from None
    except SubagentChildUnavailable as error:
        # The child read route's containment refusal (design § 9.1). Not folded
        # into the generic 404 below because the code is part of the contract:
        # the reader distinguishes nothing from it, but it is retryable, and a
        # client that cannot see WHY would have to guess whether to re-probe.
        raise HTTPException(404, {"code": error.code, "message": str(error)}) from None
    except SessionStoreUnavailable as error:
        # The catalogue refused to answer, most often because the store's
        # ``sessions/`` directory could not be walked at all (descriptor
        # exhaustion, an I/O error, a permissions change).
        #
        # 503 rather than 500, because neither the code nor the operator can
        # act on it: it is transient by construction and the correct client
        # behaviour is to keep the rows it already has and retry the poll. The
        # alternative this replaces was not an error at all — the route
        # answered ``200 {"sessions": []}`` and the sidebar, which adopts that
        # answer as MEMBERSHIP, wiped its visible catalogue until the next poll
        # succeeded. An empty list is a statement about the operator's
        # conversations; this sentence is a statement about the read, which is
        # the only true one available here.
        #
        # WHICH ARM THIS SITS AMONG is the only ordering constraint, and it is
        # satisfied by sitting above the catch-alls: no arm above matches an
        # ``OSError``, and the generic arms BELOW would report a store it could
        # not walk as a missing session. It is the third arm, not the first.
        #
        # THE BODY IS AN OBJECT BECAUSE THE STATUS ALONE MISLEADS ON THE ONE
        # ROUTE THAT IS ALSO A PROBE. ``GET /v1/desktop/sessions?limit=1`` is
        # what the desktop app asks to decide whether the daemon at an address
        # is usable with its credential, and a client that reads every non-2xx
        # as "refused" turns this transient store failure into a capability 403
        # on a daemon whose credential was never in question — which its attach
        # path answers by declining the live daemon and spawning a second one
        # over it. ``code`` is what removes the guess: 401/403 mean the
        # credential was refused, any other ANSWERED status means a daemon
        # answered. Same named-condition shape as ``DaemonRetiring`` above and
        # ``MoveIndeterminate`` below, for the same reason.
        #
        # THE CODE IS INERT UNTIL A CLIENT READS IT, and that half is not in
        # this repository: the classification lands in the app (the change that
        # makes only 401/403 mean "credential refused"). Nothing here depends
        # on it — an older client ignores the object's extra structure exactly
        # as it ignored nothing before, since it read ``detail`` as a string.
        #
        # The sentence is composed HERE rather than taken from the exception:
        # a store error's own text can name the operator's home directory, the
        # rule this ladder applies to every other category (see the
        # ConnectionError arm below).
        raise HTTPException(
            503,
            {
                "code": error.code,
                "message": (
                    "Conversations could not be read right now. "
                    "This recovers on its own; retry in a moment."
                ),
            },
        ) from None
    except KeyError:
        raise HTTPException(
            404, "Requested session, profile, team or subscription not found"
        ) from None
    except MoveIndeterminate as error:
        # 503 with the same NAMED-CONDITION body ``DaemonRetiring`` and
        # ``SubagentChildUnavailable`` use in this ladder (review round 2, N3),
        # not the 409 refusal below, and the distinction is the whole point of
        # the class: nothing was refused and NOTHING MAY BE ROLLED BACK — the
        # retire request reached the owner and no definitive answer came back, so
        # the session may already have moved. A 409 would tell the user the move
        # failed, and they would act on a directory the owner has left. 503 is
        # the ladder's own "reconcile before retrying" shape, and ``code`` is
        # what lets a renderer key on the condition instead of matching prose.
        # The client is already built for this shape: it reads
        # ``detail.message`` when ``detail`` is an object.
        #
        # ``error.detail`` is deliberately NOT echoed — it names sockets, control
        # ports and directories — and is logged at each raise site instead, which
        # is where the cause still exists to be named: the transport raise in
        # ``AttachedSession.set_working_directory`` (where the exception is still
        # live) and the settlement refusal in ``_settle_unconfirmed_move`` (with
        # all four readbacks).
        raise HTTPException(503, {"code": error.code, "message": str(error)}) from None
    except (ReceiptConflict, ValueError) as error:
        from local_operator.session.errors import (
            AttachmentUnavailable,
            ProfileRegistryUnavailable,
            RuntimeRetiring,
        )

        if isinstance(error, (AttachmentUnavailable, ProfileRegistryUnavailable, RuntimeRetiring)):
            # ``RuntimeRetiring`` IS IN THIS TUPLE FOR THE REASON THE ARM'S
            # OTHER FOUR REFUSALS ARE: the code is the contract. A retiring runtime refuses an
            # admission the client can ACT on differently from a broken one —
            # the message was provably not admitted, so the app restores its
            # echo and retries the same id instead of holding the text in the
            # composer as a draft the backend already took (design of record
            # ``docs/design-ownerless-session-attach.md`` §1.6/F5, §6 U1, which
            # read this route as carrying ``{code: "runtime_retiring", ...}``).
            # It reached the bare ``str(error)`` answer below only because it
            # was never listed here, and that answer is the one shape a renderer
            # cannot key on: the category it needs was nowhere in the body.
            #
            # ADDITIVE FOR OLDER CLIENTS, which is what makes listing it here
            # safe rather than a wire change: such a client read ``detail`` as a
            # string and read it as nothing more than that here — the sentence a
            # refusal object carries in ``message`` is character-for-character
            # the one the bare 409 carried, so the only difference it can observe
            # is an object where it expected prose, which it ignores. The same
            # claim the ``MoveIndeterminate`` arm above states for the same
            # body shape ("The client is already built for this shape: it reads
            # ``detail.message`` when ``detail`` is an object").
            raise HTTPException(409, {"code": error.code, "message": str(error)}) from None
        if isinstance(error, SupersededCompletionToken):
            # Stale, not broken: the caller's token is real but no longer current,
            # and the remedy is to re-read the conversation's attention state and
            # acknowledge the token it names. The machine code is what lets the
            # renderer take that path quietly instead of backing off as if the
            # store had refused (the `code` field of its control error).
            raise HTTPException(409, {"code": error.code, "message": str(error)}) from None
        if isinstance(error, SessionDeletionRefused):
            # A DELETION THE MACHINE REFUSED, and it is a 409 rather than the
            # generic ``str(error)`` arm below for the one reason that matters to
            # the client: the sentence is a REMEDY (stop the session, cancel the
            # wake, read the mail) and the code is what lets a renderer offer
            # that control instead of retrying a call that cannot succeed. The
            # conversation exists, so 404 would be a lie about the user's own
            # work, and nothing failed, so 500 would be a lie about the machine.
            raise HTTPException(409, {"code": error.code, "message": str(error)}) from None
        raise HTTPException(409, str(error)) from None
    except sqlite3.Error as error:
        # THREE CONDITIONS, THREE ANSWERS, and the split is the point: this arm
        # used to answer all of them (contention, a full disk, an unopenable
        # store, a corrupt one) with the CONTENTION sentence, raised ``from
        # None`` and logged nowhere. On a full volume that told the operator a
        # read state was momentarily busy and would heal itself, over the one
        # condition no amount of retrying clears -- and the client's hint is
        # exactly "send it again". ``session/store_failures`` owns the
        # classification and the copy; ``_store_refusal`` owns the log record.
        #
        # The text is still NOT echoed for the reason the ConnectionError arm
        # below refuses to echo: a store error can name file paths. ``copy`` is
        # the calling route's own sentence where it has one -- a receipt clear is
        # not a message send, and saying so is the route's job rather than the
        # classifier's (QA round 2, Q1).
        raise _store_refusal(
            request, sqlite_store_failure(error, store_root(request)), error, copy
        ) from None
    except RuntimeUnresponsiveError:
        # BEFORE the generic ConnectionError arm (it subclasses it). The runtime
        # is alive and this host reached it; it did not answer inside the desktop
        # control envelope. Answered fast and typed so the renderer can retry
        # instead of reporting a lost session at its 20 s deadline.
        raise HTTPException(
            503,
            {
                "code": RUNTIME_BUSY,
                "message": RUNTIME_UNREACHABLE_MESSAGE,
                "retryable": True,
                "retry_after_ms": RUNTIME_BUSY_RETRY_AFTER_MS,
            },
            headers={"Retry-After": str(max(1, RUNTIME_BUSY_RETRY_AFTER_MS // 1000))},
        ) from None
    except ConnectionError as error:
        # A cold session that cannot start a runtime reports WHY -- but only when
        # the reason arrives as an `ActionableConnectionError`, whose TYPE is
        # what certifies the message as one of the vetted configuration
        # sentences (`launch._ACTIONABLE_STARTUP_REASONS`).
        #
        # Vettedness is deliberately NOT inferred from the text. Echoing every
        # ConnectionError leaked `owner socket unreachable: [Errno 61] Connect
        # call failed ('127.0.0.1', 54321)` and `owner moved to another
        # conversation (<session id>)` straight to the renderer: `attach_client`
        # raises bare ConnectionErrors carrying socket errors, internal control
        # ports and other sessions' ids. Those keep the generic sentence.
        detail = str(error).strip() if getattr(error, "actionable", False) else ""
        # A CODE, NOT ONLY A SENTENCE (design D8). The status says "this backend
        # could not complete the request", which the renderer cannot tell from a
        # server-wide outage — that conflation is what painted "the Local Operator
        # server is not running" over one conversation. ``code`` is the machine
        # contract the renderer branches on; ``message`` keeps the same vetted
        # sentence it has always carried, because the shipped app matches that
        # prefix for its MCP row and the two repositories must never have to move
        # in step for copy to keep rendering.
        raise HTTPException(
            503,
            {
                "code": RUNTIME_UNREACHABLE,
                "message": detail or RUNTIME_UNREACHABLE_MESSAGE,
            },
        ) from None
    except (RuntimeError, asyncio.TimeoutError):
        raise HTTPException(
            503, {"code": RUNTIME_UNREACHABLE, "message": RUNTIME_UNREACHABLE_MESSAGE}
        ) from None
    except OSError as error:
        # THE LAST ARM, and only for the disk. Placed here rather than beside the
        # sqlite arm because ``ConnectionError`` -- caught above, with its own
        # vetted copy -- is an ``OSError``, and because
        # ``SessionStoreUnavailable`` (the third arm, an ``OSError`` subclass
        # whose sentence is about a store that could not be WALKED) must keep
        # winning for its own condition.
        #
        # Everything this ladder cannot classify is RE-RAISED untouched: it sits
        # under every desktop control-plane route, and answering for arbitrary
        # ``OSError``s would swallow the failures whose own routes have better
        # words for them -- ``move_session`` answers a bad target (an unmounted
        # volume, a symlink loop: ENOENT/ELOOP/ENOTDIR) with a 409 naming the
        # path, and that clause returns ``None`` for exactly those, so the
        # ladder must let them past rather than answer in its own voice.
        #
        # What it does answer is ENOSPC. The non-sqlite writes on the send path
        # (the transcript append, the attachment store) raise this rather than a
        # sqlite error, and a message that could not be persisted is the same
        # condition to the user as a store that could not be written.
        failure = store_failure(error, store_root(request))
        if failure is None:
            raise
        raise _store_refusal(request, failure, error) from None


@router.get("/v1/desktop/sessions", response_model=CRUDResponse[SessionList])
async def list_sessions(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    include_archived: bool = Query(default=False),
    include_peers: bool = Query(default=False),
):
    # Wrapped like its neighbours: the list gained a receipt-store read, and an
    # unmapped failure there answered the app's primary navigation surface with
    # a bare 500. The decoration is already omitted per row inside `list()`;
    # this ladder covers anything else the pool can raise — including the store
    # it could not walk, which is now a typed 503 rather than an empty 200.
    async with errors(request):
        # THE STATUS STAMPS, read WITHOUT constructing the feed. `getattr`
        # rather than `feed(request)` is deliberate: `feed()` BUILDS the
        # singleton (and the poller that comes with it), so calling it here
        # would make every list request start a feed on a backend the desktop
        # app has not opened — and would make the app's first list the reason a
        # machine-wide poller exists. A backend that never opened the feed has
        # no counters, and the rows are simply unstamped, which is the same
        # contract an older backend's rows carry.
        engine = getattr(request.app.state, "desktop_feed", None)
        stamps = engine.status_stamps() if engine is not None else None
        page = await host(request).list(
            limit,
            status_stamps=stamps,
            include_archived=include_archived,
            include_peers=include_peers,
        )
        # THE PAGE, THEN THE PINNED CONVERSATIONS IT DID NOT CARRY, as ONE list.
        # DECIDED, not left open: concatenated on the wire rather than published
        # as a second field, because of what the client does with this array — it
        # REPLACES the rows it is holding with it. A pinned row parked in a
        # sibling field would be a row the client does not hold until it learns
        # about that field, and a client that missed it renders nothing for the
        # pin, which is the exact gap the extra exists to close. A second field
        # would also mean every consumer learns a second merge path for rows it
        # must render identically, while `pinned` already distinguishes them.
        #
        # WHAT THAT COSTS, stated so the next reader does not assume the old
        # invariant: ``len(sessions)`` MAY EXCEED ``limit``. ``limit`` and
        # ``truncated`` continue to describe the PAGE ONLY — the extras are not
        # page rows and do not make the page bigger.
        #
        # ORDER: the page first, then the extras, which is the catalogue's own
        # ranking continued below the page — the same order the page's rows
        # arrive in, and the same order the TUI's ``★ Pinned`` section draws. It
        # is deliberately NOT pin recency: the store holds that (newest pin
        # first) and it is one of the few orderings the two surfaces could
        # disagree about, so ordering the extras by it would put a second
        # ordering authority inside one section and make the app's Pinned list
        # read as catalogue order followed by pin order.
        sessions = page.rows + page.pinned_off_page + page.remote
        # The sources that could not be read for THIS page. Lifted from the rows
        # rather than plumbed beside them: every row of a poll carries the same
        # verdict (one registry scan answers for the whole listing), so the
        # listing-level statement is derivable, and a second channel through
        # `list()` would be one more thing a caller can forget to pass. Sorted
        # so the set is stable across polls, and computed over what is actually
        # sent — a row the page does not carry AND this answer does not send says
        # nothing about it. (The extras below ARE sent, so they are included.)
        #
        # The stamps and this marker are INDEPENDENT facts about the same rows
        # and neither may displace the other: a stamp answers "is this row newer
        # than the frame you already applied" for a session the feed publishes,
        # the marker answers "was every read behind this row the one that
        # produced it" — a listing can be fully stamped and still be degraded.
        degraded = sorted({source for row in sessions for source in row.get("degraded") or ()})
        return reply(
            {
                "sessions": sessions,
                "truncated": page.truncated,
                "limit": limit,
                "degraded": degraded,
            }
        )


@router.get("/v1/desktop/sessions/search", response_model=CRUDResponse[SessionSearch])
async def search_sessions(
    request: Request,
    q: str = Query(default="", max_length=256),
    limit: int = Query(default=100, ge=1, le=500),
    include_archived: bool = Query(default=False),
    include_peers: bool = Query(default=False),
):
    """Past conversations matching ``q`` by name, id, or what was SAID in them.

    The same search the CLI's ``/resume`` picker runs, through the one
    implementation the picker and the phone daemon share
    (``session_search.search_store``): name and id as exact case-insensitive
    substrings over the haystack the row is RENDERED with, plus the cached body
    digest index for an exact conversation match, plus a bounded soft tier
    (prefix, word-order, edit distance <= 2 on words of 4+ characters) when the
    query is not already precisely answered. Results come back best-first with
    the tier and the reason attached.

    **Declared BEFORE ``/v1/desktop/sessions/{session_id}``** and that order is
    load-bearing: FastAPI matches routes in declaration order, so a parent route
    declared first would swallow this path and the client would get the
    snapshot of a session literally named "search" (a 404, in practice) instead
    of an answer.

    The scan and the index build run off the event loop — they read every
    session directory's head and one cache file — while ``q`` is bounded at 256
    characters because the query is only ever a user's typing, and an unbounded
    one would be projected into every digest comparison.

    ``include_archived`` defaults FALSE and the default is the contract: an
    archived conversation is not a search result, which is why the archive had
    to be threaded through the scanning layer rather than filtered here — the
    index is built from the rows the scan returned, so an archived id is never
    handed to it and cannot be reached by a body match. Raising the flag is a
    deliberate act for a surface that has already revealed them (the picker's
    reveal toggle), and every hit carries its own ``archived`` either way.
    """
    async with errors(request):
        return reply(
            {
                "sessions": await host(request).search(
                    q, limit, include_archived=include_archived, include_peers=include_peers
                ),
                "query": q,
                "limit": limit,
            }
        )


@router.post("/v1/desktop/sessions", response_model=CRUDResponse[CreatedSession])
async def create_session(body: CreateSession, request: Request):
    """Create a new conversation, optionally born on a chosen model and effort.

    The body's admissions run BEFORE the receipt is claimed, in the same order
    ``preview`` applies them (cwd, target, model), for two reasons. A refusal has
    to leave the store exactly as it found it — the claim is a durable write, and
    a claimed-then-refused request would otherwise answer its own retry with
    "outcome indeterminate" instead of the refusal. And the two routes are
    documented as answering the same refusals, which is only true if they also
    agree on WHICH refusal a body that is bad in two ways gets (review round 1,
    R4).

    All three are the admissions ``DesktopSessions.create`` itself applies, so a
    body that passes here passes there — and the pool runs them again rather than
    trusting this pre-flight, because that is the callable's own contract and a
    second caller must not be able to reach it unvalidated. Two of the three are
    pure reads (a ``stat`` for the directory; the catalogue and the metadata cache
    for the model, which writes nothing at all), and they run BEFORE the target's,
    which builds the registries ``create`` would build anyway — on a fresh root that
    materialises ``<config>/agents``, so it is the one admission with a filesystem
    effect (review round 2, R8; the preview route documents the same carve-out).

    A REPLAYED request runs none of them: a retry whose directory has since vanished
    must answer what the first attempt recorded, not turn a success into a refusal
    (review round 2, R7).

    No ``model`` ⇒ the body, the marker and the launch are byte-for-byte today's.
    """

    async def create():
        if body.peer:
            # THE PEER MINTS THE ID (``relay._op_session_create``'s rule): an id that
            # exists in two places at once is the permanent routing ambiguity the
            # mesh exists to prevent, so this device does not choose one.
            #
            # ``cwd`` IS DELIBERATELY NOT FORWARDED. The field names a path on THIS
            # machine — the renderer sends the directory it is showing — and a peer
            # asked to create a session there would either fail or, worse, land in a
            # directory that merely happens to share the path on its disk. An empty
            # cwd makes the peer default to its own home, which is the same rule
            # ``session/remote_open`` states for a remote attach.
            assert spec is not None or body.model is None
            # Imported here rather than at module scope: the mesh package is a boot
            # cost this file must not add for every backend, and a create that names
            # no peer never needs it (``routes/auth.py``'s rule for its own imports).
            from local_operator.network.types import MeshRefusal
            from local_operator.server.utils.desktop_mesh import create_on_peer
            from local_operator.server.utils.desktop_receipts import Unclaimed

            try:
                detail = await asyncio.to_thread(
                    create_on_peer,
                    host(request).root,
                    body.peer,
                    model=(
                        {
                            "provider": spec.provider,
                            "model_id": spec.model_id,
                            "reasoning_effort": spec.reasoning_effort,
                        }
                        if spec is not None
                        else None
                    ),
                )
            except MeshRefusal as error:
                document = {
                    "refused": True,
                    "code": error.code,
                    "message": str(error),
                }
                if error.code in _CREATE_UNCONFIRMED_CODES:
                    # THE REQUEST MAY HAVE ARRIVED: the peer stopped answering after
                    # this device sent the frame, so the session may exist on it. The
                    # claim is KEPT (returned, not raised) so a retry replays this
                    # answer instead of minting a second conversation.
                    return document
                raise Unclaimed(document) from None
            return {
                "session_id": str(detail.get("session_id") or ""),
                # A remote session has no LOCAL attachment: the binding names the
                # agent or team recorded on the session's own marker, which lives on
                # the peer, and guessing from this device's registries would be a
                # claim about a store that does not hold the session.
                "binding": {"agent": None, "team": None},
            }
        pool = host(request)
        target = body.target.model_dump() if body.target else None
        session_id = await pool.create(
            body.cwd,
            target=target,
            # The NORMALISED triple, not the raw body: the marker is what a later
            # engage re-resolves, so it stores the pair the validation just
            # proved servable (provider alias resolved, level lowercased).
            model=(
                {
                    "provider": spec.provider,
                    "model_id": spec.model_id,
                    "reasoning_effort": spec.reasoning_effort,
                }
                if spec is not None
                else None
            ),
        )
        return {"session_id": session_id, "binding": await pool.binding(session_id)}

    async with errors(request):
        # REFUSED BEFORE ANYTHING IS CLAIMED OR ADMITTED — before the receipt is
        # claimed and before the draft's own admissions (the working directory, the
        # model spec, the target registry) run: a refused request must leave no
        # pending receipt behind, or the client's retry against the SUCCESSOR would
        # meet the indeterminate 409 the receipts layer reserves for a crashed
        # attempt (``desktop_receipts``). ``DesktopSessions.create`` re-asks the
        # same question as its first statement, so a caller that reaches the
        # adapter another way gets the same refusal.
        host(request).assert_admitting()
        pool = host(request)
        key = "create:" + body.request_id
        spec: ModelSpec | None = None
        # A recorded key short-circuits the admissions below: see the docstring.
        # ``recorded`` is a read — it neither claims the key nor creates the store.
        if not await asyncio.to_thread(receipts(request).recorded, key):
            # The SAME admissions ``create`` applies, in the SAME order ``preview``
            # applies them: the working directory, then the model (which writes
            # nothing), then the target (whose registry build materialises
            # ``<config>/agents``). See ``DesktopSessions.create`` /
            # ``resolve_working_directory``.
            #
            # A PEER CREATE SKIPS THE TWO THAT ARE ABOUT THIS MACHINE, and refuses
            # the third rather than re-interpreting it: ``cwd`` names a path HERE and
            # ``target`` names a registry HERE, neither of which the peer shares. The
            # model IS admitted (its catalogue is global, so the same pick is servable
            # on the peer and a bad one still 422s here), and a target on a peer
            # create is refused in words instead of being silently dropped — a session
            # born on the wrong agent is exactly what the pick was for.
            if body.peer:
                if body.target is not None:
                    raise HTTPException(
                        422,
                        "a target cannot be chosen for a conversation created on another "
                        "device yet: its agents and teams live on that device. Create the "
                        "conversation here, or pick the target after moving it home.",
                    )
                if body.model is not None:
                    spec = await asyncio.to_thread(_draft_model_spec, body.model)
            else:
                await asyncio.to_thread(resolve_working_directory, body.cwd)
                if body.model is not None:
                    # Off the loop: the catalogue it reads is a disk document, and for
                    # an unshipped model the metadata resolver may consult the provider.
                    spec = await asyncio.to_thread(_draft_model_spec, body.model)
                if body.target is not None:
                    target_row = body.target.model_dump()
                    from local_operator.agents import AgentRegistry
                    from local_operator.server.utils.desktop_profiles import (
                        validate_target,
                    )
                    from local_operator.teams import TeamRegistry

                    await asyncio.to_thread(
                        validate_target,
                        AgentRegistry(pool.root),
                        TeamRegistry(pool.root),
                        target_row["kind"],
                        target_row["name"],
                    )
        result = await receipts(request).run(key, body.model_dump(), create)
        if result.get("refused"):
            # RAISED AFTER THE JOURNAL SETTLED THE CLAIM, the wakes family's rule: a
            # recorded refusal is what a retry replays, a released one re-runs. The
            # MESSAGE IS THE PEER'S own sentence, never re-derived here — that device
            # is the only party that saw which guard or which rung fired.
            code = str(result.get("code") or "peer_refused")
            raise HTTPException(
                503 if code in _CREATE_UNCONFIRMED_CODES else 409,
                {"code": code, "message": str(result.get("message") or "that device refused")},
            )
        return reply(result)


@router.post("/v1/desktop/sessions/preview", response_model=CRUDResponse[DraftPreviewPayload])
async def preview_session(body: DraftPreview, request: Request):
    """The readings a new-conversation pane may show, for a session that is not.

    **Declared BEFORE ``/v1/desktop/sessions/{session_id}``** for the same reason
    ``search`` is: FastAPI matches in declaration order, and although no POST
    route exists under that path today, the client would otherwise get whatever
    one a later edit adds, for a session literally named "preview".

    A deliberately session-LESS op, and every absence below is the point. There
    is no session record, nothing under ``sessions/`` (no directory, no marker, no
    runtime lease), nothing handed to the canonical store, and no receipt row — a
    pane that is only being OPENED must cost nothing durable. The alternative
    (create the record at pane open, then cold-GET it) is honest but leaves a
    visible empty row in the sidebar for every abandoned pane: ``create`` writes a
    directory plus a marker, and a marker-only directory is listed.

    ONE filesystem effect it does share with ``create``, named here rather than
    glossed as "no directory": a ``target`` is validated against
    ``AgentRegistry``, whose constructor materialises ``<config>/agents``
    (``agents.py``; ``TeamRegistry`` deliberately does not mkdir). That is the
    object the design mandates reusing — a second, non-mkdir'ing registry would be
    a second profile-resolution path, which is the defect class this PR exists to
    remove. The ``cwd`` check below runs FIRST, so a bad working directory is
    refused before any registry is built.

    The state comes from the SAME synthesis a real cold open uses
    (``session.cold_model``), so the draft's model chip cannot disagree with the
    model the first send actually gets — the UI swaps this payload for the first
    cold frame at ``finishDraft``, where a flicker is visible.

    The account-metadata step is SKIPPED, unlike a cold session: a draft has no
    context reading to divide, and a synthetic stickiness key must not move a
    real account's stickiness. So the window this pane publishes is the MODEL's
    own (``build_model_spec``'s), carried with
    ``context_metadata_resolved: False`` — it is NOT a placeholder, and it is NOT
    a promise about the first turn: a cold open may apply plan-scoped account
    metadata and a window that metadata scopes, which is exactly the read this op
    must not perform. The LADDER and the LEVEL do agree with the first turn by
    construction (see ``session.cold_model.resolve_birth_effort``); the effective
    window cannot, and nothing here claims it does.

    ``target`` is validated exactly as ``create`` validates it, so an
    unresolvable profile fails here rather than becoming a session that cannot
    start — the pane renders no strip instead of a reading for a session that
    cannot exist.
    """

    async def preview():
        pool = host(request)
        # The SAME admissions ``create`` applies, in the SAME order, so one body gets
        # one answer from either route (and 409, not a 200 describing a session that
        # could never be created). See ``DesktopSessions.create`` /
        # ``resolve_working_directory``.
        await asyncio.to_thread(resolve_working_directory, body.cwd)
        # The MODEL before the TARGET, as in ``create``: the model admission writes
        # nothing at all, while building the registries the target needs
        # materialises ``<config>/agents`` (review round 2, R8).
        birth_model = (
            await asyncio.to_thread(_preview_birth_model, pool.root, body.model)
            if body.model is not None
            else None
        )
        if body.target is not None:
            target = body.target.model_dump()
            from local_operator.agents import AgentRegistry
            from local_operator.server.utils.desktop_profiles import validate_target
            from local_operator.teams import TeamRegistry

            await asyncio.to_thread(
                validate_target,
                AgentRegistry(pool.root),
                TeamRegistry(pool.root),
                target["kind"],
                target["name"],
            )
        state = await synthesise_cold_state(
            config_dir=pool.root,
            session_id="",
            cwd=body.cwd,
            # The chosen selection, when there is one, is synthesised exactly as
            # the first cold frame of the session it describes will be — same
            # resolver, same metadata — so the pane's identity, its effort LADDER
            # and the LEVEL the first turn runs at are the ones the first turn
            # gets rather than the configured default's. The WINDOW is the one
            # reading that cannot agree and is not claimed to: the account-metadata
            # step above is skipped, so this pane publishes the MODEL's window with
            # ``context_metadata_resolved: False``, while a cold open may apply a
            # plan-scoped account window a draft must not read. With NO selection
            # the same synthesis answers the CONFIGURED pair, and
            # ``session.cold_model`` resolves that through the model's own metadata
            # too, so the ladder and the level are there for an unpicked draft as
            # well (review round 2's effort-reading defect).
            # Still session-less and side-effect free: this is an INPUT, and the
            # state below writes nothing (see the docstring above).
            birth_model=birth_model,
        )
        sync = FrontendSync(
            epoch=state.epoch,
            sequence=state.sequence,
            snapshot=state,
            live_cursor=state.history_cursor,
        )
        return {"frontend": sync_wire_payload(sync)}

    async with errors(request):
        return reply(await preview())


@router.get("/v1/desktop/sessions/{session_id}", response_model=CRUDResponse[SessionSnapshot])
async def snapshot(session_id: str, request: Request):
    # READ: an existing but silent owner must not fail a read. The durable answer
    # is on disk in this same process, so the attempt is bounded
    # (``READ_ATTACH_BUDGET_S``) and the cold facade serves it with a
    # ``cold_reason``; the previous envelope answered 503 "Session owner is
    # unavailable" after ~17 s for a runtime whose loop was merely busy.
    #
    # A PEER'S CONVERSATION IS ANSWERED IN WORDS, NOT AS UNKNOWN (see
    # :func:`_remote_open_refusal`): this is the read a click on a row reaches
    # first, and until the remote viewer is wired to the desktop's bridge the row
    # a user can SEE must not answer "Requested session … not found" about their
    # own conversation. Deferred deliberately — the viewer is its own slice, and a
    # half-wired one that answered 200 with nothing to drive would be worse.
    await _remote_open_refusal(request, session_id)
    async with errors(request), host(request).session(session_id, read=True) as bridge:
        return reply(await bridge.snapshot())


async def _remote_open_refusal(request: Request, session_id: str) -> None:
    """Refuse a peer's id IN WORDS when the desktop cannot open it, else do nothing.

    THE ROW A USER CAN SEE MUST NOT READ AS UNKNOWN. With ``include_peers`` a
    sidebar row can name a conversation another device holds, and a click reaches
    this route first; without this, the answer was the shared 404 ("Requested
    session, profile, team or subscription not found") about the user's own
    conversation — a dead affordance whose sentence is also false. So the id is
    looked up in the peer projection (cache-only on a hit, one cached read on a
    miss, and NO relay work at all on a machine in no network) and answered as a
    409 whose ``message`` names the device and the two ways to work with the
    conversation today.

    409 RATHER THAN 404, by the rule the delete route states for its own refusals:
    the conversation exists and the user can see it, so "not found" would be a lie
    about their own work. The ``code`` is what a renderer branches on; the sentence
    is what a person reads, and it is written in the same register as the TUI's own
    remote-session notice (that surface names the device and the way in).

    THE COST ON THE ORDINARY PATH IS ONE CACHE LOOKUP, and this is the important
    half: every local session answers from the projection's own directory check,
    which is why this can sit in front of the hottest read on this plane.
    """
    from local_operator.server.utils.desktop_mesh import remote_owner

    owner = await asyncio.to_thread(remote_owner, host(request).root, session_id)
    if owner is None:
        return
    device_id, device_name = owner
    label = device_name or device_id
    raise HTTPException(
        409,
        {
            "code": "session_is_remote",
            "message": (
                f"{session_id} lives on {label}, and this desktop cannot open a "
                "conversation on another device yet. Move it home with "
                f"`lop sessions move {session_id} --to local`, or pilot it from the "
                f"terminal with `/network sessions --peer {label} --engage {session_id}`."
            ),
        },
    )


@router.get("/v1/desktop/sessions/{session_id}/history", response_model=CRUDResponse[HistoryPage])
async def history(
    session_id: str,
    request: Request,
    before_id: str | None = Query(default=None, max_length=128),
    limit: int = Query(default=100, ge=1, le=500),
):
    # READ, for the same reason as ``snapshot`` beside it.
    async with errors(request), host(request).session(session_id, read=True) as bridge:
        return reply(await bridge.history(before_id=before_id, limit=limit))


@router.get(
    "/v1/desktop/sessions/{session_id}/children/{child_id}/transcript",
    response_model=CRUDResponse[ChildTranscriptPage],
)
async def child_transcript(
    session_id: str,
    child_id: str,
    request: Request,
    before_id: str | None = Query(default=None, max_length=128),
    limit: int = Query(default=100, ge=1, le=CHILD_PAGE_LIMIT),
):
    """One page of a subagent's own transcript, read through its parent.

    The sidebar's child reader, and the ONLY door to a child's conversation:
    the renderer sends ids and never a ``session_dir``, so the backend (see
    ``_contained_child_dir``) is what proves the child belongs to the named
    parent, is a subagent rather than the user's own conversation or a fork,
    and lives under ``sessions/``. Refusals are 404 ``child_not_found``.

    The envelope is ``/history``'s, verbatim — ``entries`` are the child's raw
    transcript rows — plus the derived ``state``. ``pending`` and ``gone`` are
    answers, not errors: neither is a status the caller can act on, and both
    must be distinguishable from a refusal or the panel would report a missing
    child as "not yours".

    Read-only in the strongest sense: no bridge, no runtime, no message
    admission — a paused conversation answers exactly like a running one.
    """
    async with errors(request):
        return reply(
            await host(request).child_transcript(
                session_id, child_id, before_id=before_id, limit=limit
            )
        )


@router.get(
    "/v1/desktop/sessions/{session_id}/children/{child_id}/attachments/{digest}",
    # Same declaration as the parent's route, and for the same reason: the
    # response is raw bytes with the stored image's own type, which is not the
    # CRUD envelope its neighbours return.
    response_class=Response,
)
async def child_attachment(
    session_id: str, child_id: str, digest: AttachmentDigest, request: Request
):
    """Raw bytes of an attachment referenced by a CHILD transcript's rows.

    The parent's route cannot serve these: it takes the session whose transcript
    holds the reference, and a child's rows reference attachments in the same
    content-addressed store. Everything else is deliberately identical — the
    digest is the traversal gate (declared in the path, so FastAPI refuses a
    non-matching shape before the handler runs), the stored mime is allowlisted
    rather than trusted, ``nosniff`` rides the response, and a missing
    attachment is an ordinary 404.

    The route's scope is its parent's: containment proves the caller may read
    THIS child, and the store is shared across conversations by design, so the
    digest is not partitioned per child. The bearer already authorises the whole
    desktop surface; what this route must not become is a way to reach a child
    that is not the named session's, which ``_contained_child_dir`` refuses.
    """
    async with errors(request):
        data, mime_type = await host(request).child_attachment(session_id, child_id, digest)
    return Response(
        content=data,
        media_type=(
            mime_type if mime_type in SUPPORTED_IMAGE_MIME_TYPES else ATTACHMENT_FALLBACK_MIME
        ),
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.get(
    "/v1/desktop/sessions/{session_id}/attachments/{digest}",
    # The image's own mime type is the response type, so the published contract
    # must not claim the CRUD JSON envelope its neighbours return. Declaring the
    # class is what keeps the schema honest; without it FastAPI documents
    # `application/json` for a route that never sends any.
    response_class=Response,
)
async def attachment(session_id: str, digest: AttachmentDigest, request: Request):
    """Raw bytes of one content-addressed attachment referenced by a history row.

    ``/history`` serves durable rows verbatim, and a durable image block is
    ``{"attachment": <digest>, "mime_type": ...}`` with the payload stripped
    (``transcript._externalize_attachments``). Without this route a frontend can
    see that an image was in the conversation and can never render it after a
    reload — the live event stream carries the base64, the durable transcript
    does not.

    Three deliberate choices:

    - **Keyed by digest, not by (entry, index).** The history row hands the
      caller the digest directly, so nothing has to re-fold history per image,
      and identical screenshots across a whole conversation collapse to one
      request and one cache entry.
    - **The digest pattern is the traversal gate.** ``_DIGEST_CHARS`` is 32 hex
      characters and the store turns a digest straight into a filename, so the
      shape is validated in the path declaration rather than inside the handler
      where a later edit could route around it. FastAPI answers a non-matching
      path with 422 before any disk access.
    - **No HTTP caching, deliberately.** The digest IS the sha256 of the
      content, so an ``immutable`` response would be correct about the BYTES —
      and it is still the wrong header here. ``managed_desktop_boundary``
      applies ``Cache-Control: no-store`` to everything under ``/v1/desktop/``
      because these responses are bearer-gated session data, and a route that
      set ``public, max-age=31536000`` would either be silently overridden (it
      was: measured ``no-store`` on the wire) or, if the middleware were
      carved out for it, would invite a shared cache to retain one user's
      screenshots. The mobile daemon's equivalent route can afford
      ``immutable`` because it is a different process behind different auth;
      copying the header without the surrounding argument would not be reuse.
      Dedup belongs to the client, which already holds a digest-keyed cache
      for exactly this reason — and the digest being content-addressed is what
      makes that cache safe.

    A missing attachment is 404, never 500: the store's contract is that an
    unresolvable reference is ordinary (interrupted write, hand-pruned store)
    and the reader degrades to a placeholder. A corrupt-but-parseable sidecar
    is the same class of ordinary, which is why the mime is allowlisted rather
    than trusted (see :data:`ATTACHMENT_FALLBACK_MIME`).

    Two scopes this route does NOT have, stated because the URL shape implies
    otherwise:

    - ``session_id`` is an EXISTENCE check, not a binding. It proves *a* user
      conversation by that name is on this machine; it does not prove this
      attachment belongs to it. The store is content-addressed and shared
      across conversations by design, so any valid user session id resolves
      any digest in it. That is not an escalation — the bearer already
      authorises the whole desktop surface, ``/history`` included — but a
      future reader must not mistake the path segment for authorization.
    - The served ``Content-Type`` is not guaranteed to equal the ``mime_type``
      on the history row. The store dedups by content digest and the FIRST
      sidecar wins, so identical bytes stored once as ``image/png`` and later
      as ``image/gif`` keep the original sidecar while the newer row reports
      ``image/gif``. Harmless for real images (the bytes decide what renders),
      but the two values are not a matched pair.
    """
    async with errors(request):
        data, mime_type = await host(request).attachment(session_id, digest)
    return Response(
        content=data,
        media_type=(
            mime_type if mime_type in SUPPORTED_IMAGE_MIME_TYPES else ATTACHMENT_FALLBACK_MIME
        ),
        # Defence in depth on the one response here that can carry a type the
        # caller did not choose: with the allowlist above, an unexpected mime
        # is served as opaque bytes, and nosniff stops a browser from
        # re-deciding that for itself. `tunnels/gateway.py` sets the same
        # header on this repo's other byte-serving surface.
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.post(
    "/v1/desktop/sessions/{session_id}/messages", response_model=CRUDResponse[MessageAdmission]
)
async def prompt(session_id: str, body: Prompt, request: Request):
    """Admit ONE user turn, or refuse because this process is leaving.

    The refusal is the pool's door (``DesktopSessions.session``), which this
    route enters before it claims anything: it is the FIRST thing the handler
    does, before the receipt is claimed, and it covers ``admit_prompt``'s
    ``_ensure_bound`` (``session/attached.py``) — the one path in the desktop
    plane that can also START a session runtime, which is what ``warm``'s own
    gate exists to prevent (review round 1, MAJOR-2 measured this route
    answering 200 on a latched daemon while ``/warm`` answered 503 against the
    same process).

    The gate is at the DOOR rather than in this handler because this handler is
    one of fourteen that obtain a bridge, not the admission path itself (review
    round 2, MAJOR-1): see ``DesktopSessions.session`` for why the
    route-by-route question was the defect.

    NOT refused while the daemon is merely ANNOUNCED, which is why the gate is
    on the latch and not on the record: an announced daemon is still the only
    place its client can work (``server/retire.py``).
    """
    # THE ACKNOWLEDGEMENT GOES OUT BEFORE THE DOOR, AND THE OUTCOME FOLLOWS IT.
    # Both positions are the whole point of this handler:
    #
    # * BEFORE ``host(request).session(...)``, because that is what acquires a
    #   bridge, and a cold acquire waits on the bind lock behind any speculative
    #   warm the visible ``/watch`` lease armed — measured at ~1.0 s
    #   (``_BACKGROUND_YIELD_BUDGET_S``), which is exactly the case where a
    #   mounted viewer is waiting to be told something. The announcement takes no
    #   reference, builds nothing and can start nothing
    #   (``DesktopSessions.announce_admission``), is idempotent per request id, so
    #   the retry this route tolerates cannot announce twice, and declines on a
    #   latched daemon rather than contradicting the door's refusal.
    # * AND A REFUSAL RESOLVES IT, through the same reference-free path, because
    #   everything that can refuse this request — the door's dial, the owner's
    #   answer, the receipt journal's own conflicts — happens AFTER the
    #   acknowledgement is on the wire. Without the outcome frame the viewer (and
    #   every later viewer reading it from the replay) is left holding a promise
    #   that never resolves, while the caller learns what happened from a 5xx it
    #   may not even surface. The frame carries this request id, so the pair
    #   reads as one submit.
    #
    # THE SPECULATIVE WARM PUBLISHES NEITHER. A visible ``/watch`` lease arms a
    # warm of its own, and that spawn races this route's engage; only the submit
    # path owns the request, so only it acknowledges and only it resolves — which
    # is what keeps each frame to one per submit rather than one per racing path.
    #
    # THE WHOLE BODY SITS INSIDE ``errors(request)`` so a store failure on the
    # way in is classified by the same ladder as one on the way through: the
    # announcement is the first thing the handler does, and it must not be the
    # one step that answers a bare 500 (review round 1, R6-note).
    async with errors(request):
        announced = await host(request).announce_admission(
            session_id, request_id=body.request_id, mode=body.mode
        )
        #: Whether the owner ADMITTED the turn. From that moment the request is
        #: the owner's: a later failure in this same request — recording the
        #: receipt, refreshing the watch lease — is not a refusal of the
        #: admission, and publishing one as ``admission.failed`` would tell every
        #: viewer the text was dropped while the turn is running. The
        #: acknowledgement resolves through the turn's own frames instead.
        admitted = False
        try:
            async with host(request).session(session_id) as bridge:

                async def admit():
                    nonlocal admitted
                    assert bridge.remote is not None
                    detail, duplicate = await bridge.remote.admit_prompt(
                        body.text,
                        command_id=body.request_id,
                        images=[image.model_dump() for image in body.images],
                        steer=body.mode == "steer",
                    )
                    admitted = True
                    # Admission can bind a cold viewer while an event
                    # subscription is already open. Apply only its still-live
                    # lease, never resurrect one.
                    await bridge.refresh_watch()
                    return {
                        "status": "admitted",
                        "command_id": body.request_id,
                        "duplicate": duplicate,
                        "detail": detail,
                    }

                return reply(
                    await receipts(request).run(
                        session_id + ":" + body.request_id,
                        body.model_dump(),
                        admit,
                        retry_safe=True,
                    )
                )
        except BaseException as error:
            # ``announced`` rather than an unconditional publish: a request this
            # host never acknowledged (no resident viewer, a latched daemon, an
            # unknown session) owes no outcome, and publishing one would invent a
            # failure for a submit nobody was told about. ``not admitted`` for the
            # mirror-image reason: a request the owner DID take is the turn's, and
            # the turn is what resolves it.
            if announced and not admitted and _refusal_resolves_the_acknowledgement(error):
                await host(request).announce_admission_failure(
                    session_id,
                    request_id=body.request_id,
                    mode=body.mode,
                    detail=_admission_failure_detail(error),
                )
            raise


def desktop_viewer_must_submit(receipt_type: Any) -> bool:
    """Whether THIS route owes the request a slash receipt carries.

    The receipt vocabulary is shared with the runtime and the ownership rule is
    one predicate — :func:`runtime_must_complete`: the RUNTIME submits the
    request only when the dialing client did NOT declare that receipt type as
    its own. The desktop viewer is a DECLARING client — ``AttachedSession``
    dials with ``slash_consumers=list(ATTACHED_SLASH_CONSUMERS)`` on every
    surface, ``desktop`` included — so for the receipts in that vocabulary the
    runtime deliberately stands down, and the submit is this host's job.

    MEMBERSHIP COMES FIRST, and that clause is load-bearing rather than
    decorative: ``runtime_must_complete`` answers False both for "the client
    declared this type" and for "this is not an action receipt at all", so the
    bare inversion would claim every notice — including one carrying no request —
    as this host's to complete.

    THE ASSUMPTION, now readable rather than assumed: this host claims the WHOLE
    vocabulary because the client it serves declares the whole list. It reads
    that declaration — ``session/attached.py::ATTACHED_SLASH_CONSUMERS``, the
    value ``AttachedSession`` actually dials with — rather than the vocabulary
    itself, so the second half of the predicate decides something: a client kind
    that declared a SUBSET would have this host stand down for the types it did
    not declare, instead of racing the runtime into two user turns from one
    command. "The declaration covers the vocabulary" is pinned by a guard file
    (``tests/unit/server/test_desktop_goal_admission.py``), so a narrowing edit
    fails a test rather than double-submitting.

    The function-local import is what keeps the drift guard honest: the tuple is
    read AT CALL TIME, so a test that extends the vocabulary watches this answer
    change, where a module-level binding would freeze the answer it set out to
    check.

    NOT THE TUI'S FULL RULE, deliberately. The TUI also completes a TYPELESS
    legacy goal receipt (``tui/app.py``'s ``legacy_goal``: an older owner that
    reported ``stored`` and never admitted the turn). This host cannot: a
    receipt with no ``type`` is not in the vocabulary, no declaration covers it,
    and the runtime a desktop session talks to is never older than the client
    that spawned it.
    """
    from local_operator.session.attached import ATTACHED_SLASH_CONSUMERS
    from local_operator.session.runtime.types import (
        SLASH_ACTION_RECEIPTS,
        runtime_must_complete,
    )

    # The two halves of the one rule, read in the order that makes them true:
    # the type is an action receipt, and this host's client DECLARED it (so the
    # runtime stood down and the submit is ours). They coincide today because
    # the declaration IS the vocabulary; both are stated because a change to
    # either half has one place to be made and one test to fail.
    return receipt_type in SLASH_ACTION_RECEIPTS and not runtime_must_complete(
        receipt_type, ATTACHED_SLASH_CONSUMERS
    )


@router.post(
    "/v1/desktop/sessions/{session_id}/commands", response_model=CRUDResponse[CommandReceipt]
)
async def command(session_id: str, body: Command, request: Request):
    """Run ONE slash command, or refuse because this process is leaving.

    Refused by the pool's door (``DesktopSessions.session``) for the same reason
    as ``prompt`` and at the same seam, and here it matters twice:
    ``bridge.remote.bind_runtime()`` below is an explicit ``_ensure_bound``, so a
    latched daemon would start a runtime for a command alone. The door runs
    before the receipt is claimed, so a refused command leaves no receipt row for
    the client's retry against the successor to trip over.
    """
    spec = slash_command_for("/" + body.command.removeprefix("/"))
    if spec is None or not spec.desktop_destination:
        raise HTTPException(422, "Unknown command")
    if spec.name == "credential" and body.args:
        # The ONE command whose trailing text this route REFUSES rather than
        # consumes, because the secret is entered in the masked form. Left as its
        # own check because the sentence is about the FORM, not about a shape
        # `command_argument_refusal` validates.
        #
        # It is NOT a row the admission rule calls prose, and that is the half
        # this comment used to get wrong: the registry publishes
        # `argument_shape=ANY` for it, so the messages endpoint reads a
        # whole-draft `/credential <secret>` as the command and answers 422 too.
        # The two 422s are one policy — the text belongs to the masked form —
        # and the registry's `ANY` is what keeps the secret out of a paid turn
        # for a client whose command surface is off and which therefore plans
        # every draft as `send`.
        raise HTTPException(
            422, "Enter credentials in the masked credential form, not command text"
        )
    refusal = command_argument_refusal(spec, body.args)
    if refusal is not None:
        # ONE derivation with the messages endpoint's admission test: the shape
        # validators live in `slash_commands` beside the registry, so the route
        # cannot start refusing text the admission rule accepts (or the reverse)
        # — the `/mcp logout` / `/login openai` class where a control was accepted
        # as a message while the route would still have run it.
        raise HTTPException(422, refusal)
    async with errors(request), host(request).session(session_id) as bridge:

        async def execute():
            if (
                (spec.name == "team" and (body.args == "chart" or body.args.startswith("chart ")))
                or (
                    spec.name == "approvals"
                    and (body.args == "default" or body.args.startswith("default "))
                )
                or spec.name not in OWNER_COMMANDS
                or (
                    not body.args
                    and spec.name
                    in {"rename", "model", "effort", "fast", "approvals", "team", "agent", "loop"}
                )
            ):
                return {"command": spec.name, "result": native_action(spec, session_id, body.args)}
            assert bridge.remote is not None
            await bridge.remote.bind_runtime()
            await bridge.refresh_watch()
            outcome = await bridge.remote.route_shared_slash(
                spec.name,
                body.args,
                images=await decode_images(body.images),
            )
            if outcome is None or outcome.get("kind") == "noop":
                return {"command": spec.name, "result": native_action(spec, session_id, body.args)}
            outcome = SlashResult.model_validate(outcome)
            result = outcome.model_dump(mode="json")
            if outcome.kind == "error" and outcome.data.get("code") in {
                "loop_invalid",
                "loop_busy",
                # The third loop refusal: `/loop --clear` while the driver RUNS.
                # It rode a 200 error receipt before, so a client that reads the
                # status could not tell the refusal from a success — on the very
                # surface the flag exists for (round 1, reviewer MINOR-4). It
                # takes the 409 arm with its siblings' `outcome.text`, the
                # sentence that names `/loop --stop`.
                "loop_running",
            }:
                raise HTTPException(
                    422 if outcome.data["code"] == "loop_invalid" else 409, outcome.text
                )
            consumed = outcome.data.get("request", "")
            # The receipt's typed discriminator is the ONLY thing that decides
            # whether a request still needs a home — the runtime returns
            # attachment metadata for an attach and the goal text for a goal,
            # never a started turn — and it is read through the shared
            # vocabulary so a newly declared receipt cannot be missed here.
            # See ``desktop_viewer_must_submit`` for the ownership rule.
            #
            # ORDER, matching the TUI's (``app.py::_cmd_goal``): the goal is
            # stored and the receipt built BEFORE this admits anything — so
            # "goal set" describes the state the run began under — and the
            # admission is reported INSIDE that same receipt rather than as a
            # second answer the caller has to correlate.
            #
            # A NON-EMPTY request is the other half, and images alone are not a
            # substitute for it: an action-less receipt (``/agent clear`` returns
            # ``agent_attached`` with an empty request) carries no ask, and the
            # body's staged images are the CALLER's, not the receipt's — an
            # image-only turn opened for one is a paid turn and a durable row
            # nobody asked for. Images still ride along with a real request:
            # ``consumed`` is what admits, and they are passed with it.
            #
            # Deliberate difference from the TUI path, documented here: the text
            # submitted is the receipt's own ``request`` — the argument as the
            # runtime recorded it — and the body's structured images are passed
            # through unchanged. There is no composer on this host, so there is
            # no attachment map to resolve ``[Image #N]`` markers or collapsed
            # pastes against, unlike ``_submit_command_prompt``.
            #
            # The admission is BOUNDED, not awaited and not fire-and-forget: a
            # reply parked on a running turn's durable append is answered 503
            # past the client's ack deadline and reads as indeterminate on retry
            # while the goal is set and the turn is queued. ``admit_receipt_request``
            # carries the reasoning, the measurement behind the bound, the steer
            # choice that matches both other hosts, and the three dispositions
            # ``admission.status`` can now report.
            if desktop_viewer_must_submit(outcome.data.get("type")) and consumed:
                admission = await admit_receipt_request(
                    bridge,
                    str(consumed),
                    command=spec.name,
                    command_id=body.request_id,
                    images=[image.model_dump() for image in body.images],
                )
                result["admission"] = {
                    "status": admission.status,
                    "detail": admission.detail,
                    "duplicate": admission.duplicate,
                }
            return {"command": spec.name, "result": result}

        return reply(
            await receipts(request).run(
                session_id + ":" + body.request_id, body.model_dump(), execute
            )
        )


async def decode_images(images: list[Image]):
    """Wire images to bounded ImageContent blocks, off the event loop.

    The shared helper resizes and re-encodes each image, which is CPU-bound;
    this runs inside a request handler, so it takes the threaded form.
    """
    from local_operator.session.runtime.server import image_blocks_in_thread

    return await image_blocks_in_thread([image.model_dump() for image in images])


@router.post(
    "/v1/desktop/sessions/{session_id}/answers", response_model=CRUDResponse[AnswerReceipt]
)
async def answer(session_id: str, body: Answer, request: Request):
    """Answer the pending gate, or refuse because this process is leaving.

    Refused by the pool's door (``DesktopSessions.session``) like the two routes
    above, and the reason is NOT that this path can start a runtime — it cannot:
    ``answer_gate`` needs a connected client and raises otherwise, so a cold
    daemon cannot be warmed into a spawn from here. The reason is that a latched
    daemon is a process whose socket is about to close, and an answer delivered
    through it is a delivery nobody can confirm; the client's correct move is the
    same one every other refusal asks for (rediscover the successor through the
    record) rather than the 409 its own "no longer pending" check would produce,
    which reads like the question expired.

    BEFORE the epoch comparison, deliberately — and the door runs before this
    handler's first statement: a stale-epoch answer on a latched daemon must not
    get a refusal that suggests retrying against this process.
    """
    async with errors(request), host(request).session(session_id) as bridge:
        assert bridge.remote is not None
        if body.epoch != bridge.remote.frontend_state.epoch:
            raise HTTPException(409, "This answer belongs to an earlier session owner")
        try:
            detail = await bridge.remote.answer_gate(
                body.request_id,
                value=body.value,
                approved=body.approved,
                question_index=body.question_index,
            )
        except OperatorAuthorityRequired as error:
            # BEFORE the ``RuntimeError`` arm below, which would otherwise
            # swallow this as "no longer pending" — the opposite of the truth:
            # the card is STILL PARKED, waiting for a console that may approve
            # it, and this caller is not one (agent review round 1 R1-2, QA Q1:
            # the route answered 409 "no longer pending" while the card was
            # still on screen, so the operator's next move was to stop looking
            # for it). ``still_pending`` says so in a field rather than only in
            # prose, and the copy names the remedies.
            raise HTTPException(
                422,
                {"code": error.code, "message": str(error), "still_pending": True},
            ) from None
        except RuntimeError:
            raise HTTPException(409, "This question or approval is no longer pending") from None
        return reply({"detail": detail})


@router.post("/v1/desktop/sessions/{session_id}/seen", response_model=CRUDResponse[AttentionState])
async def seen(session_id: str, body: Seen, request: Request):
    # Same composer as the bulk sibling: BOTH receipt routes clear read receipts
    # and neither sends a message, so the classifier's send-path nouns are false
    # about both (review round 4, M1 — this route is the shipped ``sessions.seen``
    # contract and was left on the classifier's copy).
    async with errors(request, receipts_refusal):
        return reply(await host(request).acknowledge_attention(session_id, body.completion_token))


class SeenItem(Input):
    """One completion the caller RENDERED, named by session id and token.

    Both halves are identity, not content: the id selects a session (the
    conversation identity ``session/<id>`` is DERIVED server-side, so a caller
    cannot name an identity it could not enumerate), and the token is the
    specific completion it was showing. A mark without a token is not ackable
    and must not be sent -- a timestamp or a caller's own epoch could clear a
    later, unseen result.
    """

    session_id: Annotated[str, Field(pattern=r"^[a-f0-9]{12}$")]
    completion_token: RequestID


class SeenMany(Input):
    #: 1..500, and the bound is the CATALOGUE's own maximum page
    #: (``GET /v1/desktop/sessions?limit=``, ``le=500``), so a client can always
    #: send every row it holds in one call and never has to chunk a single user
    #: gesture. The worst-case body is ~30 KB against the 900 KB control-frame
    #: limit.
    items: Annotated[list[SeenItem], Field(min_length=1, max_length=500)]


@router.post("/v1/desktop/attention/seen", response_model=CRUDResponse[dict[str, Any]])
async def seen_many(body: SeenMany, request: Request):
    """Clear the unread completion receipts a client enumerated, in ONE write.

    The bulk sibling of ``POST .../{session_id}/seen``, for the sidebar gesture
    that clears the whole pile rather than opening each conversation. Same cold
    contract, same store rule -- only the shape is additive.

    NOT A SWEEP: the body carries the completions the caller actually rendered,
    and the store compares each against the conversation's CURRENT token inside
    one write transaction, so a completion published after that render stays
    unread. A batch that clears nothing is still 200 -- the three verdict buckets
    ARE the answer, and a non-2xx would make a client throw away the partial
    result it did get -- while a per-item failure (a dead or foreign session) is
    ``unknown`` for that item rather than a 404 for the call.

    ``read`` is ``list[dict[str, Any]]`` and deliberately NOT
    ``AttentionState``: that model defaults ``supported`` to ``None``, so
    serialising through it would put ``"supported": null`` on the wire, and the
    renderer's attention merge honours only ``undefined`` as "inherit what you
    were told" -- a ``null`` would replace a known ``supported: true`` and
    silently disable its visible-read receipt. The store's own state dict has no
    such key, so the wire omits it and the merge inherits.

    The path cannot collide with ``GET /v1/desktop/sessions/{session_id}`` or
    with the per-session ``/seen`` at any registration order, hence the noun in
    the middle rather than a ``/v1/desktop/sessions/seen`` that the path
    parameter could shadow.
    """
    # This route composes its own refusal copy: a bulk read receipt has no
    # message in it and sends nothing, so the classifier's send-path sentences
    # are false about it (see ``receipts_refusal``).
    async with errors(request, receipts_refusal):
        result = await host(request).acknowledge_attention_many(
            [(item.session_id, item.completion_token) for item in body.items]
        )
        return CRUDResponse(status=200, message="Completion receipts marked read.", result=result)


@router.post(
    "/v1/desktop/sessions/{session_id}/notified", response_model=CRUDResponse[NotificationClaim]
)
async def notified(session_id: str, body: Notified, request: Request):
    """Claim the right to raise ONE banner for one completion.

    Called by the desktop app immediately before it constructs the OS
    notification, and only then: claim-then-deliver means the claimant has to
    be the deliverer, so a renderer that is going to suppress the banner
    (focused window, stale dedupe key) must not reach here. A claim taken for a
    toast nobody sees is delivered-to-nobody forever, and no other surface can
    ever pick it up.

    Cold and receipt-free, unlike ``/seen`` beside it: no bridge is acquired,
    no runtime is started, and neither ``unseen`` nor the read watermark moves.
    Notifying is not reading.
    """
    async with errors(request):
        claimed = await host(request).claim_notification(session_id, body.completion_token)
        return reply({"claimed": claimed})


@router.post("/v1/desktop/sessions/{session_id}/pin", response_model=CRUDResponse[PinState])
async def pin(session_id: str, body: Pin, request: Request):
    """Set a session's durable pin to the state the caller asked for.

    THE PIN FILE IS NOW A CROSS-SURFACE CONTRACT. It began as the sidebar's own
    index and it is now the durable record two front ends share — the TUI writes
    it with f10 and reads it on every sidebar refresh, this route writes it for
    the desktop app, and the catalogue row below reports it — so a change to its
    shape is a coordinated change between the two surfaces and the backend, not
    a private refactor of a TUI index. It stays a bare JSON array of session
    directory names for the reasons `sidebar_pins` gives; nothing here adds a
    field to it.

    DESIRED STATE, NOT A TOGGLE. The TUI's verb is a toggle because it is a
    keypress; over HTTP a toggle is not idempotent, so a request retried after a
    dropped response flips the pin BACK and the user reports "the pin keeps
    un-pinning itself". The body therefore carries the state the caller wants
    and a retry lands on the same state — re-pinning a pinned session is a no-op
    that does not even rewrite the file, which is also what keeps a retry from
    reordering the user's pins (the store is newest-pin-first).

    RECEIPT-FREE, deliberately, unlike the mutating routes around it. Receipts
    buy at-most-once for calls that ADMIT WORK (a retried send must not run a
    turn twice); this call is idempotent by construction, which is strictly
    better than putting it on the ``ReceiptConflict`` 409 ladder.

    LAST WRITER WINS across processes, accepted and documented rather than
    fixed, and the unit of arbitration is the WHOLE LIST rather than this one id:
    every write is a read-modify-write of the entire index, so two presses
    landing inside one window do not merely arbitrate over the conversation they
    share — the later ``os.replace`` is what the file holds, and anything the
    earlier writer added in that same window is gone. That can be a pin to a
    DIFFERENT conversation.

    THAT LOSS IS NOT SOMETHING THIS ROUTE INTRODUCES. The store's own docstring
    already concedes it between two ``lop`` processes — the TUI was the only
    writer, not the only possible one — and what this route changes is how often
    the window is hit: a press here beside a press in the terminal is routine in
    a way two terminal processes colliding never was, so a few-microsecond race
    stops being a curiosity. A cross-process lock for a small index has no precedent in this
    codebase, the store's own docstring records why no read-back is wanted
    either, and the only consequence a user can observe is that two presses
    within one animation resolve to the second — the correct reading of their own
    two actions. Stated rather than left to the store's comment because the
    client reconciles its row on this answer: until the next catalogue read
    agrees, a pin the app just made is not yet durable, and it never is on a
    config root the backend cannot write.

    ID SHAPE AND IS-DIR ONLY. Deliberately NOT the ``is_user_session`` check its
    neighbour ``/seen`` applies: the sidebar pins delegated runs, and a route
    that refused to unpin one would leave a pin the user can see and cannot
    remove. Unknown and malformed ids both raise ``KeyError`` into ``errors()``
    above, which answers the generic 404 — the reader cannot act on the
    difference between the two, and inventing a code for it would be a
    distinction with no remedy behind it.
    """
    async with errors(request):
        return reply(await host(request).set_pin(session_id, body.pinned))


@router.post("/v1/desktop/sessions/{session_id}/archive", response_model=CRUDResponse[ArchiveState])
async def archive(session_id: str, body: Archive, request: Request):
    """Set a session's durable archive to the state the caller asked for.

    THE PIN ROUTE'S SHAPE, field for field, and for the same reasons — read that
    docstring before changing either, because the two are one convention:
    desired state rather than a toggle (a retry of a toggle flips the archive
    back and the user reports "the archive keeps un-archiving itself"),
    receipt-free because the call is idempotent by construction, last-writer-wins
    on the whole index across processes, and ID SHAPE AND IS-DIR as the only
    admission — deliberately NOT ``is_user_session``, because the sidebar can
    show a delegated run and a state the user can see must be a state the user
    can change.

    THE 200 REPORTS THE STATE THAT WAS ASKED FOR, which on a config root this
    process cannot write is not the state the store holds: ``set_archived``
    swallows its ``OSError`` and echoes the desired value, as ``set_pin`` does.
    The pin route's discipline, restated here because a reader of THIS docstring
    should not have to know the other one to learn that the response is a
    statement about intent rather than a durability claim (review round 1, NIT).
    A client that needs the store's own answer re-reads the listing, which is
    also what settles the two-writers race this route accepts.

    WHAT IT DOES NOT DO, because the pin route's silence about it was reasoned:
    archiving NEVER removes anything, so there is no guard, no refusal and no
    409 — the worst case is a flag on a conversation that is still on disk and
    still resumable by id. It also does not make the archived flag reachable in a
    listing: every surface that offers rows filters them out through the scan
    unless its own ``include_archived`` asked for them, so a client cannot
    archive a conversation and keep seeing it in a list it did not ask to change.

    A PEER'S SESSION IS ARCHIVED ON THE PEER, through the owner's own
    implementation (``mobility.lifecycle``): the archive index is a file beside the
    session, so writing it here would archive a conversation on a device that does
    not hold it — and the row the user sees would keep its state while the owner's
    own listing said the opposite. The 200 keeps this route's contract (the state the
    caller ASKED for), because that is what the client reconciles its row on, and an
    owner's refusal carries the owner's own sentence.
    """
    async with errors(request):
        remote = await _remote_lifecycle(
            request, session_id, action="archive" if body.archived else "unarchive"
        )
        if remote is not None:
            return reply(remote)
        return reply(await host(request).set_archived(session_id, body.archived))


@router.delete("/v1/desktop/sessions/{session_id}", response_model=CRUDResponse[DeletedSession])
async def delete_session_route(session_id: str, body: ConfirmDeletion, request: Request):
    """Permanently remove ONE conversation. Irreversible, so it is guarded.

    A DELETE WITH A BODY, which is unusual enough to state: the body is not
    parameters, it is the CONFIRMATION — the request is refused without it (422,
    see :class:`ConfirmDeletion`), so a client cannot reach this route by
    accident with the right URL and the wrong method. The alternative spellings
    were considered and rejected: a query parameter is invisible to anyone
    reading a log or a URL, and a header is a convention with no precedent in
    this API.

    THREE ANSWERS:

    * **200** — ``{"session_id", "deleted": true}``. The removal happened.
    * **404** — an unknown or malformed id, including a session the user did not
      open (a delegated subagent run). One generic answer for all of them, the
      pin route's rule: separate refusals would let an authenticated renderer
      enumerate the machine's store by the difference between them.
    * **409** — a hard guard refused, carrying the sentence that names the
      condition and its remedy (a live session, an armed wake, unread spooled
      mail, or a guard that could not be read). Not 404: the conversation exists
      and the user can see it. Not 500: nothing failed.

    WHAT IT REMOVES, exactly: the addressed session's own directory and nothing
    else. Subagent runs it launched live as SIBLINGS under ``sessions/`` and are
    not touched — the client is told that in the confirmation it shows BEFORE
    this call (the TUI states it in words), because a receipt is the wrong place
    to learn the blast radius of an irreversible act.

    Consistency for the two stores that mention sessions is FREE rather than
    handled here, and deliberately so: the pin and archive indexes both PRUNE AT
    READ against the store, so an id whose directory is gone reads back as
    neither pinned nor archived, and deletion needs no cooperation from either.
    The wake index is pruned by the deletion path itself (``cleanup``), and the
    search cache is keyed by ids the caller lists, so a stale entry is never
    consulted for a conversation that no longer exists.

    THE DAEMON'S OWN RESIDENCY IS NOT FREE, so the pool drops it explicitly
    (``DesktopSessions.forget``): a conversation this process has already opened
    is served from a resident bridge without re-reading the directory, so a
    deleted one stayed reachable — measured 200 on ``sessions.get`` and
    ``/history`` — until the daemon restarted, while a FRESH daemon over the same
    store answered 404 (desktop QA round 2, PR #390). After the fix every
    session-scoped route answers 404 for the removed id, exactly as a fresh daemon
    does, and that is measured rather than argued: the snapshot, ``/history``,
    ``/mcp``, ``/report`` and ``/failovers`` reads, the ``/pin`` and ``/archive``
    desired-state writes and the ``/seen`` receipt clear were each probed against
    both daemons after a delete, and every cell agrees.
    The RECORD plane needs no cooperation and gets none, which is the half that
    was already true: the catalogue (``GET /v1/desktop/sessions``) and the search
    digest are built from a store walk, so the id is simply gone from them —
    re-read after a delete, the catalogue lists the surviving conversation and
    nothing else. What a client holds until it re-reads is its own state, not this
    daemon's.

    A PEER'S SESSION IS DELETED ON THE PEER, through the owner's own
    implementation, and the 409 ``session_delete_refused`` SHAPE IS UNCHANGED: the
    sentence is the owner's (its guards are the ones that stat the records, the wake
    index and the spool) and the code is the family's, so a client that already
    branches on it needs no change. The confirmation is forwarded as the owner's own
    ``confirmed`` flag, which is what makes a delete without one a dry run rather
    than a deletion.
    """
    async with errors(request):
        remote = await _remote_lifecycle(
            request, session_id, action="delete", confirm=bool(body.confirmed)
        )
        if remote is not None:
            return reply(remote)
        return reply(await host(request).delete(session_id))


@router.post("/v1/desktop/sessions/{session_id}/watch", response_model=CRUDResponse[WatchReceipt])
async def watch(session_id: str, body: Watch, request: Request):
    # READ: this is the renderer's presence BEAT (every 15 s), not a mutation of
    # the conversation. It must never be refused because an existing owner is
    # slow to answer — a lost beat costs one lease interval, while a 503 here
    # made the panel report a lost connection for a session that was running.
    # The visible lease this beat carries still CREATES residency (through
    # ``bridge.watch`` and its lease-warm loop); read mode bounds only the attach.
    async with errors(request), host(request).session(session_id, read=True) as bridge:
        await bridge.watch(body.subscription_id, visible=body.visible, can_notify=body.can_notify)
        return reply({"lease_seconds": 45})


@router.post("/v1/desktop/sessions/{session_id}/warm", response_model=CRUDResponse[WarmReceipt])
async def warm(session_id: str, body: Warm, request: Request):
    """Start this session's runtime without submitting anything to it.

    Closes the one gap that made the desktop app feel slower than the TUI:
    every other route here either reads or mutates, so the first message POST
    for a session paid the entire cold engage inline (1146 ms median; 12-42 ms
    once engaged). The renderer calls this on the first keystroke, so the spawn
    overlaps the time the user spends finishing their sentence.

    NOT THE ONLY WARM ANY MORE. The bridge also warms on a live VISIBLE watch
    lease, without going through this route, and keeps that intent across
    attempts: a live lease's warm is retried while it is live (see
    ``DesktopSessionBridge._lease_warm_loop``), paced by a bounded backoff after
    an attempt that failed. So a session the user is merely looking at is
    already warm when the first click arrives, and one attempted warm being
    lost is not the end of the intent. This route remains the renderer's
    explicit speculation and needs no change to keep working — it is the
    keystroke, which is a user action and is deliberately never paced.

    PRECONDITION, AND IT BINDS THE CALLER: a warm issued while nothing else
    holds the bridge is cancelled when its own request returns; the desktop UI
    satisfies this by firing the warm from the mounted, subscribed session
    panel. The bridge is reference-counted and detaching cancels an in-flight
    warm (a spawn must not outlive the facade it was started against), and this
    request is itself a user of that bridge. A caller with no subscription open
    therefore gets a 200, a ``warming`` receipt, and no warm — the send that
    follows pays the full cold engage exactly as it does today. Stated here
    because the symptom is a benchmark anomaly rather than a failure, and the
    next reader should not have to rediscover it from one.

    RECEIPT-FREE, unlike every mutating route beside it, and that is the point
    rather than an omission. Receipts buy at-most-once for calls that admit
    WORK, so a retried POST cannot run a turn twice. A warm admits nothing and
    is idempotent by construction, so a receipt would add a sqlite write per
    keystroke-debounce on the hottest new path in the app — and would put this
    route on the ``ReceiptConflict`` 409 ladder, where a speculative warm-up
    could answer a typing user with an error.

    ALWAYS 2xx FOR A WARMING FAILURE. The state at return time is genuinely
    "an engage was started"; what becomes of it is not this request's to
    report, and the send that follows reports it properly through its own
    ladder. The non-2xx answers that remain are the ones that mean the call
    itself was not admissible at all: an unknown session (404), a full
    bridge table (409), and a daemon that has LATCHED against new work (503,
    ``daemon-retiring``) — the last is the one refusal that says "this process is
    leaving", not "this call is wrong", so a client reconnects to the successor
    rather than retrying here, both from ``errors()``.

    ``body`` is declared and never read: it exists so FastAPI validates the
    request against a closed model. Dropping the parameter would make the route
    accept any JSON at all, which is the opposite of what the empty model is
    for.
    """
    del body
    async with errors(request), host(request).session(session_id) as bridge:
        assert bridge.remote is not None
        return reply({"state": await bridge.warm()})


def _work_is_running(remote: Any) -> bool:
    """Whether there is anything the interrupt rung would actually STOP.

    WHY THIS EXISTS AT ALL, and it is not an optimisation. Everything this route
    can stop is decided by the OWNER, but the answer the caller needs is a word —
    ``interrupted`` or ``idle`` — and the only structured read of the owner's
    state a follower has is its published roster. Without this, a press on a warm
    session that was merely sitting between turns ran an abort that stopped
    nothing and then reported ``interrupted``; that is exactly the class of
    overstatement the abort receipt itself was rewritten to remove, and it would
    stay invisible until something reads the field.

    The terms are the RUNG's own effects, one per thing ``abort`` does:

    * a live TURN — ``is_streaming``, the facade's own mirror of the canonical
      flag, and ``activity_phase_clock``, which is non-empty from the moment a
      turn's work begins until it ends. The second is not redundant: it covers
      the pipeline the streaming flag has not caught up with yet, which is the
      window a press would otherwise be swallowed in (see the LIMIT below);
    * a PARKED GATE — the orphan card that outlived its turn, which this rung
      settles (``_deny_pending_gates``). This term is why the predicate is not
      "is the follower streaming": that card is precisely the case the abort's
      deny-first ordering was added for, so a press that clears it DID do
      something and must not be answered ``idle``;
    * a running ``task`` JOB — a subagent, which the abort cancels;
    * a running GOAL LOOP — which the abort cancels.

    A running ``bash`` job is deliberately NOT a term. Backgrounded jobs exist to
    outlive the turn that started them (``background=true``) and this rung never
    touches them, so a session whose only live work is one has nothing for an
    interrupt to stop — the receipt names those jobs as untouched, and a caller
    that wanted them gone has the Jobs surface.

    THE FIRST THREE READS ARE CLONE-FREE ON PURPOSE. ``is_streaming`` is the
    facade's mirror, and ``pending_gate``/``activity_phase_clock`` are the store's
    own documented copy-free seams. The roster itself is NOT: ``jobs`` is a mutable
    container, so the store deliberately keeps handing its readers a deep copy for
    it (``_SHAREABLE_STATE_FIELDS``), and one press therefore pays exactly one clone
    on this path — the overwhelmingly common press (a streaming turn) pays NONE,
    because it returns on the first read.

    LIMIT, STATED RATHER THAN HIDDEN: the owner's ``_turn_lock`` flush window is
    not visible from a follower at all, so a prompt admitted but not yet started
    cannot be told from an idle session by reading canonical state. The activity
    phase narrows that window to the admission-to-first-work gap, and the residue
    is not reachable from the desktop: the button and Esc are both offered on the
    same ``streaming`` flag this reads, so a press cannot exist in a window where
    this predicate is false.

    RACE, STATED RATHER THAN HIDDEN: the follower's roster can lag the owner by a
    delta. Both directions are benign here. A stale ``False`` cannot swallow a
    press the user could make, for the reason just given. A stale ``True`` at
    worst reaches the abort a moment after the turn settled, which is the
    pre-existing behaviour of a press racing a turn's end.
    """
    if remote.is_streaming:
        return True
    if remote.pending_gate is not None:
        return True
    if remote.activity_phase_clock()[0]:
        return True
    state = remote.frontend_state
    # Bare attributes rather than getattr probes: the store hands out real
    # ``JobState`` rows (``_public_job`` detaches every one), so a probe would hide
    # a rename behind a silent ``""`` instead of failing, which is the opposite of
    # what this file's sibling reads want.
    if any(row.type == "task" and row.status == "running" for row in state.jobs):
        return True
    loop = state.loop or {}
    return loop.get("status") in {"running", "judging"}


def _running_work_counts(state: Any) -> tuple[int, int]:
    """(live subagents, live backgrounded ``bash`` jobs) from a published roster.

    Split by job TYPE because the two surviving kinds have different remaining
    levers and the notice a surface writes names one of them: a ``task`` row is
    a subagent, which the interrupt DID reach (so any row still running here
    refused to die), while a ``bash`` row was deliberately never touched and
    needs the Jobs surface. Collapsing them into one number would force the
    copy to say "3 things" about two situations with two different answers.

    Takes an already-read roster rather than the facade so one press reads the
    follower's canonical state once per question — before the press for "was
    there work", after it for "what survived" — instead of cloning it twice for
    one of them.
    """
    jobs = state.jobs
    running = [row for row in jobs if row.status == "running"]
    children = sum(1 for row in running if row.type == "task")
    background = sum(1 for row in running if row.type == "bash")
    return children, background


@router.post(
    "/v1/desktop/sessions/{session_id}/interrupt",
    response_model=CRUDResponse[InterruptReceipt],
)
async def interrupt(session_id: str, body: Interrupt, request: Request):
    """Stop this session's CURRENT WORK, and leave the session running.

    THE OP THE DESKTOP'S STOP BUTTON AND ESC MEAN, and the bug it fixes is
    that they meant nothing. The renderer posted ``sessions.command`` with
    ``command: "stop"``, which is not an ``OWNER_COMMAND``, so this API
    answered an ``native_action`` PRESENTATION for ``POST /v1/desktop/stop``
    and stopped no turn at all: the transport was fine and the button called
    the wrong op. Pointing it at ``/stop`` instead would have been worse than
    the bug — that route is the KILL SWITCH (deny gates, dispose, release the
    writer lease, unpublish, exit the runtime), and a control that promises
    "stop this session's current work" must not end the session.

    THE RUNG IT REUSES IS THE PHONE RELAY'S. ``abort`` already means exactly
    this on the control socket — stop the turn, cancel the children it
    started, leave the session and its process alive, and report honestly on
    what settled — so this route adds a way to REACH it from HTTP, not a
    second implementation of it. The mapping is one line in
    ``AttachedSession.interrupt`` (this route) to ``abort`` (the runtime op),
    and the name deliberately avoids ``/abort``: on this surface ``stop``
    already means "end the process", and a route one letter from it is a trap
    for the next reader.

    NO LADDER. A first press stops the turn and its children; a second press
    is simply a second interrupt, a no-op because nothing is left running. The
    keyboard's Esc ladder can afford a narrow first press because a second one
    is offered on screen (``DOUBLE_STOP_WINDOW_S``); this surface has no such
    offer and nothing rendering "press again", so a ladder here would be a
    press that does nothing once and explains itself nowhere. Backgrounded
    ``bash`` jobs are never touched — ``background=true`` exists so a build
    outlives the turn that started it — and the receipt names them.

    ``idle`` IS A SUCCESS, AND IT IS THE ANSWER FOR ANY SESSION WITH NOTHING TO
    STOP. A cold session is NOT engaged to answer this (an interrupt is not a
    reason to spend a process, which is what ``warm`` is for), and a warm one
    that is merely sitting between turns is answered without dialling its owner
    at all — see ``_work_is_running`` for the terms, and note that a parked gate
    WITHOUT a live turn (the orphan card this release also taught ``abort`` to
    settle) counts as work, because that press really does clear the screen. A
    client putting an error in front of a press that had nothing to do would be
    reporting the user's own success as a failure, and a client told
    ``interrupted`` for a press that stopped nothing would be shown a success
    that did not happen.

    "NO OWNER" MEANS ``owner_reachable``, NOT ``is_cold``, and the difference is
    the whole of review round 1's MAJOR-1: ``is_cold``'s third disjunct is a
    RESYNC state which is true of a connected, SERVING session for the duration
    of a frontend sync plus a history page load, so gating on it answered
    ``idle`` for a live streaming turn and stopped nothing — this PR's own defect
    class, arriving through its own new door.

    RECEIPTED ``retry_safe=True``, and both halves are deliberate. Receipted,
    because a retry after a lost response must not fire a second interrupt at a
    turn that has since moved on — the journal replays the stored answer
    verbatim. ``retry_safe=True``, because unlike ``/stop`` and ``/move`` the
    operation is idempotent: "make the current turn stop" creates and destroys
    nothing, and a pending row that never ran is re-executed to the same end.
    That is also why this route does NOT need ``/stop``'s ``assert_admitting``
    call before the claim (``desktop_lifecycle.stop``): its receipt is
    ``retry_safe=False``, so a claimed-but-unrun row is INDETERMINATE for the
    client, whereas here a retry is the remedy rather than a hazard.

    THE JOURNAL'S 409 ARM CANNOT FIRE HERE, and that is a property of the body
    rather than an omission (review round 1, MINOR-2). A receipt's fingerprint is
    a pure function of its request body, and this body has exactly one field, so
    the same ``request_id`` can only ever arrive with the same fingerprint —
    which replays — or with a body the closed model refuses, which is a 422
    before the journal is reached. ``extra="forbid"`` is what makes the second
    case a shape error rather than a silently-ignored extra, so nothing is lost
    by naming it plainly here instead of documenting a 409 a caller could never
    provoke. The journal's own rule is unchanged and tested where it lives
    (``test_receipts_survive_adapter_restart_and_reject_changed_body``).

    STATUS CODES are the shared ladder's, with one shape to state because it
    looks like a bug: an UNKNOWN session id and a MALFORMED one are both 404.
    The id validator raises ``KeyError`` for a bad shape, and ``errors()``
    answers that as "no such session" — deliberately not a 422, which is
    reserved for the BODY's shape (a non-UUID ``request_id``, or any extra
    field). 401 missing or wrong bearer, 403 a disallowed or browser-originated
    Origin, 503 a desktop capability that is not configured or an owner that
    cannot be reached (``ConnectionError``/``RuntimeError``/``TimeoutError``).
    """
    async with errors(request), host(request).session(session_id) as bridge:

        async def execute():
            assert bridge.remote is not None
            # NO REACHABLE OWNER is a no-op, NOT a runtime spawn — and the term is
            # REACHABILITY, deliberately not ``is_cold``. That property's third
            # disjunct is ``not _ready_for_events``, a RESYNC state which is true of
            # a connected, SERVING session for the whole of a frontend sync plus a
            # history page load: gating on it read a live streaming turn as `idle`
            # and stopped nothing, which is this PR's own defect class arriving
            # through its own new door (review round 1, MAJOR-1).
            if not bridge.remote.owner_reachable:
                return {
                    "status": "idle",
                    "receipt": "",
                    "children_running": 0,
                    "background_jobs": 0,
                }
            # NOTHING FOR THIS RUNG TO STOP IS THE SAME ANSWER as no owner to stop
            # it with: ``idle``, on a 200, without dialling. That question is asked
            # of the follower's published roster, which stays readable through a
            # resync — the store is installed from the attach snapshot and
            # maintained by deltas, so a mid-refresh viewer still knows whether
            # work is running.
            if not _work_is_running(bridge.remote):
                # Nothing was stopped, so "what survived" is simply what is
                # running. ``children_running`` is zero by construction (a
                # running ``task`` job is a term above), while backgrounded
                # ``bash`` jobs are reported TRUTHFULLY rather than zeroed: the
                # press did not touch them, and a field that said "no jobs"
                # beside a build still going would be a lie told on a success.
                return {
                    "status": "idle",
                    "receipt": "",
                    "children_running": 0,
                    "background_jobs": _running_work_counts(bridge.remote.frontend_state)[1],
                }
            receipt = await bridge.remote.interrupt()
            # Read AGAIN, after the press: "what survived" is a different question
            # from "was there work", and the children settle in between.
            children_running, background_jobs = _running_work_counts(bridge.remote.frontend_state)
            return {
                "status": "interrupted",
                "receipt": receipt,
                "children_running": children_running,
                "background_jobs": background_jobs,
            }

        return reply(
            await receipts(request).run(
                session_id + ":interrupt:" + body.request_id,
                body.model_dump(),
                execute,
                retry_safe=True,
            )
        )


@router.post(
    "/v1/desktop/sessions/{session_id}/working-directory",
    response_model=CRUDResponse[MoveReceipt],
)
async def move(session_id: str, body: MoveSession, request: Request):
    """Point a live session at ``body.cwd``, rebuilding its runtime if it has one.

    THE DESKTOP HALF OF THE TUI'S ``/move``, and the same operation: one
    implementation behind both surfaces (``move_session``), so "what does moving
    a session mean" has one answer. The desktop cannot reuse the TUI's route to
    it, because the TUI's is a key handler in the process that owns the session.

    WHY A DEDICATED ROUTE RATHER THAN ``/commands``. Three reasons, in order of
    weight. (1) The argument form cannot be answered through ``CommandReceipt``:
    its ``result`` is ``NativeAction | OwnerCommandResult``, a move produces
    neither, and adding a third member changes the renderer's shared
    ``SlashOutcome`` union and every consumer of it. (2) ``/commands`` executes
    slash commands and a move is not one — the runtime cannot respawn itself at
    a new cwd (its directory is baked in at spawn) and the state to change lives
    on the VIEWER, so routing it through a runtime slash handler would answer the
    user with the runtime's own "/move reads this machine's configuration…"
    notice: a false statement about ``move``, on a 200. (3) ``/warm`` is the
    structural precedent — both are lifecycle operations on this session's
    runtime, both have their own route, op and receipt. A move is a warm that
    goes somewhere else.

    ``/move`` ITSELF STAYS AS IT IS. It keeps its ``session.move`` destination in
    the command catalogue and stays out of ``OWNER_COMMANDS``: a bare ``/move``
    still answers a ``native_action`` (a request for PRESENTATION, claiming
    nothing ran) and the renderer opens its picker; a typed ``/move <path>`` is
    executed by the renderer calling THIS route. Nothing in the command path
    changes, so an older renderer against this backend still presents ``/move``
    correctly, and this backend against an older renderer is simply a route
    nobody calls.

    REACHABLE ONLY WITH ``features.session_move``, which is why that key is its
    own rather than a bump: a renderer that does not see it keeps its read-only
    working-directory chip and reports the degradation, exactly as it does
    today, instead of firing a request an older backend answers with a 404.
    That key is ``2`` from the exclusivity fence on, and a renderer additionally
    needs ``features.frontend_replace`` before it may offer the control: a move
    now refuses while another actual attach is registered (review R3) and while
    a mounted viewer cannot render the ``frontend.replace`` frame (review R4), so
    a renderer built for the unconditional ``1`` must not promise the old
    behaviour. Both refusals are 409s carrying the sentence to act on.

    WHAT THIS RESPONSE DOES NOT CLAIM: that the successor is up. The runtime
    leaves by the ``retiring`` route, and the successor is engaged by the retire
    frame on the BRIDGE (see ``DesktopSessionBridge._on_runtime_retired``), not
    by this request. The receipt says where the session now works and what
    happened to the old runtime; the renderer's chip settles when the successor
    binds and publishes its frontend state.

    RECEIPTED, unlike ``/warm`` beside it, because this DOES mutate: it writes
    the durable marker, may retire a runtime and may spend a spawn. The journal is
    AT MOST ONCE (``retry_safe=False``, review R1), and the reason is that a
    re-run is NOT provably a no-op: ``move_session`` answers ``unchanged`` only
    while the session is still in the directory the first attempt started from,
    and a relative target resolved after that attempt is resolved against the NEW
    directory (``child`` becomes ``child/child``) while an absolute one can undo a
    move that landed after it. A pending row is therefore INDETERMINATE — it
    answers the journal's own reconcile-before-retrying refusal, and a client
    that still wants the move issues a NEW request id. A row that finished
    replays its stored receipt verbatim, so a lost response is still recoverable;
    what no longer happens is raw re-execution.

    ``RuntimeError`` IS MAPPED TO 409 HERE, and that is the load-bearing line of
    this route. It is the SESSION's own refusal — "working right now", "too old
    to be moved", "could not move: <reason>" — and these are states of a HEALTHY
    session. ``errors()``'s ladder answers ``RuntimeError`` with its 503 sentence
    ("Session owner is unavailable. Reconnect and reconcile before retrying."),
    which would tell a user mid-turn that the backend is unreachable. The
    session's sentence is what the user needs, and 409 is the status the rest of
    this API uses for "your request is understood and refused".

    A REJECTED TARGET's own ``ValueError`` sentence (absent, not a directory,
    unenterable) is deliberately NOT caught here: the ladder already answers it
    with a 409 carrying the vetter's text, and catching it would be a second copy
    of that mapping to keep in step.

    ``MoveIndeterminate`` IS NOT A ``RuntimeError`` AND MUST NOT BECOME ONE. It
    is the other half of the split above (contract §A): the retire REQUEST
    reached the owner and no definitive answer came back, so the owner may
    already have accepted the new directory. ``errors()`` answers it with 503,
    which is the ladder's own "reconcile before retrying" answer, and nothing is
    rolled back — restoring the old marker there would overwrite a committed move
    with a stale one.
    """
    async with errors(request), host(request).session(session_id) as bridge:

        async def execute():
            try:
                return (await move_session(bridge, body.cwd)).model_dump(mode="json")
            except RuntimeError as error:
                raise HTTPException(409, str(error)) from None

        # ONE OWNED TASK covering receipt claim → serialized move → receipt
        # finish, awaited under a shield-and-JOIN loop (review R2 / QA Q2).
        #
        # WHY NOT A DIRECT AWAIT. Cancelling this coroutine — the HTTP waiter
        # going away, a server shutdown — must not cancel the transaction: the
        # marker writer and the retire are already in flight, and a task
        # cancelled out of that releases ``move_lock`` while its filesystem work
        # is still running, so a later writer can overwrite a committed marker
        # with the cancelled request's directory. ``asyncio.shield`` alone is
        # not enough either, because its own await is still cancellable — hence
        # the loop in :meth:`_join_owned`, which rejoins after repeated
        # cancellation and re-raises the operation's real exception.
        operation = asyncio.create_task(
            receipts(request).run(
                session_id + ":" + body.request_id,
                body.model_dump(),
                execute,
                # AT MOST ONCE (review R1). A PENDING row is INDETERMINATE, not
                # retryable: re-executing it would resolve a relative target
                # against the directory the first attempt may already have moved
                # to (`child/child`), and an absolute one could undo a later
                # accepted move. A finished row still replays its stored receipt
                # exactly, and reusing the id with different input still
                # conflicts — only the raw re-execution goes away.
                retry_safe=False,
            )
        )
        return reply(await _join_owned(operation))


@router.get("/v1/desktop/sessions/{session_id}/events")
async def events(
    session_id: str,
    request: Request,
    epoch: str | None = Query(default=None, max_length=128),
    after_seq: int = Query(default=0, ge=0),
    frontend_replace: int = Query(default=0, ge=0),
):
    # Acquire BEFORE returning response headers: invalid identity/capacity must
    # return JSON status, not a misleading 200 followed by a broken SSE stream.
    context = host(request).session(session_id, read=True)
    async with errors(request):
        bridge: DesktopSessionBridge = await context.__aenter__()
        try:
            # ADDITIVE NEGOTIATION (contract §C). ``frontend_replace=1`` says
            # this renderer can consume the desktop-only ``frontend.replace``
            # frame; an older client sends nothing, the parameter defaults to 0,
            # and the move path then refuses rather than leaving that mounted
            # viewer stale. The flag is retained on the subscription so the
            # fence in ``subscribe`` can refuse a LEGACY mount during a move.
            sub = bridge.subscribe(frontend_replace=bool(frontend_replace))
        except LegacySubscriberDuringMove as error:
            await context.__aexit__(None, None, None)
            raise HTTPException(409, str(error)) from None
        except BaseException:
            await context.__aexit__(None, None, None)
            raise

    # The bridge is acquired ABOVE, before any response exists, so that an
    # invalid session or a full subscriber table is a JSON error rather than a
    # 200 followed by a broken stream. That leaves the release owed by
    # something other than the generator: if the generator is never consumed --
    # the client disconnects between headers and body, or the response is
    # discarded before iteration -- its `finally` never runs and the bridge
    # stays acquired for the process's lifetime, holding a session attached.
    #
    # Released exactly once, from whichever path gets there first: the
    # generator's own teardown for a stream that ran, and the response's
    # background task for one that never did.
    released = False

    async def release_once() -> None:
        nonlocal released
        if released:
            return
        released = True
        await context.__aexit__(None, None, None)

    async def stream():
        try:
            async for frame in bridge.events(sub, epoch=epoch, after_seq=after_seq):
                yield "data: " + json.dumps(frame, separators=(",", ":")) + "\n\n"
        finally:
            await release_once()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
        background=BackgroundTask(release_once),
    )


def feed(request: Request) -> DesktopFeed:
    """The process's ONE desktop feed, created lazily beside the session pool.

    A process singleton rather than per-request state for the same reason the
    daemon registry is: the whole point of this channel is that a completion in
    a session nobody has open is still announced, and a per-connection poller
    would make the cost of listening scale with the number of listeners. It also
    means the delivery lease has exactly one owner to attribute it to.
    """
    value = getattr(request.app.state, "desktop_feed", None)
    if value is None:
        pool = host(request)
        value = DesktopFeed(
            request.app.state.config_manager.config_dir,
            # Read-only and by reference: the feed needs to know which sessions
            # already have a stream so it never races one, and it must never
            # ACQUIRE anything of its own — see the module docstring. The hook
            # returns the FEED's key domain (``session/<id>``) and only for
            # bridges that will actually announce — see
            # ``DesktopSessions.bridged_notify_sessions``.
            bridged=pool.bridged_notify_sessions,
        )
        request.app.state.desktop_feed = value
    return value


@router.get("/v1/desktop/events")
async def desktop_events(request: Request):
    """The machine-wide event feed: attention, catalogue and notifications.

    NO BRIDGE IS ACQUIRED AND NO RUNTIME IS SPAWNED. That is the property this
    route exists for — watching every session on the machine must not attach
    every session on the machine — and it is asserted directly in
    ``tests/unit/server/test_desktop_feed.py``.

    Unlike the per-session stream there is nothing to release eagerly before
    headers either: the subscriber table has its own ceiling and an unknown
    session cannot be asked for. The release is still wired to BOTH the
    generator's teardown and the response's background task, because a client
    that disconnects between headers and body would otherwise leak a subscriber
    and keep the process's poller alive for its lifetime.
    """
    engine = feed(request)
    if len(engine.subscribers) >= SUBSCRIBER_COUNT:
        raise HTTPException(503, "Too many desktop feed subscribers")
    subscription = engine.subscribe()
    released = False

    async def release_once() -> None:
        nonlocal released
        if released:
            return
        released = True
        engine.unsubscribe(subscription)

    async def stream():
        try:
            async for frame in engine.events(subscription):
                yield "data: " + json.dumps(frame, separators=(",", ":")) + "\n\n"
        finally:
            await release_once()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
        background=BackgroundTask(release_once),
    )


@router.post("/v1/desktop/presence", response_model=CRUDResponse[PresenceReceipt])
async def desktop_presence(
    body: PresenceBeat,
    request: Request,
):
    """Record "a desktop app is here and can attempt a banner", and where.

    A ROUTE RATHER THAN A FILE WRITTEN BY THE APP. The app may be paired to a
    backend on another host, so it cannot write to this machine's filesystem;
    the server aggregates what its live subscriptions report and materialises
    that where every sibling process can read it. Local and remote apps then
    behave identically, which is the whole reason the presence is server-side.

    The lease is held against the SUBSCRIPTION, so an id the server does not
    know is refused rather than believed: a claim to deliver for a socket that
    does not exist is exactly the "presence that cannot deliver" this mechanism
    must not manufacture.
    """
    engine = feed(request)
    if body.subscription_id not in engine.subscribers:
        raise HTTPException(404, "Unknown desktop feed subscription")
    engine.presence.update(
        body.subscription_id,
        can_notify=bool(body.can_notify),
        can_notify_kinds=list(body.can_notify_kinds),
        session_id=body.session_id,
        window=body.window.model_dump(),
    )
    return reply(PresenceReceipt(lease_seconds=int(PRESENCE_TTL_S)))

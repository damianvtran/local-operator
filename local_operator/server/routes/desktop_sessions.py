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
from typing import Annotated, Any, Literal, NamedTuple

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
from local_operator.server.models.desktop_sessions import (
    AdmissionStatus,
    AnswerReceipt,
    AttentionState,
    ChildTranscriptPage,
    CommandReceipt,
    CreatedSession,
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
    CHILD_PAGE_LIMIT,
    SUBSCRIBER_COUNT,
    DesktopSessionBridge,
    DesktopSessions,
    LegacySubscriberDuringMove,
    SubagentChildUnavailable,
    move_session,
    resolve_working_directory,
)
from local_operator.session.attention import SupersededCompletionToken
from local_operator.session.cold_model import synthesise_cold_state
from local_operator.session.errors import MoveIndeterminate, SessionStoreUnavailable
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
FAILED_ADMISSION_STATUS: AdmissionStatus = "failed"

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
ADMISSION_FAILED_FRAME = "admission.failed"


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
            {
                "request_id": command_id,
                "command": command,
                "status": FAILED_ADMISSION_STATUS,
                "detail": detail,
            },
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
    from local_operator.providers.registry import get_provider_definition

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


@asynccontextmanager
async def errors() -> AsyncIterator[None]:
    try:
        yield
    except DaemonRetiring as error:
        # The daemon has announced its retirement and refuses to start work it
        # would not finish. 503, not 500: the process is alive and deliberately
        # not admitting, so the client's move is to rediscover the successor
        # through the record (design §7) — a message and a code it can key on,
        # never a traceback.
        raise HTTPException(503, {"code": error.code, "message": str(error)}) from None
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
        )

        if isinstance(error, (AttachmentUnavailable, ProfileRegistryUnavailable)):
            raise HTTPException(409, {"code": error.code, "message": str(error)}) from None
        if isinstance(error, SupersededCompletionToken):
            # Stale, not broken: the caller's token is real but no longer current,
            # and the remedy is to re-read the conversation's attention state and
            # acknowledge the token it names. The machine code is what lets the
            # renderer take that path quietly instead of backing off as if the
            # store had refused (the `code` field of its control error).
            raise HTTPException(409, {"code": error.code, "message": str(error)}) from None
        raise HTTPException(409, str(error)) from None
    except sqlite3.Error:
        # Contention on the shared receipt store is transient and retryable, so
        # it gets a vetted sentence rather than a bare 500 carrying SQLite's own
        # wording. The text is NOT echoed for the same reason the ConnectionError
        # ladder below refuses to echo: a store error can name file paths.
        raise HTTPException(
            503, "Read state is busy right now. It will catch up on its own."
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
        raise HTTPException(
            503,
            detail or "Session owner is unavailable. Reconnect and reconcile before retrying.",
        ) from None
    except (RuntimeError, asyncio.TimeoutError):
        raise HTTPException(
            503, "Session owner is unavailable. Reconnect and reconcile before retrying."
        ) from None


@router.get("/v1/desktop/sessions", response_model=CRUDResponse[SessionList])
async def list_sessions(request: Request, limit: int = Query(default=100, ge=1, le=500)):
    # Wrapped like its neighbours: the list gained a receipt-store read, and an
    # unmapped failure there answered the app's primary navigation surface with
    # a bare 500. The decoration is already omitted per row inside `list()`;
    # this ladder covers anything else the pool can raise — including the store
    # it could not walk, which is now a typed 503 rather than an empty 200.
    async with errors():
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
        rows = await host(request).list(limit + 1, status_stamps=stamps)
        sessions = rows[:limit]
        # The sources that could not be read for THIS page. Lifted from the rows
        # rather than plumbed beside them: every row of a poll carries the same
        # verdict (one registry scan answers for the whole listing), so the
        # listing-level statement is derivable, and a second channel through
        # `list()` would be one more thing a caller can forget to pass. Sorted
        # so the set is stable across polls, and computed over what is actually
        # sent — a degraded row beyond the page says nothing about this answer.
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
                "truncated": len(rows) > limit,
                "limit": limit,
                "degraded": degraded,
            }
        )


@router.get("/v1/desktop/sessions/search", response_model=CRUDResponse[SessionSearch])
async def search_sessions(
    request: Request,
    q: str = Query(default="", max_length=256),
    limit: int = Query(default=100, ge=1, le=500),
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
    """
    async with errors():
        return reply(
            {
                "sessions": await host(request).search(q, limit),
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

    async with errors():
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
            await asyncio.to_thread(resolve_working_directory, body.cwd)
            if body.model is not None:
                # Off the loop: the catalogue it reads is a disk document, and for
                # an unshipped model the metadata resolver may consult the provider.
                spec = await asyncio.to_thread(_draft_model_spec, body.model)
            if body.target is not None:
                target_row = body.target.model_dump()
                from local_operator.agents import AgentRegistry
                from local_operator.server.utils.desktop_profiles import validate_target
                from local_operator.teams import TeamRegistry

                await asyncio.to_thread(
                    validate_target,
                    AgentRegistry(pool.root),
                    TeamRegistry(pool.root),
                    target_row["kind"],
                    target_row["name"],
                )
        return reply(await receipts(request).run(key, body.model_dump(), create))


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

    async with errors():
        return reply(await preview())


@router.get("/v1/desktop/sessions/{session_id}", response_model=CRUDResponse[SessionSnapshot])
async def snapshot(session_id: str, request: Request):
    async with errors(), host(request).session(session_id) as bridge:
        return reply(await bridge.snapshot())


@router.get("/v1/desktop/sessions/{session_id}/history", response_model=CRUDResponse[HistoryPage])
async def history(
    session_id: str,
    request: Request,
    before_id: str | None = Query(default=None, max_length=128),
    limit: int = Query(default=100, ge=1, le=500),
):
    async with errors(), host(request).session(session_id) as bridge:
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
    async with errors():
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
    async with errors():
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
    async with errors():
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
    async with errors(), host(request).session(session_id) as bridge:

        async def admit():
            assert bridge.remote is not None
            detail, duplicate = await bridge.remote.admit_prompt(
                body.text,
                command_id=body.request_id,
                images=[image.model_dump() for image in body.images],
                steer=body.mode == "steer",
            )
            # Admission can bind a cold viewer while an event subscription is
            # already open. Apply only its still-live lease, never resurrect one.
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
        # The ONE command whose trailing text the desktop never consumes: the
        # secret is entered in the masked form (`argument_shape` is NONE), so any
        # text here is prose the caller sent to the wrong route. Left as its own
        # check because the sentence is about the FORM, not about a shape the
        # admission rule reads.
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
    async with errors(), host(request).session(session_id) as bridge:

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
    async with errors(), host(request).session(session_id) as bridge:
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
        except RuntimeError:
            raise HTTPException(409, "This question or approval is no longer pending") from None
        return reply({"detail": detail})


@router.post("/v1/desktop/sessions/{session_id}/seen", response_model=CRUDResponse[AttentionState])
async def seen(session_id: str, body: Seen, request: Request):
    async with errors():
        return reply(await host(request).acknowledge_attention(session_id, body.completion_token))


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
    async with errors():
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
    DIFFERENT conversation, which is the case this route is what makes reachable:
    before it there was one writer surface (the TUI), and now a TUI and an app
    write the same file at once. A cross-process lock for a small index has no
    precedent in this codebase, the store's own docstring records why, and the
    only consequence a user can observe is that two presses within one animation
    resolve to the second — which is the correct reading of their own two
    actions. Stated rather than left to the store's comment because the client
    reconciles its row on this answer: until the next catalogue read agrees, a
    pin the app just made is not yet durable, and it never is on a config root
    the backend cannot write.

    ID SHAPE AND IS-DIR ONLY. Deliberately NOT the ``is_user_session`` check its
    neighbour ``/seen`` applies: the sidebar pins delegated runs, and a route
    that refused to unpin one would leave a pin the user can see and cannot
    remove. Unknown and malformed ids both raise ``KeyError`` into ``errors()``
    above, which answers the generic 404 — the reader cannot act on the
    difference between the two, and inventing a code for it would be a
    distinction with no remedy behind it.
    """
    async with errors():
        return reply(await host(request).set_pin(session_id, body.pinned))


@router.post("/v1/desktop/sessions/{session_id}/watch", response_model=CRUDResponse[WatchReceipt])
async def watch(session_id: str, body: Watch, request: Request):
    async with errors(), host(request).session(session_id) as bridge:
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
    async with errors(), host(request).session(session_id) as bridge:
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
    async with errors(), host(request).session(session_id) as bridge:

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
    async with errors(), host(request).session(session_id) as bridge:

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
    context = host(request).session(session_id)
    async with errors():
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

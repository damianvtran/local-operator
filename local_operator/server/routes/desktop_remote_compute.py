"""The desktop's Radient remote-compute surface (``features.remote_compute``).

WHAT THIS IS. The device dropdown's Radient group and the runs panel need five
answers: is the preview on, which Radient account is it for, which of this
device's networks a run could be bound to, can this deployment provision anything
right now, and which runs exist. Plus the three writes a panel needs — submit,
cancel, delete. Every one of them is a THIN HTTP SKIN over
:mod:`local_operator.remote_compute` (``api.RemoteRuns``, ``records``, ``repo``,
``resolve``, ``errors``): the library owns the sequence, the local record, the
repository read and the typed refusals, and this module owns authentication, the
JSON shapes and the HTTP statuses.

NOT A ``PeerRow``, AND THE READ IS NOT THE MESH READ. A peer is a device with a
handshake, presence and sessions; Radient is capacity that is rented, and filing
it under a device that does not exist would make every one of those fields a lie
(design of record §2.1). So it is its own read, in its own module, and the only
thing it shares with ``/v1/desktop/networks`` is the network store: the
``networks`` field exists because the run's tenant must be visible at the moment
it is chosen.

EVERY REFUSAL IS ``{"error": {"code", "message", "retry_after"?}}`` and the HTTP
status mirrors the server's. The body is deliberately NOT the CRUD envelope and
NOT ``detail.<envelope>``: this surface's client keys on ``error.code`` and
paints ``error.message``, and the status is what tells a retryable refusal (429
with ``Retry-After``, 503) from a terminal one (402, 403, 409). ``code`` is the
library's code — the server's refusal code when the answer carried one, a mesh
refusal code, or this surface's own name for a library-internal refusal.

WHAT IS NEVER RETURNED. No bearer, no API key, and none of the create wrapper's
secrets: ``without_secrets`` (the same filter the CLI's ``--remote-status`` uses)
drops ``run_token`` and ``tunnel`` from every run body this module answers with.
The client binds neither, so there is nothing here to remember to redact; the
filter is defence in depth.

THE FLAG IS READ PER REQUEST, from the app's config root. ``is_enabled`` reads
``config.yml`` on every call precisely so a live toggle reaches a running daemon.
WITH THE FLAG OFF, NOTHING ABOUT THE USER'S WORK LEAVES THE DEVICE — for the two
reads that would ask Radient on their own initiative: the summary answers
``{"enabled": false}`` and the runs listing answers ``{"runs": [], "server":
"disabled"}``, both with no upstream call.

THE WRITES, AND THE PER-RUN READ, STAY REACHABLE WITH THE FLAG OFF, on purpose
and for the reason ``remote_compute/cli.py`` documents for ``--remote-cancel``:
the switch stops this device from STARTING work, and a run that is already
running is still spending the operator's money. Turning the feature off must not
take away the two controls that stop it, so ``cancel`` and ``delete`` keep
answering; the per-run read keeps answering too, because it is how the panel
paints a row it already shows and how a deep link resolves. ``create`` is
refused locally (403 ``not_configured``), before any request is composed.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from local_operator.config import ConfigManager
from local_operator.i18n.keys import wire_remote_compute as _wr
from local_operator.i18n.text import tr
from local_operator.network.types import MeshRefusal
from local_operator.remote_compute import is_enabled, records, resolve_network, results
from local_operator.remote_compute.api import (
    RemoteRuns,
    RunSubmitMetadata,
    build_spec,
    repository_for,
)
from local_operator.remote_compute.client import (
    LOGS_PAGE_DEFAULT,
    RemoteComputeClient,
    check_idempotency_key,
    is_terminal,
    new_idempotency_key,
    without_secrets,
)
from local_operator.remote_compute.defaults import run_defaults
from local_operator.remote_compute.errors import (
    ATTESTATION_CODE_PREFIX,
    REFUSAL_CODES,
    BundleApplyFailed,
    CapacityUnavailable,
    Conflict,
    IdempotencyConflict,
    InsufficientCredit,
    InvalidApiOverride,
    InvalidIdempotencyKey,
    InvalidRequest,
    NetworkMismatch,
    PreviewNotEnabled,
    RemoteComputeError,
    ResultsUnavailable,
    ResultsUnreachable,
    RunBusy,
    RunIdentityMismatch,
    RunLimitReached,
    RunNotFound,
    SizeMismatch,
    Unreachable,
    UnreadableEnvelope,
    UnsafeArchive,
    UnusableRepo,
)
from local_operator.remote_compute.messages import refusal_message
from local_operator.remote_compute.request import RunCreateRequest
from local_operator.remote_compute.resolve import is_active_member
from local_operator.server.dependencies import get_config_manager
from local_operator.server.desktop import require_desktop
from local_operator.server.routes import desktop_radient
from local_operator.server.routes.auth import get_desktop_auth
from local_operator.server.routes.desktop_radient import radient_bearer
from local_operator.server.utils.desktop_auth import DesktopAuth
from local_operator.tunnels.errors import LoginRequired

logger = logging.getLogger("local_operator.server.routes.desktop_remote_compute")

router = APIRouter(tags=["Desktop remote compute"], dependencies=[Depends(require_desktop)])

_t = tr("wire.remote_compute")
#: The sentence for a tier this build cannot start.
#:
#: ROUTE-OWNED ENGLISH PROSE, which is this plane's convention for a reason:
#: ``desktop_radient`` carries its refusals the same way (``NO_CREDENTIAL_MESSAGE``),
#: because a keyed message must be translated AND audited in all seven carried
#: locales before it may ship, and one defensive sentence for a tier the v1 GUI
#: never offers does not earn an audit wave of its own. The GUI paints
#: ``error.message`` only when it has no copy for ``error.code``; the code is the
#: contract, and ``tier_not_available`` is what it branches on.
TIER_NOT_AVAILABLE_MESSAGE = (
    "This app can start only Cloud runs on Radient for now; the Dedicated tier is "
    "not offered here yet, so nothing was submitted."
)

#: The sentence for a directory path this surface cannot even resolve or read.
#:
#: ROUTE-OWNED ENGLISH PROSE for the reason ``TIER_NOT_AVAILABLE_MESSAGE`` records:
#: the code is the contract and a renderer branches on it, while a keyed message needs
#: seven translations and an audit before it may ship. What reaches this is the
#: caller's own string — a 4096-character body, which ``RunSubmission`` admits and the
#: filesystem refuses — and never a repository verdict, which ``repo.detect_repo``
#: answers as ``no_remote`` in its own words.
DIRECTORY_UNUSABLE_MESSAGE = (
    "That directory cannot be read on this machine — its path is too long, or a part "
    "of it is not a directory — so no run was submitted. Pick the repository folder "
    "again."
)

#: The contract's own code for "this deployment does not serve that".
CODE_NOT_CONFIGURED = "not_configured"

#: The sentence for a surface this Radient deployment does not serve.
#:
#: ROUTE-OWNED ENGLISH PROSE for the reason ``TIER_NOT_AVAILABLE_MESSAGE`` records,
#: and here it is also the only way to say it in one round: a keyed sentence would
#: have to join ``wire.remote_compute``, whose ledger rows then go stale and whose
#: seven carried locales UN-SHIP until an independent audit of the namespace lands
#: (`generate.py --check` regenerates ``shipped.json`` to ``en`` only, and the suite's
#: carrying-locale tests fail until the audit restores it). The code is the contract
#: a renderer branches on; this is what a person reads when it has no copy of its own.
SURFACE_NOT_CONFIGURED_MESSAGE = (
    "This Radient deployment does not serve this: the feature is not configured "
    "there, so nothing was read or changed. Retrying here will not help — it has to "
    "be switched on for the deployment."
)

#: The tiers this PR can start. ``dedicated`` is refused as ``tier_not_available``
#: rather than accepted and failed upstream: the GUI's second tier arrives with a
#: later release, and a picker row that was accepted here would submit work this
#: build cannot describe.
TIER_CLOUD_RUN = "cloud_run"
TIER_DEDICATED = "dedicated"

#: The longest prompt this surface accepts, in characters.
#:
#: DERIVED, NOT INVENTED. The remote-compute contract documents no prompt bound of
#: its own (``remote_compute.request.RunSpec`` checks that the prompt is non-empty
#: and nothing more; ``client.check_idempotency_key`` bounds only the key), so the
#: bound here is this plane's own prompt-shaped one: ``desktop_sessions.Prompt.text``
#: carries ``max_length=200_000``, and a create body is the same kind of thing. An
#: unbounded body would reach a proxy and come back a 413 the operator cannot act
#: on; the 422 this answers is the same refusal the plane's other request models
#: give.
PROMPT_MAX_CHARS = 200_000

#: The words the runs listing uses for the SERVER half of the merge. Not a
#: refusal code: the listing answers the local records either way, and a panel
#: that could not read the server must still render the work this device sent.
SERVER_OK = "ok"
SERVER_SIGNED_OUT = "signed-out"
SERVER_UNAVAILABLE = "unavailable"
#: The switch is off, so there is no server half and no request was made. Its own
#: word rather than ``unavailable``: a renderer that painted a retry over it would
#: offer to re-read a host the user has told this device to leave alone.
SERVER_DISABLED = "disabled"

#: The codes for refusals this surface names that no server code covers.
#:
#: A refusal that reached the client as typed prose still needs a STABLE code
#: here, because a renderer branches on ``code`` and never on the sentence:
#: "Radient could not be reached" is retryable, "another network's run" is not.
#: Classes the server's contract already codes are absent on purpose — their own
#: code travels with the error (see ``ClientRefusal.code``).
CODE_SIGNED_OUT = "signed_out"
CODE_UNREACHABLE = "unreachable"
CODE_UNREADABLE_ENVELOPE = "unreadable_envelope"
CODE_NETWORK_MISMATCH = "network_mismatch"
CODE_RUN_IDENTITY_MISMATCH = "run_identity_mismatch"
CODE_INVALID_API_OVERRIDE = "invalid_api_override"
CODE_INVALID_REQUEST = "invalid_request"
CODE_CAPACITY_UNAVAILABLE = "capacity_unavailable"
#: The contract's own spelling for "no such run", preserved rather than replaced
#: by a surface code: a renderer branching on ``error.code`` alone must be able to
#: tell a run that is gone from a library failure, and ``not_found`` is the word
#: the server itself sends.
CODE_RUN_NOT_FOUND = "not_found"
#: Also the contract's spelling, for a key this client refuses before sending.
CODE_INVALID_IDEMPOTENCY_KEY = "invalid_idempotency_key"
CODE_RESULTS_SIZE_MISMATCH = "results_size_mismatch"
#: The artifact host could not be reached, or the transfer broke (503, because the
#: far end is unavailable rather than the request being wrong).
CODE_RESULTS_UNREACHABLE = "results_unreachable"
#: A caller-supplied directory this surface could not even resolve or read: a path
#: whose own bytes the filesystem refuses (``ENAMETOOLONG``), not a directory
#: problem about the repository (that is ``no_remote``, from ``repo.detect_repo``).
CODE_DIRECTORY_UNUSABLE = "directory_unusable"
CODE_FAILED = "remote_compute_failed"

#: The status each refusal answers when the library carries no HTTP status of its
#: own. A refusal that DOES carry one (it came back from an HTTP answer) mirrors
#: that status instead — see :func:`status_for`.
_STATUS_BY_CLASS: tuple[tuple[type[BaseException], int], ...] = (
    (LoginRequired, 401),
    (PreviewNotEnabled, 403),
    (InsufficientCredit, 402),
    (RunNotFound, 404),
    (IdempotencyConflict, 409),
    (RunLimitReached, 409),
    (RunBusy, 409),
    (Conflict, 409),
    (CapacityUnavailable, 503),
    (InvalidRequest, 400),
    (InvalidIdempotencyKey, 400),
    (Unreachable, 502),
    (InvalidApiOverride, 409),
    (UnreadableEnvelope, 502),
    (NetworkMismatch, 502),
    (RunIdentityMismatch, 502),
    # THE RESULTS FAMILY, and the statuses are the DISTINCTION rather than a
    # convention. A run that published nothing, or a destination that is already
    # there, is a conflict with the state of the run or of this machine (409); an
    # archive this build will not write and a download that is not the size the
    # server advertised are the far end having sent something unusable (502); and a
    # clone that cannot take the bundle is again this machine's state (409).
    (ResultsUnavailable, 409),
    (BundleApplyFailed, 409),
    (UnsafeArchive, 502),
    (SizeMismatch, 502),
    # NOT 409, because the run is not in a bad state: the artifact host could not be
    # reached or the transfer broke, and the remedy is to ask for a fresh listing
    # once it is back — the same status a store that is down answers elsewhere here.
    (ResultsUnreachable, 503),
)

#: The code each library-internal refusal answers, in the same order. Read only
#: when the error carries no server code of its own.
_CODE_BY_CLASS: tuple[tuple[type[BaseException], str], ...] = (
    (LoginRequired, CODE_SIGNED_OUT),
    (Unreachable, CODE_UNREACHABLE),
    (UnreadableEnvelope, CODE_UNREADABLE_ENVELOPE),
    (NetworkMismatch, CODE_NETWORK_MISMATCH),
    (RunIdentityMismatch, CODE_RUN_IDENTITY_MISMATCH),
    (InvalidApiOverride, CODE_INVALID_API_OVERRIDE),
    (InvalidRequest, CODE_INVALID_REQUEST),
    (InvalidIdempotencyKey, CODE_INVALID_IDEMPOTENCY_KEY),
    (CapacityUnavailable, CODE_CAPACITY_UNAVAILABLE),
    (RunNotFound, CODE_RUN_NOT_FOUND),
    (SizeMismatch, CODE_RESULTS_SIZE_MISMATCH),
    (ResultsUnreachable, CODE_RESULTS_UNREACHABLE),
)

#: The status a MESH refusal answers, by the refusal's own code.
#:
#: The mesh's codes are its contract (``resolve`` mints them), so the status
#: follows the meaning rather than one blanket code: no network and an ambiguous
#: one are the same picker problem (409), an id this device never joined is not
#: found, and a network this device is no longer a member of is forbidden.
_STATUS_BY_MESH_CODE: dict[str, int] = {
    "remote_compute_no_network": 409,
    "remote_compute_ambiguous_network": 409,
    "remote_compute_unknown_network": 404,
    "remote_compute_not_a_member": 403,
    "remote_compute_invalid_network_id": 400,
    "remote_compute_duplicate_network_id": 400,
}


class ClientRefusal(Exception):
    """A refusal from this surface, answered by :func:`_refusing`.

    A plain ``Exception`` with three fields rather than an ``HTTPException``
    because the body is not FastAPI's ``{"detail": …}``: see the module docstring
    for why this surface's client reads ``error.code``. ``retry_after`` is carried
    through to the ``Retry-After`` header, so a 429's instruction reaches the
    caller as a header a generic HTTP client already understands.
    """

    def __init__(
        self, status: int, code: str, message: str, *, retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after = retry_after


def refusal_from(failure: BaseException) -> ClientRefusal:
    """The one reading of any failure this surface can meet.

    THE LIBRARY'S SENTENCES, NOT NEW ONES. ``messages.refusal_message`` is the
    mapping the CLI prints, so a refusal reads the same whichever surface the
    operator is looking at; a mesh refusal carries its own sentence (its codes are
    documented in ``remote_compute.resolve``) and never passes through that
    mapping, which knows nothing about the mesh.

    THE SERVER'S STATUS WINS when the answer carried one: a 402 is a payment, a 429
    is a wait and a 503 is a capacity problem, and only the status the server sent
    keeps that distinction intact.
    """
    if isinstance(failure, MeshRefusal):
        return ClientRefusal(
            _STATUS_BY_MESH_CODE.get(failure.code, 409), failure.code, str(failure)
        )
    if isinstance(failure, UnusableRepo):
        # The reason IS the code: each one has its own remedy and its own copy
        # (``errors.UnusableRepo``), and the answer names it so a renderer can put
        # the remedy beside the message.
        return ClientRefusal(409, failure.reason, refusal_message(failure))
    if isinstance(failure, RemoteComputeError):
        if failure.code == CODE_NOT_CONFIGURED:
            # THE FAR END'S OWN CODE OUTRANKS ITS CLASS'S SENTENCE. ``not_configured``
            # reaches this table on a ``Conflict`` (``errors.ERROR_BY_CODE``), whose
            # remedy — read the run again, use a different name — cannot fix a
            # deployment that does not serve the surface at all (QA round 1, Q1).
            # Only the far end sets this code: this plane's own switch-off refusal is
            # built by ``_disabled`` and never passes through here.
            return ClientRefusal(
                status_for(failure),
                CODE_NOT_CONFIGURED,
                SURFACE_NOT_CONFIGURED_MESSAGE,
                retry_after=failure.retry_after,
            )
        code = _contract_code(failure) or _class_code(failure)
        retry_after = failure.retry_after
        return ClientRefusal(
            status_for(failure), code, refusal_message(failure), retry_after=retry_after
        )
    if isinstance(failure, LoginRequired):
        return ClientRefusal(401, CODE_SIGNED_OUT, refusal_message(failure))
    if isinstance(failure, ValueError):
        # A locally refused body (``RunSpec``'s invariants, a run id that is not
        # one): the client composed or the caller sent something the contract does
        # not admit, and ``invalid_request``'s sentence says so once.
        return ClientRefusal(400, CODE_INVALID_REQUEST, _t(_wr.invalid_request()))
    logger.warning("remote compute: unclassified failure %s", type(failure).__name__)
    return ClientRefusal(500, CODE_FAILED, _t(_wr.failed(error=type(failure).__name__)))


def status_for(failure: RemoteComputeError) -> int:
    """The HTTP status for a library failure: the server's own, else its class's."""
    if failure.status is not None:
        return failure.status
    for error_type, status in _STATUS_BY_CLASS:
        if isinstance(failure, error_type):
            return status
    return 502


def _class_code(failure: RemoteComputeError) -> str:
    """This surface's own code for a failure the server did not name one for.

    ``RunNotFound``'s entry is why this table is not only "codes we invented": a
    per-run read that comes back 404 with NO code still answers the contract's
    ``not_found``, so a renderer that branches on ``error.code`` alone can tell
    "this run is gone" from a generic library failure without also reading the
    status (QA round 1, Q1).

    THE REASON IS THE CODE for the results families, exactly as it is for
    :class:`UnusableRepo` (which has its own arm in :func:`refusal_from`): each
    reason has its own remedy — a missing artifact, an archive this device will not
    write, a clone whose base commit is absent — and a renderer branches on the
    code. Named classes rather than ``getattr(failure, "reason", None)`` for
    everything, because two other families carry a ``reason`` that is NOT a wire
    code and this table's answer for them must not move.
    """
    if isinstance(failure, (BundleApplyFailed, ResultsUnavailable, UnsafeArchive)):
        return failure.reason
    for error_type, code in _CODE_BY_CLASS:
        if isinstance(failure, error_type):
            return code
    return CODE_FAILED


def _contract_code(failure: RemoteComputeError) -> str | None:
    """The server's own code, only when the contract's closed set names it.

    ``errors.refusal_code`` already drops an unknown string on the way in, so this
    is the second gate on the way OUT: a ``RemoteComputeError`` composed in-process
    (this surface mints several) could still carry an arbitrary ``code``, and a
    renderer keying on ``error.code`` must only ever see a member of the contract's
    vocabulary. An unattributable code falls through to :func:`_class_code`, which
    answers the surface's own word for the failure's CLASS — ``not_found`` for a
    404, never the arbitrary string — and the STATUS stays exactly what it was,
    because the status is a fact about the answer and the code is what this surface
    can say about it (QA round 1, Q1).
    """
    code = failure.code
    if code is None:
        return None
    if code in REFUSAL_CODES or code.startswith(ATTESTATION_CODE_PREFIX):
        return code
    return None


def _answer(refusal: ClientRefusal) -> JSONResponse:
    """A refusal as ``{"error": {code, message, retry_after?}}``, with its status."""
    body: dict[str, Any] = {"code": refusal.code, "message": refusal.message}
    headers: dict[str, str] = {}
    if refusal.retry_after is not None:
        # WHOLE SECONDS, ROUNDED UP: a header value is an integer, and rounding
        # down would tell a caller to come back fractionally early.
        body["retry_after"] = refusal.retry_after
        headers["Retry-After"] = str(max(1, int(refusal.retry_after + 0.999)))
    return JSONResponse({"error": body}, status_code=refusal.status, headers=headers)


def _refusing(route: Callable[..., Any]) -> Callable[..., Any]:
    """Answer this route's refusals in the one shape, from the one mapping.

    ``functools.wraps`` keeps the signature FastAPI introspects (it follows
    ``__wrapped__``), so a route stays a route rather than becoming an opaque
    callable; and the one handler is why six routes cannot answer a 429 three
    different ways.
    """

    @functools.wraps(route)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await route(*args, **kwargs)
        except ClientRefusal as refusal:
            return _answer(refusal)
        except Exception as failure:  # noqa: BLE001 — every exit is shaped here
            if isinstance(failure, HTTPException):
                # Not ours: the desktop plane's own gate (``require_desktop``)
                # answers in its own envelope before or beside this handler.
                raise
            return _answer(refusal_from(failure))

    return wrapper


class RunSubmission(BaseModel):
    """The create body: a directory, a prompt, a tier and the run's own bounds.

    ``extra="forbid"``, the rule every desktop request model on this plane
    follows: a misspelt key is a 422 rather than a silently ignored intent.

    ``directory`` IS A PATH ON THIS MACHINE, and the whole point of the field: the
    route reads the repository and HEAD from it (``remote_compute.repo``) so the
    operator never retypes a URL and a commit. It is resolved the way this plane's
    other caller-supplied directory is (``Path(cwd).expanduser().resolve()``,
    ``server/utils/desktop_sessions.resolve_working_directory``): a renderer that
    sends ``~/work/repo`` means the operator's home, and handing the raw string to
    ``detect_repo`` reported a perfectly good repository as "not a git repository
    with an origin remote". ``network_id`` is optional because
    :func:`resolve_network` answers the device's one network when the caller names
    none, and refuses — never guesses — when there are several.

    ``idempotency_key`` IS THE CALLER'S, when it has one. See :func:`create_run`
    for why the key must be suppliable and not only returned.
    """

    model_config = ConfigDict(extra="forbid")

    directory: str = Field(min_length=1, max_length=4096)
    prompt: str = Field(min_length=1, max_length=PROMPT_MAX_CHARS)
    tier: Literal["cloud_run", "dedicated"]
    size: Literal["s", "m", "l"] | None = None
    max_hours: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    network_id: str | None = None
    idempotency_key: str | None = None


class ResultsFetch(BaseModel):
    """The fetch body: where to extract, or nothing for the run's default directory.

    ``dest_dir`` IS OPTIONAL AND ABSOLUTE. Optional because the default is a
    directory this app owns (``<config>/remote-compute/results/<run id>``), and
    absolute because a relative path would resolve against the daemon's working
    directory rather than the operator's — the refusal names that rather than
    landing the files somewhere nobody looks.
    """

    model_config = ConfigDict(extra="forbid")

    dest_dir: str | None = None


class ResultsApply(BaseModel):
    """The apply body: the clone the run's bundle is fetched into.

    REQUIRED, because there is no directory this app may guess: the bundle belongs
    beside the commits it builds on, and only the operator knows which clone that
    is. The clone must already hold the run's base commit
    (``git bundle verify`` enforces it).

    ABSOLUTE AND THE REPOSITORY'S OWN ROOT, both refused by the library rather than
    here (``relative_destination``, ``not_repository_root``): the fetch writes a ref
    into ``<dir>/.git``, so a relative path would resolve against the daemon's working
    directory and a path merely INSIDE a repository would put this app's ref in a
    repository nobody named.
    """

    model_config = ConfigDict(extra="forbid")

    directory: str = Field(min_length=1)


@contextlib.asynccontextmanager
async def _transport() -> AsyncIterator[Any]:
    """The HTTP client one request uses; the seam a test replaces.

    IMPORTED HERE, not at module scope: this module is imported by ``server.app``,
    and a module-level ``httpx`` is a cost every ``lop serve`` boot pays for routes
    that only run when a desktop opens a Radient screen (the same reasoning
    ``desktop_radient`` records).
    """
    import httpx

    async with httpx.AsyncClient(follow_redirects=False) as client:
        yield client


@contextlib.asynccontextmanager
async def _runs(config_dir: Path, credential_id: int) -> AsyncIterator[RemoteRuns]:
    """The library's run API over this request's transport, recording in ``config_dir``.

    ``credential_id`` IS REQUIRED, and that is the client's own rule rather than a
    choice here: a remote run is billed to, owned by and visible to ONE Radient
    account, so a surface that left the client to resolve one would let the client's
    shell default answer a question about THIS daemon's account. The CLI pins one per
    invocation and a server pins one per request (see
    ``remote_compute.client.RemoteComputeClient``); :func:`_pin` is where this
    request's answer comes from.
    """
    async with _transport() as http:
        yield RemoteRuns(
            RemoteComputeClient(http, credential_id=credential_id), config_dir=config_dir
        )


def _config_dir(request: Request, manager: ConfigManager) -> Path:
    """The config root this daemon serves, not the process default.

    A backend started against a relocated root must read THAT machine's config,
    switch and records — the same rule every mesh read on this plane follows
    (``desktop_mesh._root``). The manager is a declared dependency rather than an
    ``app.state`` read so a test can override it, which is why it is taken first
    and the request is only the fallback for a route mounted without one.
    """
    resolved = getattr(manager, "config_dir", None)
    if resolved is not None:
        return Path(resolved)
    return Path(request.app.state.config_manager.config_dir)


# -- the reads ---------------------------------------------------------------


@router.get("/v1/desktop/remote-compute")
@_refusing
async def remote_compute(
    request: Request,
    manager: ConfigManager = Depends(get_config_manager),
    auth: DesktopAuth = Depends(get_desktop_auth),
) -> Any:
    """The device dropdown's whole Radient group, in one read.

    THREE FIELDS ARE DELIBERATELY HONEST RATHER THAN COMPLETE:

    * ``pricing`` is ``null``. The capacity read returns ``{tier, ready, detail}``
      and no rate, so a renderer printing a price would be printing a constant that
      drifts from the deployment's own config;
    * ``networks`` lists only networks this device is an ACTIVE member of — the set
      ``resolve_network`` will bind a run to. Listing a network whose membership
      this device lost would offer a choice that always refuses;
    * ``active`` carries the SERVER's live runs only. The local record cannot supply
      a tier, a deadline or a hold, so showing those rows offline is the runs
      listing's job (``/runs``), not this read's.

    ``offered`` is the deployment's answer, not this device's — see
    :func:`_capacity`. ``account.state`` outranks it for a renderer: "sign in to
    use" is the on-ramp and "not offered by this deployment" is the provider's
    switch, and only one of them is something the user can act on today.
    """
    root = _config_dir(request, manager)
    if not is_enabled(root):
        # NO UPSTREAM CALL, by design: the switch exists so that nothing about the
        # user's work leaves this device while it is off.
        return {"enabled": False}
    account = await _account(auth)
    # A local read, and answered whether or not Radient is reachable: the network
    # binding is this device's own record.
    networks = await asyncio.to_thread(_networks, root)
    capacity: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    offered = True
    if account["state"] == "signed-in":
        credential_id = await _pin(auth)
        if credential_id is None:
            # Signed in, but no bearer can be minted this moment. That is a
            # transient, and the deployment's own answer is not ours to invent:
            # ``offered`` stays true and the reads come back empty.
            offered = True
        else:
            capacity, offered = await _capacity(root, credential_id)
            # THE LIVE RUNS ARE READ EVEN WHEN THE DEPLOYMENT OFFERS NO MORE: a run
            # already in flight was admitted under the old answer and is still
            # spending money and time, so hiding it from the dropdown that reports
            # it would be the one omission this read must never make.
            active = await _active_runs(root, credential_id)
    return {
        "enabled": True,
        "account": account,
        "networks": networks,
        "capacity": capacity,
        "pricing": None,
        "active": active,
        "offered": offered,
    }


@router.get("/v1/desktop/remote-compute/runs")
@_refusing
async def list_runs(
    request: Request,
    manager: ConfigManager = Depends(get_config_manager),
    auth: DesktopAuth = Depends(get_desktop_auth),
) -> Any:
    """The runs panel: the server's runs merged with this device's own records.

    THE RECORD IS WHY THIS READ ANSWERS OFFLINE. A run outlives the network that was
    up when it was submitted, so the server half is best-effort: when Radient cannot
    be reached, or nothing is signed in, the rows this device recorded are still the
    answer (``server`` says which it was). The alternative — refusing the listing —
    would hide the operator's own work behind a dead Radient, which is the state a
    remote panel most needs to survive.

    THE MERGE IS ONE ROW PER RUN ID, server first: the server is authoritative about
    a run's present status, and a record row is appended only for an id the server
    did not return — including one marked ``deleted``, which is kept on purpose so a
    panel can say a run was deleted instead of silently losing the row.

    EVERY ROW CARRIES ``id``, whichever half it came from: the server's run object
    names it ``id`` and a record names it ``run_id``, and a renderer that had to try
    both keys on every row of every panel is the work :func:`_record_row` does once,
    here.

    THE FLAG OFF ANSWERS AN EMPTY LISTING WITH NO UPSTREAM CALL (``server:
    "disabled"``), like the summary read: the listing is the read that would ask
    Radient what this device has been doing, and the switch exists so that nothing
    about the user's work leaves the device while it is off.
    """
    root = _config_dir(request, manager)
    if not is_enabled(root):
        # NO UPSTREAM CALL, by design — and no local rows either: an empty listing
        # with its own word for why is one state a panel paints, while records
        # shown under a switch the user turned off would look like the switch
        # failed.
        return {"runs": [], "server": SERVER_DISABLED}
    server_state = SERVER_OK
    server_runs: list[dict[str, Any]] = []
    credential_id = await _pin(auth)
    if credential_id is None:
        server_state = SERVER_SIGNED_OUT
    else:
        async with _runs(root, credential_id) as runs:
            try:
                server_runs = await runs.list_runs()
            except LoginRequired:
                server_state = SERVER_SIGNED_OUT
            except RemoteComputeError as failure:
                logger.info("remote compute: runs listing fell back to records (%s)", failure)
                server_state = SERVER_UNAVAILABLE
    merged: list[dict[str, Any]] = [
        {**without_secrets(run), "source": "server"} for run in server_runs
    ]
    seen = {run.get("id") for run in server_runs}
    for record in records.load(root):
        if record.run_id in seen:
            continue
        merged.append(_record_row(record))
    return {"runs": merged, "server": server_state}


@router.get("/v1/desktop/remote-compute/runs/{identifier}")
@_refusing
async def get_run(
    identifier: str,
    request: Request,
    manager: ConfigManager = Depends(get_config_manager),
    auth: DesktopAuth = Depends(get_desktop_auth),
) -> Any:
    """One run, refreshed into the local record, with the record as the offline answer.

    ONLY WHAT THE PANEL CANNOT DO BETTER FALLS BACK. A refusal that means something
    (no such run, another network's run, a run that already finished) is the answer:
    reporting a local copy over it would tell the operator a run is where it is not.
    ``Unreachable`` — and an unusable sign-in, which is the same thing from the
    panel's point of view — answers the record row when this device has one, because
    a device whose grant cannot be refreshed right now still LISTED the run in
    ``/runs`` and an offline laptop must be able to open the row it shows.
    """
    root = _config_dir(request, manager)
    credential_id = await _pin(auth)
    if credential_id is None:
        # READ LAZILY, in both fallbacks: a run the server answers needs no local
        # read, and the record file is the one thing here that grows with use.
        record = _recorded(root, identifier)
        if record is None:
            raise _no_credential()
        return {"run": _record_row(record)}
    async with _runs(root, credential_id) as runs:
        try:
            run = await runs.get_run(identifier)
        except Unreachable:
            record = _recorded(root, identifier)
            if record is None:
                raise
            return {"run": _record_row(record)}
    return {"run": {**without_secrets(run), "source": "server"}}


@router.get("/v1/desktop/remote-compute/runs/{identifier}/logs")
@_refusing
async def run_logs(
    identifier: str,
    request: Request,
    next_token: str | None = None,
    limit: int = LOGS_PAGE_DEFAULT,
    manager: ConfigManager = Depends(get_config_manager),
    auth: DesktopAuth = Depends(get_desktop_auth),
) -> Any:
    """One page of a run's logs, in the server's own page shape.

    ONE PAGE, NOT A TAIL. The streaming loop is the library's ``follow_logs``, and
    a route that held a request open for the life of a run would be a connection a
    GUI has to keep alive, restart and time out on its own. So the renderer polls
    this with the ``next_token`` it was given, and the two stop rules — an
    unchanged token on two empty pages means no new lines, a terminal run with an
    empty page is the end — stay in the library where they are tested once.

    THE CONSENT SWITCH GATES THIS READ. With ``remote_compute.enabled`` off nothing
    about the operator's work leaves the device, and a log read is work leaving the
    device: the answer is the same 403 ``not_configured`` a create gets, and no
    upstream call is made. The flag is read per request, so a live toggle reaches a
    running daemon.
    """
    root = _config_dir(request, manager)
    _require_enabled(root)
    credential_id = _require_pin(await _pin(auth))
    async with _runs(root, credential_id) as runs:
        page = await runs.client.run_logs(identifier, next_token=next_token, limit=limit)
    return {
        "events": list(page.events),
        "next_token": page.next_token,
        "run_status": page.run_status,
    }


@router.get("/v1/desktop/remote-compute/runs/{identifier}/results")
@_refusing
async def run_results(
    identifier: str,
    request: Request,
    manager: ConfigManager = Depends(get_config_manager),
    auth: DesktopAuth = Depends(get_desktop_auth),
) -> Any:
    """The run's artifacts WITHOUT their download URLs, and its git summary.

    THE URLS DO NOT LEAVE THIS PROCESS except into the download itself. Each one is
    a bearer that authorises one GET for a few minutes, so a body carrying them
    would put the run's results in the client's cache and in every log the client
    keeps; :func:`fetch_results` is the one place a URL is used, and it never
    returns one. ``Artifact.to_json`` is that filter, so this route cannot forget
    it — the same shape ``without_secrets`` gives a run body.

    A run that has not published anything answers an empty list, not a refusal: "no
    artifacts yet" is a normal answer while a run is still going, and the panel
    paints it.
    """
    root = _config_dir(request, manager)
    _require_enabled(root)
    credential_id = _require_pin(await _pin(auth))
    async with _runs(root, credential_id) as runs:
        listing = await runs.client.run_artifacts(identifier)
    return {
        "run_status": listing.run_status,
        "artifacts": [artifact.to_json() for artifact in listing.artifacts],
        "git": listing.git,
    }


# -- the writes --------------------------------------------------------------


@router.post("/v1/desktop/remote-compute/runs")
@_refusing
async def create_run(
    body: RunSubmission,
    request: Request,
    manager: ConfigManager = Depends(get_config_manager),
    auth: DesktopAuth = Depends(get_desktop_auth),
) -> Any:
    """Submit one run from a directory on this machine.

    THE ORDER OF THE REFUSALS IS THE POINT, and every step refuses LOCALLY rather
    than letting the far end fail minutes later:

    1. the preview switch — off means no request is built at all;
    2. the tier — this build starts Cloud runs, and Dedicated answers
       ``tier_not_available`` instead of being sent to an endpoint it does not have;
    3. the sign-in — the run is billed to one account, so there must be one;
    4. the network — ``resolve_network`` issues the binding a request can only be
       built from, so a device in no network (or in two) is told that before any
       work is addressed to a host;
    5. the repository — ``repo.detect_repo`` reads the https origin and HEAD, and
       refuses an ssh-only origin or an unpushed commit;
    6. the model pair — a run does not inherit this device's provider keys, so the
       far end must be told which model to run, resolved through the same ladder a
       local exec uses.

    THE IDEMPOTENCY KEY IS THE CALLER'S WHEN IT SENDS ONE, and minted here (once per
    request) when it does not. The key exists for the retry, and the retry that
    matters is the LOST RESPONSE: a renderer whose POST timed out never received the
    key the first attempt minted, so a surface that only returned one turned every
    proxy timeout into a second run, a second pod and a second charge. A body that
    carries ``idempotency_key`` replays the same run instead. Reusing a key for
    DIFFERENT work is not a hole this opens: the server answers it ``409
    idempotency_conflict``, which is the conflict the header exists to detect. The
    key is validated by the library's own ``check_idempotency_key`` before a request
    is composed (a 400 ``invalid_idempotency_key``), is not secret — it names a
    request, not a credential — and is RETURNED either way, so a retry can also
    start from the response. The log line pairs it with the run id so a submission
    can be reconciled against the server's own record.
    """
    root = _config_dir(request, manager)
    if not is_enabled(root):
        raise _disabled()
    if body.tier != TIER_CLOUD_RUN:
        raise ClientRefusal(400, "tier_not_available", TIER_NOT_AVAILABLE_MESSAGE)
    credential_id = _require_pin(await _pin(auth))
    binding = await asyncio.to_thread(resolve_network, body.network_id, root)
    # RESOLVED, not taken raw: see ``RunSubmission.directory``. A ``~`` the caller
    # meant as the operator's home reached ``detect_repo`` as a literal folder name
    # and came back "not a git repository".
    try:
        directory = Path(body.directory).expanduser().resolve()
        repository = await asyncio.to_thread(repository_for, directory)
    except OSError:
        # THE PATH ITSELF CAN FAIL BEFORE ANY REPOSITORY QUESTION IS ASKED: a
        # 4096-character body (the field's own limit) is a component the filesystem
        # refuses with ``ENAMETOOLONG``, and reading the repository through it fails
        # the same way. That is the caller's input rather than this surface's fault,
        # so it answers a coded refusal instead of the unclassified-failure 500 the
        # blanket handler would otherwise produce (PR #4200 review round 2, Q3).
        raise ClientRefusal(400, CODE_DIRECTORY_UNUSABLE, DIRECTORY_UNUSABLE_MESSAGE) from None
    try:
        hosting, model = await asyncio.to_thread(_hosting_model, manager)
    except ValueError as failure:
        # The resolver's own sentence (``bootstrap.resolve_hosting_model``): a lower
        # layer's reason travels verbatim rather than being reworded here.
        raise ClientRefusal(409, "not_configured", str(failure)) from None
    spec = build_spec(
        prompt=body.prompt,
        repo=repository,
        hosting=hosting,
        model=model,
        defaults=run_defaults(),
        size=body.size,
        max_hours=body.max_hours,
    )
    payload = RunCreateRequest(binding=binding, body=spec.to_body())
    # THE LIBRARY'S OWN RULE, applied to a key this surface was HANDED: the client
    # checks the key it is given too, but it does so after the run sequence has
    # begun, and a refusal there is a 400 under a class this surface would have
    # reported as a generic 502.
    key = (
        check_idempotency_key(body.idempotency_key)
        if body.idempotency_key is not None
        else new_idempotency_key()
    )
    async with _runs(root, credential_id) as runs:
        submitted = await runs.create_run(
            payload,
            idempotency_key=key,
            recorded_from=RunSubmitMetadata(
                repo_url=repository.url,
                repo_sha=repository.sha,
                prompt=body.prompt,
                size=spec.size,
            ),
        )
    run = without_secrets(submitted.run)
    logger.info(
        "remote-compute run submitted run_id=%s idempotency_key=%s replayed=%s",
        run.get("id"),
        key,
        submitted.replayed,
    )
    # 201 for a run this request created, 200 for a replay: the library carries the
    # distinction because the body is identical either way, and a renderer that
    # toasted "submitted" over a replay would claim work that already existed.
    return JSONResponse(
        {
            "run": run,
            "replayed": submitted.replayed,
            "network_id": binding.network_id,
            "idempotency_key": key,
        },
        status_code=200 if submitted.replayed else 201,
    )


@router.post("/v1/desktop/remote-compute/runs/{identifier}/cancel")
@_refusing
async def cancel_run(
    identifier: str,
    request: Request,
    manager: ConfigManager = Depends(get_config_manager),
    auth: DesktopAuth = Depends(get_desktop_auth),
) -> Any:
    """Cancel a run; idempotent, and ``run`` is ``null`` when the server sent no body.

    The library refreshes the local record's status from the answer, so the panel's
    next read agrees with what this call was told.
    """
    credential_id = _require_pin(await _pin(auth))
    async with _runs(_config_dir(request, manager), credential_id) as runs:
        run = await runs.cancel_run(identifier)
    return {"run": without_secrets(run) or None}


@router.delete("/v1/desktop/remote-compute/runs/{identifier}")
@_refusing
async def delete_run(
    identifier: str,
    request: Request,
    manager: ConfigManager = Depends(get_config_manager),
    auth: DesktopAuth = Depends(get_desktop_auth),
) -> Any:
    """Delete a run's record; ``run`` is ``null`` when the server sent no body.

    The LOCAL record is kept and marked deleted (``records.mark_deleted``), so the
    runs listing can still say a run existed here — a row that vanished would look
    like this device forgot work it submitted.
    """
    credential_id = _require_pin(await _pin(auth))
    async with _runs(_config_dir(request, manager), credential_id) as runs:
        run = await runs.delete_run(identifier)
    return {"run": without_secrets(run) or None}


@router.post("/v1/desktop/remote-compute/runs/{identifier}/results/fetch")
@_refusing
async def fetch_results(
    identifier: str,
    request: Request,
    # NO BODY IS A VALID REQUEST — "extract it where the app puts results" — and a
    # required model would answer 422 before the consent gate or the sign-in check
    # could speak. The body only ever carries a destination.
    body: ResultsFetch | None = None,
    manager: ConfigManager = Depends(get_config_manager),
    auth: DesktopAuth = Depends(get_desktop_auth),
) -> Any:
    """Download and extract the run's results, and offer its session for ``--resume``.

    THE ORDER IS THE SAFETY, and it is the library's: the archive is downloaded to a
    private temp file, checked against the size the listing advertised, and only
    then extracted into a staging directory this call renames onto a destination
    that must NOT already exist. So a refused or truncated download, and an archive
    whose members this device will not write, all leave NO results directory — a
    panel therefore never offers a half-populated tree as the run's work, and the
    retry at the same (per-run default) path is answered on its merits rather than
    ``destination_exists``.

    THE SESSION IS RESTORED IN A THREAD. Copying a session directory is file I/O
    that can take a moment on a large transcript, and this loop serves every other
    request in the daemon: the same reason ``_networks`` and the repository read run
    off the loop. ``None`` means the archive carried no session directory (a run
    that produced artifacts but no conversation), which is a legitimate answer.
    """
    root = _config_dir(request, manager)
    _require_enabled(root)
    credential_id = _require_pin(await _pin(auth))
    chosen = body.dest_dir if body is not None else None
    destination = Path(chosen) if chosen else results.results_directory(identifier, config_dir=root)
    async with _runs(root, credential_id) as runs:
        fetched = await runs.fetch_results(identifier, destination)
        restored = await asyncio.to_thread(runs.restore_session, fetched.directory)
    return {
        "directory": str(fetched.directory),
        "file_count": fetched.file_count,
        "archive_bytes": fetched.archive_bytes,
        "session_id": restored.session_id if restored is not None else None,
        "session_restored": bool(restored is not None and restored.created),
    }


@router.post("/v1/desktop/remote-compute/runs/{identifier}/results/apply")
@_refusing
async def apply_bundle(
    identifier: str,
    body: ResultsApply,
    request: Request,
    manager: ConfigManager = Depends(get_config_manager),
    auth: DesktopAuth = Depends(get_desktop_auth),
) -> Any:
    """Fetch the run's repository bundle into a clone, as a ref this app owns.

    ``refs/remotes/radient/lop/<run id>`` AND NOT A BRANCH: nothing here checks
    anything out, so the run's commits are visible to ``git log`` while the
    operator's HEAD, index and branches are exactly what they were. The clone must
    already hold the run's base commit — ``git bundle verify`` enforces that, and
    its refusal is the honest answer to "this clone has never seen that code".
    """
    root = _config_dir(request, manager)
    _require_enabled(root)
    credential_id = _require_pin(await _pin(auth))
    async with _runs(root, credential_id) as runs:
        applied = await runs.apply_bundle(identifier, body.directory)
    return {"ref": applied.ref, "commit": applied.commit}


# -- the pieces the routes read ----------------------------------------------
#
# EVERY UPSTREAM CALL IS PINNED TO ONE CREDENTIAL, resolved from THIS daemon's
# store (``app.state.desktop_auth.store``). The library's default is
# ``tunnels.api.credential_id``, which opens the PROCESS default config dir: right
# for a CLI invocation, wrong for a daemon started against a relocated root, and
# wrong in a test whose store is a temporary database. So the pin is resolved here,
# once, and handed to every client this module builds.


def _require_enabled(root: Path) -> None:
    """Refuse a read that would reach Radient while the consent switch is off.

    THE SAME REFUSAL A CREATE GETS (``403 not_configured``), because it is the same
    question: with ``remote_compute.enabled`` off, nothing about the operator's work
    leaves the device, and a one-page log read or an artifact listing is that work
    leaving. Raised rather than answered with ``enabled: false`` because these
    routes have no local answer to fall back to — a log page and an artifact list
    only exist upstream, and an empty body would paint as "this run has no logs".
    """
    if not is_enabled(root):
        raise _disabled()


def _disabled() -> ClientRefusal:
    """The refusal every route here answers while the consent switch is off.

    ONE CONSTRUCTION, because the sentence is the library's ``disabled`` copy and
    the code is ``not_configured``: a renderer that branches on the code must see
    the same one from a create and from a log read.
    """
    return ClientRefusal(403, CODE_NOT_CONFIGURED, _t(_wr.disabled()))


async def _account(auth: DesktopAuth) -> dict[str, str]:
    """Which Radient account this device would spend, in the plane's own vocabulary.

    THE THREE STATES ARE ``desktop_radient``'s: ``signed-out`` when nothing is
    stored, ``expired`` when the IdP has declared the grant dead (its refusal's
    reason is ``grant_invalid``, whose remedy is a re-login), and ``signed-in``
    otherwise — INCLUDING a credential that is merely unrefreshable right now,
    because someone is still signed in and telling them to sign in again would send
    them to fix a problem a retry fixes.

    ``label`` is ``credential_identity``'s answer for the row this request would
    spend, which is the one derivation the account list and the CLI's login receipt
    also use; empty when there is no row to name.
    """
    rows = [
        row
        for row in await asyncio.to_thread(auth.store.list_credentials, "radient")
        if row.credential_type == "oauth"
    ]
    if not rows:
        return {"state": "signed-out", "label": ""}
    access = await auth.store.get_oauth_access("radient")
    if access is not None:
        return {"state": "signed-in", "label": _label(rows, access.credential_id)}
    # Nothing usable, so the store cannot name the row this request would spend.
    # The plane's own diagnosis says WHICH nothing this is, and only one of its
    # reasons is a state of the ACCOUNT rather than of this moment.
    try:
        await radient_bearer(auth, classify=True, join=False)
    except HTTPException as refusal:
        expired = _grant_is_dead(refusal)
        return {"state": "expired" if expired else "signed-in", "label": _label(rows, None)}
    return {"state": "signed-in", "label": _label(rows, None)}


def _label(rows: list[Any], selected: int | None) -> str:
    """The account label of the row ``selected``, or of the only row there is.

    The SELECTED row, not the first one: with two accounts signed in, a label from
    the other one would name an account the run would not be billed to. With no
    selection and several rows there is nothing to choose between, so the label is
    empty rather than a guess.
    """
    from local_operator.providers.auth_store import credential_identity

    for row in rows:
        if selected is not None and row.id == selected:
            return credential_identity(row) or ""
    if len(rows) == 1:
        return credential_identity(rows[0]) or ""
    return ""


def _grant_is_dead(refusal: HTTPException) -> bool:
    """Whether ``radient_bearer``'s refusal is the IdP having killed the grant.

    Read off the refusal the plane already carries (``code`` in the envelope,
    ``reason`` in its details) rather than by asking the store again: one diagnosis,
    one answer.
    """
    detail = refusal.detail
    if not isinstance(detail, dict):
        return False
    details = detail.get("details")
    return (
        detail.get("code") == "radient_credential_refused"
        and isinstance(details, dict)
        and details.get("reason") == desktop_radient.REASON_GRANT_INVALID
    )


async def _pin(auth: DesktopAuth) -> int | None:
    """The Radient credential this request's calls are pinned to, or ``None``.

    ``None`` means "no usable sign-in right now" and is NOT an error of this
    function: the read routes answer it as a state (``signed-out``), and the write
    routes turn it into the plane's own refusal through :func:`_require_pin`.
    """
    access = await auth.store.get_oauth_access("radient")
    if access is None or access.credential_id is None:
        return None
    return access.credential_id


def _require_pin(credential_id: int | None) -> int:
    """``credential_id``, or the plane's own "sign in to Radient" refusal.

    ``desktop_radient._failure``'s shape is the vocabulary every Radient surface on
    this plane already answers in (409 ``radient_no_credential``), so a create with
    no sign-in reads the same as any other Radient call made without one. Its
    envelope's ``text`` is the localized sentence and its ``code`` is the machine
    one; neither is invented here.
    """
    if credential_id is not None:
        return credential_id
    raise _no_credential()


def _no_credential() -> ClientRefusal:
    """The same refusal as a value, for a route that may keep going without one.

    The per-run read answers its local record when the sign-in is unusable, and
    raises this only when there is no record either — so it needs the refusal to
    RAISE at one branch rather than to return from a pin it never resolved.
    """
    failure = desktop_radient._failure(  # noqa: SLF001 — the plane's one refusal shape
        409, "radient_no_credential", desktop_radient.NO_CREDENTIAL_MESSAGE
    )
    detail = failure.detail if isinstance(failure.detail, dict) else {}
    return ClientRefusal(
        failure.status_code,
        str(detail.get("code") or "radient_no_credential"),
        str(detail.get("text") or desktop_radient.NO_CREDENTIAL_MESSAGE),
    )


def _recorded(root: Path, identifier: str) -> records.RunRecord | None:
    """This device's own record for ``identifier``, or ``None``.

    Read through ``records.load`` directly rather than through a ``RemoteRuns``:
    the offline answer must be available to a request that could not even resolve
    a credential, and building a client to read a local file would be the long way
    round to the same list the listing uses.
    """
    return next((row for row in records.load(root) if row.run_id == identifier), None)


def _record_row(record: records.RunRecord) -> dict[str, Any]:
    """One recorded run as a row of this surface's vocabulary.

    ``run_id`` BECOMES ``id``, and is then DROPPED. The server names a run ``id``;
    a record names it ``run_id`` because the record is the library's own on-disk
    shape, which this route does not rewrite. One array holding both vocabularies
    made every renderer try ``id or run_id`` on every row — and a row that failed
    both rendered as a nameless run with no error. Nothing reads ``run_id`` here:
    the CLI reads the record's own JSON, not this route's rows.
    """
    row = {**record.to_json(), "source": "record"}
    row["id"] = row.pop("run_id")
    return row


def _networks(root: Path) -> list[dict[str, Any]]:
    """The networks a run could be BOUND to, with this device's role in each.

    ONLY ACTIVE MEMBERSHIPS, and the test for that is ``resolve_network``'s own
    predicate (``resolve.is_active_member``, called rather than restated): a run
    submitted through a network this device is no longer a member of is refused
    before a request is built, so listing that network would offer a choice that
    always fails.
    """
    from local_operator.network import store as network_store

    rows: list[dict[str, Any]] = []
    for record in network_store.list_networks(root):
        if not is_active_member(record):
            continue
        member = record.self_member()
        rows.append(
            {
                "network_id": record.network_id,
                "name": record.name,
                "role": member.role if member is not None else record.self_role,
            }
        )
    return rows


async def _capacity(root: Path, credential_id: int) -> tuple[list[dict[str, Any]], bool]:
    """The deployment's tiers, and whether it offers remote compute at all.

    ``offered`` is answered by the DEPLOYMENT, and by two facts only: the account not
    being admitted to the preview (403 — the provider's switch, which is the one
    "no" that is painted even though the user turned the feature on), and the server
    saying ``not_configured`` or reporting no ready tier. An answer this build could
    not get (an unreachable API, a dead sign-in) leaves ``offered`` true: "not
    offered" would paint a remedy that cannot work over a network problem that a
    retry fixes.

    ONE CORNER READS IMPRECISELY, AND IS ACCEPTED RATHER THAN REPAIRED. A provider row
    whose ``tier`` this build cannot name is dropped by the client's ``capacity()``, so
    a deployment whose ONLY provider is a tier newer than this build answers ``[]``
    here and ``offered`` falls to false — the renderer's "not offered by this
    deployment", where the truth is "this build predates the tier". Telling those
    apart needs a third state on the wire plus a sentence the catalogue does not carry,
    for a corner that cannot arise while the server's two literals are the two this
    build knows; the dropped literal is logged at debug by the client, which is where
    a deployment that grew a tier is found. If that changes, the fix is a machine
    field on this response (never new copy here), so no renderer has to parse it out
    of ``offered``.
    """
    async with _runs(root, credential_id) as runs:
        try:
            rows = await runs.client.capacity()
        except PreviewNotEnabled:
            return [], False
        except LoginRequired as failure:
            logger.info("remote compute: capacity read needs a login (%s)", type(failure).__name__)
            return [], True
        except RemoteComputeError as failure:
            # ``not_configured`` is the deployment saying it does not offer this at
            # all, and it arrives under either class the contract maps it to (503 →
            # ``CapacityUnavailable``, a coded 409 → ``Conflict``), so it is read by
            # CODE rather than by type: the status the answer came with is a property
            # of how the server phrased it, and the code is the fact.
            if failure.code == "not_configured":
                return [], False
            logger.info("remote compute: capacity read failed (%s)", type(failure).__name__)
            return [], True
    tiers = [_capacity_row(row) for row in rows]
    return tiers, any(row.get("ready") is True for row in tiers)


def _capacity_row(row: dict[str, Any]) -> dict[str, Any]:
    """One capacity row, narrowed to the three fields this surface publishes.

    CAPACITY IS THE ONE SERVER-SHAPED BODY THIS MODULE ANSWERS, so it is filtered
    rather than passed through (``without_secrets`` covers the run bodies): a row
    the deployment grows a field on — a rate, an internal pool name — would otherwise
    be republished the moment it appears, without anyone deciding it should be. The
    three keys are the contract's own (``{tier, ready, detail}``), and a row missing
    one simply omits it, which is what the renderer already handles.
    """
    return {key: row[key] for key in ("tier", "ready", "detail") if key in row}


async def _active_runs(root: Path, credential_id: int) -> list[dict[str, Any]]:
    """The server's runs that have not finished.

    A ROW THIS BUILD CANNOT READ IS KEPT: a run without a status is not evidence that
    it stopped, and hiding it would make work in flight vanish from the dropdown
    exactly when the contract drifts.
    """
    async with _runs(root, credential_id) as runs:
        try:
            listed = await runs.list_runs()
        except (RemoteComputeError, LoginRequired) as failure:
            logger.info("remote compute: active read failed (%s)", type(failure).__name__)
            return []
    active: list[dict[str, Any]] = []
    for run in listed:
        try:
            finished = is_terminal(run)
        except RemoteComputeError:
            finished = False
        if not finished:
            active.append(without_secrets(run))
    return active


def _hosting_model(manager: ConfigManager) -> tuple[str, str]:
    """The model pair a run is submitted with, through the product's one resolver.

    ``bootstrap.resolve_hosting_model`` is the resolver
    ``resolve_model_configuration`` delegates the pair to, and it takes the config
    manager a surface already holds: re-deriving "hosting then model from config"
    here would be a second copy of a precedence ladder four call sites share.
    """
    from local_operator.bootstrap import resolve_hosting_model

    return resolve_hosting_model(manager, None, None, None)


__all__ = [
    "ClientRefusal",
    "RunSubmission",
    "refusal_from",
    "router",
    "status_for",
]

"""The desktop's approval routes (``features.approvals``).

THE DECISION ROUTES remain thin HTTP skins over one function in
``server.utils.desktop_approvals``: the STORE does the deciding (the write-once
decision, the signature gate, the anchor provenance), and this module maps
answers onto the contract the desktop app consumes.

ADDITIVE, AND THE FEATURE KEY IS THE GATE. Every route here is new, so a
pre-approvals renderer never calls one; ``features.approvals`` is what tells a
renderer it may. The store is device-local, so a read here opens no socket and
dials no peer — the mesh relay is not in this path at all (§2.3 of the
remote-onboarding design), and neither is the operator's secret store (the
record holds a reference NAME and this path never resolves it).

THE AUTHORISATION RULE, repeated from the utils module because this is the HTTP
boundary a reader will look for it at: the desktop bearer token gates the whole
router (``require_desktop``, the app-wide dependency), and inside it the only
authority-increasing act — approve — is the SAME presence-gated signing call the
CLI's verb runs. An HTTP request cannot approve anything on its own; the OS
prompt is what decides, and a host that cannot raise one refuses with a sentence
naming the setup action.

THE REFUSAL SHAPE is the plane's ``{code, message}``: 404 for a record this
device does not hold, 409 for everything the caller can act on (a state that
refuses a decision, a missing signing surface). The additive mesh_onboard surface
also owns bounded SSH work, but never changes the decision gate or signs through
HTTP alone.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from local_operator.i18n.keys import wire_errors_desktop as _wd
from local_operator.network.types import MeshRefusal
from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_approvals import (
    ApprovalDecision,
    ApprovalList,
    CredentialApprovalAcknowledgment,
    NetworkEnrollmentAcknowledgment,
    OnboardProbe,
    OnboardRequest,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import errors, reply
from local_operator.server.utils.desktop_approvals import approval_rows, approve, deny

router = APIRouter(tags=["Desktop approvals"], dependencies=[Depends(require_desktop)])


def _root(request: Request) -> Path:
    """This backend's config root — where the device-local records live.

    Read from the app's config manager rather than the process default, the same
    way every mesh read on this plane is keyed: a backend started against a
    relocated root must read THAT machine's records, not the environment's.
    """
    return request.app.state.config_manager.config_dir


def _http_refusal(error: MeshRefusal) -> Exception:
    """Map an approval refusal onto a status, keeping the sentence verbatim.

    The status split is the plane's "who can act on this": a record this device
    does not hold is 404, and every decidable refusal — running, already
    connected, already decided, no signing surface — is 409, because the
    operator (or their next action) can act on all of them.
    """
    status = 404 if error.code in {"unknown_approval", "approval_missing"} else 409
    if error.code in {"ssh_port_conflict", "invalid_ssh_port", "unexpected_run_body"}:
        status = 422
    body = _wd.desktop_refusal(error=str(error)).envelope(code=error.code)
    outcome = getattr(error, "apply_outcome", None)
    if isinstance(outcome, dict):
        # An APPLY-phase refusal says, in this same response, whether a retry can succeed
        # (UX #3002 r2, U8b): ``{code, retryable, reason, remedy}``, the block the row's
        # ``typed_approval.apply_outcome`` and the detail read carry. Absent on every other
        # refusal, so existing clients see the body they always did.
        body["apply_outcome"] = outcome
    return HTTPException(status, body)


def _crash_facts(error: BaseException) -> tuple[str, str]:
    """The onboarding step a crash came from, and the kind of exception it was.

    WHAT THIS FIXES. The onboarding runner is run in a worker thread; anything
    that escapes it lands in ``run_onboard``'s catch-all, which used to record a
    single canned row with ``step="worker"`` — a name that is not one of
    ``onboard.STEP_NAMES`` — and no cause at all. A run that died inside
    ``join`` was then indistinguishable, from every surface except the raw JSON
    store, from a credential-helper or store-write crash, and the durable
    receipts could not say which step had been reached.

    SANITISED BY CONSTRUCTION. ``str(exc)`` is never read: an exception message
    can carry a filesystem path, a credential reference, an SSH command line or
    a remote host's output. Only an identifier-shaped class name and a step
    name from the runner's own vocabulary leave this function, and ``worker``
    stays the fallback for a crash that never came from a step (claim release,
    a store write), so the historical row is still what such a crash records.
    """
    from local_operator.network import onboard

    step = str(getattr(error, "step", "") or "")
    if step not in onboard.STEP_NAMES:
        step = "worker"
    cause = str(getattr(error, "cause_type", "") or type(error).__name__)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", cause):
        cause = "exception"
    return step, cause


async def _local(request: Request, work: Any) -> Any:
    """Run one store call off the event loop and translate its refusals.

    The store can BLOCK on a cross-process file lock (the write discipline §2.3
    gives three writers), and ``approve`` waits on a human gesture on top of
    that — neither may run on the loop that is streaming a turn.
    """
    async with errors(request):
        try:
            return await asyncio.to_thread(work)
        except MeshRefusal as error:
            raise _http_refusal(error) from None


@router.get("/v1/desktop/approvals", response_model=CRUDResponse[ApprovalList])
async def approvals(request: Request):
    """The badge read: pending and recent approval records, oldest first."""
    return reply(await _local(request, lambda: {"approvals": approval_rows(_root(request))}))


@router.post(
    "/v1/desktop/approvals/{approval_id}/approve", response_model=CRUDResponse[ApprovalDecision]
)
async def approve_approval(request: Request, approval_id: str):
    """The operator's yes: the SAME signing call the CLI's ``approve`` verb runs."""
    return reply(await _local(request, lambda: approve(_root(request), approval_id)))


@router.post(
    "/v1/desktop/approvals/{approval_id}/deny", response_model=CRUDResponse[ApprovalDecision]
)
async def deny_approval(request: Request, approval_id: str):
    """The operator's no — ordinary, write-once, the safe direction (§2.4)."""
    return reply(await _local(request, lambda: deny(_root(request), approval_id)))


@router.post("/v1/desktop/approvals/probe", response_model=CRUDResponse[Any])
async def onboard_probe(request: Request, body: OnboardProbe):
    """Observe a public SSH host key, never authenticate or resolve a credential.

    Existing 6s banner/8s keyscan bounds and host:port parser are unchanged.
    The answer is explicitly TOFU observation, NOT verified target identity.
    """
    from local_operator.network import onboard
    from local_operator.server.utils.desktop_onboard import bound_root

    await _local(request, lambda: bound_root(_root(request)))
    embedded = onboard._port_from(body.host)
    if body.port and embedded and body.port != embedded:
        raise _http_refusal(MeshRefusal("ssh_port_conflict", "The supplied SSH ports disagree"))
    port = body.port or embedded or 22
    if not 1 <= port <= 65535:
        raise _http_refusal(
            MeshRefusal("invalid_ssh_port", "Choose an SSH port between 1 and 65535")
        )
    host = body.host.partition(":")[0] if ":" in body.host else body.host
    # Bind the observed port into the exact host field the request/run consumes;
    # otherwise a frontend could accidentally file its default port instead.
    target = f"{host}:{port}"

    def work():
        observed = onboard.probe(host, user=body.user, port=port)
        return {
            "host": target,
            "user": body.user,
            "port": port,
            "host_key_fp": observed.host_key_fp,
            "banner": observed.banner[:256],
            "reachable": observed.ok,
            "detail": observed.detail[:1000],
            "identity": "observed_tofu",
        }

    return reply(await _local(request, work))


@router.post("/v1/desktop/approvals/request", response_model=CRUDResponse[Any])
async def request_onboard(request: Request, body: OnboardRequest):
    """File the SAME CLI card. No signing, login, secret lookup or implicit grant."""
    from local_operator.network.onboard_approvals import request_device
    from local_operator.server.utils.desktop_onboard import bound_root

    def work():
        root = bound_root(_root(request))
        record = request_device(
            host=body.host,
            user=body.user,
            name=body.name,
            network=body.network_id,
            credential_ref=body.credential_ref,
            host_key_fp=body.host_key_fp,
            advertise_host=body.advertise_host,
            request_id=body.request_id,
            unattended=body.unattended,
            grants=tuple(body.grant),
            network_only=True,
            role="drive",
            root=root,
            session_id="desktop",
            surface="desktop",
        )
        return {
            "approval_id": record["approval_id"],
            "state": record["state"],
            "device": record.get("device"),
            "what": record.get("what"),
            "expires_at": record["expires_at"],
        }

    return reply(await _local(request, work))


@router.get("/v1/desktop/approvals/{approval_id}", response_model=CRUDResponse[Any])
async def onboard_detail(request: Request, approval_id: str):
    """Durable local receipts; the request closing never cancels their writer."""
    from local_operator.network.credentials import desktop_approval
    from local_operator.server.utils.desktop_onboard import detail

    def work():
        answer = detail(_root(request), approval_id)
        # Re-reading a card is the "review it again" a cancelled credential prompt asks for.
        desktop_approval.refetched([approval_id])
        return answer

    return reply(await _local(request, work))


@router.post(
    "/v1/desktop/approvals/{approval_id}/run", status_code=202, response_model=CRUDResponse[Any]
)
async def run_onboard(request: Request, approval_id: str):
    """Verify signature AND acquire admission before 202; own work beyond HTTP.

    No request-body commands or second job journal: the existing approval lease
    and its receipts are durable authority. Repeating an active run gets 409.
    Only process death uses the existing stale-lease rule; shutdown drains SSH.
    """
    from local_operator.network import onboard, onboard_approvals
    from local_operator.owned_tasks import join_owned
    from local_operator.server.utils.desktop_onboard import bound_root, detail

    if await request.body():
        raise _http_refusal(
            MeshRefusal(
                "unexpected_run_body", "Run takes no body; the signed card supplies every scope"
            )
        )

    async def start():
        def admit():
            root = bound_root(_root(request))
            claim = onboard.approval_run(approval_id, root=root)
            # ``approval_run`` yields a TargetLease that holds the signed
            # admission view. The per-target claim lives inside it and is
            # released BY the run just before it publishes an outcome
            # (onboard.TargetLease), so a retry that reads "run finished" is
            # never refused on a claim the run has already stopped needing.
            return root, claim, claim.__enter__()

        root, claim, lease = await asyncio.to_thread(admit)
        tasks = getattr(request.app.state, "mesh_onboard_tasks", None)
        if tasks is None:
            tasks = request.app.state.mesh_onboard_tasks = {}

        async def worker():
            try:
                operation = asyncio.create_task(
                    asyncio.to_thread(
                        onboard._execute_admitted_approval,
                        approval_id,
                        view=lease.view,
                        root=root,
                        release_target=lease.release,
                    )
                )
                await join_owned(operation)
            except Exception as crash:
                # Never publish raw SSH/secret-store exception text. A crash is
                # recorded as failed (same card may retry), not left live forever.
                # It DOES name what died: the step it escaped from and its
                # exception class (see onboard.OnboardStepCrash). Before that,
                # this row was the only evidence a crashed run left and it said
                # neither.
                step, cause = _crash_facts(crash)
                # Same ordering as every other outcome: the target is free
                # before the terminal row becomes readable.
                lease.release()
                await asyncio.to_thread(
                    onboard_approvals.finish,
                    approval_id,
                    "failed",
                    lease.view.run_id,
                    step=step,
                    # i18n: ignore runtime diagnostic: the suffix is a step name
                    # and an exception class, never user copy
                    detail=(
                        "Setup worker did not finish; check the receipts before retrying "
                        f"(step {step}, {cause})"
                    ),
                    root=root,
                )
            finally:
                claim.__exit__(None, None, None)
                tasks.pop(approval_id, None)

        tasks[approval_id] = asyncio.create_task(worker(), name="desktop-mesh-onboard")
        return await asyncio.to_thread(detail, root, approval_id)

    async with errors(request):
        try:
            # A canceled response still finishes admission/registration. The
            # backend registry, NOT this HTTP waiter, then owns the worker.
            return reply(await join_owned(asyncio.create_task(start())))
        except MeshRefusal as error:
            raise _http_refusal(error) from None


async def drain(app: Any) -> None:
    """Join all owned runs before event-loop teardown can strand a worker lease."""
    from local_operator.owned_tasks import join_owned

    tasks = list(getattr(app.state, "mesh_onboard_tasks", {}).values())
    for task in tasks:
        await join_owned(task)


@router.post(
    "/v1/desktop/approvals/{approval_id}/approve-network-enrollment",
    response_model=CRUDResponse[ApprovalDecision],
)
async def approve_network_enrollment(
    request: Request, approval_id: str, body: NetworkEnrollmentAcknowledgment
):
    """Same human presence, but ONLY after a validated displayed new snapshot."""
    from local_operator.network import approvals as store
    from local_operator.server.utils.desktop_approvals import _decision
    from local_operator.server.utils.desktop_onboard import bound_root

    def work():
        root = bound_root(_root(request))
        record = store.approve_network_enrollment(
            approval_id,
            authority_version=body.authority_version,
            scope=body.scope,
            digest=body.request_digest,
            root=root,
        )
        return _decision(record)

    return reply(await _local(request, work))


async def _approve_operation(request: Request, approval_id: str, body: Any) -> Any:
    """THE ONE typed sign-then-apply handler behind ``/approve-operation`` and its alias.

    Every refusal before the signature is a 409 with its code (nothing happened). Once
    signed, an apply refusal is reported as ``applied: {ok: false, code}`` beside
    ``signed: {ok: true}`` -- the decision stands and ``apply-operation`` retries it. Both
    paths call this so their responses cannot drift (pinned by the parity tests).
    """
    from local_operator.network.credentials import desktop_approval

    def work():
        return desktop_approval.approve_and_apply(
            _root(request),
            approval_id,
            op=body.operation,
            request_digest=body.request_digest,
            disclosure_digest_=body.disclosure_digest,
        )

    return reply(await _local(request, work))


async def _apply_operation(request: Request, approval_id: str, body: Any) -> Any:
    """THE ONE typed apply (the "signed, not applied" retry) behind both apply paths."""
    from local_operator.network.credentials import desktop_approval

    def work():
        return desktop_approval.apply(
            _root(request),
            approval_id,
            op=body.operation,
            request_digest=body.request_digest,
            disclosure_digest_=body.disclosure_digest,
        )

    return reply(await _local(request, work))


@router.post(
    "/v1/desktop/approvals/{approval_id}/approve-credential", response_model=CRUDResponse[Any]
)
async def approve_credential(
    request: Request, approval_id: str, body: CredentialApprovalAcknowledgment
):
    """ALIAS of ``/approve-operation`` (``server.routes.desktop_credentials``) for shipped
    desktop builds (``credential_approvals``).

    Kept with its ``Literal`` body model, so an op that is not a credential op is a 422 here
    exactly as before; the handler is the same function. THE ONE INTENDED DIFFERENCE from
    ``/approve-operation`` (QA round 1, Q1): an unregistered ``operation`` is rejected by
    validation here (422), but passes the open ``OperationApprovalAcknowledgment`` there and
    the engine refuses it (409 ``credential_operation_mismatch``). Either way nothing signs;
    every registered op answers identically on both families.
    """
    return await _approve_operation(request, approval_id, body)


@router.post(
    "/v1/desktop/approvals/{approval_id}/apply-credential", response_model=CRUDResponse[Any]
)
async def apply_credential(
    request: Request, approval_id: str, body: CredentialApprovalAcknowledgment
):
    """ALIAS of ``/apply-operation`` for shipped desktop builds.

    Same 422-here / 409-there difference for an unregistered ``operation`` as
    :func:`approve_credential`.
    """
    return await _apply_operation(request, approval_id, body)

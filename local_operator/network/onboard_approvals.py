"""The ONE place the onboarding runner names slice (a)'s approval store.

WHY THIS MODULE EXISTS (remote-onboarding design §6, interface discipline).
``local_operator/network/approvals.py`` — the record's store, lifecycle and
signature rules — is slice (a) of the same workstream, built in parallel with
this one. The runner (slice (b)) is written against the FROZEN §2.2 record
schema and the §2.4 transition matrix, and every call it makes into that store
goes through the small surface below. A rebase onto slice (a)'s merged shape
touches this file and nothing else; a test drives the runner with a fake by
monkeypatching :func:`_module`.

THE FROZEN SURFACE (reconstructed from the design note; reconcile names at
rebase — the SHAPE is what the runner promised):

* ``load(approval_id)``            → the record, or ``None``.
* ``verify_signature(record)``     → re-verify the §2.4 canonical payload
  against the local operator key; raises on any mismatch. The runner calls this
  before every credentialed step (§2.4's second verification point).
* ``begin_run(approval_id, run_id)`` → the matrix's ``approved|failed →
  connecting`` transition (retry re-enters on the SAME record with a NEW
  ``run_id``). Raises when the record cannot start a run.
* ``append_receipt(approval_id, run_id, receipt)`` → the runner's append-only
  receipt trail.
* ``finish(approval_id, state, run_id)`` → ``connected`` | ``failed`` |
  ``expired``.

Records are read through :func:`_field`, so a dict-shaped or object-shaped
record loads identically (the §2.2 schema is a JSON object; slice (a) is free to
model it as a dataclass).

NOTHING HERE PRINTS, LOGS OR STORES A SECRET: the record carries
``credential_ref`` — a reference — and this adapter's whole job is to hand that
reference to the runner untouched.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from local_operator.network.types import MeshRefusal

#: Crockford's alphabet, the same one ``network/wire.py`` uses for ids: an id a
#: human might read aloud should not carry characters that sound alike.
_CROCKFORD = "0123456789abcdefghjkmnpqrstvwxyz"

#: The record states from which a run may START, as slice (a)'s ``begin_run``
#: actually enforces them (the rebase reconciled this constant with the shipped
#: store): ``approved`` opens a run and ``failed`` re-enters with a new run id
#: (§2.4's retry). A ``connecting`` record may ALSO re-enter — but only through
#: the store's staleness rule (drill finding, 2026-10-04: a runner killed
#: mid-flight wedged a card in ``connecting`` and nothing could run it again):
#: ``begin_run`` supersedes a run that stopped reporting (``run_is_stale``) and
#: refuses a run that is still in flight, so the double-entry guard and the
#: recovery path are the same, one owner.
RUNNABLE_STATES = ("approved", "failed")


def _module() -> Any:
    """The slice (a) module, resolved DYNAMICALLY at call time.

    Function-local for the package's usual reason (``lop --version`` must not
    pay for it) — and through ``importlib`` for one more: this file is
    importable (and type-checkable) before slice (a) lands, so the rebase
    changes this function's target and nothing else. A static import would make
    every ``pyright`` run fail on the module that has not merged yet, which is
    exactly the divergence this adapter exists to absorb.
    """
    import importlib

    return importlib.import_module("local_operator.network.approvals")


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    """One call into slice (a), or a typed refusal naming the missing piece.

    A missing function is an interface mismatch, not an operator problem — the
    sentence says so rather than raising ``AttributeError`` through a CLI; the
    same applies to a missing MODULE, which is the state of this tree until
    slice (a) merges.
    """
    try:
        module = _module()
    except ModuleNotFoundError:
        raise MeshRefusal(
            "approvals_interface_missing",
            "this build has no approval store (local_operator.network.approvals), so "
            "the onboarding runner and the store it was built against have drifted",
        ) from None
    function = getattr(module, name, None)
    if function is None:
        raise MeshRefusal(
            "approvals_interface_missing",
            f"this build's approval store does not provide {name}(); the onboarding "
            "runner and the store it was built against have drifted",
        )
    return function(*args, **kwargs)


def _field(record: Any, name: str, default: Any = None) -> Any:
    """One record field, from either a mapping or an object (§2.2 is a JSON object)."""
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(name, default)
    value = getattr(record, name, default)
    return default if value is None else value


def new_run_id() -> str:
    """``run_<crockford(8)>`` — the schema's spelling, minted here.

    Random rather than derived: a run id exists to distinguish two runs over the
    SAME record (retry), and a derived id would collide exactly then.
    """
    import os

    raw = os.urandom(8)
    return "run_" + "".join(_CROCKFORD[byte & 0x1F] for byte in raw)


@dataclass(frozen=True)
class ApprovalView:
    """What the runner reads off a record — normalized, never the raw object.

    ``record`` is carried so a writer (slice (a)) can take it back; everything
    else is copied so a concurrent writer cannot change what a running step
    believed mid-step.
    """

    approval_id: str
    kind: str = ""
    state: str = ""
    expires_at: float = 0.0
    device: dict[str, Any] = field(default_factory=dict)
    what: dict[str, Any] = field(default_factory=dict)
    credential_ref: dict[str, Any] = field(default_factory=dict)
    receipts: tuple[dict[str, Any], ...] = ()
    request_id: str = ""
    request_digest: str = ""
    run_id: str = ""
    record: Any = None


def _view(record: Any, *, run_id: str = "") -> ApprovalView:
    receipts = _field(record, "receipts") or ()
    return ApprovalView(
        approval_id=str(_field(record, "approval_id", "") or ""),
        kind=str(_field(record, "kind", "") or ""),
        state=str(_field(record, "state", "") or ""),
        expires_at=float(_field(record, "expires_at", 0.0) or 0.0),
        device=dict(_field(record, "device") or {}),
        what=dict(_field(record, "what") or {}),
        credential_ref=dict(_field(record, "credential_ref") or {}),
        receipts=tuple(receipts),
        request_id=str(_field(record, "request_id", "") or ""),
        request_digest=str(_field(record, "request_digest", "") or ""),
        run_id=run_id,
        record=record,
    )


def load(approval_id: str, *, root: Any = None) -> ApprovalView | None:
    """The folded record, or ``None`` when this id is not on this device.

    ``load_record`` raises its OWN ``unknown_approval`` refusal for a missing id
    (its D5 pointer sentence); every caller here branches on ``None``, so that
    one code is translated and every other refusal propagates as-is.
    """
    try:
        record = _call("load_record", approval_id, root=root)
    except MeshRefusal as refusal:
        if refusal.code == "unknown_approval":
            return None
        raise
    return None if record is None else _view(record)


def _refuse_missing(approval_id: str) -> MeshRefusal:
    return MeshRefusal(
        "approval_missing",
        f"there is no onboarding request {approval_id!r} on this machine",
    )


def _who(view: ApprovalView) -> str:
    """The device name a reader knows — never a bare record id (design D3)."""
    return str((view.record.get("device") or {}).get("name") or "this device")


def _human_expiry(expires_at: float) -> str:
    """The window's end on the reader's own clock (design round 1, D3: an epoch
    is not a sentence). Local time, minute precision, no timezone lecture."""
    if not expires_at:
        return "an unknown time"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(expires_at)))


def _denied_refusal(approval_id: str, view: ApprovalView) -> MeshRefusal:
    """The one deny sentence: the actor a D3 reader can reach, in both moments."""
    return MeshRefusal(
        "approval_denied",
        f"request {approval_id} was denied, so nothing further will run for it; "
        f"ask Local Operator to file a new onboarding request for {_who(view)} "
        "if it should still be set up",
    )


def require_step_allowed(
    approval_id: str, *, now: float | None = None, run_id: str = "", root: Any = None
) -> ApprovalView:
    """The per-step gate: load, re-verify the signature, check state + expiry.

    Called before EVERY credentialed step (§2.4: "the runner observes state and
    expiry before EVERY step and stops, recording ``denied`` with receipts
    showing where it stopped"). Every refusal is typed, and each carries the
    state that caused it, because the caller writes it into the failing receipt.
    """
    moment = time.time() if now is None else now
    view = load(approval_id, root=root)
    if view is None:
        raise _refuse_missing(approval_id)
    state = view.state
    # THE DENY OBSERVATION IS A REFUSAL IN EITHER DIRECTION: it arrives before a
    # run (the operator answered the card) or mid-run, and the reader gets the
    # D3 sentence with the actor — a forged ``state: denied`` buys a refusal,
    # never an execution, so it is safe ahead of the cryptographic half.
    if state == "denied":
        raise _denied_refusal(approval_id, view)
    # THE SIGNATURE HALF (F4, agent review round 1): slice (a)'s
    # ``verify_for_run`` re-derives the request digest and re-checks the
    # operator's signature against this machine's key, so a tampered record
    # refuses here — before any step of the matrix below can let anything run.
    # (It ALSO re-observes a mid-run deny; the branch above already answered.)
    _call("verify_for_run", approval_id, root=root)
    if state == "connected":
        raise MeshRefusal(
            "approval_already_connected",
            f"request {approval_id} already finished — {_who(view)} is connected; one "
            "approval is one onboarding, so ask Local Operator to file a new request "
            "if anything needs to change",
        )
    if state == "expired" or (view.expires_at and moment >= view.expires_at):
        raise MeshRefusal(
            "approval_expired",
            f"request {approval_id} expired at {_human_expiry(view.expires_at)}; one "
            f"approval covers one window, so ask Local Operator to file a new "
            f"onboarding request for {_who(view)}",
        )
    if state not in ("approved", "connecting"):
        # ``requested`` (not yet signed for), ``failed`` mid-step (the caller
        # transitions before running, via begin_run), or anything unknown.
        raise MeshRefusal(
            "approval_not_approved",
            f"request {approval_id} is {state or 'in an unknown state'}; nothing runs "
            "until the operator's approval is on the record",
        )
    # The run id is CARRIED, not re-minted: every receipt one run appends must
    # name the same run, and the gate is called between every pair of steps.
    return _view(view.record, run_id=run_id or view.run_id)


def begin_run(
    approval_id: str, *, now: float | None = None, root: Any = None, operation: str = ""
) -> ApprovalView:
    """Open a run: ``approved|failed → connecting``, with a new run id.

    The state transition and the id belong to slice (a)'s locked writer; the id
    VALUE is minted here so every receipt the runner appends carries the same
    one even if a rebase moves the writer.

    A ``connecting`` record goes through slice (a)'s reassessment: a run that
    stopped reporting is superseded (new run id, a ``superseded`` receipt naming
    the stopped run), a run that is still in flight refuses
    (``approval_run_in_flight``). The runner's own pid rides into the record's
    run LEASE, which is what the next staleness question is answered from —
    and it is this process's pid because this call IS the run opening.
    """
    moment = time.time() if now is None else now
    view = load(approval_id, root=root)
    if view is None:
        raise _refuse_missing(approval_id)
    if view.kind == "device_onboard_network_only" and view.state == "failed" and view.receipts:
        last = view.receipts[-1]
        if (
            isinstance(last.get("data"), dict)
            and last["data"].get("failure_kind") == "changed_facts"
        ):
            # Re-running the old card would only refile the same contradiction.
            # This hint denies work; it cannot grant authority if edited away.
            raise MeshRefusal(
                "network_enrollment_review_required",
                "Review the replacement network enrollment card; do not retry the old target facts",
            )
    # THE PRE-RUN DENY FIRST (its own sentence assumes no run yet), THEN the
    # signature half — F4's "before ANY step" includes the first one: the invite
    # edge mints a live token and writes the admit pre-answer, so a tampered
    # record must be refused before either can happen (agent review round 1,
    # Finding 1; the verify call is slice (a)'s ``verify_for_run``).
    if "credential_finite_share" in view.what and operation != "credential_finite_share":
        # Every generic runner shares this admission seam. Reusing the normal
        # signed envelope does not authorize SSH, install, invite or joining.
        raise MeshRefusal(
            "finite_apply_required",
            "Use the exact credential share --approval command; "
            "this finite request does not authorize onboarding",
        )
    if "credential_inheritance" in view.what and operation != "credential_inheritance":
        # The same seam, the same reason: a signed future-login permission is metadata-only
        # authority and its approval names NO host, install or connect scope. Without this,
        # ``approvals run`` would launch the generic SSH ``OnboardRun`` over an approved
        # inheritance record (the card carries a device block). Only the authoring flow
        # (``credentials.inheritance.apply``) may open a run on it.
        raise MeshRefusal(
            "inherit_apply_required",
            "This is a credential permission, not onboarding: it requires "
            "`lop network credential inherit ... --approval <id>` (shown on its request); "
            "no connection, install or SSH ran",
        )
    if view.state == "denied":
        raise _denied_refusal(approval_id, view)
    _call("verify_for_run", approval_id, root=root)
    # EXPIRY IS CHECKED HERE TOO, not only at the per-step gates: the first step
    # (the invite) runs on the approval's own edge, BEFORE any gate, and an
    # expired window must not mint a token or write a confirmation for it. A
    # record whose window has passed is `expired`, full stop (F2: the signature
    # covers `expires_at`; an expiry is never extended).
    if view.expires_at and moment >= view.expires_at:
        raise MeshRefusal(
            "approval_expired",
            f"request {approval_id} expired at {_human_expiry(view.expires_at)}; one "
            f"approval covers one window, so ask Local Operator to file a new "
            f"onboarding request for {_who(view)}",
        )
    if view.state not in RUNNABLE_STATES:
        # ``connecting`` is the one state with a second answer: superseded when
        # the previous run stopped reporting, refused in flight. The store owns
        # that definition (``run_is_stale``); asking it here keeps ONE owner.
        superseding = view.state == "connecting" and bool(
            _call("run_is_stale", view.record, now=moment)
        )
        if not superseding:
            # Reuse the per-step gate's refusals so "why can't this start" answers
            # identically here and at step time.
            require_step_allowed(approval_id, now=moment, root=root)
    run_id = new_run_id()
    record = _call("begin_run", approval_id, run_id=run_id, runner_pid=os.getpid(), root=root)
    if record is None:
        record = view.record
    return _view(record, run_id=run_id)


def append_receipt(
    approval_id: str, run_id: str, receipt: dict[str, Any], *, root: Any = None
) -> None:
    """Append one runner receipt, in the store's six fields (§2.2).

    The runner's own row carries one ADDITIVE key — ``data``, the structured
    payload ``digest`` binds — and the RECORD keeps the frozen six fields only
    (slice (a)'s ``append_receipt`` takes exactly those); ``data`` stays on the
    run payload the CLI returns. The rebase fold, stated here so it is not a
    silent drop.
    """
    _call(
        "append_receipt",
        approval_id,
        run_id=run_id,
        step=str(receipt.get("step") or ""),
        ok=bool(receipt.get("ok")),
        detail=str(receipt.get("detail") or ""),
        digest=str(receipt.get("digest") or ""),
        at=receipt.get("at"),
        root=root,
    )


def finish(
    approval_id: str,
    state: str,
    run_id: str,
    *,
    step: str = "",
    detail: str = "",
    root: Any = None,
    enrollment_report: dict[str, Any] | None = None,
    enrollment_failure: dict[str, Any] | None = None,
) -> None:
    """Terminate a run through the store's terminal writes.

    ``connected`` and ``failed`` are WRITES: the store lands the terminal
    receipt in the same locked mutation (``mark_connected``/``mark_failed``),
    which is why the runner must NOT append that step's receipt separately —
    the record would show the step twice. ``denied``/``expired`` are
    observations: the deny writer and the expiry fold already carry the state,
    so there is nothing to write here.
    """
    if state == "connected":
        _call(
            "mark_connected",
            approval_id,
            run_id=run_id,
            step=step or "verify",
            detail=detail,
            root=root,
            **({"enrollment_report": enrollment_report} if enrollment_report is not None else {}),
        )
    elif state == "failed":
        _call(
            "mark_failed",
            approval_id,
            run_id=run_id,
            step=step or "failed",
            detail=detail,
            root=root,
            **(
                {"enrollment_failure": enrollment_failure} if enrollment_failure is not None else {}
            ),
        )


#: The finding kinds this refile can actually apply to a fresh card — the
#: facts §3.3 step 4's halt may correct. Anything else (``install_scope``,
#: ``sudo``) is a scope/authority finding that a fresh request repeating it
#: does not fix; the caller's sentence says what must change instead (design
#: round 1, D2).
_CORRECTABLE_CHECKS = frozenset({"host_key_fp", "build", "os", "arch"})


def refile_after_contradiction(
    approval_id: str, finding: dict[str, Any], *, root: Any = None
) -> str | None:
    """File a fresh ``requested`` record superseding ``approval_id``, or ``None``.

    §3.3 step 4's halt "files a FRESH card describing the finding". The mint is
    slice (a)'s ``create_request`` (nothing here is authority-increasing — the
    fresh record is ``requested`` and the operator approves it the normal way);
    the corrected facts are the finding's ``observed`` value applied to the
    surface it names — host key, OS, architecture, build (``_CORRECTABLE_CHECKS``)
    — so the fresh card describes what the machine ACTUALLY is; every other field
    is carried from the failed record. ``None`` means the store offers no request
    surface — or the finding has nothing a fresh request can carry: a host-key
    halt whose observation failed (minting that replacement would reproduce the
    very hole the halt names), or a scope/authority finding (``install_scope``,
    ``sudo``) no fact of which this refile can apply — an identical card minted
    under a "corrected facts" sentence is a lie one family over (design round 1,
    D2). The caller's sentence names what must change instead.
    """
    module = _module()
    create = getattr(module, "create_request", None)
    new_request_id = getattr(module, "new_request_id", None)
    if create is None or new_request_id is None:
        return None
    record = _call("load_record", approval_id, root=root)
    kind = str(record.get("kind") or "")
    block_key = "machine" if kind == "local_authority" else "device"
    block = {block_key: dict(record.get(block_key) or {})}
    what = dict(record.get("what") or {})
    check = str(finding.get("check") or "")
    observed = str(finding.get("observed") or "")
    if check == "host_key_fp" and not observed:
        # A REPLACEMENT THAT CANNOT RUN IS NOT A REMEDY (drill finding,
        # 2026-10-03): the missing-fingerprint halt used to refile a fresh request
        # WITHOUT the key — reproducing the exact hole the halt names — because
        # the finding's ``observed`` was empty. Nothing is minted then; the
        # caller's sentence says a new request is needed, and the request verb
        # now OBSERVES the key when one is filed.
        return None
    if check not in _CORRECTABLE_CHECKS:
        # NOTHING CORRECTED ⇒ NOTHING MINTED (design round 1, D2): a scope or
        # authority finding has no fact this refile can apply, and the earlier
        # shape minted an IDENTICAL card while its sentence claimed "the
        # corrected facts" — the same lie-class as the keyless refile, one
        # family over. The caller's sentence names what must change instead.
        return None
    if observed:
        if check == "host_key_fp":
            block[block_key]["host_key_fp"] = observed
        elif check == "build":
            what["build"] = observed
        elif check in ("os", "arch"):
            what[check] = observed
    try:
        created = create(
            kind=kind or "device_onboard",
            request_id=str(new_request_id()),
            requested_by=dict(record.get("requested_by") or {}),
            what=what,
            credential_ref=dict(record.get("credential_ref") or {}) or None,
            root=root,
            **block,
        )
    except MeshRefusal:
        return None
    return str((created or {}).get("approval_id") or "") or None


def request_device(
    *,
    host: str,
    user: str = "",
    name: str = "",
    device_id: str = "",
    fingerprint: str = "",
    host_key_fp: str = "",
    advertise_host: str = "",
    network: str = "",
    role: str = "drive",
    credential_ref: str = "",
    request_id: str = "",
    session_id: str = "",
    surface: str = "cli",
    unattended: bool = True,
    grants: tuple[str, ...] | None = None,
    network_only: bool = False,
    expires: float = 3600,
    root: Any = None,
) -> dict[str, Any]:
    """ONE CLI/desktop card builder; frontends cannot invent a second policy.

    This only files existing device_onboard authority. It does not install an
    operator anchor, approve, resolve credentials or increase the default role.
    The observed SSH key is TOFU observation, not independent identity proof.
    The backend caller supplies its own root and requester provenance.
    """
    from local_operator.network import approvals as store
    from local_operator.network.cli import _host_key_for_request, _resolve
    from local_operator.network.identity import load_or_mint
    from local_operator.network.types import ROLE_CAPABILITIES

    if role not in ROLE_CAPABILITIES:
        raise MeshRefusal("invalid_role", "Choose an existing mesh member role")
    trio = store.local_anchor_trio(root)
    if trio is None:
        raise MeshRefusal(
            "approval_anchor_unavailable",
            "operator authority is not set up on this machine yet, and the card has to "
            "name the key it will install: ask Local Operator to set up operator authority "
            "for this machine (one approval and one admin password prompt), then file the request",
        )
    network_id = _resolve(network, root=root).network_id if network else ""
    try:
        requester = load_or_mint(root).device_id
    except (OSError, ValueError):
        requester = ""
    moment = time.time()
    what: dict[str, Any] = {
        "connect": True,
        "install": True,
        "anchor": {key: trio[key] for key in ("key_id", "spki_fp", "statement_digest")},
        "unattended": unattended,
        "grant": (
            (["approve"] if grants is None else list(grants))
            if network_only
            else (list(grants or ()) or ["approve"])
        ),
    }
    if network_id:
        what.update(network_id=network_id, role=role)
    if network_only:
        # New authority is opt-in and cryptographically separate. Legacy cards
        # keep their historical provisioning semantics and flat wire format.
        what["provisioning"] = "network_only"
        expires = store.DEFAULT_EXPIRY_S
    observed = _host_key_for_request(host, user, host_key_fp)
    device = {
        "device_id": device_id,
        "name": name or host,
        "fingerprint": fingerprint,
        "host": host,
        "user": user,
        "transport": "ssh",
        "host_key_fp": observed,
    }
    if advertise_host:
        # Signed with the target: the runner may advertise only the address
        # the operator approved, never an unsigned runner-side override. Omit
        # the empty default so an upgrade retry of an older request stays identical.
        device["advertise_host"] = advertise_host
    return store.create_request(
        kind=store.enrollment.KIND if network_only else store.KIND_DEVICE_ONBOARD,
        request_id=request_id or store.new_request_id(),
        requested_by={"session_id": session_id, "device_id": requester, "surface": surface},
        device=device,
        what=what,
        credential_ref={"kind": "ssh", "ref": credential_ref} if credential_ref else None,
        created_at=moment,
        expires_at=moment + expires,
        root=root,
    )

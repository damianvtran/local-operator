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
#: (§2.4's retry). ``connecting`` deliberately cannot re-open — a crashed run's
#: record resolves through its window, never a silent restart — so the
#: pre-merge guess that included it is gone.
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


def begin_run(approval_id: str, *, now: float | None = None, root: Any = None) -> ApprovalView:
    """Open a run: ``approved|failed → connecting``, with a new run id.

    The state transition and the id belong to slice (a)'s locked writer; the id
    VALUE is minted here so every receipt the runner appends carries the same
    one even if a rebase moves the writer.
    """
    moment = time.time() if now is None else now
    view = load(approval_id, root=root)
    if view is None:
        raise _refuse_missing(approval_id)
    # THE PRE-RUN DENY FIRST (its own sentence assumes no run yet), THEN the
    # signature half — F4's "before ANY step" includes the first one: the invite
    # edge mints a live token and writes the admit pre-answer, so a tampered
    # record must be refused before either can happen (agent review round 1,
    # Finding 1; the verify call is slice (a)'s ``verify_for_run``).
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
        # Reuse the per-step gate's refusals so "why can't this start" answers
        # identically here and at step time.
        require_step_allowed(approval_id, now=moment, root=root)
    run_id = new_run_id()
    record = _call("begin_run", approval_id, run_id=run_id, root=root)
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
        )
    elif state == "failed":
        _call(
            "mark_failed",
            approval_id,
            run_id=run_id,
            step=step or "failed",
            detail=detail,
            root=root,
        )


def refile_after_contradiction(
    approval_id: str, finding: dict[str, Any], *, root: Any = None
) -> str | None:
    """File a fresh ``requested`` record superseding ``approval_id``, or ``None``.

    §3.3 step 4's halt "files a FRESH card describing the finding". The mint is
    slice (a)'s ``create_request`` (nothing here is authority-increasing — the
    fresh record is ``requested`` and the operator approves it the normal way);
    the corrected facts are the finding's ``observed`` value applied to the
    surface it names — host key, OS, architecture, build — so the fresh card
    describes what the machine ACTUALLY is; every other field is carried from
    the failed record. ``None`` means the store offers no request surface, and
    the caller's sentence still says a new request is needed either way.
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

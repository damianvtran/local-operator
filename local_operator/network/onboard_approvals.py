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

#: The record states from which a run may START. ``connecting`` re-enters after
#: a crashed run; ``failed`` is the design's retry-eligible state (§2.4).
RUNNABLE_STATES = ("approved", "connecting", "failed")


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


def load(approval_id: str) -> ApprovalView | None:
    record = _call("load", approval_id)
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


def require_step_allowed(
    approval_id: str, *, now: float | None = None, run_id: str = ""
) -> ApprovalView:
    """The per-step gate: load, re-verify the signature, check state + expiry.

    Called before EVERY credentialed step (§2.4: "the runner observes state and
    expiry before EVERY step and stops, recording ``denied`` with receipts
    showing where it stopped"). Every refusal is typed, and each carries the
    state that caused it, because the caller writes it into the failing receipt.
    """
    moment = time.time() if now is None else now
    view = load(approval_id)
    if view is None:
        raise _refuse_missing(approval_id)
    # THE SIGNATURE FIRST (F4): a mutated record is a typed refusal before any
    # state reading — a forged ``state: approved`` must not even reach the
    # matrix, and slice (a)'s verifier owns the canonical payload.
    _call("verify_signature", view.record)
    state = view.state
    if state == "denied":
        raise MeshRefusal(
            "approval_denied",
            f"request {approval_id} was denied, so nothing further will run for it; "
            f"ask Local Operator to file a new onboarding request for {_who(view)} "
            "if it should still be set up",
        )
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


def begin_run(approval_id: str, *, now: float | None = None) -> ApprovalView:
    """Open a run: ``approved|connecting|failed → connecting``, with a new run id.

    The state transition and the id belong to slice (a)'s locked writer; the id
    VALUE is minted here so every receipt the runner appends carries the same
    one even if a rebase moves the writer.
    """
    moment = time.time() if now is None else now
    view = load(approval_id)
    if view is None:
        raise _refuse_missing(approval_id)
    # F4'S "BEFORE EVERY STEP" INCLUDES THE FIRST ONE (agent review round 1,
    # Finding 1). The invite edge runs before any gate — ``execute_approval``
    # gates every step after it — and it has REAL effects: it mints a live
    # invite and writes the admit pre-answer. A tampered record must be refused
    # here, before either; ``verify_signature`` re-derives the request digest and
    # its refusal propagates to the caller as-is (same class as a terminal
    # state's refusal).
    _call("verify_signature", view.record)
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
        require_step_allowed(approval_id, now=moment)
    run_id = new_run_id()
    record = _call("begin_run", approval_id, run_id)
    if record is None:
        record = view.record
    return _view(record, run_id=run_id)


def append_receipt(approval_id: str, run_id: str, receipt: dict[str, Any]) -> None:
    """Append one runner receipt. ``receipt`` is the §3.5 shape: step/ok/detail/at."""
    _call("append_receipt", approval_id, run_id, dict(receipt))


def finish(approval_id: str, state: str, run_id: str) -> None:
    """Terminate a run: ``connected`` (terminal), ``failed`` (retry-eligible) or
    ``expired`` (observed at step time)."""
    _call("finish", approval_id, state, run_id)


def refile_after_contradiction(approval_id: str, finding: dict[str, Any]) -> str | None:
    """File a fresh ``requested`` record superseding ``approval_id``, if slice (a)
    exposes the call.

    BEST EFFORT WITH A NAMED BOUND: §3.3 step 4 says the halt "files a FRESH card
    describing the finding (a new ``requested`` record superseding the old id)".
    The request surface belongs to slice (a); when it does not offer ``refile``
    yet, the caller's refusal sentence still states that a fresh request is
    needed, so the operator path exists either way. Returns the new id, or
    ``None`` when the refile call is unavailable — including when the whole
    store module is absent, which is the pre-slice-(a) state this tree is in.
    """
    try:
        module = _module()
    except ModuleNotFoundError:
        return None
    function = getattr(module, "refile", None)
    if function is None:
        return None
    new_id = function(approval_id, dict(finding))
    return str(new_id) if new_id else None

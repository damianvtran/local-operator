"""Desktop review -> sign -> apply for ONE digest-bound credential approval.

WHY THIS EXISTS. A credential permission (a finite-share recovery, or a standing
inheritance) could only be signed from ``lop network approvals approve``, because the
desktop card could not state what it was signing. The operator asked for the opposite
("We should be able to approve all through the UI, less technical users would not be
able to run this"), so this module gives the desktop a TYPED path that is exactly as
strict as the CLI's: the same structured disclosure (``finite_share.disclosure`` /
``inheritance.disclosure``, which the CLI also renders), the same signer
(``approvals.sign_decision``), the same write-time verification
(``approvals._record_decision``) and the same apply (``finite_share.apply`` /
``inheritance.apply``).

WHAT THE CALLER MUST PROVE, BEFORE ANY SIGNER CALL. The body echoes the row's
``request_digest`` AND ``disclosure_digest``. The server recomputes both from the
record on disk and refuses any mismatch, an expired window, a non-``requested``
state, a tampered or mixed ``what`` shape, an operator-anchor mismatch, or an
assurance level the policy below does not allow -- all before ``sign_decision``
runs. That makes the signature bind what the card SHOWED: a UI rendering a stale or
different disclosure echoes a digest the server does not recompute, and nothing is
signed. It is not proof that pixels were seen, nor a defence against arbitrary
same-uid code (see the residual below).

WHY THE GENERIC ``/approve`` STAYS REFUSED. An old desktop build shows a generic
Approve and POSTs a bodiless ``/approve``. That request carries no digest, so it
cannot prove which disclosure was on screen; signing it would be the blind approval
the CLI-only guard was built to prevent. The typed routes are additive and gated by
``features.credential_approvals``; ``can_approve_here`` stays ``False`` so the old UI
never offers a button that would sign.

ASSURANCE, AND THE OPERATOR'S FILE-ONLY DECISION. A ``presence`` key (Secure Enclave /
CNG) raises an OS prompt per signature, so a signature proves a person was there. A
``file_only`` key is a 0600 file: any process running as the operator can sign with
it, so a desktop signature on it proves only that SOMETHING with the operator's uid
and the desktop bearer pressed the route. :data:`FILE_ONLY_SIGNING` is the ONE place
that decides which operations may still be signed on such a key, and every row and
response carries the assurance level so the UI must disclose it.

RESIDUAL RISK (accepted by the operator, not papered over). The desktop bearer token
is in the daemon's environment when the app spawned it, and the daemon's session
runtimes and their tool children inherit the environment under the default
``shell_environment.mode: inherit``; a daemon that was CLAIMED instead keeps its key
in the 0600 run-directory record, which the same uid can read. Either way an agent
tool child can reach these routes. On a presence key the OS prompt still stops it;
on a file-only key a finite share can be signed without a person. The in-flight,
cancel and rate limits below bound the abuse; they do not remove it. The upgrade
offer (:func:`upgrade`) is how the operator closes it.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping

from local_operator.network import approvals
from local_operator.network.credentials import finite_share, inheritance
from local_operator.network.types import MeshRefusal
from local_operator.network.wire import canonical_json

#: Assurance levels the wire carries (machine tokens; the UI localizes the words).
ASSURANCE_PRESENCE = "presence"
ASSURANCE_FILE_ONLY = "file_only"

#: THE POLICY SWITCH: which typed credential operations a FILE-ONLY operator key may sign
#: from the desktop. One table, consulted by both the row (``signable_here``) and the
#: sign path, so the two cannot disagree.
#:
#: * Finite share -- ALLOWED. Operator decision, ask a-b492 (2026-10-26): "Both: allow
#:   now, prompt to upgrade" ("Allow with the warning on file-only keys, and the approval
#:   sheet offers a one-click 'Set up Touch ID approvals' upgrade"). The row carries the
#:   ``unproven_human`` disclosure flag and the upgrade availability for that reason.
#: * Standing inheritance -- REFUSED on the DESKTOP SURFACE (``credential_presence_required``).
#:   Operator decision, ask a-e3a5 (2026-10-26): "Finite shares only" -- a standing grant
#:   that lasts until revoked, signed by a key that cannot prove a person approved it, is a
#:   larger risk than one finite share, and the first answer was about credential SHARES.
#:   A presence-gated key may sign it. SCOPE: this switch governs the desktop routes only,
#:   NOT a host-wide rule. The CLI ``lop network approvals approve`` verb is deliberately
#:   unchanged (operator, ask a-b815, recommended option: desktop only). Loosening it is a
#:   one-line change here and a new operator decision.
FILE_ONLY_SIGNING: Mapping[str, bool] = {
    finite_share.OP: True,
    inheritance.OP: False,
}

#: The disclosure modules by operation; their ``disclosure``/``apply`` are the CLI's own.
_MODULES: Mapping[str, Any] = {finite_share.OP: finite_share, inheritance.OP: inheritance}

#: ABUSE BOUND: at most this many credential sign ATTEMPTS per window per daemon. A human
#: approving cards one at a time never comes near it; a script looping on the route does.
RATE_LIMIT = 5
RATE_WINDOW_S = 60.0

_SIGN_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_ATTEMPTS: deque[float] = deque()
#: Approvals whose presence prompt was cancelled (or otherwise did not sign). A second
#: prompt for the same card is refused until that ONE card is re-read through its detail
#: route (``GET /v1/desktop/approvals/{id}``) -- the "review it again" step. The list read
#: deliberately does NOT clear it: the desktop polls the list every 15 s, which made the
#: latch last at most one poll (agent review #3002 round 1, R2). This is a speed bump for
#: an accidental re-prompt, NOT a bound on a bearer-holding script, which can call the
#: detail route too; the per-daemon rate limit above is what bounds that.
_CANCELLED: set[str] = set()


def _refuse(code: str, message: str) -> MeshRefusal:
    return MeshRefusal(code, message)


def operation(record: Mapping[str, Any]) -> str | None:
    """The typed credential operation this record carries, or ``None`` for any other kind.

    Key PRESENCE is the boundary (as in the generic route's guard): a malformed finite
    payload, an unknown ``credential_*`` operation, or any extra ``what`` key beside the
    operation and its anchor refuses here -- never falls through to a signer.
    """
    what = record.get("what")
    if not isinstance(what, dict):
        return None
    ops = [key for key in what if isinstance(key, str) and key.startswith("credential_")]
    if not ops:
        return None
    if len(ops) == 1 and ops[0] in _MODULES and set(what) == {ops[0], "anchor"}:
        if _MODULES[ops[0]].intent(dict(record)) is not None:
            return ops[0]
    if finite_share.OP in what:
        raise _refuse(
            "finite_disclosure_invalid",
            "The finite-share request is not the exact typed shape this build can validate "
            "and disclose. Nothing was approved; request a new finite-share recovery.",
        )
    raise _refuse(
        "credential_approval_unsupported",
        "This credential request uses an operation this build cannot validate and disclose "
        "for approval. Nothing was approved.",
    )


def disclosure_digest(facts: Mapping[str, Any]) -> str:
    """``sha256:<hex>`` over the canonical JSON of the structured disclosure."""
    return "sha256:" + hashlib.sha256(canonical_json(dict(facts))).hexdigest()


def assurance() -> tuple[str | None, str]:
    """``(assurance, level)`` for THIS host's installed operator anchor.

    Read from the root-owned anchor's recorded backend (``operator_authority_level``'s
    own rule), never a fresh OS probe. Anchor-unpinned, spawn-only and unreported
    hosts answer ``None``: nothing may be signed from the desktop there.
    """
    from local_operator.operator import (
        LEVEL_OPERATOR_FILE_ONLY,
        LEVEL_OPERATOR_PRESENCE,
        authority_level_load,
    )
    from local_operator.operator.trust import is_presence_backend

    level, loaded = authority_level_load()
    # PRESENCE needs the anchor's BACKEND to be a presence backend, not only its
    # ``presence`` flag (review #3002 round 1, N1): the signer is resolved from
    # ``anchor.backend``, so the policy must read the same field. A statement whose flag
    # and backend disagree falls to file-only, the stricter answer.
    backend = loaded.anchor.backend if loaded.anchor is not None else ""
    if level == LEVEL_OPERATOR_PRESENCE and is_presence_backend(backend):
        return ASSURANCE_PRESENCE, level
    if level == LEVEL_OPERATOR_PRESENCE:
        return ASSURANCE_FILE_ONLY, level
    if level == LEVEL_OPERATOR_FILE_ONLY:
        return ASSURANCE_FILE_ONLY, level
    return None, level


def _policy_refusal(op: str, level: str | None) -> MeshRefusal | None:
    """The ONE policy decision: may ``op`` be signed here at this assurance level?"""
    if level == ASSURANCE_PRESENCE:
        return None
    if level == ASSURANCE_FILE_ONLY:
        if FILE_ONLY_SIGNING.get(op, False):
            return None
        from local_operator.network.credentials import approval_copy

        # SURFACE-ACCURATE (UX U3): it is the DESKTOP that needs Touch ID here (a-e3a5,
        # desktop surface per a-b815); the terminal still approves it (a-4ba6). The upgrade
        # is named as the fix only when this install can actually set it up.
        return _refuse(
            "credential_presence_required",
            approval_copy.inheritance_desktop_refusal(bool(upgrade().get("available"))),
        )
    return _refuse(
        "credential_authority_unavailable",
        "Operator authority is not installed and pinned on this machine, so nothing can be "
        "approved here. Nothing was approved.",
    )


def upgrade() -> dict[str, Any]:
    """Whether a presence-gated key is AVAILABLE on this host -- reported, never assumed.

    On macOS the Secure Enclave path needs the signed key-agent helper that ships only in
    the macOS wheel; a source/snapshot install has none. ``helper`` is the pre-flight's
    verdict (present, code signature and provisioning profile verify), which is what a
    row read can afford. It deliberately does NOT run the helper's ``doctor``: that
    creates and deletes a keychain item, which is a setup act, not a read. The one-click
    setup itself is a separate, admin-gated flow and is not offered by this module.
    """
    import sys

    if sys.platform != "darwin":
        return {"available": False, "helper": "unsupported_platform"}
    from local_operator.operator.macos import keyagent

    app = keyagent.helper_bundle_path()
    try:
        stat = app.stat()
    except OSError:
        return {"available": False, "helper": "absent"}
    return _verified(str(app), stat.st_mtime_ns)


_UPGRADE_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


def _verified(app: str, mtime_ns: int) -> dict[str, Any]:
    # Keyed by the bundle's mtime: ``codesign`` is a subprocess and a row list must not pay
    # it per read, while a reinstall that replaces the bundle changes the key.
    key = (app, mtime_ns)
    cached = _UPGRADE_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    from local_operator.operator.macos import keyagent

    try:
        keyagent.verify_bundle(Path(app))
        result = {"available": True, "helper": "verified"}
    except keyagent.KeyagentError as exc:
        result = {"available": False, "helper": exc.kind or "unverified"}
    except OSError:
        result = {"available": False, "helper": "unverified"}
    _UPGRADE_CACHE.clear()
    _UPGRADE_CACHE[key] = result
    return dict(result)


def row_fields(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The ``typed_approval`` block for a credential row, or ``None`` for other kinds.

    Additive: an old UI ignores it and keeps ``can_approve_here: False``. A record this
    build cannot describe carries ``signable_here: False`` and its refusal code instead of
    a disclosure, so no renderer can offer a typed Approve for it either.
    """
    try:
        op = operation(record)
    except MeshRefusal as exc:
        return {"action": "approve-credential", "signable_here": False, "refusal": exc.code}
    if op is None:
        return None
    level, _ = assurance()
    block: dict[str, Any] = {
        "action": "approve-credential",
        "operation": op,
        "request_digest": str(record.get("request_digest") or ""),
        "assurance": level,
        # The UI MUST render this disclosure when true (operator decision a-b492): a
        # file-only signature cannot prove a person approved.
        "unproven_human": level == ASSURANCE_FILE_ONLY,
    }
    try:
        facts = _MODULES[op].disclosure(dict(record))
    except MeshRefusal as exc:
        block.update(signable_here=False, refusal=exc.code)
        return block
    block.update(disclosure=facts, disclosure_digest=disclosure_digest(facts))
    refusal = _policy_refusal(op, level)
    block["signable_here"] = refusal is None and record.get("state") == approvals.STATE_REQUESTED
    if refusal is not None:
        block["refusal"] = refusal.code
        # The sentence beside the code, so the row and the typed route say the same thing
        # (UX U3: a row whose only remedy pointed at a terminal contradicted the refusal).
        block["refusal_message"] = str(refusal)
    from local_operator.network.credentials import approval_copy

    # A FAILED card carries why it stopped, whether the same card can be applied again,
    # and the next step -- read from its durable receipt, so the list and the detail read
    # still say it after a reload or a daemon restart (UX U2). ``retryable`` is the only
    # signal a renderer may use to offer "Try again".
    failure = approval_copy.last_failure(dict(record))
    if failure is not None:
        block["apply_outcome"] = failure
    if level != ASSURANCE_PRESENCE:
        block["upgrade"] = upgrade()
    return block


def refetched(approval_ids: Any) -> None:
    """A card was re-read: a cancelled prompt for it may be raised again."""
    with _STATE_LOCK:
        for approval_id in approval_ids:
            _CANCELLED.discard(str(approval_id))


def _admit(approval_id: str) -> None:
    with _STATE_LOCK:
        if approval_id in _CANCELLED:
            raise _refuse(
                "credential_prompt_cancelled",
                "The last approval prompt for this card was cancelled or did not sign. "
                "Review the card again before approving.",
            )
        now = time.monotonic()
        while _ATTEMPTS and now - _ATTEMPTS[0] > RATE_WINDOW_S:
            _ATTEMPTS.popleft()
        if len(_ATTEMPTS) >= RATE_LIMIT:
            raise _refuse(
                "credential_sign_rate_limited",
                "Too many credential approvals were attempted in the last minute. Nothing "
                "was approved; wait a minute and review the card again.",
            )
        _ATTEMPTS.append(now)


def _validated(
    root: Path, approval_id: str, *, op: str, request_digest: str, disclosure_digest_: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every pre-signer check, in the order that names the TRUE refusal first."""
    record = approvals._load_raw(approval_id, root)
    found = operation(record)
    if found is None or found != op:
        raise _refuse(
            "credential_operation_mismatch",
            "This card is not the kind of credential approval that was reviewed. Nothing was "
            "approved.",
        )
    if approvals.request_digest(record) != record.get("request_digest"):
        raise _refuse(
            "approval_record_tampered",
            "This approval no longer matches the digest it was filed with. Nothing was "
            "approved; file a new request.",
        )
    if request_digest != record.get("request_digest"):
        raise _refuse(
            "credential_request_changed",
            "This card changed since it was reviewed. Nothing was approved; review it again.",
        )
    facts = _MODULES[op].disclosure(dict(record))
    if disclosure_digest_ != disclosure_digest(facts):
        raise _refuse(
            "credential_disclosure_changed",
            "What this card discloses changed since it was reviewed. Nothing was approved; "
            "review it again.",
        )
    return record, facts


def _require_requested(record: Mapping[str, Any]) -> None:
    expires = record.get("expires_at")
    if record.get("state") == approvals.STATE_EXPIRED or (
        isinstance(expires, (int, float)) and expires and time.time() >= float(expires)
    ):
        raise _refuse(
            "approval_expired",
            "This approval's window has passed; nothing was approved. File a new request.",
        )
    if record.get("state") != approvals.STATE_REQUESTED:
        raise _refuse(
            "approval_decision_conflict",
            f"This card is {record.get('state')!r}, not awaiting approval. Nothing was approved.",
        )


def _require_anchor(record: Mapping[str, Any], root: Path) -> dict[str, Any]:
    trio = approvals.local_anchor_trio(root)
    anchor = (record.get("what") or {}).get("anchor")
    if (
        trio is None
        or not isinstance(anchor, dict)
        or any(anchor.get(key) != trio[key] for key in ("key_id", "spki_fp", "statement_digest"))
    ):
        raise _refuse(
            "approval_anchor_mismatch",
            "This card was filed for a different operator key than this machine's. Nothing "
            "was approved; file a new request.",
        )
    return trio


def _audit(event: str, approval_id: str, root: Path, detail: dict[str, Any]) -> None:
    approvals._audit(event, actor="self", subject=approval_id, root=root, detail=detail)


def approve(
    root: Path,
    approval_id: str,
    *,
    op: str,
    request_digest: str,
    disclosure_digest_: str,
) -> dict[str, Any]:
    """Sign ONE reviewed card. Every refusal fires before the signer runs, and is audited.

    Returns ``{approval_id, state, operation, assurance, signature:{key_id}}``.

    AUDITED REFUSALS (review #3002 round 1, R3/Q1): every refusal on this path -- in
    flight, rate-limited, cancelled prompt, signer failure, digest/disclosure/operation
    mismatch, expiry, state, anchor, policy -- writes ``credential.desktop_sign_refused``
    with its code. The rate-limit, in-flight and cancel rows are the trace a prompt-bombing
    loop leaves, which is what an incident review needs to see.
    """
    from local_operator.network import enrollment

    root = enrollment.bound_profile(root)
    try:
        return _approve_locked(
            root,
            approval_id,
            op=op,
            request_digest=request_digest,
            disclosure_digest_=disclosure_digest_,
        )
    except MeshRefusal as refusal:
        try:
            authority_level = assurance()[1]
        except Exception:  # noqa: BLE001 -- the refusal must reach the caller regardless
            authority_level = ""
        _audit(
            "credential.desktop_sign_refused",
            approval_id,
            root,
            {
                "surface": "desktop",
                "operation": op,
                "code": refusal.code,
                "authority_level": authority_level,
            },
        )
        raise


def _approve_locked(
    root: Path,
    approval_id: str,
    *,
    op: str,
    request_digest: str,
    disclosure_digest_: str,
) -> dict[str, Any]:
    if not _SIGN_LOCK.acquire(blocking=False):
        # ONE credential sign in flight per daemon: a second request cannot stack a second
        # OS prompt behind the first, where a person cannot tell which card it is for.
        raise _refuse(
            "credential_sign_in_flight",
            "Another credential approval is waiting for confirmation. Finish or cancel it "
            "first; nothing was approved.",
        )
    try:
        record, facts = _validated(
            root,
            approval_id,
            op=op,
            request_digest=request_digest,
            disclosure_digest_=disclosure_digest_,
        )
        _require_requested(record)
        _require_anchor(record, root)
        level, _ = assurance()
        refusal = _policy_refusal(op, level)
        if refusal is not None:
            raise refusal
        _admit(approval_id)
        decided_at = time.time()
        try:
            signature = approvals.sign_decision(
                kind=record["kind"],
                request_id=record["request_id"],
                request_digest=record["request_digest"],
                decided_at=decided_at,
                config_root=root,
            )
        except MeshRefusal:
            with _STATE_LOCK:
                _CANCELLED.add(approval_id)
            raise
        decided = approvals._record_decision(
            approval_id,
            decision="approve",
            decided_at=decided_at,
            signature_hex=signature,
            root=root,
            expected_digest=request_digest,
            changed=_refuse(
                "credential_request_changed",
                "This card changed while it was being approved. Nothing was approved; "
                "review it again.",
            ),
        )
        key_id = str((decided.get("signature") or {}).get("key_id") or "")
        # EVERY desktop credential signature is audited with its surface and assurance, so
        # an incident review can separate "a person pressed Touch ID" from "the file key
        # signed" without reconstructing it from the anchor's history.
        _audit(
            "credential.desktop_signed",
            approval_id,
            root,
            {
                "surface": "desktop",
                "assurance": level,
                "operation": op,
                "key_id": key_id,
                "disclosure_digest": disclosure_digest(facts),
            },
        )
        return {
            "approval_id": approval_id,
            "state": decided.get("state"),
            "operation": op,
            "assurance": level,
            "signature": {"key_id": key_id},
        }
    finally:
        _SIGN_LOCK.release()


def _expected(op: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """The apply ``expected`` block, from the SIGNED record (the card, not a command line)."""
    data = _MODULES[op].intent(dict(record))
    if op == finite_share.OP:
        return {
            name: data[name]
            for name in ("owner_device", "network_id", "recipient_device", "key", "scope")
        }
    parsed = inheritance._parse(dict(record))
    return {
        "owner_device": data["owner_device"],
        "network_id": data["network_id"],
        "recipient_device": data["recipient_device"],
        "provider": parsed["provider"],
        "account": parsed["account"],
        "org": parsed["org"],
    }


def apply(
    root: Path,
    approval_id: str,
    *,
    op: str,
    request_digest: str,
    disclosure_digest_: str,
) -> dict[str, Any]:
    """Apply ONE signed card in-process (the CLI's own apply). Retryable from ``failed``.

    This is also the RETRY route's body: refusals raise (409 with the code) rather than
    becoming a phase, because nothing was signed in this call. The digests are re-checked
    so a retry applies the card that was reviewed; the signature, decision and window are
    re-verified by the apply's own ``begin_run``. A replay of a ``connected`` card refuses
    ``approval_already_connected`` there.
    """
    from local_operator.network import enrollment

    root = enrollment.bound_profile(root)
    try:
        return _apply_checked(
            root,
            approval_id,
            op=op,
            request_digest=request_digest,
            disclosure_digest_=disclosure_digest_,
        )
    except MeshRefusal as refusal:
        # A signed-but-not-applied card is the state an operator asks about; the refusal
        # code names why (review #3002 round 1, Q1).
        _audit(
            "credential.desktop_apply_refused",
            approval_id,
            root,
            {"surface": "desktop", "operation": op, "code": refusal.code},
        )
        raise


def _apply_checked(
    root: Path,
    approval_id: str,
    *,
    op: str,
    request_digest: str,
    disclosure_digest_: str,
) -> dict[str, Any]:
    record, _ = _validated(
        root,
        approval_id,
        op=op,
        request_digest=request_digest,
        disclosure_digest_=disclosure_digest_,
    )
    if record.get("state") == approvals.STATE_REQUESTED:
        # Named here rather than left to ``begin_run``: its refusal for an unsigned card is
        # "this approval carries no decision signature", which the inheritance apply labels
        # as tampering -- the wrong story for a card that simply was not approved yet.
        raise _refuse(
            "approval_not_approved",
            "This card has not been approved yet; nothing was applied. Approve it first.",
        )
    # The apply's own result (placement entry / provisioned row) stays server-side: the UI
    # needs the record's state, and the disclosure already names what was granted.
    _MODULES[op].apply(approval_id, _expected(op, record), root)
    _audit("credential.desktop_applied", approval_id, root, {"surface": "desktop", "operation": op})
    state = approvals.load_record(approval_id, root=root).get("state")
    return {
        "approval_id": approval_id,
        "operation": op,
        "state": state,
        "applied": {"ok": True, "code": ""},
    }


def approve_and_apply(
    root: Path,
    approval_id: str,
    *,
    op: str,
    request_digest: str,
    disclosure_digest_: str,
) -> dict[str, Any]:
    """Sign, then apply; report each phase so the UI can show "signed, not applied".

    A refusal BEFORE the signature propagates (nothing happened). Once signed, an apply
    refusal is a phase outcome, not an error: the decision stands and ``apply`` can be
    retried from the ``failed`` (or still ``approved``) record.
    """
    signed = approve(
        root,
        approval_id,
        op=op,
        request_digest=request_digest,
        disclosure_digest_=disclosure_digest_,
    )
    phases: dict[str, Any] = {
        "approval_id": approval_id,
        "operation": op,
        "assurance": signed["assurance"],
        "signed": {"ok": True, "key_id": signed["signature"]["key_id"]},
    }
    try:
        applied = apply(
            root,
            approval_id,
            op=op,
            request_digest=request_digest,
            disclosure_digest_=disclosure_digest_,
        )
        phases["applied"] = {"ok": True, "code": ""}
        phases["state"] = applied["state"]
    except MeshRefusal as exc:
        from local_operator.network.credentials import approval_copy

        # ``retryable`` decides whether the UI may offer "Try again" (UX U2): only a
        # refusal after which the SAME signed card can succeed. ``reason``/``remedy`` are
        # desktop-safe sentences; ``message`` stays the CLI-shaped text for compatibility.
        phases["applied"] = {"ok": False, "message": str(exc), **approval_copy.outcome(exc.code)}
        phases["state"] = approvals.load_record(approval_id, root=root).get("state")
    return phases

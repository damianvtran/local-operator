"""Project a SIGNED, EXECUTED future-login approval into brokered access.

WHY THIS MODULE EXISTS. ``consent.provision_from_run`` records executed operator
permission ("this recipient may borrow my future native OAuth logins") and
``consent.verified_permission`` can re-verify it, but nothing turned that permission
into a placement holder: a login made AFTER the approval stayed unusable on the
recipient until somebody ran ``credential share`` and the recipient pulled. This is
the one writer that closes exactly that gap, for native OAuth providers only.

WHAT IT DOES, ON THE OWNER, INSIDE THE EXISTING 15 s LIFECYCLE CADENCE (no thread,
no queue, no dialer of its own — ``lifecycle.lifecycle_tick_step`` calls it):

1. :func:`project_for` — for one ACTIVE recipient with an enabled approval, read the
   owner's native OAuth inventory (``auth.db`` opened read-only, NO credential
   material) and, for each provider the recipient does not already hold, add a holder
   row iff ``consent.verified_permission`` accepts at least one eligible native
   login. The holder carries ``granted_by="consent:<approval ref>"`` (the existing
   field; no schema change) so the serve path can tell a DERIVED grant — which is
   account-scoped — from an explicit share, which is not and is left untouched.
2. :func:`deliver` — push the owner's metadata-only rows for derived holders over the
   already authenticated, already negotiated link, so the recipient's RUNNING session
   sees the holder on its next request (``MeshCredentialClient.placement`` re-stats
   the file). The recipient applies it through :func:`accept_push`.
3. :func:`serve_gate` / :func:`derived_mint_check` — used by ``owner.grant``. A derived
   holder is only served from a native row whose ACTUAL account the signed scope
   covers. That is decided BEFORE any refresh (so a mismatch costs zero IdP POSTs and
   the cascade is told which rows it may not pick) and again at mint under the
   consent lock (so a revoke or an account swap that lands during a refresh wins).

WHAT IT NEVER DOES (each is a decision, not an omission):

* It never writes a capability. ``broker_credential`` — the dial authority — must
  already be on the owner's own row AND on the owner's row for the recipient;
  otherwise the outcome is ``notready:<remedy>`` and NOTHING is mutated. A signed
  approval of *credentials* is not a signed approval of *dial authority*, and there is
  no approval UX for the latter yet; inventing it here would be a silent grant.
* It never touches an explicit holder (finite or legacy), a deny, ``consent.json``,
  ``lifecycle.json`` or ``auth.db``.
* It never projects non-OAuth classes (MCP, ``secret:*``, GitHub, device-bound,
  static API keys): those stay explicit shares, and a derived holder is never served
  a static-key row either (see :func:`serve_gate`).
* The owner keeps the refresh token; the recipient only receives the short-lived
  access the unchanged ``grant`` op mints.

LOCK ORDER is the established ``consent -> placement``. Verification happens inside the
consent lock but outside the placement lock; ``grant_key_set``'s in-lock
``consent.check_borrow`` is what lets a concurrent ``restrict_and_revoke`` (which
writes its deny under the same consent lock) win.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from local_operator.network import store
from local_operator.network.credentials import consent, placement
from local_operator.network.types import MeshRefusal

logger = logging.getLogger(__name__)

#: ``Holder.granted_by`` prefix of a holder derived from a signed approval.
DERIVED_PREFIX = "consent:"

#: Marker on a ``net_broker kind=placement`` frame that is this module's push. A frame
#: without it keeps the pre-existing pull/merge semantics byte for byte.
PUSH_ORIGIN = "consent_reconcile"

#: Bounds on a pushed document: the recipient reads a peer's bytes.
MAX_PUSH_CREDENTIALS = 32
MAX_PUSH_BYTES = 65536
MAX_PUSH_HOLDERS = 8

#: A recipient that refused the push with a TYPED ``not_implemented`` / ``not_authorised``
#: answer (a build without the handler, or its row for us lacks ``broker_credential``) is
#: not asked again for this long: only an operator upgrade or grant changes the answer.
#: A TIMEOUT, a dead send or a transient ``busy`` is NOT a refusal and never starts this
#: window — it is retried on the next tick (see :func:`_classify_reply`).
REFUSED_RETRY_S = 300.0

#: Any other typed answer (the receiver's own validation refusal) is re-sent no sooner than
#: this, unless the pending rows changed: a malformed push repeated every 15 s helps nobody.
REJECTED_RETRY_S = 60.0

#: A memoised "nothing changed" verdict is re-derived at least this often, so drift the
#: memo key cannot see (operator anchor change, approval record edited) still heals.
MEMO_TTL_S = 300.0

#: The three codes a derived-holder refusal reaches the borrower as. Every other
#: consent refusal collapses into ``consent_withdrawn`` so the closed wire set stays small.
CODE_MISMATCH = "scope_mismatch"
CODE_UNKNOWN = "scope_unknown"
CODE_WITHDRAWN = "consent_withdrawn"
#: A momentary inability to READ what the decision needs (``auth.db`` locked, approval
#: store unreadable). Retryable and said so — NOT a withdrawal, so the borrower is not
#: told to re-approve on the owner for a blip.
CODE_UNVERIFIABLE = "consent_unverifiable"
SCOPE_CODES = frozenset({CODE_MISMATCH, CODE_UNKNOWN, CODE_WITHDRAWN, CODE_UNVERIFIABLE})

#: ``consent`` / ``approvals`` refusals that mean the operator's authority is genuinely
#: absent, denied, mismatched or tampered. Every OTHER refusal code reaching
#: :func:`wire_code` is an unreadable-store class and maps to :data:`CODE_UNVERIFIABLE`.
_WITHDRAWAL_CODES = frozenset(
    {
        "consent_not_enabled",
        "consent_subset_refused",
        "consent_execution_missing",
        "consent_source_mismatch",
        "consent_subject_mismatch",
        "consent_origin_refused",
        "consent_recipient_inactive",
        "consent_action_refused",
        "approval_denied",
        "approval_record_tampered",
    }
)

_MEMO_ATTR = "_mesh_credential_reconcile_memo"
_STATUS_ATTR = "_mesh_credential_reconcile_status"
_DELIVERY_ATTR = "_mesh_credential_reconcile_delivery"
_PUSHED_ATTR = "_mesh_credential_reconcile_pushed"
_AUDITED_ATTR = "_mesh_credential_reconcile_audited"


class NativeRow(NamedTuple):
    """One enabled native row's METADATA. Never a token, never a hash of one."""

    row_id: int
    marker: str
    context: dict[str, str]
    copied: bool
    oauth: bool


@dataclass
class ReconcileResult:
    outcome: str
    projected: list[str] = field(default_factory=list)
    refused: dict[str, str] = field(default_factory=dict)


def is_derived(holder: Any) -> bool:
    return holder is not None and str(getattr(holder, "granted_by", "") or "").startswith(
        DERIVED_PREFIX
    )


def wire_code(code: str) -> str:
    """The borrower-facing code for a consent refusal (see :data:`SCOPE_CODES`).

    Scope codes and a real withdrawal keep their meaning; anything else is an
    unreadable store, which must read as retryable rather than as a decision.
    """
    if code in (CODE_MISMATCH, CODE_UNKNOWN, CODE_WITHDRAWN, "not_a_holder", CODE_UNVERIFIABLE):
        return code
    return CODE_WITHDRAWN if code in _WITHDRAWAL_CODES else CODE_UNVERIFIABLE


def push_negotiated(link: Any) -> bool:
    """Both ends of THIS link advertised ``credential-placement-push-v1``.

    The lifecycle capability is NOT enough: it shipped in v0.71.3 without a handler for
    the push, so a peer that negotiated only that string would answer the frame with a
    not-implemented refusal (no broker) or merge it through the legacy, unvalidated pull
    path (broker present). Raw peer advertisement alone is not negotiation either — the
    LOCAL set must carry it too (the ``lifecycle.negotiated`` rule).
    """
    from local_operator.network import wire

    return wire.CREDENTIAL_PLACEMENT_PUSH_V1 in getattr(
        link, "local_capabilities", ()
    ) and wire.CREDENTIAL_PLACEMENT_PUSH_V1 in getattr(link, "capabilities", ())


def _provider_ok(provider: str) -> bool:
    from local_operator.network.credentials.github import GITHUB_KEY
    from local_operator.network.credentials.types import DEVICE_BOUND_PROVIDERS

    return (
        consent.PROVIDER.fullmatch(provider) is not None
        and provider != GITHUB_KEY
        and provider != "mcp-oauth"
        and provider not in DEVICE_BOUND_PROVIDERS
    )


def inventory(root: Path) -> dict[str, list[NativeRow]] | None:
    """Enabled native rows by provider, from one read-only snapshot; ``None`` = unknown."""
    from local_operator.providers.auth_store import AuthStore

    rows = AuthStore.native_oauth_inventory(Path(root) / "auth.db")
    if rows is None:
        return None
    result: dict[str, list[NativeRow]] = {}
    for provider, row_id, marker, account, email, org, mesh_origin, ctype in rows:
        context = {
            name: str(value)
            for name, value in (("account_id", account), ("email", email), ("org_id", org))
            if value
        }
        result.setdefault(str(provider), []).append(
            NativeRow(
                int(row_id),
                str(marker or ""),
                context,
                mesh_origin is not None,
                ctype == "oauth",
            )
        )
    return result


def _verify(
    network_id: str, owner: str, recipient: str, key: str, rows: list[NativeRow], root: Path
) -> tuple[list[NativeRow], str]:
    """``(permitted native rows, first refusal code)`` for ``key`` — pure reads."""
    native = [r for r in rows if r.oauth and not r.copied]
    if not native:
        return [], CODE_UNKNOWN if rows else ""
    if any(not r.marker for r in native):
        # A pre-stamp / ambiguous login is UNKNOWN, not provisioned (lifecycle rule).
        return [], CODE_UNKNOWN
    permitted: list[NativeRow] = []
    refusal = ""
    for row in native:
        try:
            consent.verified_permission(network_id, owner, recipient, key, row.context, root)
        except (MeshRefusal, OSError) as exc:
            code = getattr(exc, "code", CODE_UNVERIFIABLE)
            if isinstance(exc, OSError):
                logger.debug("credential reconcile: approval read failed", exc_info=True)
            refusal = refusal or wire_code(code)
            if code == CODE_MISMATCH:
                refusal = CODE_MISMATCH
            continue
        permitted.append(row)
    return permitted, "" if permitted else refusal


def serve_gate(
    network_id: str, owner: str, recipient: str, key: str, root: Path | None
) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    """``(permitted row ids, row ids the cascade must not pick, refusal code)``.

    Called by ``owner.grant`` for a DERIVED holder BEFORE the refresh, from metadata
    alone. Every enabled row of the provider that the signed scope does not cover —
    another account, a borrowed copy, a static-key row — is excluded from the
    cascade, so a denied account is never refreshed and never served. ``code`` is set
    only when NOTHING is permitted yet something exists to be refused.
    """
    inv = inventory(root) if root is not None else None
    if inv is None or root is None:
        # An unreadable ``auth.db`` is a momentary read problem, not "sign in again".
        return (), (), CODE_UNVERIFIABLE
    rows = inv.get(key, [])
    permitted, code = _verify(network_id, owner, recipient, key, rows, root)
    allowed = {r.row_id for r in permitted}
    return (
        tuple(sorted(allowed)),
        tuple(sorted(r.row_id for r in rows if r.row_id not in allowed)),
        code if rows else "",
    )


def derived_mint_check(
    network_id: str,
    owner: str,
    recipient: str,
    key: str,
    identity: Mapping[str, Any],
    root: Path | None,
    *,
    credential_id: int,
    kind: str,
) -> None:
    """Final check, under the consent lock, on what the resolve ACTUALLY picked.

    Two independent facts must hold: the identity the resolve used is covered by the
    signed scope, AND the row id it used is one :func:`serve_gate` permits now. The
    second closes the case where the cascade fell through to a tier with no row or
    no identity (an env/static key): an empty identity must never read as "scope has
    no constraints, so anything goes" for a derived holder.
    """
    try:
        if kind != "oauth" or credential_id <= 0:
            raise MeshRefusal(CODE_UNKNOWN, "a derived grant serves native OAuth rows only")
        consent.verified_permission(network_id, owner, recipient, key, dict(identity), root)
        allowed, _excluded, code = serve_gate(network_id, owner, recipient, key, root)
        if credential_id not in allowed:
            raise MeshRefusal(code or CODE_MISMATCH, "the login served is not covered")
    except (MeshRefusal, OSError) as exc:
        if isinstance(exc, OSError):
            logger.debug("credential reconcile: mint authority read failed", exc_info=True)
        raise MeshRefusal(
            wire_code(getattr(exc, "code", CODE_UNVERIFIABLE)),
            "the signed approval could not be verified for this login",
        ) from exc


# ---------------------------------------------------------------------------
# Owner: projection
# ---------------------------------------------------------------------------


def _set_status(server: Any, device_id: str, outcome: str) -> str:
    status = getattr(server, _STATUS_ATTR, None)
    if not isinstance(status, dict):
        status = {}
        setattr(server, _STATUS_ATTR, status)
    if status.get(device_id) != outcome and outcome.startswith("notready"):
        logger.info("credential reconcile for %s: %s", device_id, outcome)
    status[device_id] = outcome
    return outcome


def status_for(server: Any, device_id: str) -> str:
    """The owner-side reading for ``device_id``: the projection verdict, and — when a
    delivery is stuck — the delivery verdict with its remedy, so a skip is never silent."""
    status = getattr(server, _STATUS_ATTR, None)
    base = str(status.get(device_id, "")) if isinstance(status, dict) else ""
    delivery = getattr(server, _DELIVERY_ATTR, None)
    note = str(delivery.get(device_id, "")) if isinstance(delivery, dict) else ""
    return f"{base} | delivery={note}" if note else base


def _set_delivery(
    server: Any, network_id: str, self_device: str, device_id: str, outcome: str, remedy: str = ""
) -> str:
    """Record the push outcome. A stuck one is named once in the log and the audit trail."""
    table = getattr(server, _DELIVERY_ATTR, None)
    if not isinstance(table, dict):
        table = {}
        setattr(server, _DELIVERY_ATTR, table)
    stuck = outcome.startswith(("skipped", "refused"))
    previous = table.get(device_id)
    note = f"{outcome} ({remedy})" if stuck and remedy else outcome if stuck else ""
    if outcome == "failed":
        note = "failed:link_or_owner_unavailable (retrying next tick; check the peer connection)"
    if note:
        table[device_id] = note
    else:
        table.pop(device_id, None)
    if stuck and previous != note:
        logger.info("credential reconcile push to %s: %s", device_id, note)
        _audit_one(
            server,
            "credential.permission_undelivered",
            network_id,
            self_device,
            device_id,
            code=outcome.split(":", 1)[-1][:64],
        )
    return outcome


def _record_for(root: Path, device_id: str) -> Any:
    for record in store.list_networks(root):
        member = record.member(device_id)
        if member is not None and member.active and device_id != record.self_device_id:
            return record
    return None


def project_for(server: Any, device_id: str) -> ReconcileResult:
    """One recipient's reconcile pass. Never raises for a refusal; mutates only by adding
    derived holders, or removing its own when approval is absent (see module docstring)."""
    root = getattr(server, "root", None)
    self_device = str(getattr(getattr(server, "identity", None), "device_id", "") or "")
    if root is None or not self_device or not device_id:
        return ReconcileResult(_set_status(server, device_id, "idle"))
    root = Path(root)
    record = _record_for(root, device_id)
    if record is None:
        return ReconcileResult(_set_status(server, device_id, "idle"))
    network_id = str(record.network_id)
    # Cheap gate: no consent file => nothing can be projected. The only work left is to
    # take back a derived holder whose approval has since disappeared; with no placement
    # file either, that is one stat and out and nothing is read or written.
    if not placement.placement_path(network_id, root).with_name("consent.json").exists():
        _purge_derived(server, network_id, self_device, device_id, root, None)
        return ReconcileResult(_set_status(server, device_id, "idle"))
    try:
        authority = consent.ConsentDocument(network_id, self_device, root)
        policy = authority.load()
    except (MeshRefusal, OSError):
        logger.debug("credential reconcile: consent read failed", exc_info=True)
        # Unreadable authority is NOT "no consent": say so, retry next tick, and never
        # memoise it. (Derived holders already on disk stay fail-closed at the serve gate.)
        return ReconcileResult(
            _set_status(
                server,
                device_id,
                "notready:consent_unreadable (retrying each tick; if it persists, inspect "
                "consent.json under this device's network credentials directory)",
            )
        )
    rec = policy["recipients"].get(device_id)
    if rec is None or rec["enabled"] is not True or "approval" not in rec:
        # The approval is gone or disabled: take back ONLY the holders this module
        # derived from it (explicit finite shares are left exactly as they are).
        _purge_derived(server, network_id, self_device, device_id, root, authority)
        return ReconcileResult(_set_status(server, device_id, "idle"))
    # TRANSPORT CAPABILITIES ARE PRECONDITIONS, NEVER OUTPUTS. Read both rows; write none.
    if "broker_credential" not in record.self_capabilities:
        return ReconcileResult(
            _set_status(
                server,
                device_id,
                "notready:owner_lacks_broker_credential "
                f"(on this device: lop network member grant {network_id} {device_id} "
                "broker_credential is NOT enough; this device's own row needs it)",
            )
        )
    member = record.member(device_id)
    if member is None or "broker_credential" not in member.capabilities:
        return ReconcileResult(
            _set_status(
                server,
                device_id,
                f"notready:recipient_lacks_broker_credential (grant it: lop network member "
                f"grant {network_id} {device_id} broker_credential)",
            )
        )
    inv = inventory(root)
    if inv is None:
        # Unreadable / locked / not-yet-created auth.db: retry, never a decision.
        return ReconcileResult(
            _set_status(
                server,
                device_id,
                "notready:inventory_unreadable (retrying each tick; if it persists, check "
                "this device's auth.db is readable)",
            )
        )
    document = placement.PlacementDocument.load(network_id, root, self_device=self_device)
    from local_operator.network.credentials.sync import MARK_LOCAL_ONLY, SyncState

    marks = SyncState.load(network_id, root)
    # An explicit `local-only` mark is the operator's kill switch: the serve path refuses
    # such a key, so advertising a holder for it would promise what can never be served.
    local_only = frozenset(p for p in inv if marks.mark_for(p) == MARK_LOCAL_ONLY)
    candidates = sorted(p for p in inv if _provider_ok(p) and p not in local_only)
    held = tuple(
        (p, document.is_holder(p, device_id)) for p in candidates if document.entry(p) is not None
    )
    signature = (
        policy["revision"],
        member.public_key,
        local_only,
        tuple((p, tuple(inv[p])) for p in candidates),
        held,
    )
    memo = getattr(server, _MEMO_ATTR, None)
    if not isinstance(memo, dict):
        memo = {}
        setattr(server, _MEMO_ATTR, memo)
    memo_key = (network_id, device_id)
    previous = memo.get(memo_key)
    if (
        previous is not None
        and previous[0] == signature
        and time.monotonic() - previous[1] < MEMO_TTL_S
    ):
        return ReconcileResult(previous[2])
    # Bound the memo to live recipients (it holds metadata tuples only).
    live = {m.device_id for m in record.active_members()}
    for stale in [k for k in memo if k[1] not in live]:
        del memo[stale]
    for attr in (_DELIVERY_ATTR, _STATUS_ATTR):
        table = getattr(server, attr, None)
        if isinstance(table, dict):
            for gone in [d for d in table if d not in live]:
                del table[gone]
    pushed = getattr(server, _PUSHED_ATTR, None)
    if isinstance(pushed, dict):
        for gone in [i for i in pushed if i[1] not in live]:
            del pushed[gone]

    result = ReconcileResult("unchanged")
    eligible: list[str] = []
    for key in candidates:
        entry = document.entry(key)
        if entry is not None and (
            entry.is_holder(device_id)
            or entry.owner_device != self_device
            or entry.kind != "oauth-rotating"
        ):
            continue  # never touch a finite/legacy holder or someone else's row
        permitted, code = _verify(network_id, self_device, device_id, key, inv[key], root)
        if permitted:
            eligible.append(key)
        elif code:
            result.refused[key] = code
    if eligible:
        ref = str(rec["approval"]["ref"])
        with authority.locked():
            # Re-verify INSIDE the consent lock: restrict_and_revoke writes its deny under
            # this same lock, so a deny that landed since the pass above is seen here.
            final = [
                key
                for key in eligible
                if _verify(network_id, self_device, device_id, key, inv[key], root)[0]
            ]
            if final:
                with placement.mutate(network_id, root, self_device=self_device) as projection:
                    granted, skipped = placement.grant_key_set(
                        projection,
                        device=device_id,
                        keys=final,
                        config=root,
                        self_device=self_device,
                        owner_name=str(getattr(server.identity, "name", "") or ""),
                        provenance=DERIVED_PREFIX + ref,
                    )
                result.projected = list(granted)
                for row in skipped:
                    result.refused[row["key"]] = row["reason"]
    result.outcome = f"projected:{','.join(result.projected)}" if result.projected else "unchanged"
    _audit_changes(server, network_id, self_device, device_id, result)
    if CODE_UNVERIFIABLE in result.refused.values():
        # A read that failed this tick says nothing about the next one: never memoise it,
        # and say it is being retried rather than leaving a quiet "unchanged".
        result.outcome = "notready:approval_unverifiable (retrying each tick)"
    else:
        memo[memo_key] = (signature, time.monotonic(), result.outcome)
    return ReconcileResult(
        _set_status(server, device_id, result.outcome), result.projected, result.refused
    )


def _audit_one(
    server: Any,
    event: str,
    network_id: str,
    self_device: str,
    device_id: str,
    *,
    key: str = "",
    code: str = "",
) -> None:
    """One keyless audit row (key, device ids, a machine code); a failed write is not a failure."""
    audit = getattr(server, "audit", None)
    if audit is None:
        return
    from local_operator.network.audit import AuditEvent

    detail: dict[str, Any] = {"act": self_device, "sub": device_id}
    if key:
        detail["credential_key"] = key
    if code:
        detail["code"] = code
    try:
        audit.record(
            AuditEvent(
                event=event,
                network_id=network_id,
                actor=self_device,
                actor_kind="device",
                subject=device_id,
                detail=detail,
            )
        )
    except Exception:  # noqa: BLE001 — a log that cannot be written is not a failure
        pass


def _audit_changes(
    server: Any, network_id: str, self_device: str, device_id: str, result: ReconcileResult
) -> None:
    seen = getattr(server, _AUDITED_ATTR, None)
    if not isinstance(seen, set):
        seen = set()
        setattr(server, _AUDITED_ATTR, seen)
    rows = [("credential.permission_projected", key, "") for key in result.projected]
    rows += [("credential.permission_refused", key, code) for key, code in result.refused.items()]
    for event, key, code in rows:
        ident = (event, device_id, key, code)
        if ident in seen:
            continue  # one row per distinct fact, not one per 15 s tick
        seen.add(ident)
        _audit_one(server, event, network_id, self_device, device_id, key=key, code=code)


def _purge_derived(
    server: Any,
    network_id: str,
    self_device: str,
    device_id: str,
    root: Path,
    authority: consent.ConsentDocument | None,
) -> None:
    """Take back ONLY the holders this module derived, once their approval is gone.

    Called when the recipient has no enabled approval (disabled, replaced by nothing,
    or ``consent.json`` removed). Without it the owner keeps advertising — and the
    recipient keeps a stale, borrowable-looking copy of — a holder every borrow would
    refuse with ``consent_withdrawn``. An explicit share (``granted_by`` = the owner) is
    never touched. The cheap path (no derived holder for this device) reads one file and
    takes no lock. Lock order consent -> placement, and the decision is re-made INSIDE
    the consent lock so a concurrent re-approval wins.
    """
    document = placement.PlacementDocument.load(network_id, root, self_device=self_device)
    if not any(
        is_derived(entry.holder(device_id))
        for key in document.keys_owned_by(self_device)
        if (entry := document.entry(key)) is not None
    ):
        return
    authority = authority or consent.ConsentDocument(network_id, self_device, root)
    try:
        with authority.locked():
            try:
                rec = authority.load()["recipients"].get(device_id)
            except MeshRefusal:
                return  # unreadable authority: change nothing, the serve gate stays closed
            if rec is not None and rec["enabled"] is True and "approval" in rec:
                return
            with placement.mutate(network_id, root, self_device=self_device) as projection:
                for key in projection.keys_owned_by(self_device):
                    entry = projection.entry(key)
                    if entry is not None and is_derived(entry.holder(device_id)):
                        projection.revoke(key, device_id, by=self_device)
                        _audit_one(
                            server,
                            "credential.permission_withdrawn",
                            network_id,
                            self_device,
                            device_id,
                            key=key,
                            code="approval_absent",
                        )
    except (MeshRefusal, OSError):
        logger.debug("credential reconcile: purge for %s deferred", device_id, exc_info=True)


# ---------------------------------------------------------------------------
# Owner: delivery
# ---------------------------------------------------------------------------


def _push_rows(
    server: Any, network_id: str, root: Path, self_device: str, device_id: str
) -> list[dict[str, Any]]:
    """Metadata rows to push: derived holders, plus rows this module pushed EARLIER whose
    holder has since gone (a revoke), so the recipient's copy stops reading eligible
    without waiting for a manual pull. The holder list is reduced to the owner and this
    recipient — never another device's grant. An explicit share is never pushed."""
    document = placement.PlacementDocument.load(network_id, root, self_device=self_device)
    state = getattr(server, _PUSHED_ATTR, None)
    previously = (
        {i[2] for i in state if i[0] in {"rev", "attempted"} and i[1] == device_id}
        if isinstance(state, dict)
        else set()
    )
    rows: list[dict[str, Any]] = []
    for key in document.keys_owned_by(self_device):
        entry = document.entry(key)
        if entry is None:
            continue
        holder = entry.holder(device_id)
        if not (is_derived(holder) or (holder is None and key in previously)):
            continue
        row = entry.to_json()
        row["holders"] = [h for h in row["holders"] if h["device"] in (self_device, device_id)]
        rows.append(row)
    return rows


#: Typed answers that mean "this peer will not take the push, and retrying changes nothing
#: until an operator acts": a build without the handler, or its row for us lacks the
#: capability. The ONLY answers that start the :data:`REFUSED_RETRY_S` window.
_PEER_REFUSAL_CODES = frozenset({"not_implemented", "not_authorised"})
#: Receiver-side answers that mean "not now": retried on the next tick.
_TRANSIENT_CODES = frozenset({"busy", "unavailable"})


def _classify_reply(reply: Any) -> tuple[str, str]:
    """``(verdict, code)``: ``ok`` | ``transient`` | ``peer_refused`` | ``rejected``.

    WHY THIS IS NOT "anything but an ack is a refusal" (review R2). ``PeerLink.request``
    returns ``None`` on a timeout, a failed send and a link that died mid-request; none of
    those is a statement by the peer, and treating one as a typed refusal silenced grant
    AND revoke delivery on a healthy link for minutes. Only a reply that actually carries
    a typed refusal can back the sender off.
    """
    if not isinstance(reply, dict):
        return "transient", ""
    detail = reply.get("detail")
    if isinstance(detail, dict) and detail.get("kind") == "ack":
        return "ok", ""
    if isinstance(detail, dict) and detail.get("kind") == "error":
        code = str(detail.get("code") or "")
        if code in _TRANSIENT_CODES:
            return "transient", code
        if code in _PEER_REFUSAL_CODES:
            return "peer_refused", code
        return "rejected", code or "rejected"
    if reply.get("op") == "error":
        code = str(reply.get("code") or "")
        if code in _PEER_REFUSAL_CODES:
            return "peer_refused", code
        if code in _TRANSIENT_CODES:
            return "transient", code
        if not code:
            # The authoriser intentionally withholds its reason. Do not invent a typed
            # policy decision from a sentence: like a lost reply, retry next tick.
            return "transient", ""
        return "rejected", code
    return "transient", ""


def deliver(server: Any, link: Any, device_id: str, incarnation: Any) -> str:
    """Push derived rows (and the revokes of rows pushed earlier) the recipient has not
    acknowledged on THIS link incarnation. Returns the outcome; the same outcome is
    recorded for :func:`status_for` and, when stuck, audited once with its remedy.

    Outcomes: ``idle`` | ``in_sync`` | ``pushed:<n>`` | ``failed`` (retry next tick) |
    ``skipped:<why>`` / ``refused:<code>`` (stuck, remedy named, bounded backoff).
    """
    from local_operator.network.credentials import sync as sync_mod

    root = Path(getattr(server, "root"))
    self_device = str(server.identity.device_id)
    network_id = str(getattr(link, "network_id", "") or "")
    if not network_id:
        return "idle"
    rows = _push_rows(server, network_id, root, self_device, device_id)
    state = getattr(server, _PUSHED_ATTR, None)
    if not isinstance(state, dict):
        state = {}
        setattr(server, _PUSHED_ATTR, state)
    if state.get(("link", device_id)) != incarnation:
        # Reconnect / recipient restart: receipts and backoffs belong to a connection (the
        # lifecycle ledger's rule), and the peer may have been upgraded meanwhile. A
        # re-push is idempotent — the recipient's merge returns [].
        for ident in [i for i in state if i[1] == device_id]:
            del state[ident]
        state[("link", device_id)] = incarnation
    pending = [r for r in rows if state.get(("rev", device_id, r["key"])) != r["doc_rev"]]
    if not pending:
        return _set_delivery(server, network_id, self_device, device_id, "in_sync")
    # THE GATE IS THE PUSH CAPABILITY ON BOTH ENDS, decided before any frame (review R1):
    # the lifecycle string a v0.71.3 peer also advertises does not imply a handler.
    if not push_negotiated(link):
        return _set_delivery(
            server,
            network_id,
            self_device,
            device_id,
            "skipped:peer_lacks_placement_push",
            f"upgrade lop on {device_id} to a build that advertises "
            "credential-placement-push-v1; until then it cannot receive this grant "
            f"without a manual 'lop network credentials' pull on {device_id}",
        )
    now = time.monotonic()
    refused_at = state.get(("refused", device_id))
    if refused_at is not None and now - refused_at[0] < REFUSED_RETRY_S:
        return _set_delivery(
            server,
            network_id,
            self_device,
            device_id,
            f"refused:{refused_at[1]}",
            f"{device_id} answered the push with {refused_at[1]}; upgrade it, or make sure "
            "its member row for this device carries broker_credential",
        )
    signature = tuple((r["key"], r["doc_rev"]) for r in pending)
    rejected_at = state.get(("rejected", device_id))
    if (
        rejected_at is not None
        and rejected_at[2] == signature
        and now - rejected_at[0] < REJECTED_RETRY_S
    ):
        return _set_delivery(
            server,
            network_id,
            self_device,
            device_id,
            f"refused:{rejected_at[1]}",
            f"{device_id} rejected the pushed placement ({rejected_at[1]}); nothing changed there",
        )
    document = {
        "schema": placement.PLACEMENT_SCHEMA,
        "network_id": network_id,
        "epoch": 0,
        "written_by": self_device,
        "credentials": pending,
    }
    frame = {
        "op": "net_broker",
        "kind": "placement",
        "origin": PUSH_ORIGIN,
        "network_id": network_id,
        "from_device": self_device,
        "req": server._next_relay_req(),  # noqa: SLF001 — the relay's own counter
        "document": document,
    }
    # A timeout can mean the recipient applied the row but its ack was lost. Remember
    # ATTEMPTED keys separately from acknowledged revisions, so a subsequent revoke is
    # still delivered; this is not a success receipt and never suppresses a retry.
    for row in pending:
        state[("attempted", device_id, row["key"])] = True
    try:
        reply = link.request(frame, timeout=sync_mod.ANNOUNCE_TIMEOUT_S)
    except Exception:  # noqa: BLE001 — one member's link is that member's problem
        logger.debug("credential reconcile: push to %s failed", device_id, exc_info=True)
        reply = None
    verdict, code = _classify_reply(reply)
    if verdict == "ok":
        for row in pending:
            state[("rev", device_id, row["key"])] = row["doc_rev"]
        state.pop(("rejected", device_id), None)
        state.pop(("refused", device_id), None)
        return _set_delivery(server, network_id, self_device, device_id, f"pushed:{len(pending)}")
    if verdict == "peer_refused":
        state[("refused", device_id)] = (now, code)
        return _set_delivery(
            server,
            network_id,
            self_device,
            device_id,
            f"refused:{code}",
            f"{device_id} answered the push with {code}; upgrade it, or make sure its member "
            "row for this device carries broker_credential",
        )
    if verdict == "rejected":
        state[("rejected", device_id)] = (now, code, signature)
        return _set_delivery(
            server,
            network_id,
            self_device,
            device_id,
            f"refused:{code}",
            f"{device_id} rejected the pushed placement ({code}); nothing changed there",
        )
    # Transient: nothing is memoised and nothing is backed off — the next tick re-sends the
    # same rows (idempotent on the recipient), so a missed revoke is never forgotten.
    return _set_delivery(server, network_id, self_device, device_id, "failed")


# ---------------------------------------------------------------------------
# Recipient: accept a push
# ---------------------------------------------------------------------------


def _refuse_push(code: str, message: str) -> dict[str, Any]:
    return {"kind": "error", "code": code, "message": message}


def _text(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 256


def accept_push(
    root: Path, self_device: str, link: Any, frame: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a pushed document BEFORE any disk write, then merge it.

    Narrower than a pull's merge on purpose: this layer only admits OAuth rows the
    sender OWNS whose holders are the sender and THIS device. The merge itself
    (``placement.merge_from_peer``) still applies owner-only / higher-revision /
    no-self-forgery rules, and the relay's authoriser already required the sender's
    ``broker_credential`` capability in our row for it.

    EVERY FIELD IS TYPE-CHECKED BEFORE IT IS COMPARED (review R3): a list or dict where a
    device id belongs raised ``TypeError`` from a set-membership test, which the relay
    reported as an anonymous failure the sender then read as a refusal. A malformed push
    is now a typed ``malformed_frame`` and changes nothing. The pushed ``epoch`` is
    sender-controlled and the merge would persist ``max(local, pushed)``; the only value
    this module ever sends is ``0``, so anything else is refused rather than trusted.
    """
    from local_operator.network.credentials.sync import (  # noqa: SLF001
        _member_is_active,
    )

    if not push_negotiated(link):
        # Same gate as the sender's, applied on receipt: a peer that did not negotiate the
        # push feature has no business sending it, and the legacy pull merge must never be
        # reachable by this frame shape.
        return _refuse_push("unsupported", "the push feature was not negotiated on this link")
    sender = str(getattr(link, "device_id", "") or "")
    claimed = frame.get("from_device")
    if not sender or (claimed not in (None, "") and claimed != sender):
        return _refuse_push(
            "identity_mismatch", "the push's sender is not the authenticated device"
        )
    network_id = str(getattr(link, "network_id", "") or "")
    document = frame.get("document")
    if not isinstance(document, dict) or not network_id:
        return _refuse_push("malformed_frame", "the push carries no placement document")
    if (
        document.get("network_id") != network_id
        or document.get("schema") != placement.PLACEMENT_SCHEMA
        or sender == self_device
    ):
        return _refuse_push("identity_mismatch", "the push names another network or this device")
    epoch = document.get("epoch")
    if type(epoch) is not int or epoch != 0:
        return _refuse_push("malformed_frame", "the push carries an epoch this feature never sends")
    rows = document.get("credentials")
    if not isinstance(rows, list) or not rows or len(rows) > MAX_PUSH_CREDENTIALS:
        return _refuse_push("malformed_frame", "the push carries no rows or too many")
    try:
        if len(json.dumps(document, separators=(",", ":")).encode()) > MAX_PUSH_BYTES:
            raise ValueError("oversize")
        placement._assert_no_material(document, where="pushed placement")  # noqa: SLF001
    except (ValueError, TypeError, RecursionError):
        return _refuse_push("malformed_frame", "the push is oversized or carries material fields")
    for row in rows:
        if not isinstance(row, dict):
            return _refuse_push("malformed_frame", "a pushed row is not an object")
        holders = row.get("holders")
        if (
            not _text(row.get("key"))
            or not _text(row.get("owner_device"))
            or not isinstance(row.get("provider"), str)
            or not _text(row.get("kind"))
            or type(row.get("doc_rev")) is not int
            or not isinstance(holders, list)
            or len(holders) > MAX_PUSH_HOLDERS
        ):
            return _refuse_push("malformed_frame", "a pushed row has malformed fields")
        for holder in holders:
            if (
                not isinstance(holder, dict)
                or not _text(holder.get("device"))
                or not _text(holder.get("scope"))
            ):
                return _refuse_push("malformed_frame", "a pushed holder has malformed fields")
    # Types are now known-good; the policy checks below can compare freely.
    if not _member_is_active(root, network_id, sender):
        return _refuse_push("not_owner", "the sender is not an active member of this network")
    for row in rows:
        key = row["key"]
        if (
            row["owner_device"] != sender
            or not _provider_ok(key)
            or row["provider"] != key
            or row["kind"] != "oauth-rotating"
        ):
            return _refuse_push(
                "not_owner", "a pushed row is not a native OAuth row its sender owns"
            )
        for holder in row["holders"]:
            if holder["device"] not in {sender, self_device} or holder["scope"] not in {
                "session",
                "device",
            }:
                return _refuse_push("not_owner", "a pushed row names a holder this push may not")
    changed = placement.merge_from_peer(
        network_id, document, from_device=sender, self_device=self_device, root=root
    )
    return {"kind": "ack", "key": "", "changed": changed}


def member_placement_push(server: Any, link: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
    """``owner.relay_handler``'s route for a push — before the broker is consulted, so a
    recipient that owns nothing (no broker) accepts it exactly like one that does.

    The negotiation gate is checked FIRST, before identity or disk: a frame on a link that
    did not negotiate the push capability never reaches the merge, on either route."""
    if not push_negotiated(link):
        return _refuse_push("unsupported", "the push feature was not negotiated on this link")
    root = getattr(server, "root", None)
    self_device = str(getattr(getattr(server, "identity", None), "device_id", "") or "")
    if root is None or not self_device:
        return _refuse_push("unavailable", "this device has no identity; nothing applied")
    try:
        return accept_push(Path(root), self_device, link, frame)
    except (OSError, MeshRefusal) as exc:
        return _refuse_push(getattr(exc, "code", "") or "busy", "the push was not applied")

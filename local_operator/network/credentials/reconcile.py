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
* It never touches an existing holder (finite or legacy), a deny, ``consent.json``,
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

#: A recipient that refused the push BY NAME (older build, or its row for us lacks the
#: capability) is not asked again for this long, and never tightly: the answer changes
#: only when an operator upgrades or grants, and a retry per 15 s tick is pure noise.
REFUSED_RETRY_S = 300.0

#: A memoised "nothing changed" verdict is re-derived at least this often, so drift the
#: memo key cannot see (operator anchor change, approval record edited) still heals.
MEMO_TTL_S = 300.0

#: The three codes a derived-holder refusal reaches the borrower as. Every other
#: consent refusal collapses into ``consent_withdrawn`` so the closed wire set stays small.
CODE_MISMATCH = "scope_mismatch"
CODE_UNKNOWN = "scope_unknown"
CODE_WITHDRAWN = "consent_withdrawn"
SCOPE_CODES = frozenset({CODE_MISMATCH, CODE_UNKNOWN, CODE_WITHDRAWN})

_MEMO_ATTR = "_mesh_credential_reconcile_memo"
_STATUS_ATTR = "_mesh_credential_reconcile_status"
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
    """The borrower-facing code for a consent refusal (see :data:`SCOPE_CODES`)."""
    if code in (CODE_MISMATCH, CODE_UNKNOWN, "not_a_holder"):
        return code
    return CODE_WITHDRAWN


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
        except MeshRefusal as exc:
            refusal = refusal or wire_code(exc.code)
            if exc.code == CODE_MISMATCH:
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
        return (), (), CODE_UNKNOWN
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
    except MeshRefusal as exc:
        raise MeshRefusal(
            wire_code(exc.code), "the signed approval does not cover this login"
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
    status = getattr(server, _STATUS_ATTR, None)
    return str(status.get(device_id, "")) if isinstance(status, dict) else ""


def _record_for(root: Path, device_id: str) -> Any:
    for record in store.list_networks(root):
        member = record.member(device_id)
        if member is not None and member.active and device_id != record.self_device_id:
            return record
    return None


def project_for(server: Any, device_id: str) -> ReconcileResult:
    """One recipient's reconcile pass. Never raises for a refusal; mutates only by adding
    derived holders (see module docstring)."""
    root = getattr(server, "root", None)
    self_device = str(getattr(getattr(server, "identity", None), "device_id", "") or "")
    if root is None or not self_device or not device_id:
        return ReconcileResult(_set_status(server, device_id, "idle"))
    root = Path(root)
    record = _record_for(root, device_id)
    if record is None:
        return ReconcileResult(_set_status(server, device_id, "idle"))
    network_id = str(record.network_id)
    # Cheap gate: no consent file => one stat and out. Nothing else is read or written.
    if not placement.placement_path(network_id, root).with_name("consent.json").exists():
        return ReconcileResult(_set_status(server, device_id, "idle"))
    try:
        authority = consent.ConsentDocument(network_id, self_device, root)
        policy = authority.load()
    except MeshRefusal:
        return ReconcileResult(_set_status(server, device_id, "idle:consent_unreadable"))
    rec = policy["recipients"].get(device_id)
    if rec is None or rec["enabled"] is not True or "approval" not in rec:
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
        return ReconcileResult(_set_status(server, device_id, "idle:inventory_unknown"))
    document = placement.PlacementDocument.load(network_id, root, self_device=self_device)
    candidates = sorted(p for p in inv if _provider_ok(p))
    held = tuple(
        (p, document.is_holder(p, device_id)) for p in candidates if document.entry(p) is not None
    )
    signature = (
        policy["revision"],
        member.public_key,
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
    memo[memo_key] = (signature, time.monotonic(), result.outcome)
    return ReconcileResult(
        _set_status(server, device_id, result.outcome), result.projected, result.refused
    )


def _audit_changes(
    server: Any, network_id: str, self_device: str, device_id: str, result: ReconcileResult
) -> None:
    audit = getattr(server, "audit", None)
    if audit is None:
        return
    seen = getattr(server, _AUDITED_ATTR, None)
    if not isinstance(seen, set):
        seen = set()
        setattr(server, _AUDITED_ATTR, seen)
    from local_operator.network.audit import AuditEvent

    rows = [("credential.permission_projected", key, "") for key in result.projected]
    rows += [("credential.permission_refused", key, code) for key, code in result.refused.items()]
    for event, key, code in rows:
        ident = (event, device_id, key, code)
        if ident in seen:
            continue  # one row per distinct fact, not one per 15 s tick
        seen.add(ident)
        try:
            detail = {"credential_key": key, "act": self_device, "sub": device_id}
            if code:
                detail["code"] = code
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
        {i[2] for i in state if i[0] == "rev" and i[1] == device_id}
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


def deliver(server: Any, link: Any, device_id: str, incarnation: Any) -> str:
    """Push derived rows the recipient has not acknowledged on THIS link incarnation."""
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
        # Reconnect / recipient restart: receipts belong to a connection (the lifecycle
        # ledger's rule). A re-push is idempotent — the recipient's merge returns [].
        for ident in [i for i in state if i[1] == device_id]:
            del state[ident]
        state[("link", device_id)] = incarnation
    refused_at = state.get(("refused", device_id))
    if refused_at is not None and time.monotonic() - refused_at < REFUSED_RETRY_S:
        return "skipped:peer_lacks_placement_push"
    pending = [r for r in rows if state.get(("rev", device_id, r["key"])) != r["doc_rev"]]
    if not pending:
        return "in_sync"
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
    try:
        reply = link.request(frame, timeout=sync_mod.ANNOUNCE_TIMEOUT_S)
    except Exception:  # noqa: BLE001 — one member's link is that member's problem
        logger.debug("credential reconcile: push to %s failed", device_id, exc_info=True)
        return "failed"
    detail = reply.get("detail") if isinstance(reply, dict) else None
    if isinstance(detail, dict) and detail.get("kind") == "ack":
        for row in pending:
            state[("rev", device_id, row["key"])] = row["doc_rev"]
        return f"pushed:{len(pending)}"
    if isinstance(detail, dict) and detail.get("code") in {"busy", "unavailable"}:
        return "failed"  # transient on the recipient: the next tick retries
    # Refused BY NAME (an older build, or a recipient whose own row for us lacks the
    # capability): remember for this connection so it is never retried tightly.
    state[("refused", device_id)] = time.monotonic()
    return "skipped:peer_lacks_placement_push"


# ---------------------------------------------------------------------------
# Recipient: accept a push
# ---------------------------------------------------------------------------


def _refuse_push(code: str, message: str) -> dict[str, Any]:
    return {"kind": "error", "code": code, "message": message}


def accept_push(
    root: Path, self_device: str, link: Any, frame: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a pushed document BEFORE any disk write, then merge it.

    Narrower than a pull's merge on purpose: this layer only admits OAuth rows the
    sender OWNS whose holders are the sender and THIS device. The merge itself
    (``placement.merge_from_peer``) still applies owner-only / higher-revision /
    no-self-forgery rules, and the relay's authoriser already required the sender's
    ``broker_credential`` capability in our row for it.
    """
    from local_operator.network.credentials.sync import (  # noqa: SLF001
        _member_is_active,
    )

    sender = str(getattr(link, "device_id", "") or "")
    claimed = str(frame.get("from_device") or "")
    if not sender or (claimed and claimed != sender):
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
    if not _member_is_active(root, network_id, sender):
        return _refuse_push("not_owner", "the sender is not an active member of this network")
    rows = document.get("credentials")
    if not isinstance(rows, list) or not rows or len(rows) > MAX_PUSH_CREDENTIALS:
        return _refuse_push("malformed_frame", "the push carries no rows or too many")
    try:
        if len(json.dumps(document, separators=(",", ":")).encode()) > MAX_PUSH_BYTES:
            raise ValueError("oversize")
        placement._assert_no_material(document, where="pushed placement")  # noqa: SLF001
    except ValueError:
        return _refuse_push("malformed_frame", "the push is oversized or carries material fields")
    for row in rows:
        key = str(row.get("key") or "") if isinstance(row, dict) else ""
        if (
            not isinstance(row, dict)
            or row.get("owner_device") != sender
            or not _provider_ok(key)
            or row.get("provider") != key
            or row.get("kind") != "oauth-rotating"
            or not isinstance(row.get("holders"), list)
        ):
            return _refuse_push(
                "not_owner", "a pushed row is not a native OAuth row its sender owns"
            )
        for holder in row["holders"]:
            if (
                not isinstance(holder, dict)
                or holder.get("device") not in {sender, self_device}
                or holder.get("scope") not in {"session", "device"}
            ):
                return _refuse_push("not_owner", "a pushed row names a holder this push may not")
    changed = placement.merge_from_peer(
        network_id, document, from_device=sender, self_device=self_device, root=root
    )
    return {"kind": "ack", "key": "", "changed": changed}


def member_placement_push(server: Any, link: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
    """``owner.relay_handler``'s route for a push — before the broker is consulted, so a
    recipient that owns nothing (no broker) accepts it exactly like one that does."""
    root = getattr(server, "root", None)
    self_device = str(getattr(getattr(server, "identity", None), "device_id", "") or "")
    if root is None or not self_device:
        return _refuse_push("unavailable", "this device has no identity; nothing applied")
    try:
        return accept_push(Path(root), self_device, link, frame)
    except (OSError, MeshRefusal) as exc:
        return _refuse_push(getattr(exc, "code", "") or "busy", "the push was not applied")

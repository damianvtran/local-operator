"""Pure proposals, NOT enrollment authority; adapters must recheck every capture.

Ingress owns identity/consent-byte validation. No message bodies or credentials
belong here. One fresh account snapshot covers every tenant, including later
joins; no worker, config access, or singleton owner mutation is hidden in planning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum


class Preference(StrEnum):
    UNSET = "unset"  # A schema default False is not persisted user intent.
    EXPLICIT_TRUE = "explicit_true"
    EXPLICIT_FALSE = "explicit_false"
    UNREADABLE = "unreadable"


class ServerState(StrEnum):
    READY = "ready"
    DISABLED = "disabled"
    PAUSED = "paused"
    STALE_CONSENT = "stale_consent"


class IntentKind(StrEnum):
    CREATE = "create"
    ATTACH_CURRENT = "attach_current"
    HOLD = "hold"


class Reason(StrEnum):
    NEW_ENROLLMENT = "new_enrollment"
    CURRENT_BINDING = "current_binding"
    EXPLICIT_OFF = "explicit_off"
    CONFIG_UNREADABLE = "config_unreadable"
    SNAPSHOT_NOT_AUTHORITATIVE = "snapshot_not_authoritative"
    OWNER_AMBIGUOUS = "owner_ambiguous"
    LEGACY_ACCOUNT_DONE = "legacy_account_done"
    RECEIVER_UNBOUND = "receiver_unbound"
    RECEIVER_IDENTITY_MISMATCH = "receiver_identity_mismatch"
    TENANT_IDENTITY_MISMATCH = "tenant_identity_mismatch"
    BINDING_IDENTITY_MISMATCH = "binding_identity_mismatch"
    ENDPOINT_MISMATCH = "endpoint_mismatch"
    GENERATION_MISMATCH = "generation_mismatch"
    GENERATION_NOT_CAPTURED = "generation_not_captured"
    MEMBERSHIP_NOT_CURRENT = "membership_not_current"
    INELIGIBLE = "ineligible"
    SERVER_DISABLED = "server_disabled"
    SERVER_PAUSED = "server_paused"
    SERVER_STALE_CONSENT = "server_stale_consent"
    BINDING_NOT_CURRENT = "binding_not_current"
    RECEIPT_NOT_ACCEPTED = "receipt_not_accepted"
    REPLAY_FENCE_MISSING = "replay_fence_missing"
    BOOKKEEPING_WITHOUT_BINDING = "bookkeeping_without_binding"
    OFFER_NOT_AUTOMATIC = "offer_not_automatic"
    TENANT_REMOVED = "tenant_removed"


@dataclass(frozen=True, slots=True)
class OwnerKey:
    hub: str
    account_id: str
    tenant_id: str
    endpoint_id: str

    @property
    def namespace(self) -> str:
        # Match store._owner_key without importing its I/O-bearing store module.
        return json.dumps(
            [self.hub, self.account_id, self.tenant_id, self.endpoint_id], separators=(",", ":")
        )


@dataclass(frozen=True, slots=True)
class ReceiverBinding:
    hub: str
    account_id: str
    device_id: str
    session_id: str


@dataclass(frozen=True, slots=True)
class Offer:
    version: str
    text_ref: str
    bytes_validated: bool
    policy_accepted: bool


@dataclass(frozen=True, slots=True)
class Binding:
    owner: OwnerKey
    device_id: str
    generation: int
    receipt_version: str
    receipt_validated: bool
    pre_enable_head: int | None
    active: bool = True
    current: bool = True


@dataclass(frozen=True, slots=True)
class TenantSnapshot:
    hub: str
    account_id: str
    tenant_id: str
    membership_revision: str
    membership_current: bool = True
    eligible: bool = True
    server_state: ServerState = ServerState.READY
    binding: Binding | None = None
    enrollment_recorded: bool = False


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    hub: str
    account_id: str
    device_id: str
    #: The owner namespace's ENDPOINT half. Exact attachment requires this to
    #: match the captured binding's ``owner.endpoint_id``; a snapshot that
    #: differs only here is a different local owner, not the same row.
    endpoint_id: str
    snapshot_id: str
    preference: Preference
    tenants: tuple[TenantSnapshot, ...]
    offer: Offer | None
    accepted_receipt_versions: tuple[str, ...]
    #: The generation the caller expects for this owner namespace, when it has
    #: one. ``None`` means "not captured": the planner carries the binding's
    #: generation through but cannot judge it, and the adapter must compare it
    #: against the store's active generation before acting. A provided value is
    #: compared here, so a stale generation can never attach as current.
    generation: int | None = None
    #: Bindings the local store holds for this account/device whose tenant is
    #: NOT in ``tenants`` — an org the account has been removed from. They are
    #: planned as visible holds, never silently dropped; the adapter must supply
    #: every local binding or an omission is indistinguishable from absence.
    local_bindings: tuple[Binding, ...] = ()
    receiver: ReceiverBinding | None = None
    authoritative: bool = True
    active_owner_ambiguous: bool = False
    legacy_account_done: bool = False


@dataclass(frozen=True, slots=True)
class Intent:
    kind: IntentKind
    reason: Reason
    hub: str
    account_id: str
    device_id: str
    snapshot_id: str
    tenant_id: str
    #: ``None`` only for a tenant removed from the org, where no fresh membership
    #: revision exists; every tenant present in the snapshot carries one.
    membership_revision: str | None
    receiver: ReceiverBinding | None
    owner: OwnerKey | None
    generation: int | None
    pre_enable_head: int | None
    receipt_version: str | None
    offer_version: str | None = None
    offer_text_ref: str | None = None

    def __post_init__(self) -> None:
        # Membership revision is required for every decision about a tenant the
        # snapshot still contains. Only the removed-tenant hold can lack one,
        # because no fresh revision exists for an org the account left. Making
        # this a construction invariant keeps a future caller from expressing
        # "unknown membership" on an intent that might otherwise attach.
        if self.membership_revision is None and self.reason is not Reason.TENANT_REMOVED:
            raise ValueError("membership revision missing for a non-removal intent")


def _validate_binding(b: Binding) -> None:
    """Captured bindings are facts; a malformed one fails the whole plan.

    Shared by tenant rows and local (removed-tenant) captures so both entry
    points cannot drift into accepting different shapes of the same record.
    """
    if not isinstance(b, Binding):
        raise ValueError("typed binding required")
    if any(type(flag) is not bool for flag in (b.active, b.current, b.receipt_validated)):
        raise ValueError("validated boolean facts required, not truthy raw values")
    identifiers = (
        b.owner.hub,
        b.owner.account_id,
        b.owner.tenant_id,
        b.owner.endpoint_id,
        b.device_id,
        b.receipt_version,
    )
    if any(not isinstance(v, str) or not v.strip() for v in identifiers):
        raise ValueError("empty or invalid captured identity")
    if type(b.generation) is not int or b.generation < 1:
        raise ValueError("invalid namespace generation")
    if b.pre_enable_head is not None and (
        type(b.pre_enable_head) is not int or b.pre_enable_head < 0
    ):
        raise ValueError("invalid initial replay fence")


def _reason(s: AccountSnapshot, t: TenantSnapshot) -> Reason:
    if s.preference is Preference.EXPLICIT_FALSE:
        return Reason.EXPLICIT_OFF
    if s.preference is Preference.UNREADABLE:
        return Reason.CONFIG_UNREADABLE
    if not s.authoritative:
        return Reason.SNAPSHOT_NOT_AUTHORITATIVE
    if s.active_owner_ambiguous:
        return Reason.OWNER_AMBIGUOUS
    if s.legacy_account_done:
        return Reason.LEGACY_ACCOUNT_DONE
    if s.receiver is None:
        return Reason.RECEIVER_UNBOUND
    if (s.receiver.hub, s.receiver.account_id, s.receiver.device_id) != (
        s.hub,
        s.account_id,
        s.device_id,
    ):
        return Reason.RECEIVER_IDENTITY_MISMATCH
    if (t.hub, t.account_id) != (s.hub, s.account_id):
        return Reason.TENANT_IDENTITY_MISMATCH
    if not t.membership_current:
        return Reason.MEMBERSHIP_NOT_CURRENT
    if not t.eligible:
        return Reason.INELIGIBLE
    if t.server_state is not ServerState.READY:
        return Reason("server_" + t.server_state.value)
    b = t.binding
    if b is not None:
        if (b.owner.hub, b.owner.account_id, b.owner.tenant_id, b.device_id) != (
            s.hub,
            s.account_id,
            t.tenant_id,
            s.device_id,
        ):
            return Reason.BINDING_IDENTITY_MISMATCH
        # The namespace is JSON(hub, account, tenant, ENDPOINT): a row captured
        # for another endpoint of the same tenant is a different owner, and
        # attaching it would hand one endpoint's membership to another. Exact
        # means all four fields, so this is compared, never assumed.
        if b.owner.endpoint_id != s.endpoint_id:
            return Reason.ENDPOINT_MISMATCH
        # Generation is compared only when the caller captured the one it
        # expects; ``None`` is "not captured" and the adapter must compare it
        # against the store's active generation before acting. A supplied value
        # is authoritative here, so stale generations cannot attach.
        # "Not captured" is a boundary the pure function enforces: a caller that
        # did not supply the expected generation cannot attach. Leaving that to
        # the adapter's comparison would make the boundary advisory, and this
        # contract does not delegate boundaries. Capture it to attach.
        if s.generation is None:
            return Reason.GENERATION_NOT_CAPTURED
        if b.generation != s.generation:
            return Reason.GENERATION_MISMATCH
        if not b.active or not b.current:
            return Reason.BINDING_NOT_CURRENT
        if not b.receipt_validated or b.receipt_version not in s.accepted_receipt_versions:
            return Reason.RECEIPT_NOT_ACCEPTED
        # Late attachment cannot silently start at zero or skip unread backlog.
        # Preserve the captured initial fence; deriving one is another contract.
        if b.pre_enable_head is None:
            return Reason.REPLAY_FENCE_MISSING
        return Reason.CURRENT_BINDING
    # Bookkeeping is neither authority to recreate an endpoint nor proof we
    # created one. Its uncertainty holds this tenant only, never another org.
    if t.enrollment_recorded:
        return Reason.BOOKKEEPING_WITHOUT_BINDING
    o = s.offer
    if (
        o is None
        or o.version != "bridge_consent_v2"
        or not o.text_ref.strip()
        or not o.bytes_validated
        or not o.policy_accepted
    ):
        return Reason.OFFER_NOT_AUTOMATIC
    return Reason.NEW_ENROLLMENT


def plan_reconciliation(snapshot: AccountSnapshot) -> tuple[Intent, ...]:
    """Reject malformed/duplicate captures atomically; never partially enroll.

    Canonical hub identity is supplied by authoritative ingress, not guessed by
    URL normalization here. Generation belongs to the full owner namespace.
    These proposals do not relax the store's global active_owner ambiguity guard.
    """
    s = snapshot
    if not isinstance(s.preference, Preference) or not isinstance(s.tenants, tuple):
        raise ValueError("typed preference and immutable tenant snapshot required")
    if not isinstance(s.accepted_receipt_versions, tuple):
        raise ValueError("immutable accepted receipt policy required")
    if s.generation is not None and (type(s.generation) is not int or s.generation < 1):
        raise ValueError("invalid expected generation")
    if not isinstance(s.local_bindings, tuple):
        raise ValueError("immutable local binding capture required")
    identifiers = [
        s.hub,
        s.account_id,
        s.device_id,
        s.endpoint_id,
        s.snapshot_id,
        *s.accepted_receipt_versions,
    ]
    flags = [s.authoritative, s.active_owner_ambiguous, s.legacy_account_done]
    if s.offer is not None:
        if not isinstance(s.offer.version, str) or not isinstance(s.offer.text_ref, str):
            raise ValueError("invalid offer identity")
        flags.extend((s.offer.bytes_validated, s.offer.policy_accepted))
    if s.receiver is not None:
        identifiers.extend(
            (s.receiver.hub, s.receiver.account_id, s.receiver.device_id, s.receiver.session_id)
        )
    for t in s.tenants:
        identifiers.extend((t.hub, t.account_id, t.tenant_id, t.membership_revision))
        flags.extend((t.membership_current, t.eligible, t.enrollment_recorded))
        if not isinstance(t.server_state, ServerState):
            raise ValueError("unknown server state")
        if t.binding is not None:
            _validate_binding(t.binding)
    present = {t.tenant_id for t in s.tenants}
    for b in s.local_bindings:
        _validate_binding(b)
        if (b.owner.hub, b.owner.account_id, b.device_id) != (
            s.hub,
            s.account_id,
            s.device_id,
        ):
            raise ValueError("local binding belongs to another hub/account/device")
        if b.owner.tenant_id in present:
            # A tenant row carries its own binding; a second capture for the
            # same tenant is a malformed snapshot, not two facts to reconcile.
            raise ValueError("local binding duplicates a present tenant")
    if len({b.owner.namespace for b in s.local_bindings}) != len(s.local_bindings):
        raise ValueError("duplicate local binding")
    if any(type(flag) is not bool for flag in flags):
        raise ValueError("validated boolean facts required, not truthy raw values")
    if any(not isinstance(v, str) or not v.strip() for v in identifiers):
        raise ValueError("empty or invalid captured identity")
    if len({t.tenant_id for t in s.tenants}) != len(s.tenants):
        raise ValueError("duplicate tenant snapshot")
    result = []
    for t in sorted(s.tenants, key=lambda row: row.tenant_id):
        reason, b = _reason(s, t), t.binding
        # A foreign owner is diagnostic only, never a consumable attachment key.
        if reason in {
            Reason.TENANT_IDENTITY_MISMATCH,
            Reason.BINDING_IDENTITY_MISMATCH,
            Reason.ENDPOINT_MISMATCH,
            Reason.GENERATION_MISMATCH,
            Reason.GENERATION_NOT_CAPTURED,
        }:
            b = None
        kind = {
            Reason.NEW_ENROLLMENT: IntentKind.CREATE,
            Reason.CURRENT_BINDING: IntentKind.ATTACH_CURRENT,
        }.get(reason, IntentKind.HOLD)
        offer = s.offer if kind is IntentKind.CREATE else None
        result.append(
            Intent(
                kind=kind,
                reason=reason,
                hub=s.hub,
                account_id=s.account_id,
                device_id=s.device_id,
                snapshot_id=s.snapshot_id,
                tenant_id=t.tenant_id,
                membership_revision=t.membership_revision,
                receiver=s.receiver,
                owner=b.owner if b else None,
                generation=b.generation if b else None,
                pre_enable_head=b.pre_enable_head if b else None,
                receipt_version=b.receipt_version if b else None,
                offer_version=offer.version if offer else None,
                offer_text_ref=offer.text_ref if offer else None,
            )
        )
    # An org the account has been removed from still has a local owner row. It
    # is a HOLD with the captured owner/generation carried for the adapter to
    # preserve: silence would look identical to "nothing to do" while a user
    # still has a binding, and no fresh membership revision exists to attach
    # with. Removal is inferred ONLY from an authoritative snapshot — a
    # non-authoritative capture cannot prove absence, and its present tenants
    # already carry the account-wide hold.
    if s.authoritative:
        for b in sorted(s.local_bindings, key=lambda row: row.owner.tenant_id):
            result.append(
                Intent(
                    kind=IntentKind.HOLD,
                    reason=Reason.TENANT_REMOVED,
                    hub=s.hub,
                    account_id=s.account_id,
                    device_id=s.device_id,
                    snapshot_id=s.snapshot_id,
                    tenant_id=b.owner.tenant_id,
                    membership_revision=None,
                    receiver=s.receiver,
                    owner=b.owner,
                    generation=b.generation,
                    pre_enable_head=b.pre_enable_head,
                    receipt_version=b.receipt_version,
                )
            )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Rotation:
    owners: tuple[OwnerKey, ...]
    cursor: str | None


def select_rotation(
    owners: tuple[OwnerKey, ...], *, budget: int, cursor: str | None = None
) -> Rotation:
    """Pure bounded round robin; cursor is the last served JSON owner namespace.

    For a stable set, every owner is visited within ceil(N / budget) cycles.
    Callers cap TOTAL transport requests, not each org's independent budget.
    """
    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be a positive integer")
    ordered = sorted(owners, key=lambda owner: owner.namespace)
    keys = [owner.namespace for owner in ordered]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate owner namespace")
    if not ordered:
        return Rotation((), None)
    start = next((n for n, key in enumerate(keys) if cursor is None or key > cursor), 0)
    selected = tuple((ordered[start:] + ordered[:start])[:budget])
    return Rotation(selected, selected[-1].namespace)

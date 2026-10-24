"""Decision-only contract: no fixture may enroll, arm a worker, or read config."""

from __future__ import annotations

import ast
import inspect
import json
import os
from dataclasses import FrozenInstanceError, replace

import pytest

from local_operator.team_bridge import reconcile_plan as planner
from local_operator.team_bridge.reconcile_plan import (
    AccountSnapshot,
    Binding,
    Intent,
    IntentKind,
    Offer,
    OwnerKey,
    Preference,
    Reason,
    ReceiverBinding,
    ServerState,
    TenantSnapshot,
    plan_reconciliation,
    select_rotation,
)

V1 = "bridge_consent_v1"
V2 = "bridge_consent_v2"
HUB = "https://hub.example"


def _tenant(tenant_id: str = "tnt_1", **changes) -> TenantSnapshot:
    return replace(TenantSnapshot(HUB, "acc_a", tenant_id, "membership-7"), **changes)


def _binding(tenant_id: str = "tnt_1", **changes) -> Binding:
    return replace(
        Binding(OwnerKey(HUB, "acc_a", tenant_id, "ep_1"), "dev_1", 3, V1, True, 42),
        **changes,
    )


def _snapshot(**changes) -> AccountSnapshot:
    return replace(
        AccountSnapshot(
            hub=HUB,
            account_id="acc_a",
            device_id="dev_1",
            endpoint_id="ep_1",
            snapshot_id="snapshot-10",
            preference=Preference.UNSET,
            tenants=(_tenant(),),
            offer=Offer(V2, "sha256:validated-consent-identity", True, True),
            accepted_receipt_versions=(V1, V2),
            # The helper captures the generation it expects, matching
            # ``_binding()``; an attach therefore has a real comparison. Tests
            # for the \"not captured\" hold pass generation=None explicitly.
            generation=3,
            receiver=ReceiverBinding(HUB, "acc_a", "dev_1", "receiver_a"),
        ),
        **changes,
    )


@pytest.mark.parametrize("preference", [Preference.UNSET, Preference.EXPLICIT_TRUE])
def test_empty_config_and_explicit_true_are_eligible(preference) -> None:
    # The adapter must map a schema default False with no persisted key to UNSET,
    # not pretend that default is a user opt-out. Raw bools are not this API.
    (intent,) = plan_reconciliation(_snapshot(preference=preference))
    assert intent.kind is IntentKind.CREATE
    assert intent.offer_version == V2
    assert intent.offer_text_ref == "sha256:validated-consent-identity"
    assert intent.owner is None and intent.generation is None
    assert intent.snapshot_id == "snapshot-10"
    assert intent.membership_revision == "membership-7"
    with pytest.raises(ValueError):
        plan_reconciliation(_snapshot(preference=False))


@pytest.mark.parametrize(
    "preference,reason",
    [
        (Preference.EXPLICIT_FALSE, Reason.EXPLICIT_OFF),
        (Preference.UNREADABLE, Reason.CONFIG_UNREADABLE),
    ],
)
def test_persisted_false_and_unreadable_never_enable(preference, reason) -> None:
    for tenant in (_tenant(), _tenant(binding=_binding())):
        (intent,) = plan_reconciliation(_snapshot(preference=preference, tenants=(tenant,)))
        assert intent.kind is IntentKind.HOLD and intent.reason is reason


def test_all_eligible_orgs_and_later_join_are_planned() -> None:
    initial = _snapshot(tenants=(_tenant("tnt_2"), _tenant("tnt_1")))
    assert [i.tenant_id for i in plan_reconciliation(initial)] == ["tnt_1", "tnt_2"]
    fresh = replace(
        initial, snapshot_id="snapshot-11", tenants=initial.tenants + (_tenant("tnt_3"),)
    )
    result = plan_reconciliation(fresh)
    assert len(result) == 3 and all(i.kind is IntentKind.CREATE for i in result)
    assert all(i.snapshot_id == "snapshot-11" for i in result)


def test_existing_v1_receipt_and_new_v2_offer_are_independent() -> None:
    snapshot = _snapshot(tenants=(_tenant(binding=_binding()), _tenant("tnt_2")))
    existing, new = plan_reconciliation(snapshot)
    assert existing.kind is IntentKind.ATTACH_CURRENT
    assert existing.receipt_version == V1 and existing.offer_version is None
    assert existing.owner == _binding().owner and existing.generation == 3
    assert existing.pre_enable_head == 42
    assert new.kind is IntentKind.CREATE and new.offer_version == V2


@pytest.mark.parametrize(
    "offer",
    [
        None,
        Offer(V1, "sha256:v1", True, True),
        Offer(V2, "sha256:v2", False, True),
        Offer(V2, "sha256:v2", True, False),
        Offer(V2, "", True, True),
        Offer("unrecognized_future_version", "sha256:future", True, True),
    ],
)
def test_missing_v1_unaccepted_or_unvalidated_offer_holds_new_only(offer) -> None:
    existing, new = plan_reconciliation(
        _snapshot(offer=offer, tenants=(_tenant(binding=_binding()), _tenant("tnt_2")))
    )
    assert existing.kind is IntentKind.ATTACH_CURRENT
    assert new.kind is IntentKind.HOLD and new.reason is Reason.OFFER_NOT_AUTOMATIC


@pytest.mark.parametrize(
    "state,reason",
    [
        (ServerState.DISABLED, Reason.SERVER_DISABLED),
        (ServerState.PAUSED, Reason.SERVER_PAUSED),
        (ServerState.STALE_CONSENT, Reason.SERVER_STALE_CONSENT),
    ],
)
def test_server_state_is_preserved_even_with_valid_offer(state, reason) -> None:
    for binding in (None, _binding()):
        (intent,) = plan_reconciliation(
            _snapshot(tenants=(_tenant(server_state=state, binding=binding),))
        )
        assert intent.kind is IntentKind.HOLD and intent.reason is reason


@pytest.mark.parametrize(
    "tenant,reason",
    [
        (_tenant(membership_current=False), Reason.MEMBERSHIP_NOT_CURRENT),
        (_tenant(eligible=False), Reason.INELIGIBLE),
        (_tenant(binding=_binding(current=False)), Reason.BINDING_NOT_CURRENT),
        (_tenant(binding=_binding(active=False)), Reason.BINDING_NOT_CURRENT),
        (_tenant(binding=_binding(receipt_validated=False)), Reason.RECEIPT_NOT_ACCEPTED),
        (_tenant(binding=_binding(receipt_version="unknown")), Reason.RECEIPT_NOT_ACCEPTED),
        (_tenant(binding=_binding(pre_enable_head=None)), Reason.REPLAY_FENCE_MISSING),
    ],
)
def test_stale_membership_binding_and_receipt_are_holds(tenant, reason) -> None:
    (intent,) = plan_reconciliation(_snapshot(tenants=(tenant,)))
    assert intent.kind is IntentKind.HOLD and intent.reason is reason


@pytest.mark.parametrize("field,value", [("hub", "https://other.example"), ("account_id", "acc_b")])
def test_cross_namespace_tenant_snapshot_holds(field, value) -> None:
    valid, invalid = plan_reconciliation(
        _snapshot(tenants=(_tenant(), _tenant("tnt_2", **{field: value})))
    )
    assert valid.kind is IntentKind.CREATE
    assert invalid.kind is IntentKind.HOLD
    assert invalid.reason is Reason.TENANT_IDENTITY_MISMATCH


@pytest.mark.parametrize(
    "owner",
    [
        OwnerKey("https://other.example", "acc_a", "tnt_1", "ep_1"),
        OwnerKey(HUB, "acc_b", "tnt_1", "ep_1"),
        OwnerKey(HUB, "acc_a", "other_tenant", "ep_1"),
    ],
)
def test_binding_namespace_mismatch_is_not_an_attachment(owner) -> None:
    (intent,) = plan_reconciliation(_snapshot(tenants=(_tenant(binding=_binding(owner=owner)),)))
    assert intent.kind is IntentKind.HOLD and intent.reason is Reason.BINDING_IDENTITY_MISMATCH
    assert intent.owner is None and intent.generation is None


def test_binding_device_mismatch_is_a_hold() -> None:
    (intent,) = plan_reconciliation(
        _snapshot(tenants=(_tenant(binding=_binding(device_id="dev_2")),))
    )
    assert intent.reason is Reason.BINDING_IDENTITY_MISMATCH


def test_an_endpoint_only_difference_is_a_hold_not_an_attachment() -> None:
    """The owner namespace is JSON(hub, account, tenant, endpoint).

    A binding captured for another endpoint of the SAME tenant and account is a
    different owner; attaching it as current would hand one endpoint's
    membership to another. The snapshot differs only in endpoint_id here, so
    this is the negative the exact-namespace claim needs.
    """
    (intent,) = plan_reconciliation(
        _snapshot(endpoint_id="ep_other", tenants=(_tenant(binding=_binding()),))
    )
    assert intent.kind is IntentKind.HOLD
    assert intent.reason is Reason.ENDPOINT_MISMATCH
    assert intent.owner is None and intent.generation is None


def test_generation_is_required_to_attach_and_compared_when_supplied() -> None:
    """There is no "unknown generation" attachment.

    A contract enforces its own boundary: an absent expected generation is a
    coded hold, never ATTACH_CURRENT with the comparison delegated to the
    adapter. A supplied generation is compared, so a stale one cannot attach.
    """
    captured = _tenant(binding=_binding())
    (stale,) = plan_reconciliation(_snapshot(generation=2, tenants=(captured,)))
    assert stale.kind is IntentKind.HOLD
    assert stale.reason is Reason.GENERATION_MISMATCH
    assert stale.owner is None and stale.generation is None

    (exact,) = plan_reconciliation(_snapshot(generation=3, tenants=(captured,)))
    assert exact.kind is IntentKind.ATTACH_CURRENT
    assert exact.generation == 3

    (missing,) = plan_reconciliation(_snapshot(generation=None, tenants=(captured,)))
    assert missing.kind is IntentKind.HOLD
    assert missing.reason is Reason.GENERATION_NOT_CAPTURED
    assert missing.owner is None and missing.generation is None


def test_only_the_removal_hold_may_carry_a_missing_membership_revision() -> None:
    """The invariant is enforced on the KIND, not smuggled through the reason.

    The public constructor must refuse the pair the planner never emits: an
    ATTACH_CURRENT carrying TENANT_REMOVED, with or without a revision. A
    genuine removal hold is HOLD + TENANT_REMOVED and is the only shape allowed
    to omit the revision.
    """
    fields = dict(
        hub=HUB,
        account_id="acc_a",
        device_id="dev_1",
        snapshot_id="snapshot-10",
        tenant_id="tnt_1",
        receiver=None,
        owner=None,
        generation=None,
        pre_enable_head=None,
        receipt_version=None,
    )
    # The reviewer's exact hole: a mismatched kind/reason pair with no revision.
    with pytest.raises(ValueError):
        Intent(
            kind=IntentKind.ATTACH_CURRENT,
            reason=Reason.TENANT_REMOVED,
            membership_revision=None,
            **fields,
        )
    # Even with a revision, the pair itself is illegal.
    with pytest.raises(ValueError):
        Intent(
            kind=IntentKind.HOLD,
            reason=Reason.CURRENT_BINDING,
            membership_revision="membership-7",
            **fields,
        )
    removed = Intent(
        kind=IntentKind.HOLD,
        reason=Reason.TENANT_REMOVED,
        membership_revision=None,
        **fields,
    )
    assert removed.membership_revision is None


def test_a_locally_bound_tenant_removed_from_the_org_is_a_visible_hold() -> None:
    """Removed-from-org is HOLD, never no-intent.

    A local binding whose tenant is absent from the fresh authoritative snapshot
    is an account removed from that org. Silently omitting it looks identical to
    "nothing to do" while the user still has a binding; attaching is impossible
    because no fresh membership revision exists. A non-authoritative snapshot
    cannot prove absence, so it emits no removal hold.
    """
    removed = _binding("tnt_removed")
    result = plan_reconciliation(_snapshot(local_bindings=(removed,)))
    assert [i.tenant_id for i in result] == ["tnt_1", "tnt_removed"]
    held = result[1]
    assert held.kind is IntentKind.HOLD and held.reason is Reason.TENANT_REMOVED
    assert held.membership_revision is None
    assert held.owner == removed.owner and held.generation == 3
    assert held.pre_enable_head == 42 and held.receipt_version == V1

    uncertain = plan_reconciliation(_snapshot(authoritative=False, local_bindings=(removed,)))
    assert [i.tenant_id for i in uncertain] == ["tnt_1"]
    assert uncertain[0].reason is Reason.SNAPSHOT_NOT_AUTHORITATIVE


@pytest.mark.parametrize(
    "tenants",
    [
        (_tenant(), _tenant()),
        (_tenant(), _tenant(binding=_binding())),
        (_tenant(tenant_id=""),),
        (_tenant(membership_revision=""),),
        (_tenant(server_state="active"),),
        (_tenant(binding=_binding(generation=-1)),),
        (_tenant(binding=_binding(owner=OwnerKey(HUB, "acc_a", "tnt_1", ""))),),
    ],
)
def test_duplicate_or_invalid_rows_reject_without_partial_plan(tenants) -> None:
    with pytest.raises(ValueError):
        plan_reconciliation(_snapshot(tenants=tenants))


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"authoritative": False}, Reason.SNAPSHOT_NOT_AUTHORITATIVE),
        ({"active_owner_ambiguous": True}, Reason.OWNER_AMBIGUOUS),
        ({"legacy_account_done": True}, Reason.LEGACY_ACCOUNT_DONE),
        ({"receiver": None}, Reason.RECEIVER_UNBOUND),
    ],
)
def test_account_wide_uncertainty_is_visible_hold(changes, reason) -> None:
    intents = plan_reconciliation(_snapshot(tenants=(_tenant(), _tenant("tnt_2")), **changes))
    assert all(i.kind is IntentKind.HOLD and i.reason is reason for i in intents)


@pytest.mark.parametrize(
    "field,value",
    [
        ("hub", "https://other.example"),
        ("account_id", "acc_b"),
        ("device_id", "dev_2"),
    ],
)
def test_receiver_requires_proven_hub_account_device_binding(field, value) -> None:
    captured = _snapshot().receiver
    assert captured is not None
    receiver = replace(captured, **{field: value})
    (intent,) = plan_reconciliation(_snapshot(receiver=receiver))
    assert intent.reason is Reason.RECEIVER_IDENTITY_MISMATCH
    assert intent.kind is IntentKind.HOLD


def test_successful_per_org_bookkeeping_does_not_veto_another_org() -> None:
    snapshot = _snapshot(
        tenants=(_tenant(enrollment_recorded=True, binding=_binding()), _tenant("tnt_2"))
    )
    assert [i.kind for i in plan_reconciliation(snapshot)] == [
        IntentKind.ATTACH_CURRENT,
        IntentKind.CREATE,
    ]
    # A success marker without a current binding is uncertainty, not proof that
    # this process created an endpoint (nor permission to recreate that org).
    own, other = plan_reconciliation(
        replace(snapshot, tenants=(_tenant(enrollment_recorded=True), _tenant("tnt_2")))
    )
    assert own.reason is Reason.BOOKKEEPING_WITHOUT_BINDING
    assert other.kind is IntentKind.CREATE


def test_owner_namespace_matches_collision_safe_store_encoding() -> None:
    owner = OwnerKey(HUB, 'acc_"a', "tenant:1", "endpoint,1")
    assert owner.namespace == json.dumps(
        [HUB, 'acc_"a', "tenant:1", "endpoint,1"], separators=(",", ":")
    )
    assert owner.namespace != OwnerKey(HUB, 'acc_"a', "tenant", "1:endpoint,1").namespace


@pytest.mark.parametrize("count,budget", [(1, 1), (7, 2), (8, 3), (3, 20)])
def test_rotation_visits_every_owner_within_bounded_cycles(count, budget) -> None:
    owners = tuple(OwnerKey(HUB, "acc_a", f"tnt_{n}", f"ep_{n}") for n in range(count))
    cursor = None
    seen = set()
    for _ in range((count + budget - 1) // budget):
        rotation = select_rotation(owners[::-1], budget=budget, cursor=cursor)
        assert 0 < len(rotation.owners) <= budget
        assert len(set(rotation.owners)) == len(rotation.owners)
        seen.update(rotation.owners)
        cursor = rotation.cursor
    assert seen == set(owners)
    assert select_rotation(owners, budget=budget) == select_rotation(owners[::-1], budget=budget)


def test_rotation_cursor_handles_removed_owner_and_empty_set() -> None:
    owners = tuple(OwnerKey(HUB, "acc_a", f"tnt_{n}", "ep") for n in range(3))
    cursor = owners[1].namespace
    assert select_rotation((owners[0], owners[2]), budget=1, cursor=cursor).owners == (owners[2],)
    assert select_rotation((), budget=1, cursor=cursor).owners == ()
    assert select_rotation((), budget=1, cursor=cursor).cursor is None
    with pytest.raises(ValueError):
        select_rotation((owners[0], owners[0]), budget=1)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_rotation_rejects_nonpositive_or_noninteger_budget(budget) -> None:
    with pytest.raises(ValueError):
        select_rotation((), budget=budget)


def test_late_attach_cannot_invent_initial_fence_or_authorize_backlog() -> None:
    missing = _tenant(binding=_binding(pre_enable_head=None))
    (intent,) = plan_reconciliation(_snapshot(tenants=(missing,)))
    assert intent.kind is IntentKind.HOLD
    assert intent.reason is Reason.REPLAY_FENCE_MISSING
    assert intent.pre_enable_head is None
    # A known fence is preserved byte-for-byte as data, never replaced by a
    # fresh head or interpreted here as authorization to replay older messages.
    zero = _tenant(binding=_binding(pre_enable_head=0))
    (attached,) = plan_reconciliation(_snapshot(tenants=(zero,)))
    assert attached.pre_enable_head == 0
    assert attached.kind is IntentKind.ATTACH_CURRENT


@pytest.mark.parametrize(
    "changes",
    [
        {"authoritative": "false"},
        {"endpoint_id": ""},
        {"generation": 0},
        {"local_bindings": (_binding("tnt_1"),)},
        {"local_bindings": (_binding("tnt_9", device_id="dev_2"),)},
        {
            "local_bindings": (_binding("tnt_9"), _binding("tnt_9")),
        },
        {"tenants": (_tenant(membership_current="false"),)},
        {"offer": replace(Offer(V2, "sha256:offer", True, True), policy_accepted="false")},
    ],
)
def test_truthy_unvalidated_facts_cannot_authorize(changes) -> None:
    with pytest.raises(ValueError):
        plan_reconciliation(_snapshot(**changes))


def test_repeated_snapshot_is_deterministic_immutable_and_decision_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot()

    def refuse_io(*args, **kwargs):
        raise AssertionError("planner attempted file I/O")

    # Exercise both positive decisions under live I/O tripwires, not just a
    # source allowlist. No config/history/ledger may be read or persisted here.
    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", refuse_io)
        patch.setattr(os, "open", refuse_io)
        first = plan_reconciliation(snapshot)
        assert plan_reconciliation(snapshot) == first
        (attached,) = plan_reconciliation(_snapshot(tenants=(_tenant(binding=_binding()),)))
        assert attached.kind is IntentKind.ATTACH_CURRENT
    # Duplicate events repeat proposals; there is deliberately no executed/no-op
    # inference. The eventual executor needs its own revalidation/idempotency.
    assert first[0].kind is IntentKind.CREATE
    assert snapshot == _snapshot()
    with pytest.raises(FrozenInstanceError):
        setattr(snapshot, "preference", Preference.EXPLICIT_FALSE)
    with pytest.raises(FrozenInstanceError):
        setattr(first[0], "kind", IntentKind.HOLD)
    # The import allowlist is a negative architectural control, not a claim of
    # runtime enrollment QA: adding service/config/timer access must be explicit.
    tree = ast.parse(inspect.getsource(planner))
    imports = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports <= {"__future__", "dataclasses", "enum", "json"}
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not calls & {"open", "exec", "eval", "__import__", "Thread", "Timer"}

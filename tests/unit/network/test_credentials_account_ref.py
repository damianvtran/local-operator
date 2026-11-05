"""The owner's ``account_ref`` rides the mesh grant; a borrower never derives one.

A borrower sees a synthetic (negative) credential id and none of the owner's
store, so a ref hashed from them would collapse every key behind one owner and
differ from the ref the owner stamps for the same account. These drive the real
owner broker and borrower store (the ``borrowed`` rig) end to end.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from local_operator.harness.types import ModelSpec, StreamUsageEvent, Usage
from local_operator.network.credentials.types import Grant
from local_operator.providers.account_provenance import account_ref
from local_operator.providers.failover import _stamp_serving_spec
from tests.unit.network.test_credential_limit_skip import (  # noqa: F401 — fixtures by import
    borrowed,
    devices,
    idp,
    mesh,
    owner,
    stub_provider,
)
from tests.unit.network.test_credentials_owner import STUB_PROVIDER

_FIXTURES = (borrowed, owner, mesh, devices, idp, stub_provider)

MODEL = "claude-opus-5-5"


def _lend(rig: Any, session: str) -> dict[str, Any]:
    return rig.client.serve_grant(
        STUB_PROVIDER, provider=STUB_PROVIDER, session_id=session, model_id=MODEL
    )


def _stamped(access: Any) -> str | None:
    out = _stamp_serving_spec(
        StreamUsageEvent(usage=Usage(usd_cost=1)),
        ModelSpec(provider=STUB_PROVIDER, model_id=MODEL),
        access=access,
    )
    assert isinstance(out, StreamUsageEvent)
    return out.usage.account_ref


def _owner_ref(rig: Any, credential_id: int) -> str | None:
    row = next(r for r in rig.owner.auth.list_credentials(STUB_PROVIDER) if r.id == credential_id)
    return rig.owner.auth._account_ref_for_row(row)  # noqa: SLF001


def test_borrower_stamps_the_owners_ref_for_the_lent_row(borrowed: Any) -> None:
    rig = borrowed
    detail = _lend(rig, "equality")
    # The wire carries the owner's ref and it round-trips through the parser.
    wire_ref = detail["credential_ref"]["account_ref"]
    grant = Grant.from_detail(detail)
    assert grant.credential_ref.account_ref == wire_ref
    expected = _owner_ref(rig, grant.credential_ref.credential_id)
    assert expected is not None and wire_ref == expected

    access = rig.store._access_from_grant(grant)  # noqa: SLF001
    assert access.credential_id < 0
    assert access.account_ref == expected
    assert _stamped(access) == expected


def test_two_owner_rows_behind_one_owner_give_different_borrower_refs(borrowed: Any) -> None:
    from tests.unit.network.test_lend_selection_parity import (
        EMAIL_A,
        EMAIL_B,
        HOUR_MS,
        _usage,
    )

    rig = borrowed
    seen: dict[int, str | None] = {}
    # Move the owner's usage evidence so the owner lends each row in turn.
    for session, resets in (
        ("sib-a", {EMAIL_A: HOUR_MS, EMAIL_B: 4 * HOUR_MS}),
        ("sib-b", {EMAIL_A: 4 * HOUR_MS, EMAIL_B: HOUR_MS}),
    ):
        _usage(rig.owner.root, resets)
        grant = Grant.from_detail(_lend(rig, session))
        access = rig.store._access_from_grant(grant)  # noqa: SLF001
        # Both rows share one owner/provider, hence one synthetic id.
        assert access.credential_id < 0
        seen[grant.credential_ref.credential_id] = _stamped(access)
    assert set(seen) == {rig.row_a.id, rig.row_b.id}
    assert seen[rig.row_a.id] == _owner_ref(rig, rig.row_a.id)
    assert seen[rig.row_b.id] == _owner_ref(rig, rig.row_b.id)
    assert seen[rig.row_a.id] != seen[rig.row_b.id]


def test_old_owner_without_a_ref_leaves_a_key_unattributed(borrowed: Any) -> None:
    rig = borrowed
    detail = _lend(rig, "legacy-key")
    detail["credential_ref"].pop("account_ref")
    detail["credential_ref"]["kind"] = "api_key"
    grant = Grant.from_detail(detail)
    assert grant.credential_ref.account_ref == ""
    access = rig.store._access_from_grant(grant)  # noqa: SLF001
    assert access.account_ref is None
    # Not a ref hashed from the synthetic id.
    assert _stamped(access) is None


def test_old_owner_without_a_ref_still_yields_the_oauth_identity_ref(borrowed: Any) -> None:
    rig = borrowed
    detail = _lend(rig, "legacy-oauth")
    detail["credential_ref"].pop("account_ref")
    detail["identity"] = {"org_id": "org-1"}
    access = rig.store._access_from_grant(Grant.from_detail(detail))  # noqa: SLF001
    assert access.account_ref is None
    assert _stamped(access) == account_ref(
        STUB_PROVIDER, kind="oauth", credential_id=access.credential_id, org_id="org-1"
    )


def test_a_malformed_peer_ref_is_dropped() -> None:
    for junk in (
        "ABCDEF0123456789",
        "0123456789abcde",
        "0123456789abcdef0",
        "zzzzzzzzzzzzzzzz",
        "0123456789abcdef\n",
        "",
        None,
        7,
        ["0123456789abcdef"],
    ):
        grant = Grant.from_detail({"credential_ref": {"account_ref": junk}})
        assert grant.credential_ref.account_ref == "", junk
    ok = Grant.from_detail({"credential_ref": {"account_ref": "0123456789abcdef"}})
    assert ok.credential_ref.account_ref == "0123456789abcdef"


def test_stamp_ignores_a_namespace_less_object() -> None:
    # An access with no ref and no OAuth identity stays unattributed.
    assert _stamped(SimpleNamespace(kind="api_key", credential_id=-3)) is None

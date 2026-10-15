"""The consent reconciler (credentials/reconcile.py): projection, serve gate, push acceptor.

Reuses ``test_credentials_consent``'s rig: a real signed + executed approval (software
operator key, the existing approvals store), a real ``AuthStore`` and the real owner
broker. Nothing here fakes the consent check; the only stand-in is the relay object
(``server``), which the reconciler reads for ``root``/``identity``/``audit`` alone.
The two-relay, rotating-IdP, running-session proof is in
``test_credentials_reconcile_link.py``.
"""

from __future__ import annotations

import copy
import time
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import store
from local_operator.network.credentials import consent, placement, reconcile
from local_operator.network.credentials.types import DEVICE_BOUND_PROVIDERS
from tests.unit.network.test_credentials_consent import (  # noqa: F401 — fixture by import
    N,
    P,
    grant,
    request,
)
from tests.unit.network.test_credentials_consent import (  # noqa: F401 — fixture by import
    rig as consent_rig,
)

# The consent suite's rig, re-exported under the name the cells request. Importing it
# directly as ``rig`` trips F811 on every parameter that uses it.
rig = consent_rig


def _server(rig: Any) -> Any:
    return SimpleNamespace(root=rig.root, identity=rig.own, audit=None)


def _owner_ready(rig: Any) -> None:
    """Both transport capabilities, set the way an operator would (not by the reconciler)."""
    with store.mutate(N, rig.root) as record:
        record.self_capabilities = ["broker_credential"]
        store.save(record, rig.root)


def _fresh(rig: Any) -> None:
    """Owner has a login and NO placement row yet: the state before any share."""
    placement.placement_path(N, rig.root).unlink()


def _approve(rig: Any, **kw: Any) -> str:
    aid, run = request(rig, **kw)
    consent.provision_from_run(aid, run, rig.subject, rig.root)
    return aid


def _doc(rig: Any) -> placement.PlacementDocument:
    return placement.PlacementDocument.load(N, rig.root, self_device=rig.own.device_id)


def _inventory(rig: Any) -> dict[str, list[reconcile.NativeRow]]:
    found = reconcile.inventory(rig.root)
    assert found is not None
    return found


def _entry(rig: Any) -> Any:
    """The provider's placement entry, asserted present (so the cells read straight)."""
    entry = _doc(rig).entry(P)
    assert entry is not None
    return entry


def _bytes(rig: Any) -> bytes | None:
    path = placement.placement_path(N, rig.root)
    return path.read_bytes() if path.exists() else None


def _member_bytes(rig: Any) -> bytes:
    return store.record_path(N, rig.root).read_bytes()


def test_no_consent_file_is_one_stat_and_changes_nothing(rig: Any) -> None:
    _owner_ready(rig)
    _fresh(rig)
    members = _member_bytes(rig)
    result = reconcile.project_for(_server(rig), rig.peer.device_id)
    assert result.outcome == "idle" and _bytes(rig) is None
    assert _member_bytes(rig) == members


def test_prepared_or_disabled_consent_is_byte_identical(rig: Any) -> None:
    _owner_ready(rig)
    aid = _approve(rig)
    doc = consent.ConsentDocument(N, rig.own.device_id, rig.root)
    row = doc.load()
    row["recipients"][rig.peer.device_id]["enabled"] = False
    doc.save(row, row["revision"])
    _fresh(rig)
    members = _member_bytes(rig)
    result = reconcile.project_for(_server(rig), rig.peer.device_id)
    assert result.outcome == "idle" and _bytes(rig) is None, aid
    assert _member_bytes(rig) == members


def test_signed_future_consent_projects_a_derived_holder_once(rig: Any) -> None:
    _owner_ready(rig)
    aid = _approve(rig)
    _fresh(rig)
    members = _member_bytes(rig)
    first = reconcile.project_for(_server(rig), rig.peer.device_id)
    assert first.projected == [P], first
    holder = _entry(rig).holder(rig.peer.device_id)
    assert holder is not None and holder.granted_by == "consent:" + aid
    assert reconcile.is_derived(holder)
    rev = _entry(rig).doc_rev
    bytes_after = _bytes(rig)
    # Idempotent: a second pass (memo cleared to prove the logic, not the memo) is a no-op.
    server = _server(rig)
    again = reconcile.project_for(server, rig.peer.device_id)
    again2 = reconcile.project_for(server, rig.peer.device_id)
    assert again.projected == [] and again2.projected == []
    assert _entry(rig).doc_rev == rev and _bytes(rig) == bytes_after
    assert _member_bytes(rig) == members, "the reconciler must never write a capability"


@pytest.mark.parametrize("side", ["owner", "recipient"])
def test_missing_transport_capability_is_notready_and_mutates_nothing(rig: Any, side: str) -> None:
    _approve(rig)
    _fresh(rig)
    if side == "recipient":
        _owner_ready(rig)
        with store.mutate(N, rig.root) as record:
            member = record.member(rig.peer.device_id)
            assert member is not None
            member.capabilities = ["list"]
            store.save(record, rig.root)
    members = _member_bytes(rig)
    result = reconcile.project_for(_server(rig), rig.peer.device_id)
    assert result.outcome.startswith("notready:"), result
    assert "broker_credential" in result.outcome
    assert _bytes(rig) is None and _member_bytes(rig) == members


def test_existing_finite_holder_is_untouched(rig: Any) -> None:
    _owner_ready(rig)
    _approve(rig)  # rig placement already has an explicit holder for the peer
    before = _bytes(rig)
    result = reconcile.project_for(_server(rig), rig.peer.device_id)
    assert result.projected == [] and _bytes(rig) == before
    assert not reconcile.is_derived(_entry(rig).holder(rig.peer.device_id))


def test_only_native_oauth_classes_are_projected(rig: Any) -> None:
    """A broad signed selector still never projects static, device-bound or non-provider keys."""
    _owner_ready(rig)

    def broad(data: dict[str, Any]) -> None:
        data["inherit"]["providers"] = {"selector": "eligible-owner-native", "future_logins": True}

    _approve(rig, change=broad)
    _fresh(rig)
    rig.auth.upsert_credential("deepseek", {"type": "api_key", "key": "-".join(("static", "x"))})
    rig.auth.upsert_credential(
        "kimi",
        {"type": "oauth", "access": "a", "refresh": "r", "expires": 9999999999999, "email": "k@x"},
    )
    result = reconcile.project_for(_server(rig), rig.peer.device_id)
    assert result.projected == [P], result
    doc = _doc(rig)
    assert doc.entry("deepseek") is None and doc.entry("kimi") is None
    assert "kimi" in DEVICE_BOUND_PROVIDERS


def test_account_mismatch_projects_nothing(rig: Any) -> None:
    """The signed scope names account-one; a different account being signed in is no candidate."""
    _owner_ready(rig)
    _approve(rig)
    _fresh(rig)
    rig.auth.delete_credentials_for_provider(P)
    rig.auth.upsert_credential(
        P,
        {
            "type": "oauth",
            "email": "other@example.invalid",
            "access": "other-access",
            "refresh": "other-refresh",
            "expires": int((time.time() + 3600) * 1000),
            "account_id": "account-two",
            "org_id": "org-one",
        },
    )
    result = reconcile.project_for(_server(rig), rig.peer.device_id)
    assert result.projected == [] and result.refused.get(P) == "scope_mismatch", result
    assert _bytes(rig) is None


@pytest.mark.parametrize("selector", [{}, {"account_ref": "account-one"}])
def test_a_standing_deny_beats_the_approval(rig: Any, selector: dict[str, str]) -> None:
    _owner_ready(rig)
    _approve(rig)
    consent.restrict_and_revoke(
        N,
        rig.own.device_id,
        rig.peer.device_id,
        {"credential_key": P, **selector},
        "deny-1",
        rig.root,
    )
    _fresh(rig)
    result = reconcile.project_for(_server(rig), rig.peer.device_id)
    assert result.projected == [] and _bytes(rig) is None, result


def test_a_deny_landing_between_verify_and_lock_wins(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _owner_ready(rig)
    _approve(rig)
    _fresh(rig)
    real = reconcile._verify
    calls: list[int] = []

    def racing(*args: Any, **kwargs: Any) -> Any:
        out = real(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:  # the unlocked pass has just succeeded; now the operator denies
            consent.restrict_and_revoke(
                N, rig.own.device_id, rig.peer.device_id, {"credential_key": P}, "deny-2", rig.root
            )
        return out

    monkeypatch.setattr(reconcile, "_verify", racing)
    result = reconcile.project_for(_server(rig), rig.peer.device_id)
    assert result.projected == [] and len(calls) >= 2
    entry = _doc(rig).entry(P)
    assert entry is None or not entry.is_holder(rig.peer.device_id)


def test_copied_origin_and_unknown_marker_are_not_candidates(rig: Any) -> None:
    _owner_ready(rig)
    _approve(rig)
    _fresh(rig)
    rows = _inventory(rig)[P]
    assert len(rows) == 1 and rows[0].marker
    copied = [rows[0]._replace(copied=True)]
    unknown = [rows[0]._replace(marker="")]
    for case in (copied, unknown):
        permitted, code = reconcile._verify(
            N, rig.own.device_id, rig.peer.device_id, P, case, rig.root
        )
        assert permitted == [] and code == "scope_unknown"


def test_unknown_login_context_is_scope_unknown(rig: Any) -> None:
    """A native row with no account id cannot satisfy an account-constrained scope."""
    _owner_ready(rig)
    _approve(rig)
    rows = [_inventory(rig)[P][0]._replace(context={})]
    permitted, code = reconcile._verify(N, rig.own.device_id, rig.peer.device_id, P, rows, rig.root)
    assert permitted == [] and code == "scope_unknown"


def test_memo_skips_unchanged_work_and_reacts_to_a_new_login(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _owner_ready(rig)
    _approve(rig)
    _fresh(rig)
    rig.auth.delete_credentials_for_provider(P)  # no login yet: nothing to project
    server = _server(rig)
    verifies: list[int] = []
    real = consent.verified_permission

    def counting(*args: Any, **kwargs: Any) -> Any:
        verifies.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(consent, "verified_permission", counting)
    for _ in range(5):
        reconcile.project_for(server, rig.peer.device_id)
    assert verifies == [], "no login => no per-key verification at all"
    rig.auth.upsert_credential(
        P,
        {
            "type": "oauth",
            "email": "late@example.invalid",
            "access": "late-access",
            "refresh": "late-refresh",
            "expires": int((time.time() + 3600) * 1000),
            "account_id": "account-one",
            "org_id": "org-one",
        },
    )
    assert reconcile.project_for(server, rig.peer.device_id).projected == [P]
    spent = len(verifies)
    for _ in range(5):
        reconcile.project_for(server, rig.peer.device_id)
    assert len(verifies) == spent, "an unchanged pass must not re-verify"


# -- serve-time ---------------------------------------------------------------------


def _derive(rig: Any) -> str:
    _owner_ready(rig)
    aid = _approve(rig)
    _fresh(rig)
    assert reconcile.project_for(_server(rig), rig.peer.device_id).projected == [P]
    return aid


def test_derived_holder_is_served_the_approved_account(rig: Any) -> None:
    _derive(rig)
    reply = grant(rig)
    assert reply["kind"] == "grant", reply
    assert "refresh-marker" not in str(reply) and "refresh_token" not in reply


def test_derived_holder_never_gets_another_account_and_costs_zero_refreshes(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _derive(rig)
    # The approved login is replaced by a different (expired) account: nothing may be served
    # and nothing may be refreshed on the other account's behalf.
    rig.auth.delete_credentials_for_provider(P)
    rig.auth.upsert_credential(
        P,
        {
            "type": "oauth",
            "email": "other@example.invalid",
            "access": "other-access",
            "refresh": "other-refresh",
            "expires": int((time.time() - 3600) * 1000),
            "account_id": "account-two",
            "org_id": "org-one",
        },
    )
    posts: list[int] = []

    async def no_refresh(*args: Any, **kwargs: Any) -> Any:
        posts.append(1)
        raise AssertionError("an uncovered account must never be refreshed")

    monkeypatch.setattr(rig.auth, "ensure_oauth_fresh", no_refresh)
    reply = grant(rig)
    assert reply["kind"] == "error" and reply["code"] == "scope_mismatch", reply
    assert posts == [] and "access_token" not in reply
    assert "owner" in reply["message"].lower() or reply.get("owner_device")


def test_explicit_finite_holder_behaves_as_before_for_the_same_account_swap(rig: Any) -> None:
    """Control: an explicit share is per key, not per account — unchanged semantics."""
    rig.auth.delete_credentials_for_provider(P)
    rig.auth.upsert_credential(
        P,
        {
            "type": "oauth",
            "email": "other@example.invalid",
            "access": "other-access",
            "refresh": "other-refresh",
            "expires": int((time.time() + 3600) * 1000),
            "account_id": "account-two",
        },
    )
    assert grant(rig)["kind"] == "grant"


def test_derived_holder_skips_an_uncovered_row_and_is_served_the_covered_one(rig: Any) -> None:
    _derive(rig)
    rig.auth.upsert_credential(
        P,
        {
            "type": "oauth",
            "email": "other@example.invalid",
            "access": "other-access",
            "refresh": "other-refresh",
            "expires": int((time.time() + 3600) * 1000),
            "account_id": "account-two",
            "org_id": "org-two",
        },
    )
    rows = _inventory(rig)[P]
    assert len(rows) == 2, rows
    for _ in range(4):  # the cascade round-robins; the uncovered row must never win
        reply = grant(rig)
        assert reply["kind"] == "grant", reply
        assert reply["access_token"] != "other-access"


def test_a_later_deny_or_disable_stops_a_derived_holder(rig: Any) -> None:
    _derive(rig)
    assert grant(rig)["kind"] == "grant"
    doc = consent.ConsentDocument(N, rig.own.device_id, rig.root)
    row = doc.load()
    row["recipients"][rig.peer.device_id]["enabled"] = False
    doc.save(row, row["revision"])
    reply = grant(rig)
    assert reply["kind"] == "error" and reply["code"] == "consent_withdrawn", reply


# -- the push acceptor -----------------------------------------------------------------


def _push(rig: Any, mutate: Any = None, *, sender: str | None = None) -> dict[str, Any]:
    _derive(rig)
    row = _entry(rig).to_json()
    row["holders"] = [h for h in row["holders"]]
    document = {
        "schema": placement.PLACEMENT_SCHEMA,
        "network_id": N,
        "epoch": 0,
        "written_by": rig.own.device_id,
        "credentials": [row],
    }
    # Re-home the row onto the PEER as owner, and the rig's owner as the recipient.
    for holder in row["holders"]:
        holder["device"] = (
            rig.own.device_id if holder["device"] == rig.peer.device_id else holder["device"]
        )
    row["owner_device"] = rig.peer.device_id
    row["holders"] = [h for h in row["holders"] if h["device"] == rig.own.device_id]
    row["holders"].append(
        {"device": rig.peer.device_id, "scope": "device", "granted_at": 1.0, "granted_by": ""}
    )
    if mutate:
        mutate(document, row)
    link = SimpleNamespace(device_id=sender or rig.peer.device_id, network_id=N)
    placement.placement_path(N, rig.root).unlink()
    frame = {
        "op": "net_broker",
        "kind": "placement",
        "origin": reconcile.PUSH_ORIGIN,
        "from_device": sender or rig.peer.device_id,
        "document": document,
    }
    return reconcile.accept_push(rig.root, rig.own.device_id, link, frame)


def test_push_from_the_owner_is_merged(rig: Any) -> None:
    reply = _push(rig)
    assert reply["kind"] == "ack" and reply["changed"] == [P], reply
    assert _entry(rig).owner_device == rig.peer.device_id


@pytest.mark.parametrize(
    "name,mutate,code",
    [
        (
            "third-party owner",
            lambda d, r: r.__setitem__("owner_device", "d_" + "9" * 32),
            "not_owner",
        ),
        (
            "third-party holder",
            lambda d, r: r["holders"].append({"device": "d_" + "9" * 32, "scope": "session"}),
            "not_owner",
        ),
        ("static kind", lambda d, r: r.__setitem__("kind", "api-key-static"), "not_owner"),
        ("mcp key", lambda d, r: r.update(key="mcp:https://x", provider=""), "not_owner"),
        ("github key", lambda d, r: r.update(key="github", provider="github"), "not_owner"),
        (
            "other network",
            lambda d, r: d.__setitem__("network_id", "n_" + "2" * 24),
            "identity_mismatch",
        ),
        ("material field", lambda d, r: r.__setitem__("access_token", "x"), "malformed_frame"),
        (
            "too many",
            lambda d, r: d.__setitem__("credentials", [copy.deepcopy(r) for _ in range(40)]),
            "malformed_frame",
        ),
    ],
)
def test_hostile_push_is_refused_before_disk(rig: Any, name: str, mutate: Any, code: str) -> None:
    reply = _push(rig, mutate)
    assert reply["kind"] == "error" and reply["code"] == code, (name, reply)
    assert _bytes(rig) is None, name


def test_push_claiming_another_sender_is_refused(rig: Any) -> None:
    def forge(document: Any, row: Any) -> None:
        pass

    _derive(rig)
    link = SimpleNamespace(device_id=rig.peer.device_id, network_id=N)
    frame = {"kind": "placement", "from_device": "d_" + "8" * 32, "document": {}}
    assert (
        reconcile.accept_push(rig.root, rig.own.device_id, link, frame)["code"]
        == "identity_mismatch"
    )

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
from local_operator.network.types import MeshRefusal
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


def _push_link(device: str, *, negotiated: bool = True) -> Any:
    """A link as ``accept_push`` reads it; ``negotiated`` toggles the push capability."""
    from local_operator.network import wire

    caps = frozenset({wire.CREDENTIAL_PLACEMENT_PUSH_V1}) if negotiated else frozenset()
    return SimpleNamespace(
        device_id=device, network_id=N, capabilities=caps, local_capabilities=caps
    )


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
    row["owner_device"] = rig.peer.device_id
    row["holders"] = [
        {"device": rig.peer.device_id, "scope": "device", "granted_at": 1.0, "granted_by": ""},
        {"device": rig.own.device_id, "scope": "session", "granted_at": 1.0, "granted_by": ""},
    ]
    if mutate:
        mutate(document, row)
    link = _push_link(sender or rig.peer.device_id)
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
    link = _push_link(rig.peer.device_id)
    frame = {"kind": "placement", "from_device": "d_" + "8" * 32, "document": {}}
    assert (
        reconcile.accept_push(rig.root, rig.own.device_id, link, frame)["code"]
        == "identity_mismatch"
    )


# =====================================================================================
# Review round 1 / QA round 1 remediation (R1-R7, Q1-Q2)
# =====================================================================================


class _FakeLink:
    """An owner->recipient link as ``deliver`` reads it; ``replies`` are returned in order."""

    def __init__(self, rig: Any, replies: list[Any], *, negotiated: bool = True) -> None:
        from local_operator.network import wire

        caps = frozenset({wire.CREDENTIAL_PLACEMENT_PUSH_V1}) if negotiated else frozenset()
        self.network_id = N
        self.device_id = rig.peer.device_id
        self.capabilities = caps
        self.local_capabilities = caps
        self.replies = list(replies)
        self.frames: list[dict[str, Any]] = []

    def request(self, frame: dict[str, Any], timeout: float | None = None) -> Any:
        self.frames.append(frame)
        return self.replies.pop(0) if self.replies else _ACK


_ACK = {"op": "ack", "detail": {"kind": "ack", "key": "", "changed": []}}


def _owner_server(rig: Any) -> Any:
    server = _server(rig)
    server._next_relay_req = iter(range(1, 10**6)).__next__
    return server


def test_the_push_capability_is_advertised_with_a_real_handler() -> None:
    """R1: the push has its own string (never the shipped lifecycle one) and a real handler."""
    from local_operator.network import wire

    assert wire.CREDENTIAL_PLACEMENT_PUSH_V1 == "credential-placement-push-v1"
    assert wire.CREDENTIAL_PLACEMENT_PUSH_V1 != wire.CREDENTIAL_LIFECYCLE_V1
    assert wire.LINK_CAPABILITIES.count(wire.CREDENTIAL_PLACEMENT_PUSH_V1) == 1
    assert callable(reconcile.member_placement_push) and callable(reconcile.accept_push)


@pytest.mark.parametrize(
    "local,remote",
    [(True, False), (False, True), (False, False)],
)
def test_push_requires_the_capability_on_both_ends(local: bool, remote: bool) -> None:
    from local_operator.network import wire

    cap = wire.CREDENTIAL_PLACEMENT_PUSH_V1
    link = SimpleNamespace(
        capabilities=frozenset({cap}) if remote else frozenset({wire.CREDENTIAL_LIFECYCLE_V1}),
        local_capabilities=frozenset({cap}) if local else frozenset({wire.CREDENTIAL_LIFECYCLE_V1}),
    )
    assert reconcile.push_negotiated(link) is False
    assert (
        reconcile.push_negotiated(SimpleNamespace(capabilities={cap}, local_capabilities={cap}))
        is True
    )


def test_a_lifecycle_only_peer_gets_zero_frames_and_a_named_upgrade_remedy(rig: Any) -> None:
    """R1/Q1: the v0.71.3 shape (lifecycle string, no push string) receives NOTHING."""
    _derive(rig)
    from local_operator.network.audit import AuditLog

    server = _owner_server(rig)
    server.audit = AuditLog(rig.root)
    link = _FakeLink(rig, [], negotiated=False)
    outcome = reconcile.deliver(server, link, rig.peer.device_id, "inc-1")
    assert outcome == "skipped:peer_lacks_placement_push"
    assert link.frames == [], "a peer without the push capability must receive zero frames"
    reading = reconcile.status_for(server, rig.peer.device_id)
    assert "credential-placement-push-v1" in reading and "upgrade" in reading, reading
    assert reconcile.deliver(server, link, rig.peer.device_id, "inc-1") == outcome
    assert link.frames == []
    rows = [
        row for row in server.audit.tail() if row["event"] == "credential.permission_undelivered"
    ]
    assert len(rows) == 1, "unchanged upgrade status must not cause one audit row per tick"
    assert rows[0]["detail"] == {
        "act": rig.own.device_id,
        "sub": rig.peer.device_id,
        "code": "peer_lacks_placement_push",
    }


def test_the_receiver_refuses_an_unnegotiated_push_on_both_routes(rig: Any) -> None:
    """R1: a broker-less AND a broker-present receiver both refuse before any merge."""
    reply = _push_unnegotiated(rig)
    assert reply["kind"] == "error" and reply["code"] == "unsupported", reply
    assert _bytes(rig) is None
    # The legacy broker-present route: owner.on_broker must NOT hand this frame shape to the
    # unvalidated pull merge when the push capability was not negotiated.
    frame = {
        "op": "net_broker",
        "kind": "placement",
        "origin": reconcile.PUSH_ORIGIN,
        "req": 7,
        "from_device": rig.peer.device_id,
        "document": {"schema": 1, "network_id": N, "epoch": 0, "credentials": []},
    }
    link = _push_link(rig.peer.device_id, negotiated=False)
    detail = rig.broker.on_broker(link, frame)["detail"]
    assert detail["kind"] == "error" and detail["code"] == "unsupported", detail


def _push_unnegotiated(rig: Any) -> dict[str, Any]:
    _derive(rig)
    placement.placement_path(N, rig.root).unlink()
    link = _push_link(rig.peer.device_id, negotiated=False)
    frame = {"kind": "placement", "from_device": rig.peer.device_id, "document": {}}
    return reconcile.accept_push(rig.root, rig.own.device_id, link, frame)


def test_a_timeout_is_a_transient_failure_retried_next_tick_not_a_refusal(rig: Any) -> None:
    """R2: ``request`` returning ``None`` (timeout/dead send) must not start any backoff."""
    _derive(rig)
    server = _owner_server(rig)
    link = _FakeLink(rig, [None, None, _ACK])
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "failed"
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "failed"
    assert len(link.frames) == 2, "a timed-out push must be re-sent on the very next tick"
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "pushed:1"
    assert len(link.frames) == 3
    # Only a SUCCESS is remembered: nothing more is sent for an unchanged row...
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "in_sync"
    assert len(link.frames) == 3
    assert "delivery=" not in reconcile.status_for(server, rig.peer.device_id)


def test_a_failed_revoke_push_is_retried_and_never_forgotten(rig: Any) -> None:
    _derive(rig)
    server = _owner_server(rig)
    link = _FakeLink(rig, [_ACK])
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "pushed:1"
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": P}, "t-revoke", rig.root
    )
    link.replies = [None, None]
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "failed"
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "failed"
    sent = link.frames[-1]["document"]["credentials"][0]
    assert [h["device"] for h in sent["holders"]] == [rig.own.device_id], sent
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "pushed:1"
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "in_sync"


def test_only_a_typed_peer_refusal_backs_off_and_it_is_bounded_and_named(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _derive(rig)
    server = _owner_server(rig)
    refusal = {"op": "error", "code": "not_implemented", "message": "no"}
    link = _FakeLink(rig, [refusal])
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "refused:not_implemented"
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "refused:not_implemented"
    assert len(link.frames) == 1, "a typed not_implemented backs off; it is not hammered"
    assert "upgrade" in reconcile.status_for(server, rig.peer.device_id)
    clock = {"t": reconcile.time.monotonic()}
    monkeypatch.setattr(
        reconcile.time, "monotonic", lambda: clock["t"] + reconcile.REFUSED_RETRY_S + 1
    )
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "pushed:1"


def test_a_reconnect_clears_a_peer_refusal_backoff_at_once(rig: Any) -> None:
    """The peer may have been upgraded while disconnected: a new link incarnation re-asks."""
    _derive(rig)
    server = _owner_server(rig)
    refusal = {"op": "error", "code": "not_implemented", "message": "no"}
    link = _FakeLink(rig, [refusal, _ACK])
    assert reconcile.deliver(server, link, rig.peer.device_id, "old") == "refused:not_implemented"
    assert reconcile.deliver(server, link, rig.peer.device_id, "old") == "refused:not_implemented"
    assert len(link.frames) == 1
    assert reconcile.deliver(server, link, rig.peer.device_id, "new") == "pushed:1"


def test_a_codeless_transport_refusal_is_not_a_typed_policy_answer(rig: Any) -> None:
    """Without a typed cause, do not invent an authorization or upgrade decision."""
    _derive(rig)
    server = _owner_server(rig)
    link = _FakeLink(rig, [{"op": "error", "message": "denied"}, _ACK])
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "failed"
    assert "retrying next tick" in reconcile.status_for(server, rig.peer.device_id)
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "pushed:1"
    assert len(link.frames) == 2


def test_lifecycle_tick_returns_the_push_outcome(rig: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """R1: the tick's return value carries a stuck push (it used to be discarded)."""
    from local_operator.network.credentials import lifecycle

    _derive(rig)
    server = _owner_server(rig)
    link = _FakeLink(rig, [], negotiated=False)
    server._ensure_link = lambda device: link
    monkeypatch.setattr(lifecycle, "_announce_over", lambda *a, **k: "in_sync")
    from local_operator.network import definitions

    monkeypatch.setattr(definitions, "unholdable_capability", lambda *a, **k: "")
    out = lifecycle.lifecycle_tick_step(server, rig.peer.device_id)
    assert out == "in_sync+push:skipped:peer_lacks_placement_push", out


# -- R3 / R4: the acceptor's type and epoch discipline --------------------------------


@pytest.mark.parametrize(
    "name,mutate",
    [
        ("list device", lambda d, r: r["holders"][0].__setitem__("device", ["x"])),
        ("dict scope", lambda d, r: r["holders"][0].__setitem__("scope", {"a": 1})),
        ("int device", lambda d, r: r["holders"][1].__setitem__("device", 7)),
        ("list owner", lambda d, r: r.__setitem__("owner_device", ["x"])),
        ("str doc_rev", lambda d, r: r.__setitem__("doc_rev", "9")),
        ("bool doc_rev", lambda d, r: r.__setitem__("doc_rev", True)),
        ("holders not list", lambda d, r: r.__setitem__("holders", {"a": 1})),
        ("row not object", lambda d, r: d.__setitem__("credentials", ["x"])),
        ("too many holders", lambda d, r: r["holders"].extend([r["holders"][0]] * 9)),
    ],
)
def test_malformed_field_types_are_a_typed_refusal_and_zero_change(
    rig: Any, name: str, mutate: Any
) -> None:
    reply = _push(rig, mutate)
    assert reply["kind"] == "error" and reply["code"] != "internal", (name, reply)
    assert _bytes(rig) is None, name


@pytest.mark.parametrize("epoch", [1, 10**9, -1, 1.5, "0", True, None])
def test_a_sender_chosen_epoch_is_refused_not_persisted(rig: Any, epoch: Any) -> None:
    reply = _push(rig, lambda d, r: d.__setitem__("epoch", epoch))
    assert reply["kind"] == "error", (epoch, reply)
    assert _bytes(rig) is None, "a refused push must leave the placement file untouched"


def test_an_accepted_push_leaves_the_local_epoch_unchanged(rig: Any) -> None:
    reply = _push(rig)
    assert reply["kind"] == "ack"
    assert _doc(rig).epoch == 0


# -- R5: the mint-time guard, with the guard actually exercised -----------------------


def _flip_during_refresh(rig: Any, monkeypatch: pytest.MonkeyPatch, flip: Any) -> None:
    """Run ``flip`` AFTER the owner's resolve picked a row and BEFORE it mints."""
    original = rig.auth.get_oauth_access

    async def resolve_then_flip(*args: Any, **kwargs: Any) -> Any:
        access = await original(*args, **kwargs)
        flip()
        return access

    monkeypatch.setattr(rig.auth, "get_oauth_access", resolve_then_flip)


def test_approval_withdrawn_mid_refresh_wins_at_mint(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _derive(rig)

    def disable() -> None:
        doc = consent.ConsentDocument(N, rig.own.device_id, rig.root)
        row = doc.load()
        row["recipients"][rig.peer.device_id]["enabled"] = False
        doc.save(row, row["revision"])

    _flip_during_refresh(rig, monkeypatch, disable)
    reply = grant(rig)
    assert reply["kind"] == "error" and reply["code"] == "consent_withdrawn", reply
    assert "access_token" not in reply


def test_account_swap_mid_refresh_wins_at_mint(rig: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _derive(rig)

    def swap() -> None:
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

    _flip_during_refresh(rig, monkeypatch, swap)
    reply = grant(rig)
    assert reply["kind"] == "error" and reply["code"] == "scope_mismatch", reply
    assert "access_token" not in reply


def test_the_mint_guard_is_what_stops_a_mid_refresh_flip(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEGATIVE CONTROL: with ``derived_mint_check`` neutralised the same flip is served.

    Proves the two cells above fail when the guard is removed (review R5: the previous
    37 cells all passed with the guard deleted).
    """
    _derive(rig)
    monkeypatch.setattr(reconcile, "derived_mint_check", lambda *a, **k: None)

    def disable() -> None:
        doc = consent.ConsentDocument(N, rig.own.device_id, rig.root)
        row = doc.load()
        row["recipients"][rig.peer.device_id]["enabled"] = False
        doc.save(row, row["revision"])

    _flip_during_refresh(rig, monkeypatch, disable)
    assert grant(rig)["kind"] == "grant", "the control must be able to fail the guard cells"


# -- R6: local-only and purge ----------------------------------------------------------


def test_an_explicit_local_only_mark_is_not_projected(rig: Any) -> None:
    from local_operator.network.credentials.sync import MARK_LOCAL_ONLY, SyncState

    _owner_ready(rig)
    _approve(rig)
    _fresh(rig)
    marks = SyncState.load(N, rig.root)
    marks.record_mark(P, MARK_LOCAL_ONLY)
    marks.save()
    result = reconcile.project_for(_server(rig), rig.peer.device_id)
    assert result.projected == [] and _bytes(rig) is None, result


def test_a_disabled_approval_takes_back_only_the_derived_holder(rig: Any) -> None:
    _derive(rig)
    # An explicit share of a SECOND key to the same recipient must survive the purge.
    with placement.mutate(N, rig.root, self_device=rig.own.device_id) as doc:
        doc.declare(
            "explicitkey",
            owner_device=rig.own.device_id,
            provider="explicitkey",
            by=rig.own.device_id,
        )
        doc.grant("explicitkey", rig.peer.device_id, by=rig.own.device_id)
    row = consent.ConsentDocument(N, rig.own.device_id, rig.root)
    data = row.load()
    data["recipients"][rig.peer.device_id]["enabled"] = False
    row.save(data, data["revision"])
    server = _server(rig)
    reconcile.project_for(server, rig.peer.device_id)
    doc = _doc(rig)
    assert doc.is_holder(P, rig.peer.device_id) is False, "the derived holder must be taken back"
    assert doc.is_holder("explicitkey", rig.peer.device_id) is True, "explicit shares are untouched"
    assert doc.is_holder(P, rig.own.device_id) is True, "the owner stays its own holder"
    rev = _entry(rig).doc_rev
    reconcile.project_for(server, rig.peer.device_id)
    assert _entry(rig).doc_rev == rev, "the purge is idempotent"


def test_a_removed_consent_file_also_takes_back_the_derived_holder(rig: Any) -> None:
    _derive(rig)
    consent.ConsentDocument(N, rig.own.device_id, rig.root).path.unlink()
    reconcile.project_for(_server(rig), rig.peer.device_id)
    assert _doc(rig).is_holder(P, rig.peer.device_id) is False


def test_an_explicit_finite_holder_survives_a_missing_approval(rig: Any) -> None:
    """The purge is keyed on provenance: an owner-granted holder is not 'derived'."""
    _owner_ready(rig)
    _approve(rig)
    row = consent.ConsentDocument(N, rig.own.device_id, rig.root)
    data = row.load()
    data["recipients"][rig.peer.device_id]["enabled"] = False
    row.save(data, data["revision"])
    before = _bytes(rig)
    reconcile.project_for(_server(rig), rig.peer.device_id)
    assert _bytes(rig) == before
    assert _doc(rig).is_holder(P, rig.peer.device_id) is True


# -- R7: a failed read is retryable, not a decision -----------------------------------


@pytest.mark.parametrize(
    "code,expected",
    [
        ("scope_mismatch", "scope_mismatch"),
        ("scope_unknown", "scope_unknown"),
        ("consent_not_enabled", "consent_withdrawn"),
        ("approval_denied", "consent_withdrawn"),
        ("approval_record_tampered", "consent_withdrawn"),
        ("unknown_approval", "consent_unverifiable"),
        ("approval_anchor_unavailable", "consent_unverifiable"),
        ("consent_invalid", "consent_unverifiable"),
        ("busy", "consent_unverifiable"),
    ],
)
def test_wire_code_separates_a_decision_from_an_unreadable_store(code: str, expected: str) -> None:
    assert reconcile.wire_code(code) == expected


def test_an_unreadable_inventory_is_retryable_and_not_scope_unknown(rig: Any) -> None:
    _derive(rig)
    (rig.root / "auth.db").rename(rig.root / "auth.db.away")
    allowed, excluded, code = reconcile.serve_gate(
        N, rig.own.device_id, rig.peer.device_id, P, rig.root
    )
    assert (allowed, code) == ((), "consent_unverifiable"), (allowed, excluded, code)
    reply = grant(rig)
    assert reply["kind"] == "error" and reply["code"] == "consent_unverifiable", reply


@pytest.mark.parametrize("raw_io_error", [False, True])
def test_an_unreadable_approval_store_at_serve_time_is_retryable(
    rig: Any, monkeypatch: pytest.MonkeyPatch, raw_io_error: bool
) -> None:
    from local_operator.network import approvals

    _derive(rig)

    def unavailable(*a: Any, **k: Any) -> Any:
        if raw_io_error:
            raise OSError("approval store temporarily unreadable")
        raise MeshRefusal("unknown_approval", "no such approval right now")

    monkeypatch.setattr(approvals, "verify_for_run", unavailable)
    reply = grant(rig)
    assert reply["kind"] == "error" and reply["code"] == "consent_unverifiable", reply


def test_a_transient_read_failure_is_not_memoised_and_heals_on_the_next_tick(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network import approvals

    _owner_ready(rig)
    _approve(rig)
    _fresh(rig)
    server = _server(rig)
    real = approvals.verify_for_run
    monkeypatch.setattr(
        approvals, "verify_for_run", lambda *a, **k: (_ for _ in ()).throw(MeshRefusal("busy", "x"))
    )
    first = reconcile.project_for(server, rig.peer.device_id)
    assert first.projected == [] and "approval_unverifiable" in first.outcome, first
    assert "notready" in reconcile.status_for(server, rig.peer.device_id)
    monkeypatch.setattr(approvals, "verify_for_run", real)
    healed = reconcile.project_for(server, rig.peer.device_id)
    assert healed.projected == [P], "a failed read must not have been memoised as 'unchanged'"


def test_an_unreadable_auth_db_during_projection_is_notready_not_idle(rig: Any) -> None:
    _owner_ready(rig)
    _approve(rig)
    _fresh(rig)
    (rig.root / "auth.db").rename(rig.root / "auth.db.away")
    server = _server(rig)
    result = reconcile.project_for(server, rig.peer.device_id)
    assert result.outcome.startswith("notready:inventory_unreadable"), result
    (rig.root / "auth.db.away").rename(rig.root / "auth.db")
    assert reconcile.project_for(server, rig.peer.device_id).projected == [P]


def test_the_unverifiable_code_has_a_sentence_naming_the_owner() -> None:
    from local_operator.network.credentials.messages import render_broker_error
    from local_operator.network.credentials.types import BROKER_ERROR_CODES, BrokerError

    assert "consent_unverifiable" in BROKER_ERROR_CODES
    text = render_broker_error(
        BrokerError(
            code="consent_unverifiable", key=P, owner_device="d_x", owner_device_name="owner-box"
        ),
        key=P,
        owner_name="owner-box",
        provider=P,
    )
    assert "owner-box" in text and "momentary" in text.lower(), text


def test_the_new_audit_events_are_registered_and_keyless() -> None:
    from local_operator.network import audit

    for name in ("credential.permission_withdrawn", "credential.permission_undelivered"):
        assert name in audit.DETAIL_KEYS
        assert audit.DETAIL_KEYS[name] <= {"credential_key", "act", "sub", "code"}


def test_a_lost_first_ack_does_not_forget_a_later_revoke(rig: Any) -> None:
    """The peer may have applied a grant even though its ack never reached the owner."""
    _derive(rig)
    server = _owner_server(rig)
    link = _FakeLink(rig, [None, None, _ACK])
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "failed"
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": P}, "lost-ack-revoke", rig.root
    )
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "failed"
    row = link.frames[-1]["document"]["credentials"][0]
    assert rig.peer.device_id not in [holder["device"] for holder in row["holders"]]
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "pushed:1"
    assert reconcile.deliver(server, link, rig.peer.device_id, "i") == "in_sync"


def test_unverifiable_records_retryable_unavailability_not_withdrawal(rig: Any) -> None:
    from local_operator.network.credentials import effective
    from local_operator.network.credentials.types import BrokerError

    effective.note_resolution(
        rig.root,
        P,
        BrokerError(code="consent_unverifiable", key=P, owner_device=rig.own.device_id),
    )
    receipt = effective._receipts[(str(rig.root.resolve()), P)]  # noqa: SLF001 — recorded evidence
    assert receipt.state == "owner_unavailable"

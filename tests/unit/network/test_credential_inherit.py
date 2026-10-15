"""Operator authoring of a signed future-login permission, driven through the REAL CLI.

Every positive cell runs the shipped entrypoints in their real order -- ``credential inherit
--request-approval`` -> ``approvals approve`` -> ``credential inherit --approval`` ->
``--status`` -- through the real ``lop network`` parser. The only substitute is the signer:
``approvals.sign_decision`` is replaced by the existing software file-only operator key
(``test_approvals_store._sign``), because no Touch ID gesture exists headless. That gesture is
therefore NOT exercised here; everything the signature protects (digest, decision, anchor
provenance, signature verification at apply) is the real code.

Nothing here calls ``consent.provision_from_run`` or ``begin_run`` directly except the two
cells that deliberately prove a refusal AT those seams.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import approvals as A
from local_operator.network import onboard_approvals, store
from local_operator.network.credentials import consent, inheritance, placement
from local_operator.network.types import MeshRefusal
from local_operator.providers.auth_store import AuthStore
from tests.unit.network.test_approvals_store import _sign
from tests.unit.network.test_credentials_consent import N, P
from tests.unit.network.test_credentials_consent import rig as consent_rig
from tests.unit.network.test_credentials_real_link import _lop_network

# The existing software-authority fixture: owner root, one admin recipient, one native
# login (account-one / org-one) with a stamped incarnation, and an explicit placement row.
rig = consent_rig

PRIVATE_MARKERS = ("private-marker", "refresh-marker", "example.invalid")


def software_signer(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing software operator key in place of the Touch ID gesture (see module doc)."""

    def sign(**kw: Any) -> str:
        return _sign(
            root,
            A.signed_payload(
                kind=kw["kind"],
                request_id=kw["request_id"],
                request_digest=kw["request_digest"],
                decision=kw.get("decision", "approve"),
                decided_at=kw["decided_at"],
            ),
        )

    monkeypatch.setattr(A, "sign_decision", sign)


def with_owner_transport(rig: Any) -> None:
    """Both broker rows present, as an operator would have set them with ``member grant``."""
    record = store.load(N, rig.root)
    record.self_capabilities = ["broker_credential"]
    store.save(record, rig.root)


@pytest.fixture
def ready(rig: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    software_signer(rig.root, monkeypatch)
    with_owner_transport(rig)
    return rig


def base(rig: Any, *extra: str) -> list[str]:
    return ["credential", "inherit", P, "--to", rig.peer.device_id, "--network", N, *extra]


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    capsys.readouterr()
    code = _lop_network(*argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def request_id(rig: Any, capsys: pytest.CaptureFixture[str]) -> str:
    code, out, err = run(capsys, *base(rig, "--request-approval", "--json"))
    assert code == 0, err
    return json.loads(out)["approval_id"]


def policy_bytes(rig: Any) -> bytes | None:
    path = consent.ConsentDocument(N, rig.own.device_id, rig.root).path
    return path.read_bytes() if path.exists() else None


def snapshot(rig: Any) -> dict[str, bytes | None]:
    """Everything an inheritance REQUEST/refused APPLY must leave byte-identical."""
    return {
        "policy": policy_bytes(rig),
        "network": store.record_path(N, rig.root).read_bytes(),
        "placement": placement.placement_path(N, rig.root).read_bytes(),
    }


def approve(rig: Any, aid: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    return run(capsys, "approvals", "approve", aid, "--json")


def status(rig: Any, capsys: pytest.CaptureFixture[str], *extra: str) -> dict[str, Any]:
    code, out, err = run(capsys, *base(rig, "--status", "--json", *extra))
    assert code == 0, err
    return json.loads(out)


# ---------------------------------------------------------------------------
# The product flow, end to end
# ---------------------------------------------------------------------------


def test_request_approve_apply_status_through_the_real_entrypoints(
    ready: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = ready
    assert status(rig, capsys)["state"] == "no-consent"
    assert "authentication gap" in "\n".join(status(rig, capsys)["lines"])

    members = store.record_path(N, rig.root).read_bytes()
    code, out, err = run(capsys, *base(rig, "--request-approval", "--json"))
    assert code == 0, err
    requested = json.loads(out)
    aid = requested["approval_id"]
    assert requested["state"] == "requested" and "borrow your" in requested["summary"]
    # A request signs and writes nothing but its own record.
    assert policy_bytes(rig) is None
    assert store.record_path(N, rig.root).read_bytes() == members
    assert status(rig, capsys)["state"] == "awaiting-approval"

    # The generic runner can never launch SSH onboarding over this record.
    code, out, err = run(capsys, "approvals", "run", aid, "--json")
    assert code == 1 and "credential inherit" in err and "no connection" in err

    code, out, err = approve(rig, aid, capsys)
    assert code == 0
    assert json.loads(out)["state"] == "approved"  # stdout stays ONE json object
    # The full disclosure is on stderr BEFORE signing, derived from the signed record.
    assert "Account: account-one" in err
    assert f"Recipient device: {rig.peer.device_id}" in err
    assert "Recipient key fingerprint: sha256:" in err
    assert "UNTIL YOU REVOKE IT" in err and "NOT the lifetime" in err
    assert "credential revoke" in err and "broker_credential" in err
    assert policy_bytes(rig) is None, "approving records a decision, not the permission"
    state = status(rig, capsys)
    assert state["state"] == "approved-not-applied" and state["approval_id"] == aid

    before = snapshot(rig)
    code, out, err = run(
        capsys, *base(rig, "--account", "account-one", "--approval", aid, "--json")
    )
    assert code == 0, err
    applied = json.loads(out)
    assert applied["state"] == "connected" and applied["activation"].startswith("static_storage")
    row = consent.ConsentDocument(N, rig.own.device_id, rig.root).load()["recipients"][
        rig.peer.device_id
    ]
    assert row["enabled"] is True and row["approval"]["ref"] == aid
    assert row["inherit"]["providers"] == {"ids": [P], "future_logins": True}
    assert row["inherit"]["mcp"] == {"urls": [], "future_logins": False}
    assert row["inherit"]["origin"] == "owner-native-login"
    assert "selector" not in json.dumps(row["inherit"])
    # Capability record and placement are byte-identical: authoring never writes either.
    after = snapshot(rig)
    assert after["network"] == before["network"] and after["placement"] == before["placement"]

    record = A.load_record(aid, root=rig.root)
    assert record["state"] == "connected"
    assert any(r["step"] == consent.STEP and r["ok"] for r in record["receipts"])

    ready_state = status(rig, capsys)
    assert ready_state["state"] == "ready"
    text = "\n".join(ready_state["lines"] + [ready_state["scope"]])
    assert "PERMISSION only" in text and "until revoked" in text
    assert "account account-one" in text and "credential revoke" in text

    # The card stays honest after its own one-hour window: connected, never `expired`, and
    # the permission (which has no TTL) still verifies.
    real_time = time.time
    monkeypatch.setattr(A.time, "time", lambda: real_time() + 3 * 3600)
    assert A.load_record(aid, root=rig.root)["state"] == "connected"
    A.sweep(root=rig.root)
    assert A.load_record(aid, root=rig.root)["state"] == "connected"
    consent.verified_permission(
        N, rig.own.device_id, rig.peer.device_id, P, {"account_id": "account-one"}, rig.root
    )

    # One-shot: a replay refuses and changes nothing.
    frozen = snapshot(rig)
    code, out, err = run(capsys, *base(rig, "--account", "account-one", "--approval", aid))
    assert code == 1 and "already" in err
    assert snapshot(rig) == frozen


def test_no_key_material_reaches_output_record_policy_or_audit(
    ready: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = ready
    aid = request_id(rig, capsys)
    code, out, err = approve(rig, aid, capsys)
    assert code == 0
    code, applied_out, applied_err = run(capsys, *base(rig, "--approval", aid, "--json"))
    assert code == 0
    pieces = [out, err, applied_out, applied_err, json.dumps(status(rig, capsys))]
    pieces.append(json.dumps(A.load_record(aid, root=rig.root)))
    pieces.append((policy_bytes(rig) or b"").decode())
    audit = store.audit_path(rig.root)
    events = [json.loads(line) for line in audit.read_text().splitlines()] if audit.exists() else []
    pieces.append(json.dumps(events))
    joined = "\n".join(pieces)
    for marker in PRIVATE_MARKERS:
        assert marker not in joined, f"{marker!r} leaked"
    names = {e["event"] for e in events}
    assert {"credential.inherit_requested", "credential.inherit_applied"} <= names
    for event in events:
        if event["event"].startswith("credential.inherit_"):
            assert set(event["detail"]) == {"credential_key", "act", "sub", "approval_id"}


def test_the_card_never_calls_the_permission_expiring(
    ready: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = ready
    code, out, err = run(capsys, *base(rig, "--request-approval"))
    assert code == 0
    text = out.lower()
    assert "until you revoke it" in text and "no automatic expiry" in text
    assert "not the lifetime of the permission" in text
    # Every sentence that mentions an hour/window is about the CARD.
    for line in out.splitlines():
        if "window" in line.lower() or "hour" in line.lower():
            assert line.startswith("Request window:"), line


# ---------------------------------------------------------------------------
# Request refusals: nothing is filed
# ---------------------------------------------------------------------------


def records(rig: Any) -> list[dict[str, Any]]:
    return A.list_records(root=rig.root)


@pytest.mark.parametrize("which", ["owner", "recipient"])
def test_missing_broker_row_refuses_with_the_exact_remedy_and_zero_mutation(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], which: str
) -> None:
    software_signer(rig.root, monkeypatch)
    record = store.load(N, rig.root)
    if which == "recipient":
        record.self_capabilities = ["broker_credential"]
        record.member(rig.peer.device_id).capabilities = ["list"]  # type: ignore[union-attr]
    store.save(record, rig.root)
    before = snapshot(rig)
    code, out, err = run(capsys, *base(rig, "--request-approval"))
    assert code == 1
    assert f"{which}_lacks_broker_credential" in err and "member grant" in err
    assert records(rig) == [] and snapshot(rig) == before


@pytest.mark.parametrize(
    "provider,needle",
    [
        ("github", "not a native OAuth login class"),
        ("kimi", "not a native OAuth login class"),
        ("mcp-oauth", "not a native OAuth login class"),
        ("Bad Name", "not a native OAuth login class"),
        ("never-signed-in", "no native sign-in"),
    ],
)
def test_ineligible_or_absent_provider_refuses_and_files_nothing(
    ready: Any, capsys: pytest.CaptureFixture[str], provider: str, needle: str
) -> None:
    rig = ready
    code, out, err = run(
        capsys,
        "credential", "inherit", provider, "--to", rig.peer.device_id, "--network", N,
        "--request-approval",
    )  # fmt: skip
    assert code == 1 and needle in err, err
    assert records(rig) == [] and policy_bytes(rig) is None


def test_noisy_inventory_still_binds_exactly_the_one_native_account(
    ready: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = ready
    auth = AuthStore(db_path=rig.root / "auth.db")
    try:
        auth.upsert_credential(
            "openai", {"type": "api_key", "key": "static-key-marker", "account_id": "static-acct"}
        )
        auth.upsert_credential(
            "anthropic",
            {
                "type": "oauth",
                "access": "a",
                "refresh": "r",
                "expires": int((time.time() + 3600) * 1000),
                "account_id": "other-provider-account",
            },
        )
        auth.upsert_credential(
            P,
            {
                "type": "oauth",
                "access": "borrowed",
                "refresh": "borrowed",
                "expires": int((time.time() + 3600) * 1000),
                "account_id": "copied-account",
                "mesh_origin": {"owner": "elsewhere"},
            },
        )
    finally:
        auth.close()
    aid = request_id(rig, capsys)
    scope = inheritance._parse(A.load_record(aid, root=rig.root))
    assert (scope["provider"], scope["account"], scope["org"]) == (P, "account-one", "")
    assert "copied-account" not in json.dumps(A.load_record(aid, root=rig.root))


def test_two_native_accounts_require_an_explicit_choice(
    ready: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = ready
    auth = AuthStore(db_path=rig.root / "auth.db")
    try:
        auth.upsert_credential(
            P,
            {
                "type": "oauth",
                "access": "x",
                "refresh": "y",
                "expires": int((time.time() + 3600) * 1000),
                "account_id": "account-two",
            },
        )
    finally:
        auth.close()
    code, out, err = run(capsys, *base(rig, "--request-approval"))
    assert code == 1 and "choose one with --account" in err and records(rig) == []
    code, out, err = run(capsys, *base(rig, "--request-approval", "--account", "nope"))
    assert code == 1 and "account 'nope'" in err and records(rig) == []
    code, out, err = run(capsys, *base(rig, "--request-approval", "--account", "account-two"))
    assert code == 0, err
    assert "Account: account-two" in out


def test_org_must_belong_to_the_chosen_account(
    ready: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = ready
    code, out, err = run(capsys, *base(rig, "--request-approval", "--org", "not-this-org"))
    assert code == 1 and "org 'not-this-org'" in err and records(rig) == []
    code, out, err = run(capsys, *base(rig, "--request-approval", "--org", "org-one", "--json"))
    assert code == 0, err
    parsed = inheritance._parse(A.load_record(json.loads(out)["approval_id"], root=rig.root))
    assert parsed["org"] == "org-one"


def test_a_second_open_request_for_the_same_grant_is_refused(
    ready: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = ready
    aid = request_id(rig, capsys)
    code, out, err = run(capsys, *base(rig, "--request-approval"))
    assert code == 1 and aid in err and len(records(rig)) == 1


def test_a_revoked_recipient_is_not_regranted_and_the_deny_survives(
    ready: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = ready
    aid = request_id(rig, capsys)
    approve(rig, aid, capsys)
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": P}, "owner-revoke:t", rig.root
    )
    frozen = snapshot(rig)
    # Neither a new request nor the already-signed one can clear the sticky withdrawal.
    code, out, err = run(capsys, *base(rig, "--request-approval"))
    assert code == 1 and "withdrawn" in err
    code, out, err = run(capsys, *base(rig, "--approval", aid))
    assert code == 1 and "withdrawn" in err
    assert snapshot(rig) == frozen
    assert A.load_record(aid, root=rig.root)["state"] == "approved", "refused before begin_run"
    assert status(rig, capsys)["state"] == "withdrawn"


def test_an_existing_standing_permission_is_not_silently_replaced(
    ready: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = ready
    aid = request_id(rig, capsys)
    approve(rig, aid, capsys)
    assert run(capsys, *base(rig, "--approval", aid))[0] == 0
    code, out, err = run(capsys, *base(rig, "--request-approval"))
    assert code == 1 and "already holds a standing inherit permission" in err


# ---------------------------------------------------------------------------
# Apply refusals: before any policy write
# ---------------------------------------------------------------------------


def settle(rig: Any, capsys: pytest.CaptureFixture[str], *, signed: bool = True) -> str:
    aid = request_id(rig, capsys)
    if signed:
        assert approve(rig, aid, capsys)[0] == 0
    return aid


@pytest.mark.parametrize(
    "bad",
    [
        "unapproved",
        "denied",
        "expired",
        "tampered-digest",
        "tampered-signature",
        "wrong-provider",
        "wrong-recipient",
        "wrong-account-arg",
        "fingerprint-changed",
        "login-account-changed",
        "owner-broker-row-removed",
        "recipient-broker-row-removed",
        "recipient-removed",
    ],
)
def test_every_invalid_apply_fails_before_any_policy_write(
    ready: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    rig = ready
    aid = settle(rig, capsys, signed=bad != "unapproved")
    argv = base(rig, "--approval", aid)
    if bad == "denied":
        A.deny(aid, root=rig.root)
    if bad == "expired":
        real = time.time
        monkeypatch.setattr(A.time, "time", lambda: real() + 2 * 3600)
        monkeypatch.setattr(onboard_approvals.time, "time", lambda: real() + 2 * 3600)
    if bad == "tampered-digest":
        record = A.load_record(aid, root=rig.root)
        record["what"][inheritance.OP]["scope"]["account_id"] = "someone-else"
        A._write_record(record, rig.root)
    if bad == "tampered-signature":
        record = A.load_record(aid, root=rig.root)
        record["signature"]["sig"] = "00" * 64
        A._write_record(record, rig.root)
    if bad == "wrong-provider":
        argv[2] = "anthropic"
    if bad == "wrong-recipient":
        argv[4] = rig.own.device_id
    if bad == "wrong-account-arg":
        argv += ["--account", "account-other"]
    if bad == "fingerprint-changed":
        record = store.load(N, rig.root)
        member = record.member(rig.peer.device_id)
        assert member is not None
        member.public_key = rig.own.public_key
        store.save(record, rig.root)
    if bad == "login-account-changed":
        auth = AuthStore(db_path=rig.root / "auth.db")
        try:
            auth.delete_credentials_for_provider(P)
            auth.upsert_credential(
                P,
                {
                    "type": "oauth",
                    "access": "n",
                    "refresh": "n",
                    "expires": int((time.time() + 3600) * 1000),
                    "account_id": "different-account",
                },
            )
        finally:
            auth.close()
    if bad == "owner-broker-row-removed":
        record = store.load(N, rig.root)
        record.self_capabilities = []
        store.save(record, rig.root)
    if bad == "recipient-broker-row-removed":
        record = store.load(N, rig.root)
        record.member(rig.peer.device_id).capabilities = ["list"]  # type: ignore[union-attr]
        store.save(record, rig.root)
    if bad == "recipient-removed":
        record = store.load(N, rig.root)
        record.member(rig.peer.device_id).removed_at = time.time()  # type: ignore[union-attr]
        store.save(record, rig.root)
    frozen = snapshot(rig)
    state_before = A.load_record(aid, root=rig.root)["state"]
    code, out, err = run(capsys, *argv)
    assert code == 1, (bad, out, err)
    assert snapshot(rig) == frozen, f"{bad}: a refused apply wrote policy/network/placement"
    state_after = A.load_record(aid, root=rig.root)["state"]
    assert state_after in {state_before, "expired"}
    assert state_after not in {"connecting", "connected", "failed"}
    assert policy_bytes(rig) is None


def test_apply_with_a_failed_signature_leaves_a_retry_eligible_record_not_a_stuck_run(
    ready: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = ready
    aid = settle(rig, capsys)
    real = consent.provision_from_run

    def boom(*a: Any, **k: Any) -> Any:
        raise MeshRefusal("consent_conflict", "forced")

    monkeypatch.setattr(consent, "provision_from_run", boom)
    code, out, err = run(capsys, *base(rig, "--approval", aid))
    assert code == 1 and "consent_conflict" in err
    assert A.load_record(aid, root=rig.root)["state"] == "failed"
    monkeypatch.setattr(consent, "provision_from_run", real)
    assert run(capsys, *base(rig, "--approval", aid))[0] == 0
    assert A.load_record(aid, root=rig.root)["state"] == "connected"


# ---------------------------------------------------------------------------
# The generic-runner trap and the desktop route
# ---------------------------------------------------------------------------


def test_begin_run_without_the_authoring_operation_is_refused_for_this_record(
    ready: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = ready
    aid = settle(rig, capsys)
    with pytest.raises(MeshRefusal) as raised:
        onboard_approvals.begin_run(aid, root=rig.root)
    assert raised.value.code == "inherit_apply_required"
    assert A.load_record(aid, root=rig.root)["state"] == "approved"


def test_approvals_run_never_launches_the_ssh_runner_for_this_record(
    ready: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.network import onboard

    def explode(*a: Any, **k: Any) -> Any:
        raise AssertionError("an SSH transport/runner was constructed")

    monkeypatch.setattr(onboard, "SshTransport", explode)
    monkeypatch.setattr(onboard, "step_runner", explode, raising=False)
    rig = ready
    aid = settle(rig, capsys)
    code, out, err = run(capsys, "approvals", "run", aid)
    assert code == 1 and "inherit" in err and "SSH" in err
    assert A.load_record(aid, root=rig.root)["state"] == "approved"
    assert policy_bytes(rig) is None


def test_approvals_show_and_list_describe_a_permission_not_an_onboarding(
    ready: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = ready
    aid = request_id(rig, capsys)
    for argv in (("approvals", "show", aid), ("approvals", "list")):
        code, out, err = run(capsys, *argv)
        assert code == 0
        assert "Recipient key fingerprint" in out
        for forbidden in ("install anchor", "definitions, MCP servers", "provision"):
            assert forbidden not in out, (argv, forbidden)


# ---------------------------------------------------------------------------
# Status states and not-ready reasons (permission, never delivery)
# ---------------------------------------------------------------------------


def test_status_of_an_ineligible_class_is_not_applicable_and_suggests_no_refresh(
    ready: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = ready
    before = snapshot(rig)
    code, out, err = run(
        capsys,
        "credential", "inherit", "github", "--to", rig.peer.device_id, "--network", N, "--status",
        "--json",
    )  # fmt: skip
    assert code == 0, err
    report = json.loads(out)
    text = "\n".join(report["lines"])
    assert report["state"] == "not-applicable" and "credential share" in text
    assert "refresh" not in text.replace("no sign-in or refresh changes that", "")
    assert snapshot(rig) == before


def test_status_is_read_only_and_creates_nothing(
    rig: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    before = sorted(str(p.relative_to(rig.root)) for p in rig.root.rglob("*"))
    snap = snapshot(rig)
    report = status(rig, capsys)
    assert report["state"] == "no-consent"
    assert sorted(str(p.relative_to(rig.root)) for p in rig.root.rglob("*")) == before
    assert snapshot(rig) == snap


def test_ready_is_named_notready_for_a_changed_account_a_gap_or_a_missing_row(
    ready: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = ready
    aid = settle(rig, capsys)
    assert run(capsys, *base(rig, "--approval", aid))[0] == 0
    assert status(rig, capsys)["state"] == "ready"

    record = store.load(N, rig.root)
    record.self_capabilities = []
    store.save(record, rig.root)
    report = status(rig, capsys)
    assert report["state"] == "notready"
    assert "owner_lacks_broker_credential" in "\n".join(report["lines"])
    assert "permission: until revoked" in "\n".join(report["lines"])
    record.self_capabilities = ["broker_credential"]
    store.save(record, rig.root)

    auth = AuthStore(db_path=rig.root / "auth.db")
    try:
        auth.delete_credentials_for_provider(P)
        assert status(rig, capsys)["state"] == "notready"
        assert "no_native_login" in "\n".join(status(rig, capsys)["lines"])
        auth.upsert_credential(
            P,
            {
                "type": "oauth",
                "access": "n",
                "refresh": "n",
                "expires": int((time.time() + 3600) * 1000),
                "account_id": "account-two",
            },
        )
    finally:
        auth.close()
    report = status(rig, capsys)
    text = "\n".join(report["lines"])
    assert report["state"] == "notready" and "scope_mismatch" in text
    assert "approved for account account-one; signed in now as account-two" in text


# ---------------------------------------------------------------------------
# Desktop surfaces: the app cannot approve what it cannot state
# ---------------------------------------------------------------------------


def test_the_desktop_route_refuses_to_sign_a_credential_permission(
    ready: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.server.utils import desktop_approvals as desktop

    rig = ready
    aid = request_id(rig, capsys)

    def must_not_sign(**_: Any) -> str:
        raise AssertionError("the desktop path reached the signer")

    monkeypatch.setattr(A, "sign_decision", must_not_sign)
    with pytest.raises(MeshRefusal) as raised:
        desktop.approve(rig.root, aid)
    assert raised.value.code == "inherit_terminal_approval_required"
    assert f"lop network approvals approve {aid}" in str(raised.value)
    assert A.load_record(aid, root=rig.root)["state"] == "requested"
    # Deny is the safe direction and stays available from the app.
    assert desktop.deny(rig.root, aid)["state"] == "denied"


def test_desktop_rows_carry_optional_keyless_extras_only_for_this_kind(
    ready: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.server.utils import desktop_approvals as desktop
    from tests.unit.network.test_approvals_store import _device_request

    rig = ready
    aid = request_id(rig, capsys)
    plain = A.create_request(**_device_request(A.new_request_id()), root=rig.root)
    rows = {row["approval_id"]: row for row in desktop.approval_rows(rig.root)}
    row = rows[aid]
    assert row["can_approve_here"] is False
    assert f"lop network approvals approve {aid}" in row["approve_remedy"]
    assert row["summary"].startswith("Let ") and "SAME account" in row["summary"]
    assert any(line.startswith("Duration: UNTIL YOU REVOKE IT") for line in row["details"])
    assert not any(marker in json.dumps(row) for marker in PRIVATE_MARKERS)
    # The builder's own frozen key set (``machine`` is a ``local_authority`` key; the route's
    # model adds it as ``None``): an ordinary row gains NOTHING.
    assert set(rows[plain["approval_id"]]) == set(A.badge_row(plain))


def test_an_undescribable_record_is_never_presented_as_approvable(
    ready: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = ready
    aid = request_id(rig, capsys)
    record = A.load_record(aid, root=rig.root)
    record["what"][inheritance.OP]["inherit"]["providers"] = {"selector": "eligible-owner-native"}
    A._write_record(record, rig.root)
    extras = inheritance.row_extras(A.load_record(aid, root=rig.root))
    assert extras is not None and extras["can_approve_here"] is False
    assert "cannot describe" in extras["summary"]
    with pytest.raises(MeshRefusal):
        inheritance.details(A.load_record(aid, root=rig.root))
    code, out, err = run(capsys, "approvals", "approve", aid, "--json")
    assert code == 1 and "will not be approved or applied" in err
    assert A.load_record(aid, root=rig.root)["state"] == "requested"


def test_offload_readiness_is_unchanged_by_authoring(
    ready: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """No permission leaves read-only pairing intact: no placement/capability is written."""
    rig = ready
    before = snapshot(rig)
    request_id(rig, capsys)
    status(rig, capsys)
    after = snapshot(rig)
    assert after["network"] == before["network"] and after["placement"] == before["placement"]

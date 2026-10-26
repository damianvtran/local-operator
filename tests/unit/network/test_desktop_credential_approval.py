"""Desktop review -> sign -> apply for credential approvals, through the REAL routes.

Every cell drives the bearer-gated FastAPI router over the existing isolated software
authority rig (``test_credentials_consent.rig``): real approval store, real consent
policy, real placement, real signature verification at write and at apply. Two seams
are substituted, both named:

* the SIGNER: ``approvals.sign_decision`` is the rig's software file-only key (no Touch
  ID exists headless). The "presence" cells wrap it in a FAKE presence signer that
  counts the prompt it would raise -- never the real keychain or Secure Enclave;
* the ASSURANCE READ: ``desktop_approval.assurance`` is pinned per cell, because the rig
  has no root-owned installed anchor and the level is otherwise a host fact.

Every refusal cell asserts the signer ran ZERO times: the digests, window, state, shape,
anchor and policy are all checked before any signature exists.
"""

from __future__ import annotations

import copy
import json
import time
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import approvals as A
from local_operator.network.audit import AuditLog
from local_operator.network.credentials import (
    consent,
    desktop_approval,
    finite_share,
    inheritance,
    placement,
)
from local_operator.network.types import MeshRefusal
from tests.unit.network.test_credential_inherit import request_id as inherit_request_id
from tests.unit.network.test_credential_inherit import (
    software_signer,
    with_owner_transport,
)
from tests.unit.network.test_credentials_consent import N, P
from tests.unit.network.test_credentials_consent import rig as consent_rig
from tests.unit.network.test_credentials_real_link import _lop_network
from tests.unit.network.test_finite_share import (
    _desktop_client,
    _record_with_what,
    _signer,
)

# Reuse the existing software-authority fixture, not a new signer stack.
rig = consent_rig

FILE_ONLY = ("file_only", "operator-file-only")
PRESENCE = ("presence", "operator-presence")
ZERO = "sha256:" + "0" * 64


@pytest.fixture(autouse=True)
def _fresh_limits() -> Any:
    """The abuse bounds are per-daemon process state; every cell starts from none."""
    desktop_approval._ATTEMPTS.clear()
    desktop_approval._CANCELLED.clear()
    yield
    desktop_approval._ATTEMPTS.clear()
    desktop_approval._CANCELLED.clear()


def _level(monkeypatch: pytest.MonkeyPatch, level: tuple[str | None, str]) -> None:
    monkeypatch.setattr(desktop_approval, "assurance", lambda: level)


def _counted(monkeypatch: pytest.MonkeyPatch, *, presence: bool = False) -> dict[str, int]:
    """Count signer calls; with ``presence``, a FAKE presence signer that counts prompts."""
    software = A.sign_decision
    counters = {"signer": 0, "prompts": 0}

    def counted(**kw: Any) -> str:
        counters["signer"] += 1
        if presence:
            counters["prompts"] += 1
        return software(**kw)

    monkeypatch.setattr(A, "sign_decision", counted)
    return counters


def _finite(rig: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    _signer(rig, monkeypatch)
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": P}, "desktop-one", rig.root
    )
    return finite_share.request(N, rig.own.device_id, rig.peer.device_id, P, "session", rig.root)


def _inherit(rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> dict[str, Any]:
    software_signer(rig.root, monkeypatch)
    with_owner_transport(rig)
    return A.load_record(inherit_request_id(rig, capsys), root=rig.root)


async def _row(client: Any, approval_id: str) -> dict[str, Any]:
    response = await client.get("/v1/desktop/approvals")
    assert response.status_code == 200, response.text
    rows = {row["approval_id"]: row for row in response.json()["result"]["approvals"]}
    return rows[approval_id]


def _ack(row: dict[str, Any], **override: str) -> dict[str, str]:
    typed = row["typed_approval"]
    body = {
        "operation": typed["operation"],
        "request_digest": typed["request_digest"],
        "disclosure_digest": typed["disclosure_digest"],
    }
    body.update(override)
    return body


def _events(rig: Any) -> list[dict[str, Any]]:
    log = AuditLog(root=rig.root)
    try:
        return log.tail(limit=10_000)
    finally:
        log.close()


# ---------------------------------------------------------------------------
# Happy paths: file-only finite, presence inheritance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_only_finite_signs_applies_and_audits_the_assurance(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    counters = _counted(monkeypatch)
    aid = record["approval_id"]
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        typed = row["typed_approval"]
        # The old UI's affordance stays off; only a typed renderer can sign.
        assert row["can_approve_here"] is False
        assert typed["signable_here"] is True and typed["assurance"] == "file_only"
        assert typed["unproven_human"] is True and "upgrade" in typed
        assert typed["disclosure"]["grant_lifetime"] == {"kind": "until_revoked"}
        assert typed["disclosure"]["request_window"]["expires_at"] == record["expires_at"]
        response = await client.post(
            f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row)
        )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["assurance"] == "file_only"
    assert result["signed"]["ok"] is True and result["applied"] == {"ok": True, "code": ""}
    assert result["state"] == "connected"
    assert counters["signer"] == 1
    entry = placement.PlacementDocument.load(N, rig.root).entry(P)
    assert entry is not None and entry.is_holder(rig.peer.device_id)
    signed = [e for e in _events(rig) if e["event"] == "credential.desktop_signed"]
    assert len(signed) == 1
    assert signed[0]["detail"]["surface"] == "desktop"
    assert signed[0]["detail"]["assurance"] == "file_only"
    assert signed[0]["detail"]["disclosure_digest"] == typed["disclosure_digest"]


@pytest.mark.asyncio
async def test_presence_key_signs_inheritance_through_one_prompt(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _inherit(rig, monkeypatch, capsys)
    _level(monkeypatch, PRESENCE)
    counters = _counted(monkeypatch, presence=True)
    aid = record["approval_id"]
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        assert row["typed_approval"]["signable_here"] is True
        assert row["typed_approval"]["unproven_human"] is False
        response = await client.post(
            f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row)
        )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["assurance"] == "presence" and result["state"] == "connected"
    assert result["applied"]["ok"] is True
    assert counters == {"signer": 1, "prompts": 1}
    row_policy = consent.ConsentDocument(N, rig.own.device_id, rig.root).load()
    assert row_policy["recipients"][rig.peer.device_id]["enabled"] is True


# ---------------------------------------------------------------------------
# The operator's boundary: standing inheritance needs a presence key (a-e3a5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_only_inheritance_is_refused_before_the_signer(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _inherit(rig, monkeypatch, capsys)
    _level(monkeypatch, FILE_ONLY)
    counters = _counted(monkeypatch)
    aid = record["approval_id"]
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        assert row["typed_approval"]["signable_here"] is False
        assert row["typed_approval"]["refusal"] == "credential_presence_required"
        response = await client.post(
            f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row)
        )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "credential_presence_required"
    assert counters["signer"] == 0
    assert A.load_record(aid, root=rig.root)["state"] == "requested"
    assert desktop_approval.FILE_ONLY_SIGNING == {finite_share.OP: True, inheritance.OP: False}


@pytest.mark.asyncio
@pytest.mark.parametrize("level", [(None, "spawn-capability-only"), (None, "anchor-unpinned")])
async def test_no_pinned_authority_refuses_every_typed_sign(
    rig: Any, monkeypatch: pytest.MonkeyPatch, level: tuple[None, str]
) -> None:
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, level)
    counters = _counted(monkeypatch)
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, record["approval_id"])
        response = await client.post(
            f"/v1/desktop/approvals/{record['approval_id']}/approve-credential", json=_ack(row)
        )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "credential_authority_unavailable"
    assert counters["signer"] == 0


# ---------------------------------------------------------------------------
# Digest binding and pre-signer refusals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field, code",
    [
        ("request_digest", "credential_request_changed"),
        ("disclosure_digest", "credential_disclosure_changed"),
    ],
)
async def test_a_stale_digest_is_refused_with_zero_signer_calls(
    rig: Any, monkeypatch: pytest.MonkeyPatch, field: str, code: str
) -> None:
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    counters = _counted(monkeypatch)
    aid = record["approval_id"]
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        response = await client.post(
            f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row, **{field: ZERO})
        )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == code
    assert counters["signer"] == 0
    assert A.load_record(aid, root=rig.root)["state"] == "requested"


@pytest.mark.asyncio
async def test_a_mutated_disclosure_field_changes_the_digest_the_server_recomputes(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A UI that rendered (and hashed) a different withdrawal list cannot get a signature."""
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    counters = _counted(monkeypatch)
    aid = record["approval_id"]
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        shown = copy.deepcopy(row["typed_approval"]["disclosure"])
        shown["withdrawals"] = shown["withdrawals"][:0]
        response = await client.post(
            f"/v1/desktop/approvals/{aid}/approve-credential",
            json=_ack(row, disclosure_digest=desktop_approval.disclosure_digest(shown)),
        )
    assert response.json()["detail"]["code"] == "credential_disclosure_changed"
    assert counters["signer"] == 0


@pytest.mark.asyncio
async def test_an_expired_card_is_refused_before_the_signer(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    counters = _counted(monkeypatch)
    aid = record["approval_id"]
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        later = time.time() + 7200
        monkeypatch.setattr(
            desktop_approval,
            "time",
            SimpleNamespace(time=lambda: later, monotonic=time.monotonic),
        )
        response = await client.post(
            f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row)
        )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "approval_expired"
    assert counters["signer"] == 0


@pytest.mark.asyncio
async def test_a_card_not_awaiting_approval_is_refused(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    counters = _counted(monkeypatch)
    aid = record["approval_id"]
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        assert (await client.post(f"/v1/desktop/approvals/{aid}/deny")).status_code == 200
        response = await client.post(
            f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row)
        )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "approval_decision_conflict"
    assert counters["signer"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["mixed-unknown", "mixed-install", "wrong-operation"])
async def test_tampered_or_mixed_shapes_never_reach_the_signer(
    rig: Any, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    valid = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    counters = _counted(monkeypatch)
    what = copy.deepcopy(valid["what"])
    if shape == "mixed-unknown":
        what["credential_unknown_typed_operation"] = {"scope": "unknown"}
    elif shape == "mixed-install":
        what["install"] = True
    record = valid if shape == "wrong-operation" else _record_with_what(valid, what, rig.root)
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, valid["approval_id"])
        body = _ack(row, request_digest=record["request_digest"])
        if shape == "wrong-operation":
            body["operation"] = inheritance.OP
        response = await client.post(
            f"/v1/desktop/approvals/{record['approval_id']}/approve-credential", json=body
        )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] in {
        "finite_disclosure_invalid",
        "credential_operation_mismatch",
    }
    assert counters["signer"] == 0


@pytest.mark.asyncio
async def test_an_extra_or_untyped_body_is_rejected_by_the_model(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    counters = _counted(monkeypatch)
    aid = record["approval_id"]
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        extra = await client.post(
            f"/v1/desktop/approvals/{aid}/approve-credential", json={**_ack(row), "auto": True}
        )
        bare = await client.post(f"/v1/desktop/approvals/{aid}/approve-credential")
    assert extra.status_code == 422 and bare.status_code == 422
    assert counters["signer"] == 0


@pytest.mark.asyncio
async def test_an_anchor_mismatch_is_refused_before_the_signer(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    counters = _counted(monkeypatch)
    trio = A.local_anchor_trio(rig.root)
    assert trio is not None
    monkeypatch.setattr(A, "local_anchor_trio", lambda root=None: {**trio, "spki_fp": "other"})
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, record["approval_id"])
        response = await client.post(
            f"/v1/desktop/approvals/{record['approval_id']}/approve-credential", json=_ack(row)
        )
    assert response.json()["detail"]["code"] == "approval_anchor_mismatch"
    assert counters["signer"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["finite", "inheritance"])
async def test_the_generic_approve_route_still_refuses_both_kinds(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    # One kind per cell: the finite rig withdraws the very key an inheritance would name.
    if kind == "finite":
        record, code = _finite(rig, monkeypatch), "finite_terminal_approval_required"
    else:
        record, code = _inherit(rig, monkeypatch, capsys), "inherit_terminal_approval_required"
    _level(monkeypatch, PRESENCE)
    counters = _counted(monkeypatch)
    async with _desktop_client(rig, monkeypatch) as client:
        response = await client.post(f"/v1/desktop/approvals/{record['approval_id']}/approve")
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == code
        assert (await _row(client, record["approval_id"]))["can_approve_here"] is False
    assert counters["signer"] == 0
    assert A.load_record(record["approval_id"], root=rig.root)["state"] == "requested"


# ---------------------------------------------------------------------------
# Abuse bounds: one in flight, a cancelled prompt, the rate limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_credential_sign_in_flight_per_daemon(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    counters = _counted(monkeypatch)
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, record["approval_id"])
        assert desktop_approval._SIGN_LOCK.acquire(blocking=False)
        try:
            response = await client.post(
                f"/v1/desktop/approvals/{record['approval_id']}/approve-credential",
                json=_ack(row),
            )
        finally:
            desktop_approval._SIGN_LOCK.release()
    assert response.json()["detail"]["code"] == "credential_sign_in_flight"
    assert counters["signer"] == 0


@pytest.mark.asyncio
async def test_a_cancelled_prompt_is_not_raised_again_until_the_card_is_refetched(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, PRESENCE)
    software = A.sign_decision
    calls = {"n": 0}

    def cancelled_once(**kw: Any) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise MeshRefusal("approval_signing_unavailable", "the presence prompt was cancelled")
        return software(**kw)

    monkeypatch.setattr(A, "sign_decision", cancelled_once)
    aid = record["approval_id"]
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        first = await client.post(f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row))
        second = await client.post(
            f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row)
        )
        assert first.json()["detail"]["code"] == "approval_signing_unavailable"
        assert second.json()["detail"]["code"] == "credential_prompt_cancelled"
        assert calls["n"] == 1
        row = await _row(client, aid)  # the re-fetch is the "review it again"
        third = await client.post(f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row))
    assert third.status_code == 200, third.text
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_sign_attempts_are_rate_limited(rig: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    counters = _counted(monkeypatch)
    now = time.monotonic()
    desktop_approval._ATTEMPTS.extend([now] * desktop_approval.RATE_LIMIT)
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, record["approval_id"])
        response = await client.post(
            f"/v1/desktop/approvals/{record['approval_id']}/approve-credential", json=_ack(row)
        )
    assert response.json()["detail"]["code"] == "credential_sign_rate_limited"
    assert counters["signer"] == 0


# ---------------------------------------------------------------------------
# Sign then apply: a concurrent change, the retry, the replay, deny
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_policy_change_between_sign_and_apply_leaves_a_retryable_failed_card(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defect B through the desktop: signed, not applied -> ``failed``, never ``connecting``."""
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    software = A.sign_decision

    def sign_then_revoke(**kw: Any) -> str:
        signature = software(**kw)
        # A concurrent withdrawal elsewhere moves the policy revision the card signed.
        consent.restrict_and_revoke(
            N,
            rig.own.device_id,
            rig.peer.device_id,
            {"credential_key": "unrelated"},
            "concurrent",
            rig.root,
        )
        return signature

    monkeypatch.setattr(A, "sign_decision", sign_then_revoke)
    aid = record["approval_id"]
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        response = await client.post(
            f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row)
        )
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert result["signed"]["ok"] is True
        assert result["applied"]["ok"] is False
        assert result["applied"]["code"] == "consent_conflict"
        assert result["state"] == "failed"
        retry = await client.post(f"/v1/desktop/approvals/{aid}/apply-credential", json=_ack(row))
    # The retry re-enters (a new run) and meets the TRUE refusal again, not "run in flight".
    assert retry.status_code == 409
    assert retry.json()["detail"]["code"] == "consent_conflict"
    assert A.load_record(aid, root=rig.root)["state"] == "failed"


@pytest.mark.asyncio
async def test_signed_not_applied_then_apply_retry_succeeds_and_replay_refuses(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    aid = record["approval_id"]
    real_apply = finite_share.apply
    calls = {"n": 0}

    def flaky(*args: Any, **kw: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise MeshRefusal("finite_write_unconfirmed", "synthetic transient failure")
        return real_apply(*args, **kw)

    monkeypatch.setattr(finite_share, "apply", flaky)
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        # Apply before any approval: named as "not approved", never as tampering.
        early = await client.post(f"/v1/desktop/approvals/{aid}/apply-credential", json=_ack(row))
        assert early.status_code == 409
        assert early.json()["detail"]["code"] == "approval_not_approved"
        first = await client.post(f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row))
        assert first.json()["result"]["applied"]["code"] == "finite_write_unconfirmed"
        assert first.json()["result"]["state"] == "approved"
        retry = await client.post(f"/v1/desktop/approvals/{aid}/apply-credential", json=_ack(row))
        assert retry.status_code == 200, retry.text
        assert retry.json()["result"]["state"] == "connected"
        replay = await client.post(f"/v1/desktop/approvals/{aid}/apply-credential", json=_ack(row))
        again = await client.post(f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row))
    assert replay.status_code == 409
    assert replay.json()["detail"]["code"] == "approval_already_connected"
    assert again.json()["detail"]["code"] == "approval_decision_conflict"


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["requested", "approved", "failed"])
async def test_deny_works_from_every_open_state(
    rig: Any, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    aid = record["approval_id"]
    if state != "requested":
        monkeypatch.setattr(
            finite_share,
            "apply",
            lambda *a, **k: (_ for _ in ()).throw(MeshRefusal("finite_write_unconfirmed", "x")),
        )
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        if state != "requested":
            await client.post(f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row))
        if state == "failed":
            stored = A._load_raw(aid, rig.root)
            run = A.begin_run(aid, run_id=A.new_run_id(), root=rig.root)
            A.mark_failed(aid, run_id=run["run"]["run_id"], step="x", root=rig.root)
            assert stored["state"] == "approved"
        assert A.load_record(aid, root=rig.root)["state"] == state
        denied = await client.post(f"/v1/desktop/approvals/{aid}/deny")
    assert denied.status_code == 200, denied.text
    assert denied.json()["result"]["state"] == "denied"


# ---------------------------------------------------------------------------
# One disclosure: the dict the desktop signs is what `approvals show` renders
# ---------------------------------------------------------------------------


def test_finite_disclosure_matches_the_cli_show_field_by_field(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _finite(rig, monkeypatch)
    facts = finite_share.disclosure(record)
    assert facts is not None
    capsys.readouterr()
    assert _lop_network("approvals", "show", record["approval_id"]) == 0
    shown = capsys.readouterr().out.splitlines()
    assert f"Signed policy revision: {facts['policy_revision']}" in shown
    for number, deny in enumerate(facts["withdrawals"], 1):
        assert f"Withdrawal {number}: " + json.dumps(deny, sort_keys=True) in shown
    assert f"Recipient device: {facts['recipient']['device_id']}" in shown
    assert f"Recipient key fingerprint: {facts['recipient']['key_fingerprint']}" in shown
    assert f"Network: {facts['network_id']}" in shown
    assert any(facts["signing_principal"] in line for line in shown)
    assert any(line.startswith("Duration: UNTIL YOU REVOKE IT") for line in shown)
    assert any(line.startswith("Request window:") for line in shown)
    assert repr(facts["credential_key"]) in shown[0] and facts["scope"] in shown[0]
    # The dict carries no CLI command and no English prose (the UI localizes labels).
    assert "lop " not in json.dumps(facts)


def test_inheritance_disclosure_matches_the_cli_show_field_by_field(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _inherit(rig, monkeypatch, capsys)
    facts = inheritance.disclosure(record)
    assert facts is not None
    capsys.readouterr()
    assert _lop_network("approvals", "show", record["approval_id"]) == 0
    shown = capsys.readouterr().out.splitlines()
    assert any(line.startswith(f"Provider: {facts['provider']} ") for line in shown)
    assert f"Account: {facts['account']}" in shown
    # "" is the structured "not constrained"; the CLI words it, the UI localizes it.
    assert "Org: " + (facts["org"] or "not constrained (any org of this account)") in shown
    name = facts["recipient"]["name"]
    device = facts["recipient"]["device_id"]
    assert f"Recipient device: {f'{name} ({device})' if name else device}" in shown
    assert f"Recipient key fingerprint: {facts['recipient']['key_fingerprint']}" in shown
    assert f"Network: {facts['network_id']}" in shown
    assert (
        f"Signing principal: your operator key {facts['signing_principal']} on this machine"
        in shown
    )
    assert "lop " not in json.dumps(facts)


# ---------------------------------------------------------------------------
# Defect B at the CLI/apply seam: fails on origin/main (record stays `connecting`)
# ---------------------------------------------------------------------------


def test_a_finite_apply_refusal_releases_its_run_instead_of_wedging_it(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _finite(rig, monkeypatch)
    aid = record["approval_id"]
    assert _lop_network("approvals", "approve", aid, "--json") == 0
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": "unrelated"}, "moved", rig.root
    )
    expected = {
        "owner_device": rig.own.device_id,
        "network_id": N,
        "recipient_device": rig.peer.device_id,
        "key": P,
        "scope": "session",
    }
    with pytest.raises(MeshRefusal) as first:
        finite_share.apply(aid, expected, rig.root)
    assert first.value.code == "consent_conflict"
    assert A.load_record(aid, root=rig.root)["state"] == "failed"
    # The retry in the SAME live process meets the true refusal, not approval_run_in_flight.
    with pytest.raises(MeshRefusal) as second:
        finite_share.apply(aid, expected, rig.root)
    assert second.value.code == "consent_conflict"

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
from local_operator.network import store
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


def _refusal_codes(rig: Any) -> list[str]:
    """The codes of every ``credential.desktop_sign_refused`` row, oldest first."""
    return [
        e["detail"]["code"] for e in _events(rig) if e["event"] == "credential.desktop_sign_refused"
    ]


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
    assert _refusal_codes(rig) == ["credential_sign_in_flight"]


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
        # A list poll -- what the desktop does every 15 s -- does NOT clear the latch
        # (review #3002 round 1, R2): looping list + approve stays refused, with no prompt.
        for _ in range(3):
            row = await _row(client, aid)
            looped = await client.post(
                f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row)
            )
            assert looped.json()["detail"]["code"] == "credential_prompt_cancelled"
        assert calls["n"] == 1
        # Only the per-card detail read is the "review it again".
        assert (await client.get(f"/v1/desktop/approvals/{aid}")).status_code == 200
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
    # Unwrapped lines: the CLI word-wraps to the terminal (design D3), the facts are what count.
    monkeypatch.setenv("COLUMNS", "1000")
    assert _lop_network("approvals", "show", record["approval_id"]) == 0
    shown = capsys.readouterr().out.splitlines()
    assert f"Signed policy revision: {facts['policy_revision']}" in shown
    for number, deny in enumerate(facts["withdrawals"], 1):
        assert f"Withdrawal {number}: " + json.dumps(deny, sort_keys=True) in shown
    assert f"Recipient device: {facts['recipient']['device_id']}" in shown
    assert f"Recipient key fingerprint: {facts['recipient']['key_fingerprint']}" in shown
    assert f"Network: {facts['network_id']}" in shown
    assert any(facts["signing_principal"] in line for line in shown)
    assert any(line.startswith("Duration: until you revoke it") for line in shown)
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


# ---------------------------------------------------------------------------
# Round 1 remediation (review #3002 R1, R3/Q1; QA Q2, Q3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_record_edited_during_the_sign_window_with_its_digest_field_kept_is_refused(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1: the write re-check RECOMPUTES the digest; the stored field alone is not trusted."""
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    aid = record["approval_id"]
    software = A.sign_decision

    def tamper_then_sign(**kw: Any) -> str:
        # Inside the prompt window: edit the signed intent, keep the stored digest field.
        stored = A._load_raw(aid, rig.root)
        stored["what"][finite_share.OP]["scope"] = "device"
        A._write_record(stored, rig.root)
        return software(**kw)

    monkeypatch.setattr(A, "sign_decision", tamper_then_sign)
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        response = await client.post(
            f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row)
        )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "credential_request_changed"
    stored = A._load_raw(aid, rig.root)
    assert stored["state"] == "requested" and stored.get("signature") is None
    events = [e["event"] for e in _events(rig)]
    assert "credential.desktop_signed" not in events and "onboard_approved" not in events
    assert _refusal_codes(rig) == ["credential_request_changed"]


@pytest.mark.asyncio
async def test_every_pre_signer_refusal_is_audited_with_its_code(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3/Q1: digest, rate-limit and cancelled-prompt refusals each leave an audit row."""
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    aid = record["approval_id"]

    def cancelled(**_: Any) -> str:
        raise MeshRefusal("approval_signing_unavailable", "the presence prompt was cancelled")

    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        await client.post(
            f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row, request_digest=ZERO)
        )
        monkeypatch.setattr(A, "sign_decision", cancelled)
        await client.post(f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row))
        await client.post(f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row))
        desktop_approval._CANCELLED.clear()
        now = time.monotonic()
        desktop_approval._ATTEMPTS.extend([now] * desktop_approval.RATE_LIMIT)
        await client.post(f"/v1/desktop/approvals/{aid}/approve-credential", json=_ack(row))
    assert _refusal_codes(rig) == [
        "credential_request_changed",
        "approval_signing_unavailable",
        "credential_prompt_cancelled",
        "credential_sign_rate_limited",
    ]
    assert A._load_raw(aid, rig.root)["state"] == "requested"


@pytest.mark.asyncio
async def test_an_apply_refusal_is_audited(rig: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    aid = record["approval_id"]
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        await client.post(f"/v1/desktop/approvals/{aid}/apply-credential", json=_ack(row))
    refused = [e for e in _events(rig) if e["event"] == "credential.desktop_apply_refused"]
    assert [e["detail"]["code"] for e in refused] == ["approval_not_approved"]


def test_a_finite_card_names_its_recipient(rig: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Q2: the member's display name is recorded on the card and disclosed."""
    network = store.load(N, rig.root)
    member = network.member(rig.peer.device_id)
    assert member is not None
    member.name = "qa-recipient"
    store.save(network, rig.root)
    record = _finite(rig, monkeypatch)
    facts = finite_share.disclosure(record)
    assert facts is not None and facts["recipient"]["name"] == "qa-recipient"
    assert record["device"]["name"] == "qa-recipient"


def test_a_busy_apply_offers_a_retry_not_a_new_approval(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q3: a transient lock refusal tells the operator to apply the same card again."""
    record = _finite(rig, monkeypatch)
    aid = record["approval_id"]

    def busy(*_: Any, **__: Any) -> Any:
        raise MeshRefusal("busy", "another writer")

    monkeypatch.setattr(finite_share, "_apply", busy)
    with pytest.raises(MeshRefusal) as raised:
        finite_share.apply(aid, {}, rig.root)
    assert raised.value.code == "busy"
    assert "again in a moment" in str(raised.value)
    assert "Regenerate" not in str(raised.value) and "lop " not in str(raised.value)


def test_assurance_reads_the_backend_the_signer_uses(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N1: a presence flag on a non-presence backend is NOT presence (the stricter answer)."""
    import dataclasses

    import local_operator.operator as operator_pkg
    from local_operator.operator import trust

    staged = trust.load_staged_anchor(rig.root)
    assert staged is not None

    def pinned(anchor: Any) -> None:
        # ``authority_level_load`` reads the package's own ``load_anchor`` binding.
        monkeypatch.setattr(
            operator_pkg,
            "load_anchor",
            lambda uid=None: trust.AnchorLoad(
                anchor=anchor,
                path=trust.anchor_path(uid),
                root_owned=True,
                reason="ok",
                exists=True,
            ),
        )

    pinned(dataclasses.replace(staged, backend="file-only", presence=True))
    assert desktop_approval.assurance() == ("file_only", "operator-presence")
    pinned(dataclasses.replace(staged, backend="secure-enclave", presence=True))
    assert desktop_approval.assurance() == ("presence", "operator-presence")
    pinned(dataclasses.replace(staged, backend="file-only", presence=False))
    assert desktop_approval.assurance() == ("file_only", "operator-file-only")


# ---------------------------------------------------------------------------
# Design/UX review round 1 (U1, U2, U3): a failed card says why and what next; retry only
# when it can work; the desktop refusal states the real rule
# ---------------------------------------------------------------------------


def _policy_moved_failed_card(rig: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    """A finite card signed, then a concurrent withdrawal moves the policy -> ``failed``."""
    record = _finite(rig, monkeypatch)
    aid = record["approval_id"]
    at = time.time()
    signature = A.sign_decision(
        kind=record["kind"],
        request_id=record["request_id"],
        request_digest=record["request_digest"],
        decided_at=at,
    )
    A.approve(aid, signature_hex=signature, decided_at=at, root=rig.root)
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": "other-key"}, "moved", rig.root
    )
    with pytest.raises(MeshRefusal):
        finite_share.apply(aid, _finite_expected(rig), rig.root)
    assert A.load_record(aid, root=rig.root)["state"] == "failed"
    return aid


def _finite_expected(rig: Any) -> dict[str, Any]:
    return {
        "owner_device": rig.own.device_id,
        "network_id": N,
        "recipient_device": rig.peer.device_id,
        "key": P,
        "scope": "session",
    }


def test_a_failed_finite_card_says_why_and_what_to_do_in_show_and_list(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """U1: ``show`` and ``list`` name the stored reason and the real next step."""
    aid = _policy_moved_failed_card(rig, monkeypatch)
    capsys.readouterr()
    assert _lop_network("approvals", "show", aid) == 0
    shown = capsys.readouterr().out
    assert "Why it stopped: Who may use this credential changed" in shown, shown
    assert "can't be used again; a new request is needed" in shown
    assert "--request-approval" in shown  # the terminal route rides BESIDE the sentence
    assert _lop_network("approvals", "list") == 0
    listed = capsys.readouterr().out
    assert "Why it stopped:" in listed and "a new request is needed" in listed
    assert f"lop network approvals show {aid}" in listed
    assert "Recipient key fingerprint" not in listed, "list is the compact card (U6)"


def test_re_applying_a_permanently_failed_card_refuses_with_the_same_words_and_no_new_run(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U1: Apply again on a spent card answers at once; no extra ``failed`` receipt."""
    aid = _policy_moved_failed_card(rig, monkeypatch)
    receipts = len(A.load_record(aid, root=rig.root)["receipts"])
    with pytest.raises(MeshRefusal) as again:
        finite_share.apply(aid, _finite_expected(rig), rig.root)
    assert again.value.code == "consent_conflict"
    assert "can't be used again; a new request is needed" in str(again.value)
    assert len(A.load_record(aid, root=rig.root)["receipts"]) == receipts


@pytest.mark.asyncio
async def test_the_desktop_offers_a_retry_only_when_one_can_succeed_and_the_row_keeps_it(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U2: ``retryable`` false + a no-command remedy for a permanent refusal, persisted."""
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    software = A.sign_decision

    def sign_then_revoke(**kw: Any) -> str:
        signature = software(**kw)
        consent.restrict_and_revoke(
            N, rig.own.device_id, rig.peer.device_id, {"credential_key": "x"}, "c", rig.root
        )
        return signature

    monkeypatch.setattr(A, "sign_decision", sign_then_revoke)
    aid = record["approval_id"]
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        result = (
            await client.post(f"/v1/desktop/approvals/{aid}/approve-" "credential", json=_ack(row))
        ).json()["result"]
        reloaded = await _row(client, aid)
    applied = result["applied"]
    assert applied["ok"] is False and applied["code"] == "consent_conflict"
    assert applied["retryable"] is False
    assert "a new request is needed" in applied["remedy"] and "lop " not in applied["remedy"]
    assert "Who may use this credential changed" in applied["reason"]
    # PERSISTED: a later list read (a reload) still carries the outcome.
    outcome = reloaded["typed_approval"]["apply_outcome"]
    assert outcome == {k: applied[k] for k in ("code", "retryable", "reason", "remedy")}


def test_a_transient_refusal_is_the_only_kind_marked_retryable() -> None:
    """U2: the one classification every surface reads; unknown codes are permanent."""
    from local_operator.network.credentials import approval_copy

    assert approval_copy.outcome("busy")["retryable"] is True
    assert "again in a moment" in approval_copy.outcome("busy")["remedy"]
    for code in ("consent_conflict", "finite_denial_mismatch", "approval_expired", "x_new"):
        assert approval_copy.outcome(code)["retryable"] is False, code


@pytest.mark.asyncio
@pytest.mark.parametrize("available", [False, True])
async def test_the_desktop_inheritance_refusal_states_the_desktop_rule_and_a_real_fix(
    rig: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    available: bool,
) -> None:
    """U3: the rule is the DESKTOP's; an unavailable upgrade is never offered as the fix."""
    record = _inherit(rig, monkeypatch, capsys)
    _level(monkeypatch, FILE_ONLY)
    monkeypatch.setattr(
        desktop_approval,
        "upgrade",
        lambda: {"available": available, "helper": "verified" if available else "absent"},
    )
    aid = record["approval_id"]
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        refused = await client.post(
            f"/v1/desktop/approvals/{aid}/approve-" "credential", json=_ack(row)
        )
    message = refused.json()["detail"]["text"]
    assert refused.json()["detail"]["code"] == "credential_presence_required"
    assert "the desktop app approves it only with" in message
    assert "can only be approved with" not in message, "not a host-wide rule (a-b815)"
    if available:
        assert "Set up Touch ID approvals on this Mac" in message
    else:
        assert "can't be set up on this install" in message
        assert "Set up Touch ID approvals" not in message
    # The row says the same thing: its refusal sentence IS the route's.
    assert row["typed_approval"]["refusal_message"] == message
    assert "terminal" in row["approve_remedy"] and "command line" not in row["approve_remedy"]


def test_cli_approve_on_a_file_only_key_warns_once_and_names_the_apply_step(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """U3 (copy, not a gate) + U5: the warning, one disclosure, then how to apply."""
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    capsys.readouterr()
    assert _lop_network("approvals", "approve", record["approval_id"]) == 0
    captured = capsys.readouterr()
    both = captured.err + captured.out
    assert "cannot prove that a person approved" in both
    assert both.count("Recipient key fingerprint") == 1, "the disclosure prints once"
    out_lines = captured.out.splitlines()
    assert "Approved, not applied yet. Run:" in out_lines
    assert any(line.startswith("lop network credential share ") for line in out_lines)
    assert A.load_record(record["approval_id"], root=rig.root)["state"] == "approved"


# ---------------------------------------------------------------------------
# Design r2 / UX r2 / QA r3 on #3002: commands copy as one line; the detail read carries the
# outcome; the 409 says whether a retry can work; the card's words match its state
# ---------------------------------------------------------------------------


class _Stream:
    """A stdout/stderr stand-in that reports whether it is a terminal."""

    def __init__(self, tty: bool) -> None:
        self.tty = tty
        self.parts: list[str] = []

    def write(self, text: str) -> int:
        self.parts.append(text)
        return len(text)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return self.tty

    def text(self) -> str:
        return "".join(self.parts)


def _printed(monkeypatch: pytest.MonkeyPatch, *argv: str, cols: int, tty: bool) -> tuple[int, str]:
    """``lop network …`` at a given terminal width, on a TTY or into a pipe."""
    import sys

    out, err = _Stream(tty), _Stream(tty)
    monkeypatch.setenv("COLUMNS", str(cols))
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    code = _lop_network(*argv)
    return code, out.text() + err.text()


def _parses(command: str) -> None:
    """The printed line is a command the real ``lop`` parser accepts as typed."""
    import shlex

    from tests.unit.network.test_credentials_real_link import _parser

    argv = shlex.split(command)
    assert argv[0] == "lop"
    _parser().parse_args(argv[1:])


_WIDTHS = [(80, True), (120, True), (80, False)]


@pytest.mark.parametrize(("cols", "tty"), _WIDTHS)
def test_every_printed_command_is_one_whole_line_at_any_width_and_in_a_pipe(
    rig: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cols: int,
    tty: bool,
) -> None:
    """D6 = U7 = Q7: no line break or inserted indent inside a command, ever."""

    # The inheritance card first: once the finite share is withdrawn, inherit requests refuse.
    inherit = _inherit(rig, monkeypatch, capsys)
    failed = _policy_moved_failed_card(rig, monkeypatch)
    expected = _finite_expected(rig)
    capsys.readouterr()
    new_request = finite_share._request_command(expected)
    inherit_apply = inheritance._apply_command(inheritance._parse(inherit), inherit["approval_id"])
    inherit_revoke = inheritance._revoke_command(inheritance._parse(inherit))
    checks = [
        (("approvals", "show", failed), [new_request]),
        (("approvals", "show", inherit["approval_id"]), [inherit_apply, inherit_revoke]),
        (
            (
                "credential",
                "share",
                P,
                "--with",
                rig.peer.device_id,
                "--network",
                N,
                "--scope",
                "session",
                "--approval",
                failed,
            ),
            [new_request],
        ),
    ]
    for argv, commands in checks:
        _, text = _printed(monkeypatch, *argv, cols=cols, tty=tty)
        lines = text.splitlines()
        for command in commands:
            assert command in lines, (argv, cols, tty, command, text)
            _parses(command)
        if not tty:
            # A pipe gets the logical lines: nothing is wrapped at all (Q6).
            assert not any(line.startswith("  ") and line.strip() for line in lines[:3]), text


@pytest.mark.parametrize(("cols", "tty"), _WIDTHS)
def test_run_and_apply_again_commands_are_one_whole_line(
    rig: Any, monkeypatch: pytest.MonkeyPatch, cols: int, tty: bool
) -> None:
    """D6: "Run:" after approve and "Apply again:" after a transient refusal, whole."""
    import shlex

    fresh = _finite(rig, monkeypatch)
    _, text = _printed(
        monkeypatch, "approvals", "approve", fresh["approval_id"], cols=cols, tty=tty
    )
    run_line = shlex.join(finite_share._apply_argv(_finite_expected(rig), fresh["approval_id"]))
    assert "Approved, not applied yet. Run:" in text.splitlines()
    assert run_line in text.splitlines(), text
    _parses(run_line)
    monkeypatch.setattr(
        finite_share, "_apply", lambda *_a, **_k: (_ for _ in ()).throw(MeshRefusal("busy", "x"))
    )
    _, text = _printed(
        monkeypatch,
        "credential",
        "share",
        P,
        "--with",
        rig.peer.device_id,
        "--network",
        N,
        "--scope",
        "session",
        "--approval",
        fresh["approval_id"],
        cols=cols,
        tty=tty,
    )
    assert "Apply again:" in text.splitlines() and run_line in text.splitlines(), text


def test_prose_still_wraps_on_words_on_a_terminal(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3 stays fixed: on a TTY, prose lines fit the width; only command lines may not."""
    failed = _policy_moved_failed_card(rig, monkeypatch)
    _, text = _printed(monkeypatch, "approvals", "show", failed, cols=80, tty=True)
    for line in text.splitlines():
        if "lop " not in line and not line.lstrip().startswith(("{", "sha256:")):
            assert len(line) <= 80, line


def test_a_device_name_containing_lop_still_wraps_its_prose(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D10 = U13: only a line that STARTS with a command is left unwrapped, not one naming it."""
    network = store.load(N, rig.root)
    member = network.member(rig.peer.device_id)
    assert member is not None
    member.name = "Studio develop Mac"
    store.save(network, rig.root)
    record = _finite(rig, monkeypatch)
    _, text = _printed(monkeypatch, "approvals", "show", record["approval_id"], cols=80, tty=True)
    lines = text.splitlines()
    assert any("Studio develop Mac" in line for line in lines), text
    for line in lines:
        if not line.lstrip().startswith(("lop ", "{", "sha256:")):
            assert len(line) <= 80, line


@pytest.mark.asyncio
async def test_the_detail_read_carries_the_same_outcome_as_the_list_and_show_after_restart(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q5 = U9: list row, detail read and ``approvals show`` agree, before and after restart."""
    aid = _policy_moved_failed_card(rig, monkeypatch)
    expected_block = None
    for _restart in range(2):  # a fresh app + client is a daemon restart over the same store
        async with _desktop_client(rig, monkeypatch) as client:
            listed = (await _row(client, aid))["typed_approval"]["apply_outcome"]
            detail = await client.get(f"/v1/desktop/approvals/{aid}")
        assert detail.status_code == 200, detail.text
        assert detail.json()["result"]["apply_outcome"] == listed
        assert set(listed) == {"code", "retryable", "reason", "remedy"}
        assert listed["retryable"] is False and "lop " not in listed["remedy"]
        expected_block = expected_block or listed
        assert listed == expected_block
    assert expected_block is not None
    capsys.readouterr()
    monkeypatch.setenv("COLUMNS", "1000")
    assert _lop_network("approvals", "show", aid) == 0
    shown = capsys.readouterr().out.splitlines()
    assert f"Why it stopped: {expected_block['reason']}" in shown
    assert f"What to do: {expected_block['remedy']}" in shown


@pytest.mark.asyncio
async def test_a_permanent_refusal_after_busy_says_so_on_the_409(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U8b: Try again after ``busy`` that meets a permanent refusal carries the outcome."""
    record = _finite(rig, monkeypatch)
    _level(monkeypatch, FILE_ONLY)
    aid = record["approval_id"]
    real_apply = finite_share._apply
    calls = {"n": 0}

    def busy_once(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise MeshRefusal("busy", "held")
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(finite_share, "_apply", busy_once)
    async with _desktop_client(rig, monkeypatch) as client:
        row = await _row(client, aid)
        first = (
            await client.post(f"/v1/desktop/approvals/{aid}/approve-" "credential", json=_ack(row))
        ).json()["result"]
        assert first["applied"]["code"] == "busy" and first["applied"]["retryable"] is True
        consent.restrict_and_revoke(
            N, rig.own.device_id, rig.peer.device_id, {"credential_key": "y"}, "moved", rig.root
        )
        retry = await client.post(
            f"/v1/desktop/approvals/{aid}/apply-" "credential", json=_ack(row)
        )
    assert retry.status_code == 409
    body = retry.json()["detail"]
    assert body["code"] == "consent_conflict"
    outcome = body["apply_outcome"]
    assert outcome["retryable"] is False and "a new request is needed" in outcome["remedy"]
    assert "lop " not in body["text"] and "--" not in body["text"]
    assert outcome["reason"] in body["text"]


def test_a_failed_card_does_not_offer_its_window_and_the_list_says_nothing_is_below(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D7 + D8: the compact card points at nothing below; a failed card's window is closed."""
    aid = _policy_moved_failed_card(rig, monkeypatch)
    capsys.readouterr()
    monkeypatch.setenv("COLUMNS", "1000")
    assert _lop_network("approvals", "show", aid) == 0
    shown = capsys.readouterr().out
    assert "can be approved and applied until" not in shown
    assert "Request window: closed; this card is failed." in shown.splitlines()
    assert _lop_network("approvals", "list") == 0
    assert "listed below" not in capsys.readouterr().out


def test_the_inheritance_card_names_its_apply_command_once_on_approve(
    rig: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """N3: the disclosure before signing no longer carries the apply step; it prints once."""
    record = _inherit(rig, monkeypatch, capsys)
    command = inheritance._apply_command(inheritance._parse(record), record["approval_id"])
    capsys.readouterr()
    monkeypatch.setenv("COLUMNS", "1000")
    assert _lop_network("approvals", "approve", record["approval_id"]) == 0
    captured = capsys.readouterr()
    both = captured.err + captured.out
    assert both.splitlines().count(command) == 1, both
    assert "Then run:" not in both


def test_the_desktop_refusal_does_not_promise_a_time_limited_share() -> None:
    """U10: the finite card has no automatic expiry, so the copy must not call it time-limited."""
    from local_operator.network.credentials import approval_copy

    text = approval_copy.inheritance_desktop_refusal(False)
    assert "time-limited" not in text and "just the current sign-in" in text
    assert "review the card and approve again" in approval_copy.KEY_DID_NOT_SIGN  # U11

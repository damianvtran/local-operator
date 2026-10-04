"""The approval store's lifecycle, pinned: F1 idempotency, F2 matrix, F3
write-once decision, F4 signature contract, expiry fold, tombstones.

These cells drive the REAL store functions against an isolated config root with
a REAL (file-only) operator key — no doubles for the crypto half, because the
point of F4 is that the product path cannot be loosened without a signature and
a test that stubbed the verifier would pin the stub.

Isolation notes: the installed anchor is pinned ABSENT (``trust.load_anchor``),
so a developer's machine state cannot select a different branch; the staged
statement is written exactly as ``lop operator init`` leaves it. Nothing here
touches the operator's login keychain (``file-only`` writes a 0600 file under
the isolated root).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import approvals as A
from local_operator.network.types import MeshRefusal

#: Timestamps RELATIVE to now: a record created "a minute ago" is inside every
#: live window below, so nothing here is born expired on a machine whose clock
#: says today. Cells that need a past window pass explicit values instead.
_NOW = time.time()
CREATED_AT = _NOW - 60.0
EXPIRES_AT = _NOW + 3600.0


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated config root, with "this host has no installed anchor" pinned."""
    from local_operator.operator import trust

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))

    def absent(uid: int | str | None = None) -> Any:
        return trust.AnchorLoad(
            anchor=None,
            path=trust.anchor_path(uid),
            root_owned=False,
            reason="pinned absent by the test",
            exists=False,
        )

    monkeypatch.setattr(trust, "load_anchor", absent)
    return config_dir


def _device_request(request_id: str, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": A.KIND_DEVICE_ONBOARD,
        "request_id": request_id,
        "requested_by": {"session_id": "s1", "device_id": "d_self", "surface": "cli"},
        "device": {
            "device_id": "d_node",
            "name": "cloud-node-1",
            "fingerprint": "ABCD-EF12-3456",
            "host": "99.79.190.164",
            "user": "ec2-user",
            "transport": "ssh",
            "host_key_fp": "SHA256:abc",
        },
        "what": {"install": True, "connect": True, "unattended": True, "grant": ["approve"]},
        "credential_ref": {"kind": "ssh", "ref": "secret:node-key"},
        "created_at": CREATED_AT,
        "expires_at": EXPIRES_AT,
    }
    base.update(overrides)
    return base


def _make_key(config_root: Path) -> Any:
    """A real file-only operator key, staged exactly as ``lop operator init``.

    Idempotent per root: a second call in one test returns the staged anchor
    rather than re-creating the key (the file backend refuses to overwrite).
    """
    from local_operator.operator.keychain import FILE_ONLY
    from local_operator.operator.sign import anchor_for_handle, create_key
    from local_operator.operator.trust import (
        anchor_bytes,
        load_staged_anchor,
        staging_path,
    )

    staged = staging_path(config_root)
    existing = load_staged_anchor(config_root)
    if existing is not None and staged.exists():
        return existing
    handle = create_key(config_root=config_root, preference=FILE_ONLY)
    anchor = anchor_for_handle(handle, label="approvals-test")
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(anchor_bytes(anchor))
    return anchor


def _sign(config_root: Path, message: bytes) -> str:
    from local_operator.operator.sign import load_signer, sign_message

    signer = load_signer(config_root=config_root, backend_name="file-only")
    assert signer is not None, "the rig's operator key did not load"
    try:
        return sign_message(signer, message).sig
    finally:
        signer.close()


def _sign_decision(config_root: Path, record: dict[str, Any], decision: str, decided_at: float):
    message = A.signed_payload(
        kind=record["kind"],
        request_id=record["request_id"],
        request_digest=record["request_digest"],
        decision=decision,
        decided_at=decided_at,
    )
    return _sign(config_root, message)


# ---------------------------------------------------------------------------
# F1 — idempotency
# ---------------------------------------------------------------------------


def test_same_id_and_digest_returns_the_record_and_a_changed_payload_conflicts(
    root: Path,
) -> None:
    request_id = A.new_request_id()
    first = A.create_request(**_device_request(request_id), root=root)
    again = A.create_request(**_device_request(request_id), root=root)
    assert again["approval_id"] == first["approval_id"]
    assert again["request_digest"] == first["request_digest"]

    before = (root / "network" / "approvals" / f"{first['approval_id']}.json").read_text()
    with pytest.raises(MeshRefusal) as raised:
        A.create_request(
            **_device_request(
                request_id, device={**_device_request(request_id)["device"], "host": "5.5.5.5"}
            ),
            root=root,
        )
    assert raised.value.code == "approval_request_conflict"
    after = (root / "network" / "approvals" / f"{first['approval_id']}.json").read_text()
    assert before == after, "a conflicting re-request changed the file"


def test_a_re_request_after_a_terminal_state_returns_it_unchanged(root: Path) -> None:
    request_id = A.new_request_id()
    record = A.create_request(**_device_request(request_id), root=root)
    A.deny(record["approval_id"], decided_at=CREATED_AT + 5.0, root=root)
    again = A.create_request(**_device_request(request_id), root=root)
    assert again["state"] == "denied", "a deny must not be reset by a re-request"


def test_the_digest_binds_every_immutable_field(root: Path) -> None:
    """One field at a time: any changed immutable field is a conflict, never a merge.

    The WINDOW fields are the deliberate exception (QA round 1, Q1): a retry
    re-derives them per call, so they are not part of the idempotency intent —
    the retry cell below pins the moored behaviour they have instead.
    """
    request_id = A.new_request_id()
    A.create_request(**_device_request(request_id), root=root)
    for overrides in (
        {"what": {"install": False}},
        {"credential_ref": None},
        {"requested_by": {"session_id": "s2", "device_id": "d_self", "surface": "cli"}},
    ):
        with pytest.raises(MeshRefusal) as raised:
            A.create_request(**_device_request(request_id, **overrides), root=root)
        assert raised.value.code == "approval_request_conflict", overrides


# ---------------------------------------------------------------------------
# F2 — the frozen matrix
# ---------------------------------------------------------------------------


def test_deny_is_write_once_and_approve_after_deny_conflicts(root: Path) -> None:
    record = A.create_request(**_device_request(A.new_request_id()), root=root)
    A.deny(record["approval_id"], decided_at=CREATED_AT + 1.0, root=root)
    with pytest.raises(MeshRefusal) as second:
        A.deny(record["approval_id"], root=root)
    assert second.value.code == "approval_decision_conflict"
    with pytest.raises(MeshRefusal) as approve_after:
        A.approve(record["approval_id"], signature_hex="00", decided_at=1.0, root=root)
    assert approve_after.value.code == "approval_decision_conflict"


def test_deny_lands_before_the_first_step_and_mid_run_the_runner_stops(root: Path) -> None:
    """The frozen deny matrix: approved→denied, connecting→denied, failed→denied.

    A deny is accepted any time before the runner's NEXT step check: from
    ``approved`` (ordinary), from ``connecting`` (the write lands and the runner
    observes it and stops — receipts show where), and from ``failed``
    (abandoning a failed-but-retryable request). It is refused only AFTER
    ``connected``.
    """
    _make_key(root)
    record = _approved_device_record(root)
    # `approved` with no receipts: a deny is still the safe direction.
    A.deny(record["approval_id"], decided_at=CREATED_AT + 10.0, root=root)
    assert A.load_record(record["approval_id"], root=root)["state"] == "denied"

    # connecting → denied: the write lands, a receipt records the stop, and the
    # runner's per-step check refuses to execute past it.
    other = _approved_device_record(root)
    A.begin_run(other["approval_id"], run_id="run_1", root=root)
    A.append_receipt(other["approval_id"], run_id="run_1", step="connect", ok=True, root=root)
    denied = A.deny(other["approval_id"], decided_at=CREATED_AT + 20.0, root=root)
    assert denied["state"] == "denied"
    assert [r["step"] for r in denied["receipts"]] == ["connect", "deny"], denied["receipts"]
    with pytest.raises(MeshRefusal) as stopped:
        A.verify_for_run(other["approval_id"], root=root)
    assert stopped.value.code == "approval_denied"

    # failed → denied: an operator abandoning a failed-but-retryable request.
    failed_rec = _approved_device_record(root)
    A.begin_run(failed_rec["approval_id"], run_id="run_1", root=root)
    A.mark_failed(
        failed_rec["approval_id"], run_id="run_1", step="install", detail="boom", root=root
    )
    abandoned = A.deny(failed_rec["approval_id"], decided_at=CREATED_AT + 30.0, root=root)
    assert abandoned["state"] == "denied"
    # Write-once, never reset: a second deny (or an approve) still conflicts.
    with pytest.raises(MeshRefusal) as again:
        A.deny(failed_rec["approval_id"], root=root)
    assert again.value.code == "approval_decision_conflict"

    # AFTER connected: the one refusal that stays.
    done = _approved_device_record(root)
    A.begin_run(done["approval_id"], run_id="run_1", root=root)
    A.mark_connected(done["approval_id"], run_id="run_1", step="verify", root=root)
    with pytest.raises(MeshRefusal) as late:
        A.deny(done["approval_id"], root=root)
    assert late.value.code == "approval_already_connected"


def test_run_re_entry_from_failed_uses_a_new_run_id(root: Path) -> None:
    _make_key(root)
    record = _approved_device_record(root)
    A.begin_run(record["approval_id"], run_id="run_1", root=root)
    A.append_receipt(record["approval_id"], run_id="run_1", step="connect", ok=True, root=root)
    A.mark_failed(record["approval_id"], run_id="run_1", step="install", detail="boom", root=root)
    failed = A.load_record(record["approval_id"], root=root)
    assert failed["state"] == "failed"

    A.begin_run(record["approval_id"], run_id="run_2", root=root)
    A.append_receipt(record["approval_id"], run_id="run_2", step="connect", ok=True, root=root)
    A.mark_connected(record["approval_id"], run_id="run_2", step="verify", root=root)
    done = A.load_record(record["approval_id"], root=root)
    assert done["state"] == "connected"
    runs = {r["run_id"] for r in done["receipts"]}
    assert runs == {"run_1", "run_2"}
    with pytest.raises(MeshRefusal) as raised:
        A.begin_run(record["approval_id"], run_id="run_3", root=root)
    assert raised.value.code == "approval_not_runnable", "a connected record must never re-run"


# ---------------------------------------------------------------------------
# The stale-lease re-entry — drill finding, 2026-10-04 (F1)
# ---------------------------------------------------------------------------
#
# A runner killed mid-flight left a card in ``connecting`` that refused every
# retry: the frozen matrix lets only the RUNNER write ``failed``, and the runner
# was the thing that died. These cells pin the resolution chosen by the drill
# decision — ``begin_run`` supersedes a run whose lease is stale, refuses one
# still in flight, and never invents a step failure (the truthful receipt is
# ``step=superseded``, a value that cannot be mistaken for one of the eight
# step names — round-1 D5). A live runner can never be double-entered: the pid
# in the lease is the guard, and ``procstate.pid_alive`` fails closed toward
# "alive".


def test_a_connecting_record_with_a_dead_lease_is_superseded_by_a_new_run(root: Path) -> None:
    record = _approved_device_record(root)
    A.begin_run(record["approval_id"], run_id="run_dead", root=root)
    A.append_receipt(record["approval_id"], run_id="run_dead", step="invite", ok=True, root=root)
    # A REAL dead process's pid: spawned and reaped here, so no clock is asked
    # to stand in for exit.
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    stopped = A._load_raw(record["approval_id"], root)
    stopped["run"] = {"run_id": "run_dead", "pid": child.pid, "started_at": time.time() - 100.0}
    A._write_record(stopped, root)

    reopened = A.begin_run(
        record["approval_id"], run_id="run_new", runner_pid=os.getpid(), root=root
    )
    assert reopened["state"] == A.STATE_CONNECTING
    assert reopened["run"]["run_id"] == "run_new"
    assert reopened["run"]["pid"] == os.getpid()
    steps = [r["step"] for r in reopened["receipts"]]
    assert steps == ["invite", "superseded"], reopened["receipts"]
    superseded = reopened["receipts"][-1]
    assert superseded["run_id"] == "run_dead"
    assert superseded["ok"] is False
    assert "stopped reporting" in superseded["detail"]
    # Round-1 D5: the receipt cannot be read at a glance as a step result.
    assert "no step result is recorded here" in superseded["detail"]
    # The audit trail shows the gap between the approval that ran and the run
    # that replaced it.
    assert reopened["audit"][-1] == "onboard_superseded"


def test_a_live_runner_is_never_double_entered(root: Path) -> None:
    """The guard from the other side: a lease whose pid IS alive refuses, and
    the refusal leaves the record byte-identical — the in-flight run keeps its
    receipts and its lease, whatever the caller intended."""
    record = _approved_device_record(root)
    A.begin_run(record["approval_id"], run_id="run_live", runner_pid=os.getpid(), root=root)
    before = A._load_raw(record["approval_id"], root)
    with pytest.raises(MeshRefusal) as raised:
        A.begin_run(record["approval_id"], run_id="run_2", root=root)
    assert raised.value.code == "approval_run_in_flight"
    after = A._load_raw(record["approval_id"], root)
    assert after == before, "a refused re-entry must not touch the record"


def test_a_lease_less_record_falls_back_to_the_silence_bound(root: Path) -> None:
    """Records written before the lease existed: silence past the bound (no
    step can legally be in flight that long) may be superseded; a recent
    receipt refuses, because it cannot be told apart from a live step."""
    record = _approved_device_record(root)
    A.begin_run(record["approval_id"], run_id="run_old", root=root)
    A.append_receipt(record["approval_id"], run_id="run_old", step="install", ok=True, root=root)
    raw = A._load_raw(record["approval_id"], root)
    del raw["run"]
    raw["receipts"][-1]["at"] = time.time() - (A.STALE_RUN_AFTER_S + 60.0)
    A._write_record(raw, root)
    reopened = A.begin_run(record["approval_id"], run_id="run_next", root=root)
    assert reopened["state"] == A.STATE_CONNECTING
    assert [r["step"] for r in reopened["receipts"]] == ["install", "superseded"]

    other = _approved_device_record(root)
    A.begin_run(other["approval_id"], run_id="run_old2", root=root)
    A.append_receipt(other["approval_id"], run_id="run_old2", step="install", ok=True, root=root)
    raw2 = A._load_raw(other["approval_id"], root)
    del raw2["run"]
    raw2["receipts"][-1]["at"] = time.time() - 1.0
    A._write_record(raw2, root)
    with pytest.raises(MeshRefusal) as raised:
        A.begin_run(other["approval_id"], run_id="run_x", root=root)
    assert raised.value.code == "approval_run_in_flight"


def test_a_lease_less_record_with_no_receipts_fails_closed(root: Path) -> None:
    """Nothing dates the run at all — no lease, no receipt — so nothing can
    prove it dead; guessing would risk double-running a live first step."""
    record = _approved_device_record(root)
    A.begin_run(record["approval_id"], run_id="run_bare", root=root)
    raw = A._load_raw(record["approval_id"], root)
    del raw["run"]
    raw["receipts"] = []
    A._write_record(raw, root)
    with pytest.raises(MeshRefusal) as raised:
        A.begin_run(record["approval_id"], run_id="run_x", root=root)
    assert raised.value.code == "approval_run_in_flight"


def test_the_staleness_bound_covers_the_runners_largest_step_timeout(root: Path) -> None:
    """The bound's derivation, pinned: a live runner may be silent for exactly
    its largest step timeout between two receipts, so the staleness bound must
    exceed it. If either constant moves, this fails before silence starts
    meaning something it does not."""
    from local_operator.network import onboard

    assert A.STALE_RUN_AFTER_S >= max(onboard.STEP_TIMEOUTS.values()) + 60.0


def test_receipts_never_touch_state_or_signature_and_stop_at_terminal(root: Path) -> None:
    _make_key(root)
    record = _approved_device_record(root)
    before = A.load_record(record["approval_id"], root=root)
    A.begin_run(record["approval_id"], run_id="run_1", root=root)
    A.append_receipt(record["approval_id"], run_id="run_1", step="connect", ok=True, root=root)
    after = A.load_record(record["approval_id"], root=root)
    assert after["state"] == "connecting"
    assert after["signature"] == before["signature"]
    A.mark_connected(record["approval_id"], run_id="run_1", step="verify", root=root)
    with pytest.raises(MeshRefusal) as raised:
        A.append_receipt(record["approval_id"], run_id="run_1", step="later", ok=True, root=root)
    assert raised.value.code == "approval_receipt_refused"


# ---------------------------------------------------------------------------
# Expiry: a fold for readers, a materialized transition for writers
# ---------------------------------------------------------------------------


def test_expiry_folds_on_read_and_materializes_on_the_first_write(root: Path) -> None:
    request_id = A.new_request_id()
    record = A.create_request(
        **_device_request(request_id, created_at=1000.0, expires_at=1001.0), root=root
    )
    # The create-time retention sweep materializes it immediately (the window is
    # long past); the view has folded either way.
    assert record["state"] == "expired" or A.presented(record)["state"] == "expired"
    raw = json.loads((root / "network" / "approvals" / f"{record['approval_id']}.json").read_text())
    assert raw["state"] == "expired"
    with pytest.raises(MeshRefusal) as raised:
        A.approve(record["approval_id"], signature_hex="00", decided_at=1.0, root=root)
    assert raised.value.code == "approval_expired"


def test_a_read_folds_without_writing(root: Path) -> None:
    record = A.create_request(
        **_device_request(A.new_request_id(), created_at=CREATED_AT, expires_at=EXPIRES_AT),
        root=root,
    )
    path = root / "network" / "approvals" / f"{record['approval_id']}.json"
    raw = json.loads(path.read_text())
    raw["expires_at"] = time.time() - 10.0
    path.write_text(json.dumps(raw))
    loaded = A.load_record(record["approval_id"], root=root)
    assert loaded["state"] == "expired"
    assert json.loads(path.read_text())["state"] == "requested", "a read materialized a write"


# ---------------------------------------------------------------------------
# F4 — the signature contract, on the real signer
# ---------------------------------------------------------------------------


def test_the_signed_payload_framing_round_trips_through_operator_verify(root: Path) -> None:
    from local_operator.operator.verify import APPROVAL_DOMAIN, verify_signature

    assert APPROVAL_DOMAIN == A.APPROVAL_DOMAIN_LITERAL, "the domain tag drifted from verify.py"
    record = A.create_request(**_device_request(A.new_request_id()), root=root)
    message = A.signed_payload(
        kind=record["kind"],
        request_id=record["request_id"],
        request_digest=record["request_digest"],
        decision="approve",
        decided_at=1790734000.123456,
    )
    _make_key(root)
    signature = bytes.fromhex(_sign(root, message))
    trio = A.local_anchor_trio(root)
    assert trio is not None
    assert verify_signature(spki=trio["spki"], message=message, signature=signature)
    assert not verify_signature(spki=trio["spki"], message=message + b"x", signature=signature)


def test_approve_verifies_the_signature_before_the_file_is_touched(root: Path) -> None:
    anchor = _make_key(root)
    trio = A.local_anchor_trio(root)
    assert trio is not None
    request_id = A.new_request_id()
    record = A.create_request(
        **_device_request(
            request_id,
            what={
                "install": True,
                "connect": True,
                "anchor": {k: trio[k] for k in ("key_id", "spki_fp", "statement_digest")},
            },
        ),
        root=root,
    )
    path = root / "network" / "approvals" / f"{record['approval_id']}.json"

    decided_at = CREATED_AT + 30.0
    with pytest.raises(MeshRefusal) as bad:
        A.approve(record["approval_id"], signature_hex="00" * 8, decided_at=decided_at, root=root)
    assert bad.value.code == "approval_signature_invalid"
    assert json.loads(path.read_text())["state"] == "requested", "a refused decision wrote anyway"

    good = _sign_decision(root, record, "approve", decided_at)
    approved = A.approve(
        record["approval_id"], signature_hex=good, decided_at=decided_at, root=root
    )
    assert approved["state"] == "approved"
    assert approved["signature"]["key_id"] == anchor.key_id
    assert approved["decided_at"] == decided_at


def test_approve_refuses_a_record_whose_anchor_is_not_the_local_key(root: Path) -> None:
    _make_key(root)
    record = A.create_request(
        **_device_request(
            A.new_request_id(),
            what={
                "install": True,
                "anchor": {
                    "key_id": "lop-op-other",
                    "spki_fp": "AAAA-BBBB-CCCC",
                    "statement_digest": "sha256:" + "0" * 64,
                },
            },
        ),
        root=root,
    )
    decided_at = CREATED_AT + 30.0
    signature = _sign_decision(root, record, "approve", decided_at)
    with pytest.raises(MeshRefusal) as raised:
        A.approve(record["approval_id"], signature_hex=signature, decided_at=decided_at, root=root)
    assert raised.value.code == "approval_anchor_mismatch"
    # F5 review, Q1: the clause names the fields in product words and says the
    # remedy (refile) — the old sentence reused the rotation wording verbatim
    # and told the reader nothing to do.
    sentence = str(raised.value)
    assert "(key id, key fingerprint, statement digest)" in sentence
    assert "refile" in sentence
    assert "nothing was written" in sentence


def test_approve_names_only_the_field_a_stale_card_disagrees_on(root: Path) -> None:
    """F5 review, Q1: the stale-format shape (the retired 12-hex fingerprint)
    disagrees on the fingerprint ALONE, and the refusal says which field — not
    just "does not match" — while the untouched fields stay out of the clause."""
    _make_key(root)
    trio = A.local_anchor_trio(root)
    assert trio is not None
    record = A.create_request(
        **_device_request(
            A.new_request_id(),
            what={
                "install": True,
                "anchor": {
                    "key_id": trio["key_id"],
                    "spki_fp": "6DC6-1AAB-622A",
                    "statement_digest": trio["statement_digest"],
                },
            },
        ),
        root=root,
    )
    decided_at = CREATED_AT + 30.0
    signature = _sign_decision(root, record, "approve", decided_at)
    with pytest.raises(MeshRefusal) as raised:
        A.approve(record["approval_id"], signature_hex=signature, decided_at=decided_at, root=root)
    sentence = str(raised.value)
    assert "(key fingerprint)" in sentence
    assert "key id" not in sentence and "statement digest" not in sentence
    assert "refile" in sentence


def test_approve_refuses_a_record_with_no_anchor_provenance(root: Path) -> None:
    """R1-5 / QA Q2: the ABSENCE branch must fail CLOSED, not skip the check.

    No shipped mint omits the trio (the CLI derives it), but the comparison used
    to run only ``if`` it was present — so a record stripped of provenance
    approved with NO comparison at all. It refuses now, with the provenance-
    shaped code, and the file is untouched.
    """
    _make_key(root)
    record = A.create_request(**_device_request(A.new_request_id()), root=root)
    assert "anchor" not in record["what"]
    decided_at = CREATED_AT + 30.0
    signature = _sign_decision(root, record, "approve", decided_at)
    with pytest.raises(MeshRefusal) as raised:
        A.approve(record["approval_id"], signature_hex=signature, decided_at=decided_at, root=root)
    assert raised.value.code == "approval_anchor_missing"
    assert "nothing was written" in str(raised.value)
    again = json.loads(A.record_path(record["approval_id"], root).read_text())
    assert again["state"] == A.STATE_REQUESTED, "a refused decision wrote anyway"


def test_a_windowness_retry_with_the_same_request_id_is_idempotent(root: Path) -> None:
    """QA round 1, Q1, at the store: the window is NOT part of the intent.

    Retries re-derive their window per call (the CLI passes its own computed
    pair), so the comparison adopts the STORED window: a retry returns the first
    record, and the window stays moored to the first acceptance of the id — a
    retry cannot extend a card. A changed INTENT still conflicts.
    """
    first = A.create_request(
        **_device_request(A.new_request_id(), created_at=None, expires_at=None), root=root
    )
    second = A.create_request(
        **_device_request(first["request_id"], created_at=None, expires_at=None), root=root
    )
    assert second["approval_id"] == first["approval_id"], (first, second)

    # Same id, a DIFFERENT window: moored, so this is the same request — not a
    # conflict, and not a new record.
    third = A.create_request(
        **_device_request(first["request_id"], created_at=1001.0, expires_at=2001.0), root=root
    )
    assert third["approval_id"] == first["approval_id"]
    assert third["expires_at"] == first["expires_at"], "the window must stay moored"

    # ...but a changed intent still refuses.
    with pytest.raises(MeshRefusal) as raised:
        A.create_request(
            **_device_request(
                first["request_id"],
                what={"install": False, "connect": True, "unattended": True, "grant": ["approve"]},
            ),
            root=root,
        )
    assert raised.value.code == "approval_request_conflict"


def test_verify_for_run_refuses_a_tampered_record(root: Path) -> None:
    _make_key(root)
    record = _approved_device_record(root)
    assert A.verify_for_run(record["approval_id"], root=root)["state"] == "approved"
    path = root / "network" / "approvals" / f"{record['approval_id']}.json"
    raw = json.loads(path.read_text())
    raw["what"] = {**raw["what"], "install": False}  # an immutable field, edited
    path.write_text(json.dumps(raw))
    with pytest.raises(MeshRefusal) as raised:
        A.verify_for_run(record["approval_id"], root=root)
    assert raised.value.code == "approval_record_tampered"


# ---------------------------------------------------------------------------
# Retention: tombstones guard a spent id
# ---------------------------------------------------------------------------


def test_a_tombstoned_id_is_refused_for_180_days(root: Path) -> None:
    request_id = A.new_request_id()
    record = A.create_request(**_device_request(request_id), root=root)
    A.deny(record["approval_id"], decided_at=CREATED_AT + 1.0, root=root)
    later = time.time() + A.TERMINAL_PRUNE_AGE_S + 10.0
    counts = A.sweep(root=root, now=later)
    assert counts["pruned"] == 1
    assert not (root / "network" / "approvals" / f"{record['approval_id']}.json").exists()

    with pytest.raises(MeshRefusal) as raised:
        A.create_request(**_device_request(request_id), root=root)
    assert raised.value.code == "approval_request_conflict"

    # Past 180 days the tombstone itself ages out and the id is reuseable.
    A.sweep(root=root, now=later + A.TOMBSTONE_PRUNE_AGE_S + 1.0)
    again = A.create_request(**_device_request(request_id), root=root)
    assert again["state"] in {"requested", "expired"}


# ---------------------------------------------------------------------------
# F5 — one derivation, one owner for the anchor trio
# ---------------------------------------------------------------------------


def test_the_card_trio_and_the_runner_share_one_derivation(root: Path) -> None:
    """F5 cross-module contract: the value the CARD records (this module's
    ``local_anchor_trio``, what the mint verb calls) and the value the RUNNER
    re-derives before planting (``trust.anchor_trio``, what ``step_anchor``
    compares) are byte-for-byte the same — and both are ``verify.spki_fp``.
    Before F5 the card recorded a 12-hex 3-group truncation while the runner
    derived a 16-hex 4-group one, so every card this build filed refused with
    ``{"mismatch": ["spki_fp"]}`` (drill run_eett85dv). Either half drifting
    back fails this cell.
    """
    from local_operator.operator import trust
    from local_operator.operator.verify import key_id_for, spki_fp

    anchor = _make_key(root)
    card = A.local_anchor_trio(root)
    assert card is not None
    assert card["key_id"] == key_id_for(anchor.spki)
    assert card["spki_fp"] == spki_fp(anchor.spki), "the runner compares THIS value"
    assert {k: card[k] for k in ("key_id", "spki_fp", "statement_digest")} == trust.anchor_trio(
        anchor
    )
    # The runner's compare is a 4×4-group string; the retired 12-hex shape is gone.
    assert len(card["spki_fp"].replace("-", "")) == 16


def test_the_local_read_prefers_the_installed_statement_over_the_staged(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F5: ONE store read, ONE precedence — the installed-and-usable statement
    first, the staged statement as the fallback (the bootstrap window between
    ``init``/``setup`` and the privileged install).

    The two sides used to read in OPPOSITE orders (the trio installed-first,
    ``step_anchor`` staged-first), dormant while the two files are identical and
    a second refusal class the moment they diverge (a re-init before its
    install). The installed anchor is what the runtime honours, so it is what
    the card records and what the runner ships; a staged candidate becomes
    readable — and mintable — the moment its own install lands it.
    """
    from local_operator.operator import trust
    from local_operator.operator.verify import key_id_for

    staged = _make_key(root)
    spki = b"\x04" + bytes(range(64, 128))
    other = trust.OperatorAnchor(
        key_id=key_id_for(spki), spki=spki, backend="file-only", presence=False
    )

    def installed(*_args: Any, **_: Any) -> Any:
        return trust.AnchorLoad(
            anchor=other,
            path=Path("/etc/other.json"),
            root_owned=True,
            reason="ok",
            exists=True,
        )

    monkeypatch.setattr(trust, "load_anchor", installed)
    picked = trust.load_local_anchor(root)
    assert picked is not None and picked.spki == other.spki, "installed must win"
    card = A.local_anchor_trio(root)
    assert card is not None
    assert card["spki_fp"] == trust.anchor_trio(other)["spki_fp"]

    # A present-but-NOT-usable installed statement is not authoritative: the
    # staged statement is the source in the bootstrap window, and the runner's
    # reader agrees with the mint's there too.
    def unusable(*_args: Any, **_: Any) -> Any:
        return trust.AnchorLoad(
            anchor=other,
            path=Path("/etc/other.json"),
            root_owned=False,
            reason="the anchor is owned by uid 501, not root",
            exists=True,
        )

    monkeypatch.setattr(trust, "load_anchor", unusable)
    picked = trust.load_local_anchor(root)
    assert picked is not None and picked.spki == staged.spki
    card = A.local_anchor_trio(root)
    assert card is not None
    assert card["statement_digest"] == trust.statement_digest(staged)


# ---------------------------------------------------------------------------
# The sibling kind
# ---------------------------------------------------------------------------


def test_local_authority_rides_the_same_store(root: Path) -> None:
    record = A.create_request(
        kind=A.KIND_LOCAL_AUTHORITY,
        request_id=A.new_request_id(),
        requested_by={"session_id": "s1", "device_id": "d_self", "surface": "desktop"},
        machine={
            "hostname": "mac",
            "platform": "darwin",
            "uid": "501",
            "backend": "file-only",
            "level": "operator-file-only",
        },
        what={"install": True},
        credential_ref=None,
        created_at=CREATED_AT,
        expires_at=EXPIRES_AT,
        root=root,
    )
    assert "machine" in record and "device" not in record
    assert record["credential_ref"] is None
    assert "proposed" in A.LOCAL_AUTHORITY_STEPS
    digest_again = A.request_digest(record)
    assert digest_again == record["request_digest"]


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def test_an_unknown_id_refuses_and_listing_orders_by_creation(root: Path) -> None:
    with pytest.raises(MeshRefusal) as raised:
        A.load_record("ap_nothinghere", root=root)
    assert raised.value.code == "unknown_approval"

    older = A.create_request(
        **_device_request(
            A.new_request_id(), created_at=CREATED_AT - 100, expires_at=EXPIRES_AT + 100
        ),
        root=root,
    )
    newer = A.create_request(**_device_request(A.new_request_id()), root=root)
    rows = A.list_records(root=root)
    assert [row["approval_id"] for row in rows] == [older["approval_id"], newer["approval_id"]]


# ---------------------------------------------------------------------------


def _approved_device_record(root: Path) -> dict[str, Any]:
    """A device_onboard record approved with a REAL signature over its payload."""
    _make_key(root)
    trio = A.local_anchor_trio(root)
    assert trio is not None
    record = A.create_request(
        **_device_request(
            A.new_request_id(),
            what={
                "install": True,
                "connect": True,
                "anchor": {k: trio[k] for k in ("key_id", "spki_fp", "statement_digest")},
            },
        ),
        root=root,
    )
    decided_at = CREATED_AT + 5.0
    signature = _sign_decision(root, record, "approve", decided_at)
    return A.approve(
        record["approval_id"], signature_hex=signature, decided_at=decided_at, root=root
    )


# ---------------------------------------------------------------------------
# Withdrawal — the requester's own settle (self-settled, no operator involved)
# ---------------------------------------------------------------------------


def test_a_requester_withdraws_its_own_unanswered_request(root: Path) -> None:
    """The drill's honest end-state for the redundant card: the REQUESTER settles
    its OWN un-actioned request. One write, no signature (no operator gesture
    rode it), the record's own trail naming the requester as the withdrawer,
    and the result is terminal — it can never run and never re-fold as
    expired."""
    record = A.create_request(**_device_request(A.new_request_id()), root=root)
    withdrawn = A.withdraw(
        record["approval_id"], requested_by=dict(record["requested_by"]), root=root
    )

    assert withdrawn["state"] == "withdrawn"
    assert A.is_terminal(withdrawn["state"])
    # The trail reads "filed by X, withdrawn by X" without re-deriving the rule —
    # and the SHARED row carries it too (design review round 1, D2).
    assert withdrawn["withdrawn_by"] == record["requested_by"]
    assert A.badge_row(withdrawn)["withdrawn_by"] == record["requested_by"]
    assert withdrawn["audit"] == ["onboard_requested", "onboard_withdrawn"]
    raw = A._load_raw(record["approval_id"], root)
    assert raw["signature"] is None, "no operator gesture rides a withdrawal"
    assert raw["receipts"] == [], "nothing ran, so nothing is recorded as run"
    assert float(raw["decided_at"]) > 0.0
    # Terminal, both directions: never re-runnable, and the expiry fold leaves it
    # settled as withdrawn rather than re-reading it as a lapse.
    with pytest.raises(MeshRefusal) as raised:
        A.begin_run(record["approval_id"], run_id="run_1", root=root)
    assert raised.value.code == "approval_not_runnable"
    later = A.presented(raw, now=float(record["expires_at"]) + 600.0)
    assert later["state"] == "withdrawn"


def test_a_non_requester_cannot_withdraw(root: Path) -> None:
    """Only the surface that filed the request can settle it; a foreign requester
    block refuses and writes NOTHING."""
    record = A.create_request(**_device_request(A.new_request_id()), root=root)
    with pytest.raises(MeshRefusal) as raised:
        A.withdraw(
            record["approval_id"],
            requested_by={"session_id": "s9", "device_id": "d_other", "surface": "cli"},
            root=root,
        )
    assert raised.value.code == "approval_requester_mismatch"
    assert "filed by cli s1" in raised.value.sentence
    assert "did not supply that requester's identity" in raised.value.sentence
    assert "nothing was written" in raised.value.sentence
    raw = A._load_raw(record["approval_id"], root)
    assert raw["state"] == "requested"
    assert raw["audit"] == ["onboard_requested"]
    assert "withdrawn_by" not in raw


def test_a_withdrawal_is_refused_once_answered_or_running(root: Path) -> None:
    """Past ``requested`` the normal paths own the record — a withdrawal never
    competes with approve/deny, the run, retry or expiry. And the operator
    answering a withdrawn record gets a self-explaining refusal, never "the
    first decision wins" (no operator decision existed)."""
    approved = _approved_device_record(root)
    with pytest.raises(MeshRefusal) as answered:
        A.withdraw(approved["approval_id"], requested_by=dict(approved["requested_by"]), root=root)
    assert answered.value.code == "approval_withdraw_conflict"

    running = _approved_device_record(root)
    A.begin_run(running["approval_id"], run_id="run_1", root=root)
    with pytest.raises(MeshRefusal) as live:
        A.withdraw(running["approval_id"], requested_by=dict(running["requested_by"]), root=root)
    assert live.value.code == "approval_withdraw_conflict"
    # The refusal changed nothing: the run keeps its owner, and deny still lands.
    assert A.load_record(running["approval_id"], root=root)["state"] == "connecting"
    A.deny(running["approval_id"], decided_at=CREATED_AT + 30.0, root=root)
    assert A.load_record(running["approval_id"], root=root)["state"] == "denied"

    settled = A.create_request(**_device_request(A.new_request_id()), root=root)
    A.withdraw(settled["approval_id"], requested_by=dict(settled["requested_by"]), root=root)
    with pytest.raises(MeshRefusal) as no_answer:
        A.deny(settled["approval_id"], root=root)
    assert no_answer.value.code == "approval_decision_conflict"
    assert "withdrawn by its requester" in no_answer.value.sentence
    assert "first decision wins" not in no_answer.value.sentence


def test_a_request_id_spent_on_a_withdrawal_returns_it_unchanged(root: Path) -> None:
    """F1 across the new terminal: a retry of the spent id returns the withdrawn
    record verbatim — a re-request never resets a settle."""
    payload = _device_request(A.new_request_id())
    record = A.create_request(**payload, root=root)
    A.withdraw(record["approval_id"], requested_by=dict(record["requested_by"]), root=root)
    again = A.create_request(**payload, root=root)
    assert again["approval_id"] == record["approval_id"]
    assert again["state"] == "withdrawn"


def test_both_arrivals_of_an_expired_withdraw_answer_alike(root: Path) -> None:
    """NIT-2 (review round 1): the two ways a withdraw can meet a lapsed window
    — the fold this call materializes itself, and one a prior writer already
    landed — answer ALIKE (``approval_expired``), exactly as deny's two
    equivalents do. Neither arrival writes a withdrawal."""
    # (a) The window lapses before the call: withdraw itself folds it.
    lapsed = A.create_request(**_device_request(A.new_request_id()), root=root)
    path = root / "network" / "approvals" / f"{lapsed['approval_id']}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["expires_at"] = time.time() - 10.0
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(MeshRefusal) as folded_here:
        A.withdraw(lapsed["approval_id"], requested_by=dict(lapsed["requested_by"]), root=root)
    assert folded_here.value.code == "approval_expired"
    assert A._load_raw(lapsed["approval_id"], root)["state"] == "expired"

    # (b) Born past its window: the filing sweep folded it already, so withdraw
    # meets the materialized state instead of folding it.
    born = A.create_request(
        **_device_request(A.new_request_id(), created_at=1000.0, expires_at=1001.0),
        root=root,
    )
    assert born["state"] == "expired"
    with pytest.raises(MeshRefusal) as folded_then:
        A.withdraw(born["approval_id"], requested_by=dict(born["requested_by"]), root=root)
    assert folded_then.value.code == "approval_expired"
    assert folded_then.value.sentence == folded_here.value.sentence

    # Neither arrival wrote a withdrawal.
    final = A._load_raw(born["approval_id"], root)
    assert "withdrawn_by" not in final
    assert final["audit"][-1] == "onboard_expired"


# ---------------------------------------------------------------------------
# The audit trail
# ---------------------------------------------------------------------------


def test_every_transition_lands_a_mesh_audit_event(root: Path) -> None:
    """The taxonomy is closed AND the store actually reaches it.

    Read back from the real ``audit.jsonl``: the detail a transition carries
    must survive the whitelist (a key the whitelist drops would make the event
    useless exactly where an incident reader looks), and every emitted name
    must be in the closed set the writer enforces.
    """
    from local_operator.network import audit as audit_mod

    record = A.create_request(**_device_request(A.new_request_id()), root=root)
    A.deny(record["approval_id"], decided_at=CREATED_AT + 1.0, root=root)

    path = root / "network" / "audit.jsonl"
    events = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    names = [event["event"] for event in events]
    assert "onboard_requested" in names, names
    assert "onboard_denied" in names, names
    requested = next(event for event in events if event["event"] == "onboard_requested")
    assert requested["detail"].get("kind") == A.KIND_DEVICE_ONBOARD, requested
    denied = next(event for event in events if event["event"] == "onboard_denied")
    assert denied["subject"] == record["approval_id"], denied
    # A withdrawal lands its own row, and the requester's surface/session survive
    # the whitelist — the drill's redundant card must leave a self-explaining trail.
    second = A.create_request(**_device_request(A.new_request_id()), root=root)
    A.withdraw(second["approval_id"], requested_by=dict(second["requested_by"]), root=root)
    events = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    names = [event["event"] for event in events]
    assert "onboard_withdrawn" in names, names
    withdrawn = next(event for event in events if event["event"] == "onboard_withdrawn")
    assert withdrawn["subject"] == second["approval_id"], withdrawn
    assert withdrawn["detail"].get("surface") == "cli", withdrawn
    assert withdrawn["detail"].get("session_id") == "s1", withdrawn
    assert all(event["event"] in audit_mod.EVENT_KINDS for event in events), names


def test_records_tolerate_unknown_keys_and_forward_them(root: Path) -> None:
    """Forward-compat (slice (b), PR #1876): unknown keys are ignored, never fatal.

    The receipt vocabulary is additive — slice (b) landed a ``data`` key beside
    the frozen six (step/at/ok/detail/digest) — and readers must not error on it
    (or on any future addition). Equally load-bearing in the other direction: a
    mutation WRITES BACK what it read, so an extra key must not be silently
    dropped by this build either, or a reader-run that rewrites the record would
    erase the very field it does not understand. Both directions are asserted,
    because either one alone is a half-contract.
    """
    record = A.create_request(**_device_request(A.new_request_id()), root=root)
    approval_id = record["approval_id"]

    # A record as a LATER build writes it: one unknown top-level key and one
    # receipt carrying slice (b)'s additive ``data``.
    path = A.record_path(approval_id, root)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["future_field"] = {"nested": True}
    raw["receipts"] = [
        {
            "run_id": "r1",
            "step": "install",
            "at": CREATED_AT + 1.0,
            "ok": True,
            "detail": "",
            "digest": "",
            "data": {"bytes": 1234},
        }
    ]
    raw["state"] = A.STATE_CONNECTING
    path.write_text(json.dumps(raw), encoding="utf-8")

    # READS: fold + presentation, no error, extras carried.
    view = A.presented(A._load_raw(approval_id, root))
    assert view["state"] == A.STATE_CONNECTING
    assert view["future_field"] == {"nested": True}
    assert view["receipts"][0]["data"] == {"bytes": 1234}

    # WRITES: the next mutation preserves both extras (a run stays successful
    # through this build), which is what keeps a mixed fleet from pruning a
    # field its sibling wrote.
    decided = A.mark_connected(approval_id, run_id="r1", root=root)
    assert decided["state"] == A.STATE_CONNECTED
    again = json.loads(path.read_text(encoding="utf-8"))
    assert again["future_field"] == {"nested": True}
    assert again["receipts"][0]["data"] == {"bytes": 1234}

"""Closed new authority framing/lifecycle, with authentic legacy positive controls.

Software FILE_ONLY keys prove real cryptographic verification, not Touch ID.
Every negative begins with one valid card and changes one cause. No SSH, service,
account or live credential access is needed for the codec/presence boundaries.
"""

from __future__ import annotations

import copy
import json
import time

import pytest

from local_operator.network import approvals as a
from local_operator.network import enrollment as e
from local_operator.network.types import MeshRefusal
from local_operator.server.utils import desktop_approvals
from tests.unit.network.test_approvals_store import _make_key
from tests.unit.network.test_approvals_store import root as isolated_root

root = isolated_root


def fields(root, **changes):
    _make_key(root)
    trio = a.local_anchor_trio(root)
    result = {
        "kind": e.KIND,
        "request_id": "req_closed_fixture",
        "requested_by": {"session_id": "desktop", "device_id": "d_fixture", "surface": "desktop"},
        "device": {
            "device_id": "",
            "name": "fixture",
            "fingerprint": "",
            "host": "fixture.test:22",
            "user": "fixture",
            "transport": "ssh",
            "host_key_fp": "SHA256:fixture",
        },
        "what": {
            "connect": True,
            "install": True,
            "anchor": {k: trio[k] for k in ("key_id", "spki_fp", "statement_digest")},
            "network_id": "n_fixture",
            "role": "drive",
            "grant": ["approve"],
            "unattended": True,
            "provisioning": "network_only",
        },
        "credential_ref": {"kind": "ssh", "ref": "fixture-ssh-reference"},
    }
    result.update(changes)
    return result


def create(root, **changes):
    return a.create_request(**fields(root, **changes), root=root)


def approve(root, record):
    return a.approve_network_enrollment(
        record["approval_id"],
        authority_version=1,
        scope="network_only",
        digest=record["request_digest"],
        root=root,
    )


def assert_wrapped(root, aid):
    raw = json.loads(a.record_path(aid, root).read_text())
    assert set(raw) == {"format", "version", "record"}
    assert raw["format"] == e.FORMAT and type(raw["version"]) is int and raw["version"] == 1
    assert "approval_id" not in raw
    assert raw["record"]["approval_id"] == aid
    assert not (a.approvals_dir(root) / (aid + ".json")).exists()
    return raw


def test_exact_domains_and_nonrenewing_lookup_first(root, monkeypatch):
    first = create(root)
    aid = first["approval_id"]
    before = a.record_path(aid, root).read_bytes()
    assert_wrapped(root, aid)
    immutable = a._immutable_request(first)
    import hashlib

    expected = (
        "sha256:"
        + hashlib.sha256(
            b"lop-network-enrollment-request-v1\0"
            + a.canonical_json(
                {
                    "authority_version": 1,
                    "approval_id": aid,
                    "request_id": first["request_id"],
                    "request": immutable,
                }
            )
        ).hexdigest()
    )
    assert first["request_digest"] == expected
    assert a.signed_payload(
        kind=e.KIND,
        request_id=first["request_id"],
        request_digest=expected,
        decision="approve",
        decided_at=1,
    ).startswith(b"lop-network-enrollment-v1\0")
    assert a.signed_payload(
        kind=a.KIND_DEVICE_ONBOARD,
        request_id="req_old",
        request_digest="old",
        decision="approve",
        decided_at=1,
    ).startswith(b"lop-approval-v1\0")
    # Minting a candidate ID even on a hit is an error: only genuine misses own a window.
    monkeypatch.setattr(
        a, "token_bytes", lambda n: (_ for _ in ()).throw(AssertionError("minted on retry"))
    )
    monkeypatch.setattr(a.time, "time", lambda: first["created_at"] + 120)
    repeated = create(root)
    for key in ("approval_id", "request_digest", "created_at", "expires_at"):
        assert repeated[key] == first[key]
    assert a.record_path(aid, root).read_bytes() == before
    altered = fields(root)
    altered["device"]["host"] = "other.fixture.test"
    with pytest.raises(MeshRefusal, match="already exists"):
        a.create_request(**altered, root=root)
    assert len(a.list_records(root=root)) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "outer_extra",
        "flat",
        "version_bool",
        "version_string",
        "version_unknown",
        "inner_version",
        "kind",
        "id",
        "request_id",
        "target",
        "scope",
        "role",
        "grant",
        "window",
        "reference",
    ],
)
def test_one_cause_codec_tamper_refuses_before_signer_or_run(root, mutation, monkeypatch):
    record = create(root)
    aid = record["approval_id"]
    wrapped = assert_wrapped(root, aid)
    raw = copy.deepcopy(wrapped)
    inner = raw["record"]
    if mutation == "outer_extra":
        raw["approval_id"] = aid
    elif mutation == "flat":
        raw = inner
    elif mutation == "version_bool":
        raw["version"] = True
    elif mutation == "version_string":
        raw["version"] = "1"
    elif mutation == "version_unknown":
        raw["version"] = 2
    elif mutation == "inner_version":
        inner["authority_version"] = True
    elif mutation == "kind":
        inner["kind"] = a.KIND_DEVICE_ONBOARD
    elif mutation == "id":
        inner["approval_id"] = "apn_0000000000000"
    elif mutation == "request_id":
        inner["request_id"] = "req_changed"
    elif mutation == "target":
        inner["device"]["host"] = "other.fixture.test"
    elif mutation == "scope":
        inner["what"]["provisioning"] = "all"
    elif mutation == "role":
        inner["what"]["role"] = "admin"
    elif mutation == "grant":
        inner["what"]["grant"] = ["broker_credential"]
    elif mutation == "window":
        inner["expires_at"] += 60
    elif mutation == "reference":
        inner["credential_ref"]["ref"] = "different-reference"
    a.record_path(aid, root).write_text(json.dumps(raw))
    calls = []
    monkeypatch.setattr(a, "sign_decision", lambda **kwargs: calls.append(True))
    with pytest.raises(MeshRefusal) as err:
        approve(root, record)
    assert err.value.code == "network_enrollment_invalid"
    with pytest.raises(MeshRefusal) as err:
        a.verify_for_run(aid, root=root)
    assert err.value.code == "network_enrollment_invalid" and not calls


def test_generic_never_signs_typed_ack_once_and_write_race_is_rechecked(root, monkeypatch):
    record = create(root)
    aid = record["approval_id"]
    calls = []
    original = a.sign_decision

    def signer(**kwargs):
        calls.append(True)
        return original(**kwargs)

    monkeypatch.setattr(a, "sign_decision", signer)
    with pytest.raises(MeshRefusal) as err:
        desktop_approvals.approve(root, aid)
    assert err.value.code == e.CONFIRMATION and not calls
    with pytest.raises(MeshRefusal) as err:
        a.approve_network_enrollment(
            aid, authority_version=1, scope="network_only", digest="sha256:" + "0" * 64, root=root
        )
    assert err.value.code == "network_enrollment_snapshot_changed" and not calls
    for version in (True, "1", 2):
        with pytest.raises(MeshRefusal):
            a.approve_network_enrollment(
                aid,
                authority_version=version,
                scope="network_only",
                digest=record["request_digest"],
                root=root,
            )
    assert not calls
    approved = approve(root, record)
    assert len(calls) == 1 and approved["state"] == "approved"
    a.verify_for_run(aid, root=root)
    assert_wrapped(root, aid)
    # Presence for an expired/already answered card is not re-requested.
    with pytest.raises(MeshRefusal):
        approve(root, record)
    assert len(calls) == 1
    other = create(root, request_id="req_write_race")

    def racing_signer(**kwargs):
        signature = signer(**kwargs)
        a.deny(other["approval_id"], root=root)
        return signature

    monkeypatch.setattr(a, "sign_decision", racing_signer)
    with pytest.raises(MeshRefusal) as err:
        approve(root, other)
    assert err.value.code == "approval_decision_conflict"
    assert a.load_record(other["approval_id"], root=root)["state"] == "denied"
    assert_wrapped(root, other["approval_id"])


def test_all_shared_lifecycle_writes_wrap_and_tombstone_in_own_namespace(root):
    r = create(root)
    aid = r["approval_id"]
    approve(root, r)
    a.begin_run(aid, run_id="run_first", runner_pid=99999999, root=root)
    assert_wrapped(root, aid)
    a.append_receipt(aid, run_id="run_first", step="pre_read", ok=True, detail="fixture", root=root)
    assert_wrapped(root, aid)
    a.mark_failed(aid, run_id="run_first", step="join", detail="controlled interruption", root=root)
    assert_wrapped(root, aid)
    a.begin_run(aid, run_id="run_second", runner_pid=99999999, root=root)
    a.mark_connected(aid, run_id="run_second", root=root)
    assert_wrapped(root, aid)
    # New writers share exact existing retention, but not legacy index/lookup.
    a.sweep(root=root, now=time.time() + a.TERMINAL_PRUNE_AGE_S + 10)
    assert not a.record_path(aid, root).exists()
    assert len(a.read_index(root, enrollment_only=True)) == 1
    assert not a.read_index(root)
    with pytest.raises(MeshRefusal) as err:
        create(root)
    assert err.value.code == "approval_request_conflict"
    withdrawn = create(root, request_id="req_withdraw_wrapped")
    a.withdraw(withdrawn["approval_id"], requested_by=withdrawn["requested_by"], root=root)
    assert_wrapped(root, withdrawn["approval_id"])
    expired = create(root, request_id="req_expiry_wrapped")
    a.sweep(root=root, now=expired["expires_at"] + 1)
    assert a.load_record(expired["approval_id"], root=root)["state"] == "expired"
    assert_wrapped(root, expired["approval_id"])


def test_frozen_old_wire_copy_guard_and_real_signature_positive_controls(root):
    from tests.fixtures import approval_0731_wire as old
    from tests.unit.network.test_approvals_store import _sign

    pending = create(root)
    approved = approve(root, create(root, request_id="req_old_wire_approved"))
    directory = root / "old-runtime-directory"
    directory.mkdir()
    signs = []
    for record in (pending, approved):
        raw = assert_wrapped(root, record["approval_id"])
        # Exact original bytes, then same bytes under a renamed OLD identifier.
        for name in (record["approval_id"], "ap_renamed_fixture"):
            (directory / (name + ".json")).write_text(json.dumps(raw))
            with pytest.raises(ValueError, match="unknown_approval"):
                old.load(directory, name)
            with pytest.raises(ValueError, match="unknown_approval"):
                old.presign(directory, name, lambda r: signs.append(True))
            assert not old.rows(directory) and not signs
    spki = a.local_anchor_trio(root)["spki"]
    for kind in (a.KIND_DEVICE_ONBOARD, "unknown_new_kind_positive_control"):
        legacy = {**copy.deepcopy(approved), "kind": kind, "approval_id": "ap_control"}
        legacy["request_digest"] = old.digest(legacy)
        legacy["signature"] = {"sig": _sign(root, old.payload(legacy))}
        assert old.verifies(
            legacy, spki
        ), "broken frozen verifier cannot certify negative replay safety"
        assert (
            old.digest(legacy)
            == "sha256:"
            + __import__("hashlib")
            .sha256(a.canonical_json(a._immutable_request(legacy)))
            .hexdigest()
        )
    # Unwrapping is NOT pre-presence safe; old execution still cannot consume
    # the new commitment. This is deliberately a nonzero signer control.
    (directory / (pending["approval_id"] + ".json")).write_text(json.dumps(pending))
    old.presign(directory, pending["approval_id"], lambda r: signs.append(True))
    assert len(signs) == 1 and not old.verifies(approved, spki)


def test_new_and_old_decision_domains_are_independently_disjoint(root):
    from tests.fixtures import approval_0731_wire as old
    from tests.unit.network.test_approvals_store import _sign

    record = approve(root, create(root))
    aid = record["approval_id"]
    spki = a.local_anchor_trio(root)["spki"]
    a.verify_for_run(aid, root=root)  # authentic new-domain positive first
    raw = assert_wrapped(root, aid)
    raw["record"]["signature"]["sig"] = _sign(root, old.payload(record))
    a.record_path(aid, root).write_text(json.dumps(raw))
    # Modern commitment is unchanged: ONLY decision domain differs.
    assert e.commitment(raw["record"]) == record["request_digest"]
    with pytest.raises(MeshRefusal) as err:
        a.verify_for_run(aid, root=root)
    assert err.value.code == "approval_record_tampered"
    # Conversely give old verifier a legacy-compatible request commitment but
    # a genuine new-domain signature over THAT same commitment. Domain is now
    # the sole failure cause, not a simultaneous request-digest mismatch.
    changed = copy.deepcopy(record)
    changed["request_digest"] = old.digest(changed)
    message = a.signed_payload(
        kind=e.KIND,
        request_id=changed["request_id"],
        request_digest=changed["request_digest"],
        decision="approve",
        decided_at=changed["decided_at"],
    )
    changed["signature"]["sig"] = _sign(root, message)
    assert not old.verifies(changed, spki)
    changed["signature"]["sig"] = _sign(root, old.payload(changed))
    assert old.verifies(changed, spki)


def test_unknown_versions_and_unwrapped_modern_files_never_fall_back(root):
    r = create(root)
    aid = r["approval_id"]
    p = a.record_path(aid, root)
    p.write_text(json.dumps(r))
    with pytest.raises(MeshRefusal):
        a.load_record(aid, root=root)
    p.write_text(json.dumps({"format": e.FORMAT, "version": 1, "record": r, "approval_id": aid}))
    with pytest.raises(MeshRefusal):
        a.load_record(aid, root=root)
    for outer, inner in ((True, 1), (1, True), ("1", 1), (1, "1"), (2, 1), (1, 2)):
        changed = copy.deepcopy(r)
        changed["authority_version"] = inner
        p.write_text(json.dumps({"format": e.FORMAT, "version": outer, "record": changed}))
        with pytest.raises(MeshRefusal):
            a.load_record(aid, root=root)


def test_same_target_lease_spans_new_and_legacy_kinds(root):
    from local_operator.network import onboard
    from tests.unit.server.test_desktop_approvals import _file_device_record

    new = approve(root, create(root))
    old = _file_device_record(root, with_anchor=True)
    # An authentic legacy card with EXACT same target/network is distinct
    # authority but cannot install concurrently in this new runtime.
    old["device"].update(new["device"])
    old["device"]["host"] = new["device"]["host"].removesuffix(":22")
    old["what"]["network_id"] = new["what"]["network_id"]
    old["request_digest"] = a.request_digest(old)
    a._write_record(old, root)
    desktop_approvals.approve(root, old["approval_id"])
    import os
    import selectors
    import signal
    import subprocess
    import sys

    script = """
import os,sys
from pathlib import Path
from local_operator.operator import trust
from local_operator.network import onboard
root=Path(sys.argv[1])
trust.load_anchor=lambda uid=None: trust.AnchorLoad(
    anchor=None,path=root/'fixture-absent',root_owned=False,
    reason='fixture OS boundary',exists=False)
with onboard.approval_run(sys.argv[2],root=root):
    print('OWNED',flush=True)
    sys.stdin.readline()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(root), new["approval_id"]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        with selectors.DefaultSelector() as poll:
            poll.register(child.stdout, selectors.EVENT_READ)
            assert poll.select(10), "signed target-lease child did not become ready"
        assert child.stdout.readline().strip() == "OWNED"
        with pytest.raises(MeshRefusal) as err:
            with onboard.approval_run(old["approval_id"], root=root):
                pytest.fail("second target owner admitted across namespace/process boundary")
        assert err.value.code == "approval_target_in_flight"
    finally:
        if child.poll() is None:
            child.stdin.write("\n")
            child.stdin.flush()
            try:
                child.wait(5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(5)
        for pipe in (child.stdin, child.stdout, child.stderr):
            pipe.close()
    assert child.returncode == 0
    # The target claim was not split across authority namespaces or unlinked.
    assert list(a.approvals_dir(root).glob(".target-*.lock"))
    assert not list(a.approvals_dir(root, enrollment_only=True).glob(".target-*.lock"))


def test_new_namespace_catalog_and_symlinks_never_alias_legacy(root):
    record = create(root)
    aid = record["approval_id"]
    assert {p.name for p in (root / "network").iterdir() if p.is_dir()} == {"enrollment-approvals"}
    assert not a.approvals_dir(root).exists()
    p = a.record_path(aid, root)
    original = p.read_bytes()
    other = p.with_name("fixture-copy.json")
    other.write_bytes(original)
    p.unlink()  # Only this test's own just-created fixture, not operator data.
    p.symlink_to(other)
    with pytest.raises(MeshRefusal):
        a.load_record(aid, root=root)
    p.unlink()
    p.write_bytes(original)
    legacy = a.ensure_approvals_dir(root)
    modern = a.approvals_dir(root, enrollment_only=True)
    moved = modern.with_name("fixture-original-directory")
    modern.rename(moved)
    modern.symlink_to(legacy, target_is_directory=True)
    with pytest.raises(MeshRefusal):
        a.load_record(aid, root=root)
    assert not list(legacy.glob("apn_*.json"))


def test_network_only_verify_requires_known_positive_targeted_producer_facts():
    def report():
        return {
            "ok": False,
            "identity_present": True,
            "checks": [
                {
                    "check": "reachability",
                    "class": "admission",
                    "device_id": "d_target",
                    "ok": True,
                    "observed": {"outcome": "connected"},
                },
                {
                    "check": "readiness",
                    "class": "admission",
                    "device_id": "d_target",
                    "ok": True,
                    "capability": "build",
                    "source": "peer",
                },
                {
                    "check": "readiness",
                    "class": "equipment",
                    "device_id": "d_target",
                    "ok": True,
                    "capability": "operator_authority",
                    "source": "peer",
                },
                {
                    "check": "readiness",
                    "class": "equipment",
                    "device_id": "d_target",
                    "ok": False,
                    "capability": "model_credential",
                    "source": "peer",
                    "code": "not_configured",
                    "detail": "fixture lacks model",
                },
            ],
        }

    positive = report()
    rows = e.ready_evidence(positive, device_id="d_target", network_id="n_target")
    assert rows[0]["capability"] == "model_credential" and positive["ok"] is False
    variants = [
        None,
        {},
        {**report(), "checks": []},
        {**report(), "schema": 2},
        {**report(), "identity_present": False},
    ]
    for key, value in (
        ("class", []),
        ("class", "unknown"),
        ("ok", 1),
        ("device_id", "wrong_target"),
    ):
        changed = report()
        changed["checks"][0][key] = value
        variants.append(changed)
    for index, key, value in (
        (1, "capability", {}),
        (1, "source", "unknown"),
        (2, "ok", False),
        (2, "class", "admission"),
    ):
        changed = report()
        changed["checks"][index][key] = value
        variants.append(changed)
    changed = report()
    changed["checks"].pop(1)
    variants.append(changed)
    for invalid in variants:
        with pytest.raises(MeshRefusal):
            e.ready_evidence(invalid, device_id="d_target", network_id="n_target")


def test_cli_typed_disclosure_and_explicit_empty_grant_do_not_use_generic_approve(root, capsys):
    from argparse import Namespace

    from local_operator.network import cli

    r = create(root)
    rc = cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="approve-network-enrollment",
            approval=r["approval_id"],
            request_digest=r["request_digest"],
            json=True,
        )
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out)["state"] == "approved"
    assert "Network enrollment only; account logins are not shared" in captured.err
    assert "shared matching logins" not in captured.err
    for key in ("key_id", "spki_fp", "statement_digest"):
        assert r["what"]["anchor"][key] in captured.err
    assert "fixture-ssh-reference" not in captured.out + captured.err

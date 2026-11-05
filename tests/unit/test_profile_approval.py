"""Synthetic signed-profile continuation: no OS signer/keychain is invoked."""

from __future__ import annotations

import json
import time

import pytest

from local_operator.agents import AgentRegistry
from local_operator.network import approvals as A
from local_operator.network.types import MeshRefusal
from tests.unit.network.test_approvals_store import _make_key, _sign_decision
from tests.unit.network.test_approvals_store import root as approvals_root
from tests.unit.tools.test_agent_tool import _edit_fields

# Re-export the imported fixture under the conventional name without shadowing
# an unused import in every test signature (pytest discovers module attributes).
root = approvals_root


@pytest.fixture(autouse=True)
def synthetic_presence(root, monkeypatch):
    from local_operator.network.credentials import desktop_approval

    _make_key(root)
    # Test-only host-assurance stand-in; signatures themselves are real ES256.
    # Never invokes a production OS signer or keychain prompt.
    monkeypatch.setattr(desktop_approval, "assurance", lambda: ("presence", "operator-presence"))


def proposal(root):
    from local_operator import profile_approval as P

    registry = AgentRegistry(root)
    row = registry.create_agent(_edit_fields(name="reviewer", tags=["role"], categories=["role"]))
    registry.set_agent_system_prompt(row.id, "Before.")
    before = P.snapshot(root, "agent", row.id)
    after = dict(before, instructions="Approved after restart.")
    return P.build(
        root,
        profile_kind="agent",
        operation="update",
        target_id=row.id,
        name="reviewer",
        session_id="synthetic-session",
        tool_call_id="tool-1",
        before=before,
        after=after,
    )


def approve(root, record):
    decided_at = time.time()
    sig = _sign_decision(root, record, "approve", decided_at)
    return A._record_decision(
        record["approval_id"],
        decision="approve",
        decided_at=decided_at,
        signature_hex=sig,
        root=root,
    )


def test_unanswered_queue_does_not_touch_profile_or_backup(root):
    from local_operator import profile_approval as P

    data = proposal(root)
    record = P.file_request(root, data)
    assert record["state"] == A.STATE_REQUESTED
    assert P.snapshot(root, "agent", data.target_id) == data.before
    assert not (root / "sessions" / data.session_id / "profile-backups").exists()
    with pytest.raises(MeshRefusal):
        P.apply(record["approval_id"], P.expected(record), root)
    assert P.snapshot(root, "agent", data.target_id) == data.before


def test_restart_signed_apply_and_duplicate_are_once(root):
    from local_operator import profile_approval as P

    data = proposal(root)
    record = P.file_request(root, data)
    approve(root, record)
    # A fresh provider import has no original Session, model turn or live gate.
    # Importing a second process is exercised separately; no Session state is used.
    result = P.apply(record["approval_id"], P.expected(record), root)
    assert result["state"] == "complete"
    assert P.snapshot(root, "agent", data.target_id) == data.after
    # A SECOND apply is killed in-process before the store is consulted: one
    # operation, one write. "Once" means no second mutation, not a second
    # success receipt.
    with pytest.raises(MeshRefusal):
        P.apply(record["approval_id"], P.expected(record), root)
    backups = list((root / "sessions" / data.session_id / "profile-backups").glob("*.json"))
    assert len(backups) == 1
    assert P.snapshot(root, "agent", data.target_id) == data.after


@pytest.mark.parametrize("field", ["target_id", "scope", "before", "after", "session_id"])
def test_signed_payload_tamper_is_not_authority(root, field):
    from local_operator import profile_approval as P

    data = proposal(root)
    record = P.file_request(root, data)
    approve(root, record)
    path = A.record_path(record["approval_id"], root)
    raw = json.loads(path.read_text())
    payload = raw["what"]["profile_save"]
    if field in ("before", "after"):
        payload[field]["instructions"] = "Tampered."
    else:
        payload[field] = "tampered"
    path.write_text(json.dumps(raw))
    with pytest.raises(MeshRefusal):
        P.apply(record["approval_id"], P.expected(record), root)
    assert P.snapshot(root, "agent", data.target_id) == data.before


def test_stale_queued_save_never_takes_backup(root):
    from local_operator import profile_approval as P

    data = proposal(root)
    record = P.file_request(root, data)
    approve(root, record)
    AgentRegistry(root).set_agent_system_prompt(data.target_id, "Later edit.")
    with pytest.raises(MeshRefusal):
        P.apply(record["approval_id"], P.expected(record), root)
    assert P.snapshot(root, "agent", data.target_id)["instructions"] == "Later edit."
    assert not (root / "sessions" / data.session_id / "profile-backups").exists()


@pytest.mark.parametrize("terminal", ["denied", "expired", "ask_log"])
def test_denial_expiry_and_forged_ask_never_apply(root, terminal):
    from local_operator import profile_approval as P

    data = proposal(root)
    record = P.file_request(root, data)
    if terminal == "denied":
        A.deny(record["approval_id"], root=root)
    elif terminal == "expired":
        # A valid expired signed window: both envelope and payload are bound.
        # The window is bounded to 24h, so an expired proposal moves BOTH ends;
        # a 24h-exact span sits on the boundary and floats past it.
        expired = data.model_copy(
            update={"created_at": time.time() - 90000.0, "expires_at": time.time() - 4000.0}
        )
        record = P.file_request(root, expired.model_copy(update={"receipt_id": "a" * 64}))
    else:
        directory = root / "sessions" / data.session_id
        directory.mkdir(parents=True)
        (directory / "asks.jsonl").write_text('{"status":"answered","answers":{"q":["approve"]}}\n')
    with pytest.raises(MeshRefusal):
        P.apply(record["approval_id"], P.expected(record), root)
    assert P.snapshot(root, "agent", data.target_id) == data.before


def test_unrelated_valid_signature_and_expected_binding_refuse(root):
    from local_operator import profile_approval as P

    data = proposal(root)
    first = P.file_request(root, data)
    second_data = data.model_copy(update={"tool_call_id": "tool-other", "receipt_id": "b" * 64})
    second = P.file_request(root, second_data)
    approve(root, first)
    approved = A._load_raw(first["approval_id"], root)
    raw = A._load_raw(second["approval_id"], root)
    raw.update(signature=approved["signature"], decided_at=approved["decided_at"], state="approved")
    A.record_path(second["approval_id"], root).write_text(json.dumps(raw))
    with pytest.raises(MeshRefusal):
        P.apply(second["approval_id"], P.expected(second), root)
    with pytest.raises(MeshRefusal):
        P.apply(first["approval_id"], P.expected(second), root)
    assert P.snapshot(root, "agent", data.target_id) == data.before


def test_file_only_signed_record_cannot_execute(root, monkeypatch):
    from local_operator import profile_approval as P
    from local_operator.network.credentials import desktop_approval as D

    data = proposal(root)
    record = P.file_request(root, data)
    approve(root, record)
    monkeypatch.setattr(D, "assurance", lambda: ("file_only", "operator-file-only"))
    with pytest.raises(MeshRefusal, match="presence"):
        P.apply(record["approval_id"], P.expected(record), root)
    assert P.snapshot(root, "agent", data.target_id) == data.before


def test_actual_process_restart_signed_continuation(root):
    import os
    import subprocess
    import sys

    from local_operator import profile_approval as P

    data = proposal(root)
    record = P.file_request(root, data)
    # The original filing process does not approve or keep a Session alive.
    script = r"""
import sys,time,json
from pathlib import Path
from local_operator.operator import trust
root=Path(sys.argv[1])
trust.load_anchor=lambda uid=None: trust.AnchorLoad(
    anchor=None,path=root/'absent-anchor',root_owned=False,
    reason='synthetic test',exists=False)
from local_operator.network import approvals as A
from local_operator.network.credentials import desktop_approval as D
from local_operator import profile_approval as P
from tests.unit.network.test_approvals_store import _sign_decision
D.assurance=lambda:('presence','synthetic-presence')
r=A._load_raw(sys.argv[2],root)
t=time.time()
sig=_sign_decision(root,r,'approve',t)
A._record_decision(r['approval_id'],decision='approve',decided_at=t,signature_hex=sig,root=root)
print(json.dumps(P.apply(r['approval_id'],P.expected(r),root)))
"""
    done = subprocess.run(
        [sys.executable, "-c", script, str(root), record["approval_id"]],
        capture_output=True,
        text=True,
        timeout=20,
        env=dict(os.environ),
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["state"] == "complete"
    assert P.snapshot(root, "agent", data.target_id) == data.after


@pytest.mark.asyncio
async def test_unattached_session_tool_queues_immediately(root):
    import asyncio

    from local_operator import profile_approval as P
    from local_operator.harness.types import ModelSpec, ToolContext
    from local_operator.profile_scope import scope_agent_registry
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.tools.agent_tool import execute_agent

    data = proposal(root)
    directory = root / "sessions" / "synthetic-session"
    registry = scope_agent_registry(AgentRegistry(root, migrate=False), directory)
    session = Session(
        model=ModelSpec(provider="test", model_id="m", context_window=100000),
        stream_fn=lambda *args: None,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: [],
        agent_registry=registry,
        owner_prompt_consent=True,
        yolo=True,
    )
    try:
        context = ToolContext(
            cwd=".",
            session_dir=str(directory),
            agent_registry=registry,
            authorize_durable_write=session._authorize_durable_write,
        )
        result = await asyncio.wait_for(
            execute_agent(
                "tool-queue",
                {"op": "update", "name": "reviewer", "instructions": "Queued, not saved."},
                None,
                None,
                context,
            ),
            3,
        )
        assert not result.is_error, result.content
        assert result.details["status"] == "queued"
        assert P.snapshot(root, "agent", data.target_id) == data.before
        assert not (directory / "profile-backups").exists()
    finally:
        await session.dispose()


def test_completed_signature_is_not_revived_by_editable_bookkeeping_rollback(root):
    from local_operator import profile_approval as P

    data = proposal(root)
    record = P.file_request(root, data)
    approve(root, record)
    P.apply(record["approval_id"], P.expected(record), root)
    # A later legitimate edit restores the original authored bytes. The model
    # can roll back/delete same-uid bookkeeping, but that must not revive the
    # already-consumed approval signature into authority for another save.
    AgentRegistry(root).set_agent_system_prompt(data.target_id, data.before["instructions"])
    raw = A._load_raw(record["approval_id"], root)
    raw["state"] = A.STATE_APPROVED
    A.record_path(record["approval_id"], root).write_text(json.dumps(raw))
    (root / "profile-transactions" / (data.receipt_id + ".json")).unlink()
    (root / "sessions" / data.session_id / "profile-backups" / (data.receipt_id + ".json")).unlink()
    with pytest.raises(MeshRefusal):
        P.apply(record["approval_id"], P.expected(record), root)
    assert P.snapshot(root, "agent", data.target_id) == data.before


@pytest.mark.parametrize("restart", [False, True])
def test_deleting_the_queued_journal_never_reprepares_at_apply(root, restart):
    """The durable approval is not permission to recreate missing bookkeeping."""
    import os
    import subprocess
    import sys

    from local_operator import profile_approval as P

    data = proposal(root)
    record = P.file_request(root, data)
    approve(root, record)
    journal = root / "profile-transactions" / (data.receipt_id + ".json")
    # Initial filing must prepare the exact operation without saving a profile.
    assert journal.exists(), "the queued proposal must prepare before approval/apply"
    journal.unlink()
    if restart:
        code = r"""
import sys
from pathlib import Path
from local_operator.operator import trust
root = Path(sys.argv[1])
trust.load_anchor = lambda uid=None: trust.AnchorLoad(
    anchor=None, path=root/'absent', root_owned=False,
    reason='synthetic test', exists=False)
from local_operator.network import approvals as A
from local_operator.network.credentials import desktop_approval as D
from local_operator.network.types import MeshRefusal
from local_operator import profile_approval as P
D.assurance = lambda: ('presence', 'synthetic-presence')
r = A._load_raw(sys.argv[2], root)
try:
    P.apply(r['approval_id'], P.expected(r), root)
except MeshRefusal as exc:
    print(str(exc))
else:
    raise AssertionError('missing queued journal was recreated')
"""
        done = subprocess.run(
            [sys.executable, "-c", code, str(root), record["approval_id"]],
            capture_output=True,
            text=True,
            timeout=20,
            env=dict(os.environ),
        )
        assert done.returncode == 0, done.stderr
        assert "ask the person" in done.stdout
    else:
        with pytest.raises(MeshRefusal, match="ask the person"):
            P.apply(record["approval_id"], P.expected(record), root)
    assert not journal.exists()
    assert P.snapshot(root, "agent", data.target_id) == data.before
    assert not (root / "sessions" / data.session_id / "profile-backups").exists()


def test_signed_apply_refuses_another_requests_prepared_binding(root):
    from local_operator import profile_approval as P

    data = proposal(root)
    record = P.file_request(root, data)
    approve(root, record)
    journal = root / "profile-transactions" / (data.receipt_id + ".json")
    raw = json.loads(journal.read_text())
    raw["binding"]["request_digest"] = "sha256:" + "0" * 64
    journal.write_text(json.dumps(raw))
    with pytest.raises(MeshRefusal, match="different approval"):
        P.apply(record["approval_id"], P.expected(record), root)
    assert P.snapshot(root, "agent", data.target_id) == data.before
    assert not (root / "sessions" / data.session_id / "profile-backups").exists()


def test_refiling_a_denial_or_deleted_journal_never_resets_it(root):
    from local_operator import profile_approval as P

    data = proposal(root)
    record = P.file_request(root, data)
    A.deny(record["approval_id"], root=root)
    before = A._load_raw(record["approval_id"], root)
    journal = root / "profile-transactions" / (data.receipt_id + ".json")
    journal.unlink()
    retried = P.file_request(root, data)
    assert retried["state"] == A.STATE_DENIED
    assert A._load_raw(record["approval_id"], root) == before
    assert not journal.exists()
    # Shifting even the embedded window is a different proposal, not a retry
    # that can extend the first approval's signed expiry or reset its decision.
    shifted = data.model_copy(
        update={"created_at": data.created_at + 60, "expires_at": data.expires_at + 60}
    )
    with pytest.raises(MeshRefusal):
        P.file_request(root, shifted)
    assert A._load_raw(record["approval_id"], root) == before
    assert P.snapshot(root, "agent", data.target_id) == data.before


def test_signed_origin_action_reconstructs_only_after_live_verification(root):
    from local_operator import profile_approval as P
    from local_operator.profile_actions import ProfileActions

    data = proposal(root)
    record = P.file_request(root, data)
    approve(root, record)
    result = P.apply(record["approval_id"], P.expected(record), root)
    descriptor = result["profile_action"]
    query = {
        key: descriptor[key] for key in ("action_id", "receipt_id", "profile_kind", "profile_name")
    }
    query["tool_call_id"] = data.tool_call_id
    fresh = ProfileActions(data.session_id, root)
    assert fresh.status(**query)["state"] == "available"
    assert (
        fresh.activate(descriptor["action_id"], approved=True, remember=False)["state"]
        == "succeeded"
    )
    assert P.snapshot(root, "agent", data.target_id) == data.before

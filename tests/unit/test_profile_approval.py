"""Synthetic signed-profile continuation: no OS signer/keychain is invoked."""

from __future__ import annotations

import json
import time

import pytest

from local_operator.agents import AgentRegistry
from local_operator.harness.types import Message
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
    assert before is not None
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
    checked_image_0 = P.snapshot(root, "agent", data.target_id)
    assert checked_image_0 is not None
    assert checked_image_0["instructions"] == "Later edit."
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
    from tests.unit.session.test_account_profile_consent import _never_stream

    session = Session(
        model=ModelSpec(provider="test", model_id="m", context_window=100000),
        stream_fn=_never_stream,
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
        assert result.details is not None
        assert result.details["status"] == "proposed"
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
    assert data.before is not None
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


def test_signed_revert_stays_consumed_after_legitimate_aba_and_restart(root):
    from local_operator import profile_approval as P
    from local_operator.profile_actions import ProfileActions

    data = proposal(root)
    record = P.file_request(root, data)
    approve(root, record)
    result = P.apply(record["approval_id"], P.expected(record), root)
    action = result["profile_action"]
    query = {
        key: action[key] for key in ("action_id", "receipt_id", "profile_kind", "profile_name")
    }
    query["tool_call_id"] = data.tool_call_id
    first = ProfileActions(data.session_id, root)
    assert first.status(**query)["state"] == "available"
    assert (
        first.activate(action["action_id"], approved=True, remember=False)["state"] == "succeeded"
    )
    P._write(root, data, data.after)  # A later legitimate edit restores B, no journal tampering.
    restarted = ProfileActions(data.session_id, root)
    assert restarted.status(**query)["state"] == "succeeded"
    assert (
        restarted.activate(action["action_id"], approved=True, remember=False)["state"]
        == "succeeded"
    )
    assert P.snapshot(root, "agent", data.target_id) == data.after

    import subprocess
    import sys

    script = r"""
import sys,json
from pathlib import Path
from local_operator.operator import trust
root=Path(sys.argv[1])
trust.load_anchor=lambda uid=None: trust.AnchorLoad(
    anchor=None,path=root/'absent',root_owned=False,reason='synthetic',exists=False)
from local_operator.network.credentials import desktop_approval as D
D.assurance=lambda:('presence','synthetic')
from local_operator.profile_actions import ProfileActions
query=json.loads(sys.argv[3])
actions=ProfileActions(sys.argv[2],root)
print(json.dumps([actions.status(**query),actions.activate(query['action_id'],approved=True,remember=False)]))
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(root), data.session_id, json.dumps(query)],
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert child.returncode == 0, child.stderr
    assert [row["state"] for row in json.loads(child.stdout)] == ["succeeded", "succeeded"]
    assert P.snapshot(root, "agent", data.target_id) == data.after


@pytest.mark.parametrize("damage", ["missing", "corrupt", "applying"])
@pytest.mark.parametrize("consumed", [False, True])
def test_revert_journal_damage_never_recreates_authority(root, damage, consumed):
    from local_operator import profile_approval as P
    from local_operator.profile_actions import ProfileActions

    data = proposal(root)
    record = P.file_request(root, data)
    approve(root, record)
    action = P.apply(record["approval_id"], P.expected(record), root)["profile_action"]
    if consumed:
        query = {
            key: action[key] for key in ("action_id", "receipt_id", "profile_kind", "profile_name")
        }
        query["tool_call_id"] = data.tool_call_id
        owner = ProfileActions(data.session_id, root)
        owner.status(**query)
        assert (
            owner.activate(action["action_id"], approved=True, remember=False)["state"]
            == "succeeded"
        )
        P._write(root, data, data.after)
    path = root / "profile-transactions" / ("revert_" + action["action_id"] + ".json")
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_text("[]")
    else:
        value = json.loads(path.read_text())
        value["state"] = "applying"
        path.write_text(json.dumps(value))
    query = {
        key: action[key] for key in ("action_id", "receipt_id", "profile_kind", "profile_name")
    }
    query["tool_call_id"] = data.tool_call_id
    fresh = ProfileActions(data.session_id, root)
    assert fresh.status(**query)["state"] in {"unavailable", "pending"}
    assert fresh.activate(action["action_id"], approved=True, remember=False)["state"] in {
        "unavailable",
        "pending",
    }
    assert P.snapshot(root, "agent", data.target_id) == data.after


def test_postcommit_exception_is_proven_success_not_a_failed_nochange(root, monkeypatch):
    from local_operator import profile_approval as P
    from local_operator.profile_actions import ProfileActions

    data = proposal(root)
    record = P.file_request(root, data)
    approve(root, record)
    action = P.apply(record["approval_id"], P.expected(record), root)["profile_action"]
    query = {
        key: action[key] for key in ("action_id", "receipt_id", "profile_kind", "profile_name")
    }
    query["tool_call_id"] = data.tool_call_id
    fresh = ProfileActions(data.session_id, root)
    assert fresh.status(**query)["state"] == "available"
    original = P.write_image

    def postcommit(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("synthetic exception AFTER actual authored commit")

    monkeypatch.setattr(P, "write_image", postcommit)
    assert (
        fresh.activate(action["action_id"], approved=True, remember=False)["state"] == "succeeded"
    )
    assert P.snapshot(root, "agent", data.target_id) == data.before


def test_interrupted_terminal_bookkeeping_reconciles_without_second_mutation(root, monkeypatch):
    from local_operator import definition_transaction as T
    from local_operator import profile_approval as P
    from local_operator.profile_actions import ProfileActions

    data = proposal(root)
    record = P.file_request(root, data)
    approve(root, record)
    action = P.apply(record["approval_id"], P.expected(record), root)["profile_action"]
    query = {
        key: action[key] for key in ("action_id", "receipt_id", "profile_kind", "profile_name")
    }
    query["tool_call_id"] = data.tool_call_id
    fresh = ProfileActions(data.session_id, root)
    fresh.status(**query)
    original = T._durable_json

    def interrupted(path, body):
        if (
            path.name == "revert_" + action["action_id"] + ".json"
            and body.get("state") == "complete"
        ):
            raise OSError("synthetic lost completion fsync")
        original(path, body)

    with monkeypatch.context() as m:
        m.setattr(T, "_durable_json", interrupted)
        assert (
            fresh.activate(action["action_id"], approved=True, remember=False)["state"] != "failed"
        )
    assert P.snapshot(root, "agent", data.target_id) == data.before
    restarted = ProfileActions(data.session_id, root)
    assert restarted.status(**query)["state"] == "succeeded"
    P._write(root, data, data.after)
    assert (
        restarted.activate(action["action_id"], approved=True, remember=False)["state"]
        == "succeeded"
    )
    assert P.snapshot(root, "agent", data.target_id) == data.after


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["agent", "team"])
@pytest.mark.parametrize("live_delivery", [False, True])
async def test_offline_signed_continuation_replaces_native_queued_result_once(
    root, monkeypatch, kind, live_delivery
):
    from local_operator import profile_approval as P
    from local_operator.harness.types import (
        Message,
        ModelSpec,
        ToolCall,
        ToolContext,
        ToolExecutionEndEvent,
    )
    from local_operator.network import approvals as A
    from local_operator.profile_scope import scope_agent_registry
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.tools.agent_tool import execute_agent

    proposal(root)
    directory = root / "sessions" / "synthetic-session"
    from local_operator.profile_scope import scope_team_registry
    from local_operator.server.features import static_feature_flags
    from local_operator.teams import TeamEditFields, TeamRegistry
    from local_operator.tools.team_tool import execute_team

    assert static_feature_flags()["profile_approvals"] == 1
    TeamRegistry(root).create_team(
        TeamEditFields(name="lopdev", manager="reviewer", instructions="Old team.")
    )

    def new_session():
        return Session(
            model=ModelSpec(provider="test", model_id="m", context_window=100000),
            stream_fn=lambda *args: (_ for _ in ()).throw(AssertionError("NO model turn")),
            tools=[],
            transcript=Transcript(directory),
            system_blocks_provider=lambda: [],
            agent_registry=scope_agent_registry(AgentRegistry(root, migrate=False), directory),
            team_registry=scope_team_registry(TeamRegistry(root), directory),
            owner_prompt_consent=True,
            yolo=True,
        )

    original = new_session()
    args = {
        "op": "update",
        "name": "reviewer" if kind == "agent" else "lopdev",
        "instructions": "Approved offline guidance.",
    }
    context = ToolContext(
        cwd=".",
        session_dir=str(directory),
        agent_registry=scope_agent_registry(AgentRegistry(root, migrate=False), directory),
        team_registry=scope_team_registry(TeamRegistry(root), directory),
        authorize_durable_write=original._authorize_durable_write,
    )
    queued = await (execute_agent if kind == "agent" else execute_team)(
        "same-real-call", args, None, None, context
    )
    assert queued.details is not None
    assert queued.details["status"] == "proposed"
    from local_operator import profile_proposals as PP
    from local_operator.harness.types import TextContent, ToolResult

    proposed = PP.get(root, queued.details["proposal_id"])
    assert proposed is not None
    legacy = P.build(
        root,
        profile_kind=kind,
        operation="update",
        target_id=proposed["target_id"],
        name=proposed["name"],
        session_id=original._session_id,
        tool_call_id=queued.tool_call_id,
        before=proposed["before"],
        after=proposed["after"],
    )
    legacy_record = P.file_request(root, legacy)
    queued = ToolResult(
        tool_call_id=queued.tool_call_id,
        tool_name=kind,
        content=[TextContent(text="Queued legacy profile save for signed approval.")],
        details={
            "status": "queued",
            "approval_id": legacy_record["approval_id"],
            "receipt_id": legacy.receipt_id,
            "profile_kind": kind,
            "profile_name": proposed["name"],
            "scope": "global",
        },
    )
    assert queued.details is not None and queued.details["status"] == "queued"
    # The ordinary harness native persistence boundary, using the ACTUAL call/result.
    call = Message.assistant(
        tool_calls=[ToolCall(id=queued.tool_call_id, name=queued.tool_name, arguments=args)]
    )
    queued_message = Message.tool_result(queued)
    await original.transcript.append_messages([call, queued_message])
    original._context.messages.extend([call, queued_message])
    if not live_delivery:
        await original.dispose()
    resumed = original if live_delivery else new_session()
    events = []
    resumed.subscribe(events.append)
    live = None
    try:
        await resumed.seed_history(resumed.transcript.build_llm_history())
        if live_delivery:
            from tests.unit.session.runtime import test_approval_authority_seam as seam

            monkeypatch.setattr(seam, "_AttachableSession", lambda: resumed)
            live = await seam._serve(root, operator_cap=None)
        assert queued.details is not None
        record = A._load_raw(queued.details["approval_id"], root)
        approve(root, record)
        # A peer-supplied hint, even for an approved but UNEXECUTED request,
        # cannot manufacture a saved result or write the profile.
        await resumed.receive_profile_save_result("f" * 64)
        assert queued.details is not None
        await resumed.receive_profile_save_result(queued.details["receipt_id"])
        assert not any(isinstance(event, ToolExecutionEndEvent) for event in events)
        applied = P.apply(record["approval_id"], P.expected(record), root)
        from types import SimpleNamespace

        from local_operator.session.runtime.process import _drain_inbox_into

        if live_delivery:
            import asyncio

            async with asyncio.timeout(5):
                while not any(isinstance(event, ToolExecutionEndEvent) for event in events):
                    await asyncio.sleep(0.01)
        else:
            assert (
                await _drain_inbox_into(
                    SimpleNamespace(
                        _session=resumed, receive_peer_message=resumed.receive_peer_message
                    )
                )
                == 1
            )
        # The live carrier is not a third peer provenance kind. It invokes the
        # real Session verifier; a forged hint cannot add a model/owner row.
        from local_operator.session.runtime.server import RuntimeServer

        carrier = object.__new__(RuntimeServer)
        carrier._started = True

        async def verify_hint(receipt_id: str) -> str:
            await resumed.receive_profile_save_result(receipt_id)
            return "verified hint"

        async def never_peer(**fields):
            pytest.fail("an internal signed-result hint must not become peer/model input")

        monkeypatch.setattr(
            carrier,
            "_handle",
            SimpleNamespace(
                receive_profile_save_result=verify_hint, receive_peer_message=never_peer
            ),
            raising=False,
        )
        frame = {
            "op": "peer_message",
            "text": "internal hint",
            "sender": {"source": "profile_save"},
            "custom_type": "profile_save_result",
            "mode": "mailbox",
            "wake": False,
            "message_id": queued.details["receipt_id"],
        }
        assert await carrier._dispatch("peer_message", frame) == "verified hint"
        before_forged = len(events)
        assert (
            await carrier._dispatch("peer_message", frame | {"message_id": "unsigned-hint"})
            == "verified hint"
        )
        assert len(events) == before_forged
        with pytest.raises(ValueError, match="non-waking"):
            await carrier._dispatch("peer_message", frame | {"wake": True})
        native = [event for event in events if isinstance(event, ToolExecutionEndEvent)]
        assert len(native) == 1
        assert native[0].tool_call_id == queued.tool_call_id
        assert queued.details is not None
        assert native[0].result.details is not None
        assert native[0].result.details["receipt_id"] == queued.details["receipt_id"]
        assert native[0].result.details is not None
        assert native[0].result.details["profile_action"] == applied["profile_action"]
        history = Transcript(directory).build_llm_history()
        tool = [
            message
            for message in history
            if getattr(message, "tool_call_id", None) == queued.tool_call_id
        ]
        assert len(tool) == 1
        assert isinstance(tool[0], Message)
        assert tool[0].provider_payload is not None
        assert tool[0].provider_payload["details"]["status"] == "applied"
        P._publish(root, P._proposal(record), applied)
        assert await _drain_inbox_into(
            SimpleNamespace(_session=resumed, receive_peer_message=resumed.receive_peer_message)
        ) in (0, 1)
        assert len([event for event in events if isinstance(event, ToolExecutionEndEvent)]) == 1
        import os
        import subprocess
        from pathlib import Path

        evidence = Path(os.environ.get("PROFILE_CONTINUATION_EVIDENCE", str(root)))
        git_env = dict(
            os.environ,
            GIT_CONFIG_SYSTEM="/dev/null",
            GIT_TERMINAL_PROMPT="0",
            GIT_ASKPASS="/bin/false",
        )
        golden = {
            "meta": {
                "source_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True, env=git_env
                ).strip(),
                "uncommitted_source": bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"], text=True, env=git_env
                    ).strip()
                ),
                "provenance": (
                    "Actual execute_agent queued result; original Session disposed; "
                    "signed apply; real inbox consumer emits native event and upserts "
                    "cold history. No LLM dispatch."
                ),
            },
            "source_session_id": resumed._session_id,
            "queued_tool_result": queued.model_dump(mode="json"),
            "applied_tool_result": native[0].result.model_dump(mode="json"),
            "tool_execution_end_event": native[0].model_dump(mode="json"),
            "cold_history_message": tool[0].model_dump(mode="json"),
        }
        if live_delivery:
            golden["meta"]["provenance"] = (
                "Actual publisher -> non-waking peer_message socket -> verified native consumer "
                "-> fresh cold history. No LLM dispatch."
            )
            (evidence / f"profile-continuation-live-{kind}.json").write_text(
                json.dumps(golden, indent=2)
            )
        else:
            (evidence / f"profile-continuation-golden-{kind}.json").write_text(
                json.dumps(golden, indent=2)
            )
            if kind == "agent":
                (evidence / "profile-continuation-golden.json").write_text(
                    json.dumps(golden, indent=2)
                )
        assert not resumed._run_owner_texts
        assert P.snapshot(root, kind, P._proposal(record).target_id) == P._proposal(record).after
        if not live_delivery:
            await resumed.transcript.compact_file(min_reclaim_bytes=0)
            restarted = new_session()
            after_restart = []
            restarted.subscribe(after_restart.append)
            try:
                assert queued.details is not None
                await restarted.receive_profile_save_result(queued.details["receipt_id"])
                assert not any(isinstance(event, ToolExecutionEndEvent) for event in after_restart)
                replay = Transcript(directory).build_llm_history()
                assert (
                    len(
                        [
                            m
                            for m in replay
                            if getattr(m, "tool_call_id", None) == queued.tool_call_id
                        ]
                    )
                    == 1
                )
            finally:
                await restarted.dispose()

    finally:
        if live is not None:
            await live.close(root)
        else:
            await resumed.dispose()


def test_signed_save_split_metadata_prompt_fault_rolls_back(root, monkeypatch):
    from local_operator import profile_approval as P

    data = proposal(root)
    data = P.build(
        root,
        profile_kind="agent",
        operation="update",
        target_id=data.target_id,
        name=data.name,
        session_id=data.session_id,
        tool_call_id=data.tool_call_id,
        before=data.before,
        after=dict(data.after, description="New approved description"),
    )
    record = P.file_request(root, data)
    approve(root, record)
    original = AgentRegistry.set_agent_system_prompt
    calls = []

    def fail_once(self, identity, text):
        calls.append(text)
        if len(calls) == 1:
            raise OSError("fault after metadata before prompt")
        return original(self, identity, text)

    monkeypatch.setattr(AgentRegistry, "set_agent_system_prompt", fail_once)
    with pytest.raises(MeshRefusal, match="fault after metadata"):
        P.apply(record["approval_id"], P.expected(record), root)
    assert P.snapshot(root, "agent", data.target_id) == data.before
    journal = json.loads((root / "profile-transactions" / f"{data.receipt_id}.json").read_text())
    assert journal["state"] == "rolled_back"
    assert data.before is not None
    assert calls == [data.after["instructions"], data.before["instructions"]]


_POLICY_CASES = {
    "tools": {"tags": ["role", "tools:read"]},
    "effort": {"tags": ["role", "effort:hi"]},
    "delegate": {"tags": ["role", "delegate:yes"]},
    "saver": {"tags": ["role", "saver:on"]},
    "class": {"tags": ["role", "class:proactive"]},
    "model": {"model": "test/approved"},
    "label": {"label": "Approved label"},
    "description": {"description": "Approved description"},
    "categories": {"categories": ["role", "approved-category"]},
    "combined": {
        "tags": ["role", "tools:read", "effort:hi", "delegate:yes", "saver:on", "class:proactive"],
        "model": "test/approved",
        "label": "Approved label",
        "description": "Approved description",
        "categories": ["role", "approved-category"],
        "instructions": "Approved policy guidance.",
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize("field", list(_POLICY_CASES))
async def test_signed_projected_policy_publishes_native_event_and_cold_history(root, field):
    """Exercise signed apply and its trusted consumer, not just the formatter."""
    from local_operator import profile_approval as P
    from local_operator.harness.types import ModelSpec, ToolExecutionEndEvent
    from local_operator.profile_scope import scope_agent_registry
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    data = proposal(root)
    assert data.before is not None
    data = P.build(
        root,
        profile_kind="agent",
        operation="update",
        target_id=data.target_id,
        name=data.name,
        session_id=data.session_id,
        tool_call_id=data.tool_call_id,
        before=data.before,
        after=dict(data.before, **_POLICY_CASES[field]),
    )
    record = P.file_request(root, data)
    assert P.snapshot(root, "agent", data.target_id) == data.before
    approve(root, record)
    result = P.apply(record["approval_id"], P.expected(record), root)
    assert result["state"] == "complete"
    assert P.snapshot(root, "agent", data.target_id) == data.after
    directory = root / "sessions" / data.session_id
    session = Session(
        model=ModelSpec(provider="test", model_id="m", context_window=100000),
        stream_fn=lambda *args: (_ for _ in ()).throw(AssertionError("NO model turn")),
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: [],
        agent_registry=scope_agent_registry(AgentRegistry(root, migrate=False), directory),
        owner_prompt_consent=True,
        yolo=True,
    )
    events = []
    session.subscribe(events.append)
    try:
        from types import SimpleNamespace

        from local_operator.session.runtime.process import _drain_inbox_into

        assert (
            await _drain_inbox_into(
                SimpleNamespace(_session=session, receive_peer_message=session.receive_peer_message)
            )
            == 1
        )
        native = [e for e in events if isinstance(e, ToolExecutionEndEvent)]
        assert len(native) == 1
        details = native[0].result.details
        expected = (
            {
                "tools",
                "effort",
                "delegate",
                "saver",
                "class",
                "model",
                "label",
                "description",
                "categories",
                "instructions",
            }
            if field == "combined"
            else {field}
        )
        assert details is not None
        assert set(details["fields"]) == expected
        assert details is not None
        assert all(
            any(line.startswith(name + ":") for line in details["diff"]) for name in expected
        )
        assert native[0].tool_call_id == data.tool_call_id
        history = Transcript(directory).build_llm_history()
        rows = [m for m in history if getattr(m, "tool_call_id", None) == data.tool_call_id]
        assert isinstance(rows[0], Message)
        assert rows[0].provider_payload is not None
        assert len(rows) == 1 and rows[0].provider_payload["details"] == details
        if field == "combined":
            import os
            import subprocess
            from pathlib import Path

            output = Path(os.environ.get("PROFILE_R4_EVIDENCE", str(root)))
            output.mkdir(parents=True, exist_ok=True)
            (output / "protected-policy-native-golden.json").write_text(
                json.dumps(
                    {
                        "meta": {
                            "source_sha": subprocess.check_output(
                                ["git", "rev-parse", "HEAD"], text=True
                            ).strip(),
                            "uncommitted_source": bool(
                                subprocess.check_output(
                                    ["git", "status", "--porcelain"], text=True
                                ).strip()
                            ),
                            "production_source_sha256": {
                                str(path): __import__("hashlib")
                                .sha256(path.read_bytes())
                                .hexdigest()
                                for path in [
                                    Path("local_operator/definition_transaction.py"),
                                    Path("local_operator/profile_approval.py"),
                                ]
                            },
                        },
                        "tool_execution_end_event": native[0].model_dump(mode="json"),
                        "cold_history_message": rows[0].model_dump(mode="json"),
                    },
                    indent=2,
                )
            )
    finally:
        await session.dispose()

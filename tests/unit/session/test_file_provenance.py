"""Future-live provenance controls using the owning Session and real journal.

Native providers are synthetic. File metadata, transcript/fsync and process
boundaries are real; these tests do not claim OS/account/presence proof.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.harness.types import (
    AgentTool,
    CustomMessage,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.network import identity
from local_operator.session import file_provenance as F
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript


class Stream:
    def __init__(self, turns):
        self.turns = iter(turns)

    def __call__(self, request, signal):
        events = next(
            self.turns, [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")]
        )

        async def generate():
            for event in events:
                yield event

        return generate()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    root = tmp_path / "config"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("HOME", str(tmp_path))
    owner = identity.load_or_mint(root)
    directory = root / "sessions" / "a123456789bc"
    transcript = Transcript(directory, defer_materialise=True)
    provenance = F.OwnerProvenance(transcript, config_root=root, cwd=work, home=tmp_path)
    result = SimpleNamespace(
        root=root,
        work=work,
        directory=directory,
        transcript=transcript,
        owner=provenance,
        public_key=owner.public_key,
    )
    try:
        yield result
    finally:
        transcript.close()


def ids(transcript):
    return [
        proof["id"]
        for entry in transcript.entries()
        if entry.payload.get("custom_type") == F.RECEIPT
        for proof in entry.payload["details"]["proofs"]
    ]


async def append_live(rig, text):
    message = Message.user(text)
    rig.owner.bind(message.id, rig.owner.capture("user", F.text_candidates(text)))
    await rig.transcript.append_messages([message], _file_captures=rig.owner.take([message]))
    return message


def make_session(rig, turns=(), tools=()):
    return Session(
        model=ModelSpec(provider="test", model_id="m", context_window=100000),
        stream_fn=Stream(turns),
        tools=list(tools),
        transcript=rig.transcript,
        cwd=str(rig.work),
        system_blocks_provider=lambda: ["stable"],
    )


@pytest.mark.asyncio
async def test_first_real_session_prompt_and_assistant_are_live_sources(rig):
    target = rig.work / "report.txt"
    target.write_text("real content not copied")
    assert not rig.directory.exists()
    session = make_session(
        rig, [[StreamTextDelta(delta=f"Saved {target}"), StreamEndEvent(stop_reason="stop")]]
    )
    try:
        await session.prompt(f"Inspect {target}", message_id="native-user", owner_typed=False)
        proof_ids = ids(rig.transcript)
        assert len(proof_ids) == 2
        proofs = [F.lookup_proof(rig.root, rig.directory.name, opaque) for opaque in proof_ids]
        assert all(proofs)
        assert {proof["source_class"] for proof in proofs if proof} == {"user", "assistant"}
        user = next(row for row in rig.transcript.entries() if row.id == "native-user")
        assert user.payload.get("provider_payload"), "remote locality marker must remain"
        assert "real content not copied" not in "".join(
            p.read_text() for p in (rig.directory / F.DIRECTORY).glob("*.json")
        )
        receipt = next(
            row for row in rig.transcript.entries() if row.payload.get("custom_type") == F.RECEIPT
        )
        assert str(target) not in receipt.to_json()
        print(
            "P2 FIRST PROMPT",
            json.dumps({"proof_ids": proof_ids, "states": [p["state"] for p in proofs if p]}),
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_capture_before_queue_does_not_restat_newly_appearing_file(rig):
    target = rig.work / "late.txt"
    session = make_session(rig)
    try:
        ticket = session._capture_file_input(str(target), "original-command")
        assert not ticket.files and ticket.unknown
        target.write_text("too late")
        await session.prompt(str(target), _file_capture=ticket)
        assert ids(rig.transcript) == []
        assert not (rig.directory / F.DIRECTORY).exists()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_cold_spool_or_injected_user_does_not_mint(rig):
    target = rig.work / "untrusted.txt"
    target.write_text("x")
    session = make_session(rig)
    try:
        await session.prompt(str(target), _file_capture=None, owner_typed=False)
        await session.prompt(str(target), harness_injected=True)
        assert ids(rig.transcript) == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_default_append_import_and_forged_role_do_not_mint(rig):
    target = rig.work / "import.txt"
    target.write_text("x")
    await rig.transcript.append_message(Message.user(str(target)))
    await rig.transcript.append_message(
        CustomMessage(
            custom_type="peer_message",
            details={"role": "user", "source": "user", "text": str(target)},
        )
    )
    assert ids(rig.transcript) == []
    assert not (rig.directory / F.DIRECTORY).exists()
    forged = {"source": "user", "files": [{"path": str(target)}]}
    message = Message.user(str(target))
    await rig.transcript.append_messages([message], _file_captures={message.id: [forged]})
    assert not (rig.directory / F.DIRECTORY).exists()


def test_ticket_and_admission_cannot_cross_json_or_pickle(rig):
    ticket = rig.owner.capture("user", [])
    with pytest.raises(TypeError):
        pickle.dumps(ticket)
    with pytest.raises(TypeError):
        F.CaptureTicket(object(), source="user", files=(), unknown=())
    admission = F.user_admission(
        SimpleNamespace(locality="remote", capabilities=frozenset({"prompt"}))
    )
    assert F.valid_admission(admission) and not F.valid_admission({"locality": "local"})
    with pytest.raises(TypeError):
        pickle.dumps(admission)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    ["symlink", "parent-link", "hardlink", "fifo", "directory", "traversal", "sensitive", "config"],
)
async def test_unsafe_original_names_are_unproven(rig, kind):
    target = rig.work / "candidate.txt"
    if kind == "fifo":
        os.mkfifo(target)
    elif kind == "directory":
        target.mkdir()
    elif kind == "config":
        target = rig.root / "hidden.txt"
        target.write_text("x")
    elif kind == "sensitive":
        target = rig.work / ".env"
        target.write_text("synthetic")
    else:
        target.write_text("x")
        if kind == "symlink":
            linked = rig.work / "link.txt"
            linked.symlink_to(target)
            target = linked
        elif kind == "hardlink":
            os.link(target, rig.work / "alias.txt")
        elif kind == "parent-link":
            real = rig.work / "real"
            real.mkdir()
            (real / "child.txt").write_text("x")
            linked = rig.work / "linked"
            linked.symlink_to(real, target_is_directory=True)
            target = linked / "child.txt"
        elif kind == "traversal":
            target = rig.work / "missing" / ".." / target.name
    ticket = rig.owner.capture("user", [F.Candidate(str(target), "literal")])
    assert not ticket.files
    assert ticket.unknown


@pytest.mark.asyncio
async def test_exact_retained_redaction_prevents_premask_binding(rig):
    target = rig.work / "secretly-removed.txt"
    target.write_text("x")
    message = Message.user("reference withheld")
    rig.owner.bind(message.id, rig.owner.capture("user", F.text_candidates(str(target))))
    await rig.transcript.append_messages([message], _file_captures=rig.owner.take([message]))
    assert ids(rig.transcript) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["rewrite", "replace", "hardlink"])
async def test_changed_target_never_refreshes_existing_proof(rig, mutation):
    target = rig.work / "version.txt"
    target.write_text("v1")
    await append_live(rig, str(target))
    opaque = ids(rig.transcript)[0]
    assert F.lookup_proof(rig.root, rig.directory.name, opaque)
    if mutation == "replace":
        target.unlink()
        target.write_text("v2")
    elif mutation == "rewrite":
        target.write_text("new version")
    else:
        os.link(target, rig.work / "other.txt")
    assert F.lookup_proof(rig.root, rig.directory.name, opaque) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["pending", "finish"])
async def test_proof_failure_preserves_ordinary_transcript(rig, monkeypatch, stage):
    target = rig.work / "durable.txt"
    target.write_text("x")
    original = F._private_write

    def fail(path, data):
        if data.get("state") == ("pending" if stage == "pending" else "committed"):
            raise OSError("synthetic-provenance-disk-failure")
        return original(path, data)

    monkeypatch.setattr(F, "_private_write", fail)
    message = await append_live(rig, str(target))
    assert rig.transcript.has_entry(message.id)
    assert message.id in rig.transcript.path.read_text()
    for opaque in ids(rig.transcript):
        assert F.lookup_proof(rig.root, rig.directory.name, opaque) is None


@pytest.mark.asyncio
async def test_actual_tool_completion_binds_new_result_id_and_originating_call(rig):
    target = rig.work / "new-output.txt"

    async def execute(call_id, args, signal, on_update, context):
        Path(args["path"]).write_text("newly produced")
        return ToolResult(
            tool_call_id=call_id,
            tool_name="write",
            content=[TextContent(text="done")],
            details={"path": args["path"]},
        )

    tool = AgentTool(
        name="write",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        execute=execute,
    )
    session = make_session(
        rig,
        [
            [
                StreamToolCallDelta(
                    index=0,
                    id="actual-call",
                    name="write",
                    argument_delta=json.dumps({"path": str(target)}),
                ),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ],
        [tool],
    )
    try:
        await session.prompt("create it")
        proofs = [F.lookup_proof(rig.root, rig.directory.name, p) for p in ids(rig.transcript)]
        completed = [p for p in proofs if p and p["source_class"] == "tool_completion"]
        assert completed
        result = next(
            row
            for row in rig.transcript.entries()
            if row.payload.get("tool_call_id") == "actual-call"
        )
        assert result.id != "actual-call"
        assert all(p["event_id"] == result.id and p["origin_id"] for p in completed)
        print(
            "P2 TOOL COMPLETION",
            json.dumps({"result_id": result.id, "proofs": [p["id"] for p in completed]}),
        )
    finally:
        await session.dispose()

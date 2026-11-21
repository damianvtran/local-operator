"""Cross-process point verification and adverse ordered-publication controls."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from local_operator.harness.types import Message
from local_operator.session import file_provenance as F
from tests.unit.session.test_file_provenance import append_live, ids
from tests.unit.session.test_file_provenance import rig as _provenance_rig

rig = _provenance_rig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "broken",
    [
        "proof-missing",
        "proof-pending",
        "proof-corrupt",
        "marker-missing",
        "marker-corrupt",
        "meta-incarnation",
        "source-changed",
        "receipt-changed",
        "store-link",
        "mode",
        "schema-bool",
    ],
)
async def test_corrupt_missing_or_redirected_witness_never_repairs(rig, broken):
    target = rig.work / "proof.txt"
    target.write_text("x")
    await append_live(rig, str(target))
    opaque = ids(rig.transcript)[0]
    proof_path = rig.directory / F.DIRECTORY / (opaque + ".json")
    proof = json.loads(proof_path.read_text())
    if broken == "proof-missing":
        proof_path.unlink()
    elif broken == "proof-pending":
        proof["state"] = "pending"
        proof_path.write_text(json.dumps(proof))
    elif broken == "proof-corrupt":
        proof_path.write_text("{")
    elif broken == "marker-missing":
        (rig.directory / F.SENTINEL).unlink()
    elif broken == "marker-corrupt":
        (rig.directory / F.SENTINEL).write_text("{")
    elif broken in {"meta-incarnation", "schema-bool"}:
        p = rig.directory / F.DIRECTORY / "meta.json"
        d = json.loads(p.read_text())
        d["incarnation" if broken == "meta-incarnation" else "schema"] = (
            "different" if broken == "meta-incarnation" else True
        )
        p.write_text(json.dumps(d))
    elif broken == "store-link":
        saved = rig.directory / "other-private"
        (rig.directory / F.DIRECTORY).rename(saved)
        (rig.directory / F.DIRECTORY).symlink_to(saved, target_is_directory=True)
    elif broken == "mode":
        proof_path.chmod(0o644)
    else:
        span = proof["source" if broken == "source-changed" else "receipt"]
        with rig.transcript.path.open("r+b") as f:
            f.seek(span["offset"])
            f.write(b"!")
    before = sorted(str(p.relative_to(rig.directory)) for p in rig.directory.rglob("*"))
    assert F.lookup_proof(rig.root, rig.directory.name, opaque) is None
    assert before == sorted(str(p.relative_to(rig.directory)) for p in rig.directory.rglob("*"))


@pytest.mark.asyncio
async def test_partial_initialization_never_repairs_from_next_live_event(rig, monkeypatch):
    target = rig.work / "partial.txt"
    target.write_text("x")
    write = F._private_write

    def fail_meta(path, data):
        if path.name == "meta.json":
            raise OSError("synthetic-first-init-failure")
        return write(path, data)

    monkeypatch.setattr(F, "_private_write", fail_meta)
    first = await append_live(rig, str(target))
    assert rig.transcript.has_entry(first.id)
    assert (rig.directory / F.SENTINEL).exists()
    monkeypatch.setattr(F, "_private_write", write)
    second = await append_live(rig, str(target))
    assert rig.transcript.has_entry(second.id)
    assert ids(rig.transcript) == []
    assert not (rig.directory / F.DIRECTORY / "meta.json").exists()


@pytest.mark.asyncio
async def test_unknown_writer_exclusivity_does_not_change_ordinary_commit(rig, monkeypatch):
    from local_operator.session import compression

    target = rig.work / "lock.txt"
    target.write_text("x")
    monkeypatch.setattr(compression, "acquire_writer", lambda *args, **kwargs: False)
    message = await append_live(rig, str(target))
    assert rig.transcript.has_entry(message.id)
    assert not ids(rig.transcript)


@pytest.mark.asyncio
async def test_source_and_receipt_share_one_write_and_cancellation_shields_finish(rig, monkeypatch):
    target = rig.work / "cancel.txt"
    target.write_text("x")
    original = rig.transcript._write_entries
    entered, release = threading.Event(), threading.Event()

    def write(rows, **kwargs):
        assert len(rows) == 2 and rows[-1].payload["custom_type"] == F.RECEIPT
        entered.set()
        release.wait()
        return original(rows, **kwargs)

    monkeypatch.setattr(rig.transcript, "_write_entries", write)
    task = asyncio.create_task(append_live(rig, str(target)))
    await asyncio.to_thread(entered.wait)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ids(rig.transcript)
    assert F.lookup_proof(rig.root, rig.directory.name, ids(rig.transcript)[0])


@pytest.mark.asyncio
async def test_private_metadata_does_not_fork_or_sync_and_old_rows_replay(rig):
    from local_operator.fork import EXCLUDED_SIDECARS, fork_session
    from local_operator.network.sync import EXCLUDED_ENTRIES
    from local_operator.session.transcript import Transcript

    target = rig.work / "fork.txt"
    target.write_text("x")
    await append_live(rig, str(target))
    for name in (F.DIRECTORY, F.SENTINEL):
        assert name in EXCLUDED_SIDECARS and name in EXCLUDED_ENTRIES
    destination = fork_session(rig.root, rig.directory.name)
    assert not (rig.root / "sessions" / destination / F.DIRECTORY).exists()
    assert not (rig.root / "sessions" / destination / F.SENTINEL).exists()
    assert F.lookup_proof(rig.root, destination, ids(rig.transcript)[0]) is None
    opened = Transcript(rig.directory, read_only=True)
    try:
        assert opened.build_llm_history()
    finally:
        opened.close()


@pytest.mark.asyncio
async def test_metadata_growth_and_inline_capture_are_measured_not_capped(rig):
    times = []
    for index in range(12):
        target = rig.work / f"growth-{index}.txt"
        target.write_text("x")
        start = time.perf_counter_ns()
        ticket = rig.owner.capture("user", F.text_candidates(str(target)))
        times.append(time.perf_counter_ns() - start)
        message = Message.user(str(target))
        rig.owner.bind(message.id, ticket)
        await rig.transcript.append_messages([message], _file_captures=rig.owner.take([message]))
    proof_ids = ids(rig.transcript)
    assert len(proof_ids) == 12
    sizes = [p.stat().st_size for p in (rig.directory / F.DIRECTORY).glob("*.json")]
    assert len(sizes) == 13
    print(
        "P2 METADATA MEASUREMENTS",
        json.dumps(
            {
                "n": len(times),
                "capture_ns": times,
                "private_bytes": sum(sizes),
                "record_count": len(sizes),
                "proof_count": len(proof_ids),
            }
        ),
    )


WRITER = """
import asyncio,json,os,sys
from pathlib import Path
from local_operator.harness.types import ModelSpec,StreamEndEvent,StreamTextDelta
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.session.file_provenance import RECEIPT
async def main():
 root,work,target=map(Path,sys.argv[1:4]); sid=sys.argv[4]
 def stream(request,signal):
  async def gen():
   yield StreamTextDelta(delta="done")
   yield StreamEndEvent(stop_reason="stop")
  return gen()
 t=Transcript(root/"sessions"/sid,defer_materialise=True)
 s=Session(model=ModelSpec(provider="test",model_id="m",context_window=100000),stream_fn=stream,tools=[],transcript=t,cwd=str(work),system_blocks_provider=lambda:["stable"])
 try:
  await s.prompt(str(target))
  proof_ids=[p["id"] for e in t.entries()
             if e.payload.get("custom_type")==RECEIPT
             for p in e.payload["details"]["proofs"]]
  print(json.dumps({"pid":os.getpid(),
                    "source":__import__("local_operator").__file__,
                    "ids":proof_ids}),flush=True)
 finally: await s.dispose()
asyncio.run(main())
"""
READER = """
import json,os,sys
from pathlib import Path
from local_operator.session.file_provenance import lookup_proof
p=lookup_proof(Path(sys.argv[1]),sys.argv[2],sys.argv[3])
print(json.dumps({"pid":os.getpid(),"state":p["state"] if p else "unknown",
                  "id":p["id"] if p else None}),flush=True)
"""


def child(code, args, rig):
    env = {
        "HOME": str(rig.work.parent),
        "LOCAL_OPERATOR_CONFIG_DIR": str(rig.root),
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", code, *map(str, args)],
        cwd=rig.work,
        env=env,
        start_new_session=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        out, err = process.communicate()
        assert process.returncode == 0, (out, err)
        result = json.loads(out.splitlines()[-1])
        assert result["pid"] == process.pid
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
        with pytest.raises(ProcessLookupError):
            os.killpg(process.pid, 0)
        return result
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()


def test_actual_writer_reader_restart_uses_disk_not_issuer_memory(rig):
    target = rig.work / "process.txt"
    target.write_text("file remains unchanged")
    written = child(WRITER, [rig.root, rig.work, target, rig.directory.name], rig)
    assert len(written["ids"]) == 1
    opaque = written["ids"][0]
    first = child(READER, [rig.root, rig.directory.name, opaque], rig)
    second = child(READER, [rig.root, rig.directory.name, opaque], rig)
    assert first["state"] == second["state"] == "committed"
    restarted = child(WRITER, [rig.root, rig.work, target, rig.directory.name], rig)
    assert opaque in restarted["ids"] and len(restarted["ids"]) == 2
    third = child(READER, [rig.root, rig.directory.name, opaque], rig)
    assert third["state"] == "committed"
    proof_path = rig.directory / F.DIRECTORY / (opaque + ".json")
    proof = json.loads(proof_path.read_text())
    proof["state"] = "pending"
    proof_path.write_text(json.dumps(proof))
    cold = child(READER, [rig.root, rig.directory.name, opaque], rig)
    assert cold["state"] == "unknown"
    print(
        "P2 SEPARATE W/R",
        json.dumps(
            {
                "W": written["pid"],
                "R": first["pid"],
                "R_restart": second["pid"],
                "W_restart": restarted["pid"],
                "R_after_W_restart": third["pid"],
                "cold_pending": cold,
            }
        ),
    )

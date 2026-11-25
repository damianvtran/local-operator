"""B1 real audit/proof/snapshot/seal/prepare, with synthetic P1 account/presence.

No sockets/native attestation or emission: the fixture reuses P1's explicit
shape-double boundary, real SQLite/signatures/durability and P2 live transcript.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.harness.types import Message
from local_operator.network import account_binding as B
from local_operator.network import files
from local_operator.network import mentioned_file_read as M
from local_operator.network import store
from local_operator.network.audit import AuditLog
from local_operator.network.types import MeshRefusal
from local_operator.session import file_provenance as P
from local_operator.session.transcript import Transcript
from tests.unit.network import test_account_binding as binding_tests

rig = binding_tests.rig


@pytest.fixture
def lane(rig, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    target = work / "report.txt"
    target.write_bytes(b"immutable prepared bytes from a future-live mentioned file")
    directory = rig.root / "sessions" / "a123456789bc"
    transcript = Transcript(directory, defer_materialise=True)
    owner = P.OwnerProvenance(transcript, config_root=rig.root, cwd=work, home=tmp_path)
    text = f"Please use {target}"
    message = Message.user(text)
    owner.bind(message.id, owner.capture("user", P.text_candidates(text)))
    asyncio.run(transcript.append_messages([message], _file_captures=owner.take([message])))
    proof_id = next(
        p["id"]
        for entry in transcript.entries()
        if entry.payload.get("custom_type") == P.RECEIPT
        for p in entry.payload["details"]["proofs"]
    )
    proof = P.lookup_proof(rig.root, directory.name, proof_id)
    assert proof is not None
    evidence = binding_tests._active(rig)
    rig.link.server.audit = AuditLog(rig.root)
    reader = M.OwnerMentionedFileReader(
        incoming_link=rig.link,
        auth=rig.auth,
        owner_account=B.ExactNativeAccountSelection(credential_id=rig.row.id),
        evidence=evidence,
        required_subject=B.CanonicalSubject(
            issuer=binding_tests.F.DEFAULT_RADIENT_API_BASE_URL, opaque_id="account-a"
        ),
    )
    read = binding_tests._read(
        rig,
        session_id=directory.name,
        source_row_id=proof["source"]["id"],
        provenance_id=proof_id,
        proof_digest="sha256:" + proof["proof_digest"],
        length=17,
        file_version="not-a-sealed-version",
    )
    value = SimpleNamespace(
        rig=rig,
        work=work,
        target=target,
        transcript=transcript,
        proof=proof,
        reader=reader,
        read=read,
        log=rig.link.server.audit,
    )
    try:
        yield value
    finally:
        transcript.close()


def rows(lane):
    path = lane.log._path
    return (
        [
            row
            for line in path.read_text().splitlines()
            if (row := json.loads(line))["event"] == "mentioned_file_read_admitted"
        ]
        if path.exists()
        else []
    )


def establish(lane, **changes):
    request = M.begin_establishment(lane.read.model_copy(update=changes))
    return asyncio.run(lane.reader.establish(request))


def read_for(version, **changes):
    return version.first_read.model_copy(
        update={
            "request_id": "synthetic-next-request",
            "transfer_nonce": "d" * 32,
            **changes,
        }
    )


def test_actual_new_rows_sealed_commitment_and_exact_immutable_preparation(lane):
    assert rows(lane) == [], "no P3 entry exists before this operation"
    version, prepared = establish(lane)
    actual_rows = rows(lane)
    assert len(actual_rows) == 2
    assert [r["detail"]["operation"] for r in actual_rows] == ["establish", "fetch"]
    assert actual_rows[0]["detail"]["read_binding"]["file_version"].startswith("establish:")
    assert actual_rows[1]["detail"]["read_binding"] == M._read_identity(prepared.read_binding)
    assert prepared.admission.record == lane.log._path.read_bytes().splitlines(keepends=True)[-1]
    assert prepared.payload == lane.target.read_bytes()[:17]
    assert prepared.span_digest == "sha256:" + hashlib.sha256(prepared.payload).hexdigest()
    assert prepared.whole_digest == "sha256:" + hashlib.sha256(lane.target.read_bytes()).hexdigest()
    seal = json.loads(version._seal_bytes)
    assert seal["establishment_record"].encode() == version.establishment.record
    assert json.loads(version.establishment.record) == actual_rows[0]
    assert seal["first_read"] == M._read_identity(prepared.read_binding)
    assert seal["nonce"] == version.nonce
    body = {k: v for k, v in seal.items() if k not in {"commitment", "first_read"}}
    assert version.commitment == "sealed:" + M._digest(body)
    assert (version._directory / "snapshot").read_bytes() == lane.target.read_bytes()
    assert not ((version._directory / "snapshot").stat().st_mode & 0o077)
    assert not (version._directory.stat().st_mode & 0o077)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(prepared, "payload", b"changed")
    with pytest.raises(TypeError):
        M.PreparedResponse()
    assert not hasattr(prepared, "send") and not hasattr(prepared, "to_dict")
    # Corrupting mutable storage after preparation cannot alter frozen result bytes.
    (version._directory / "snapshot").chmod(0o600)
    (version._directory / "snapshot").write_bytes(b"X" * version.size)
    assert prepared.payload == b"immutable prepare"


def test_seal_length_substitution_refuses_before_content_allocation(lane, monkeypatch):
    version, _ = establish(lane)
    seal = version._directory / "seal.json"
    seal.write_bytes(seal.read_bytes() + b"X")
    info = seal.stat()
    reads = []
    original = files._read_span

    def span(fd, offset, length):
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) == (info.st_dev, info.st_ino):
            reads.append((offset, length))
        return original(fd, offset, length)

    monkeypatch.setattr(files, "_read_span", span)
    with pytest.raises(MeshRefusal):
        asyncio.run(lane.reader.prepare(version, read_for(version)))
    assert reads == []


def test_each_phase_gets_own_exact_fresh_audit(lane):
    version, _ = establish(lane)
    for index, phase in enumerate(("stat", "fetch", "verify", "resume")):
        read = read_for(
            version,
            phase=phase,
            offset=0,
            length=0 if phase == "stat" else 9,
            request_id=f"synthetic-{phase}",
            transfer_nonce=f"{index + 1:032x}",
        )
        before = len(rows(lane))
        response = asyncio.run(lane.reader.prepare(version, read))
        current = rows(lane)
        assert len(current) == before + 1
        assert current[-1]["detail"]["operation"] == phase
        assert current[-1]["detail"]["read_binding"] == M._read_identity(read)
        assert response.payload == (
            lane.target.read_bytes()[:9] if phase in {"fetch", "resume"} else b""
        )
        assert response.admission.sequence == current[-1]["seq"]


def test_second_establishment_new_version_and_record_never_overwrites(lane):
    first, _ = establish(lane)
    old_seal = (first._directory / "seal.json").read_bytes()
    second, _ = establish(lane)
    assert first.digest == second.digest
    assert first.commitment != second.commitment and first.nonce != second.nonce
    assert first._directory != second._directory
    assert (first._directory / "seal.json").read_bytes() == old_seal
    assert len(rows(lane)) == 4
    assert rows(lane)[0]["detail"]["nonce"] != rows(lane)[2]["detail"]["nonce"]


@pytest.mark.parametrize(
    "failure", ["disabled", "degraded", "append", "file_fsync", "dir_fsync", "readback"]
)
def test_failed_audit_has_zero_target_reads_seals_or_prepared_responses(lane, monkeypatch, failure):
    reads = []
    original_span = files._read_span
    target_info = lane.target.stat()

    def span(fd, offset, length):
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) == (target_info.st_dev, target_info.st_ino):
            reads.append((offset, length))
        return original_span(fd, offset, length)

    monkeypatch.setattr(files, "_read_span", span)
    if failure == "disabled":
        lane.log._enabled = False
    elif failure == "degraded":
        lane.log.degraded = True
    elif failure == "append":
        monkeypatch.setattr(lane.log, "flush", lambda: None)
    elif failure in {"file_fsync", "dir_fsync"}:
        original_fsync = os.fsync

        def fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode) == (failure == "dir_fsync"):
                raise OSError("synthetic audit sync failure")
            return original_fsync(fd)

        # Install only at the audit call so P1/P2 setup/readbacks remain real.
        original_required = lane.log.record_mentioned_file_required

        def required(identity):
            with monkeypatch.context() as patch:
                patch.setattr(os, "fsync", fsync)
                return original_required(identity)

        monkeypatch.setattr(lane.log, "record_mentioned_file_required", required)
    else:
        original_record = lane.log.record

        def record(event):
            original_record(event)
            lines = lane.log._path.read_text().splitlines()
            row = json.loads(lines[-1])
            assert row["event"] == "mentioned_file_read_admitted"
            row["detail"]["binding"]["subject_id"] = "substituted-account"
            intended = lane.log._required_line
            lane.log._path.write_text("\n".join(lines[:-1]) + "\n" + json.dumps(row) + "\n")
            assert lane.log._required_line == intended
            assert "substituted-account" in lane.log._path.read_text()

        monkeypatch.setattr(lane.log, "record", record)
    with pytest.raises(MeshRefusal):
        establish(lane)
    assert reads == []
    assert not list(lane.transcript.directory.glob("mentioned-file-versions/*/seal.json"))


@pytest.mark.parametrize(
    "failure",
    ["snapshot", "snapshot_corrupt", "snapshot_fsync", "seal", "seal_readback", "seal_spacing"],
)
def test_pending_or_torn_establishment_is_never_a_version(lane, monkeypatch, failure):
    initial = M.begin_establishment(lane.read)
    if failure == "snapshot":
        monkeypatch.setattr(files, "_read_span", lambda *a: b"")
    elif failure == "snapshot_corrupt":
        real_write = os.write
        monkeypatch.setattr(os, "write", lambda fd, block: real_write(fd, b"Z" * len(block)))
    elif failure == "snapshot_fsync":
        real_sync = os.fsync

        def sync_snapshot(fd):
            if os.fstat(fd).st_mode & 0o777 == 0o400:
                raise OSError("synthetic snapshot fsync failure")
            return real_sync(fd)

        monkeypatch.setattr(os, "fsync", sync_snapshot)
    else:
        real = store._write_private_json_durable

        def write(path, payload):
            result = real(path, payload)
            if path.name == "seal.json":
                if failure == "seal":
                    raise OSError("synthetic post-publication indeterminate seal")
                path.write_bytes(
                    path.read_bytes() + b" " if failure == "seal_spacing" else b"{torn"
                )
            return result

        monkeypatch.setattr(store, "_write_private_json_durable", write)
    with pytest.raises(MeshRefusal):
        asyncio.run(lane.reader.establish(initial))
    assert [r["detail"]["operation"] for r in rows(lane)] == ["establish"]
    with pytest.raises(MeshRefusal):
        asyncio.run(lane.reader.prepare(initial, initial.read_binding))
    with pytest.raises(TypeError):
        M.SealedVersion()


@pytest.mark.parametrize(
    "corruption", ["snapshot", "seal_truncated", "seal_mutated", "seal_replaced", "seal_spacing"]
)
def test_corrupt_snapshot_or_exact_seal_never_prepares(lane, corruption):
    version, _ = establish(lane)
    path = version._directory / ("snapshot" if corruption == "snapshot" else "seal.json")
    raw = path.read_bytes()
    if corruption == "snapshot":
        path.chmod(0o600)
        path.write_bytes(b"Z" * len(raw))
    elif corruption == "seal_truncated":
        path.write_bytes(raw[:-1])
    elif corruption == "seal_spacing":
        path.write_bytes(raw + b" ")
    elif corruption == "seal_mutated":
        record = json.loads(raw)
        record["first_read"]["offset"] = 1
        path.write_text(json.dumps(record))
    else:
        replacement = version._directory / "substitute"
        replacement.write_bytes(raw)
        os.replace(replacement, path)
    with pytest.raises(MeshRefusal):
        asyncio.run(lane.reader.prepare(version, read_for(version)))


def test_intersecting_chunks_verified_before_slicing_and_no_mutable_reread(lane, monkeypatch):
    # Reissue the mention after changing the source: old P2 metadata must not be
    # weakened just to make a large-file fixture pass.
    payload = b"a" * files.FILE_CHUNK_BYTES + b"second-chunk-content"
    lane.target.write_bytes(payload)
    text = f"Please use {lane.target}"
    message = Message.user(text)
    owner = P.OwnerProvenance(
        lane.transcript, config_root=lane.rig.root, cwd=lane.work, home=lane.work.parent
    )
    owner.bind(message.id, owner.capture("user", P.text_candidates(text)))
    asyncio.run(lane.transcript.append_messages([message], _file_captures=owner.take([message])))
    proof_id = next(
        p["id"]
        for e in reversed(list(lane.transcript.entries()))
        if e.payload.get("custom_type") == P.RECEIPT
        for p in e.payload["details"]["proofs"]
    )
    proof = P.lookup_proof(lane.rig.root, lane.transcript.directory.name, proof_id)
    assert proof is not None
    lane.read = lane.read.model_copy(
        update={
            "source_row_id": proof["source"]["id"],
            "provenance_id": proof_id,
            "proof_digest": "sha256:" + proof["proof_digest"],
        }
    )
    version, _ = establish(lane)
    snapshot = version._directory / "snapshot"
    opened = snapshot.stat()
    real = files._read_span
    calls = []

    def read(fd, offset, length):
        block = real(fd, offset, length)
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == (opened.st_dev, opened.st_ino):
            calls.append((offset, length))
        return block

    monkeypatch.setattr(files, "_read_span", read)
    request = read_for(version, offset=files.FILE_CHUNK_BYTES - 3, length=9)
    result = asyncio.run(lane.reader.prepare(version, request))
    assert result.payload == b"aaasecond"
    assert calls == [
        (0, files.FILE_CHUNK_BYTES),
        (files.FILE_CHUNK_BYTES, len(b"second-chunk-content")),
    ]
    # Corrupt bytes outside the requested slice but IN its committed chunk:
    # hashing only the slice would miss this, full chunk verification refuses.
    snapshot.chmod(0o600)
    with snapshot.open("r+b") as handle:
        handle.seek(1)
        handle.write(b"Z")
    with pytest.raises(MeshRefusal):
        asyncio.run(lane.reader.prepare(version, request))


@pytest.mark.parametrize("mutation", ["replace", "rewrite"])
def test_same_size_mtime_mutation_is_early_p2_identity_refusal(lane, mutation):
    version, _ = establish(lane)
    info = lane.target.stat()
    if mutation == "replace":
        replacement = lane.work / "replacement"
        replacement.write_bytes(b"B" * info.st_size)
        os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
        os.replace(replacement, lane.target)
    else:
        lane.target.write_bytes(b"B" * info.st_size)
        os.utime(lane.target, ns=(info.st_atime_ns, info.st_mtime_ns))
        after = lane.target.stat()
        assert (after.st_dev, after.st_ino) == (info.st_dev, info.st_ino)
        assert after.st_ctime_ns != info.st_ctime_ns, "same-inode control changed ctime"
    assert lane.target.stat().st_size == info.st_size
    assert lane.target.stat().st_mtime_ns == info.st_mtime_ns
    before = len(rows(lane))
    assert P.lookup_proof(lane.rig.root, lane.read.session_id, lane.read.provenance_id) is None
    with pytest.raises(MeshRefusal):
        asyncio.run(lane.reader.prepare(version, read_for(version)))
    assert len(rows(lane)) == before, "P2 rejects replacement before new admission"


@pytest.mark.parametrize(
    "change",
    [
        {"source_row_id": "different-row"},
        {"provenance_id": "f" * 32},
        {"proof_digest": "sha256:" + "f" * 64},
        {"session_id": "different-session"},
        {"file_version": "establish:" + "f" * 32},
        {"link_id": "different-link"},
        {"transcript_hash": "f" * 64},
        {"offset": 999},
        {"length": 999},
        {"phase": "stat"},
    ],
)
def test_read_context_substitution_never_reuses_version_admission(lane, change):
    version, _ = establish(lane)
    before = len(rows(lane))
    with pytest.raises(MeshRefusal):
        asyncio.run(lane.reader.prepare(version, read_for(version, **change)))
    assert len(rows(lane)) == before


def test_fresh_verification_required_and_revoked_generation_denies(lane):
    version, _ = establish(lane)
    before = len(rows(lane))
    # Real signed consent revoke; synthetic fixture only, no live device change.
    revocation = binding_tests.approval_tests.request(lane.rig, action="revoke")
    binding_tests.approval_tests.approve(lane.rig, revocation)
    binding_tests.F.revoke(
        revocation["approval_id"],
        binding_tests.F.expected(revocation),
        lane.rig.root,
    )
    with pytest.raises(MeshRefusal):
        asyncio.run(lane.reader.prepare(version, read_for(version)))
    assert len(rows(lane)) == before


def test_reconnect_requires_new_live_binding_and_own_admission(lane):
    version, _ = establish(lane)
    old = lane.rig.link
    new = object.__new__(type(old))
    new.__dict__.update(old.__dict__)
    new.link_id = "synthetic-reconnected-link"
    new.result = dataclasses.replace(
        old.result,
        link_id=new.link_id,
        transcript_hash=hashlib.sha256(b"synthetic-reconnected-handshake").hexdigest(),
    )
    new.server.links = {new.link_id: new}
    reader = M.OwnerMentionedFileReader(
        incoming_link=new,
        auth=lane.rig.auth,
        owner_account=lane.reader.owner_account,
        evidence=lane.reader.evidence,
        required_subject=lane.reader.subject,
    )
    before = len(rows(lane))
    with pytest.raises(MeshRefusal):
        asyncio.run(reader.prepare(version, read_for(version)))
    assert len(rows(lane)) == before
    read = read_for(version, link_id=new.link_id, transcript_hash=new.result.transcript_hash)
    prepared = asyncio.run(reader.prepare(version, read))
    assert prepared.payload == lane.target.read_bytes()[:17]
    assert len(rows(lane)) == before + 1
    assert rows(lane)[-1]["detail"]["read_binding"] == M._read_identity(read)
    with pytest.raises(MeshRefusal):
        asyncio.run(lane.reader.prepare(version, read_for(version)))


@pytest.mark.parametrize("invalidated", ["source_deleted", "proof_deleted", "sensitive"])
def test_snapshot_never_preserves_current_authority(lane, monkeypatch, invalidated):
    version, _ = establish(lane)
    before = len(rows(lane))
    if invalidated == "source_deleted":
        lane.target.unlink()
    elif invalidated == "proof_deleted":
        (lane.transcript.directory / P.DIRECTORY / (lane.read.provenance_id + ".json")).unlink()
    else:
        from local_operator.supplements import denylist

        monkeypatch.setattr(denylist, "is_sensitive_for_supplements", lambda path: True)
    with pytest.raises(MeshRefusal):
        asyncio.run(lane.reader.prepare(version, read_for(version)))
    assert len(rows(lane)) == before


@pytest.mark.parametrize("changed", ["account", "source"])
def test_final_live_refresh_and_after_await_source_recheck(lane, monkeypatch, changed):
    original = lane.reader._verified
    calls = []

    async def verified(read):
        calls.append(read.file_version)
        if len(calls) == 2 and changed == "account":
            lane.reader.subject = B.CanonicalSubject(
                issuer=lane.reader.subject.issuer, opaque_id="different-account"
            )
        result = await original(read)
        if len(calls) == 2 and changed == "source":
            lane.target.write_bytes(b"B" * lane.target.stat().st_size)
        return result

    monkeypatch.setattr(lane.reader, "_verified", verified)
    with pytest.raises(MeshRefusal):
        establish(lane)
    assert len(calls) == 2
    assert [row["detail"]["operation"] for row in rows(lane)] == ["establish"]
    assert not list(lane.transcript.directory.glob("mentioned-file-versions/*/seal.json"))


@pytest.mark.parametrize("operation", ["establish", "prepare"])
def test_source_change_during_audit_refuses_before_protected_content(lane, monkeypatch, operation):
    version = establish(lane)[0] if operation == "prepare" else None
    observed = lane.target if version is None else version._directory / "snapshot"
    info = observed.stat()
    content_reads = []
    original_span = files._read_span
    original_required = lane.log.record_mentioned_file_required

    def span(fd, offset, length):
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) == (info.st_dev, info.st_ino):
            content_reads.append((offset, length))
        return original_span(fd, offset, length)

    def required(identity):
        receipt = original_required(identity)
        assert receipt is not None, "control must first commit the actual admission"
        lane.target.write_bytes(b"B" * lane.target.stat().st_size)
        return receipt

    monkeypatch.setattr(files, "_read_span", span)
    monkeypatch.setattr(lane.log, "record_mentioned_file_required", required)
    with pytest.raises(MeshRefusal):
        if version is None:
            establish(lane)
        else:
            asyncio.run(lane.reader.prepare(version, read_for(version)))
    assert content_reads == []


def test_parent_race_after_safe_open_closes_untransferred_descriptor(lane, monkeypatch):
    opened = []
    real_open = files._open_source
    real_identity = P._directory_identity

    def open_source(source, session_id, owner):
        result = real_open(source, session_id, owner)
        if source.path == lane.target:
            opened.append(result[0])
        return result

    def directory_identity(path):
        if path == lane.target.parent and len(opened) == 2:
            raise OSError("synthetic parent metadata race")
        return real_identity(path)

    monkeypatch.setattr(files, "_open_source", open_source)
    monkeypatch.setattr(P, "_directory_identity", directory_identity)
    with pytest.raises(MeshRefusal):
        establish(lane)
    assert len(opened) == 2  # P2's metadata capture, then our exact reopen.
    with pytest.raises(OSError):
        os.fstat(opened[-1])
    assert rows(lane) == []


#: The serving module is the primitive's one sanctioned caller, so reaching IT
#: is reaching the primitive: the scan follows that edge too.
_MODULES = ("mentioned_file_read", "mentioned_file_serve")


def _reaching_nodes(tree):
    names = {
        *_MODULES,
        "OwnerMentionedFileReader",
        "begin_establishment",
        "SealedVersion",
        "MentionedFileServing",
    }
    literals = {*_MODULES, *("local_operator.network." + m for m in _MODULES)}
    # Imports, aliases, direct symbol use and literal dynamic imports. This
    # guards accidental serving reachability, not arbitrary hostile Python.
    return [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Import)
            and any(
                alias.name in _MODULES or alias.name.endswith(tuple("." + m for m in _MODULES))
                for alias in node.names
            )
        )
        or (
            isinstance(node, ast.ImportFrom)
            and (
                (node.module or "").endswith(tuple("." + m for m in _MODULES))
                or any(alias.name in names for alias in node.names)
            )
        )
        or (isinstance(node, ast.Name) and node.id in names)
        or (isinstance(node, ast.Attribute) and node.attr in names)
        or (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in literals
        )
    ]


def _reaches_primitive(source):
    return bool(_reaching_nodes(ast.parse(source)))


def _gated_install_edge(tree):
    """The nodes of files.install's ONE sanctioned edge, or None if it is not exact.

    Exactly: inside ``def install``, an ``if _mentioned_file_serving_enabled(...)``
    whose body is the lazy ``from local_operator.network import
    mentioned_file_serve`` plus ``mentioned_file_serve.install(server)``.
    """
    (install,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "install"]
    gates = [
        n
        for n in install.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Call)
        and isinstance(n.test.func, ast.Name)
        and n.test.func.id == "_mentioned_file_serving_enabled"
        and not n.orelse
    ]
    if len(gates) != 1:
        return None
    body = gates[0].body
    if (
        len(body) != 2
        or not isinstance(body[0], ast.ImportFrom)
        or body[0].module != "local_operator.network"
        or [(a.name, a.asname) for a in body[0].names] != [("mentioned_file_serve", None)]
        or ast.unparse(body[1]) != "mentioned_file_serve.install(server)"
    ):
        return None
    return {id(node) for statement in body for node in ast.walk(statement)}


def _production_offenders(sources):
    """Product files reaching the primitive, minus EXACTLY the two sanctioned edges.

    WHY THE EXCEPTION IS EXACT AND MUST STAY SO. Serving is reachable only
    behind the default-off ``mentioned_file_serving`` gate: ``files.install``'s
    gated lazy import and the thin serving module itself. Anything wider (a
    module-level import in files.py, a second caller, a relay/CLI/tool/route
    reference) would make the primitive loadable with the gate off, or give it
    a second entry nobody reviewed. Widening this needs its own review.
    """
    offenders = []
    for name, source in sources.items():
        if name in {"network/mentioned_file_read.py", "network/mentioned_file_serve.py"}:
            continue
        tree = ast.parse(source)
        found = _reaching_nodes(tree)
        if name == "network/files.py":
            allowed = _gated_install_edge(tree)
            if allowed is None:
                offenders.append(name + " (gated edge is not exact)")
                continue
            found = [node for node in found if id(node) not in allowed]
        if found:
            offenders.append(name)
    return offenders


def _product_sources():
    product = Path(__file__).resolve().parents[3] / "local_operator"
    return {
        path.relative_to(product).as_posix(): path.read_text(encoding="utf-8")
        for path in product.rglob("*.py")
    }


@pytest.mark.parametrize(
    "caller",
    [
        "from local_operator.network import mentioned_file_read as reader",
        "import local_operator.network.mentioned_file_read as reader",
        "from local_operator.network.mentioned_file_read import OwnerMentionedFileReader as Reader",
        "importlib.import_module('local_operator.network.mentioned_file_read')",
        "context.reader = OwnerMentionedFileReader(incoming_link=link)",
    ],
)
def test_dormancy_guard_detects_a_new_production_caller(caller):
    assert _reaches_primitive(caller), "negative control must detect an exposure"
    assert not _reaches_primitive("event = 'mentioned_file_read_admitted'")


def test_no_production_registration_or_import_reaches_dormant_primitive():
    root = Path(__file__).resolve().parents[3]
    product = root / "local_operator"
    # Explicit serving/dispatch inventories plus every product Python caller.
    surfaces = (
        "network/files.py",
        "network/relay.py",
        "server/routes",
        "cli.py",
        "tools",
        "session_files",
        "remote_files",
    )
    assert all((product / name).exists() for name in surfaces)
    sources = _product_sources()
    # The sanctioned edge exists and is what files.py reaches through.
    assert _reaches_primitive(sources["network/files.py"])
    offenders = _production_offenders(sources)
    assert offenders == [], f"B1 must stay gated; production references: {offenders}"
    assert "mentioned_file_read" not in (root / "pyproject.toml").read_text()
    assert "mentioned_file_serve" not in (root / "pyproject.toml").read_text()


@pytest.mark.parametrize(
    "name,extra",
    [
        ("network/relay.py", "\nfrom local_operator.network import mentioned_file_serve\n"),
        ("cli.py", "\nimport local_operator.network.mentioned_file_read as reader\n"),
        ("network/files.py", "\nfrom local_operator.network import mentioned_file_serve\n"),
        ("network/files.py", "\ndef other(server):\n    mentioned_file_serve.install(server)\n"),
        ("network/new_surface.py", "x = MentionedFileServing(server)\n"),
    ],
)
def test_a_third_production_reference_still_trips_the_scan(name, extra):
    sources = _product_sources()
    sources[name] = sources.get(name, "") + extra
    assert _production_offenders(sources), f"{name} gained an edge the scan missed"


def test_files_install_edge_loosened_from_its_gate_trips_the_scan():
    sources = _product_sources()
    gate = "    if _mentioned_file_serving_enabled(server):\n"
    assert sources["network/files.py"].count(gate) == 1
    sources["network/files.py"] = sources["network/files.py"].replace(gate, "    if True:\n")
    assert _production_offenders(sources) == ["network/files.py (gated edge is not exact)"]

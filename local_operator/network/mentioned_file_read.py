"""Dormant owner-only audited mentioned-file preparation (P3/B1).

There is NO handler, protocol, CLI/tool registration, wire projection or emission
method here. B2 must fence actual mutators and the socket writer before C can
expose this through Files. A prepared response is immutable data, not permission
to enqueue it through the existing dict writer. Physical/native identity and
hostile-owner anti-rollback are not claims of this Python owner-process seam.

Authority remains P1's purpose-only live account/device binding plus P2's exact
future-live original transcript proof. Snapshots never preserve authority after
source/proof deletion, replacement or policy changes. Each operation checks both
again; snapshots commit bytes, not a continuing grant. State is session-owned
and follows ordinary session retention; pending state has no recovery-to-positive
API, no history scan, no new TTL, size policy or background sweeper.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn, SupportsIndex

from local_operator.network import account_binding as B
from local_operator.network import file_device_approval as F
from local_operator.network import files, store, sync
from local_operator.network.audit import (
    AuditLog,
    MentionedFileAdmissionReceipt,
    MentionedFileAuditIdentity,
)
from local_operator.session import file_provenance as P

_DIRECTORY = "mentioned-file-versions"


def _refuse() -> NoReturn:
    # No raw owner path (or underlying OSError text) escapes this seam.
    raise files._refuse("mentioned_file_unavailable", "the owner cannot prepare this file")


def _directory_identity(path: Path) -> list[int]:
    try:
        return P._directory_identity(path)
    except (OSError, sync.SyncRefused):
        _refuse()


def _json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode()).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _binding(verified: B.VerifiedAccountDeviceBinding) -> dict[str, Any]:
    """Public identity only; full public keys and native rows are committed, not logged raw."""
    return {
        "subject_issuer": verified.subject.issuer,
        "subject_id": verified.subject.opaque_id,
        "owner_key_digest": _digest(verified.owner_key.model_dump(mode="json")),
        "requester_key_digest": _digest(verified.requester_key.model_dump(mode="json")),
        "native_account_digest": _digest(verified.native_account.model_dump(mode="json")),
        "anchor_digest": _digest(verified.anchor.model_dump(mode="json")),
        "owner_admission_digest": _digest(verified.owner_admission),
        "requester_admission_digest": _digest(verified.requester_admission),
        "association": verified.association,
        "authority_epoch": verified.authority_epoch,
        "generation": verified.generation,
        "journal_revision": verified.journal_revision,
        "deny_generation": verified.deny_generation,
        "evidence_digest": verified.evidence_digest,
        "relay_instance": verified.relay_instance,
    }


def _read_identity(read: B.ReadBinding) -> dict[str, Any]:
    value = read.model_dump(mode="json")
    value["owner_key_digest"] = _digest(value.pop("owner_key"))
    return value


def _stamp(info: os.stat_result) -> tuple[int, ...]:
    # P2's exact six fields, including ctime, plus descriptor mode. Do not reduce
    # this to size/mtime: a same-stamp rewrite commonly changes ctime first.
    return (*P._identity(info), info.st_mode)


@dataclass(frozen=True, slots=True, init=False)
class Establishment:
    """Typed initial operation, deliberately NOT a sealed version."""

    nonce: str
    read_binding: B.ReadBinding

    def __new__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("use begin_establishment in the owner process")


def begin_establishment(read: B.ReadBinding) -> Establishment:
    """Assign an owner nonce; caller's opaque version is never a commitment."""
    if type(read) is not B.ReadBinding:
        _refuse()
    nonce = secrets.token_hex(16)
    read = B.ReadBinding.model_validate({**read.model_dump(), "file_version": "establish:" + nonce})
    value = object.__new__(Establishment)
    object.__setattr__(value, "nonce", nonce)
    object.__setattr__(value, "read_binding", read)
    return value


@dataclass(frozen=True, slots=True, init=False)
class SealedVersion:
    """Owner-local committed version; only successful establishment constructs one.

    Exact sidecar bytes and descriptor identity are retained, so a torn, edited
    or substituted seal cannot be interpreted as another valid manifest. There
    is no dict deserializer or pending-state recovery path.
    """

    commitment: str
    nonce: str
    size: int
    digest: str
    chunks: tuple[tuple[int, str], ...]
    establishment: MentionedFileAdmissionReceipt
    first_read: B.ReadBinding
    _binding_json: str = field(repr=False)
    _directory: Path = field(repr=False)
    _seal_bytes: bytes = field(repr=False)
    _seal_stamp: tuple[int, ...] = field(repr=False)
    _directory_identity: tuple[int, ...] = field(repr=False)
    _snapshot_identity: tuple[int, int] = field(repr=False)

    def __new__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("sealed versions are constructed only by establishment")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("sealed versions cannot leave the owner process")


@dataclass(frozen=True, slots=True, init=False)
class PreparedResponse:
    """Exact immutable bytes/metadata/read identity. Not a transport capability.

    B2 must consume this behind its actual writer/revoke fence, not through a
    new public dict-to-wire helper. Even stat/verify metadata is read-bound.
    """

    read_binding: B.ReadBinding
    verified_binding: B.VerifiedAccountDeviceBinding
    admission: MentionedFileAdmissionReceipt
    version: str
    size: int
    whole_digest: str
    span_digest: str
    payload: bytes

    def __new__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("prepared responses are constructed only after verification")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("prepared responses cannot leave the owner process")


class OwnerMentionedFileReader:
    """Private owner primitive, deliberately unregistered with every serving surface.

    Always call the real P1 adapter, never accept a truthy audit callback, cached
    subject, viewer path or caller-made verified context. The server's existing
    AuditLog is the sole writer. This is preparation, NOT the final release fence.
    """

    def __init__(
        self,
        *,
        incoming_link: Any,
        auth: Any,
        owner_account: B.ExactNativeAccountSelection,
        evidence: B.OwnerFileEvidenceReference,
        required_subject: B.CanonicalSubject,
    ):
        self.link = incoming_link
        self.root = F._link_root(incoming_link)
        self.root_identity = tuple(_directory_identity(self.root))
        self.auth = auth
        self.owner_account = owner_account
        self.evidence = evidence
        self.subject = required_subject
        self.audit = incoming_link.server.audit
        if type(self.audit) is not AuditLog or self.audit._path != store.audit_path(self.root):
            _refuse()

    async def _verified(self, read: B.ReadBinding) -> B.VerifiedAccountDeviceBinding:
        verified = await B.verify_live(
            incoming_link=self.link,
            auth=self.auth,
            owner_account=self.owner_account,
            evidence=self.evidence,
            required_subject=self.subject,
            required_purpose=F.PURPOSE,
            read_binding=read,
        )
        verified.require_read(self.link, read)
        if (
            F._link_root(self.link) != self.root
            or tuple(_directory_identity(self.root)) != self.root_identity
        ):
            _refuse()
        return verified

    def _current(
        self,
        read: B.ReadBinding,
        verified: B.VerifiedAccountDeviceBinding,
    ) -> tuple[int, os.stat_result]:
        """Point-read P2 plus current S1 descriptor checks; reads NO target content."""
        verified.require_read(self.link, read)
        proof = P.lookup_proof(self.root, read.session_id, read.provenance_id)
        # P2 persists bare SHA-256 hex; ReadBinding deliberately spells the
        # algorithm. Normalize only that representation, never the proof value.
        if (
            proof is None
            or "sha256:" + proof["proof_digest"] != read.proof_digest
            or proof["source"]["id"] != read.source_row_id
            or proof["owner"] != verified.owner_key.public_key
        ):
            _refuse()
        frozen = proof["file"]
        # lookup_proof rechecks relocated config/denylist, parent and file identity.
        # Reopen using the SAME S1 scope/regular/no-follow/single-link rules, not
        # _resolve_source (which remains PAD/digest-only, with no path expansion).
        path = Path(frozen["path"])
        source = files._Source(
            path,
            files._read_scope(Path(path.anchor), path.parent, "mentioned file"),
            path.name,
            "application/octet-stream",
            "",
            "mentioned file",
        )
        fd, info = files._open_source(source, read.session_id, "owner")
        try:
            if (
                list(P._identity(info)) != frozen["identity"]
                or _directory_identity(path.parent) != frozen["root_identity"]
            ):
                _refuse()
            return fd, info
        except BaseException:
            # A parent disappearance/refusal after open must not strand a file
            # descriptor. Ownership transfers to the caller only on return.
            os.close(fd)
            raise

    def _admit(
        self,
        operation: str,
        verified: B.VerifiedAccountDeviceBinding,
        read: B.ReadBinding,
        stamp: tuple[int, ...],
        nonce: str,
    ) -> MentionedFileAdmissionReceipt:
        verified.require_read(self.link, read)
        receipt = self.audit.record_mentioned_file_required(
            MentionedFileAuditIdentity(
                operation,
                _json(_binding(verified)),
                _json(_read_identity(read)),
                stamp,
                nonce,
            )
        )
        if type(receipt) is not MentionedFileAdmissionReceipt:
            _refuse()
        return receipt

    def _recheck(
        self,
        read: B.ReadBinding,
        verified: B.VerifiedAccountDeviceBinding,
        expected: tuple[int, ...],
    ) -> None:
        fd, info = self._current(read, verified)
        try:
            if _stamp(info) != expected:
                _refuse()
        finally:
            os.close(fd)

    async def establish(self, initial: Establishment) -> tuple[SealedVersion, PreparedResponse]:
        """Audit BEFORE the first source read, then snapshot, seal and prepare.

        A new random directory and nonce distinguish every establishment even
        for identical content/subject/source. Never overwrite an older version.
        Failed state is left session-owned/pending; only a fully read-back seal
        can yield a SealedVersion object, and no method loads pending objects.
        """
        if (
            type(initial) is not Establishment
            or initial.read_binding.file_version != "establish:" + initial.nonce
        ):
            _refuse()
        read = initial.read_binding
        verified = await self._verified(read)
        fd, info = self._current(read, verified)
        before = _stamp(info)
        try:
            admission = self._admit("establish", verified, read, before, initial.nonce)
            # Durability can block. Its receipt is not a lease on a source that
            # disappeared/changed while we were syncing the trail.
            self._recheck(read, verified, before)
            directory = self.root / "sessions" / read.session_id / _DIRECTORY / initial.nonce
            store._ensure_private_directory_durable(directory.parent)
            directory.mkdir(mode=0o700)  # exclusive creation, never reuse an old nonce
            store._sync_private_directory(directory.parent)
            store._sync_private_directory(directory)
            directory_identity = tuple(_directory_identity(directory))
            snapshot = directory / "snapshot"
            output = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            chunks: list[tuple[int, str]] = []
            whole = hashlib.sha256()
            position = 0
            try:
                while position < info.st_size:
                    block = files._read_span(
                        fd, position, min(files.FILE_CHUNK_BYTES, info.st_size - position)
                    )
                    if not block:
                        _refuse()
                    whole.update(block)
                    chunks.append((len(block), _bytes_digest(block)))
                    pending = memoryview(block)
                    while pending:
                        written = os.write(output, pending)
                        if written <= 0:
                            _refuse()
                        pending = pending[written:]
                    position += len(block)
                if _stamp(os.fstat(fd)) != before:
                    _refuse()
                os.fchmod(output, 0o400)
                os.fsync(output)
                snapshot_info = os.fstat(output)
            finally:
                os.close(output)
            store._sync_private_directory(directory)
            # fsync acknowledges an inode, not the manifest's exact bytes. Read
            # the newly persisted snapshot back against every captured chunk
            # before a version exists; no failed/torn capture becomes positive.
            scope = files._read_scope(Path(snapshot.anchor), directory, "mentioned file snapshot")
            captured = sync.open_for_read(snapshot, scope, nonblock=True)
            try:
                captured_info = os.fstat(captured)
                if (
                    captured_info.st_nlink != 1
                    or captured_info.st_mode & 0o077
                    or captured_info.st_size != position
                    or (captured_info.st_dev, captured_info.st_ino)
                    != (snapshot_info.st_dev, snapshot_info.st_ino)
                ):
                    _refuse()
                verified_whole = hashlib.sha256()
                at = 0
                for length, digest in chunks:
                    persisted = files._read_span(captured, at, length)
                    if len(persisted) != length or _bytes_digest(persisted) != digest:
                        _refuse()
                    verified_whole.update(persisted)
                    at += length
                if verified_whole.digest() != whole.digest() or _stamp(
                    os.fstat(captured)
                ) != _stamp(captured_info):
                    _refuse()
            finally:
                os.close(captured)
        except (OSError, sync.SyncRefused):
            _refuse()
        finally:
            os.close(fd)
        self._recheck(read, verified, before)
        body = {
            "schema": "lop.mentioned-file-seal.v1",
            "state": "committed",
            "nonce": initial.nonce,
            "binding": _binding(verified),
            "initial_read": _read_identity(read),
            "source_stamp": list(before),
            "establishment_record": admission.record.decode("utf-8"),
            "establishment_inode": [admission.device, admission.inode],
            "establishment_generation": admission.generation,
            "chunks": chunks,
            "size": position,
            "whole_digest": "sha256:" + whole.hexdigest(),
            "snapshot_identity": [snapshot_info.st_dev, snapshot_info.st_ino],
        }
        commitment = "sealed:" + _digest(body)
        final_read = B.ReadBinding.model_validate({**read.model_dump(), "file_version": commitment})
        # The new exact final identity requires a fresh live P1 observation.
        # Revalidate source/proof AFTER this await; establishment is no grant.
        final_verified = await self._verified(final_read)
        if _binding(final_verified) != _binding(verified):
            _refuse()
        self._recheck(final_read, final_verified, before)
        seal = {**body, "commitment": commitment, "first_read": _read_identity(final_read)}
        try:
            seal_path = directory / "seal.json"
            # Match the existing sole private-JSON writer's actual encoding,
            # not merely parsed equivalence. Even whitespace substitution in
            # this publication window is not the exact seal we committed.
            expected_seal = json.dumps(seal, ensure_ascii=False).encode("utf-8")
            store._write_private_json_durable(seal_path, seal)
            seal_bytes, seal_stamp = self._private_bytes(seal_path, expected_seal)
            if seal_bytes != expected_seal:
                _refuse()
            # Retain exact persisted encoding too, not only parsed equivalence.
            if tuple(_directory_identity(directory)) != directory_identity:
                _refuse()
        except (OSError, ValueError, sync.SyncRefused):
            _refuse()
        version = object.__new__(SealedVersion)
        values = {
            "commitment": commitment,
            "nonce": initial.nonce,
            "size": position,
            "digest": body["whole_digest"],
            "chunks": tuple(chunks),
            "establishment": admission,
            "first_read": final_read,
            "_binding_json": _json(_binding(verified)),
            "_directory": directory,
            "_seal_bytes": seal_bytes,
            "_seal_stamp": seal_stamp,
            "_directory_identity": directory_identity,
            "_snapshot_identity": (snapshot_info.st_dev, snapshot_info.st_ino),
        }
        for key, value in values.items():
            object.__setattr__(version, key, value)
        # Even the first prepared response gets its own final-version admission;
        # no establishment marker or establishment row is reused as a span grant.
        prepared = await self.prepare(version, final_read)
        return version, prepared

    @staticmethod
    def _private_bytes(path: Path, expected: bytes) -> tuple[bytes, tuple[int, ...]]:
        """Read the exact seal length, not an arbitrary substituted metadata size.

        This is equality with committed bytes, not a new size policy: a corrupt
        sidecar cannot make preparation allocate an uncommitted file's length.
        """
        scope = files._read_scope(Path(path.anchor), path.parent, "mentioned file seal")
        fd = sync.open_for_read(path, scope, nonblock=True)
        try:
            info = os.fstat(fd)
            if info.st_nlink != 1 or info.st_mode & 0o077 or info.st_size != len(expected):
                _refuse()
            data = files._read_span(fd, 0, len(expected))
            if data != expected or _stamp(os.fstat(fd)) != _stamp(info):
                _refuse()
            return data, _stamp(info)
        finally:
            os.close(fd)

    async def prepare(self, version: SealedVersion, read: B.ReadBinding) -> PreparedResponse:
        """Fresh exact audit, then verify committed chunks BEFORE slicing bytes.

        Stat metadata comes from the exact immutable seal; verify hashes a span
        while retaining at most one FILE_CHUNK_BYTES block. Fetch/resume retain
        at most that existing protocol granularity, never a whole-file buffer.
        """
        if type(version) is not SealedVersion or type(read) is not B.ReadBinding:
            _refuse()
        fixed = (
            "owner_key",
            "network_id",
            "network_epoch",
            "session_id",
            "source_row_id",
            "provenance_id",
            "proof_digest",
        )
        if (
            read.file_version != version.commitment
            or any(getattr(read, name) != getattr(version.first_read, name) for name in fixed)
            or read.offset > version.size
            or read.length > version.size - read.offset
            or (read.phase in {"fetch", "resume"} and read.length > files.FILE_CHUNK_BYTES)
            or (read.phase == "stat" and (read.offset != 0 or read.length != 0))
        ):
            _refuse()
        verified = await self._verified(read)
        if _json(_binding(verified)) != version._binding_json:
            _refuse()
        fd, info = self._current(read, verified)
        before = _stamp(info)
        os.close(fd)
        admission = self._admit(read.phase, verified, read, before, version.nonce)
        self._recheck(read, verified, before)
        try:
            if tuple(_directory_identity(version._directory)) != version._directory_identity:
                _refuse()
            raw, stamp = self._private_bytes(version._directory / "seal.json", version._seal_bytes)
            if raw != version._seal_bytes or stamp != version._seal_stamp:
                _refuse()
            # The manifest is the commitment, not a freshly hashed mutable source.
            seal = json.loads(raw)
            committed_body = {
                k: v for k, v in seal.items() if k not in {"commitment", "first_read"}
            }
            if "sealed:" + _digest(committed_body) != version.commitment:
                _refuse()
            snapshot = version._directory / "snapshot"
            scope = files._read_scope(
                Path(snapshot.anchor), snapshot.parent, "mentioned file snapshot"
            )
            fd = sync.open_for_read(snapshot, scope, nonblock=True)
            digest = hashlib.sha256()
            payload: list[bytes] = []
            try:
                snapshot_info = os.fstat(fd)
                if (
                    snapshot_info.st_nlink != 1
                    or snapshot_info.st_mode & 0o077
                    or snapshot_info.st_size != version.size
                    or (snapshot_info.st_dev, snapshot_info.st_ino) != version._snapshot_identity
                ):
                    _refuse()
                position = 0
                for length, expected in version.chunks:
                    end = position + length
                    if end > read.offset and position < read.offset + read.length:
                        block = files._read_span(fd, position, length)
                        if len(block) != length or _bytes_digest(block) != expected:
                            _refuse()
                        # Slice the VERIFIED bytes themselves, never reread the
                        # snapshot after a successful hash. bytes are immutable.
                        selected = block[
                            max(0, read.offset - position) : min(
                                length, read.offset + read.length - position
                            )
                        ]
                        digest.update(selected)
                        if read.phase in {"fetch", "resume"}:
                            payload.append(selected)
                    position = end
                if _stamp(os.fstat(fd)) != _stamp(snapshot_info):
                    _refuse()
            finally:
                os.close(fd)
        except (OSError, ValueError, KeyError, TypeError, sync.SyncRefused):
            _refuse()
        self._recheck(read, verified, before)
        verified.require_read(self.link, read)
        prepared = object.__new__(PreparedResponse)
        values = {
            "read_binding": read,
            "verified_binding": verified,
            "admission": admission,
            "version": version.commitment,
            "size": version.size,
            "whole_digest": version.digest,
            "span_digest": "sha256:" + digest.hexdigest(),
            "payload": b"".join(payload),
        }
        for key, value in values.items():
            object.__setattr__(prepared, key, value)
        return prepared

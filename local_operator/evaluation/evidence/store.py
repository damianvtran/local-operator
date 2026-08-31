"""Crash-safe, confined persistence for evaluation evidence bundles.

A writer holds one non-blocking kernel lock for its full lifetime and serializes
threads with an ``RLock``.  Kernel owner death releases the lock, but recovery is
intentionally abandon-only: process-local lifecycle permits and external
side-effect authority cannot be reconstructed from bytes after a crash.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import io
import json
import os
import secrets
import stat
import sys
import threading
import time
import weakref
from collections.abc import Iterable, Mapping
from typing import Any, BinaryIO, Protocol, cast
from urllib.parse import unquote

from local_operator.evaluation.evidence.models import (
    AbandonmentReason,
    AbandonmentRecord,
    EventKind,
    EventPayload,
    EventRecord,
    EvidenceArtifactRef,
    EvidenceManifest,
    FinalizationIntent,
    ModelResponsePayload,
    OutcomeDraft,
    OutcomeSeal,
    ScoringResultPayload,
    ScoringStartPayload,
    StateMarker,
    canonical_bytes,
)
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import RedactionSet

_LOCK = ".lock"
_MANIFEST = "manifest.json"
_EVENTS = "events.jsonl"
_ARTIFACTS = "artifacts"
_STATE = "state.json"
_OUTCOME = "outcome.json"
_ABANDONMENT = "abandonment.json"
_DIGEST_CHARS = frozenset("0123456789abcdef")
_WRITE_FLAGS = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_DIR_FLAGS = _READ_FLAGS | getattr(os, "O_DIRECTORY", 0)


class EvidenceError(RuntimeError):
    """Generic durable-boundary failure which never includes evidence bytes."""


class EvidenceUnsupported(EvidenceError):
    pass


class EvidenceBundleBusy(EvidenceError):
    pass


class EvidenceBundleInvalid(EvidenceError):
    pass


class EvidenceTerminal(EvidenceError):
    pass


class EvidenceRecoveryOnly(EvidenceError):
    pass


class Syscalls(Protocol):
    """Injectable crash cutpoints used to prove ordering without monkeypatch races."""

    def write(self, fd: int, data: bytes) -> int: ...

    def fsync(self, fd: int) -> None: ...

    def link(self, src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int) -> None: ...

    def unlink(self, path: str, *, dir_fd: int) -> None: ...


class _OSCalls:
    def write(self, fd: int, data: bytes) -> int:
        return os.write(fd, data)

    def fsync(self, fd: int) -> None:
        os.fsync(fd)

    def link(self, src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
        os.link(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    def unlink(self, path: str, *, dir_fd: int) -> None:
        os.unlink(path, dir_fd=dir_fd)


_OS_CALLS = _OSCalls()


def _safe_file(fd: int) -> None:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise EvidenceBundleInvalid("unsafe evidence file")
    _safe_owner_mode(info)


def _safe_dir(fd: int) -> None:
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode):
        raise EvidenceBundleInvalid("unsafe evidence directory")
    _safe_owner_mode(info)


def _safe_owner_mode(info: os.stat_result) -> None:
    if not hasattr(os, "geteuid"):
        raise EvidenceUnsupported("evidence ownership checks require geteuid")
    if info.st_uid != os.geteuid():
        raise EvidenceBundleInvalid("unsafe evidence ownership")
    if stat.S_IMODE(info.st_mode) & 0o022:
        raise EvidenceBundleInvalid("unsafe evidence permissions")


def _write_all(fd: int, data: bytes, calls: Syscalls) -> None:
    view = memoryview(data)
    interrupted = 0
    while view:
        try:
            written = calls.write(fd, view.tobytes())
        except InterruptedError:
            interrupted += 1
            if interrupted >= 16:
                raise EvidenceError("evidence write repeatedly interrupted") from None
            continue
        if written <= 0:
            raise EvidenceError("incomplete evidence write")
        interrupted = 0
        view = view[written:]


def _project_artifact(data: bytes, media_type: str) -> Any:
    if media_type == "application/json":
        try:
            decoded = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise EvidenceBundleInvalid("artifact media validation failed") from error
        if canonical_bytes(decoded) != data:
            raise EvidenceBundleInvalid("artifact media validation failed")
        return decoded
    if media_type == "text/plain":
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise EvidenceBundleInvalid("artifact media validation failed") from error
    if media_type == "image/png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise EvidenceBundleInvalid("artifact media validation failed")
    if media_type == "image/jpeg" and not (
        data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9")
    ):
        raise EvidenceBundleInvalid("artifact media validation failed")
    if media_type == "image/gif" and not data.startswith((b"GIF87a", b"GIF89a")):
        raise EvidenceBundleInvalid("artifact media validation failed")
    if media_type == "image/webp" and not (
        len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    ):
        raise EvidenceBundleInvalid("artifact media validation failed")
    # Binary formats are scanned only through their safe ASCII projection; this
    # catches encoded canaries without interpreting arbitrary provider bytes.
    return "".join(chr(byte) if 32 <= byte <= 126 else " " for byte in data)


class EvidenceWriter:
    """Sole append/finalization authority for one evidence directory."""

    def __init__(
        self,
        *,
        root: str,
        root_fd: int,
        artifacts_fd: int,
        lock_fd: int,
        events_fd: int,
        manifest: EvidenceManifest,
        redactions: RedactionSet,
        sequence: int,
        head: str,
        last_monotonic_ns: int,
        last_wall_time_ms: int,
        recovery_only: bool,
        syscalls: Syscalls,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self._root_fd = root_fd
        self._artifacts_fd = artifacts_fd
        self._lock_fd = lock_fd
        self._events_fd = events_fd
        self._redactions = redactions
        self._sequence = sequence
        self._head = head
        self._last_monotonic_ns = last_monotonic_ns
        self._last_wall_time_ms = last_wall_time_ms
        self._recovery_only = recovery_only
        self._calls = syscalls
        self._thread_lock = threading.RLock()
        self._closed = False
        self._poisoned = False
        self._creator_pid = os.getpid()
        reference = weakref.ref(self)

        def inherited_child() -> None:
            writer = reference()
            if writer is not None:
                writer._invalidate_inherited_child()

        # flock state is attached to the inherited open-file description. The
        # child closes only its copies; the parent's descriptor keeps the lock.
        os.register_at_fork(after_in_child=inherited_child)

    @classmethod
    def create(
        cls,
        root: os.PathLike[str] | str,
        manifest: EvidenceManifest,
        redactions: RedactionSet,
        *,
        syscalls: Syscalls | None = None,
    ) -> "EvidenceWriter":
        """Create or safely reopen only the still-empty bundle owned by this run."""

        cls._supported()
        path = os.path.abspath(os.fspath(root))
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
        root_fd = os.open(path, _DIR_FLAGS)
        try:
            _safe_dir(root_fd)
            lock_fd = cls._lock(root_fd)
            try:
                if cls._entry_exists(root_fd, _OUTCOME) or cls._entry_exists(root_fd, _ABANDONMENT):
                    raise EvidenceTerminal("bundle already has an immutable terminal")
                artifacts_fd = cls._open_artifacts(root_fd, create=True)
                try:
                    try:
                        cls._create_immutable(
                            root_fd, _MANIFEST, manifest.to_canonical_json(), _OS_CALLS
                        )
                    except EvidenceTerminal:
                        # Manifest alone is idempotent: exact canonical bytes may
                        # be reused, but are never replaced or resumed after events.
                        pass
                    raw_manifest = cls._read_regular(root_fd, _MANIFEST)
                    if raw_manifest != manifest.to_canonical_json():
                        raise EvidenceBundleInvalid("existing manifest does not match bundle")
                    events_fd = os.open(
                        _EVENTS,
                        _WRITE_FLAGS | os.O_APPEND | os.O_CREAT,
                        0o600,
                        dir_fd=root_fd,
                    )
                    _safe_file(events_fd)
                    raw_events = cls._read_regular(root_fd, _EVENTS)
                    if raw_events:
                        raise EvidenceBundleInvalid("existing bundle cannot resume execution")
                    cls._write_state(
                        root_fd,
                        StateMarker(
                            state="open",
                            bundle_id=manifest.bundle_id,
                            updated_wall_time_ms=int(time.time_ns() // 1_000_000),
                        ),
                        _OS_CALLS,
                    )
                    return cls(
                        root=path,
                        root_fd=root_fd,
                        artifacts_fd=artifacts_fd,
                        lock_fd=lock_fd,
                        events_fd=events_fd,
                        manifest=manifest,
                        redactions=redactions,
                        sequence=0,
                        head=manifest.manifest_digest,
                        last_monotonic_ns=-1,
                        last_wall_time_ms=-1,
                        recovery_only=False,
                        syscalls=syscalls or _OS_CALLS,
                    )
                except BaseException:
                    os.close(artifacts_fd)
                    raise
            except BaseException:
                os.close(lock_fd)
                raise
        except BaseException:
            os.close(root_fd)
            raise

    @classmethod
    def open_for_abandon(
        cls,
        root: os.PathLike[str] | str,
        redactions: RedactionSet,
        *,
        syscalls: Syscalls | None = None,
    ) -> "EvidenceWriter":
        """Acquire a dead owner's bundle only after independent verification.

        The returned object exposes ordinary methods for one API shape, but every
        execution/scoring path rejects because durable bytes cannot recreate the
        cooperative permits, environment leases, or model-call authority.
        """

        cls._supported()
        path = os.path.abspath(os.fspath(root))
        root_fd = os.open(path, _DIR_FLAGS)
        try:
            _safe_dir(root_fd)
            lock_fd = cls._lock(root_fd)
            try:
                report = verify_bundle(path)
                allowed = {"state_stale"}
                if any(
                    issue.severity == "error" and issue.code not in allowed
                    for issue in report.issues
                ):
                    raise EvidenceBundleInvalid("bundle failed recovery verification")
                if (
                    report.terminal_state in ("sealed", "abandoned", "invalid")
                    or report.manifest is None
                ):
                    raise EvidenceTerminal("bundle is already terminal or invalid")
                artifacts_fd = cls._open_artifacts(root_fd, create=False)
                events_fd = os.open(_EVENTS, _WRITE_FLAGS | os.O_APPEND, dir_fd=root_fd)
                _safe_file(events_fd)
                events = report.events
                return cls(
                    root=path,
                    root_fd=root_fd,
                    artifacts_fd=artifacts_fd,
                    lock_fd=lock_fd,
                    events_fd=events_fd,
                    manifest=report.manifest,
                    redactions=redactions,
                    sequence=len(events),
                    head=events[-1].event_id if events else report.manifest.manifest_digest,
                    last_monotonic_ns=events[-1].monotonic_ns if events else -1,
                    last_wall_time_ms=events[-1].wall_time_ms if events else -1,
                    recovery_only=True,
                    syscalls=syscalls or _OS_CALLS,
                )
            except BaseException:
                os.close(lock_fd)
                raise
        except BaseException:
            os.close(root_fd)
            raise

    @staticmethod
    def _supported() -> None:
        if os.name != "posix" or sys.platform.startswith("win") or not hasattr(fcntl, "flock"):
            raise EvidenceUnsupported(
                "evidence bundles require POSIX flock and directory descriptors"
            )

    @staticmethod
    def _entry_exists(root_fd: int, name: str) -> bool:
        try:
            os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    @staticmethod
    def _lock(root_fd: int) -> int:
        fd = os.open(_LOCK, _WRITE_FLAGS | os.O_CREAT, 0o600, dir_fd=root_fd)
        try:
            _safe_file(fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in (errno.EACCES, errno.EAGAIN):
                    raise EvidenceBundleBusy("evidence bundle already has a writer") from error
                raise
            return fd
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _open_artifacts(root_fd: int, *, create: bool) -> int:
        if create:
            try:
                os.mkdir(_ARTIFACTS, 0o700, dir_fd=root_fd)
                os.fsync(root_fd)
            except FileExistsError:
                pass
        fd = os.open(_ARTIFACTS, _DIR_FLAGS, dir_fd=root_fd)
        try:
            _safe_dir(fd)
            return fd
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _read_regular(dir_fd: int, name: str) -> bytes:
        fd = os.open(name, _READ_FLAGS, dir_fd=dir_fd)
        try:
            _safe_file(fd)
            chunks: list[bytes] = []
            while chunk := os.read(fd, 1024 * 1024):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(fd)

    @staticmethod
    def _create_immutable(dir_fd: int, name: str, data: bytes, calls: Syscalls) -> None:
        """Publish by hard link so an existing terminal can never be replaced."""

        temp = f".{name}.{secrets.token_hex(16)}.tmp"
        fd = os.open(temp, _WRITE_FLAGS | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=dir_fd)
        try:
            _write_all(fd, data, calls)
            calls.fsync(fd)
        except BaseException:
            os.close(fd)
            try:
                calls.unlink(temp, dir_fd=dir_fd)
            except FileNotFoundError:
                pass
            raise
        else:
            os.close(fd)
        try:
            calls.link(temp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            calls.fsync(dir_fd)
        except FileExistsError:
            raise EvidenceTerminal("immutable evidence file already exists")
        finally:
            try:
                calls.unlink(temp, dir_fd=dir_fd)
            except FileNotFoundError:
                pass

    @staticmethod
    def _write_state(root_fd: int, marker: StateMarker, calls: Syscalls) -> None:
        """Replace only diagnostic state; immutable terminal files remain authority."""

        temp = f".{_STATE}.{secrets.token_hex(16)}.tmp"
        fd = os.open(temp, _WRITE_FLAGS | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=root_fd)
        try:
            _write_all(fd, marker.to_canonical_json(), calls)
            calls.fsync(fd)
        except BaseException:
            os.close(fd)
            try:
                calls.unlink(temp, dir_fd=root_fd)
            except FileNotFoundError:
                pass
            raise
        else:
            os.close(fd)
        os.rename(temp, _STATE, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        calls.fsync(root_fd)

    def _invalidate_inherited_child(self) -> None:
        self._poisoned = True
        self._closed = True
        for name in ("_events_fd", "_artifacts_fd", "_lock_fd", "_root_fd"):
            fd = getattr(self, name, -1)
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, name, -1)

    def _poison(self) -> None:
        self._poisoned = True
        if self._events_fd >= 0:
            try:
                os.close(self._events_fd)
            except OSError:
                pass
            self._events_fd = -1

    def _ensure_open(self) -> None:
        if os.getpid() != self._creator_pid:
            self._invalidate_inherited_child()
            raise EvidenceRecoveryOnly("fork-inherited writer is invalid")
        if self._poisoned:
            raise EvidenceRecoveryOnly("ambiguous I/O requires abandon-only recovery")
        if self._closed:
            raise EvidenceError("evidence writer is closed")

    def _assert_redacted(self, value: Any) -> None:
        def decode_percent(item: Any) -> Any:
            if isinstance(item, str):
                return unquote(item)
            if isinstance(item, Mapping):
                return {unquote(str(key)): decode_percent(nested) for key, nested in item.items()}
            if isinstance(item, (list, tuple)):
                return [decode_percent(nested) for nested in item]
            return item

        try:
            self._redactions.assert_clear(value)
            # urllib leaves RFC-unreserved bytes such as '-' literal when it
            # creates a canary. Decode evidence too so adversarial %2D spelling
            # cannot bypass the common deterministic encoding boundary.
            self._redactions.assert_clear(decode_percent(value))
        except ValueError as error:
            # Preserve the generic message and never echo a canary through a
            # pydantic traceback, filesystem error, or caller log.
            raise EvidenceBundleInvalid("evidence redaction rejected content") from error

    def append(
        self,
        kind: EventKind,
        payload: EventPayload | Mapping[str, Any],
        *,
        monotonic_ns: int | None = None,
        wall_time_ms: int | None = None,
    ) -> EventRecord:
        with self._thread_lock:
            self._ensure_open()
            if self._recovery_only:
                raise EvidenceRecoveryOnly("recovered bundle may only be abandoned")
            marker = StateMarker.from_canonical_json(self._read_regular(self._root_fd, _STATE))
            if marker.state != "open":
                raise EvidenceTerminal("ordinary evidence is closed after finalization begins")
            return self._append_locked(
                kind,
                payload,
                monotonic_ns=monotonic_ns,
                wall_time_ms=wall_time_ms,
            )

    def _append_locked(
        self,
        kind: EventKind,
        payload: EventPayload | Mapping[str, Any],
        *,
        monotonic_ns: int | None = None,
        wall_time_ms: int | None = None,
    ) -> EventRecord:
        now_monotonic = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
        now_wall = int(time.time_ns() // 1_000_000) if wall_time_ms is None else wall_time_ms
        if now_monotonic < self._last_monotonic_ns or now_wall < self._last_wall_time_ms:
            raise EvidenceBundleInvalid("evidence timestamps cannot move backward")
        value = (
            cast(Any, payload).model_dump(mode="json")
            if not isinstance(payload, Mapping)
            else dict(payload)
        )
        self._assert_redacted(value)
        event = EventRecord.model_validate(
            {
                "sequence": self._sequence,
                "previous_event_sha256": self._head,
                "monotonic_ns": now_monotonic,
                "wall_time_ms": now_wall,
                "kind": kind,
                "payload": value,
            },
            strict=True,
        )
        encoded = event.to_canonical_json() + b"\n"
        try:
            _write_all(self._events_fd, encoded, self._calls)
            # The memory head advances only after the durable journal commit. A
            # failing fsync leaves memory pointing at the last known durable record.
            self._calls.fsync(self._events_fd)
        except (OSError, EvidenceError):
            self._poison()
            raise
        self._sequence += 1
        self._head = event.event_id
        self._last_monotonic_ns = now_monotonic
        self._last_wall_time_ms = now_wall
        return event

    def publish_artifact(
        self,
        source: bytes | bytearray | memoryview | BinaryIO | Iterable[bytes],
        *,
        media_type: str,
        expected_sha256: str | None = None,
        expected_byte_count: int | None = None,
    ) -> EvidenceArtifactRef:
        with self._thread_lock:
            self._ensure_open()
            if self._recovery_only:
                raise EvidenceRecoveryOnly("recovered bundle may only be abandoned")
            marker = StateMarker.from_canonical_json(self._read_regular(self._root_fd, _STATE))
            if marker.state != "open":
                raise EvidenceTerminal("artifact publication is closed after finalization begins")
            stream: Iterable[bytes]
            if isinstance(source, (bytes, bytearray, memoryview)):
                stream = (bytes(source),)
            elif hasattr(source, "read"):
                reader = cast(BinaryIO, source)

                def chunks() -> Iterable[bytes]:
                    while chunk := reader.read(1024 * 1024):
                        yield chunk

                stream = chunks()
            else:
                stream = source
            temp = f".artifact.{secrets.token_hex(16)}.tmp"
            fd = os.open(
                temp, _WRITE_FLAGS | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=self._artifacts_fd
            )
            digest = hashlib.sha256()
            count = 0
            collected = io.BytesIO()
            try:
                for chunk in stream:
                    if not isinstance(chunk, bytes):
                        raise EvidenceBundleInvalid("artifact stream must yield bytes")
                    digest.update(chunk)
                    count += len(chunk)
                    collected.write(chunk)
                    _write_all(fd, chunk, self._calls)
                data = collected.getvalue()
                projection = _project_artifact(data, media_type)
                self._assert_redacted(projection)
                actual = digest.hexdigest()
                if expected_sha256 is not None and expected_sha256 != actual:
                    raise EvidenceBundleInvalid("artifact digest assertion failed")
                if expected_byte_count is not None and expected_byte_count != count:
                    raise EvidenceBundleInvalid("artifact byte count assertion failed")
                ref = EvidenceArtifactRef(sha256=actual, media_type=media_type, byte_count=count)
                self._calls.fsync(fd)
            except BaseException as error:
                os.close(fd)
                try:
                    self._calls.unlink(temp, dir_fd=self._artifacts_fd)
                except FileNotFoundError:
                    pass
                if isinstance(error, (OSError, EvidenceError)) and not isinstance(
                    error, EvidenceBundleInvalid
                ):
                    self._poison()
                raise
            os.close(fd)
            try:
                self._calls.link(
                    temp,
                    ref.sha256,
                    src_dir_fd=self._artifacts_fd,
                    dst_dir_fd=self._artifacts_fd,
                )
            except FileExistsError:
                existing = self._read_regular(self._artifacts_fd, ref.sha256)
                if (
                    hashlib.sha256(existing).hexdigest() != ref.sha256
                    or len(existing) != ref.byte_count
                ):
                    raise EvidenceBundleInvalid("existing artifact conflicts with digest")
                _project_artifact(existing, media_type)
            finally:
                try:
                    self._calls.unlink(temp, dir_fd=self._artifacts_fd)
                except FileNotFoundError:
                    pass
            try:
                self._calls.fsync(self._artifacts_fd)
            except OSError:
                self._poison()
                raise
            return ref

    def begin_finalization(
        self,
        finalization_id: str,
        scoring_operation_id: str | None,
        intent: FinalizationIntent,
        *,
        monotonic_ns: int | None = None,
        wall_time_ms: int | None = None,
    ) -> EventRecord | None:
        with self._thread_lock:
            self._ensure_open()
            if self._recovery_only:
                raise EvidenceRecoveryOnly("recovered bundle may only be abandoned")
            marker = StateMarker.from_canonical_json(self._read_regular(self._root_fd, _STATE))
            if marker.state != "open":
                raise EvidenceTerminal("finalization has already begun")
            if (intent.kind == "score") != (scoring_operation_id is not None):
                raise EvidenceBundleInvalid("scoring intent and operation ID disagree")
            # Marker durability precedes scoring_start. If death lands between the
            # two, recovery observes an ambiguous no-rescore boundary and abandons.
            try:
                self._write_state(
                    self._root_fd,
                    StateMarker(
                        state="finalizing",
                        bundle_id=self.manifest.bundle_id,
                        updated_wall_time_ms=int(time.time_ns() // 1_000_000),
                        finalization_id=finalization_id,
                        scoring_operation_id=scoring_operation_id,
                        intent=intent.kind,
                        intent_digest=intent.intent_digest,
                    ),
                    self._calls,
                )
            except (OSError, EvidenceError):
                self._poison()
                raise
            if intent.kind == "unscored":
                return None
            assert scoring_operation_id is not None
            assert intent.scorer_id is not None and intent.scorer_version is not None
            return self._append_locked(
                "scoring_start",
                ScoringStartPayload(
                    finalization_id=finalization_id,
                    scoring_operation_id=scoring_operation_id,
                    scorer_id=intent.scorer_id,
                    scorer_version=intent.scorer_version,
                    intent_digest=intent.intent_digest,
                ),
                monotonic_ns=monotonic_ns,
                wall_time_ms=wall_time_ms,
            )

    def record_final_lifecycle(
        self,
        payload: Any,
        *,
        monotonic_ns: int | None = None,
        wall_time_ms: int | None = None,
    ) -> EventRecord:
        """Persist only the completed/failed/cancelled snapshot during finalization."""

        from local_operator.evaluation.evidence.models import LifecycleTransitionPayload

        with self._thread_lock:
            self._ensure_open()
            if self._recovery_only:
                raise EvidenceRecoveryOnly("recovered bundle may only be abandoned")
            if not isinstance(payload, LifecycleTransitionPayload) or payload.state not in (
                "completed",
                "failed",
                "cancelled",
            ):
                raise EvidenceBundleInvalid(
                    "finalization accepts only a terminal lifecycle snapshot"
                )
            marker = StateMarker.from_canonical_json(self._read_regular(self._root_fd, _STATE))
            if marker.state != "finalizing":
                raise EvidenceTerminal("bundle is not finalizing")
            return self._append_locked(
                "lifecycle_transition",
                payload,
                monotonic_ns=monotonic_ns,
                wall_time_ms=wall_time_ms,
            )

    def record_scoring_result(
        self,
        payload: ScoringResultPayload,
        *,
        monotonic_ns: int | None = None,
        wall_time_ms: int | None = None,
    ) -> EventRecord:
        with self._thread_lock:
            self._ensure_open()
            if self._recovery_only:
                raise EvidenceRecoveryOnly("recovered bundle may only be abandoned")
            report = verify_bundle(self.root)
            starts = [
                event.payload
                for event in report.events
                if isinstance(event.payload, ScoringStartPayload)
            ]
            results = [
                event.payload
                for event in report.events
                if isinstance(event.payload, ScoringResultPayload)
            ]
            if len(starts) != 1 or results:
                raise EvidenceTerminal("scoring is not available exactly once")
            if (
                starts[0].finalization_id != payload.finalization_id
                or starts[0].scoring_operation_id != payload.scoring_operation_id
            ):
                raise EvidenceBundleInvalid("scoring result does not match scoring start")
            return self._append_locked(
                "scoring_result",
                payload,
                monotonic_ns=monotonic_ns,
                wall_time_ms=wall_time_ms,
            )

    def seal(self, draft: OutcomeDraft) -> OutcomeSeal:
        with self._thread_lock:
            self._ensure_open()
            if self._recovery_only:
                raise EvidenceRecoveryOnly("recovered bundle may only be abandoned")
            report = verify_bundle(self.root)
            if not report.valid or report.manifest is None or report.counters is None:
                raise EvidenceBundleInvalid("bundle failed independent seal verification")
            if report.terminal_state != "finalizing" or report.outcome or report.abandonment:
                raise EvidenceTerminal("bundle is not open for sealing")
            events = report.events
            starts = [
                event.payload for event in events if isinstance(event.payload, ScoringStartPayload)
            ]
            results = [
                event.payload for event in events if isinstance(event.payload, ScoringResultPayload)
            ]
            marker = StateMarker.from_canonical_json(self._read_regular(self._root_fd, _STATE))
            if draft.result.status == "scored":
                if len(starts) != 1 or len(results) != 1 or results[0].score != draft.result:
                    raise EvidenceBundleInvalid("seal draft disagrees with durable score")
            elif starts or results or marker.intent != "unscored":
                raise EvidenceBundleInvalid("unscored seal disagrees with finalization intent")
            completed = [
                event.payload
                for event in events
                if event.kind == "lifecycle_transition"
                and getattr(event.payload, "state", None) == "completed"
            ]
            if not completed:
                raise EvidenceBundleInvalid("seal requires a completed lifecycle snapshot")
            state = cast(Any, completed[-1])
            assertions = (
                (state.preflight_seal_id, draft.preflight_seal_id),
                (state.commitment_id, draft.commitment_id),
                (state.reconciliation_id, draft.reconciliation_id),
                (state.cleanup_result_id, draft.cleanup_result_id),
            )
            if any(actual != expected for actual, expected in assertions):
                raise EvidenceBundleInvalid("seal draft disagrees with lifecycle receipts")
            reportable = draft.reportability_label == "reportable"
            if reportable and (
                state.reconciliation_reportable is not True
                or state.rescue_required is not False
                or draft.result.status != "scored"
            ):
                raise EvidenceBundleInvalid("reportable seal lacks reportable durable receipts")
            routes_by_key = {
                (
                    event.payload.served_route.provider_id,
                    event.payload.served_route.route_id,
                    event.payload.served_route.model_id,
                ): event.payload.served_route
                for event in events
                if isinstance(event.payload, ModelResponsePayload)
            }
            routes = tuple(routes_by_key[key] for key in sorted(routes_by_key))
            outcome = OutcomeSeal(
                bundle_id=self.manifest.bundle_id,
                manifest_digest=self.manifest.manifest_digest,
                event_count=len(events),
                final_event_sha256=events[-1].event_id if events else self.manifest.manifest_digest,
                artifacts=report.artifacts,
                finalization_id=draft.finalization_id,
                preflight_seal_id=draft.preflight_seal_id,
                commitment_id=draft.commitment_id,
                reconciliation_id=draft.reconciliation_id,
                score_id=draft.result.score_id,
                cleanup_result_id=draft.cleanup_result_id,
                result=draft.result,
                reportable=reportable,
                reportability_label=draft.reportability_label,
                comparable=draft.comparability_label == "comparable",
                comparability_label=draft.comparability_label,
                requested_route=self.manifest.requested_route,
                served_routes=routes,
                counters=report.counters,
                started_wall_time_ms=self.manifest.created_wall_time_ms,
                ended_wall_time_ms=draft.ended_wall_time_ms,
            )
            try:
                self._create_immutable(
                    self._root_fd, _OUTCOME, outcome.to_canonical_json(), self._calls
                )
                # Outcome publication is authoritative before this diagnostic update;
                # a crash here still derives sealed from immutable outcome.json.
                self._write_state(
                    self._root_fd,
                    StateMarker(
                        state="sealed",
                        bundle_id=self.manifest.bundle_id,
                        updated_wall_time_ms=int(time.time_ns() // 1_000_000),
                        terminal_id=outcome.evidence_root,
                    ),
                    self._calls,
                )
            except (OSError, EvidenceError):
                self._poison()
                raise
            return outcome

    def abandon(self, reason: AbandonmentReason, diagnostic_code: str) -> AbandonmentRecord:
        with self._thread_lock:
            self._ensure_open()
            report = verify_bundle(self.root)
            if report.outcome is not None or report.abandonment is not None:
                raise EvidenceTerminal("bundle already has an immutable terminal")
            if report.manifest is None or any(issue.severity == "error" for issue in report.issues):
                raise EvidenceBundleInvalid("bundle failed independent abandonment verification")
            events = report.events
            record = AbandonmentRecord(
                bundle_id=self.manifest.bundle_id,
                manifest_digest=self.manifest.manifest_digest,
                reason=reason,
                diagnostic_code=diagnostic_code,
                last_event_sequence=len(events) - 1 if events else None,
                last_event_sha256=events[-1].event_id if events else self.manifest.manifest_digest,
                event_count=len(events),
                abandoned_wall_time_ms=int(time.time_ns() // 1_000_000),
            )
            try:
                self._create_immutable(
                    self._root_fd,
                    _ABANDONMENT,
                    record.to_canonical_json(),
                    self._calls,
                )
                self._write_state(
                    self._root_fd,
                    StateMarker(
                        state="abandoned",
                        bundle_id=self.manifest.bundle_id,
                        updated_wall_time_ms=int(time.time_ns() // 1_000_000),
                        terminal_id=record.abandonment_id,
                    ),
                    self._calls,
                )
            except (OSError, EvidenceError):
                self._poison()
                raise
            return record

    def close(self) -> None:
        if os.getpid() != self._creator_pid:
            self._invalidate_inherited_child()
            return
        with self._thread_lock:
            if self._closed:
                return
            self._closed = True
            for fd in (self._events_fd, self._artifacts_fd, self._lock_fd, self._root_fd):
                if fd >= 0:
                    os.close(fd)

    def __enter__(self) -> "EvidenceWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

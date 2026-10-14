"""One engine-owned durable local transaction journal for all artifact adapters.

Content, integrated remote baseline and directional receipt are separate files.
A lease alone cannot make them crash-atomic: an adapter prepares this fsynced
record before its logical content commit, then publishes baseline/receipt and
marks complete. Recovery observes actual content hashes, never restores a stale
snapshot over an unexpected user edit. Publication proof lives here, not in an
imported document or a caller's claim that preview succeeded.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

from local_operator.config_writer import config_write_lock, nonblocking_file_claim
from local_operator.hub_sync.identity import Kind, LinkError, LinkIdentity
from local_operator.hub_sync.provenance import now_iso

SCHEMA = 1
MAX_RECORD_BYTES = 8 * 1024 * 1024
_ID = re.compile(r"^[a-f0-9]{32}$")
_FINGERPRINT = re.compile(r"^[a-f0-9]{64}$")
Phase = Literal["prepared", "content-committed", "complete", "aborted", "blocked"]
Direction = Literal["pull", "push"]


class TransactionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _directory(root: Path) -> Path:
    hub = Path(root) / "hub"
    directory = hub / "transactions"
    if hub.is_symlink() or directory.is_symlink():
        raise TransactionError("local-concurrent-edit", "hub transaction directory is unsafe")
    return directory


def _sync_ancestors(directory: Path, stop: Path) -> None:
    """Fsync a directory and its newly-created ancestors up to ``stop`` (exclusive).

    A successful rename only durably renames the parent whose directory entry
    changed. When ``hub``/``transactions``/``backups``/``staging`` are FIRST
    created, their entries live in grandparents that were never fsynced, so a
    power loss could leave the directory absent after the rename that used it.
    Syncing the exact chain (not the whole tree) is what reviewer R4 requires.
    """
    current = directory
    while True:
        _sync_directory(current)
        if current == stop or current.parent == current:
            return
        current = current.parent


def _sync_directory(directory: Path) -> None:
    if os.name == "nt":  # directory fsync is not supported on Windows
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Private fsynced JSON replacement, bounded before touching the destination."""
    if path.is_symlink() or path.parent.is_symlink():
        raise TransactionError("local-concurrent-edit", "hub transaction record is unsafe")
    try:
        body = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise TransactionError(
            "local-concurrent-edit", "hub transaction record is invalid"
        ) from None
    if len(body) > MAX_RECORD_BYTES:
        raise TransactionError("skill-too-large", "hub transaction record exceeds its bound")
    path.parent.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.mkdir(mode=0o700, exist_ok=True)
    # The record's own directory and any ancestors this call first created must
    # be durable before the record is trusted after a crash (reviewer R4).
    _sync_ancestors(path.parent, path.parent.parent.parent)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".transaction-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class LocalTransaction:
    operation_id: str
    kind: Kind
    local_id: str
    link: Mapping[str, Any]
    direction: Direction
    before_fingerprint: str | None
    after_fingerprint: str | None
    payload: Mapping[str, Any]
    phase: Phase = "prepared"
    created_at: str = field(default_factory=now_iso)
    basis: str = "integrated-baseline"
    error_class: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str) or not _ID.fullmatch(self.operation_id):
            raise TransactionError("local-concurrent-edit", "transaction operation id is invalid")
        try:
            identity = LinkIdentity.from_json(self.link)
        except (LinkError, TypeError):
            raise TransactionError("origin-conflict", "transaction origin is invalid") from None
        if identity.kind != self.kind or identity.local_id != self.local_id:
            raise TransactionError(
                "origin-conflict", "transaction origin does not match the artifact"
            )
        if self.direction not in ("pull", "push") or self.phase not in (
            "prepared",
            "content-committed",
            "complete",
            "aborted",
            "blocked",
        ):
            raise TransactionError("local-concurrent-edit", "transaction state is invalid")
        if not isinstance(self.payload, Mapping):
            raise TransactionError("local-concurrent-edit", "transaction payload is invalid")
        for fingerprint in (self.before_fingerprint, self.after_fingerprint):
            if fingerprint is not None and (
                not isinstance(fingerprint, str) or not _FINGERPRINT.fullmatch(fingerprint)
            ):
                # Recovery compares ACTUAL artifact hashes. A malformed host
                # journal must not create an alternate equality/absence basis.
                raise TransactionError(
                    "local-concurrent-edit", "transaction fingerprint is invalid"
                )
        if not isinstance(self.created_at, str) or not isinstance(self.basis, str):
            raise TransactionError("local-concurrent-edit", "transaction provenance is invalid")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "operation_id": self.operation_id,
            "kind": self.kind,
            "local_id": self.local_id,
            "link": dict(self.link),
            "direction": self.direction,
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "payload": dict(self.payload),
            "phase": self.phase,
            "created_at": self.created_at,
            "basis": self.basis,
            "error_class": self.error_class,
        }


class TransactionJournal:
    def __init__(self, config_dir: Path) -> None:
        self.config_dir = Path(config_dir)

    def _path(self, operation_id: str) -> Path:
        if not isinstance(operation_id, str) or not _ID.fullmatch(operation_id):
            raise TransactionError("local-concurrent-edit", "transaction operation id is invalid")
        return _directory(self.config_dir) / f"{operation_id}.json"

    @contextmanager
    def claim(self, operation_id: str) -> Iterator[bool]:
        """Nonblocking preparation/recovery lease, separate from the short root lock.

        Writer takes this before releasing the root lock after prepare. Readers
        may skip a live preparation (current alias is still unchanged), rather
        than aborting its staged tree. Recovery takes root->TRY lease, never waits
        on a writer that needs lease->root, so there is no reversed-order deadlock.
        Kernel release on process death makes crash recovery independent of PID
        reuse or an in-memory claim that cannot survive a restart.
        """
        self._path(operation_id)  # validates the opaque operation component
        directory = self.config_dir / "hub" / "transaction-leases"
        with nonblocking_file_claim(directory / f"{operation_id}.lock") as claimed:
            yield claimed

    def prepare(
        self,
        identity: LinkIdentity,
        direction: Direction,
        *,
        before_fingerprint: str | None,
        after_fingerprint: str | None,
        payload: Mapping[str, Any],
        basis: str = "integrated-baseline",
    ) -> LocalTransaction:
        record = LocalTransaction(
            operation_id=uuid.uuid4().hex,
            kind=identity.kind,
            local_id=identity.local_id,
            link=identity.to_json(),
            direction=direction,
            before_fingerprint=before_fingerprint,
            after_fingerprint=after_fingerprint,
            payload=payload,
            basis=basis,
        )
        self.write(record)
        return record

    def write(self, record: LocalTransaction) -> None:
        with config_write_lock(self.config_dir):
            atomic_json(self._path(record.operation_id), record.to_json())

    def transition(
        self, record: LocalTransaction, phase: Phase, *, error_class: str | None = None
    ) -> LocalTransaction:
        changed = replace(record, phase=phase, error_class=error_class)
        self.write(changed)
        return changed

    def read(self, operation_id: str) -> LocalTransaction | None:
        path = self._path(operation_id)
        if path.is_symlink():
            raise TransactionError("local-concurrent-edit", "hub transaction record is unsafe")
        try:
            with path.open("rb") as stream:
                body = stream.read(MAX_RECORD_BYTES + 1)
        except FileNotFoundError:
            return None
        if len(body) > MAX_RECORD_BYTES:
            raise TransactionError(
                "local-concurrent-edit", "hub transaction record exceeds its bound"
            )
        try:
            value = json.loads(body)
            if (
                not isinstance(value, dict)
                or type(value.get("schema")) is not int
                or value["schema"] != SCHEMA
            ):
                raise ValueError("invalid schema")
            if value.get("operation_id") != operation_id:
                raise ValueError("invalid operation")
            return LocalTransaction(
                operation_id=value["operation_id"],
                kind=value["kind"],
                local_id=value["local_id"],
                link=value["link"],
                direction=value["direction"],
                before_fingerprint=value["before_fingerprint"],
                after_fingerprint=value["after_fingerprint"],
                payload=value["payload"],
                phase=value["phase"],
                created_at=value["created_at"],
                basis=value.get("basis", "integrated-baseline"),
                error_class=value.get("error_class"),
            )
        except (ValueError, KeyError, TypeError, RecursionError, UnicodeError):
            raise TransactionError(
                "local-concurrent-edit", "hub transaction record is invalid"
            ) from None

    def pending(
        self, *, kind: Kind | None = None, local_id: str | None = None
    ) -> list[LocalTransaction]:
        directory = _directory(self.config_dir)
        if not directory.exists():
            return []
        records = []
        for path in sorted(directory.glob("*.json")):
            record = self.read(path.stem)
            if record is None or record.phase in ("complete", "aborted"):
                continue
            if (
                kind is not None
                and record.kind != kind
                or local_id is not None
                and record.local_id != local_id
            ):
                continue
            records.append(record)
        return records

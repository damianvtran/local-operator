"""Default-off, Linux owner-wide negative coordination; no runtime consumers.

This is NOT account proof, source verification, audit, or a serving permission.
Trusted future callers must begin BEFORE any affected effect/privilege prompt,
perform and independently verify that effect OUTSIDE our innermost lock, and
finish only their exact transaction. Unknown outcomes stay pending indefinitely.
No process-death, timeout, profile, or key-rotation recovery exists here.

The owner-wide coordination root is managed control state, never eligible file
content. Future activation must reuse central managed-root/config exclusions;
this prerequisite does not add a denylist or any serving integration.

A guard body may perform at most ONE nonblocking emission. It must not acquire
external locks, read sources, sync audit, prompt, check presence, wait for socket
readiness, or use sendall. A positive observation proves only this coordinator's
freshness: no publishers participate yet, so there is NO exhaustive revoke fence.
Owner-private integrity/coherent all-state rollback is the trust boundary, not
hardware anti-rollback. Initializing is an explicit never-used prerequisite, not
a read-path recovery operation; failed/partial initialization is not repairable.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator
from uuid import UUID, uuid4


class MutationOutcome(str, Enum):
    """Exactly three outcomes; indeterminate is never success or a retry token."""

    FINISHED = "finished"
    DENIED = "denied"
    INDETERMINATE = "indeterminate"


class CoordinatorRefusal(RuntimeError):
    """Unavailable, busy, disabled, malformed, or unconfirmed: never permission."""

    outcome = MutationOutcome.DENIED


class CoordinatorIndeterminate(CoordinatorRefusal):
    """No acknowledgement; disk may have committed. This handle stays negative."""

    outcome = MutationOutcome.INDETERMINATE


@dataclass(frozen=True)
class EffectIdentity:
    """Opaque operation + boundary keys, including caller's exact profile scope.

    These keys are not paths to open, authority, or permission. Future callers
    must make them distinguish every affected external operation/profile; a
    transaction for a different key can never finish/adopt this effect.
    """

    operation_key: str
    boundary_key: str

    def __post_init__(self) -> None:
        if any(type(key) is not str or not key for key in (self.operation_key, self.boundary_key)):
            raise TypeError("exact nonempty effect keys required")


@dataclass(frozen=True)
class VerifiedOutcome:
    """Trusted caller's independently read-back SHA256 of intended result.

    Construction does NOT do verification: only create after observing the exact
    intended effect outside our lock. A constant, callback, or truthy result is
    not permission and cannot substitute for P1/P2/audit or outcome verification.
    """

    effect: EffectIdentity
    fingerprint: str

    def __post_init__(self) -> None:
        if type(self.effect) is not EffectIdentity:
            raise TypeError("exact effect identity required")
        if (
            type(self.fingerprint) is not str
            or len(self.fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in self.fingerprint)
        ):
            raise TypeError("canonical SHA256 outcome fingerprint required")


@dataclass(frozen=True)
class CoordinatorReceipt:
    """Capture before slow verification; bind the entire owner-wide domain."""

    domain: str
    incarnation: str
    revision: int

    def __post_init__(self) -> None:
        if (
            type(self.domain) is not str
            or not self.domain
            or type(self.revision) is not int
            or self.revision < 0
        ):
            raise TypeError("exact domain and nonnegative revision required")
        _uuid(self.incarnation)


@dataclass(frozen=True)
class MutationReceipt:
    """Exact ownership handle; not a generic abort/clear capability."""

    domain: str
    incarnation: str
    transaction: str
    effect: EffectIdentity

    def __post_init__(self) -> None:
        if type(self.domain) is not str or not self.domain:
            raise TypeError("exact domain required")
        _uuid(self.incarnation)
        _uuid(self.transaction)
        if type(self.effect) is not EffectIdentity:
            raise TypeError("exact effect identity required")


@dataclass(frozen=True)
class CompletionEvidence:
    """Read-only exact committed fact, NEVER permission or an upgraded old result."""

    transaction: MutationReceipt
    verified: VerifiedOutcome
    revision: int

    def __post_init__(self) -> None:
        if (
            type(self.transaction) is not MutationReceipt
            or type(self.verified) is not VerifiedOutcome
            or self.transaction.effect != self.verified.effect
            or type(self.revision) is not int
            or self.revision < 2
        ):
            raise TypeError("exact completion evidence required")


@dataclass(frozen=True)
class FinishedMutation:
    """Successful completion ACK, returned only after durability and readback."""

    evidence: CompletionEvidence
    outcome = MutationOutcome.FINISHED

    def __post_init__(self) -> None:
        if type(self.evidence) is not CompletionEvidence:
            raise TypeError("exact completion evidence required")


@dataclass(frozen=True)
class CoordinatorObservation:
    """Mechanics only. Consumers must still supply every independent authority."""

    current: bool

    def __post_init__(self) -> None:
        if type(self.current) is not bool:
            raise TypeError("exact observation bool required")


@dataclass(frozen=True)
class _State:
    incarnation: str
    revision: int
    pending: tuple[MutationReceipt, ...]
    completed: tuple[CompletionEvidence, ...] = ()


# flock does not serialize threads sharing one open description. Match store's
# per-path thread-lock + stable separate file-lock composition, without its
# mkdir-on-read or placement's timeout policy. Contention simply refuses.
_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _owner_home(uid: int) -> Path:
    import pwd  # Unsupported platforms must be importable while disabled.

    return Path(pwd.getpwuid(uid).pw_dir)


def _anchor_path(uid: int) -> Path:
    from local_operator.operator.trust import anchor_path

    return anchor_path(uid)


def _uuid(value: object) -> str:
    if type(value) is not str or str(UUID(value)) != value:
        raise ValueError("invalid coordinator identifier")
    return value


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate coordinator field")
        result[key] = value
    return result


def _encode(value: dict[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class FileReleaseCoordinator:
    """Explicit opt-in mechanics, with no environment/config/CLI enable switch.

    Location comes only from the OS owner and canonical anchor PATH spelling,
    never anchor key content or HOME/XDG/profile. OS location is not identity.
    All exceptions are negative; do not proceed with an affected effect after a
    begin failure, or retry completion as though a failed write had rolled back.
    """

    def __init__(self, *, file_release_coordinator_enabled: bool = False) -> None:
        if type(file_release_coordinator_enabled) is not bool:
            raise TypeError("file_release_coordinator_enabled must be an exact bool")
        self._enabled = file_release_coordinator_enabled
        self._identity: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._failed = False

    def _location(self) -> tuple[int, Path, tuple[str, ...]]:
        if not self._enabled or sys.platform != "linux" or self._failed:
            raise CoordinatorRefusal("coordinator disabled or unavailable")
        try:
            uid = os.getuid()
            home = _owner_home(uid)
            anchor = _anchor_path(uid)
            if not home.is_absolute() or home == Path("/") or not anchor.is_absolute():
                raise ValueError("unresolved owner location")
            # strict=False canonicalizes spelling WITHOUT opening/creating an
            # anchor, consulting its key bytes, or requiring an installation.
            domain = hashlib.sha256(os.fsencode(anchor.resolve())).hexdigest()
            return (
                uid,
                home,
                (
                    ".local",
                    "state",
                    "local-operator",
                    "file-release",
                    f"uid-{uid}",
                    f"anchor-{domain}",
                ),
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            raise CoordinatorRefusal("owner location unresolved") from exc

    @staticmethod
    def _node(info: os.stat_result) -> tuple[int, int]:
        return info.st_dev, info.st_ino

    @staticmethod
    def _check(info: os.stat_result, uid: int, *, directory: bool, private: bool = True) -> None:
        kind = stat.S_ISDIR if directory else stat.S_ISREG
        mode = 0o700 if directory else 0o600
        if (
            not kind(info.st_mode)
            or info.st_uid != uid
            or (private and stat.S_IMODE(info.st_mode) != mode)
            or (not private and info.st_mode & 0o022)
            or (not directory and info.st_nlink != 1)
        ):
            raise CoordinatorRefusal("unsafe coordinator metadata")

    def _open_directory(
        self, *, create: bool = False
    ) -> tuple[int, Path, tuple[tuple[int, int], ...]]:
        uid, home, suffix = self._location()
        if ".." in home.parts:
            raise CoordinatorRefusal("noncanonical owner home")
        path = home.joinpath(*suffix)
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        nodes: list[tuple[int, int]] = []
        try:
            # Directory-relative opens never follow a symlink, including a HOME
            # ancestor. Pin the whole walk so a renamed ancestor cannot silently
            # redirect later operations to an old but still-open domain.
            components = home.parts[1:] + suffix
            for index, part in enumerate(components):
                if create and index >= len(home.parts) - 1:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                    else:
                        os.fsync(fd)
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                try:
                    info = os.fstat(child)
                    if index >= len(home.parts) - 2:
                        # Shared .local/state parents need not be 0700, but must
                        # be owner-held and not writable by another OS user.
                        self._check(info, uid, directory=True, private=index >= len(home.parts) + 2)
                    if self._node(info) != self._node(
                        os.stat(part, dir_fd=fd, follow_symlinks=False)
                    ):
                        raise CoordinatorRefusal("directory identity changed")
                    nodes.append(self._node(info))
                    if create:
                        os.fsync(child)
                except BaseException:
                    os.close(child)
                    raise
                os.close(fd)
                fd = child
            return fd, path, tuple(nodes)
        except BaseException:
            os.close(fd)
            raise

    def _check_directory(self, nodes: tuple[tuple[int, int], ...]) -> None:
        fd, _, current = self._open_directory()
        os.close(fd)
        if current != nodes:
            raise CoordinatorRefusal("coordinator path identity changed")

    def _read(self, fd: int, name: str) -> dict[str, object]:
        uid = os.getuid()
        source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            info = os.fstat(source)
            self._check(info, uid, directory=False)
            with os.fdopen(os.dup(source), "rb") as stream:
                raw = stream.read()
            result = json.loads(raw, object_pairs_hook=_pairs)
            if type(result) is not dict or _encode(result) != raw:
                raise CoordinatorRefusal("noncanonical coordinator state")
            named = os.stat(name, dir_fd=fd, follow_symlinks=False)
            self._check(named, uid, directory=False)
            if self._node(info) != self._node(named):
                raise CoordinatorRefusal("coordinator file identity changed")
            return result
        finally:
            os.close(source)

    def _write(self, fd: int, name: str, payload: dict[str, object]) -> None:
        # Same-directory staging, file fsync, rename, parent fsync, exact readback
        # matches store's opt-in durable writer. dir_fd makes every operation
        # refer to our validated directory; unlike the legacy helper this never
        # mkdirs/chmods on a read and never accepts a substituted parent path.
        raw = _encode(payload)
        temporary = f".{name}.{uuid4()}.tmp"
        target = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd
        )
        try:
            with os.fdopen(os.dup(target), "wb") as stream:
                stream.write(raw)
                stream.flush()
            os.fsync(target)
            info = os.fstat(target)
            self._check(info, os.getuid(), directory=False)
            # Rename is the visibility linearization point, NOT a successful
            # acknowledgement. Only parent fsync + exact readback acknowledge.
            os.replace(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
            os.fsync(fd)
            if self._node(os.stat(name, dir_fd=fd, follow_symlinks=False)) != self._node(info):
                raise CoordinatorRefusal("publication identity changed")
            if self._read(fd, name) != payload:
                raise CoordinatorRefusal("coordinator readback mismatch")
        finally:
            os.close(target)
            try:
                os.unlink(temporary, dir_fd=fd)
            except FileNotFoundError:
                pass

    @contextmanager
    def _locked(self, *, initialize: bool = False) -> Iterator[tuple[int, str, tuple[int, int]]]:
        from local_operator.wakes.lock import _try_lock, _unlock

        fd = lock = None
        held = False
        thread_lock = None
        try:
            fd, path, nodes = self._open_directory(create=initialize)
            with _LOCKS_GUARD:
                thread_lock = _LOCKS.setdefault(path, threading.Lock())
            if not thread_lock.acquire(blocking=False):
                thread_lock = None
                raise CoordinatorRefusal("coordinator busy")
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
            if initialize:
                # The never-unlinked lock also witnesses an attempted init.
                # Missing state after a crash can therefore NEVER bootstrap.
                if os.listdir(fd):
                    raise CoordinatorRefusal("coordinator already used")
                flags |= os.O_CREAT | os.O_EXCL
            lock = os.open(".mutation.lock", flags, 0o600, dir_fd=fd)
            info = os.fstat(lock)
            self._check(info, os.getuid(), directory=False)
            if not _try_lock(lock):
                raise CoordinatorRefusal("coordinator busy")
            held = True
            lock_node = self._node(info)
            identity = (self._node(os.fstat(fd)), lock_node)
            if self._identity is not None and self._identity != identity:
                raise CoordinatorRefusal("coordinator handle identity changed")
            self._check_directory(nodes)
            named = os.stat(".mutation.lock", dir_fd=fd, follow_symlinks=False)
            self._check(named, os.getuid(), directory=False)
            if self._node(named) != lock_node:
                raise CoordinatorRefusal("coordinator lock identity changed")
            self._identity = identity
            if initialize:
                os.fsync(lock)
                os.fsync(fd)
            yield fd, hashlib.sha256(os.fsencode(path)).hexdigest(), lock_node
            self._check_directory(nodes)
            if self._node(os.stat(".mutation.lock", dir_fd=fd, follow_symlinks=False)) != lock_node:
                raise CoordinatorRefusal("coordinator lock identity changed")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise CoordinatorRefusal("coordinator state unconfirmed") from exc
        finally:
            if lock is not None:
                try:
                    if held:
                        _unlock(lock)
                finally:
                    os.close(lock)
            if fd is not None:
                os.close(fd)
            if thread_lock is not None:
                thread_lock.release()

    @staticmethod
    def _receipt_data(receipt: MutationReceipt) -> dict[str, object]:
        return {
            "domain": receipt.domain,
            "incarnation": receipt.incarnation,
            "transaction": receipt.transaction,
            "effect": {
                "operation_key": receipt.effect.operation_key,
                "boundary_key": receipt.effect.boundary_key,
            },
        }

    @staticmethod
    def _parse_receipt(raw: object) -> MutationReceipt:
        if type(raw) is not dict or set(raw) != {"domain", "incarnation", "transaction", "effect"}:
            raise CoordinatorRefusal("invalid transaction record")
        effect = raw["effect"]
        if type(effect) is not dict or set(effect) != {"operation_key", "boundary_key"}:
            raise CoordinatorRefusal("invalid effect identity")
        return MutationReceipt(
            raw["domain"],
            raw["incarnation"],
            raw["transaction"],
            EffectIdentity(effect["operation_key"], effect["boundary_key"]),
        )

    def _load(self, fd: int, lock_node: tuple[int, int], domain: str) -> _State:
        witness = self._read(fd, "initialized.json")
        if (
            set(witness) != {"schema", "incarnation", "lock_device", "lock_inode"}
            or type(witness["schema"]) is not int
            or witness["schema"] != 1
            or type(witness["lock_device"]) is not int
            or type(witness["lock_inode"]) is not int
            or (witness["lock_device"], witness["lock_inode"]) != lock_node
        ):
            raise CoordinatorRefusal("invalid initialization witness")
        incarnation = _uuid(witness["incarnation"])
        raw = self._read(fd, "state.json")
        if (
            set(raw) != {"schema", "incarnation", "revision", "pending", "completed"}
            or type(raw["schema"]) is not int
            or raw["schema"] != 1
            or raw["incarnation"] != incarnation
            or type(raw["revision"]) is not int
            or raw["revision"] < 0
            or type(raw["pending"]) is not list
            or type(raw["completed"]) is not list
        ):
            raise CoordinatorRefusal("invalid coordinator state")
        pending = tuple(self._parse_receipt(item) for item in raw["pending"])
        completed: list[CompletionEvidence] = []
        for item in raw["completed"]:
            if type(item) is not dict or set(item) != {"transaction", "fingerprint", "revision"}:
                raise CoordinatorRefusal("invalid completion record")
            receipt = self._parse_receipt(item["transaction"])
            completed.append(
                CompletionEvidence(
                    receipt, VerifiedOutcome(receipt.effect, item["fingerprint"]), item["revision"]
                )
            )
        transactions = [item.transaction for item in pending] + [
            item.transaction.transaction for item in completed
        ]
        if (
            len(transactions) != len(set(transactions))
            or pending != tuple(sorted(pending, key=lambda item: item.transaction))
            or completed != sorted(completed, key=lambda item: item.transaction.transaction)
            or raw["revision"] != len(pending) + 2 * len(completed)
            or len({item.revision for item in completed}) != len(completed)
            or any(item.revision > raw["revision"] for item in completed)
            or any(
                item.domain != domain or item.incarnation != incarnation
                for item in (*pending, *(entry.transaction for entry in completed))
            )
        ):
            raise CoordinatorRefusal("ambiguous transaction history")
        return _State(incarnation, raw["revision"], pending, tuple(completed))

    def _save(self, fd: int, state: _State) -> None:
        self._write(
            fd,
            "state.json",
            {
                "schema": 1,
                "incarnation": state.incarnation,
                "revision": state.revision,
                "pending": [self._receipt_data(item) for item in state.pending],
                # Exact committed facts are retained, with no invented expiry,
                # cap, PID/time repair, or history-based positive reconstruction.
                "completed": [
                    {
                        "transaction": self._receipt_data(item.transaction),
                        "fingerprint": item.verified.fingerprint,
                        "revision": item.revision,
                    }
                    for item in state.completed
                ],
            },
        )

    def initialize(self) -> None:
        """Explicit never-used prerequisite ONLY, before any consumer exists.

        An interrupted initializer is negative, not an invitation to rerun it.
        No public reset, repair, decommission, or abort-to-clear is provided.
        """
        with self._locked(initialize=True) as (fd, domain, lock_node):
            incarnation = str(uuid4())
            self._write(
                fd,
                "initialized.json",
                {
                    "schema": 1,
                    "incarnation": incarnation,
                    "lock_device": lock_node[0],
                    "lock_inode": lock_node[1],
                },
            )
            self._save(fd, _State(incarnation, 0, ()))
            self._load(fd, lock_node, domain)

    def capture(self) -> CoordinatorReceipt:
        """Before slow external verification; no read-path file creation."""
        with self._locked() as (fd, domain, lock_node):
            state = self._load(fd, lock_node, domain)
            if state.pending:
                raise CoordinatorRefusal("coordinator pending")
            return CoordinatorReceipt(domain, state.incarnation, state.revision)

    @contextmanager
    def guard(self, receipt: CoordinatorReceipt) -> Iterator[CoordinatorObservation]:
        """Innermost exclusion through at most ONE nonblocking emission.

        A mismatch raises, never yields a positive result. All source, account,
        provenance and audit verification must precede this boundary. Holding it
        is not proof those independent checks or all-mutator coverage exist.
        """
        if type(receipt) is not CoordinatorReceipt:
            raise CoordinatorRefusal("exact coordinator receipt required")
        with self._locked() as (fd, domain, lock_node):
            state = self._load(fd, lock_node, domain)
            if state.pending or receipt != CoordinatorReceipt(
                domain, state.incarnation, state.revision
            ):
                raise CoordinatorRefusal("stale coordinator receipt")
            yield CoordinatorObservation(current=True)

    def begin_mutation(self, *, effect: EffectIdentity) -> MutationReceipt:
        """Durably publish PENDING BEFORE any affected effect or prompt.

        Overlapping/nested calls keep distinct transactions. External publication
        and its independent readback happen after this method releases the lock.
        A failure means do not start the effect; it does not mean rollback.
        """
        if type(effect) is not EffectIdentity:
            raise CoordinatorRefusal("exact effect identity required")
        attempted = False
        try:
            with self._locked() as (fd, domain, lock_node):
                state = self._load(fd, lock_node, domain)
                transaction = MutationReceipt(domain, state.incarnation, str(uuid4()), effect)
                attempted = True
                self._save(
                    fd,
                    _State(
                        state.incarnation,
                        state.revision + 1,
                        tuple(
                            sorted((*state.pending, transaction), key=lambda item: item.transaction)
                        ),
                        state.completed,
                    ),
                )
                return transaction
        except BaseException as exc:
            self._failed = True
            if attempted and isinstance(exc, Exception):
                raise CoordinatorIndeterminate(
                    "begin not acknowledged; do not perform effect"
                ) from exc
            raise

    def finish_mutation(
        self, transaction: MutationReceipt, *, verified_outcome: VerifiedOutcome
    ) -> FinishedMutation:
        """Complete ONLY this transaction after trusted independent verification.

        ``verified_outcome`` is closed, exact-effect, fingerprint-bound evidence
        from the TRUSTED CALLER's independent readback, not a boolean/callback or
        permission. Unknown/failed outcomes cannot finish. Never call this from
        a generic finally. Effect/profile/transaction mismatches stay negative.

        Completion visibility linearizes at state rename; success requires file
        and parent fsync plus exact readback. If completion committed but its
        final sync/readback/ack fails, this handle latches negative and raises
        indeterminate. A fresh process MAY find that verified completion already
        committed. That is not recovery or permission: the pre-begin receipt is
        still invalid, and future readers must capture afresh BEFORE all their
        independent authority verification. Pending/partial/unknown stays denied.
        """
        if (
            type(transaction) is not MutationReceipt
            or type(verified_outcome) is not VerifiedOutcome
            or transaction.effect != verified_outcome.effect
        ):
            raise CoordinatorRefusal("verified exact transaction required")
        attempted = False
        try:
            with self._locked() as (fd, domain, lock_node):
                state = self._load(fd, lock_node, domain)
                if (
                    transaction.domain != domain
                    or transaction.incarnation != state.incarnation
                    or transaction not in state.pending
                ):
                    raise CoordinatorRefusal("stale mutation receipt")
                evidence = CompletionEvidence(transaction, verified_outcome, state.revision + 1)
                attempted = True
                self._save(
                    fd,
                    _State(
                        state.incarnation,
                        state.revision + 1,
                        tuple(item for item in state.pending if item != transaction),
                        tuple(
                            sorted(
                                (*state.completed, evidence),
                                key=lambda item: item.transaction.transaction,
                            )
                        ),
                    ),
                )
            # Includes the lock/path exit checks: a return from inside the with
            # would also unwind it, but making the ACK site explicit matters.
            return FinishedMutation(evidence)
        except BaseException as exc:
            self._failed = True
            if attempted and isinstance(exc, Exception):
                raise CoordinatorIndeterminate(
                    "completion not acknowledged; state indeterminate"
                ) from exc
            raise

    def inspect_completion(
        self, transaction: MutationReceipt, *, verified_outcome: VerifiedOutcome
    ) -> CompletionEvidence:
        """Read ONLY an exact committed fact; never upgrade an earlier outcome.

        A fresh handle may observe an already-committed verified completion after
        an ACK failure. The original indeterminate result/latched handle stays
        negative forever. This neither finishes/re-executes an effect nor repairs
        data, supplies permission, clears pending work, or skips new independent
        authority verification. Absent/different effect/profile/tx/fingerprint
        refuses; never infer that the external effect should be retried.
        """
        if (
            type(transaction) is not MutationReceipt
            or type(verified_outcome) is not VerifiedOutcome
            or transaction.effect != verified_outcome.effect
        ):
            raise CoordinatorRefusal("exact completion identity required")
        with self._locked() as (fd, domain, lock_node):
            state = self._load(fd, lock_node, domain)
            for item in state.completed:
                if item.transaction == transaction and item.verified == verified_outcome:
                    return item
            raise CoordinatorRefusal("matching committed completion absent")

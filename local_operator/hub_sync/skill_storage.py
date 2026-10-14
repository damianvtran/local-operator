"""Managed text skill adapter: staged whole-directory commits and journal recovery.

Only the active profile's native skills root is writable. Project, ecosystem and
packaged roots remain publication INPUTS, never destinations. Readers use the
same short config-root transaction lock, so the two directory renames cannot
be observed as a missing/half-new skill by another harness reader. External
editors do not honor advisory locks: every commit/recovery compares actual bytes
and preserves unexpected edits instead of restoring a stale snapshot.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping

from local_operator.config import ConfigManager, read_config_values
from local_operator.config_writer import config_write_lock
from local_operator.hub_sync.identity import LinkError, LinkIdentity
from local_operator.hub_sync.policies import materialize_policy, read_policy
from local_operator.hub_sync.provenance import (
    BaselineRecord,
    alias_owner,
    parse_baseline_record,
    read_baseline,
    write_baseline,
)
from local_operator.hub_sync.skill_document import SkillDocument, validate_name
from local_operator.hub_sync.transactions import (
    LocalTransaction,
    TransactionError,
    TransactionJournal,
    _sync_ancestors,
    _sync_directory,
)


@dataclass(frozen=True)
class SkillSnapshot:
    local_id: str
    alias: str
    document: SkillDocument
    fingerprint: str


class ManagedSkillStorage:
    def __init__(self, config_dir: Path) -> None:
        self.config_dir = Path(config_dir).expanduser().resolve()
        self.root = self.config_dir / "skills"
        self.journal = TransactionJournal(self.config_dir)

    def path(self, alias: str) -> Path:
        validate_name(alias)
        path = self.root / alias
        if self.root.is_symlink() or path.is_symlink():
            raise LinkError("skill-validation", "managed skill root or alias is unsafe")
        return path

    def _document(self, alias: str) -> SkillDocument | None:
        path = self.path(alias)
        if not path.exists():
            return None
        if not path.is_dir():
            raise LinkError(
                "origin-conflict", "managed alias is occupied by another local artifact"
            )
        document = SkillDocument.from_directory(path)
        if document.name != alias:
            raise LinkError("skill-validation", "managed skill name must match its recorded alias")
        return document

    def snapshot(self, local_id: str) -> SkillSnapshot:
        with config_write_lock(self.config_dir):
            self.recover(local_id=local_id)
            record = read_baseline(self.config_dir, "skill", local_id)
            if record is None or record.local_alias is None:
                raise LinkError("baseline-unknown", "managed skill has no recorded local alias")
            document = self._document(record.local_alias)
            if document is None:
                raise LinkError("local-concurrent-edit", "managed skill is missing locally")
            return SkillSnapshot(local_id, record.local_alias, document, document.fingerprint)

    def _stage(self, record: LocalTransaction, document: SkillDocument) -> Path:
        hub = self.config_dir / "hub"
        staging = hub / "staging"
        operation = staging / record.operation_id
        content = operation / "content"
        for directory in (hub, staging, operation, content):
            if directory.is_symlink():
                raise LinkError("skill-validation", "managed skill staging is unsafe")
            directory.mkdir(mode=0o700, exist_ok=True)
        for file in document.files:
            target = content / file.path
            parent = content
            for component in Path(file.path).parts[:-1]:
                parent = parent / component
                parent.mkdir(mode=0o700, exist_ok=True)
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(file.content.encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
        for directory, _children, _files in os.walk(content, topdown=False):
            _sync_directory(Path(directory))
        if SkillDocument.from_directory(content).fingerprint != document.fingerprint:
            raise LinkError("local-concurrent-edit", "staged skill does not match its proposal")
        return content

    def _locations(self, transaction: LocalTransaction) -> tuple[Path, Path, Path]:
        # validate_name AUTHENTICATES the payload value (Any -> a checked
        # kebab-case name) and returns it; binding the return carries that
        # runtime verification into the type at this one payload boundary.
        alias = validate_name(transaction.payload.get("local_alias"))
        destination = self.path(alias)
        staged = self.config_dir / "hub" / "staging" / transaction.operation_id / "content"
        backups = self.config_dir / "hub" / "backups"
        backup = backups / f"skill-{transaction.local_id}-{transaction.operation_id}"
        for directory in (
            self.config_dir / "hub",
            self.config_dir / "hub" / "staging",
            staged.parent,
            staged,
            backups,
            backup,
        ):
            if directory.is_symlink():
                raise LinkError("skill-validation", "managed skill transaction path is unsafe")
        return destination, staged, backup

    def _fingerprint_path(self, path: Path) -> str | None:
        if not path.exists():
            return None
        return SkillDocument.from_directory(path).fingerprint

    def _manager(self) -> ConfigManager:
        # A recovery reader must not seed defaults or move an unreadable file
        # aside merely by constructing the existing config loader.
        if (
            read_config_values(self.config_dir) is None
            and (self.config_dir / "config.yml").exists()
        ):
            raise LinkError("invalid-link-policy", "configuration is unreadable")
        return ConfigManager(self.config_dir)

    def _finalize(self, transaction: LocalTransaction) -> None:
        value = transaction.payload.get("baseline")
        if not isinstance(value, Mapping):
            raise TransactionError("baseline-unknown", "skill integration baseline is missing")
        # The adapter wrote this complete baseline before content commit. Parse
        # it through the same strict side-store reader, never an imported skill
        # metadata field, and keep every other baseline/receipt field intact.
        identity = LinkIdentity.from_json(transaction.link)
        if identity.kind != "skill":
            raise LinkError("origin-conflict", "skill integration origin is invalid")
        existing = read_baseline(self.config_dir, "skill", transaction.local_id)
        if existing is not None and existing.identity is not None:
            existing.identity.require_same_origin(identity)
        if value.get("kind") != "skill" or value.get("local_id") != transaction.local_id:
            raise LinkError(
                "origin-conflict", "skill integration baseline targets another artifact"
            )
        if value.get("identity") != identity.to_json():
            raise LinkError("origin-conflict", "skill integration baseline targets another origin")
        restored = parse_baseline_record(dict(value), "skill", transaction.local_id)
        if restored is None or restored.local_alias != transaction.payload["local_alias"]:
            raise TransactionError("baseline-unknown", "skill integration baseline is invalid")
        if SkillDocument.from_json(dict(restored.fields)).fingerprint != restored.fingerprint:
            raise TransactionError(
                "baseline-unknown", "skill integration baseline fingerprint is invalid"
            )
        manager = self._manager()
        write_baseline(self.config_dir, restored)
        materialize_policy(manager, identity, legacy_subscribe_hint=restored.legacy_subscribe_hint)
        self.journal.transition(transaction, "complete")

    def apply(
        self,
        identity: LinkIdentity,
        alias: str,
        document: SkillDocument,
        baseline: BaselineRecord,
        *,
        expected_fingerprint: str | None,
        expected_policy_revision: str | None = None,
        automatic: bool = False,
        basis: str = "integrated-baseline",
        checkpoint: Callable[[str], None] | None = None,
    ) -> LocalTransaction:
        """Commit already-validated merged bytes; no model/network calls here.

        ``document`` is the exact LOCAL alias-rendered proposal. Baseline fields
        are actual canonical remote R, not that local result. A new destination
        must be absent, or already belong to this immutable local id; an unknown
        authored folder is never overwritten by a pull.
        """
        if (
            identity.kind != "skill"
            or baseline.identity != identity
            or baseline.local_alias != alias
        ):
            raise LinkError("origin-conflict", "managed skill integration identity does not match")
        validate_name(alias)
        if document.name != alias:
            raise LinkError("skill-validation", "managed skill name must match its local alias")
        # Test fault-injection ONLY; production notifications run after return,
        # never inside the config/definition transaction lock.
        notify = checkpoint or (lambda _phase: None)
        with config_write_lock(self.config_dir):
            self.recover(local_id=identity.local_id)
            current_record = read_baseline(self.config_dir, "skill", identity.local_id)
            current = self._document(alias)
            # ALIAS OWNERSHIP IS DURABLE, NOT LIVE-DIRECTORY-LOCAL. A crash
            # between the two renames leaves the alias absent from the root while
            # its baseline still owns it; reading only the live directory let a
            # DIFFERENT origin install the same alias and then serve the other
            # origin's text (reviewer R5). An alias that belongs to another
            # immutable local id is refused, present or not.
            owner = alias_owner(self.config_dir, "skill", alias)
            if owner is not None and owner != identity.local_id:
                raise LinkError(
                    "origin-conflict", "managed alias belongs to another local skill origin"
                )
            if current is not None and (
                current_record is None
                or current_record.local_alias != alias
                or current_record.identity != identity
            ):
                raise LinkError("origin-conflict", "refusing to overwrite an unknown local skill")
            before = current.fingerprint if current is not None else None
            if before != expected_fingerprint:
                raise LinkError("local-concurrent-edit", "managed skill changed before integration")
            values = read_config_values(self.config_dir)
            if values is None and (self.config_dir / "config.yml").exists():
                raise LinkError("invalid-link-policy", "configuration is unreadable")
            policy = read_policy(values or {}, identity)
            if expected_policy_revision is not None and policy.revision != expected_policy_revision:
                raise LinkError("policy-revision-changed", "link policy changed before integration")
            if automatic and not policy.policy.subscribe:
                raise LinkError("policy-revision-changed", "subscription is off")
            transaction = self.journal.prepare(
                identity,
                "pull",
                before_fingerprint=before,
                after_fingerprint=document.fingerprint,
                payload={
                    "adapter": "managed-skill",
                    "local_alias": alias,
                    "baseline": baseline.to_json(),
                    "policy_revision": policy.revision,
                    "automatic": automatic,
                },
                basis=basis,
            )
            lease = self.journal.claim(transaction.operation_id)
            if not lease.__enter__():
                lease.__exit__(None, None, None)
                raise TransactionError(
                    "local-concurrent-edit", "skill transaction is already active"
                )
        try:
            # Slow fsync preparation is outside the short read/commit lock. Actual
            # bytes/policy are checked again when the prepared directory is ready.
            staged = self._stage(transaction, document)
            notify("staged")
            with config_write_lock(self.config_dir):
                destination, _staged, backup = self._locations(transaction)
                fresh = self._fingerprint_path(destination)
                if fresh != before:
                    self.journal.transition(
                        transaction, "blocked", error_class="local-concurrent-edit"
                    )
                    raise LinkError(
                        "local-concurrent-edit", "managed skill changed during preparation"
                    )
                values = read_config_values(self.config_dir)
                if values is None and (self.config_dir / "config.yml").exists():
                    raise LinkError("invalid-link-policy", "configuration is unreadable")
                fresh_policy = read_policy(values or {}, identity)
                if fresh_policy.revision != transaction.payload["policy_revision"]:
                    self.journal.transition(
                        transaction, "aborted", error_class="policy-revision-changed"
                    )
                    raise LinkError(
                        "policy-revision-changed", "link policy changed during preparation"
                    )
                if self._fingerprint_path(staged) != document.fingerprint:
                    self.journal.transition(
                        transaction, "blocked", error_class="local-concurrent-edit"
                    )
                    raise LinkError(
                        "local-concurrent-edit", "managed staging changed before commit"
                    )
                self.root.mkdir(mode=0o700, exist_ok=True)
                backup.parent.mkdir(mode=0o700, exist_ok=True)
                # A rename is durable only in the directory whose entries changed.
                # The backup/staging PARENT entries (and any ancestor created this
                # call) are fsynced exactly once before the swap, so a power loss
                # cannot leave the newly-created hubs undurable; no blanket fsync of
                # unrelated tree members (reviewer R4).
                _sync_ancestors(backup.parent, self.config_dir / "hub")
                _sync_ancestors(staged.parent, self.config_dir / "hub")
                if before != document.fingerprint:
                    if destination.exists():
                        os.replace(destination, backup)
                        _sync_directory(self.root)
                        notify("old-renamed")
                        # Catch an external edit between precondition read and the
                        # old directory rename; the backup contains those real bytes.
                        if self._fingerprint_path(backup) != before:
                            os.replace(backup, destination)
                            self.journal.transition(
                                transaction, "blocked", error_class="local-concurrent-edit"
                            )
                            raise LinkError(
                                "local-concurrent-edit", "managed skill changed during commit"
                            )
                    os.replace(staged, destination)
                    _sync_directory(self.root)
                    notify("new-renamed")
                if self._fingerprint_path(destination) != document.fingerprint:
                    self.journal.transition(
                        transaction, "blocked", error_class="local-concurrent-edit"
                    )
                    raise LinkError("local-concurrent-edit", "managed skill changed after commit")
                transaction = self.journal.transition(transaction, "content-committed")
                notify("content-committed")
                self._finalize(transaction)
                notify("complete")
                return self.journal.read(transaction.operation_id) or transaction

        finally:
            lease.__exit__(None, None, None)

    def recover(self, *, local_id: str | None = None) -> None:
        """Recover only skill commits; never lock/recover an agent from a reader."""
        with config_write_lock(self.config_dir):
            for transaction in self.journal.pending(kind="skill", local_id=local_id):
                if transaction.payload.get("adapter") != "managed-skill":
                    continue
                with self.journal.claim(transaction.operation_id) as claimed:
                    if claimed:
                        self._recover_one(transaction)

    def _recover_one(self, transaction: LocalTransaction) -> None:
        if transaction.phase == "blocked":
            return
        destination, staged, backup = self._locations(transaction)
        actual = self._fingerprint_path(destination)
        if actual == transaction.after_fingerprint:
            self._finalize(transaction)
            return
        if actual is None and backup.exists() and staged.exists():
            if self._fingerprint_path(backup) != transaction.before_fingerprint:
                # External editors do not hold the advisory lease. Even a crash
                # gap is not permission to promote an unexpectedly changed backup.
                self.journal.transition(transaction, "blocked", error_class="local-concurrent-edit")
                return
            # A crash in the two-rename gap restores the real original;
            # it does not blindly finish a now-unconfirmed prepared pull.
            os.replace(backup, destination)
            _sync_directory(self.root)
            self.journal.transition(transaction, "aborted")
            return
        if actual == transaction.before_fingerprint:
            self.journal.transition(transaction, "aborted")
            if staged.exists():
                shutil.rmtree(staged)  # only this operation's own prepared tree
            return
        self.journal.transition(transaction, "blocked", error_class="local-concurrent-edit")


@contextmanager
def managed_skill_read(config_dir: Path) -> Iterator[None]:
    """A consistent owned-root discovery/body/reference read, with crash recovery.

    Only callers' explicitly injected ACTIVE native root uses this writer/read
    discipline. No project/ecosystem/builtin directory is modified or recovered.
    A live preparation is skipped by its nonblocking kernel lease; swaps happen
    under this same short root lock, not under slow staging/model work.
    """
    storage = ManagedSkillStorage(config_dir)
    if not storage.root.exists() and not (storage.config_dir / "hub" / "transactions").exists():
        # Discovery of a never-created root is read-only. A first installation
        # swaps its complete staged directory into an absent alias atomically;
        # there is no old directory's two-rename gap to guard here. Existing
        # journals still require recovery even if a crashed swap hid the root.
        yield
        return
    with config_write_lock(config_dir):
        storage.recover()
        yield
